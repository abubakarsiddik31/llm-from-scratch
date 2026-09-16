# ruff: noqa
"""
Pre-Training Loop for the ~125M GPT on WikiText-103

Pipeline position: prepare_data.py → THIS → generate.py

Same skeleton as the chapter 1 training loop (batch → forward → backward →
step) with the additions that only matter at 125M parameters:

- GRADIENT ACCUMULATION: the model is updated every GRAD_ACCUM_STEPS
  forward/backward passes, so BATCH_SIZE stays within 8 GB while the
  effective batch matches GPT-3-style token budgets (32,768 tokens/step).
- MIXED PRECISION (bf16): forward/backward in bfloat16, master weights and
  optimizer state in fp32. bf16 has fp32-range exponent, so unlike fp16 it
  needs no loss scaling (Ampere+, which the RTX 3050 is).
- LEARNING RATE SCHEDULE: linear warmup then cosine decay to 10% of peak
  (GPT-2/GPT-3). At this scale a constant LR loses several points of loss.
- PARAMETER GROUPS: weight decay applies to weight matrices only, not to
  LayerNorm gains - the GPT-3 Appendix B / nanoGPT convention.
- GRADIENT CLIPPING at 1.0 global norm (GPT-2 used 1.0): one bad batch
  should not throw the optimizer's second-moment estimates into a bad
  state.

CHECKPOINTS keep the repository-wide five-key format:
    {'iter', 'model_state_dict', 'optimizer_state_dict',
     'train_loss', 'val_loss'}
so chapter tooling can read them. The final save additionally carries the
architecture as a plain dict (weights_only-loadable) so generate.py can
rebuild the exact model. All loads use torch.load(..., weights_only=True):
our checkpoints contain only tensors and primitives, and weights_only
refuses anything that could execute code.
"""

import argparse
import math
import pathlib
import time

import numpy as np
import torch

import config
from model import GPT, GPTConfig


# =============================================================================
# DATA
# =============================================================================


def load_token_bins(data_dir: pathlib.Path):
    """
    Memory-map the uint16 token files written by prepare_data.py.

    np.memmap means the OS pages tokens in on demand: training never holds
    the ~230 MB corpus (let alone a 100 GB web corpus) in RAM, and random
    window sampling is just an indexed read.
    """
    train_bins = data_dir / "train.bin"
    val_bins = data_dir / "val.bin"
    if not train_bins.is_file() or not val_bins.is_file():
        raise FileNotFoundError(
            f"Missing {train_bins} or {val_bins}. Run prepare_data.py first."
        )
    train_data = np.memmap(train_bins, dtype=np.uint16, mode="r")
    val_data = np.memmap(val_bins, dtype=np.uint16, mode="r")
    print(f"Train tokens: {len(train_data):,}")
    print(f"Val tokens:   {len(val_data):,}")
    return train_data, val_data


def get_batch(split_data, batch_size: int, block_size: int, device: str):
    """
    Sample a (B, T) input window and its (B, T) shifted-by-one targets.

    Same teacher-forcing scheme as chapter 1, but the windows start at
    random offsets in a memmap, so every step sees a different slice of the
    corpus without loading it.
    """
    ix = torch.randint(len(split_data) - block_size - 1, (batch_size,))
    x = np.stack([split_data[i:i + block_size].astype(np.int64) for i in ix])
    y = np.stack([split_data[i + 1:i + block_size + 1].astype(np.int64) for i in ix])
    x = torch.from_numpy(x).pin_memory().to(device, non_blocking=True)
    y = torch.from_numpy(y).pin_memory().to(device, non_blocking=True)
    return x, y


# =============================================================================
# LR SCHEDULE
# =============================================================================


def get_lr(it: int) -> float:
    """
    Linear warmup -> cosine decay to MIN_LEARNING_RATE (GPT-3, Appendix B).
    """
    if it < config.WARMUP_ITERS:
        return config.LEARNING_RATE * (it + 1) / config.WARMUP_ITERS
    if it >= config.MAX_ITERS:
        return config.MIN_LEARNING_RATE
    progress = (it - config.WARMUP_ITERS) / (config.MAX_ITERS - config.WARMUP_ITERS)
    cos = 0.5 * (1 + math.cos(math.pi * progress))
    return config.MIN_LEARNING_RATE + cos * (
        config.LEARNING_RATE - config.MIN_LEARNING_RATE
    )


# =============================================================================
# OPTIMIZER
# =============================================================================


def configure_optimizer(model):
    """
    AdamW with decay on weight matrices only (ndim >= 2); LayerNorm gains
    are excluded. Betas follow GPT-2/GPT-3. fused=True uses the CUDA-fused
    implementation when available (meaningfully faster at 126M params).
    """
    decay, no_decay = [], []
    for param in model.parameters():
        if param.requires_grad:
            (decay if param.ndim >= 2 else no_decay).append(param)
    groups = [
        {"params": decay, "weight_decay": config.WEIGHT_DECAY},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    try:
        return torch.optim.AdamW(
            groups,
            lr=config.LEARNING_RATE,
            betas=(config.BETA1, config.BETA2),
            eps=1e-8,
            fused=(config.DEVICE == "cuda"),
        )
    except (TypeError, RuntimeError):
        return torch.optim.AdamW(
            groups, lr=config.LEARNING_RATE, betas=(config.BETA1, config.BETA2)
        )


# =============================================================================
# MODEL CONFIG <-> PLAIN DICT
# =============================================================================


def arch_dict():
    """Architecture as a plain dict (stays inside weights_only checkpoints)."""
    return {
        "VOCAB_SIZE": config.VOCAB_SIZE,
        "N_EMBD": config.N_EMBD,
        "N_HEAD": config.N_HEAD,
        "N_LAYER": config.N_LAYER,
        "BLOCK_SIZE": config.BLOCK_SIZE,
        "DROPOUT": config.DROPOUT,
    }


def model_from_arch(arch: dict) -> GPT:
    """Build the model from an architecture dict (checkpoint or config)."""
    return GPT(GPTConfig(
        VOCAB_SIZE=arch["VOCAB_SIZE"],
        N_EMBD=arch["N_EMBD"],
        N_HEAD=arch["N_HEAD"],
        N_LAYER=arch["N_LAYER"],
        BLOCK_SIZE=arch["BLOCK_SIZE"],
        DROPOUT=arch.get("DROPOUT", 0.0),
    ))


# =============================================================================
# EVALUATION
# =============================================================================


@torch.no_grad()
def estimate_loss(model, train_data, val_data, autocast_ctx):
    """Average loss over EVAL_ITERS batches per split (model.eval mode)."""
    model.eval()
    out = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(config.EVAL_ITERS)
        for k in range(config.EVAL_ITERS):
            x, y = get_batch(data, config.BATCH_SIZE, config.BLOCK_SIZE, config.DEVICE)
            with autocast_ctx():
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# =============================================================================
# TRAINING LOOP
# =============================================================================


def train(model, optimizer, train_data, val_data, checkpoint_dir, args):
    autocast_ctx = (
        lambda: torch.autocast("cuda", dtype=torch.bfloat16)
        if (config.USE_BF16 and config.DEVICE == "cuda")
        else torch.autocast("cpu", enabled=False)
    )

    tokens_per_step = config.BATCH_SIZE * config.GRAD_ACCUM_STEPS * config.BLOCK_SIZE
    print("\n" + "=" * 60)
    print("STARTING PRE-TRAINING")
    print(f"  {tokens_per_step:,} tokens/optimizer step, "
          f"{config.MAX_ITERS:,} steps "
          f"({config.MAX_ITERS * tokens_per_step / 1e6:,.0f}M tokens)")
    print("=" * 60 + "\n")

    checkpoint_dir = pathlib.Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_path = checkpoint_dir / "checkpoint_latest.pt"
    best_path = checkpoint_dir / "checkpoint_best.pt"

    best_val_loss = float("inf")
    start_iter = 0

    # ---------------------------------------------------------------
    # RESUME (weights_only: tensors + primitives only, never executes)
    # ---------------------------------------------------------------
    if args.resume and latest_path.is_file():
        ckpt = torch.load(latest_path, map_location=config.DEVICE, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_iter = ckpt["iter"] + 1
        best_val_loss = ckpt["val_loss"]
        print(f"Resumed from {latest_path} at iter {start_iter}")

    model.train()
    t_last = time.time()
    tokens_since_last = 0
    iters_since_last = 0

    for it in range(start_iter, config.MAX_ITERS):
        # -----------------------------------------------------------
        # periodic eval + checkpointing
        # -----------------------------------------------------------
        if it % config.EVAL_INTERVAL == 0 or it == config.MAX_ITERS - 1:
            lr_now = get_lr(it)
            losses = estimate_loss(model, train_data, val_data, autocast_ctx)
            dt = time.time() - t_last
            if iters_since_last > 0 and dt > 0:
                per_iter = dt / iters_since_last
                eta_s = (config.MAX_ITERS - it) * per_iter
                print(f"iter {it:6d}/{config.MAX_ITERS} | "
                      f"train {losses['train']:.4f} | val {losses['val']:.4f} | "
                      f"lr {lr_now:.2e} | "
                      f"{tokens_since_last / dt / 1e3:.1f}k tok/s | "
                      f"eta {eta_s / 3600:.1f}h")
            else:
                print(f"iter {it:6d}/{config.MAX_ITERS} | "
                      f"train {losses['train']:.4f} | val {losses['val']:.4f} | "
                      f"lr {lr_now:.2e}")
            tokens_since_last = 0
            iters_since_last = 0
            t_last = time.time()

            snapshot = {
                "iter": it,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": losses["train"],
                "val_loss": losses["val"],
            }
            torch.save(snapshot, latest_path)
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                torch.save(snapshot, best_path)
                print(f"  → new best val loss {best_val_loss:.4f}, saved {best_path.name}")

        # -----------------------------------------------------------
        # one optimizer step = GRAD_ACCUM_STEPS micro-batches
        # -----------------------------------------------------------
        lr = get_lr(it)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        for _ in range(config.GRAD_ACCUM_STEPS):
            x, y = get_batch(train_data, config.BATCH_SIZE, config.BLOCK_SIZE,
                             config.DEVICE)
            with autocast_ctx():
                _, loss = model(x, y)
            # scale so the accumulated gradient equals the mean over all
            # micro-batches, not the sum
            (loss / config.GRAD_ACCUM_STEPS).backward()
            tokens_since_last += x.numel()
            iters_since_last += 1

        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
        optimizer.step()

    # ---------------------------------------------------------------
    # FINAL SAVE (adds the architecture dict for generate.py)
    # ---------------------------------------------------------------
    losses = estimate_loss(model, train_data, val_data, autocast_ctx)
    final_path = checkpoint_dir / "model_final.pt"
    torch.save({
        "iter": config.MAX_ITERS - 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": losses["train"],
        "val_loss": losses["val"],
        "config": arch_dict(),
    }, final_path)
    print(f"\nSaved final model to {final_path}")
    print(f"Final val loss: {losses['val']:.4f} "
          f"(perplexity {math.exp(losses['val']):.1f})")

    # ---------------------------------------------------------------
    # qualitative sanity sample
    # ---------------------------------------------------------------
    from tokenizer import BPETokenizer
    model.eval()
    tokenizer = BPETokenizer.load()
    context = torch.full((1, 1), config.SPECIAL_TOKENS["<BOS>"],
                         dtype=torch.long, device=config.DEVICE)
    with autocast_ctx():
        generated = model.generate(
            context,
            max_new_tokens=200,
            temperature=config.TEMPERATURE,
            top_k=config.TOP_K,
        )[0].tolist()
    model.train()
    print("\nSample from <BOS>:")
    print("-" * 60)
    print(tokenizer.decode(generated))
    print("-" * 60)


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Pre-train the ~125M GPT")
    parser.add_argument("--max_iters", type=int, default=None,
                        help="Override config.MAX_ITERS (smoke tests)")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--grad_accum", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", action="store_true",
                        help="Continue from checkpoints/project4/checkpoint_latest.pt")
    args = parser.parse_args()

    if args.max_iters:
        config.MAX_ITERS = args.max_iters
        # keep warmup a sensible fraction of a shortened schedule
        config.WARMUP_ITERS = min(config.WARMUP_ITERS, max(1, args.max_iters // 5))
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.grad_accum:
        config.GRAD_ACCUM_STEPS = args.grad_accum
    if args.device:
        config.DEVICE = args.device

    config.validate_config()
    config.print_config()

    data_dir = pathlib.Path(config.TOKENIZED_DATA_DIR)
    train_data, val_data = load_token_bins(data_dir)

    model = model_from_arch(arch_dict())
    model.to(config.DEVICE)

    optimizer = configure_optimizer(model)
    print(f"Optimizer: AdamW lr={config.LEARNING_RATE} "
          f"betas=({config.BETA1}, {config.BETA2}) wd={config.WEIGHT_DECAY}")

    train(model, optimizer, train_data, val_data, config.CHECKPOINT_DIR, args)


if __name__ == "__main__":
    main()
