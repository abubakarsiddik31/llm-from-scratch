# ruff: noqa
"""
LoRA Fine-Tuning Loop: the Base Model on Alpaca, Adapters Only

Pipeline position: project 5's prepare_data.py → THIS → generate.py

The loop skeleton (batch → forward → backward → step, grad accumulation,
bf16, warmup/cosine LR, clipping) is project 5's training loop. What
changes is WHAT LEARNS:

1. THE BASE CHECKPOINT LOADS, THEN LoRA WRAPS IT.
   checkpoints/project4/model_final.pt (or --base_checkpoint) loads into
   the plain model first; apply_lora() then swaps every c_attn for a
   LoRALinear. lora_B starts at zero, so step 0 computes exactly what the
   base model computes.

2. ONLY ADAPTERS RECEIVE GRADIENTS. All original weights are frozen
   (requires_grad=False). At the default r=8 on c_attn that is 393,216
   trainable parameters, 0.311% of the model - test_model.py asserts the
   count. The optimizer therefore holds moments for 0.4M parameters
   instead of 126M, which is where LoRA's memory savings live.

3. HIGHER LEARNING RATE. 1e-4 vs project 5's 2e-5: the frozen base bounds
   how far any step can move the function, and the paper's GPT-3 sweeps
   sit at 1e-4..3e-4 for LoRA (Hu et al., 2021, Table 4).

4. NO WEIGHT DECAY. Pulling the low-rank update toward zero fights the
   objective for no benefit on a short schedule.

Everything else mirrors project 5 on purpose: same encoded Alpaca arrays
(data/project5), same 64-examples-per-step batch shape, same 2,400-step
(~3 epochs) schedule, same five-key checkpoints plus a 'lora_config'
dict, so the loss curves of the two projects are directly comparable.

CHECKPOINT FORMAT: {'iter', 'model_state_dict', 'optimizer_state_dict',
'train_loss', 'val_loss'} + {'config', 'lora_config', 'base_checkpoint'}
on the final save. model_state_dict carries the frozen base weights too,
so a checkpoint is self-contained; generate.py applies LoRA before
loading and refuses a lora_config/plain mismatch.
"""

import argparse
import math
import pathlib
import time

import numpy as np
import torch

import config
import lora
from generate import sample_response
from model import GPT, GPTConfig
from template import format_prompt
from tokenizer import BPETokenizer


# =============================================================================
# DATA (same arrays project 5 encoded)
# =============================================================================


def load_sft_arrays(data_dir: pathlib.Path):
    """
    Memory-map the padded token/label arrays project 5's prepare_data.py
    wrote (reused verbatim: same data is what makes the full-SFT vs LoRA
    comparison controlled).
    """
    paths = {
        "train": data_dir / "sft_train.npy",
        "train_labels": data_dir / "sft_train_labels.npy",
        "val": data_dir / "sft_val.npy",
        "val_labels": data_dir / "sft_val_labels.npy",
    }
    missing = [p for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing[0]}. Run "
            f"phase2_finetuning/project5_sft/prepare_data.py first "
            f"(project 6 reuses project 5's encoded data)."
        )
    arrays = {name: np.load(path, mmap_mode="r") for name, path in paths.items()}
    print(f"Train examples: {arrays['train'].shape[0]:,}")
    print(f"Val examples:   {arrays['val'].shape[0]:,}")
    return arrays


def _gather(rows, labels, indices, loss_mode: str) -> tuple:
    """
    Build (x, y) int64 tensors from padded rows.

    The shift is the same teacher forcing as every chapter: input position
    t holds token t, target position t holds the TEACHER LABEL for token
    t+1. With loss_mode="response" that label is -1 wherever token t+1 is
    prompt or padding, so only response/<EOS> positions contribute loss.
    """
    idx = np.asarray(indices, dtype=np.int64)
    x = rows[idx][:, :-1].astype(np.int64)
    y = (labels[idx][:, 1:] if loss_mode == "response" else rows[idx][:, 1:]).astype(np.int64)
    return torch.from_numpy(x).pin_memory(), torch.from_numpy(y).pin_memory()


def get_batch(arrays, indices, loss_mode: str, device: str):
    """One training batch (see _gather), from the train arrays."""
    x, y = _gather(arrays["train"], arrays["train_labels"], indices, loss_mode)
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def get_val_batch(arrays, indices, loss_mode: str, device: str):
    """One validation batch (see _gather), from the val arrays."""
    x, y = _gather(arrays["val"], arrays["val_labels"], indices, loss_mode)
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


# =============================================================================
# LR SCHEDULE
# =============================================================================


def get_lr(it: int) -> float:
    """
    Linear warmup -> cosine decay to MIN_LEARNING_RATE (GPT-3, Appendix B;
    same shape as projects 4/5, higher peak - see module docstring).
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
# OPTIMIZER (adapters only)
# =============================================================================


def configure_optimizer(model):
    """
    AdamW over the LoRA parameters only (all of them 2-D matrices). No
    parameter groups needed: config.WEIGHT_DECAY is 0.0 for LoRA, so
    decay-vs-no-decay distinction is moot.
    """
    params = lora.trainable_parameters(model)
    try:
        return torch.optim.AdamW(
            params,
            lr=config.LEARNING_RATE,
            betas=(config.BETA1, config.BETA2),
            eps=1e-8,
            fused=(config.DEVICE == "cuda"),
        )
    except (TypeError, RuntimeError):
        return torch.optim.AdamW(
            params, lr=config.LEARNING_RATE, betas=(config.BETA1, config.BETA2)
        )


# =============================================================================
# BASE CHECKPOINT + LORA WIRING
# =============================================================================

ARCH_KEYS = ("VOCAB_SIZE", "N_EMBD", "N_HEAD", "N_LAYER", "BLOCK_SIZE", "DROPOUT")


def load_base_model(base_checkpoint: str, device: str) -> tuple:
    """Rebuild the model from an earlier project's checkpoint (see
    project 5's train.py for the full version; same logic)."""
    path = pathlib.Path(base_checkpoint)
    if not path.is_file():
        raise FileNotFoundError(
            f"No base checkpoint at {path}. Pre-train with "
            f"phase1_foundation/project4_pretrain/train.py first."
        )
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    arch = ckpt.get("config")
    if arch is None:
        print(f"WARNING: {path.name} has no embedded config; trusting local config.py")
        arch = {
            "VOCAB_SIZE": config.VOCAB_SIZE, "N_EMBD": config.N_EMBD,
            "N_HEAD": config.N_HEAD, "N_LAYER": config.N_LAYER,
            "BLOCK_SIZE": config.BLOCK_SIZE, "DROPOUT": config.DROPOUT,
        }
    assert arch["VOCAB_SIZE"] == config.VOCAB_SIZE, "base vocab mismatch"
    assert arch["BLOCK_SIZE"] == config.BLOCK_SIZE, "base block size mismatch"

    model = GPT(GPTConfig(**{k: arch[k] for k in ARCH_KEYS}))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    val_loss = ckpt.get("val_loss")
    print(f"Base model: {path}")
    print(f"  iter {ckpt.get('iter')}"
          f"{f', val loss {val_loss:.4f}' if val_loss is not None else ''} "
          f"-> LoRA continues from these weights")
    return model, arch


def apply_lora_to(model, lora_cfg=None) -> list:
    """Wrap the target modules; print the frozen-vs-trainable report."""
    if lora_cfg is None:
        lora_cfg = {"r": config.LORA_R, "alpha": config.LORA_ALPHA,
                    "targets": config.LORA_TARGETS}
    replaced = lora.apply_lora(model, lora_cfg["targets"], lora_cfg["r"], lora_cfg["alpha"])
    trainable, total = lora.lora_parameter_report(model)
    print(f"LoRA: r={lora_cfg['r']} alpha={lora_cfg['alpha']} "
          f"targets={lora_cfg['targets']}")
    print(f"  wrapped {len(replaced)} modules "
          f"(e.g. {replaced[0]}, {replaced[-1]})")
    print(f"  trainable: {trainable:,} / {total:,} parameters "
          f"({trainable / total:.3%})")
    return replaced


# =============================================================================
# EVALUATION
# =============================================================================


@torch.no_grad()
def estimate_loss(model, arrays, loss_mode: str, autocast_ctx):
    """
    Average loss over EVAL_ITERS batches per split (model.eval mode).

    Batch indices come from a fresh rng seeded with config.SEED, so every
    eval scores the SAME batches - val losses are comparable across steps
    and against project 5's run (the arrays are identical too).
    """
    model.eval()
    out = {}
    rng = np.random.default_rng(config.SEED)
    for split in ["train", "val"]:
        n_rows = arrays[split].shape[0]
        losses = torch.zeros(config.EVAL_ITERS)
        for k in range(config.EVAL_ITERS):
            idx = rng.integers(0, n_rows, size=config.BATCH_SIZE)
            if split == "train":
                x, y = get_batch(arrays, idx, loss_mode, config.DEVICE)
            else:
                x, y = get_val_batch(arrays, idx, loss_mode, config.DEVICE)
            with autocast_ctx():
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# =============================================================================
# TRAINING LOOP
# =============================================================================


def epoch_permutations(n_examples: int):
    """One seeded shuffle per epoch."""
    epoch = 0
    while True:
        yield np.random.default_rng(config.SEED + epoch).permutation(n_examples)
        epoch += 1


def train(model, optimizer, arrays, checkpoint_dir, args, arch, lora_cfg):
    autocast_ctx = (
        lambda: torch.autocast("cuda", dtype=torch.bfloat16)
        if (config.USE_BF16 and config.DEVICE == "cuda")
        else torch.autocast("cpu", enabled=False)
    )

    loss_mode = "response" if config.LOSS_ON_RESPONSE_ONLY else "full"
    examples_per_step = config.BATCH_SIZE * config.GRAD_ACCUM_STEPS

    print("\n" + "=" * 60)
    print("STARTING LORA FINE-TUNING")
    print(f"  loss on: {'response tokens only' if loss_mode == 'response' else 'full sequence'}")
    print(f"  {examples_per_step} examples/optimizer step "
          f"({config.MAX_ITERS:,} steps = "
          f"{config.MAX_ITERS * examples_per_step:,} example visits)")
    print("=" * 60 + "\n")

    checkpoint_dir = pathlib.Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_path = checkpoint_dir / "checkpoint_latest.pt"
    best_path = checkpoint_dir / "checkpoint_best.pt"

    best_val_loss = float("inf")
    start_iter = 0

    if args.resume and latest_path.is_file():
        ckpt = torch.load(latest_path, map_location=config.DEVICE, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_iter = ckpt["iter"] + 1
        best_val_loss = ckpt["val_loss"]
        print(f"Resumed from {latest_path} at iter {start_iter}")

    n_train = arrays["train"].shape[0]
    model.train()
    t_last = time.time()
    iters_since_last = 0
    loss_sum_since_last = 0.0
    micro_batches_since_last = 0
    epoch_no = 0

    step = start_iter
    for perm in epoch_permutations(n_train):
        if step >= config.MAX_ITERS:
            break
        steps_this_epoch = len(perm) // examples_per_step
        if steps_this_epoch == 0:
            break
        print(f"--- epoch {epoch_no + 1} ({steps_this_epoch} optimizer steps)")
        epoch_no += 1

        for s in range(steps_this_epoch):
            if step >= config.MAX_ITERS:
                break

            if step % config.EVAL_INTERVAL == 0:
                lr_now = get_lr(step)
                losses = estimate_loss(model, arrays, loss_mode, autocast_ctx)
                dt = time.time() - t_last
                running = (loss_sum_since_last / micro_batches_since_last
                           if micro_batches_since_last else losses["train"])
                if iters_since_last > 0 and dt > 0:
                    per_iter = dt / iters_since_last
                    eta_s = (config.MAX_ITERS - step) * per_iter
                    print(f"iter {step:6d}/{config.MAX_ITERS} | "
                          f"train {running:.4f} | val {losses['val']:.4f} | "
                          f"lr {lr_now:.2e} | "
                          f"eta {eta_s / 60:.0f}min")
                else:
                    print(f"iter {step:6d}/{config.MAX_ITERS} | "
                          f"train {running:.4f} | val {losses['val']:.4f} | "
                          f"lr {lr_now:.2e}")
                iters_since_last = 0
                loss_sum_since_last = 0.0
                micro_batches_since_last = 0
                t_last = time.time()

                snapshot = {
                    "iter": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": running,
                    "val_loss": losses["val"],
                }
                torch.save(snapshot, latest_path)
                if losses["val"] < best_val_loss:
                    best_val_loss = losses["val"]
                    torch.save(snapshot, best_path)
                    print(f"  → new best val loss {best_val_loss:.4f}, "
                          f"saved {best_path.name}")

            lr = get_lr(step)
            for group in optimizer.param_groups:
                group["lr"] = lr

            chunk = perm[s * examples_per_step:(s + 1) * examples_per_step]
            optimizer.zero_grad(set_to_none=True)
            for m in range(config.GRAD_ACCUM_STEPS):
                micro = chunk[m * config.BATCH_SIZE:(m + 1) * config.BATCH_SIZE]
                x, y = get_batch(arrays, micro, loss_mode, config.DEVICE)
                with autocast_ctx():
                    _, loss = model(x, y)
                # scale so the accumulated gradient equals the mean over all
                # micro-batches, not the sum
                (loss / config.GRAD_ACCUM_STEPS).backward()
                loss_sum_since_last += loss.item()
                micro_batches_since_last += 1
            torch.nn.utils.clip_grad_norm_(
                lora.trainable_parameters(model), config.GRAD_CLIP
            )
            optimizer.step()

            iters_since_last += 1
            step += 1

    # ---------------------------------------------------------------
    # FINAL SAVE (full state dict + architecture + lora config)
    # ---------------------------------------------------------------
    losses = estimate_loss(model, arrays, loss_mode, autocast_ctx)
    final_path = checkpoint_dir / "model_final.pt"
    torch.save({
        "iter": step - 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": losses["train"],
        "val_loss": losses["val"],
        "config": dict(arch),
        "lora_config": dict(lora_cfg),
        "base_checkpoint": pathlib.Path(args.base_checkpoint).name,
    }, final_path)
    print(f"\nSaved final model to {final_path}")
    print(f"Final val loss (response tokens): {losses['val']:.4f} "
          f"(perplexity {math.exp(losses['val']):.1f})")

    # ---------------------------------------------------------------
    # qualitative sanity check
    # ---------------------------------------------------------------
    model.eval()
    tokenizer = BPETokenizer.load()
    instructions = [
        ("Give three tips for staying healthy.", ""),
        ("What is the capital of France?", ""),
        ("Rewrite this sentence in past tense: 'She walks to school.'", ""),
    ]
    print("\nSamples from the LoRA model:")
    for instruction, user_input in instructions:
        prompt = format_prompt(instruction, user_input)
        response = sample_response(
            model, tokenizer, prompt,
            max_new_tokens=config.MAX_NEW_TOKENS,
            temperature=config.TEMPERATURE, top_k=config.TOP_K,
            device=config.DEVICE,
        )
        print("-" * 60)
        print(f"Q: {instruction}")
        print(f"A: {response}")


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="LoRA fine-tune the base model on Alpaca (adapters only)")
    parser.add_argument("--max_iters", type=int, default=None,
                        help="Override config.MAX_ITERS (smoke tests)")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--grad_accum", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--loss_on", choices=["response", "full"], default=None)
    parser.add_argument("--base_checkpoint", type=str, default=config.BASE_CHECKPOINT,
                        help="Checkpoint to start from (project 4 default; pass "
                             "project 5's model_final.pt for continued adaptation)")
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--resume", action="store_true",
                        help="Continue from checkpoints/project6/checkpoint_latest.pt")
    args = parser.parse_args()

    if args.max_iters:
        config.MAX_ITERS = args.max_iters
        config.WARMUP_ITERS = min(config.WARMUP_ITERS, max(1, args.max_iters // 5))
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.grad_accum:
        config.GRAD_ACCUM_STEPS = args.grad_accum
    if args.device:
        config.DEVICE = args.device
    if args.loss_on:
        config.LOSS_ON_RESPONSE_ONLY = args.loss_on == "response"
    if args.lora_r:
        config.LORA_R = args.lora_r

    config.validate_config()
    config.print_config()

    arrays = load_sft_arrays(pathlib.Path(config.TOKENIZED_DATA_DIR))

    model, arch = load_base_model(args.base_checkpoint, config.DEVICE)
    lora_cfg = {"r": config.LORA_R, "alpha": config.LORA_ALPHA,
                "targets": list(config.LORA_TARGETS)}
    apply_lora_to(model)
    optimizer = configure_optimizer(model)
    print(f"Optimizer: AdamW lr={config.LEARNING_RATE} "
          f"betas=({config.BETA1}, {config.BETA2}) wd={config.WEIGHT_DECAY} "
          f"(adapters only)")

    train(model, optimizer, arrays, config.CHECKPOINT_DIR, args, arch, lora_cfg)


if __name__ == "__main__":
    main()
