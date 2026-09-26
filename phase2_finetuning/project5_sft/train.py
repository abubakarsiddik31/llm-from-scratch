# ruff: noqa
"""
Supervised Fine-Tuning Loop: the Project 4 Base Model on Alpaca

Pipeline position: prepare_data.py → THIS → generate.py

The loop skeleton (batch → forward → backward → step, grad accumulation,
bf16, warmup/cosine LR, clipping) is the project 4 training loop. Four
things change, and each is the point of the project:

1. THE WEIGHTS START FROM THE BASE CHECKPOINT, not from init.
   checkpoints/project4/model_final.pt is loaded before step zero; the
   optimizer starts fresh. SFT continues training, it does not retrain.

2. THE LOSS SEES ONLY THE RESPONSE. Targets carry -1 (ignore_index) on
   prompt and pad positions, so gradients teach "given this instruction,
   produce this answer", never "given these prompt tokens, predict the
   next prompt token". --loss_on full switches to Alpaca's actual choice
   (loss over the whole sequence) for comparison.

3. THE DATA IS EPOCHS OVER A FIXED SET. Pre-training sampled random
   windows from a stream the model would see once; here ~51k examples are
   revisited ~3 times (shuffled per epoch with a seed). Revisiting is what
   makes overfitting the failure mode to watch - hence the val estimate
   every EVAL_INTERVAL steps and checkpoint_best.pt by val loss.

4. LEARNING RATE DROPS 12x (2e-5 vs 2.5e-4, Alpaca's setting). The base
   weights sit in a good basin; large steps wreck the pre-training
   distribution faster than the new behavior can replace it.

CHECKPOINTS keep the repository-wide five-key format:
    {'iter', 'model_state_dict', 'optimizer_state_dict',
     'train_loss', 'val_loss'}
with two extra keys on the final save: 'config' (architecture dict, so
generate.py can rebuild the model) and 'base_checkpoint' (which project 4
checkpoint this one descended from). All loads use weights_only=True.
"""

import argparse
import math
import pathlib
import time

import numpy as np
import torch

import config
from generate import sample_response
from model import GPT, GPTConfig
from template import format_prompt
from tokenizer import BPETokenizer


# =============================================================================
# DATA
# =============================================================================


def load_sft_arrays(data_dir: pathlib.Path):
    """
    Memory-map the padded token/label arrays written by prepare_data.py.

    sft_train.npy is (N, MAX_SEQ_LEN) uint16; the aligned labels file
    carries -1 on every position we do not teach. mmap keeps the ~110 MB
    pair resident page-by-page instead of up front.
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
            f"Missing {missing[0]}. Run prepare_data.py first."
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
    With "full" (Alpaca's choice) every position teaches.
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
    same shape as project 4, lower peak - see module docstring).
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
    are excluded. Betas follow the GPT-3/nanoGPT convention (Alpaca used
    PyTorch defaults; see config.py).
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
# BASE CHECKPOINT
# =============================================================================


def load_base_model(base_checkpoint: str, device: str) -> tuple:
    """
    Rebuild the model from the project 4 checkpoint and load its weights.

    The final pre-training save carries a 'config' architecture dict; if a
    bare five-key checkpoint is passed instead, the local config.py must
    match it (it does by construction - project 5's copy exists precisely
    to keep these in step) and we say so.
    """
    path = pathlib.Path(base_checkpoint)
    if not path.is_file():
        raise FileNotFoundError(
            f"No base checkpoint at {path}. Pre-train first with "
            f"phase1_foundation/project4_pretrain/train.py."
        )

    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    arch = ckpt.get("config")
    if arch is None:
        print(f"WARNING: {path.name} has no embedded config; trusting local config.py")
        arch = {
            "VOCAB_SIZE": config.VOCAB_SIZE,
            "N_EMBD": config.N_EMBD,
            "N_HEAD": config.N_HEAD,
            "N_LAYER": config.N_LAYER,
            "BLOCK_SIZE": config.BLOCK_SIZE,
            "DROPOUT": config.DROPOUT,
        }
    for key, expected in [
        ("VOCAB_SIZE", config.VOCAB_SIZE), ("BLOCK_SIZE", config.BLOCK_SIZE),
    ]:
        assert arch[key] == expected, (
            f"base checkpoint {key}={arch[key]} != config {expected}"
        )

    model = GPT(GPTConfig(
        VOCAB_SIZE=arch["VOCAB_SIZE"],
        N_EMBD=arch["N_EMBD"],
        N_HEAD=arch["N_HEAD"],
        N_LAYER=arch["N_LAYER"],
        BLOCK_SIZE=arch["BLOCK_SIZE"],
        DROPOUT=arch.get("DROPOUT", 0.0),
    ))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    val_loss = ckpt.get("val_loss")
    print(f"Base model: {path}")
    print(f"  iter {ckpt.get('iter')}"
          f"{f', val loss {val_loss:.4f}' if val_loss is not None else ''} "
          f"-> SFT continues from these weights")
    return model, arch


# =============================================================================
# EVALUATION
# =============================================================================


@torch.no_grad()
def estimate_loss(model, arrays, loss_mode: str, autocast_ctx):
    """
    Average loss over EVAL_ITERS batches per split (model.eval mode).

    The batch indices are drawn from a fresh rng seeded with config.SEED,
    so every eval scores the SAME batches: differences between evals come
    from the model alone, which makes val losses comparable across steps,
    runs, and the response-only vs full-loss experiment.
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
    """One seeded shuffle per epoch: same order every time an epoch index
    recurs, different order across epochs."""
    epoch = 0
    while True:
        yield np.random.default_rng(config.SEED + epoch).permutation(n_examples)
        epoch += 1


def train(model, optimizer, arrays, checkpoint_dir, args, arch):
    autocast_ctx = (
        lambda: torch.autocast("cuda", dtype=torch.bfloat16)
        if (config.USE_BF16 and config.DEVICE == "cuda")
        else torch.autocast("cpu", enabled=False)
    )

    loss_mode = "response" if config.LOSS_ON_RESPONSE_ONLY else "full"
    examples_per_step = config.BATCH_SIZE * config.GRAD_ACCUM_STEPS

    print("\n" + "=" * 60)
    print("STARTING SUPERVISED FINE-TUNING")
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
        steps_this_epoch = len(perm) // examples_per_step  # drop last partial
        if steps_this_epoch == 0:
            break
        print(f"--- epoch {epoch_no + 1} ({steps_this_epoch} optimizer steps)")
        epoch_no += 1

        for s in range(steps_this_epoch):
            if step >= config.MAX_ITERS:
                break

            # -----------------------------------------------------------
            # periodic eval + checkpointing
            # -----------------------------------------------------------
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

            # -----------------------------------------------------------
            # one optimizer step = GRAD_ACCUM_STEPS micro-batches over one
            # shuffled chunk of the epoch (64 examples, each seen once
            # per epoch)
            # -----------------------------------------------------------
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            optimizer.step()

            iters_since_last += 1
            step += 1

    # ---------------------------------------------------------------
    # FINAL SAVE (adds the architecture dict + provenance)
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
        "base_checkpoint": pathlib.Path(args.base_checkpoint).name,
    }, final_path)
    print(f"\nSaved final model to {final_path}")
    print(f"Final val loss (response tokens): {losses['val']:.4f} "
          f"(perplexity {math.exp(losses['val']):.1f})")

    # ---------------------------------------------------------------
    # qualitative sanity check: the base model continued text; this one
    # should answer
    # ---------------------------------------------------------------
    model.eval()
    tokenizer = BPETokenizer.load()
    instructions = [
        ("Give three tips for staying healthy.", ""),
        ("What is the capital of France?", ""),
        ("Rewrite this sentence in past tense: 'She walks to school.'", ""),
    ]
    print("\nSamples from the fine-tuned model:")
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
        description="Supervised fine-tune the project 4 base model on Alpaca")
    parser.add_argument("--max_iters", type=int, default=None,
                        help="Override config.MAX_ITERS (smoke tests)")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--grad_accum", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--loss_on", choices=["response", "full"], default=None,
                        help="response = InstructGPT-style masking (default), "
                             "full = Alpaca-style loss over the whole sequence")
    parser.add_argument("--base_checkpoint", type=str, default=config.BASE_CHECKPOINT,
                        help="Project 4 checkpoint to start from")
    parser.add_argument("--resume", action="store_true",
                        help="Continue from checkpoints/project5/checkpoint_latest.pt")
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
    if args.loss_on:
        config.LOSS_ON_RESPONSE_ONLY = args.loss_on == "response"

    config.validate_config()
    config.print_config()

    data_dir = pathlib.Path(config.TOKENIZED_DATA_DIR)
    arrays = load_sft_arrays(data_dir)

    model, arch = load_base_model(args.base_checkpoint, config.DEVICE)
    optimizer = configure_optimizer(model)
    print(f"Optimizer: AdamW lr={config.LEARNING_RATE} "
          f"betas=({config.BETA1}, {config.BETA2}) wd={config.WEIGHT_DECAY}")

    train(model, optimizer, arrays, config.CHECKPOINT_DIR, args, arch)


if __name__ == "__main__":
    main()
