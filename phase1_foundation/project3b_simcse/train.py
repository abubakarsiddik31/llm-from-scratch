# ruff: noqa
"""
Training Script for Unsupervised SimCSE

Fine-tunes the Project 3 BERT encoder with the SimCSE objective following:
- "SimCSE: Simple Contrastive Learning of Sentence Embeddings"
  (Gao, Yao & Chen, EMNLP 2021)

THE TRAINING STEP (paper, section 3):
-------------------------------------
1. Sample a batch of N sentences (paper: N = 64).
2. Tokenize: [CLS] sentence [SEP], max length 32.
3. Forward pass TWICE with independent dropout masks:
       z1_emb = encoder(batch)   # dropout mask z
       z2_emb = encoder(batch)   # dropout mask z' (independent)
   These are the positives; other batch entries are negatives.
4. Contrastive loss (equation 4) with temperature tau = 0.05.
5. AdamW update with warmup + linear decay.

WHAT THE PAPER DOES THAT WE KEEP:
---------------------------------
- Standard dropout (p=0.1) as the ONLY augmentation - nothing added.
- Trainable temperature fixed at tau = 0.05 for unsupervised SimCSE.
- MLP head with tanh used ONLY in the training-time loss; evaluation
  uses the raw [CLS] embedding (see model.py MLPHead docstring).
- 1 epoch over the corpus, linear schedule with warmup.

WHAT WE SCALE DOWN (RTX 3050 / 8GB demo):
-----------------------------------------
- Encoder: 6-layer, 256-hidden BERT (Project 3) instead of BERT-base.
- Corpus: 10^5 sentences instead of 10^6.
- Learning rate 5e-5 (smaller model tolerates a slightly larger rate).

USAGE:
------
uv run python phase1_foundation/project3b_simcse/train.py
"""

import math
import time
from pathlib import Path

import torch
from tqdm import tqdm

import config
import model as simcse_model
from data import Tokenizer, load_sentences, load_tokenizer, tokenize_batch
from evaluate import evaluate_stsb
from model import SimCSEEncoder, simcse_loss


def find_p3_checkpoint(explicit_path: str = None) -> str:
    """
    Locate the Project 3 MLM pre-trained checkpoint to start from.

    The paper stresses starting from a pre-trained encoder: "starting
    from a pre-trained checkpoint is crucial, for it provides good
    initial alignment."

    Args:
        explicit_path: User-supplied path (wins if provided and exists)

    Returns:
        Path to a .pt checkpoint
    """
    if explicit_path and Path(explicit_path).exists():
        return explicit_path

    p3_dir = Path(config.ROOT_DIR) / "checkpoints" / "project3"
    candidates = sorted(p3_dir.glob("bert_iter*.pt"), key=lambda p: p.stat().st_mtime)
    if candidates:
        return str(candidates[-1])

    raise FileNotFoundError(
        "No Project 3 checkpoint found in checkpoints/project3/. Train it first:\n"
        "  uv run python phase1_foundation/project3_contextual_embeddings/train.py"
    )


def get_lr_multiplier(step: int, total_steps: int, warmup_steps: int) -> float:
    """
    Linear warmup followed by linear decay (paper's standard schedule).

    Args:
        step: Current optimizer step (0-indexed)
        total_steps: Total number of optimizer steps
        warmup_steps: Number of initial warmup steps

    Returns:
        Multiplier in [0, 1] applied to the base learning rate
    """
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))


def save_checkpoint(encoder, optimizer, step, loss, spearman):
    """Save the fine-tuned encoder."""
    path = Path(config.CHECKPOINT_DIR) / f"simcse_step{step}_spearman{spearman:.2f}.pt"
    torch.save(
        {
            "step": step,
            "loss": loss,
            "spearman": spearman,
            "model_state_dict": encoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )
    print(f"✓ Saved checkpoint to {path}")
    return str(path)


def train(p3_checkpoint: str = None):
    """
    Run unsupervised SimCSE training.

    Args:
        p3_checkpoint: Explicit path to the Project 3 encoder checkpoint
    """
    config.validate_config()
    config.print_config()

    device = config.DEVICE

    # ==================================================================
    # 1. DATA
    # ==================================================================
    tokenizer_dict = load_tokenizer()
    tokenizer = Tokenizer(tokenizer_dict)
    vocab_size = len(tokenizer_dict["vocab"])
    print(f"Vocabulary size: {vocab_size:,}")

    sentences = load_sentences(max_sentences=config.MAX_SENTENCES)
    rng = torch.Generator().manual_seed(42)
    perm = torch.randperm(len(sentences), generator=rng).tolist()
    sentences = [sentences[i] for i in perm]

    n_val = max(1, int(len(sentences) * config.VAL_RATIO))
    val_sentences = sentences[:n_val]
    train_sentences = sentences[n_val:]
    print(f"Train: {len(train_sentences):,} | Val: {len(val_sentences):,}")

    # ==================================================================
    # 2. ENCODER (Project 3 BERT + MLP head)
    # ==================================================================
    p3_path = find_p3_checkpoint(p3_checkpoint)
    _, p3_config = simcse_model._load_project3_bert()
    p3_config.VOCAB_SIZE = vocab_size

    encoder = SimCSEEncoder(p3_config).to(device)
    encoder.load_pretrained(p3_path)
    encoder.train()  # dropout ON - it IS the augmentation

    # Baseline: STS-B before SimCSE (raw MLM encoder, CLS embedding)
    print("\nEvaluating the MLM pre-trained encoder on STS-B (baseline)...")
    baseline = evaluate_stsb(encoder, split="validation")
    print(f"  Baseline STS-B dev Spearman: {baseline:.2f}")

    # ==================================================================
    # 3. OPTIMIZER + SCHEDULE
    # ==================================================================
    optimizer = torch.optim.AdamW(
        encoder.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    steps_per_epoch = math.ceil(len(train_sentences) / config.BATCH_SIZE)
    total_steps = steps_per_epoch * config.EPOCHS
    warmup_steps = int(total_steps * config.WARMUP_RATIO)
    print(f"Total steps: {total_steps} (warmup: {warmup_steps})")

    # ==================================================================
    # 4. TRAINING LOOP
    # ==================================================================
    best_spearman = -float("inf")
    best_path = None
    step = 0
    start = time.time()
    losses = []

    pbar = tqdm(range(config.EPOCHS), desc="SimCSE epochs")
    for epoch in pbar:
        for batch_start in range(0, len(train_sentences), config.BATCH_SIZE):
            batch = train_sentences[batch_start : batch_start + config.BATCH_SIZE]
            idx, attention_mask = tokenize_batch(batch, tokenizer)
            idx, attention_mask = idx.to(device), attention_mask.to(device)

            # Two forward passes = two independent dropout masks (the
            # entire "augmentation" of unsupervised SimCSE)
            z1 = encoder.encode(idx, attention_mask, project=config.USE_MLP_HEAD)
            z2 = encoder.encode(idx, attention_mask, project=config.USE_MLP_HEAD)

            loss = simcse_loss(z1, z2, config.TEMPERATURE)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)

            # Linear warmup + decay
            lr_mult = get_lr_multiplier(step, total_steps, warmup_steps)
            for group in optimizer.param_groups:
                group["lr"] = config.LEARNING_RATE * lr_mult
            optimizer.step()

            losses.append(loss.item())
            step += 1

            if step % config.LOG_INTERVAL == 0:
                pbar.set_postfix(
                    loss=f"{sum(losses[-config.LOG_INTERVAL:]) / config.LOG_INTERVAL:.4f}",
                    lr=f"{config.LEARNING_RATE * lr_mult:.2e}",
                )

            if step % config.EVAL_INTERVAL == 0:
                spearman = evaluate_stsb(encoder, split="validation")
                print(f"\n  step {step}: STS-B dev Spearman = {spearman:.2f}")
                if spearman > best_spearman:
                    best_spearman = spearman
                    best_path = save_checkpoint(
                        encoder, optimizer, step, loss.item(), spearman
                    )
                encoder.train()

    # ==================================================================
    # 5. FINAL EVALUATION
    # ==================================================================
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    final_spearman = evaluate_stsb(encoder, split="validation")
    print(f"  Baseline (MLM-only) STS-B dev: {baseline:.2f}")
    print(f"  Final SimCSE        STS-B dev: {final_spearman:.2f}")

    if final_spearman > best_spearman:
        best_spearman = final_spearman
        best_path = save_checkpoint(encoder, optimizer, step, loss.item(), final_spearman)

    print(f"\nBest checkpoint: {best_path} (Spearman {best_spearman:.2f})")
    print(f"Training took {(time.time() - start) / 60:.1f} min on {device}")
    return best_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train unsupervised SimCSE")
    parser.add_argument(
        "--p3_checkpoint",
        type=str,
        default=None,
        help="Path to the Project 3 MLM checkpoint (default: newest in checkpoints/project3)",
    )
    args = parser.parse_args()
    train(p3_checkpoint=args.p3_checkpoint)
