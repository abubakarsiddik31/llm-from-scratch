# ruff: noqa
"""
STS-B Evaluation for SimCSE

Implements the standard SimCSE evaluation protocol following:
- "SimCSE: Simple Contrastive Learning of Sentence Embeddings"
  (Gao, Yao & Chen, EMNLP 2021)
- "Semantic Textual Similarity Benchmark" (Cer et al., 2017)

EVALUATION PROTOCOL (paper, section 4):
---------------------------------------
"We evaluate on several semantic textual similarity (STS) tasks...
We concatenate all STS-B sentences... For each sentence pair, we
compute cosine similarity between the two sentence embeddings, and
report Spearman correlation with the gold labels."

IMPLEMENTATION:
---------------
1. Encode ALL STS-B sentences (s1 and s2 of every pair) in one batched
   pass with the encoder in EVAL MODE (dropout OFF - deterministic
   embeddings) and WITHOUT the training-only MLP head.
2. For each pair, compute cosine similarity between the two embeddings.
3. Report Spearman rank correlation between similarities and human
   scores (0-5). Spearman (not Pearson) is standard because STS quality
   is judged by RANKING agreement.

PAPER REFERENCE NUMBERS (STS-B dev, BERT-base):
-----------------------------------------------
- Unsupervised SimCSE:  82.5 Spearman
- BERT-base-flow:       77.1
- BERT-base (raw CLS):  ~64
Our tiny 6-layer/256d encoder fine-tuned on 10^5 sentences will land
well below the paper's number - the pipeline and protocol are identical,
only the model capacity and data scale are reduced.
"""

from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

import config
from data import tokenize_batch


def load_sts_b(split: str = "validation"):
    """
    Load STS-B sentence pairs and gold scores.

    Args:
        split: "validation" (dev) or "test"

    Returns:
        (sentences1, sentences2, scores) as lists / numpy array
    """
    from datasets import load_dataset

    dataset = load_dataset("nyu-mll/glue", "stsb", split=split)
    return (
        list(dataset["sentence1"]),
        list(dataset["sentence2"]),
        np.array(dataset["label"], dtype=np.float32),
    )


@torch.no_grad()
def encode_sentences(encoder, sentences, batch_size=128, device=None):
    """
    Encode a list of sentences into embeddings (eval mode, no MLP head).

    DROPOUT IS OFF (encoder.eval()): embeddings must be deterministic at
    evaluation time - the two-view dropout trick is a TRAINING device.

    Args:
        encoder: SimCSEEncoder
        sentences: list of strings
        batch_size: encoding batch size
        device: torch device

    Returns:
        (N, D) numpy array of embeddings
    """
    device = device or config.DEVICE
    was_training = encoder.training
    encoder.eval()

    embeddings = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        idx, attention_mask = tokenize_batch(batch, max_len=config.MAX_SEQ_LEN)
        idx = idx.to(device)
        attention_mask = attention_mask.to(device)
        emb = encoder.encode(idx, attention_mask=attention_mask, project=False)
        embeddings.append(emb.cpu())

    if was_training:
        encoder.train()

    return torch.cat(embeddings, dim=0).numpy()


def evaluate_stsb(encoder, split: str = "validation", device=None) -> float:
    """
    Evaluate the encoder on STS-B and return Spearman correlation.

    Args:
        encoder: SimCSEEncoder
        split: "validation" or "test"
        device: torch device

    Returns:
        Spearman correlation x 100 (paper convention, e.g. 82.5)
    """
    from scipy.stats import spearmanr

    device = device or config.DEVICE
    s1, s2, gold = load_sts_b(split)

    # GLUE test-split labels are hidden (-1 placeholders); Spearman is
    # only defined on the labeled dev split.
    if (gold < 0).all():
        print(f"  ⚠ STS-B {split} has no public labels (GLUE leaderboard "
              f"only) - skipping. Use the validation split.")
        return float("nan")

    emb1 = encode_sentences(encoder, s1, device=device)
    emb2 = encode_sentences(encoder, s2, device=device)

    # Cosine similarity per pair (embeddings are unnormalized -> normalize)
    e1 = F.normalize(torch.from_numpy(emb1), p=2, dim=-1)
    e2 = F.normalize(torch.from_numpy(emb2), p=2, dim=-1)
    cosine = (e1 * e2).sum(dim=-1).numpy()

    spearman = spearmanr(cosine, gold).statistic
    return float(spearman) * 100


def main():
    """Run STS-B evaluation on a saved SimCSE checkpoint."""
    import argparse

    import model as simcse_model
    from data import load_tokenizer
    from model import SimCSEEncoder, load_checkpoint

    parser = argparse.ArgumentParser(description="Evaluate SimCSE on STS-B")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="SimCSE checkpoint (.pt). Default: newest in checkpoints/project3b",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="validation",
        help="Comma-separated STS-B splits to evaluate (test has hidden labels)",
    )
    args = parser.parse_args()

    # Find checkpoint
    if args.checkpoint is None:
        candidates = sorted(Path(config.CHECKPOINT_DIR).glob("simcse_*.pt"))
        if not candidates:
            print("✗ No SimCSE checkpoints found. Train first: uv run python .../train.py")
            raise SystemExit(1)
        args.checkpoint = str(candidates[-1])

    # Load tokenizer to size the vocabulary
    tokenizer = load_tokenizer(config.TOKENIZER_PATH)
    vocab_size = len(tokenizer["vocab"])

    # Build encoder (Project 3 config with the tokenizer's vocab size)
    _, p3_config = simcse_model._load_project3_bert()
    p3_config.VOCAB_SIZE = vocab_size
    encoder = SimCSEEncoder(p3_config).to(config.DEVICE)

    checkpoint = load_checkpoint(args.checkpoint)
    encoder.load_state_dict(checkpoint["model_state_dict"])
    print(f"✓ Loaded {args.checkpoint}")

    for split in args.splits.split(","):
        score = evaluate_stsb(encoder, split=split.strip())
        print(f"  STS-B {split:>10}: {score:.2f} Spearman")


if __name__ == "__main__":
    main()
