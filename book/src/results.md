# Results Log

Measured results from real runs in this repository. Hardware: NVIDIA
GeForce RTX 3050 (8 GB), PyTorch 2.11 + CUDA 12.8. Models and datasets are
deliberately small so every number is reproducible on a laptop GPU.

## Chapter 1 — Character-Level GPT

| Metric | Value |
|---|---|
| Dataset | Tiny Shakespeare (~1M chars) |
| Vocabulary | ~65 characters |
| Model size | ~10M parameters |
| Training time | < 1 hour on GPU |
| Target validation loss | 1.5–1.8 |

## Chapter 2 — BPE Tokenizer

| Metric | Value |
|---|---|
| Corpus | WikiText-2 (~11 MB) |
| Vocabulary | 5,000 tokens (3,983 learned merges) |
| Training time (optimized loop) | ~12 minutes |
| Training time (naive full-rescan loop) | ~5 hours |
| Speedup from distinct-word representation | ~50× |

The optimization: count adjacent pairs over **distinct words with
multiplicities** instead of rescanning the full corpus per merge.

## Chapter 3 — BERT-style Encoder

| Metric | Value |
|---|---|
| Corpus | WikiText-2 |
| Model size | 4.8M parameters (6 layers, 256 hidden, 8 heads) |
| Iterations | 10,000 (batch 32, block 128) |
| Training time | ~12 minutes on GPU |
| MLM loss | 4.2 → ~1.4 (train), best val 2.85 |

## Chapter 4 — SimCSE

Fine-tuned from the Chapter 3 checkpoint on 100k Wikipedia sentences,
batch 64, τ = 0.05, lr 5e-5, max sequence length 32.

| Model | STS-B dev Spearman ↑ |
|---|---|
| MLM encoder, raw `[CLS]` (baseline) | 16.6 |
| SimCSE, 1 epoch | 27.0 |
| **SimCSE, 3 epochs (final)** | **34.7** |
| SimCSE, 5 epochs | 33.5 |

| Metric | Value |
|---|---|
| Time per epoch | ~26 s |
| Improvement over baseline | **+18.1 Spearman** |
| Paper reference (BERT-base, 110M params, 10⁶ sentences) | 82.5 |

Evaluation protocol: cosine similarity vs human scores, Spearman rank
correlation, dropout OFF, without the training-only MLP head. STS-B *test*
labels are hidden (GLUE leaderboard only), so dev is the reportable split.

## Reproducing

Every number above comes from the commands in
[Getting Set Up](./setup.md) with default configurations. Checkpoints are
written to `checkpoints/` (gitignored); training logs are deterministic
modulo CUDA nondeterminism.
