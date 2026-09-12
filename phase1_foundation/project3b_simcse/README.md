# Project 3B: SimCSE — Sentence Embeddings

<div align="center">

### Unsupervised Contrastive Learning of Sentence Embeddings

**Dropout is all you need: the same sentence is its own positive pair**

</div>

---

## Overview

Fine-tunes the Project 3 BERT encoder into a **sentence embedding** model using
unsupervised SimCSE. No labels, no paraphrase data — just the encoder seeing
each sentence **twice with different dropout masks**.

**Papers:**
- "SimCSE: Simple Contrastive Learning of Sentence Embeddings" (Gao, Yao & Chen, EMNLP 2021)
- "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2018) — encoder

---

## Results (this repo, RTX 3050)

| Model | STS-B dev Spearman |
|---|---|
| Project 3 MLM encoder, raw `[CLS]` (baseline) | 16.6 |
| SimCSE, 1 epoch | 27.0 |
| **SimCSE, 3 epochs (final config)** | **34.7** |
| SimCSE, 5 epochs (overfits) | 33.5 |

Checkpoint: `checkpoints/project3b/simcse_step1104_spearman34.67.pt`
(+18.1 Spearman over baseline). Note: STS-B **test** labels are hidden
(GLUE leaderboard only) — the dev split is the reportable number.

For scale: the paper's BERT-base (110M params, 10⁶ sentences) reaches
82.5 with the identical pipeline.

---

## Key Concepts

### The SimCSE Objective (paper, equation 4)

```
                 exp(sim(h_i^z, h_i^z')/τ)
l_i = -log -------------------------------------------
           Σ_j exp(sim(h_i^z, h_j^z')/τ)
```

| Symbol | Meaning |
|--------|---------|
| `h_i^z` | Embedding of sentence `x_i` under dropout mask `z` |
| `h_i^z'` | The SAME sentence under an independently sampled mask `z'` |
| `sim(·,·)` | Cosine similarity |
| `τ` | Temperature (0.05 for unsupervised SimCSE) |

**The positive pair is `(x_i, x_i)` itself.** All other batch sentences are
negatives. This only works because dropout is sampled independently in the two
forward passes — the paper shows that "fixed" dropout or no dropout leads to
representation collapse (their Table 3).

### The "MLP Trick"

A one-layer MLP with tanh sits on top of the `[CLS]` embedding **during
training only**. The contrastive pressure shapes the space through the
projection; at evaluation the raw `[CLS]` embedding is used (worth ~1-2
Spearman points).

### Evaluation: STS-B

Spearman rank correlation between cosine similarity of embeddings and human
semantic-relatedness scores (0-5) on 1,500 dev / 1,379 test pairs.

| Model (paper, BERT-base) | STS-B dev Spearman |
|---|---|
| BERT-base raw `[CLS]` | ~64 |
| Unsupervised SimCSE | **82.5** |

Our encoder is ~50× smaller (6 layers, 256 hidden) and trains on 10⁵ sentences
instead of 10⁶, so our absolute numbers are lower — the pipeline, objective and
protocol are the paper's, unchanged.

---

## Pipeline

```
WikiText-2 ──▶ sentence splitter ──▶ 100k sentences
                                        │
Project 3 BERT (MLM pre-trained) ───────┤
                                        ▼
        SimCSE fine-tuning: same batch ▶ two dropout views
                                        ▶ contrastive loss (τ=0.05)
                                        ▶ 1 epoch, AdamW, warmup+decay
                                        ▼
        STS-B: cosine sim vs human scores ▶ Spearman
```

---

## Usage

```bash
# 0. Prerequisites (Projects 2 & 3 artifacts)
uv run python phase1_foundation/project2_tokenizer/train_tokenizer.py --vocab_size 5000
uv run python phase1_foundation/project3_contextual_embeddings/download_data.py --size small
uv run python phase1_foundation/project3_contextual_embeddings/train.py

# 1. Prepare SimCSE data (sentence corpus + STS-B)
uv run python phase1_foundation/project3b_simcse/download_data.py

# 2. Sanity-check the implementation (fast, runs anywhere)
uv run python phase1_foundation/project3b_simcse/test_model.py

# 3. Train SimCSE on the GPU
uv run python phase1_foundation/project3b_simcse/train.py

# 4. Evaluate a checkpoint on STS-B
uv run python phase1_foundation/project3b_simcse/evaluate.py --splits validation,test
```

---

## Files

| File | Purpose |
|------|---------|
| `config.py` | All hyperparameters with paper references (`TEMPERATURE=0.05`, `BATCH_SIZE=64`, `MAX_SEQ_LEN=32`, ...) |
| `model.py` | `SimCSEEncoder` (wraps Project 3 BERT, CLS/mean pooling), `MLPHead`, `simcse_loss`, alignment/uniformity metrics |
| `data.py` | Sentence corpus loading, BPE tokenization, BERT-style `[CLS] s [SEP] [PAD]` batching |
| `download_data.py` | Sentence extraction from WikiText-2 + STS-B download |
| `train.py` | Two-view contrastive training loop with warmup/decay + periodic STS-B eval |
| `evaluate.py` | STS-B Spearman evaluation protocol |
| `test_model.py` | Sanity tests: dropout independence, positive ranking, gradient flow, MLP trick |

---

## Implementation Notes

- **Encoder reuse:** Project 3's BERT class is loaded via `importlib` with its
  config module isolated, so both projects keep independent configurations.
- **Pooling:** `[CLS]` (position 0) by default; `pooler="mean"` (non-padding
  positions) available. Project 3's attention has no padding mask, so PAD
  tokens participate in attention — identical across the two views, so
  contrastive training is unaffected.
- **Tokenizer convention:** the Project 2 BPE tokenizer's special tokens are
  `<PAD>/<UNK>/<BOS>/<EOS>` at ids 0-3; SimCSE frames sequences by position
  (0 = pad, 2 = `[CLS]`, 3 = `[SEP]`) rather than by name.
- **Determinism:** evaluation runs with dropout OFF; the two-view dropout
  trick exists only during training.

---

## References

1. Gao, T., Yao, X., & Chen, D. (2021). [SimCSE: Simple Contrastive Learning of Sentence Embeddings](https://arxiv.org/abs/2104.08821). EMNLP 2021.
2. Cer, D., et al. (2017). [SemEval-2017 Task 1: Semantic Textual Similarity](https://alt.qcri.org/semeval2017/task1/) (STS-B).
3. Devlin, J., et al. (2018). [BERT](https://arxiv.org/abs/1810.04805).
4. Wang, T., & Isola, P. (2020). [Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere](https://arxiv.org/abs/2005.10242) (the alignment/uniformity metrics).
