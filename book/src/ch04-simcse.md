# SimCSE: Sentence Embeddings

> **Project source:** [`phase1_foundation/project3b_simcse/`](https://github.com/abubakarsiddik31/llm-from-scratch/tree/main/phase1_foundation/project3b_simcse)
>
> **Papers:** *"SimCSE: Simple Contrastive Learning of Sentence Embeddings"* (Gao, Yao & Chen, EMNLP 2021) · *"Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere"* (Wang & Isola, 2020) · STS-B (Cer et al., 2017)

## The problem

Chapter 3's encoder produces contextual *token* embeddings. But downstream
tasks — semantic search, clustering, duplicate detection — need a single
vector per *sentence* where **similar sentences are close together**. Simply
averaging the encoder's outputs doesn't have that geometry.

SimCSE fixes it with an almost absurdly simple recipe:

> Feed the same sentence into the encoder **twice**. Because dropout masks are
> sampled independently, you get two slightly different embeddings. Train the
> encoder to pull those two together and push every *other* sentence in the
> batch away.

No paraphrase data. No translation pairs. **Dropout is the data
augmentation.** The paper puts it plainly:

> "dropout acts as minimal data augmentation, and removing it leads to a
> representation collapse."

## The objective

For a mini-batch of *N* sentences (equation 4 of the paper):

```
                  exp(sim(h_i^z, h_i^z')/τ)
l_i = -log -------------------------------------------
           Σ_j exp(sim(h_i^z, h_j^z')/τ)
```

| Symbol | Meaning |
|--------|---------|
| `h_i^z` | Embedding of sentence `x_i` under dropout mask `z` |
| `h_i^z'` | The same sentence under an independent mask `z'` |
| `sim(·,·)` | Cosine similarity |
| `τ` | Temperature — **0.05** for unsupervised SimCSE |

This is an *N*-way classification over the batch: each row of the similarity
matrix must find its own positive on the diagonal. The loss is symmetrized by
swapping the two views.

### Why it works: alignment and uniformity

Wang & Isola (2020) characterize good contrastive spaces with two metrics
(both implemented in `model.py`):

- **Alignment** — similar sentences map to nearby points (positives are close)
- **Uniformity** — embeddings spread evenly over the hypersphere

The paper's diagnostic: models with *no* dropout, or with a *fixed* dropout
mask shared by both copies, trivially make the two views identical. Their
alignment collapses to zero but their representations clump — uniformity is
destroyed and STS-B degrades dramatically. Independent dropout masks keep
alignment steady while still improving uniformity. Starting from a
pre-trained checkpoint matters too: "it provides good initial alignment."

## The "MLP trick"

During training, a one-layer MLP with tanh sits on top of the `[CLS]`
embedding, and the contrastive pressure flows through it. At **evaluation**
the head is discarded and the raw `[CLS]` embedding is used — worth 1–2
Spearman points. The intuition: let the loss shape the space through a
disposable projection, leaving the underlying embedding closer to its
pre-trained manifold.

## Evaluation: STS-B

The Semantic Textual Similarity Benchmark (Cer et al., 2017) provides 1,500
dev sentence pairs scored 0–5 by humans. Encode both sentences of every pair
(dropout OFF — deterministic), compute cosine similarity, and report the
**Spearman rank correlation** with the human scores. The paper's BERT-base
result: **82.5** dev Spearman.

> Note: the GLUE *test* split's labels are hidden (leaderboard only), so the
> dev split is the reportable number.

## Results

Trained on 100k Wikipedia sentences, batch 64, τ = 0.05, lr 5e-5, max
sequence length 32 — on an RTX 3050:

| Model | STS-B dev Spearman |
|---|---|
| Project 3 MLM encoder, raw `[CLS]` (baseline) | 16.6 |
| SimCSE, 1 epoch | 27.0 |
| **SimCSE, 3 epochs (final)** | **34.7** |
| SimCSE, 5 epochs (overfits) | 33.5 |

**+18.1 Spearman over baseline.** The paper's number is higher because their
model is 110M parameters on 10⁶ sentences; the *pipeline* here is theirs,
unchanged. Each epoch takes ~26 seconds on the GPU — hyperparameter
experiments are cheap.

## Running it

```bash
# Sentence corpus + STS-B
uv run python phase1_foundation/project3b_simcse/download_data.py

# Sanity-test the mechanics before training (runs anywhere, seconds)
uv run python phase1_foundation/project3b_simcse/test_model.py

# Train, then evaluate
uv run python phase1_foundation/project3b_simcse/train.py
uv run python phase1_foundation/project3b_simcse/evaluate.py
```

The sanity tests verify the machinery itself: two forward passes with
dropout ON produce *different* embeddings (if not, the positive pair is
degenerate); aligned positives yield lower loss than misaligned ones;
gradients reach both the encoder and the MLP head.

## Code tour

| File | What to read |
|------|--------------|
| [`model.py`](https://github.com/abubakarsiddik31/llm-from-scratch/blob/main/phase1_foundation/project3b_simcse/model.py) | `SimCSEEncoder` (wraps Project 3's BERT), `MLPHead`, `simcse_loss`, `alignment_and_uniformity` |
| [`data.py`](https://github.com/abubakarsiddik31/llm-from-scratch/blob/main/phase1_foundation/project3b_simcse/data.py) | Sentence loading, `[CLS] s [SEP] [PAD]` batching |
| [`train.py`](https://github.com/abubakarsiddik31/llm-from-scratch/blob/main/phase1_foundation/project3b_simcse/train.py) | Two-view training loop, warmup + linear decay |
| [`evaluate.py`](https://github.com/abubakarsiddik31/llm-from-scratch/blob/main/phase1_foundation/project3b_simcse/evaluate.py) | STS-B protocol |
| [`test_model.py`](https://github.com/abubakarsiddik31/llm-from-scratch/blob/main/phase1_foundation/project3b_simcse/test_model.py) | The sanity suite |

## Exercises

1. Turn dropout off (`p=0`) and retrain. Watch the collapse the paper
   predicts (Section 3, Table 3).
2. Swap `[CLS]` pooling for mean pooling. Which wins at this scale?
3. Train on the full extracted corpus (raise `MAX_SENTENCES`) — does more
   data close the gap faster than more epochs?

## What's next

Phase 2 applies the same from-scratch discipline to fine-tuning: SFT, LoRA,
and DPO. Phase 3 then makes inference fast: mixed precision, KV-cache, and
Flash Attention.
