# Contextual Embeddings: BERT from Scratch

> **Project source:** [`phase1_foundation/project3_contextual_embeddings/`](https://github.com/abubakarsiddik31/llm-from-scratch/tree/main/phase1_foundation/project3_contextual_embeddings)
>
> **Papers:** *"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"* (Devlin et al., 2018) · *"Attention Is All You Need"* (Vaswani et al., 2017)

## From generation to understanding

GPT is a left-to-right generator: each token sees only its past. For
*understanding* tasks — classification, similarity, retrieval — that is a
handicap. BERT reads in both directions and learns by **predicting words it
cannot see**.

| Aspect | GPT (Chapter 1) | BERT (this chapter) |
|--------|-----------------|---------------------|
| Direction | Unidirectional (causal) | Bidirectional |
| Attention | Causal mask | Full context |
| Pre-training task | Next-token prediction | Masked token prediction |
| Output | Text generation | Contextual embeddings |
| Use case | Generation | Understanding / classification |

## Architecture

```
Input: [CLS] Sentence A [SEP] Sentence B [SEP]
   ↓
Token Embeddings + Position Embeddings + Segment Embeddings
   ↓
Transformer Encoder Blocks × N   (bidirectional self-attention)
   ↓
Output Heads:
  - MLM Head: predict masked tokens  (vocab_size outputs)
  - NSP Head: predict next sentence  (binary output)
```

Two things are new versus Chapter 1:

- **Segment embeddings.** A learned embedding distinguishes "belongs to
  sentence A" from "belongs to sentence B", added to token + position
  embeddings. Sentence A tokens get segment id 0, sentence B tokens id 1.
- **Two heads.** The MLM head projects every position to the vocabulary; the
  NSP head classifies the `[CLS]` position into *is-next / not-next*.

## Pre-training tasks

### Masked Language Modeling (MLM)

From the BERT paper — mask 15% of tokens, but not always with `[MASK]`:

| Probability | Replacement | Why |
|---|---|---|
| 80% | `[MASK]` | The actual task |
| 10% | Random token | Force the model to *not* trust any token |
| 10% | Keep original | There is a mismatch between pre-training and fine-tuning; this reduces it |

The model predicts the *original* tokens at masked positions — cross-entropy
over masked positions only.

### Next Sentence Prediction (NSP)

50% of the time, sentence B genuinely follows A; 50% it is a random sentence.
The model classifies which. *(Later work suggested NSP is not necessary —
RoBERTa dropped it — but we keep it for completeness.)*

## Configuration

`config.py` documents every choice against the paper. The demonstration
model is deliberately small:

| Hyperparameter | BERT-base | This project |
|---|---|---|
| Layers | 12 | 6 |
| Hidden size | 768 | 256 |
| Heads | 12 | 8 |
| Params | 110M | ~4.8M |
| MLM probability | 15% | 15% |

## Running it

```bash
uv run python phase1_foundation/project3_contextual_embeddings/download_data.py --size small
uv run python phase1_foundation/project3_contextual_embeddings/train.py
```

Checkpoints land in `checkpoints/project3/` as `bert_iter{N}_loss{L}.pt`
containing model + optimizer state and the loss. On the RTX 3050, 10,000
iterations train in about **12 minutes** (MLM loss 4.2 → ~1.4).

## Code tour

| File | What to read |
|------|--------------|
| [`model.py`](https://github.com/abubakarsiddik31/llm-from-scratch/blob/main/phase1_foundation/project3_contextual_embeddings/model.py) | `BidirectionalSelfAttention` (no causal mask!), `TransformerEncoderBlock`, the `BERT` class |
| [`train.py`](https://github.com/abubakarsiddik31/llm-from-scratch/blob/main/phase1_foundation/project3_contextual_embeddings/train.py) | `mask_tokens` (the 80/10/10 scheme), NSP pair construction, the training loop |
| [`config.py`](https://github.com/abubakarsiddik31/llm-from-scratch/blob/main/phase1_foundation/project3_contextual_embeddings/config.py) | All hyperparameters with paper quotes |

## Exercises

1. Set `MLM_PROB` to 0.5 — how does training stability change?
2. Disable the random-token branch (make it 90/10/0) and compare MLM loss.
3. (Concept) Why can't this model generate text? Trace the forward pass.

## What's next

A pre-trained encoder produces *token* embeddings. Chapter 4 shapes them into
*sentence* embeddings with contrastive learning — where the data augmentation
is dropout itself.
