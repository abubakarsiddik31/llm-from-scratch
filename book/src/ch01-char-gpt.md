# Character-Level GPT

> **Project source:** [`phase1_foundation/project1_minimal_gpt/`](https://github.com/abubakarsiddik31/llm-from-scratch/tree/main/phase1_foundation/project1_minimal_gpt)
>
> **Papers:** *"Attention Is All You Need"* (Vaswani et al., 2017) · *"Improving Language Understanding by Generative Pre-Training"* (GPT-1, 2018) · *"Language Models are Unsupervised Multitask Learners"* (GPT-2, 2019)

## Objective

Build a GPT model from scratch and train it on Shakespeare. By the end you
will have implemented, without libraries doing the work for you:

1. **Self-attention** — the core mechanism of the Transformer
2. **Multi-head attention** — parallel heads learning different patterns
3. **The transformer block** — attention + feed-forward with residuals
4. **Positional embeddings** — a sense of sequence order
5. **The training loop** — loss, backpropagation, optimization
6. **Text generation** — temperature, top-k sampling

## Why start at character level?

| | Character-level | Word/subword level |
|---|---|---|
| Tokenization | Trivial (each character is a token) | Needs BPE (Chapter 2) |
| Vocabulary | ~65 tokens | 50,000+ |
| Time to first result | Minutes | Hours |
| Debugging | You can visually inspect everything | Harder |

The model is small (~10M parameters) and the dataset is small (~1M
characters), so training completes in **under an hour on a consumer GPU** —
fast enough to iterate on ideas instead of waiting.

## Architecture

```
Input (B, T) → Token Embedding → Position Embedding → Add
    ↓
Transformer Blocks × N (each block has:)
    ├── Multi-Head Self-Attention
    ├── Add & Norm (Residual + LayerNorm)
    ├── Feed-Forward Network
    └── Add & Norm
    ↓
Final LayerNorm → Linear Head (vocab_size)
    ↓
Output Logits (B, T, vocab_size)
```

Note the **pre-LN** arrangement (LayerNorm *before* each sublayer) — the
GPT-2 style, which trains more stably than the original post-LN design.

## The core ideas

### Self-attention

Every token asks two questions: *what am I looking for?* (query) and *what do
I contain?* (key/value).

```
For each token position:
- Query (Q): "What am I looking for?"
- Key (K):   "What do I contain?"
- Value (V): "What information do I provide?"

Attention = softmax(Q × Kᵀ / √d_k) × V
```

The `√d_k` scaling keeps softmax inputs in a healthy range; without it,
logits grow with dimension and softmax saturates.

### Causal masking

GPT generates text left-to-right, so position *t* may only attend to
positions ≤ *t*. A triangular mask enforces this before the softmax:

- Position 1 sees: `[1]`
- Position 2 sees: `[1, 2]`
- Position T sees: `[1, 2, ..., T]`

This single change is the difference between GPT (generation) and BERT
(understanding) — Chapter 3 removes it.

### Multi-head attention

Instead of one attention computation, we run several in parallel on smaller
projected subspaces. Different heads specialize — one might track syntax,
another pronoun references. Their outputs are concatenated and projected back.

### Residuals + LayerNorm

Each sub-layer is wrapped as `x = x + Sublayer(LayerNorm(x))`. The residual
gives gradients a highway through the network; the normalization stabilizes
activation scales across depth.

## Training

```bash
uv run python phase1_foundation/project1_minimal_gpt/train.py
```

Typical trajectory on Shakespeare:

```
step 0:    val loss 4.50   (random predictions)
step 500:  val loss 2.80   (learning patterns)
step 1000: val loss 2.20   (basic structure)
step 2000: val loss 1.90   (coherent text)
step 5000: val loss 1.60   (Shakespeare-like)
```

Target: **1.5–1.8** validation loss in ~30 minutes–2 hours on GPU.

## Generation

```bash
# Single prompt
uv run python phase1_foundation/project1_minimal_gpt/generate.py \
    --prompt "ROMEO:" --temperature 0.8 --top_k 50

# Interactive mode (supports `temp <x>`, `topk <x>`, `clear`)
uv run python phase1_foundation/project1_minimal_gpt/generate.py --interactive
```

**Temperature** divides logits before softmax: values below 1 sharpen the
distribution (safer, more repetitive), values above 1 flatten it (riskier,
more diverse). **Top-k** restricts sampling to the *k* most likely tokens,
cutting off the long tail of implausible continuations.

## Code tour

| File | What to read |
|------|--------------|
| [`model.py`](https://github.com/abubakarsiddik31/llm-from-scratch/blob/main/phase1_foundation/project1_minimal_gpt/model.py) | `CausalSelfAttention`, the transformer block, the full GPT class |
| [`train.py`](https://github.com/abubakarsiddik31/llm-from-scratch/blob/main/phase1_foundation/project1_minimal_gpt/train.py) | Batching, loss, the training loop, checkpointing |
| [`generate.py`](https://github.com/abubakarsiddik31/llm-from-scratch/blob/main/phase1_foundation/project1_minimal_gpt/generate.py) | Autoregressive decoding, temperature/top-k |
| [`config.py`](https://github.com/abubakarsiddik31/llm-from-scratch/blob/main/phase1_foundation/project1_minimal_gpt/config.py) | Every hyperparameter, documented |

## Exercises

1. Set the causal mask to identity (no masking) and watch generation collapse
   into nonsense — the model now sees the future.
2. Change `N_HEAD` to 1. How does the loss curve change?
3. Set `temperature` to 0.1 and to 2.0, and compare output quality.

## What's next

The model understands *characters*, not *words*. Chapter 2 builds the
subword tokenizer that real LLMs use.
