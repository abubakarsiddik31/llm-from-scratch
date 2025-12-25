# Project 1: Character-Level GPT from Scratch

## Objective

Build a GPT (Generative Pre-trained Transformer) model from scratch and train it on Shakespeare text. This project will give you deep understanding of:

1. **Self-Attention Mechanism** - The core of Transformer architecture
2. **Multi-Head Attention** - Parallel attention heads learning different patterns
3. **Transformer Block** - The complete building block with attention + feed-forward
4. **Positional Embeddings** - Giving the model sense of sequence order
5. **Training Loop** - Loss computation, backpropagation, optimization
6. **Text Generation** - Sampling strategies (temperature, top-k, top-p)

## Why Character-Level?

Starting with character-level modeling (instead of subword/word-level) has advantages:

- **Simpler tokenization**: No need for BPE - each character is a token
- **Smaller vocabulary**: ~65 characters vs 50k+ for subword models
- **Faster to train**: Can see results in hours, not days
- **Easier to debug**: You can visually inspect what the model learns
- **Complete implementation**: You understand every component

## Model Architecture

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

## Key Concepts Deep Dive

### 1. Self-Attention

Self-attention allows each token to "look at" all previous tokens to decide what information to incorporate.

```
For each token position:
- Query (Q): "What am I looking for?"
- Key (K): "What do I contain?"
- Value (V): "What information do I provide?"

Attention = softmax(Q × K^T / √d_k) × V
```

### 2. Causal Masking

We use a causal (triangular) mask so each token can only attend to previous tokens:
- Position 1 sees: [1]
- Position 2 sees: [1, 2]
- Position T sees: [1, 2, ..., T]

This is essential for autoregressive generation (GPT-style).

### 3. Multi-Head Attention

Instead of one attention mechanism, we use multiple "heads":
- Each head learns different relationship patterns
- Head 1 might learn syntax
- Head 2 might learn semantics
- Head 3 might learn pronoun references
- etc.

### 4. Residual Connections + LayerNorm

Each sub-layer (attention, FFN) is wrapped with:
- Residual connection: `x = x + Sublayer(x)` - helps gradient flow
- Layer normalization: stabilizes training

## Files in This Project

```
project1_minimal_gpt/
├── README.md              (this file - overview and concepts)
├── model.py               (complete model implementation)
├── train.py               (training script)
├── generate.py            (inference script)
└── config.py              (hyperparameters)
```

## Training Target

- **Dataset**: Shakespeare (~1M characters)
- **Vocabulary**: ~65 unique characters
- **Model size**: ~10M parameters (tiny by modern standards)
- **Training time**: ~30 minutes to 2 hours on GPU
- **Final loss**: Target ~1.5-1.8 (lower is better)

## What You'll See During Training

```
step 0: val loss 4.50    (random predictions)
step 500: val loss 2.80  (learning some patterns)
step 1000: val loss 2.20 (basic structure)
step 2000: val loss 1.90 (coherent text)
step 5000: val loss 1.60 (Shakespeare-like)
```

## Next Steps

1. Read through `model.py` - understand each component
2. Run `train.py` - see the model train in real-time
3. Experiment with `generate.py` - generate text with different settings
4. Modify hyperparameters in `config.py` - see effects on training

Let's dive in!
