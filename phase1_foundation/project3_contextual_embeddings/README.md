# Project 3: Contextual Embeddings (BERT-Style)

<div align="center">

### Bidirectional Transformer with Masked Language Modeling

**Implementing BERT-style contextual embeddings from scratch**

</div>

---

## Overview

This project implements a BERT-style bidirectional transformer with Masked Language Modeling (MLM) and Next Sentence Prediction (NSP) for pre-training contextual embeddings.

**Papers:**
- "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (Devlin et al., 2018)
- "Attention is All You Need" (Vaswani et al., 2017)

---

## Key Concepts

### BERT vs GPT

| Aspect | GPT (Project 1) | BERT (This Project) |
|--------|----------------|---------------------|
| **Direction** | Unidirectional (causal) | Bidirectional |
| **Attention** | Causal masking | No masking (full context) |
| **Task** | Next token prediction | Masked token prediction |
| **Output** | Text generation | Contextual embeddings |
| **Use Case** | Generation | Understanding/classification |

### BERT Architecture

```
Input: [CLS] Sentence A [SEP] Sentence B [SEP]
   ↓
Token Embeddings + Position Embeddings + Segment Embeddings
   ↓
Transformer Encoder Blocks × N (Bidirectional Self-Attention)
   ↓
Output Heads:
  - MLM Head: Predict masked tokens (vocab_size output)
  - NSP Head: Predict next sentence (binary output)
```

### Pre-training Tasks

1. **Masked Language Modeling (MLM)**
   - Randomly mask 15% of tokens
   - 80% → Replace with ั token
   - 10% → Replace with random token
   - 10% → Keep original token
   - Predict original tokens at masked positions

2. **Next Sentence Prediction (NSP)**
   - Given two sentences (A, B)
   - Predict: Is B the actual next sentence?
   - 50% positive, 50% negative pairs

---

## Files

| File | Description |
|------|-------------|
| `config.py` | BERT-style hyperparameters and configuration |
| `model.py` | BERT model implementation with bidirectional attention |
| `train.py` | MLM + NSP training loop |
| `test_model.py` | Validation and testing scripts |
| `download_data.py` | Data acquisition for pre-training |

---

## Quick Start

```bash
# 1. Download training data
uv run python phase1_foundation/project3_contextual_embeddings/download_data.py --size small

# 2. Train the model
uv run python phase1_foundation/project3_contextual_embeddings/train.py

# 3. Test the model
uv run python phase1_foundation/project3_contextual_embeddings/test_model.py
```

---

## Configuration

```python
# Model Architecture
N_EMBD = 256        # Embedding dimension (BERT-base: 768)
N_HEAD = 8          # Number of attention heads (BERT-base: 12)
N_LAYER = 6         # Number of transformer blocks (BERT-base: 12)
BLOCK_SIZE = 128    # Maximum sequence length (BERT: 512)

# MLM Configuration
MLM_PROB = 0.15     # 15% of tokens masked
MASK_PROB = 0.8     # 80% → ั token
RANDOM_PROB = 0.1   # 10% → random token
KEEP_PROB = 0.1     # 10% → keep original

# Training
BATCH_SIZE = 32
MAX_ITERS = 10000
LEARNING_RATE = 2e-4
```

---

## Model Architecture Details

### Bidirectional Self-Attention

**Key difference from GPT:** No causal mask!

```python
# GPT: Each token attends to previous tokens only
att = att.masked_fill(mask == 0, float("-inf"))  # Causal mask

# BERT: Each token attends to ALL tokens
# No masking! Full context at each position
```

**Why Bidirectional?**

For sentence "The cat sat on the mat":

- GPT at "sat": Sees ["The", "cat", "sat"]
- BERT at "sat": Sees ["The", "cat", "sat", "on", "the", "mat"]

Full context enables better understanding for:
- Classification (sentiment, topic)
- NER (named entity recognition)
- QA (question answering)

### Segment Embeddings

BERT distinguishes sentence pairs with segment embeddings:

```
[CLS] Sentence A [SEP] Sentence B [SEP]
  0      0  ...  0    0      1  ...  1    1
  ↓      ↓      ↓    ↓      ↓      ↓    ↓
Segment: 0      0  ...  0    0      1  ...  1    1
```

- Sentence A tokens: segment_id = 0
- Sentence B tokens: segment_id = 1

### Special Tokens

| Token | ID | Purpose |
|-------|-----|---------|
| `[CLS]` | 2 | Classification token (sentence start) |
| `[SEP]` | 3 | Separator token (sentence boundary) |
| `[MASK]` | 4 | Mask token for MLM |
| `[PAD]` | 0 | Padding token |
| `[UNK]` | 1 | Unknown token |

---

## Training Process

### 1. Data Preparation

```python
# Create sentence pairs for NSP
pairs = create_sentence_pairs(text)

# Apply MLM masking
masked_pairs, labels = mask_tokens(pairs)
```

### 2. Forward Pass

```python
mlm_logits, nsp_logits, mlm_loss, nsp_loss = model(
    input_ids,
    segment_ids,
    masked_labels,
    nsp_labels
)
```

### 3. Loss Computation

```python
# MLM Loss: Cross-entropy on masked positions only
mlm_loss = F.cross_entropy(
    mlm_logits.view(-1, vocab_size),
    masked_labels.view(-1),
    ignore_index=-100  # Ignore non-masked positions
)

# NSP Loss: Binary cross-entropy
nsp_loss = F.cross_entropy(nsp_logits, nsp_labels)

# Total loss
loss = mlm_loss + nsp_loss
```

---

## Usage Examples

### MLM Prediction

```python
# Input: "The [MASK] sat on the mat."
# Model predicts: "cat", "dog", "rat", etc.

masked_input = "[CLS] The े sat on the mat . [SEP]"
input_ids = tokenizer.encode(masked_input)

mlm_logits, _, _, _ = model(input_ids, segment_ids)
predictions = torch.argmax(mlm_logits, dim=-1)

# Get prediction for mask position
mask_pos = input_ids.index(MASK_TOKEN_ID)
predicted_token = tokenizer.decode([predictions[0, mask_pos]])
print(f"Predicted: {predicted_token}")  # "cat"
```

### NSP Prediction

```python
# Sentence A: "The sky is blue."
# Sentence B: "Grass is green." (IsNext)
# Sentence C: "Python is a language." (NotNext)

sentence_a = "[CLS] The sky is blue . [SEP]"
sentence_b = "[SEP] Grass is green . [SEP]"

input_ids = tokenizer.encode(sentence_a + sentence_b)
segment_ids = [0] * len(sentence_a) + [1] * len(sentence_b)

_, nsp_logits, _, _ = model(input_ids, segment_ids)
is_next = torch.argmax(nsp_logits, dim=-1)

print(f"Prediction: {'IsNext' if is_next == 1 else 'NotNext'}")
```

### Extracting Contextual Embeddings

```python
# For downstream tasks, extract contextual embeddings

# Forward pass through encoder
with torch.no_grad():
    mlm_logits, _, _, _ = model(input_ids, segment_ids)

# Get [CLS] embedding for classification
cls_embedding = model.blocks[-1].output[:, 0, :]  # [CLS] position

# Get token embeddings for NER/QA
token_embeddings = model.blocks[-1].output[:, 1:-1, :]  # All tokens
```

---

## Paper References

1. **BERT** (Devlin et al., 2018)
   - "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
   - Key contribution: Bidirectional pre-training with MLM

2. **Transformer** (Vaswani et al., 2017)
   - "Attention is All You Need"
   - Key contribution: Self-attention mechanism

3. **Layer Normalization** (Ba et al., 2016)
   - Used for stable training

4. **Adam Optimizer** (Kingma & Ba, 2014)
   - Used for optimization

---

## Model Sizes

| Model | Layers | Hidden | Heads | Parameters |
|-------|--------|--------|-------|------------|
| BERT-tiny (this) | 6 | 256 | 8 | ~10M |
| BERT-base | 12 | 768 | 12 | 110M |
| BERT-large | 24 | 1024 | 16 | 340M |

---

## Success Criteria

- [x] Bidirectional self-attention (no causal mask)
- [x] Masked Language Modeling (MLM) training
- [x] Next Sentence Prediction (NSP) training
- [x] Segment embeddings for sentence pairs
- [x] Special tokens ([CLS], [SEP], [MASK])
- [x] Contextual embedding extraction
- [x] Extensive documentation with paper references

---

## Next Steps

After completing this project, you'll have:
1. Understanding of bidirectional vs unidirectional attention
2. Knowledge of MLM pre-training objective
3. Contextual embeddings for downstream tasks
4. Foundation for fine-tuning on specific tasks

**Next Project:** Project 4 - Pre-train 125M Model
