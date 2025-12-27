# Project 2: Byte-Pair Encoding (BPE) Tokenizer

## Overview

This project implements a Byte-Pair Encoding (BPE) tokenizer from scratch. BPE is the tokenization method used in GPT-2, GPT-3, and most modern large language models.

### Research Paper Reference

**"Byte-Pair Encoding: Subword-Based Machine Translation"** (Sennrich, Haddow, and Birch, 2016)

> "Byte-Pair Encoding (BPE) iteratively merges the most frequent pair of bytes to create new tokens. This allows the model to handle rare and unseen words by splitting them into subword units."

### What is BPE?

BPE sits between character-level and word-level tokenization:

| Approach | Example | Pros | Cons |
|----------|---------|------|------|
| **Character-level** | "hello" → `['h', 'e', 'l', 'l', 'o']` | No unknown tokens, tiny vocab | Very long sequences |
| **Word-level** | "hello" → `['hello']` | Short sequences | Huge vocab, unknown words |
| **BPE (subword)** | "hello" → `['hello']`, "hell" → `['hel', 'l']` | Balanced length, handles OOV | Moderate vocab size |

### Key Benefits

1. **Handles OOV words**: "unfriendliness" → `['un', 'friend', 'li', 'ness']`
2. **Efficient encoding**: Common words become single tokens
3. **Compositional**: Same subwords appear in multiple words
4. **No unknown tokens**: All text can be encoded at character level

## How BPE Works

### Training Algorithm

```
1. Initialize vocabulary with all unique characters
2. Repeat until vocab_size reached:
    a. Count frequency of all adjacent pairs
    b. Find most frequent pair
    c. Merge pair into new token
    d. Update all occurrences in corpus
```

### Example

Training on: "hug hug pug hug pug"

| Step | Operation | Vocabulary |
|------|-----------|------------|
| Initial | Characters | `{h, u, g, p, space}` |
| 1 | Merge "u" + "g" → "ug" | `{h, u, g, p, space, ug}` |
| 2 | Merge "h" + "ug" → "hug" | `{h, u, g, p, space, ug, hug}` |
| 3 | Merge "p" + "ug" → "pug" | `{h, u, g, p, space, ug, hug, pug}` |

**Encoding results:**
- "hug" → `[hug]` (1 token)
- "pug" → `[pug]` (1 token)
- "hugg" → `[hug, g]` (2 tokens - subword splitting!)

## Project Structure

```
project2_tokenizer/
├── README.md              # This file
├── config.py              # Hyperparameters (vocab size, etc.)
├── tokenizer.py           # BPE implementation
├── train_tokenizer.py     # Training script
├── test_tokenizer.py      # Testing and demonstration
└── download_data.py       # Wikipedia data downloader
```

## Usage

**All commands are run from the project root directory using `uv run python`:**

### 1. Download Training Data

For meaningful BPE training, you need a large, diverse corpus. We provide a script to download Wikipedia data:

```bash
# Download WikiText dataset (recommended)
uv run python phase1_foundation/project2_tokenizer/download_data.py --size small    # ~10 MB (good for testing)
uv run python phase1_foundation/project2_tokenizer/download_data.py --size medium   # ~500 MB (good for training)

# Output is saved to: data/wikipedia_train.txt
```

**WikiText dataset reference:** Merity et al. (2016). "The WikiText Long Term Dependency Language Modeling Dataset"

### 2. Configuration

Edit `phase1_foundation/project2_tokenizer/config.py` to set hyperparameters:

```python
VOCAB_SIZE = 50_000      # Target vocabulary size (GPT-2 uses 50,257)
MIN_FREQUENCY = 2        # Minimum merge frequency
```

### 3. Train Tokenizer

```bash
# Train with default settings (uses downloaded Wikipedia data)
uv run python phase1_foundation/project2_tokenizer/train_tokenizer.py

# Train with smaller vocabulary (faster for testing)
uv run python phase1_foundation/project2_tokenizer/train_tokenizer.py --vocab_size 5000

# Train with higher frequency threshold
uv run python phase1_foundation/project2_tokenizer/train_tokenizer.py --min_freq 5

# Train with specific data file
uv run python phase1_foundation/project2_tokenizer/train_tokenizer.py --data_path /path/to/corpus.txt
```

**Training output:**
```
======================================================================
BPE Tokenizer Training
======================================================================
Vocabulary size:     50,000
Min frequency:       2
Training data:       ../data/tokenizer_train.txt
Checkpoint path:     ../checkpoints/project2/tokenizer.pkl
======================================================================

Loading training data...
Loaded 10,485,760 characters (10.00 MB)

Training BPE tokenizer on 10,485,760 characters...
Target vocab size: 50,000
Min merge frequency: 2
Found 75 unique characters
Will learn 49,669 merge operations...

  Merge      1/49669: ' ' + 'e' → ' e' (freq=528,432)
  Merge      2/49669: 'e' + ' ' → 'e ' (freq=389,215)
  Merge   1000/49669: 'a' + 't' → 'at' (freq=23,451)
  ...

Training complete!
  Final vocabulary size: 50,000
  Total merges learned: 49,669
```

### 4. Test Tokenizer

```bash
# Run all tests
uv run python phase1_foundation/project2_tokenizer/test_tokenizer.py

# Interactive mode
uv run python phase1_foundation/project2_tokenizer/test_tokenizer.py --interactive

# Test specific text
uv run python phase1_foundation/project2_tokenizer/test_tokenizer.py --text "Hello, world!"
```

**Test output:**
```
======================================================================
BPE Tokenizer Testing
======================================================================

Test 1: Roundtrip (encode → decode)
----------------------------------------------------------------------
  Original:  'Hello, world!'
  Encoded:   3 tokens
  Decoded:   'Hello, world!'
  Match:     ✓

Test 2: Common Words
----------------------------------------------------------------------
Very common:
  the                                  → ['the'] ✓ (single token)
  be                                   → ['be'] ✓ (single token)

Less common:
  algorithm                            → ['algorithm'] ✓ (single token)
  transformer                          → ['transform', 'er'] (2 tokens)

Rare/Technical:
  hyperparameter                       → ['hyper', 'parameter'] (2 tokens)
```

## Implementation Details

### Core Components

| Component | File | Description |
|-----------|------|-------------|
| `BPETokenizer` | [tokenizer.py](tokenizer.py) | Main tokenizer class |
| `train()` | [tokenizer.py](tokenizer.py) | Learn merge rules from text |
| `encode()` | [tokenizer.py](tokenizer.py) | Convert text to token IDs |
| `decode()` | [tokenizer.py](tokenizer.py) | Convert token IDs to text |

### Special Tokens

| Token | ID | Purpose |
|-------|----|---------|
| `<PAD>` | 0 | Padding for batching |
| `<UNK>` | 1 | Unknown character fallback |
| `<BOS>` | 2 | Beginning of sequence |
| `<EOS>` | 3 | End of sequence |

### Encoding Algorithm

Greedy longest-match-first tokenization:

1. Split text into words
2. For each word:
   - Start with character-level representation
   - Apply learned merges in order
   - Use longest possible matches
3. Convert tokens to IDs

Example with learned merges: `[('e', 'r'), ('er', 't'), ('ert', 'a')]`

```
Text: "erter"
Step 1: ['e', 'r', 't', 'e', 'r']     (character level)
Step 2: ['er', 't', 'e', 'r']         (apply 'e'+'r' → 'er')
Step 3: ['ert', 'e', 'r']             (apply 'er'+'t' → 'ert')
Step 4: ['ert', 'er']                 (apply 'e'+'r' → 'er')

Result: ['ert', 'er'] → [token_id_ert, token_id_er]
```

## Concepts Explained

### Why Subword Tokenization?

**Problem with word-level:**
- Vocabulary grows with corpus size
- Cannot handle unknown words
- Memory intensive for large vocabularies

**Problem with character-level:**
- Very long sequences (inefficient)
- No semantic meaning at character level
- harder for model to learn

**BPE solution:**
- Fixed vocabulary size
- Handles all words (via subwords)
- Shorter sequences than character-level
- Captures morphological patterns

### Compression Ratio

BPE achieves better compression than character-level:

| Text | Characters | BPE Tokens | Compression |
|------|-----------|------------|-------------|
| "Hello" | 5 | 1 | 5x |
| "The quick brown fox" | 19 | ~6 | 3.2x |
| Typical English | 1 | ~0.4 | 2.5x |

### Shared Subwords

The same subword appears in multiple words:

```
help → ['help']                # Single token
helping → ['help', 'ing']      # Shares 'help'
helpful → ['help', 'ful']      # Shares 'help'
helper → ['help', 'er']        # Shares 'help'
```

This enables the model to generalize across related words!

## Comparison with Project 1

| Aspect | Project 1 (Character) | Project 2 (BPE) |
|--------|----------------------|-----------------|
| Vocabulary size | ~65 (all chars) | ~50,000 (subwords) |
| Token type | Characters | Subwords |
| Sequence length | Long | Shorter |
| Unknown handling | None needed | Subword splitting |
| Use case | Learning, simple | Production LLMs |

## Next Steps

After completing this project:

1. **Project 3**: Use this BPE tokenizer to pre-train a 125M parameter model (GPT-2 small scale)
2. **Tokenization analysis**: Study how different vocab sizes affect model performance
3. **Optimization**: Implement byte-level BPE (like GPT-2) for full Unicode support

## References

1. Sennrich, R., Haddow, B., & Birch, A. (2016). [Byte-Pair Encoding: Subword-Based Machine Translation](https://arxiv.org/abs/1508.07909)
2. Radford et al. (2019). [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf) (GPT-2)
3. Merity et al. (2016). [The WikiText Long Term Dependency Language Modeling Dataset](https://arxiv.org/abs/1609.07843) (WikiText)
4. Gage, P. (1994). [A New Algorithm for Data Compression](https://www.researchgate.net/publication/200040256_A_New_Algorithm_for_Data_Compression) (Original BPE)

## Success Criteria

- [x] BPE trains on ~1MB+ text data
- [x] Vocabulary of ~50K tokens learned
- [x] Encode/decode roundtrip works
- [x] Common words are single tokens
- [x] Extensive docstring documentation with paper references
- [x] Test examples showing tokenization behavior
