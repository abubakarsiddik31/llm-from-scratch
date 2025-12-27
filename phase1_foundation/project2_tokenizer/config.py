# ruff: noqa
"""
Byte-Pair Encoding Tokenizer Configuration

This file defines hyperparameters for BPE training following:
- "Byte-Pair Encoding: Subword-Based Machine Translation" (Sennrich et al., 2016)
- "Language Models are Unsupervised Multitask Learners" (GPT-2, 2019)

CONFIGURATION REFERENCE:
------------------------
From GPT-2 paper:
"We use a byte-level BPE (byte-pair encoding) vocabulary with a size
of 50,257... 50,000 merges plus 256 byte tokens and a special end-of-text token"

BPE HYPERPARAMETERS:
-------------------
VOCAB_SIZE: Total number of tokens in vocabulary
- GPT-2 small: 50,257 tokens
- GPT-3: 50,257 tokens
- This project: 50,000 tokens (similar scale)

MIN_FREQUENCY: Minimum frequency for a merge to be considered
- Prevents merging rare character combinations
- Typically 2-10
- Lower = more merges, higher = fewer merges

SPECIAL_TOKENS: Special tokens for specific purposes
- <PAD>: Padding token (for batching)
- <UNK>: Unknown token (fallback for unknown characters)
- <BOS>: Beginning of sequence
- <EOS>: End of sequence
"""

import os

# =============================================================================
# BPE HYPERPARAMETERS
# =============================================================================

"""
From GPT-2: "We use a Byte-Pair Encoding (BPE) tokenizer...
with a vocabulary size of 50,257"

VOCAB_SIZE:
- Number of unique tokens in vocabulary
- Includes: character tokens + merged subwords + special tokens
- GPT-2 uses 50,257: 50,000 merges + 256 bytes + 1 special token
- We use 50,000 as a clean number (still GPT-2 scale)
"""
VOCAB_SIZE = 50_000

"""
MIN_FREQUENCY:
- Minimum number of times a pair must appear to be merged
- Prevents overfitting to rare character combinations
- From BPE paper: "Stop after N merges" (we use frequency threshold)

Values:
- 2: Merge any pair that appears at least twice (aggressive)
- 5: More conservative, fewer merges
- 10: Very conservative, mostly common patterns

Why 2?
- Allows learning of moderately rare but useful patterns
- Prevents noise from single-occurrence typos
"""
MIN_FREQUENCY = 2

"""
SPECIAL_TOKENS:
Special tokens with reserved IDs for specific purposes.

From "Sequence to Sequence Learning with Neural Networks" (Sutskever et al., 2014):
Special tokens help model understand sequence boundaries.

PADDING (<PAD>):
- Used to make all sequences in a batch the same length
- ID 0: Convention (PyTorch CrossEntropyLoss ignores index 0 by default)
- Example: [hello, world, <PAD>, <PAD>]

UNKNOWN (<UNK>):
- Fallback for characters not in vocabulary
- Should rarely be used if vocabulary is comprehensive
- ID 1: Standard convention

BEGINNING OF SEQUENCE (<BOS>):
- Marks start of generated text
- Useful for conditioning generation
- ID 2: Standard convention

END OF SEQUENCE (<EOS>):
- Marks end of text / generation stopping point
- Model learns to predict this when done
- ID 3: Standard convention (GPT-2 uses <|endoftext|>)
"""
SPECIAL_TOKENS = {
    "<PAD>": 0,
    "<UNK>": 1,
    "<BOS>": 2,
    "<EOS>": 3,
}

# =============================================================================
# DATA CONFIGURATION
# =============================================================================

"""
TRAIN_DATA_PATH:
Path to training text for learning BPE merges.

Requirements:
- Large enough to learn meaningful patterns (>1MB recommended)
- Diverse: Mix of text types for general vocabulary
- Clean: Remove excessive formatting/markup

Options:
1. Wikipedia dump (diverse, encyclopedic) - DEFAULT
2. Project Gutenberg (literature, formal English)
3. WebText sample (web text, more casual)
4. Shakespeare (small, but good for testing)

For this project, we use Wikipedia WikiText dataset by default.
Download with: python download_data.py --size small
"""
TRAIN_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data",
    "wikipedia_train.txt",
)

"""
CHECKPOINT_DIR:
Directory to save trained tokenizer.

Format: .pkl file containing:
- vocab: Dictionary mapping token string to ID
- merges: List of merge operations (in order)
- special_tokens: Special token mappings
"""
CHECKPOINT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "checkpoints",
    "project2",
)

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

"""
Special tokens are reserved and NOT part of the learned vocabulary.
They are assigned IDs first (0-3), then BPE learns remaining tokens.

Example:
- Special tokens: 4 tokens with IDs 0, 1, 2, 3
- BPE learns: VOCAB_SIZE - 4 = 49,996 additional tokens
- Total: 50,000 tokens
"""
NUM_SPECIAL_TOKENS = len(SPECIAL_TOKENS)
NUM_MERGES = VOCAB_SIZE - NUM_SPECIAL_TOKENS - 256  # Reserve 256 for base bytes


# =============================================================================
# VALIDATION
# =============================================================================


def validate_config():
    """
    Validate configuration parameters.

    RAISES:
        ValueError: If configuration is invalid

    VALIDATIONS:
    1. VOCAB_SIZE must be positive
    2. NUM_MERGES must be positive
    3. Special token IDs must be sequential from 0
    """
    if VOCAB_SIZE <= 0:
        raise ValueError(f"VOCAB_SIZE must be positive, got {VOCAB_SIZE}")

    if NUM_MERGES <= 0:
        raise ValueError(
            f"NUM_MERGES must be positive (vocab too small for special tokens), "
            f"got {NUM_MERGES}"
        )

    # Check special tokens are sequential from 0
    special_ids = sorted(SPECIAL_TOKENS.values())
    expected = list(range(len(SPECIAL_TOKENS)))
    if special_ids != expected:
        raise ValueError(
            f"Special token IDs must be sequential from 0, "
            f"expected {expected}, got {special_ids}"
        )


def print_config():
    """
    Print configuration in a readable format.

    Displays:
    - Vocabulary size
    - Number of merges
    - Special tokens
    - Data paths
    """
    print("=" * 60)
    print("BPE Tokenizer Configuration")
    print("=" * 60)
    print(f"VOCAB_SIZE:        {VOCAB_SIZE:,}")
    print(f"NUM_MERGES:        {NUM_MERGES:,}")
    print(f"MIN_FREQUENCY:     {MIN_FREQUENCY}")
    print(f"NUM_SPECIAL_TOKENS:{NUM_SPECIAL_TOKENS}")
    print()
    print("SPECIAL TOKENS:")
    for token, idx in SPECIAL_TOKENS.items():
        print(f"  {token:10s} -> ID {idx}")
    print()
    print("DATA PATHS:")
    print(f"  Train data:      {TRAIN_DATA_PATH}")
    print(f"  Checkpoint dir:  {CHECKPOINT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    validate_config()
    print_config()
