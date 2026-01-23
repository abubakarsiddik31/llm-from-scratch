# ruff: noqa
"""
Configuration for BERT-Style Contextual Embeddings

This file implements hyperparameters for BERT-style bidirectional encoding
following: "BERT: Pre-training of Deep Bidirectional Transformers for Language
Understanding" (Devlin et al., 2018)

BERT ARCHITECTURE DIFFERENCES FROM GPT:
----------------------------------------
From BERT paper (2018):
"We introduce a new language representation model called BERT, which stands
for Bidirectional Encoder Representations from Transformers."

KEY DIFFERENCES:
1. Bidirectional attention: Can see all tokens (no causal mask)
2. Masked Language Modeling (MLM): Predict masked tokens
3. Next Sentence Prediction (NSP): Binary classification task
4. Segment embeddings: Distinguish sentence pairs

BIGGER IS BETTER:
-----------------
BERT-base: 12 layers, 768 hidden, 12 heads, 110M params
BERT-large: 24 layers, 1024 hidden, 16 heads, 340M params

We use a smaller model for demonstration.
"""

import os
import torch

# =============================================================================
# DATA CONFIGURATION
# =============================================================================

# Batch size: How many sequences we process in parallel
# - BERT paper uses 256 for base model, 128 for large
# - We use smaller for memory efficiency
BATCH_SIZE = 32

# Block size (Context window): Maximum sequence length
# - BERT uses 512 tokens as max sequence length
# - This is a key architectural choice from the paper
# - Longer = more context, more memory, slower training
BLOCK_SIZE = 128

# =============================================================================
# MODEL ARCHITECTURE (BERT-style)
# =============================================================================

# Embedding dimension: Size of vector representing each token
# - BERT-base: 768
# - BERT-large: 1024
# - We use 256 for smaller model (still bigger than GPT project)
N_EMBD = 256

# Number of attention heads
# - BERT-base: 12 heads
# - BERT-large: 16 heads
# - Must divide n_embd evenly
N_HEAD = 8

# Number of transformer blocks (layers)
# - BERT-base: 12 layers
# - BERT-large: 24 layers
# - We use 6 layers for demonstration
N_LAYER = 6

# Intermediate size for feed-forward network
# - BERT uses 4× expansion (same as GPT)
# - BERT-base: 768 → 3072
FFN_HIDDEN_SIZE = 4 * N_EMBD

# Dropout probability
# - BERT uses 0.1 for all dropout layers
DROPOUT = 0.1

# =============================================================================
# MASKED LANGUAGE MODELING (MLM) CONFIGURATION
# =============================================================================

# Masking probability: What fraction of tokens to mask
# - BERT paper: 15% of tokens masked
# - We use same value as paper
MLM_PROB = 0.15

# Mask token distribution (how to handle masked tokens)
# From BERT paper:
# "80% of the time: replace with [MASK] token"
# "10% of the time: replace with random token"
# "10% of the time: keep original token"
MASK_PROB = 0.8  # 80% -> [MASK]
RANDOM_PROB = 0.1  # 10% -> random token
KEEP_PROB = 0.1  # 10% -> keep original

# =============================================================================
# NEXT SENTENCE PREDICTION (NSP) CONFIGURATION
# =============================================================================

# Whether to include NSP task
# - BERT uses this for pre-training
# - Recent work suggests NSP may not be necessary
# - We include it for educational purposes
USE_NSP = True

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

# Total training iterations
# - BERT-base: 1M steps on 3.3B words
# - We use much smaller for demonstration
MAX_ITERS = 10000

# How often to evaluate on validation set
EVAL_INTERVAL = 100

# Number of batches to evaluate
EVAL_ITERS = 50

# Learning rate
# - BERT paper: 1e-4 for base, 5e-5 for large (Adam with warmup)
# - We use 2e-4 for smaller model
LEARNING_RATE = 2e-4

# Weight decay for regularization
# - BERT uses 0.01
WEIGHT_DECAY = 0.01

# =============================================================================
# SPECIAL TOKENS (for MLM)
# =============================================================================

# BERT uses special tokens for masked language modeling
SPECIAL_TOKENS = {
    "[PAD]": 0,  # Padding token
    "[UNK]": 1,  # Unknown token
    "[CLS]": 2,  # Classification token (sentence start)
    "[SEP]": 3,  # Separator token (sentence boundary)
    "[MASK]": 4,  # Mask token for MLM
}

NUM_SPECIAL_TOKENS = len(SPECIAL_TOKENS)

# =============================================================================
# SYSTEM CONFIGURATION
# =============================================================================

# Device: Where to run computations
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# DATA PATHS
# =============================================================================

# Root directory (for path calculations)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# Training data path
TRAIN_DATA_PATH = os.path.join(ROOT_DIR, "data", "wikitext_train.txt")

# Validation data path
VAL_DATA_PATH = os.path.join(ROOT_DIR, "data", "wikitext_val.txt")

# Checkpoint directory
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints", "project3")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# =============================================================================
# DERIVED CONFIGURATIONS (Don't modify manually)
# =============================================================================

VOCAB_SIZE = None  # Will be set based on tokenizer


def validate_config():
    """Validate that hyperparameters are consistent."""
    assert N_EMBD % N_HEAD == 0, f"n_embd ({N_EMBD}) must be divisible by n_head ({N_HEAD})"
    assert BLOCK_SIZE > 0, "block_size must be positive"
    assert BATCH_SIZE > 0, "batch_size must be positive"
    assert 0 < LEARNING_RATE < 1, "learning_rate must be between 0 and 1"
    assert 0 < MLM_PROB < 1, "mlm_prob must be between 0 and 1"
    assert abs(MASK_PROB + RANDOM_PROB + KEEP_PROB - 1.0) < 1e-6, \
        "Mask probabilities must sum to 1.0"
    print("✓ Configuration validated")


def print_config():
    """Print all configuration values."""
    print("=" * 60)
    print("BERT-STYLE MODEL CONFIGURATION")
    print("=" * 60)
    print(f"Vocabulary size:     {VOCAB_SIZE if VOCAB_SIZE else 'To be determined'}")
    print(f"Block size:          {BLOCK_SIZE}")
    print(f"Batch size:          {BATCH_SIZE}")
    print("-" * 60)
    print("MODEL ARCHITECTURE")
    print("-" * 60)
    print(f"Embedding dimension: {N_EMBD}")
    print(f"Attention heads:     {N_HEAD}")
    print(f"Transformer layers:  {N_LAYER}")
    print(f"FFN hidden size:     {FFN_HIDDEN_SIZE}")
    print(f"Dropout:             {DROPOUT}")
    print("-" * 60)
    print("MLM CONFIGURATION")
    print("-" * 60)
    print(f"Mask probability:    {MLM_PROB}")
    print(f"Mask → [MASK]:       {MASK_PROB}")
    print(f"Mask → random:       {RANDOM_PROB}")
    print(f"Mask → keep:         {KEEP_PROB}")
    print(f"Use NSP:             {USE_NSP}")
    print("-" * 60)
    print("TRAINING CONFIGURATION")
    print("-" * 60)
    print(f"Max iterations:      {MAX_ITERS}")
    print(f"Eval interval:       {EVAL_INTERVAL}")
    print(f"Learning rate:       {LEARNING_RATE}")
    print(f"Weight decay:        {WEIGHT_DECAY}")
    print(f"Device:              {DEVICE}")
    print("=" * 60)


if __name__ == "__main__":
    validate_config()
    print_config()
