"""
Configuration for Character-Level GPT

Every hyperparameter here is chosen deliberately.
Let's understand what each one does and why.
"""

import torch

# =============================================================================
# DATA CONFIGURATION
# =============================================================================

# Batch size: How many sequences we process in parallel
# - Higher = more stable gradients, faster training (more parallelism)
# - Lower = less GPU memory, more frequent updates
# - 64 is a good balance for 4-8GB GPU
BATCH_SIZE = 64

# Block size (Context window): Maximum sequence length
# - Model sees at most this many characters when predicting next token
# - Larger = more context, better long-range dependencies
# - Smaller = less memory, faster training
# - 256 is reasonable for character-level (enough for a few sentences)
BLOCK_SIZE = 256

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

# Embedding dimension: Size of vector representing each token
# - Each character gets embedded into a vector of this size
# - Larger = more expressive power, slower computation
# - Should be divisible by n_head (for multi-head attention)
# - 384 is small but sufficient for character-level
N_EMBD = 384

# Number of attention heads
# - Each head learns different patterns
# - n_embd must be divisible by n_head
# - Each head will have head_size = n_embd / n_head = 384 / 6 = 64
N_HEAD = 6

# Number of transformer blocks (layers)
# - Deeper network = more capacity to learn complex patterns
# - Each block has: Multi-head attention + Feed-forward + 2 LayerNorms
# - 6 layers is on the small side (GPT-3 has 96 layers)
# - Sufficient for character-level text generation
N_LAYER = 6

# Dropout probability
# - Randomly drops units during training to prevent overfitting
# - 0.2 = 20% of activations are randomly zeroed
# - Helps model generalize better
DROPOUT = 0.2

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

# Total training iterations
# - Each iteration processes BATCH_SIZE sequences
# - 5000 iterations × 64 batch × 256 tokens = 81.9M tokens processed
# - Shakespeare is ~1M chars, so we see the dataset ~80 times
MAX_ITERS = 5000

# How often to evaluate on validation set
# - Every EVAL_INTERVAL steps, we pause training and compute validation loss
# - More frequent = better tracking, slower training
# - 500 is reasonable (10 evaluations during training)
EVAL_INTERVAL = 500

# Number of batches to evaluate
# - We compute loss on multiple batches and average for stability
# - Use 200 for quick evaluation during training
EVAL_ITERS = 200

# Learning rate: Step size for gradient descent
# - Too large: training diverges or becomes unstable
# - Too small: training takes forever
# - 3e-4 is a good default for AdamW (standard for transformers)
LEARNING_RATE = 3e-4

# =============================================================================
# INFERENCE CONFIGURATION
# =============================================================================

# Maximum tokens to generate
# - When sampling, how many new tokens to generate
# - 500 is enough to see a few paragraphs of text
MAX_NEW_TOKENS = 500

# =============================================================================
# SYSTEM CONFIGURATION
# =============================================================================

# Device: Where to run computations
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# DERIVED CONFIGURATIONS (Don't modify manually)
# =============================================================================

# These will be set after loading the data
VOCAB_SIZE = None  # Will be set based on unique characters in dataset


def validate_config():
    """Validate that hyperparameters are consistent."""
    assert N_EMBD % N_HEAD == 0, f"n_embd ({N_EMBD}) must be divisible by n_head ({N_HEAD})"
    assert BLOCK_SIZE > 0, "block_size must be positive"
    assert BATCH_SIZE > 0, "batch_size must be positive"
    assert 0 < LEARNING_RATE < 1, "learning_rate must be between 0 and 1"
    print("✓ Configuration validated")


def print_config():
    """Print all configuration values."""
    print("=" * 60)
    print("MODEL CONFIGURATION")
    print("=" * 60)
    print(f"Vocabulary size:     {VOCAB_SIZE if VOCAB_SIZE else 'To be determined'}")
    print(f"Block size:          {BLOCK_SIZE}")
    print(f"Batch size:          {BATCH_SIZE}")
    print(f"Embedding dimension: {N_EMBD}")
    print(f"Attention heads:     {N_HEAD}")
    print(f"Transformer layers:  {N_LAYER}")
    print(f"Dropout:             {DROPOUT}")
    print("-" * 60)
    print("TRAINING CONFIGURATION")
    print("-" * 60)
    print(f"Max iterations:      {MAX_ITERS}")
    print(f"Eval interval:       {EVAL_INTERVAL}")
    print(f"Learning rate:       {LEARNING_RATE}")
    print(f"Device:              {DEVICE}")
    print("=" * 60)


if __name__ == "__main__":
    validate_config()
    print_config()
