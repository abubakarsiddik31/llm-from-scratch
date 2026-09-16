# ruff: noqa
"""
Configuration for GPT Pre-Training (~125M parameters)

This file defines every hyperparameter for Project 4, where we scale the
chapter 1 model up to GPT-2-small class and pre-train it on WikiText-103.

PAPER REFERENCES:
-----------------
- "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019)
  GPT-2: 12-layer / 768-hidden / 12-head model, 50,257 BPE vocab,
  context 1024, Adam with lr 2.5e-4.
- "Language Models are Few-Shot Learners" (Brown et al., 2020)
  GPT-3: cosine LR decay to 10% of peak, warmup, weight decay 0.1,
  gradient clipping 1.0, batch measured in TOKENS (not sequences).
- "Training Compute-Optimal Large Language Models" (Hoffmann et al., 2022)
  Chinchilla: token budget should scale with parameter count (~20 tokens
  per parameter for compute-optimal training). We fall far short of that
  on one RTX 3050 and say so in the results.
- "FlashAttention" (Dao et al., 2022) / PyTorch SDPA: fused attention kernels.

THE ONE DELIBERATE DEVIATION FROM GPT-2:
----------------------------------------
GPT-2 small reaches 124M parameters with a 50,257-token vocab. Our from-
scratch BPE (chapter 2 algorithm, retrained on WikiText-103) targets a
16,384-token vocab, which makes the embedding table ~26M parameters smaller.
To land back in the 125M class we use 16 layers instead of 12:

    GPT-2 small:   12 layers x 768 x 50,257 vocab  ~= 124M params
    Ours:          16 layers x 768 x 16,384 vocab  ~= 126M params

Everything else (width, heads, FFN expansion, pre-LN) matches GPT-2 small.
"""

import os

import torch

# =============================================================================
# PATHS (built from __file__, always run scripts from the repo root)
# =============================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "wikitext")  # extracted corpus text
TOKENIZED_DATA_DIR = os.path.join(DATA_DIR, "project4")  # encoded .bin files
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "project4")
TOKENIZER_FILE = os.path.join(CHECKPOINT_DIR, "tokenizer.pkl")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# =============================================================================
# SPECIAL TOKENS (same IDs as chapters 2-4, they depend on these positions)
# =============================================================================

SPECIAL_TOKENS = {
    "<PAD>": 0,
    "<UNK>": 1,
    "<BOS>": 2,
    "<EOS>": 3,
}

# =============================================================================
# TOKENIZER CONFIGURATION
# =============================================================================

# Total vocabulary size for our from-scratch BPE tokenizer.
# GPT-2 uses 50,257; we train our own 16,384-token BPE on WikiText-103.
# 16,384 < 65,536 so tokens fit in uint16 arrays (see prepare_data.py).
VOCAB_SIZE = 16_384

# Re-trained on WikiText-103 rather than reusing the chapter 2 tokenizer:
# that one saw 11 MB of WikiText-2 with a 5,000-token vocab, which would
# fragment WikiText-103 into longer sequences and slow training.
TOKENIZER_SAMPLE_MB = 100  # how much of wiki.train.tokens to train BPE on
MIN_FREQUENCY = 2  # minimum pair frequency for a merge (same as chapter 2)

# =============================================================================
# DATA CONFIGURATION
# =============================================================================

# Sequences per optimizer step is BATCH_SIZE x GRAD_ACCUM_STEPS.
# GPT-2 trains on 512 seqs x 1024 tokens ~= 0.5M tokens per batch, which
# needs datacenter GPUs. We hold ~32k tokens per step and compensate with a
# longer schedule (GPT-3, Appendix B, scales batch/LR together for this).
BATCH_SIZE = 8  # sequences per forward pass (fits 8 GB in bf16)
GRAD_ACCUM_STEPS = 8  # forward/backward passes per optimizer step
BLOCK_SIZE = 512  # context length. GPT-2 uses 1024; halving it roughly
# doubles throughput on the 3050 and fits the same batch footprint.

# =============================================================================
# MODEL ARCHITECTURE (GPT-2 small shape, 16 layers instead of 12 - see header)
# =============================================================================

N_EMBD = 768
N_HEAD = 12
N_LAYER = 16
DROPOUT = 0.0
# GPT-2 used dropout 0.1; nanoGPT and most modern pre-training runs use 0.0
# because a large corpus is seen ~once (regularization comes from data
# volume, not dropout). Raise it if you fine-tune on small data later.

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

MAX_ITERS = 4_000
# 4,000 steps x 32,768 tokens/step ~= 131M tokens ~= one pass over
# WikiText-103 (~117M BPE tokens). Chinchilla-optimal for 126M params would
# be ~2.5B tokens; that is weeks on this GPU, not hours.

LEARNING_RATE = 2.5e-4  # GPT-2's peak LR
MIN_LEARNING_RATE = LEARNING_RATE * 0.1  # GPT-3: cosine floor at 10%
WARMUP_ITERS = 200  # linear LR warmup (GPT-2/GPT-3 practice)
WEIGHT_DECAY = 0.1  # GPT-3 Appendix B (applied to matrices only, see train.py)
BETA1, BETA2 = 0.9, 0.95  # GPT-2 used (0.9, 0.999); (0.9, 0.95) is the
# standard large-model setting (GPT-3, nanoGPT): less second-moment memory,
# more responsive to recent gradients.
GRAD_CLIP = 1.0  # clip global grad norm (GPT-2 used 1.0)

EVAL_INTERVAL = 100  # steps between validation estimates
EVAL_ITERS = 30  # batches averaged per loss estimate
LOG_INTERVAL = 10

# Mixed precision: bf16 on Ampere (RTX 3050, sm_86) needs no loss scaling.
# Falls back to fp32 automatically when unsupported.
USE_BF16 = True

# torch.compile can fuse the whole model but is not supported on Windows
# CUDA builds; leave off unless you know your build supports it.
USE_COMPILE = False

# =============================================================================
# INFERENCE CONFIGURATION
# =============================================================================

MAX_NEW_TOKENS = 500
TEMPERATURE = 0.8
TOP_K = 50

# =============================================================================
# SYSTEM CONFIGURATION
# =============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# VALIDATION / PRINTING
# =============================================================================


def validate_config():
    """Validate that hyperparameters are consistent."""
    assert N_EMBD % N_HEAD == 0, f"n_embd ({N_EMBD}) must be divisible by n_head ({N_HEAD})"
    assert BLOCK_SIZE > 0 and BATCH_SIZE > 0 and GRAD_ACCUM_STEPS > 0
    assert 0 < LEARNING_RATE < 1, "learning_rate must be between 0 and 1"
    assert VOCAB_SIZE - len(SPECIAL_TOKENS) > 0, "vocab too small for special tokens"
    assert VOCAB_SIZE <= 65_536, "vocab must fit in uint16 for the .bin data files"
    assert WARMUP_ITERS < MAX_ITERS, "warmup must be shorter than the schedule"
    print("✓ Configuration validated")


def print_config():
    """Print all configuration values."""
    tokens_per_step = BATCH_SIZE * GRAD_ACCUM_STEPS * BLOCK_SIZE
    print("=" * 60)
    print("PRE-TRAINING CONFIGURATION (~125M GPT)")
    print("=" * 60)
    print(f"Vocabulary size:     {VOCAB_SIZE:,}")
    print(f"Block size:          {BLOCK_SIZE}")
    print(f"Layers / heads:      {N_LAYER} / {N_HEAD}")
    print(f"Embedding dim:       {N_EMBD}")
    print(f"Dropout:             {DROPOUT}")
    print("-" * 60)
    print("TRAINING CONFIGURATION")
    print("-" * 60)
    print(f"Batch:               {BATCH_SIZE} seqs x {GRAD_ACCUM_STEPS} accum")
    print(f"  = {BATCH_SIZE * GRAD_ACCUM_STEPS} seqs x {BLOCK_SIZE} tokens")
    print(f"  = {tokens_per_step:,} tokens per optimizer step")
    print(f"Max iterations:      {MAX_ITERS:,}")
    print(f"  = {MAX_ITERS * tokens_per_step / 1e6:,.0f}M tokens total")
    print(f"Peak LR:             {LEARNING_RATE} (cosine to {MIN_LEARNING_RATE})")
    print(f"Weight decay:        {WEIGHT_DECAY}")
    print(f"Device:              {DEVICE}")
    print("=" * 60)


if __name__ == "__main__":
    validate_config()
    print_config()
