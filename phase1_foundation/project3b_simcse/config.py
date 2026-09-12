# ruff: noqa
"""
Configuration for SimCSE (Sentence Embeddings)

This file implements hyperparameters for unsupervised SimCSE following:
- "SimCSE: Simple Contrastive Learning of Sentence Embeddings"
  (Gao, Yao & Chen, EMNLP 2021)

THE BIG IDEA (from the paper):
------------------------------
"we take a collection of sentences {x_i} and use x_i+ = x_i. The key
ingredient to get this to work with identical positive pairs is through
the use of independently sampled dropout masks."

Feed the SAME sentence through the encoder twice with different dropout
masks -> the two embeddings form a positive pair. Other sentences in the
batch are negatives. Standard contrastive loss pulls positives together
and pushes negatives apart.

PAPER HYPERPARAMETERS (BERT-base reference, scaled down for our model):
----------------------------------------------------------------------
- Dropout:           p = 0.1 (the encoder's existing dropout - nothing added)
- Batch size:        64
- Learning rate:     3e-5 (BERT-base)
- Temperature:       tau = 0.05 (unsupervised; 0.001 for supervised SimCSE)
- Epochs:            1 over 10^6 Wikipedia sentences
- Max seq length:    32 tokens
- Projection head:   2-layer MLP with tanh, used ONLY during training
- Evaluation:        Spearman correlation on STS-B with cosine similarity

OUR SCALING: we fine-tune the small BERT from Project 3 (6 layers,
256 hidden) on a subset of Wikipedia sentences, so we use a proportionally
larger learning rate and fewer sentences. Principles stay identical.
"""

import os
import torch

# =============================================================================
# SYSTEM CONFIGURATION
# =============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# =============================================================================
# ENCODER (inherits from Project 3)
# =============================================================================

# Path to the Project 3 MLM pre-trained checkpoint used to initialize
# the encoder. The paper stresses that "starting from a pre-trained
# checkpoint is crucial, for it provides good initial alignment."
P3_CHECKPOINT_PATH = os.path.join(ROOT_DIR, "checkpoints", "project3", "bert_pretrained.pt")

# The Project 3 model uses these (must match its config.py)
ENCODER_CONFIG = {
    "N_EMBD": 256,
    "N_HEAD": 8,
    "N_LAYER": 6,
    "BLOCK_SIZE": 128,
    "DROPOUT": 0.1,
}

# =============================================================================
# DATA CONFIGURATION
# =============================================================================

# Sentence corpus for unsupervised SimCSE.
# The paper samples 10^6 sentences from English Wikipedia; we extract
# sentences from WikiText-2 (already downloaded for Project 3) and cap
# the count for a tractable demo run.
SENTENCE_DATA_PATH = os.path.join(ROOT_DIR, "data", "wikitext_train.txt")

# Max number of training sentences (paper: 1,000,000)
MAX_SENTENCES = 100_000

# Sentence length limits (in characters) when extracting from corpus
MIN_SENTENCE_CHARS = 20
MAX_SENTENCE_CHARS = 200

# Paper: max sequence length of 32 tokens works best
MAX_SEQ_LEN = 32

# Validation split for monitoring contrastive loss
VAL_RATIO = 0.01

# STS-B evaluation data (glue/stsb) is cached by the HF datasets library;
# no local file needed. Set to True to download on first evaluation.
USE_STS_B = True

# =============================================================================
# CONTRASTIVE LEARNING CONFIGURATION
# =============================================================================

# Temperature for the contrastive loss (paper section 5: unsupervised
# SimCSE works best with a fixed tau = 0.05; supervised uses 0.001)
TEMPERATURE = 0.05

# Batch size (paper: 64)
BATCH_SIZE = 64

# Whether to use the MLP projection head during training.
# From the paper: "we use an MLP layer (with one tanh activation) on top
# of the [CLS] representation ... we use it only for training but not
# for evaluation" - the "MLP trick" that improves results by ~1-2 points.
USE_MLP_HEAD = True

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

# Number of training epochs over the sentence corpus.
# Paper: 1 epoch for BERT-base on 10^6 sentences. Our smaller encoder
# overfits the 10^5-sentence corpus faster; a quick sweep found 3
# epochs best on STS-B dev (1 ep: 27.0, 3 ep: 34.7, 5 ep: 33.5).
EPOCHS = 3

# Learning rate (paper: 3e-5 for BERT-base; our encoder is ~50x smaller,
# so we can afford a slightly larger rate)
LEARNING_RATE = 5e-5

# Weight decay (paper: 0.01)
WEIGHT_DECAY = 0.01

# Warmup proportion of total steps
WARMUP_RATIO = 0.1

# How often to log training loss (in steps)
LOG_INTERVAL = 50

# How often to run STS-B evaluation during training (in steps)
EVAL_INTERVAL = 300

# =============================================================================
# TOKENIZER
# =============================================================================

# BPE tokenizer trained in Project 2 (same vocab as the encoder)
TOKENIZER_PATH = os.path.join(ROOT_DIR, "checkpoints", "project2", "tokenizer.pkl")

# Special tokens (same as Project 3)
SPECIAL_TOKENS = {
    "[PAD]": 0,
    "[UNK]": 1,
    "[CLS]": 2,
    "[SEP]": 3,
    "[MASK]": 4,
}

PAD_TOKEN_ID = SPECIAL_TOKENS["[PAD]"]
CLS_TOKEN_ID = SPECIAL_TOKENS["[CLS]"]
SEP_TOKEN_ID = SPECIAL_TOKENS["[SEP]"]

# =============================================================================
# OUTPUT
# =============================================================================

CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints", "project3b")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def validate_config():
    """Validate that hyperparameters are consistent."""
    assert ENCODER_CONFIG["N_EMBD"] % ENCODER_CONFIG["N_HEAD"] == 0, (
        f"N_EMBD ({ENCODER_CONFIG['N_EMBD']}) must be divisible by "
        f"N_HEAD ({ENCODER_CONFIG['N_HEAD']})"
    )
    assert MAX_SEQ_LEN <= ENCODER_CONFIG["BLOCK_SIZE"], (
        f"MAX_SEQ_LEN ({MAX_SEQ_LEN}) must not exceed encoder BLOCK_SIZE "
        f"({ENCODER_CONFIG['BLOCK_SIZE']})"
    )
    assert 0 < TEMPERATURE < 1, "temperature must be in (0, 1)"
    assert BATCH_SIZE > 1, "contrastive loss needs batch_size > 1 for negatives"
    assert 0 < WARMUP_RATIO < 1, "warmup_ratio must be in (0, 1)"
    print("✓ Configuration validated")


def print_config():
    """Print all configuration values."""
    print("=" * 60)
    print("SIMCSE (UNSUPERVISED) CONFIGURATION")
    print("=" * 60)
    print("-" * 60)
    print("ENCODER (from Project 3)")
    print("-" * 60)
    print(f"  Embedding dim:     {ENCODER_CONFIG['N_EMBD']}")
    print(f"  Attention heads:   {ENCODER_CONFIG['N_HEAD']}")
    print(f"  Transformer layers:{ENCODER_CONFIG['N_LAYER']}")
    print(f"  Pre-trained ckpt:  {P3_CHECKPOINT_PATH}")
    print("-" * 60)
    print("CONTRASTIVE LEARNING")
    print("-" * 60)
    print(f"  Temperature (tau): {TEMPERATURE}")
    print(f"  Batch size:        {BATCH_SIZE}")
    print(f"  MLP head (train):  {USE_MLP_HEAD}")
    print("-" * 60)
    print("DATA")
    print("-" * 60)
    print(f"  Sentence corpus:   {SENTENCE_DATA_PATH}")
    print(f"  Max sentences:     {MAX_SENTENCES:,}")
    print(f"  Max seq len:       {MAX_SEQ_LEN}")
    print("-" * 60)
    print("TRAINING")
    print("-" * 60)
    print(f"  Epochs:            {EPOCHS}")
    print(f"  Learning rate:     {LEARNING_RATE}")
    print(f"  Weight decay:      {WEIGHT_DECAY}")
    print(f"  Warmup ratio:      {WARMUP_RATIO}")
    print(f"  Device:            {DEVICE}")
    print("=" * 60)


if __name__ == "__main__":
    validate_config()
    print_config()
