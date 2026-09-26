# ruff: noqa
"""
Configuration for LoRA Fine-Tuning, Project 6

Project 5 fine-tuned all 126,273,024 parameters to teach instruction
following. This project reaches the same goal from the same base
checkpoint on the same Alpaca data while training a fraction of a
percent of the parameters, using LoRA (Hu et al., 2021): freeze the
weights W, learn a low-rank update dW = B x A instead.

PAPER REFERENCES:
-----------------
- "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
  The source of everything here: freeze W, learn B @ A with r << min(d, k),
  scale by alpha/r, apply to attention projections, initialize B = 0 so
  dW starts at zero, and merge B @ A back into W at deploy time so the
  merged model has the original architecture and zero added latency.
  The paper's GPT-3 experiments adapt W_q and W_v; our fused QKV
  projection c_attn covers all three of Q, K, V in one matrix, so
  targeting c_attn adapts strictly more than the paper's default at
  similar parameter cost.
- "Training Language Models to Follow Instructions with Human Feedback"
  (Ouyang et al., 2022): the task (stage-one alignment SFT) that project 5
  performed with full fine-tuning.
- "Stanford Alpaca" (Taori et al., 2023): the dataset, template, and the
  3-epoch schedule this run mirrors for comparability with project 5.

THE EXPERIMENT THIS PROJECT EXISTS FOR:
---------------------------------------
Full SFT (project 5) vs LoRA (this project), same base checkpoint, same
encoded Alpaca arrays (data/project5, reused verbatim), same schedule
shape. The chapters compare loss curves, samples, and cost. The base
checkpoint defaults to project 4's (LoRA as an ALTERNATIVE to full SFT);
pass --base_checkpoint pointing at the project 5 checkpoint to use LoRA
for continued adaptation instead.

LORA HYPERPARAMETERS:
---------------------
    LORA_R = 8      the rank of the update. The paper uses r = 1..64 on
                    GPT-3 with little sensitivity; 8 sits mid-range.
    LORA_ALPHA = 16 the scale, divided by r: effective dW scale is
                    alpha/r = 2. Keeping alpha = 2r is the common choice.
    LORA_TARGETS = ["c_attn"]   attention QKV projections only.

    Trainable parameters at these values: per layer, A is 8 x 768 and
    B is 2304 x 8 -> 24,576; across 16 layers = 393,216 = 0.311% of the
    model. test_model.py asserts the count so the number stays honest.

NO LORA DROPOUT: the paper reports dropout on the LoRA path (0.1) for
some settings. This repository trains short schedules with dropout 0.0
everywhere (see project 5's config note); adding a second dropout knob
for one path was not worth the config surface.

WHAT IS REUSED, WHAT IS NEW:
----------------------------
- model.py: same GPT as projects 4/5 (code identical). LoRA wraps its
  Linear submodules at RUNTIME (lora.py); the class code never changes.
- tokenizer.py / template.py: same load-only tokenizer and Alpaca
  template as project 5 (verbatim copies, self-contained-project
  convention).
- data: reuses data/project5/*.npy (encoded in project 5). No download
  or encoding scripts here on purpose - same data is the point of the
  comparison. prepare it once with project 5's pipeline.
- lora.py: the LoRA layer, the applier, the merge.
"""

import os

import torch

# =============================================================================
# PATHS (built from __file__, always run scripts from the repo root)
# =============================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Encoded SFT arrays produced by project 5 (shared on purpose: same data,
# same split, so project 5 vs project 6 is a controlled comparison).
TOKENIZED_DATA_DIR = os.path.join(DATA_DIR, "project5")

# Artifacts from earlier phases.
PROJECT4_CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "project4")
PROJECT5_CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "project5")
BASE_CHECKPOINT = os.path.join(PROJECT4_CHECKPOINT_DIR, "model_final.pt")
TOKENIZER_FILE = os.path.join(PROJECT4_CHECKPOINT_DIR, "tokenizer.pkl")

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "project6")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# =============================================================================
# SPECIAL TOKENS (same IDs as chapters 2-4)
# =============================================================================

SPECIAL_TOKENS = {
    "<PAD>": 0,
    "<UNK>": 1,
    "<BOS>": 2,
    "<EOS>": 3,
}

# =============================================================================
# TOKENIZER (loaded, never retrained)
# =============================================================================

VOCAB_SIZE = 16_384  # must match the base checkpoint's embedding table

# =============================================================================
# LORA CONFIGURATION
# =============================================================================

LORA_R = 8  # update rank (Hu et al., 2021; mid-range of the paper's sweep)
LORA_ALPHA = 16  # scale; effective dW multiplier = LORA_ALPHA / LORA_R = 2
LORA_TARGETS = ["c_attn"]  # module-name suffixes to wrap (fused QKV here)

# =============================================================================
# DATA CONFIGURATION (same shapes as project 5, for comparability)
# =============================================================================

BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 8  # 64 examples per optimizer step, as in project 5
BLOCK_SIZE = 512
MAX_SEQ_LEN = 512  # prepared example length; training feeds T = 511

VAL_EXAMPLES = 1_000  # informational; the arrays are already split
SEED = 42

# =============================================================================
# MODEL ARCHITECTURE (must match the base checkpoint; validated on load)
# =============================================================================

N_EMBD = 768
N_HEAD = 12
N_LAYER = 16
DROPOUT = 0.0

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

MAX_ITERS = 2_400
# Same schedule shape as project 5 (64 examples/step ~= 3 epochs over the
# 50,973-example train split) so the loss curves are comparable.

LEARNING_RATE = 1e-4
# The LoRA paper's GPT-3 runs use lr 1e-4..3e-4 for LoRA vs 1e-5..5e-5 for
# full fine-tuning: with almost all weights frozen, updates must move
# further to matter, and the frozen base bounds how much damage a big step
# can do. 1e-4 is the paper's lower/mid setting, right for a 126M model.
MIN_LEARNING_RATE = LEARNING_RATE * 0.1
WARMUP_ITERS = 100
WEIGHT_DECAY = 0.0
# Weight decay on A/B would pull the update toward zero for no reason
# here (the paper tunes wd per task; our full-SFT runs used 0.1 on the
# whole model). LoRA's B starts at zero and the schedule is short.
BETA1, BETA2 = 0.9, 0.95
GRAD_CLIP = 1.0

EVAL_INTERVAL = 100
EVAL_ITERS = 20
LOG_INTERVAL = 10

USE_BF16 = True
USE_COMPILE = False

# =============================================================================
# INFERENCE CONFIGURATION (same as project 5)
# =============================================================================

MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
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
    assert MAX_SEQ_LEN <= BLOCK_SIZE, (
        f"MAX_SEQ_LEN ({MAX_SEQ_LEN}) must fit BLOCK_SIZE ({BLOCK_SIZE})"
    )
    assert 0 < LEARNING_RATE < 1, "learning_rate must be between 0 and 1"
    assert VOCAB_SIZE - len(SPECIAL_TOKENS) > 0, "vocab too small for special tokens"
    assert LORA_R > 0 and LORA_R < N_EMBD, "LoRA rank must be in (0, n_embd)"
    assert LORA_ALPHA > 0, "LoRA alpha must be positive"
    assert len(LORA_TARGETS) > 0, "need at least one LoRA target module"
    assert WARMUP_ITERS < MAX_ITERS, "warmup must be shorter than the schedule"
    print("✓ Configuration validated")


def print_config():
    """Print all configuration values."""
    examples_per_step = BATCH_SIZE * GRAD_ACCUM_STEPS
    print("=" * 60)
    print("LORA FINE-TUNING CONFIGURATION (~125M GPT on Alpaca)")
    print("=" * 60)
    print(f"Base checkpoint:     {os.path.relpath(BASE_CHECKPOINT, PROJECT_ROOT)}")
    print(f"Data (reused):       {os.path.relpath(TOKENIZED_DATA_DIR, PROJECT_ROOT)} "
          f"(encoded by project 5)")
    print(f"Vocabulary size:     {VOCAB_SIZE:,}")
    print(f"LoRA:                r={LORA_R}, alpha={LORA_ALPHA}, "
          f"targets={LORA_TARGETS}")
    print(f"Layers / heads:      {N_LAYER} / {N_HEAD}")
    print(f"Embedding dim:       {N_EMBD}")
    print("-" * 60)
    print("TRAINING CONFIGURATION")
    print("-" * 60)
    print(f"Batch:               {BATCH_SIZE} seqs x {GRAD_ACCUM_STEPS} accum")
    print(f"  = {examples_per_step} examples per optimizer step")
    print(f"Max iterations:      {MAX_ITERS:,}")
    print(f"  = {MAX_ITERS * examples_per_step:,} example visits")
    print(f"Peak LR:             {LEARNING_RATE} (cosine to {MIN_LEARNING_RATE})")
    print(f"Weight decay:        {WEIGHT_DECAY}")
    print(f"Device:              {DEVICE}")
    print("=" * 60)


if __name__ == "__main__":
    validate_config()
    print_config()
