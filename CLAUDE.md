# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a hands-on LLM implementation roadmap with 10 phases and 30 projects, building from character-level GPT to production deployment. Each project implements core components from scratch with extensive research paper references.

**Philosophy:** Learn by doing. Each project builds on the previous one, culminating in a fully deployed LLM application.

## Repository Structure

```
implementation/
├── phase1_foundation/
│   ├── project1_minimal_gpt/       ✅ Complete
│   ├── project2_tokenizer/         ✅ Complete
│   └── project3_pretrain/          ⏳ Pending
├── phase2_finetuning/
├── phase3_core_inference/
├── phase4_advanced_inference/
├── phase5_quantization/
├── phase6_compression/
├── phase7_architecture/
├── phase8_parallelism/
├── phase9_compiler/
├── phase10_deployment/
├── data/
│   ├── shakespeare.txt
│   └── wikipedia_train.txt
└── checkpoints/
    ├── project1/
    └── project2/
```

## Common Commands

**IMPORTANT: All commands are run from the project root directory using `uv run python`:**

```bash
# Install dependencies (uses uv)
uv sync --group all

# Example: Run any project script from root
uv run python phase1_foundation/project1_minimal_gpt/train.py
uv run python phase1_foundation/project2_tokenizer/train_tokenizer.py
```

### Project 1 - Character GPT
```bash
# Train model
uv run python phase1_foundation/project1_minimal_gpt/train.py

# Generate text (single)
uv run python phase1_foundation/project1_minimal_gpt/generate.py --prompt "ROMEO:" --temperature 0.8 --top_k 50

# Generate text (interactive)
uv run python phase1_foundation/project1_minimal_gpt/generate.py --interactive
```

### Project 2 - BPE Tokenizer
```bash
# Download training data
uv run python phase1_foundation/project2_tokenizer/download_data.py --size small

# Train tokenizer
uv run python phase1_foundation/project2_tokenizer/train_tokenizer.py --vocab_size 5000

# Test tokenizer
uv run python phase1_foundation/project2_tokenizer/test_tokenizer.py --interactive
```

## Standard Project Structure

Each project follows this consistent structure:

```
projectX_name/
├── README.md              # Project overview and usage
├── config.py              # Hyperparameters and configuration
├── <implementation>.py    # Core implementation(s)
├── train_<component>.py   # Training script
├── test_<component>.py    # Testing/validation script
└── download_data.py       # Data acquisition (if needed)
```

## Code Documentation Style

**CRITICAL:** All code must follow the documentation pattern established in Projects 1 & 2. Each implementation file includes:

1. **File header** with referenced research papers:
```python
"""
Component Name

This file implements X following:
- "Paper Title" (Authors, Year) - Key contribution
- "Paper Title 2" (Authors, Year) - Key contribution 2

Each component is explained with paper references for educational purposes.
"""
```

2. **Class/Function docstrings** with structured sections:
   - `PAPER:` Reference to original research
   - `PAPER CONTEXT:` Explanation from papers
   - `INTUITION:` Conceptual explanation
   - `IMPLEMENTATION:` How it's coded
   - `WHY:` Design rationale

3. **Inline comments** explaining algorithmic steps with paper references

See [model.py](phase1_foundation/project1_minimal_gpt/model.py) and [tokenizer.py](phase1_foundation/project2_tokenizer/tokenizer.py) as templates.

## Configuration Management

Each project has a `config.py` file with:
- Model hyperparameters (architecture)
- Training hyperparameters (optimization)
- System configuration (device)
- Data paths

Configuration validation is required before training:
```python
config.validate_config()
config.print_config()
```

**Key constraints:**
- `N_EMBD` must be divisible by `N_HEAD` (for multi-head attention)
- `VOCAB_SIZE - NUM_SPECIAL_TOKENS` must be positive for BPE

## Data Management Pattern

Projects requiring training data should follow this pattern:

1. **`download_data.py`** - Downloads/prepares training data
   - Supports multiple data sources/sizes
   - Uses Hugging Face datasets when possible
   - Provides clear progress indicators
   - Outputs to `data/<dataset_name>.txt`

2. **`config.py`** - Points to downloaded data
   ```python
   TRAIN_DATA_PATH = os.path.join(
       os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
       "data",
       "<dataset_name>.txt"
   )
   ```

3. **Data format:** Single text file, UTF-8 encoded, one document per line

## Model Architecture (Project 1)

```
Input (B, T)
    ↓
Token Embedding + Position Embedding
    ↓
Transformer Blocks × N
    ├── CausalSelfAttention (Q, K, V projections + multi-head)
    ├── Add & Norm (Residual + LayerNorm)
    ├── FeedForward (4× expansion)
    └── Add & Norm
    ↓
Final LayerNorm
    ↓
Linear Head (vocab_size)
    ↓
Output Logits (B, T, vocab_size)
```

**Pre-LN Architecture:** LayerNorm is applied BEFORE sublayers (GPT-2 style), not after.

## BPE Tokenization (Project 2)

**Paper:** "Byte-Pair Encoding: Subword-Based Machine Translation" (Sennrich et al., 2016)

- Iteratively merges most frequent character pairs
- Balances vocabulary size vs sequence length
- Handles OOV words via subword splitting
- Greedy longest-match-first encoding

## Checkpoint Format

Checkpoints are saved as `.pt` or `.pkl` files:

**Model checkpoints (.pt):**
```python
{
    'iter': int,
    'model_state_dict': dict,
    'optimizer_state_dict': dict,
    'train_loss': float,
    'val_loss': float,
}
```

**Tokenizer checkpoints (.pkl):**
```python
{
    'vocab': Dict[str, int],      # token → ID
    'inverse_vocab': Dict[int, str],  # ID → token
    'merges': List[Tuple[str, str]],  # merge operations in order
}
```

## Paper References Core to This Project

1. **"Attention is All You Need"** (Vaswani et al., 2017) - Transformer architecture
2. **"Improving Language Understanding by Generative Pre-Training"** (GPT-1, 2018)
3. **"Language Models are Unsupervised Multitask Learners"** (GPT-2, 2019)
4. **"Byte-Pair Encoding: Subword-Based Machine Translation"** (Sennrich et al., 2016)
5. **"The WikiText Long Term Dependency Language Modeling Dataset"** (Merity et al., 2016)
6. **"Adam: A Method for Stochastic Optimization"** (Kingma & Ba, 2014)
7. **"Layer Normalization"** (Ba et al., 2016)

## Implementation Notes for Future Projects

- **Project 3 (Pre-training):** Use BPE tokenizer from Project 2, scale to 125M parameters
- **Project 4+**: Each phase builds incrementally - don't skip foundations
- Always use `uv run python` for running scripts from project root
- All paths in scripts should be relative to project root

## GPU Requirements

- Minimum: 8GB VRAM recommended
- Fallback: CPU mode available (`DEVICE = "cuda" if torch.cuda.is_available() else "cpu"`)
- Project 1 trains in ~30 minutes to 2 hours on consumer GPU
- Project 2 tokenizer trains in ~5-15 minutes depending on vocab size

## Success Criteria Checklist

For each project, ensure:
- [ ] Extensive docstring documentation with paper references
- [ ] Config validation with `validate_config()` and `print_config()`
- [ ] Download script for required data
- [ ] Test script with validation examples
- [ ] README with clear usage instructions
- [ ] All commands use `uv run python` from project root
