# Getting Set Up

## Hardware

Everything in Part I was built and trained on a single consumer GPU:

| Requirement | Details |
|-------------|---------|
| **GPU** | CUDA-capable; 8 GB VRAM recommended (all runs here used an RTX 3050 8 GB) |
| **Python** | 3.10–3.13 |
| **Env manager** | [`uv`](https://docs.astral.sh/uv/) |
| **Knowledge** | Neural networks & attention (theory only — we implement the rest) |

## Installation

```bash
# Clone
git clone git@github.com:abubakarsiddik31/llm-from-scratch.git
cd llm-from-scratch

# Install all dependencies (PyTorch CUDA wheels are pre-configured)
uv sync --group all
```

### GPU notes

The repository's `pyproject.toml` pins PyTorch to the **CUDA 12.8 wheel
index**, so `uv sync` installs a GPU build by default:

```toml
[tool.uv.sources]
torch = [{ index = "pytorch-cu128" }]

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true
```

On a CPU-only machine, remove that block to fall back to CPU builds. Verify
your setup:

```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# True NVIDIA GeForce RTX 3050
```

> **Tip:** do *not* keep the repository inside a cloud-synced folder (OneDrive,
> Dropbox). Checkpoint writes during training are large and frequent; sync
> clients cause lock conflicts and slow I/O. Keep it on a local disk.

## Running the projects

All commands run from the repository root via `uv run`:

```bash
# Chapter 1 — train character-level GPT, then generate
uv run python phase1_foundation/project1_minimal_gpt/train.py
uv run python phase1_foundation/project1_minimal_gpt/generate.py --prompt "ROMEO:" --interactive

# Chapter 2 — download data, train tokenizer, test it
uv run python phase1_foundation/project2_tokenizer/download_data.py --size small
uv run python phase1_foundation/project2_tokenizer/train_tokenizer.py --vocab_size 5000
uv run python phase1_foundation/project2_tokenizer/test_tokenizer.py --interactive

# Chapter 3 — BERT-style MLM/NSP pre-training
uv run python phase1_foundation/project3_contextual_embeddings/download_data.py --size small
uv run python phase1_foundation/project3_contextual_embeddings/train.py

# Chapter 4 — SimCSE
uv run python phase1_foundation/project3b_simcse/download_data.py
uv run python phase1_foundation/project3b_simcse/test_model.py   # sanity tests
uv run python phase1_foundation/project3b_simcse/train.py
uv run python phase1_foundation/project3b_simcse/evaluate.py
```

## Repository layout

```
llm-from-scratch/
├── phase1_foundation/
│   ├── project1_minimal_gpt/           # Chapter 1
│   ├── project2_tokenizer/             # Chapter 2
│   ├── project3_contextual_embeddings/ # Chapter 3
│   └── project3b_simcse/               # Chapter 4
├── data/            # gitignored: downloaded corpora
└── checkpoints/     # gitignored: trained models
```

Each project follows the same structure: `config.py` (hyperparameters with
paper references), the core implementation, a training script, a test script,
and a data downloader. Checkpoints are `.pt`/`.pkl` files with model state,
optimizer state, and metrics.
