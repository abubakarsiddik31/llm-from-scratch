# Project 4: Pre-Train a ~125M GPT on WikiText-103

Scale the chapter 1 model from 10M parameters to GPT-2-small class
(~126M) and pre-train it on WikiText-103 with the chapter 2 tokenizer,
retrained at 16,384 tokens. This is the model Phase 2 (SFT, LoRA, DPO)
will fine-tune.

## Papers

| Topic | Paper |
|---|---|
| Architecture | Radford et al., 2019, *Language Models are Unsupervised Multitask Learners* (GPT-2) |
| Training recipe | Brown et al., 2020, *Language Models are Few-Shot Learners* (GPT-3), Appendix B |
| Token budget | Hoffmann et al., 2022, *Training Compute-Optimal LLMs* (Chinchilla) |
| Tokenizer | Sennrich et al., 2016, *Byte-Pair Encoding of Subword Units*; fastBPE (2018) for incremental pair counts |
| Attention kernel | Dao et al., 2022, *FlashAttention* (via PyTorch SDPA) |
| Corpus | Merity et al., 2016, *Pointer Sentinel Mixture Models* (WikiText) |

## What's different from chapter 1

- **16 layers instead of 12.** GPT-2 small reaches 124M params with a
  50,257-token vocab; our from-scratch 16,384-token BPE makes the
  embedding table ~26M smaller, so we buy the difference in depth:
  16 layers x 768 width x 12 heads x 16k vocab = 126,273,024 parameters
  (verified in `test_model.py`).
- **Fused causal attention** (`F.scaled_dot_product_attention`) instead of
  the hand-written softmax: same math, no materialized T x T matrix.
- **GPT-2 initialization** with the 1/sqrt(2*N_LAYER) residual shrink.
- **Weight tying** between `wte` and `lm_head` (saves 12.6M params).
- **Training loop**: bf16 autocast, gradient accumulation (8 x 8 x 512 =
  32,768 tokens/step), warmup + cosine LR, grad clipping, fused AdamW,
  memmap'd uint16 token files.
- **Fast tokenizer**: the chapter 2 rescan loop would need most of a day
  at WikiText-103 scale; `tokenizer.py` updates pair counts incrementally
  and `test_model.py` proves it identical to the naive algorithm.

## Pipeline (run in order, from the repo root)

```bash
uv sync --group all   # once

# 1. corpus (~190 MB download for the full WikiText-103-raw)
uv run python phase1_foundation/project4_pretrain/download_data.py --size full
#    (--size small fetches WikiText-2-raw, ~5 MB, for smoke tests)

# 2. tokenizer (16,384 vocab from a seeded 100 MB sample; minutes)
uv run python phase1_foundation/project4_pretrain/train_tokenizer.py

# 3. encode splits to uint16 token files (~230 MB on disk)
uv run python phase1_foundation/project4_pretrain/prepare_data.py

# 4. sanity-check everything
uv run python phase1_foundation/project4_pretrain/test_model.py

# 5. pre-train (default: 4,000 steps ~= one pass over the corpus; hours)
uv run python phase1_foundation/project4_pretrain/train.py
#    interrupt-safe: relaunch with --resume to continue from
#    checkpoints/project4/checkpoint_latest.pt

# 6. sample
uv run python phase1_foundation/project4_pretrain/generate.py --prompt "The theory"
uv run python phase1_foundation/project4_pretrain/generate.py --interactive
```

## Hardware notes (RTX 3050 8 GB)

- bf16 autocast (Ampere sm_86): no gradient scaler needed.
- BATCH_SIZE 8 x BLOCK_SIZE 512 fits comfortably (verified); if you OOM,
  drop `--batch_size 4` and double `--grad_accum` (same tokens/step).
- BLOCK_SIZE 512 instead of GPT-2's 1024 roughly doubles throughput; the
  config documents the deviation.
- Measured on a 3-step smoke run at the default batch shape (8 x 512,
  accum 8, bf16): ~5.7k tokens/s; the real run settles at ~14.1k tokens/s
  once kernels warm up. The default 4,000-step run (~131M tokens, one pass
  over WikiText-103) took 2 h 36 min at 5.8 GB of GPU memory; final val
  loss 3.408 (perplexity 30.2), best 3.385 at iter 3,800. Numbers also
  recorded in `book/src/results.md`.
- Chinchilla-optimal for 126M params would be ~2.5B tokens, about 19x this
  run (~50 hours at the measured throughput); the loss curves we actually
  get are recorded honestly as runs complete.

## Smoke validation (WikiText-2, 2026-09-17)

The full pipeline was verified end to end on the small corpus before the
real run: corpus download/extract, 16,384-token BPE training (15,367
merges on 10.4 MB in 24.9 s; the chapter 2 rescan loop needed ~12 minutes
for 3,983 merges on a same-size corpus), split encoding (2.42M train
tokens, ~0.0% UNK), 5-iteration train on GPU (loss 9.88 -> 8.54 vs
ln(16,384) = 9.70 at random init), and checkpoint reload + generation.
All 11 tests in `test_model.py` pass, including exact merge-list equality
between the incremental trainer and the naive chapter 2 algorithm.

## Important: don't train from a synced folder

Training writes checkpoints continuously. Run training from the local-disk
clone, never from OneDrive/Dropbox (see `book/src/setup.md` and the
repository AGENTS.md - the `.git` folder was lost to a sync conflict once
already).

## Files

| File | Purpose |
|---|---|
| `config.py` | All hyperparameters with paper references; `validate_config()` / `print_config()` |
| `download_data.py` | WikiText-2-raw / WikiText-103-raw from the allow-listed Zenodo mirror |
| `tokenizer.py` | Fast BPE: incremental pair counts (heap + lazy deletion), rank-based encode with per-word memoization; chapter 2-compatible checkpoints |
| `train_tokenizer.py` | Train the 16,384-token BPE on a seeded corpus sample |
| `prepare_data.py` | Encode splits to `data/project4/*.bin` with `<BOS>`/`<EOS>` at document boundaries |
| `model.py` | The ~126M GPT (SDPA attention, GPT-2 init, weight tying) |
| `train.py` | bf16 + grad-accum training loop, warmup/cosine LR, checkpoints, resume |
| `generate.py` | Sampling CLI (temperature, top-k, interactive) |
| `test_model.py` | Tokenizer equivalence + round-trip; causality, overfit, checkpoint tests |
