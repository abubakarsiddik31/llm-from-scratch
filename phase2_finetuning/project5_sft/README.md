# Project 5: Supervised Fine-Tuning (SFT)

Attach the Project 4 base model (~126M, pre-trained on WikiText-103) to
instruction data and continue training it so it answers instructions
instead of merely continuing text. First project of Phase 2; LoRA and DPO
later attach to the same base checkpoint.

## Papers

| Topic | Paper |
|---|---|
| SFT stage of the alignment recipe | Ouyang et al., 2022, *Training Language Models to Follow Instructions with Human Feedback* (InstructGPT) |
| Dataset, template, hyperparameters | Taori et al., 2023, *Stanford Alpaca: An Instruction-following Llama Model* |
| How the dataset was made | Wang et al., 2023, *Self-Instruct: Aligning Language Models with Self-Generated Instructions* |
| Instruction tuning background | Wei et al., 2022, *Finetuned Language Models Are Zero-Shot Learners* (FLAN) |
| Data-quantity counterpoint | Zhou et al., 2023, *LIMA: Less Is More for Alignment* |
| LR schedule / training loop conventions | Brown et al., 2020, *Language Models are Few-Shot Learners* (GPT-3), Appendix B |

## What's different from Project 4

- **The architecture does not move.** `model.py` is the same code as
  `project4_pretrain/model.py`; SFT changes the data distribution and the
  loss mask. The base checkpoint's weights load before step zero and the
  optimizer starts fresh.
- **Response-only loss masking.** Targets carry `-1` (`ignore_index`) on
  prompt and pad positions. InstructGPT computes the SFT loss only on
  assistant tokens; Alpaca's actual code trains on the whole sequence.
  Both are supported: `--loss_on {response,full}` (default `response`),
  and the chapter measures the difference.
- **Alpaca's plain-text template** (`### Instruction:` / `### Input:` /
  `### Response:`) instead of new special tokens: the base tokenizer
  already knows every symbol, so no embedding resize touches the base
  weights. A `<EOS>` token (id 3) after each response teaches the model to
  stop; `generate.py` listens for it.
- **Epochs over a fixed set** (~51k examples, 3 epochs, seeded shuffles)
  instead of one pass over a stream: overfitting is the failure mode to
  watch, hence `checkpoint_best.pt` by val loss.
- **Lower learning rate**: 2e-5 (Alpaca) vs 2.5e-4 pre-training.

## The tokenizer and one honest limitation

The Project 4 tokenizer (16,384 vocab) is loaded, never retrained
(`tokenizer.py` is the encoding half of Project 4's, load-only). Because
that tokenizer normalizes whitespace (WikiText is a word stream), SFT
examples encode as one long line: the `###` markers carry the structure,
and the model never learns newline placement. Also, encoding a prompt
word-by-word (`encode_words(..., final=False)`) matters: the trailing
space convention means `encode(prompt)` is NOT a prefix of
`encode(prompt + response)` - `template.py` owns that junction and
`test_model.py` proves the identity.

## Pipeline (run in order, from the repo root)

```bash
uv sync --group all   # once

# 1. dataset (~23 MB JSON; 52,002 entries, 51,974 usable)
uv run python phase2_finetuning/project5_sft/download_data.py --size full
#    (--size small keeps a seeded 2,000-example sample for smoke tests)

# 2. encode to padded token/label arrays (uint16 + int16)
uv run python phase2_finetuning/project5_sft/prepare_data.py
#    (--size small for the smoke subset)

# 3. sanity-check everything (11 tests)
uv run python phase2_finetuning/project5_sft/test_model.py

# 4. fine-tune (default: 2,400 steps = 64 examples/step ~= 3 epochs; hours)
uv run python phase2_finetuning/project5_sft/train.py
#    variants:
#    --loss_on full            Alpaca-style loss over the whole sequence
#    --resume                  continue from checkpoint_latest.pt
#    --max_iters 50            smoke run

# 5. ask it things
uv run python phase2_finetuning/project5_sft/generate.py --instruction "Give three tips for staying healthy."
uv run python phase2_finetuning/project5_sft/generate.py --interactive
```

## Hardware notes (RTX 3050 8 GB)

- Same batch shape as pre-training (8 x 512, accumulation 8 -> 64
  examples/step, bf16 autocast), so the memory profile carries over.
- Data validated (2026-09-26): the full Alpaca JSON holds 52,002 entries;
  28 have empty outputs (dropped), leaving 51,974. Encoding runs at
  ~1,600 examples/s (the full set takes ~33 s) with a seeded
  1,000-example val split: 50,973 train / 1,000 val, 1 example skipped
  for a too-long prompt, averaging 65.8 prompt + 70.3 response tokens
  with 819 `<UNK>` tokens (0.003% of positions). The seeded 2,000-example
  smoke subset (1,800/200) shows the same shape.
- Smoke training run (2026-09-26): 3 optimizer steps on the small subset
  (192 example visits, including the ~1.5 GB base checkpoint load) took
  24 s wall clock; the response-only val loss moved 4.945 -> 4.413. All
  11 tests in `test_model.py` pass, including the base-checkpoint
  compatibility check (126,273,024 parameters load into this project's
  model class, and the AST-verified copy of `model.py` is code-identical
  to Project 4's).
- After 3 steps the samples are still WikiText-style continuation with
  degenerate loops - expected. Instruction following and `<EOS>` stopping
  are what the full 2,400-step run is supposed to teach; its numbers go
  to `book/src/results.md` when complete.

## Important: don't train from a synced folder

Same rule as every project that writes checkpoints: run from the
local-disk clone, never from OneDrive/Dropbox (see `book/src/setup.md`
and the repository AGENTS.md).

## Files

| File | Purpose |
|---|---|
| `config.py` | All hyperparameters with paper references; base-checkpoint/tokenizer paths; `validate_config()` / `print_config()` |
| `template.py` | Alpaca prompt template + `encode_labeled()` (response mask arithmetic, trailing-space junction) |
| `tokenizer.py` | Load-only BPE tokenizer (Project 4 checkpoint); `encode_words()` for the word-by-word prompt encoding |
| `model.py` | Same GPT as Project 4 (code identical; docstrings note the SFT context) |
| `download_data.py` | Stanford Alpaca JSON from an allow-listed host; seeded small subset |
| `prepare_data.py` | Encode to `data/project5/*.npy` (tokens + response-masked labels), seeded train/val split |
| `train.py` | SFT loop: base checkpoint loading, masked loss, epochs, bf16 + grad accumulation, checkpoints, resume |
| `generate.py` | Template-conditioned sampling with `<EOS>` stop; interactive mode |
| `test_model.py` | Template + mask identity, loss semantics, NaN edge, base-checkpoint load, EOS-stop sampling |
