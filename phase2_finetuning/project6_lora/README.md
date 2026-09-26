# Project 6: LoRA Fine-Tuning

Reach the project 5 goal (instruction following) from the same base
checkpoint on the same Alpaca data while training 0.31% of the
parameters, using LoRA (Hu et al., 2021): freeze the weights, learn
low-rank updates B x A on the attention projections instead.

## Papers

| Topic | Paper |
|---|---|
| The method | Hu et al., 2021, *LoRA: Low-Rank Adaptation of Large Language Models* |
| The task (stage-one alignment SFT) | Ouyang et al., 2022, *Training Language Models to Follow Instructions with Human Feedback* (InstructGPT) |
| Dataset, template, schedule | Taori et al., 2023, *Stanford Alpaca* |
| Training-loop conventions | Brown et al., 2020, *Language Models are Few-Shot Learners* (GPT-3), Appendix B |

## What's different from Project 5

- **393,216 trainable parameters instead of 126,273,024** (0.311%;
  asserted in `test_model.py`). Each `c_attn` (the fused QKV projection,
  768 -> 2304) gets a rank-8 update: A is 8 x 768, B is 2304 x 8, and
  y = W x + (alpha / r) * B (A x) with alpha = 16.
- **The base is frozen** (`requires_grad=False` everywhere but the
  adapters), so the optimizer holds moments for 0.4M instead of 126M
  parameters. That is where LoRA's memory story lives.
- **B starts at zero** (paper init), so step 0 computes exactly what the
  base model computes; `test_model.py` checks the identity.
- **Higher LR, no weight decay**: 1e-4 (the paper's GPT-3 sweeps sit at
  1e-4..3e-4) and 0.0 (pulling the update toward zero fights the
  objective on a short schedule).
- **`merge()`** folds the trained update into the frozen weight, the
  paper's deploy-time trick: the merged model has the original
  architecture and zero added latency. Outputs match to float tolerance.

Everything else is deliberately identical to project 5: same base
checkpoint (`checkpoints/project4/model_final.pt` by default), same
encoded Alpaca arrays (`data/project5/*.npy`, reused verbatim - no
download/encode scripts here on purpose), same 64-examples-per-step batch
shape, same 2,400-step (~3 epochs) schedule, response-only loss masking,
bf16 + grad accumulation, five-key checkpoints plus a `lora_config` dict.
The point is a controlled full-SFT-vs-LoRA comparison for chapter 7.

## The experiment

LoRA as an ALTERNATIVE to full fine-tuning (paper-faithful, the default):
base = project 4 checkpoint. For continued adaptation instead, point
`--base_checkpoint` at project 5's `model_final.pt`.

## Pipeline (run in order, from the repo root)

```bash
uv sync --group all   # once

# 1. data: reuse project 5's (encode once there if you haven't)
uv run python phase2_finetuning/project5_sft/download_data.py --size full
uv run python phase2_finetuning/project5_sft/prepare_data.py

# 2. sanity-check LoRA (8 tests, CPU-friendly)
uv run python phase2_finetuning/project6_lora/test_model.py

# 3. fine-tune adapters only (default: 2,400 steps ~= 3 epochs)
uv run python phase2_finetuning/project6_lora/train.py
#    variants:
#    --base_checkpoint checkpoints/project5/model_final.pt   (continued adaptation)
#    --lora_r 16            bigger adapters
#    --resume               continue from checkpoint_latest.pt

# 4. ask it things (also loads plain project 5 checkpoints)
uv run python phase2_finetuning/project6_lora/generate.py --instruction "Give three tips for staying healthy."
uv run python phase2_finetuning/project6_lora/generate.py --interactive
```

## Hardware notes (RTX 3050 8 GB)

- Same batch shape as projects 4/5 (8 x 512, accumulation 8, bf16);
  adapter-only gradients shave optimizer-state memory, so this fits with
  more headroom than the full-SFT run.
- CPU test suite (8 tests, 2026-09-26): init identity, freeze and
  trainable-count arithmetic, merge equivalence, checkpoint round-trip
  (LoRA and plain), base-checkpoint wrap (393,216 trainable /
  126,666,240 total = 0.310%), EOS-stop sampling. One honest note from
  the CPU tests: adapter-only training on a RANDOM tiny base plateaus
  early (4.82 -> 4.20) because a random frozen base offers noise
  features; LoRA's premise is a pre-trained base, which is what the real
  run supplies.
- Run numbers for the real fine-tuning go to `book/src/results.md` when
  complete.

## Important: don't train from a synced folder

Same rule as every project that writes checkpoints: run from the
local-disk clone, never from OneDrive/Dropbox (see `book/src/setup.md`
and the repository AGENTS.md).

## Files

| File | Purpose |
|---|---|
| `config.py` | LoRA hyperparameters (r, alpha, targets) with paper references; reuses project 5's data by design |
| `lora.py` | `LoRALinear`, `apply_lora()` (freeze + wrap), `merge()`; state-dict-compatible keys |
| `model.py` | Same GPT as projects 4/5 (code identical); adapters wrap it at runtime |
| `tokenizer.py` / `template.py` | Verbatim copies of project 5's load-only tokenizer and Alpaca template |
| `train.py` | LoRA training loop: base load, freeze + wrap, adapter-only optimizer, checkpoints |
| `generate.py` | Loads LoRA checkpoints (and plain ones); `<EOS>`-stopping sampler |
| `test_model.py` | Init identity, freeze/count, merge equivalence, round-trip, base wrap, EOS stop |
