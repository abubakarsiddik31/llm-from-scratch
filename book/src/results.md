# Results Log

Measured results from real runs in this repository. Hardware: NVIDIA
GeForce RTX 3050 (8 GB), PyTorch 2.11 + CUDA 12.8. Models and datasets are
deliberately small so every number is reproducible on a laptop GPU.

## Chapter 1 — Character-Level GPT

| Metric | Value |
|---|---|
| Dataset | Tiny Shakespeare (~1M chars) |
| Vocabulary | ~65 characters |
| Model size | ~10M parameters |
| Training time | < 1 hour on GPU |
| Target validation loss | 1.5–1.8 |

## Chapter 2 — BPE Tokenizer

| Metric | Value |
|---|---|
| Corpus | WikiText-2 (~11 MB) |
| Vocabulary | 5,000 tokens (3,983 learned merges) |
| Training time (optimized loop) | ~12 minutes |
| Training time (naive full-rescan loop) | ~5 hours |
| Speedup from distinct-word representation | ~50× |

The optimization: count adjacent pairs over **distinct words with
multiplicities** instead of rescanning the full corpus per merge.

## Chapter 3 — BERT-style Encoder

| Metric | Value |
|---|---|
| Corpus | WikiText-2 |
| Model size | 4.8M parameters (6 layers, 256 hidden, 8 heads) |
| Iterations | 10,000 (batch 32, block 128) |
| Training time | ~12 minutes on GPU |
| MLM loss | 4.2 → ~1.4 (train), best val 2.85 |

## Chapter 4 — SimCSE

Fine-tuned from the Chapter 3 checkpoint on 100k Wikipedia sentences,
batch 64, τ = 0.05, lr 5e-5, max sequence length 32.

| Model | STS-B dev Spearman ↑ |
|---|---|
| MLM encoder, raw `[CLS]` (baseline) | 16.6 |
| SimCSE, 1 epoch | 27.0 |
| **SimCSE, 3 epochs (final)** | **34.7** |
| SimCSE, 5 epochs | 33.5 |

| Metric | Value |
|---|---|
| Time per epoch | ~26 s |
| Improvement over baseline | **+18.1 Spearman** |
| Paper reference (BERT-base, 110M params, 10⁶ sentences) | 82.5 |

Evaluation protocol: cosine similarity vs human scores, Spearman rank
correlation, dropout OFF, without the training-only MLP head. STS-B *test*
labels are hidden (GLUE leaderboard only), so dev is the reportable split.

## Chapter 5 — Pre-Training a 125M GPT

Tokenizer (16,384-token BPE, trained on a seeded 100 MB sample of
`wiki.train.raw`):

| Metric | Value |
|---|---|
| Vocabulary | 16,384 (2,612 characters + 13,768 merges + 4 special tokens) |
| Tokenizer training time | 113.1 s (incremental trainer) |
| Distinct words in sample | 284,771 |
| Compression (probe sentence) | 4.7 chars/token |
| Trainer equivalence test | 116/116 merges identical to naive rescan |

For comparison, the chapter 2 rescan loop needed ~12 minutes for 3,983
merges on WikiText-2; the incremental trainer did 15,367 merges on the
same corpus in 24.9 s.

Encoded corpus (uint16 memmap files):

| Split | Documents | Tokens | `<UNK>` rate |
|---|---|---|---|
| train | 305,497 | 123,564,629 | 0.005% |
| val | 620 | 257,564 | 0.038% |
| test | 708 | 296,257 | 0.007% |

Training run (4,000 steps × 32,768 tokens/step ≈ 131M tokens, one pass
over the train split):

| Metric | Value |
|---|---|
| Wall time | 2 h 36 min (RTX 3050, bf16) |
| Throughput | ~14,100 tokens/s steady |
| GPU memory | 5.8 GB of 8 GB |
| Val loss (iter 0 → best) | 9.83 → **3.385** (iter 3,800) |
| Final val loss | 3.408 (perplexity 30.2) |
| Sample quality | article headings, dated Wikipedia voice, confabulated facts |

Context: GPT-2's 37.5 perplexity on WikiText-103 is zero-shot (trained on
WebText, different tokenizer) and not comparable to this in-domain run.
A Chinchilla-optimal token budget for 126M parameters would be ~2.5B
tokens, about 19× this run (~50 hours at the measured throughput).

## Chapter 6 — Supervised Fine-Tuning

Fine-tuned the Chapter 5 base model on Stanford Alpaca (52,002 triples;
28 dropped for empty outputs; 50,973 train / 1,000 val after encoding and
a seeded split), Alpaca template, response-only loss masking, lr 2e-5,
2,400 steps = 153,600 example visits (~3.0 epochs).

| Metric | Value |
|---|---|
| Encoding speed | ~1,600 examples/s (~33 s for the full set) |
| Corpus averages | 65.8 prompt + 70.3 response tokens; 0.003% `<UNK>` |
| Wall time | ~1 h 42 min (bf16, 8 × 8 × 511 batch shape, ~2.5 s/step) |
| GPU memory | ~5.8 GB of 8 GB |
| Val loss (response tokens), base model | 4.994 |
| Val loss, 100 / 400 / 800 steps | 3.708 / 3.168 / 2.954 |
| Final val loss (2,400 steps) | **2.766** (perplexity 15.9) |
| Best checkpoint | iter 2,300 (val 2.769); val monotone down, no overfit in 3 epochs |
| Behavior change | answers in the template slot and stops via `<EOS>`; facts confabulated, shallow transformations (copied input on a past-tense rewrite) |

Before SFT, the base model continued the template into WikiText prose
("...### Response:= = = = Contents of the House = = = =..."). After SFT it
produces answer-shaped spans and ends its turn; sample quality limits are
recorded honestly in the chapter.

## Chapter 7 — LoRA Fine-Tuning

Adapted the Chapter 5 base model with LoRA (r = 8, alpha = 16, targets:
the fused QKV `c_attn` in all 16 blocks) on the same encoded Alpaca
arrays as Chapter 6, lr 1e-4, 2,400 steps = 153,600 example visits
(~3.0 epochs), response-only loss.

| Metric | Value |
|---|---|
| Trainable parameters | 393,216 of 126,666,240 (0.310%) |
| Init identity check | iter-0 val loss 4.9941 = the base model's number |
| Wall time | ~78 min (bf16, ~2.0 s/step) |
| GPU memory | ~4.0 GB of 8 GB (full fine-tune: ~5.8 GB) |
| Final val loss (2,400 steps) | 3.802 (perplexity 44.8) |
| Head-to-head vs Chapter 6 | 4.994 → 2.766 (full) vs 4.994 → 3.802 (LoRA); ~53% of the loss improvement with 0.31% of the parameters |
| Val curve | monotone down, no overfit in 3 epochs |
| Sample behavior | format partially learned; repetition loops over short fragments (full SFT looped over plausible sentences) |

## Reproducing

Every number above comes from the commands in
[Getting Set Up](./setup.md) with default configurations. Checkpoints are
written to `checkpoints/` (gitignored); training logs are deterministic
modulo CUDA nondeterminism.
