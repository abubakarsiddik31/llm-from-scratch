# The Roadmap: 10 Phases, 32 Projects

The full curriculum, from a character-level GPT to a deployed LLM
application. Chapters written so far are linked; the rest follow the same
pattern as the repository grows.

## Phase 1 — Foundation (Part I of this book)

| Project | Topic | Status |
|---------|-------|--------|
| 1 | Character-Level GPT | ✅ [Chapter 1](./ch01-char-gpt.md) |
| 2 | BPE Tokenizer | ✅ [Chapter 2](./ch02-bpe-tokenizer.md) |
| 3 | Contextual Embeddings (BERT-style) | ✅ [Chapter 3](./ch03-bert-embeddings.md) |
| 3B | SimCSE (Sentence Embeddings) | ✅ [Chapter 4](./ch04-simcse.md) |
| 4 | Pre-train 125M Model | ✅ [Chapter 5](./ch05-pretrain.md) |

**Focus:** multi-head attention, transformer blocks, training loops, text
generation, masked language modeling, contrastive learning.

## Phase 2 — Fine-Tuning

| Project | Topic | Status |
|---------|-------|--------|
| 5 | Supervised Fine-Tuning (SFT) | ✅ [Chapter 6](./ch06-sft.md) |
| 6 | LoRA Fine-Tuning | |
| 7 | DPO (Direct Preference Optimization) | |

**Focus:** instruction formatting, memory-efficient training, preference
alignment.

## Phase 3 — Core Inference Optimizations

| Project | Topic | Speedup |
|---------|-------|---------|
| 8 | Mixed Precision Training & Inference | 2–4× |
| 9 | KV-Cache | 10–30× |
| 10 | Flash Attention | 2–4× |

**Papers:** Micikevicius 2018 · Transformer-XL · Flash Attention 1 & 2

## Phase 4 — Advanced Inference Optimizations

| Project | Topic | Speedup |
|---------|-------|---------|
| 11 | Prompt Caching | 5–50× |
| 12 | Speculative Decoding | 2–3× |
| 13 | Dynamic Batching | 3–10× |
| 14 | Paged Attention | Near-zero memory waste |

**Papers:** SemCache · vLLM · Orca · Speculative Sampling

## Phase 5 — Quantization

| Project | Topic | Benefit |
|---------|-------|---------|
| 15 | Post-Training Quantization (PTQ) | 2–4× smaller |
| 16 | KV-Cache Quantization | 50% cache reduction |
| 17 | Quantization-Aware Training (QAT) | Better accuracy |

**Papers:** GPTQ · LLM.int8() · QAT (Jacob 2018)

## Phase 6 — Model Compression

| Project | Topic | Reduction |
|---------|-------|-----------|
| 18 | Pruning (Structured & Unstructured) | 30–60% |
| 19 | Knowledge Distillation | Smaller models |
| 20 | Weight Sharing | 10–30% |

**Papers:** Wanda · Distilling Knowledge · ALBERT

## Phase 7 — Advanced Architectures

| Project | Topic | Complexity |
|---------|-------|------------|
| 21 | Sparse Attention | O(n√n) |
| 22 | Mixture-of-Experts (MoE) | Same compute, more params |
| 23 | Memory-Efficient Attention | 2–4× less memory |

**Papers:** Longformer · BigBird · Switch Transformers · Mixtral

## Phase 8 — Parallelism & Scaling

| Project | Topic | Outcome |
|---------|-------|---------|
| 24 | Tensor Parallelism | Multi-GPU training |
| 25 | Pipeline Parallelism | Better GPU utilization |

**Papers:** Megatron-LM · GPipe · PipeDream

## Phase 9 — Compiler Optimizations

| Project | Topic | Speedup |
|---------|-------|---------|
| 26 | Operator Fusion | 20–40% |
| 27 | Graph Optimization | 1.5–3× |
| 28 | Early Exit | 30–50% |

**Papers:** Triton · XLA · TVM · PABEE

## Phase 10 — Production Deployment

| Project | Topic | Outcome |
|---------|-------|---------|
| 29 | Model Serving Optimization | Production API |
| 30 | Docker Deployment | One-command deploy |
| 31 | Interactive UI (Gradio) | User-friendly |

## The learning pathway

| Phases | Theme |
|--------|-------|
| 1–2 | Build and fine-tune LLMs |
| 3–4 | Speed up generation |
| 5–6 | Shrink models |
| 7–8 | Train larger models |
| 9–10 | Ship to production |

Projects are **cumulative**: the tokenizer of Phase 1 feeds the pre-training
of Project 4, the model of Project 4 is what gets fine-tuned in Phase 2, and
the same model is the one optimized in Phases 3–9. Skipping foundations
catches up with you by Phase 3.
