# Introduction

<div class="book-title-page">

# 🚀 Hands-On LLM Implementation

### Build everything from scratch. Ship to production.

*A practical, implementation-first guide to training and deploying Large Language Models*

</div>

## Who this book is for

This book is for people who **understand the theory** and want to **build**.

Most LLM content stops at explanations. This book stops only when the code
runs. Every chapter is a working project in the companion repository
[`llm-from-scratch`](https://github.com/abubakarsiddik31/llm-from-scratch) —
every component (tokenizer, attention, training loop, contrastive objective)
is implemented from scratch, with the original research papers cited inline in
the source.

The pipeline we build across the book:

> **Tokenization → Pre-training → Fine-tuning → Optimization → Production Deployment**

## Philosophy

**Learn by doing.** Each project builds on the previous one:

1. **Chapter 1 — Character-Level GPT.** Attention, transformer blocks, and a
   training loop, small enough to understand every line.
2. **Chapter 2 — The BPE Tokenizer.** Subword tokenization, the same method
   GPT-2 and GPT-3 use.
3. **Chapter 3 — Contextual Embeddings.** A BERT-style bidirectional encoder
   trained with masked language modeling.
4. **Chapter 4 — SimCSE.** Contrastive learning that turns the encoder into a
   sentence-embedding model — where *dropout* is the only data augmentation.

Later phases (fine-tuning, inference optimization, quantization, parallelism,
deployment) follow the same pattern: implement, measure against the paper,
then ship.

## What makes this different from a tutorial

- **Paper-first documentation.** Every module's docstrings quote the paper it
  implements — `PAPER:`, `PAPER CONTEXT:`, `INTUITION:`, `IMPLEMENTATION:`,
  `WHY:` — so the code and the literature never drift apart.
- **Measured results.** Each chapter ends with real numbers from real runs on
  a single consumer GPU (an 8 GB RTX 3050), not aspirational ones.
- **No black boxes.** If a component appears in the code, it appears in the
  book with an explanation of *why* it is designed that way.

## Conventions used in this book

- Code blocks reference files in the repository, e.g.
  [`phase1_foundation/project3b_simcse/model.py`](https://github.com/abubakarsiddik31/llm-from-scratch/blob/main/phase1_foundation/project3b_simcse/model.py)
- All commands run from the repository root with `uv run python ...`
- Citations like *(Gao, Yao & Chen, 2021)* refer to the papers listed at the
  end of each chapter.

Start with [Getting Set Up](./setup.md), then head to
[Chapter 1](./ch01-char-gpt.md).
