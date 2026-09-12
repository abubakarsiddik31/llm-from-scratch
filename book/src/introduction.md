# Introduction

<div class="book-title-page">

# Hands-On LLM Implementation

### Build everything from scratch. Ship to production.

*A practical, implementation-first guide to training and deploying large language models*

</div>

## Who this book is for

You already understand the theory. You want to build the thing.

Every chapter here is backed by a working project in the companion repository,
[`llm-from-scratch`](https://github.com/abubakarsiddik31/llm-from-scratch). The
tokenizer, attention, training loop, and contrastive objective are all written
from scratch, with the original research papers cited in the source code.

Across the book we assemble one pipeline: tokenization, pre-training,
fine-tuning, optimization, and production deployment.

## What the first four chapters cover

1. **Character-Level GPT.** Attention, transformer blocks, and a training
   loop, small enough to understand every line.
2. **The BPE Tokenizer.** Subword tokenization, the same method GPT-2 and
   GPT-3 use.
3. **Contextual Embeddings.** A BERT-style bidirectional encoder trained with
   masked language modeling.
4. **SimCSE.** Contrastive learning that turns the encoder into a
   sentence-embedding model, where dropout is the only data augmentation.

Later phases (fine-tuning, inference optimization, quantization, parallelism,
deployment) follow the same pattern: implement it, measure it against the
paper, ship it.

## How this differs from a tutorial

- The documentation is paper-first. Every module's docstrings quote the paper
  it implements (`PAPER:`, `PAPER CONTEXT:`, `INTUITION:`, `IMPLEMENTATION:`,
  `WHY:`), so the code and the literature don't drift apart.
- The results are measured. Each chapter ends with numbers from real runs on
  a single consumer GPU (an 8 GB RTX 3050), not the ones we wish we'd got.
- Nothing is a black box. If a component appears in the code, the book
  explains why it was designed that way.

## Conventions used in this book

- Code blocks reference files in the repository, e.g.
  [`phase1_foundation/project3b_simcse/model.py`](https://github.com/abubakarsiddik31/llm-from-scratch/blob/main/phase1_foundation/project3b_simcse/model.py)
- All commands run from the repository root with `uv run python ...`
- Citations like *(Gao, Yao & Chen, 2021)* refer to the papers listed at the
  end of each chapter.

Start with [Getting Set Up](./setup.md), then head to
[Chapter 1](./ch01-char-gpt.md).
