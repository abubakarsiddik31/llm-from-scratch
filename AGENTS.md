# AGENTS.md

Guidance for AI agents (and humans) working in this repository. This is the
canonical instruction file; `CLAUDE.md` points here.

## What this repository is

A hands-on LLM implementation roadmap: 10 phases, ~32 projects, building from
a character-level GPT to a deployed LLM application. Every component is
implemented from scratch with the original research papers cited in the
source code.

- **Remote:** `git@github.com:abubakarsiddik31/llm-from-scratch.git` (branch `main`)
- **Live book:** https://abubakarsiddik31.github.io/llm-from-scratch/
- **Owner:** Abu Bakar Siddik
- **Canonical location:** `C:\Users\abuba\projects\llm-from-scratch` (local
  disk). This is the ONLY copy of the repository on this machine. A copy
  that lived at `C:\Users\abuba\OneDrive\Desktop\project\llm-from-scratch`
  was removed on 2026-09-17 at the owner's direction: **nothing from this
  project (repo, data, checkpoints, logs) belongs on OneDrive or any
  cloud-synced folder.** The `.git` folder was already lost to a sync
  conflict once. Never clone, move, or open the project from a synced
  folder, and never run training from one; open sessions at the local-disk
  path.

Two things live side by side here:

1. `book/` — an mdBook (source in `book/src/`) that documents the training
   session as a readable guide, with hand-written SVG figures.
2. `phase1_foundation/` — the runnable projects each chapter is based on.

## Current status

| Project | Directory | Chapter | Status |
|---|---|---|---|
| 1. Character-Level GPT | `phase1_foundation/project1_minimal_gpt/` | `ch01-char-gpt.md` | complete |
| 2. BPE Tokenizer | `phase1_foundation/project2_tokenizer/` | `ch02-bpe-tokenizer.md` | complete |
| 3. Contextual Embeddings (BERT-style) | `phase1_foundation/project3_contextual_embeddings/` | `ch03-bert-embeddings.md` | complete |
| 3B. SimCSE | `phase1_foundation/project3b_simcse/` | `ch04-simcse.md` | complete |
| 4. Pre-train 125M model | `phase1_foundation/project4_pretrain/` | `ch05-pretrain.md` | complete |

Phase 2 (SFT, LoRA, DPO) onward: see the roadmap in `README.md` and
`book/src/curriculum.md`. The full pipeline is tokenization → pre-training →
fine-tuning → optimization → deployment.

## Commands

Everything runs from the repository root via `uv` (Python 3.10–3.13,
PyTorch CUDA 12.8 wheels are pinned in `pyproject.toml`):

```bash
# install / sync
uv sync --group all

# Chapter 1 — GPT
uv run python phase1_foundation/project1_minimal_gpt/train.py
uv run python phase1_foundation/project1_minimal_gpt/generate.py --prompt "ROMEO:" --interactive

# Chapter 2 — tokenizer
uv run python phase1_foundation/project2_tokenizer/download_data.py --size small
uv run python phase1_foundation/project2_tokenizer/train_tokenizer.py --vocab_size 5000
uv run python phase1_foundation/project2_tokenizer/test_tokenizer.py --interactive

# Chapter 3 — BERT-style MLM/NSP
uv run python phase1_foundation/project3_contextual_embeddings/download_data.py --size small
uv run python phase1_foundation/project3_contextual_embeddings/train.py

# Chapter 4 — SimCSE
uv run python phase1_foundation/project3b_simcse/download_data.py
uv run python phase1_foundation/project3b_simcse/test_model.py
uv run python phase1_foundation/project3b_simcse/train.py
uv run python phase1_foundation/project3b_simcse/evaluate.py
```

### Book

```bash
# build + serve locally (mdbook must be installed)
mdbook serve book          # http://localhost:3000
mdbook build book          # output in book/book/  (gitignored)

# install mdbook if missing:
# cargo install mdbook   or grab a release binary from
# https://github.com/rust-lang/mdBook/releases
```

Deploy is automatic: pushing to `main` with changes under `book/` or
`.github/workflows/book.yml` triggers `.github/workflows/book.yml`, which
builds the book and publishes it to GitHub Pages (Pages source must be set
to "GitHub Actions").

## Code conventions (established in Projects 1–2, follow them)

- **Paper-first documentation.** Every implementation file opens with a
  docstring listing the papers it follows. Classes/functions carry
  structured sections: `PAPER:`, `PAPER CONTEXT:`, `INTUITION:`,
  `IMPLEMENTATION:`, `WHY:`. Templates: `project1_minimal_gpt/model.py`,
  `project2_tokenizer/tokenizer.py`.
- **Per-project structure:** `README.md`, `config.py`, core implementation,
  `train_*.py`, `test_*.py`, `download_data.py`.
- **`config.py`** holds all hyperparameters (with paper references), device,
  and data paths; scripts call `config.validate_config()` and
  `config.print_config()` before training. Constraints: `N_EMBD % N_HEAD == 0`;
  `VOCAB_SIZE - NUM_SPECIAL_TOKENS > 0`.
- **Data:** `download_data.py` writes UTF-8 text files to `data/` (gitignored);
  configs reference them with paths built from `__file__`, relative to the
  repo root.
- **Checkpoints** (gitignored, written to `checkpoints/`):
  - `.pt`: `{'iter', 'model_state_dict', 'optimizer_state_dict', 'train_loss', 'val_loss'}`
  - tokenizer `.pkl`: `{'vocab', 'inverse_vocab', 'merges'}` (merges kept in learned order)
- All commands run from the project root with `uv run python`; never assume CWD
  is the project directory.

## Book conventions

mdBook with `[preprocessor.index]` and `[preprocessor.links]`; HTML output
config in `book/book.toml` (site-url `/llm-from-scratch/`, custom CSS in
`book/src/custom.css`). Chapters live in `book/src/`, order defined by
`book/src/SUMMARY.md`:

- `introduction.md`, `setup.md`, then Part I chapters (`ch01`–`ch04`),
  then appendices (`curriculum.md`, `results.md`).
- Each chapter: project source + papers blockquote up top, objective, core
  ideas, run commands, code tour (file table), exercises, "what's next".
- **Chapters embed real code with paper citations.** Each of ch01-ch04 has a
  "## The code" section quoting the actual `phase1_foundation` excerpts
  (attention forward pass, the BPE merge loop, the 80/10/10 masking, the
  SimCSE loss), trimmed of the long commentary docstrings but keeping the
  `# PAPER:` citation lines. The excerpts must stay in sync with the source:
  if you change the code in `phase1_foundation/`, update the matching excerpt
  in the chapter (or the chapter's claim that "the logic is untouched"
  becomes false). When adding chapters for future projects, follow the same
  pattern: walk through the load-bearing code with its paper references, in
  the chapter, not just in the code tour table.
- **Figures are hand-written SVGs** in `book/src/figs/` (no generated/stock
  images). Conventions: light `#fcfcfa` rounded background so they read in
  both mdBook themes; restrained palette (slate text `#1e293b`, blue
  `#3b82f6` accents, green `#16a34a` for "pull/positive/residual", red
  `#dc2626` for "push/negative/cut"); system sans font; shared
  arrow `<marker>` defs. Embed with `<figure class="figure">` +
  `<img>` + `<figcaption>` (styles in `custom.css`).
  Existing figures: `gpt-block.svg`, `causal-mask.svg`,
  `temperature-topk.svg` (ch01), `bpe-merge-loop.svg` (ch02),
  `bert-masking.svg` (ch03), `simcse-matrix.svg` (ch04).

### Writing style (important — the book must not read like AI output)

The book was edited with the humanizer / no-ai-slop rules. Keep it that way
when adding chapters:

- **State facts, don't stage them.** No "not X but Y" contrasts unless the
  negative half corrects a real belief. No one-line dramatic closers. No
  "Let's dive in" run-ups.
- **No emojis in headings or prose.** No decorative bold; bold only for terms
  a reader scans for. Headings in sentence case.
- **No em dashes as a rhythm device.** Use commas, colons, parentheses, or
  rewrite. Numeric ranges (`1.5–1.8`, `80/10/10`) are fine.
- **Vary sentence length; write like a build log.** First person is fine
  ("we measured", "the naive loop took 5 hours"). Admit trade-offs and
  failed runs where they exist.
- **Every number must be measured.** Results come from real runs on the RTX
  3050; record new numbers in `book/src/results.md` and in the chapter.
  Never inflate or round up to sound better.
- Keep code blocks, commands, paths, and link targets exactly correct; the
  code tour tables must match real files.

### Be more explanatory — this is a book

The book's job is to teach, not to summarize. A reader who knows the theory
should be able to follow a chapter from their armchair and then reproduce
the build at their desk. Concretely:

- **Explain concepts on first use, even when a paper reference exists.** The
  citation points the reader deeper; it never replaces the explanation. A
  reader should not need to open the paper to follow the chapter.
- **Why before how.** Before showing a mechanism, say what problem it solves
  and what breaks without it (e.g. why `√d_k` scaling exists, why the
  80/10/10 masking mix, why τ = 0.05). Then show the implementation.
- **Walk through code, don't just link it.** When a chapter references a
  file, quote or paraphrase the load-bearing lines and explain them; a bare
  "see `model.py`" is not a book.
- **Use small worked examples.** Concrete micro-examples (a 3-merge BPE run,
  a 4-sentence batch similarity matrix, one token's journey through a
  forward pass) beat abstract statements. Add or extend an SVG figure when
  a picture explains it better than prose.
- **Refresh earlier concepts.** Readers arrive at chapter N having skimmed
  chapter N-1. When a concept returns, give a one-line reminder with a link
  back to the chapter that introduced it.
- **Define jargon the moment it appears**, then use the term consistently
  afterwards. No synonym cycling for technical terms.

## Key technical facts (use these, don't re-derive or invent)

- **Hardware:** RTX 3050 8 GB, PyTorch 2.x + CUDA 12.8. Everything trains on
  a single consumer GPU; CPU fallback works but slow.
- **Ch1 GPT:** ~10M params, Tiny Shakespeare (~1M chars), ~65-char vocab,
  pre-LN blocks, causal mask, target val loss 1.5–1.8 in <1h.
- **Ch2 BPE:** WikiText-2 (~11 MB), 5,000 vocab (3,983 merges). Naive
  full-rescan loop ≈ 5 h; distinct-words-with-multiplicities loop ≈ 12 min
  (~50× faster, identical vocab). Every training word carries a trailing
  space so tokens like `"the "` are learnable. Special token IDs: `<PAD>`=0,
  `<UNK>`=1, `<BOS>`=2, `<EOS>`=3 — chapters 3/4 rely on these positions
  (0 = pad, 2 = sentence start, 3 = separator).
- **Ch3 BERT-style:** 4.8M params (6 layers, 256 hidden, 8 heads), WikiText-2,
  MLM 80/10/10 + NSP, 10k iters ≈ 12 min, MLM loss 4.2 → ~1.4 (best val 2.85).
- **Ch4 SimCSE:** unsupervised (dropout-only augmentation), τ = 0.05, batch
  64, 100k Wikipedia sentences, ~26 s/epoch. STS-B dev Spearman: baseline
  16.6 → 34.7 at 3 epochs (+18.1; 5 epochs overfits at 33.5). Paper reference
  (BERT-base, 110M params, 10⁶ sentences): 82.5. Dev split is the reportable
  number (GLUE test labels are hidden).
- `alignment_and_uniformity` (Wang & Isola 2020) is implemented in
  `project3b_simcse/model.py`.

## Gotchas

- **Keep the repo off cloud-sync folders.** It lived in OneDrive and the
  `.git` folder was lost to a sync conflict; the repo has since moved to
  `C:\Users\abuba\projects\llm-from-scratch` on the local disk. For training
  runs, always work from a local-disk path — checkpoint writes fight sync
  clients and cause lock conflicts (see the warning in `book/src/setup.md`).
- `data/` and `checkpoints/` are gitignored; don't commit them.
- `book/build` output directory is build output; don't commit it.
- After editing anything under `book/`, run `mdbook build book` (or push and
  watch the Pages workflow) to catch broken links/paths
  (`create-missing = false` makes mdbook fail on missing chapter files).
