# CLAUDE.md

All guidance for working in this repository lives in [AGENTS.md](AGENTS.md).

It covers: repository purpose and status, run/build commands, code
conventions (paper-first docstrings, config validation, project structure),
book (mdBook) conventions including the SVG figure and writing-style rules,
measured results per chapter, and known gotchas (OneDrive + `.git`).

Quick reference:

```bash
# code
uv sync --group all
uv run python phase1_foundation/project1_minimal_gpt/train.py

# book
mdbook serve book   # or: mdbook build book
```
