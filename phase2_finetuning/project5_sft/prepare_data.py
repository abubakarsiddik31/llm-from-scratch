# ruff: noqa
"""
Encode the Alpaca dataset into padded token/label arrays for SFT.

Pipeline position: download_data.py → THIS → train.py

Pre-training (project 4) packed one long token stream into .bin files and
sampled random windows: no example boundaries existed. SFT data is a list
of VARIABLE-LENGTH examples where position matters (prompt vs response),
so each example is encoded separately with a response mask and padded to
MAX_SEQ_LEN. train.py loads the two aligned arrays and shifts them into
(x, y) batches at step time.

OUTPUT FILES (data/project5/):
------------------------------
    sft_train.npy        uint16 (N_train, MAX_SEQ_LEN) token ids
    sft_train_labels.npy int16   same shape; response/EOS positions carry
                         the token to teach, everything else -1
    sft_val.npy          uint16 (VAL_EXAMPLES, MAX_SEQ_LEN)
    sft_val_labels.npy   int16   aligned labels

int16 covers -1..16,383, which fits both the ignore index and our
16,384-token vocabulary.

Encoding goes through template.encode_labeled(), which owns the
trailing-space junction (see template.py's module docstring): the prompt
half and the response half are encoded word-by-word and concatenated, so
the response start is known EXACTLY without searching decoded text.

EXAMPLES THAT CANNOT BE TAUGHT ARE SKIPPED AND COUNTED:
-------------------------------------------------------
- empty instruction/output (download_data.py already filters the latter)
- prompts too long to leave room for a response + <EOS> within
  MAX_SEQ_LEN (responses are truncated instead when it is only the
  response that overflows - losing the end of an answer is acceptable,
  losing the question is not)
"""

import argparse
import json
import pathlib

import numpy as np
from tqdm import tqdm

import config
from template import encode_labeled, format_prompt
from tokenizer import BPETokenizer


def load_split(raw_dir: pathlib.Path, size: str) -> list:
    """Load the JSON split download_data.py wrote."""
    name = "alpaca_small.json" if size == "small" else "alpaca_data.json"
    path = raw_dir / name
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run download_data.py "
            f"({'--size small' if size == 'small' else ''}) first."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    print(f"Loaded {len(data):,} examples from {path.name}")
    return data


def encode_split(tokenizer: BPETokenizer, examples: list, desc: str):
    """
    Encode every example to (tokens, labels) rows, skipping unteachable
    ones. Returns stacked uint16/int16 arrays plus a stats dict.
    """
    n = len(examples)
    token_rows, label_rows = [], []
    stats = {
        "skipped_empty": 0,
        "skipped_too_long": 0,
        "prompt_tokens": 0,
        "response_tokens": 0,
        "unk_tokens": 0,
        "truncated": 0,
    }

    unk = config.SPECIAL_TOKENS["<UNK>"]
    for ex in tqdm(examples, desc=desc, unit="ex"):
        instruction = str(ex.get("instruction", ""))
        user_input = str(ex.get("input", ""))
        output = str(ex.get("output", "")).strip()
        if not output:
            stats["skipped_empty"] += 1
            continue
        try:
            tokens, labels = encode_labeled(
                tokenizer, instruction, user_input, output
            )
        except ValueError:
            stats["skipped_too_long"] += 1
            continue

        # response_start is the first label != -1; -1 for <BOS>
        response_start = next(i for i, l in enumerate(labels) if l != -1)
        stats["prompt_tokens"] += response_start - 1  # minus <BOS>
        row_tokens = tokens[response_start:]
        stats["response_tokens"] += sum(1 for t in row_tokens if t != 0)
        stats["unk_tokens"] += tokens.count(unk)

        token_rows.append(np.array(tokens, dtype=np.uint16))
        label_rows.append(np.array(labels, dtype=np.int16))

    print(f"  encoded: {len(token_rows):,}")
    print(f"  skipped (empty output): {stats['skipped_empty']:,}")
    print(f"  skipped (prompt too long): {stats['skipped_too_long']:,}")
    return np.stack(token_rows), np.stack(label_rows), stats


def main():
    parser = argparse.ArgumentParser(
        description="Encode Alpaca into padded SFT token/label arrays"
    )
    parser.add_argument("--size", choices=["small", "full"], default="full",
                        help="which download_data.py artifact to encode")
    args = parser.parse_args()

    config.validate_config()
    tokenizer = BPETokenizer.load()

    raw_dir = pathlib.Path(config.RAW_DATA_DIR)
    out_dir = pathlib.Path(config.TOKENIZED_DATA_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    examples = load_split(raw_dir, args.size)

    # Seeded shuffle, then hold out the LAST VAL_EXAMPLES for validation.
    # Same seed -> same split every run, so val losses are comparable
    # across experiments (the response-only vs full-loss comparison in the
    # chapter depends on this).
    rng = np.random.default_rng(config.SEED)
    perm = rng.permutation(len(examples))
    n_val = min(config.VAL_EXAMPLES, len(examples) // 10)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train_examples = [examples[i] for i in train_idx]
    val_examples = [examples[i] for i in val_idx]
    print(f"Split: {len(train_examples):,} train / {len(val_examples):,} val "
          f"(seed {config.SEED})")

    print("\nExample template (first train example):")
    print("-" * 60)
    print(format_prompt(train_examples[0].get("instruction", ""),
                        train_examples[0].get("input", "")))
    print("<response below>")
    print(train_examples[0].get("output", "")[:200])
    print("-" * 60)

    train_tokens, train_labels, train_stats = encode_split(
        tokenizer, train_examples, "Encoding train")
    val_tokens, val_labels, _ = encode_split(
        tokenizer, val_examples, "Encoding val")

    for name, arr in [("sft_train", train_tokens), ("sft_train_labels", train_labels),
                      ("sft_val", val_tokens), ("sft_val_labels", val_labels)]:
        path = out_dir / f"{name}.npy"
        np.save(path, arr)
        print(f"Saved {path}  shape={arr.shape}  {path.stat().st_size / 1e6:.1f} MB")

    n_rows = len(train_tokens)
    print("\nCorpus statistics (train split):")
    print(f"  avg prompt tokens:     {train_stats['prompt_tokens'] / n_rows:.1f}")
    print(f"  avg response tokens:   {train_stats['response_tokens'] / n_rows:.1f}")
    print(f"  <UNK> tokens:          {train_stats['unk_tokens']:,} "
          f"({train_stats['unk_tokens'] / (n_rows * config.MAX_SEQ_LEN):.3%} of all positions)")
    print("\nDone. Next steps:")
    print("  1. uv run python phase2_finetuning/project5_sft/test_model.py")
    print("  2. uv run python phase2_finetuning/project5_sft/train.py")


if __name__ == "__main__":
    main()
