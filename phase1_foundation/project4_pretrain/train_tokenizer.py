# ruff: noqa
"""
Train the Project 4 BPE tokenizer on WikiText.

Pipeline position: download_data.py → THIS → prepare_data.py → train.py

We retrain the chapter 2 tokenizer instead of reusing its 5,000-token
vocabulary: 5k fragments WikiText-103 into longer sequences and slows the
125M-parameter model down. Target: config.VOCAB_SIZE (16,384) tokens
learned from a seeded sample of wiki.train.raw.

PAPER REFERENCE:
- "Byte-Pair Encoding of Subword Units" (Sennrich, Haddow, Birch, 2016)
- "Language Models are Unsupervised Multitask Learners" (GPT-2, 2019)

The trainer itself lives in tokenizer.py. It uses incremental pair
statistics (heap + lazy deletion) instead of the chapter 2 rescan loop;
on this corpus that is the difference between minutes and most of a day.
test_model.py verifies the incremental trainer against a naive reference.

USAGE:
------
uv run python phase1_foundation/project4_pretrain/train_tokenizer.py
uv run python phase1_foundation/project4_pretrain/train_tokenizer.py --sample_mb 50
"""

import argparse
import pathlib
import random
import time

import config
from tokenizer import BPETokenizer, get_tokenizer_stats


def sample_corpus(path: pathlib.Path, sample_mb: float, seed: int) -> str:
    """
    Take a seeded random line sample of the corpus.

    Reading 515 MB into memory just to pick 100 MB is wasteful, and taking
    the FIRST 100 MB would bias the vocabulary toward whatever articles
    happen to start the file. Instead: stream lines, keep each with
    probability sample_mb/filesize (fixed RNG seed = reproducible), stop
    once the target is reached.
    """
    file_bytes = path.stat().st_size
    target_bytes = min(int(sample_mb * 1024 * 1024), file_bytes)
    keep_probability = min(1.0, target_bytes / file_bytes)

    rng = random.Random(seed)
    kept: list[str] = []
    kept_bytes = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if kept_bytes >= target_bytes:
                break
            if rng.random() < keep_probability:
                kept.append(line)
                kept_bytes += len(line.encode("utf-8"))

    print(f"Sampled {kept_bytes / (1024 * 1024):.1f} MB of "
          f"{file_bytes / (1024 * 1024):.1f} MB "
          f"(p={keep_probability:.3f}, seed={seed})")
    return "".join(kept)


def main():
    parser = argparse.ArgumentParser(
        description="Train the project 4 BPE tokenizer"
    )
    parser.add_argument(
        "--vocab_size", type=int, default=config.VOCAB_SIZE,
        help=f"Total vocabulary size (default: {config.VOCAB_SIZE})",
    )
    parser.add_argument(
        "--min_freq", type=int, default=config.MIN_FREQUENCY,
        help=f"Minimum merge frequency (default: {config.MIN_FREQUENCY})",
    )
    parser.add_argument(
        "--sample_mb", type=float, default=config.TOKENIZER_SAMPLE_MB,
        help="MB of wiki.train.tokens to train on (default: "
             f"{config.TOKENIZER_SAMPLE_MB})",
    )
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    train_file = pathlib.Path(config.RAW_DATA_DIR) / "wiki.train.raw"
    if not train_file.is_file():
        raise FileNotFoundError(
            f"Corpus not found at {train_file}. "
            f"Run download_data.py first."
        )

    print("=" * 70)
    print("BPE TOKENIZER TRAINING (project 4)")
    print("=" * 70)
    print(f"Corpus:       {train_file}")
    print(f"Vocab target: {args.vocab_size:,}")
    print(f"Min freq:     {args.min_freq}")
    print(f"Sample:       {args.sample_mb} MB (seed {args.seed})")
    print("=" * 70)
    print()

    text = sample_corpus(train_file, args.sample_mb, args.seed)

    tokenizer = BPETokenizer()
    start = time.time()
    tokenizer.train(text, vocab_size=args.vocab_size, min_frequency=args.min_freq)
    elapsed = time.time() - start
    print(f"\nTokenizer trained in {elapsed:.1f}s")

    stats = get_tokenizer_stats(tokenizer)
    print("\n" + "=" * 70)
    print("Tokenizer statistics")
    print("=" * 70)
    print(f"Vocabulary size:      {stats['vocab_size']:,}")
    print(f"Merge rules:          {stats['num_merges']:,}")
    print(f"Avg token length:     {stats['avg_token_length']:.2f} chars")
    print(f"Max token length:     {stats['max_token_length']} chars")

    # Round-trip check on held-out-ish text before saving
    sample = ("The earliest known appearance of the name "
              "Alexander of Macedon appears in a 4th-century manuscript.")
    ids = tokenizer.encode(sample)
    recovered = tokenizer.decode(ids)
    print("\nRound-trip check:")
    print(f"  original: {sample!r}")
    print(f"  tokens:   {len(ids)} for {len(sample)} chars "
          f"({len(sample) / len(ids):.1f} chars/token)")
    assert recovered == sample, f"Round-trip failed: {recovered!r}"
    print("  round-trip OK")

    tokenizer.save()
    print(f"\nNext: uv run python phase1_foundation/project4_pretrain/prepare_data.py")


if __name__ == "__main__":
    main()
