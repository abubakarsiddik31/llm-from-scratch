# ruff: noqa
"""
Data Download Script for Unsupervised SimCSE

Prepares the two data sources needed for unsupervised SimCSE:

1. SENTENCE CORPUS (training):
   The paper samples 10^6 sentences from English Wikipedia
   ("We randomly sample 10^6 sentences from English Wikipedia and
   fine-tune BERT-base with learning rate = 3e-5, N = 64").
   We reuse the WikiText-2 corpus already downloaded for Project 3,
   split it into sentences, filter by length, and save a flat
   one-sentence-per-line file. Length filtering matters: the paper
   trains with max sequence length 32 tokens, so very long sentences
   would be truncated into unnatural snippets.

2. STS-B (evaluation):
   Semantic Textual Similarity Benchmark (Cer et al., 2017) - 8,628
   sentence pairs with human similarity scores 0-5. Downloaded from
   Hugging Face (nyu-mll/glue, config "stsb"). Standard SimCSE
   evaluation: Spearman correlation between human scores and cosine
   similarity of embeddings. The paper reports 82.5 dev Spearman for
   BERT-base unsupervised SimCSE.

USAGE:
------
uv run python phase1_foundation/project3b_simcse/download_data.py
"""

import argparse
import os
import re
from pathlib import Path

import config

# Repo data directory
DATA_DIR = Path(config.ROOT_DIR) / "data"


def extract_sentences(
    corpus_path: Path,
    output_path: Path,
    max_sentences: int,
    min_chars: int,
    max_chars: int,
):
    """
    Split a text corpus into sentences and save one per line.

    IMPLEMENTATION:
    ---------------
    - Split on sentence-ending punctuation (. ! ?) followed by whitespace
      and an uppercase letter/quote/digit (avoids splitting on
      abbreviations like "e.g." more than a naive split would).
    - Collapse whitespace (WikiText has line breaks mid-sentence).
    - Keep sentences in [min_chars, max_chars]: short ones are noise
      (headers, fragments); long ones would truncate at 32 tokens.

    Args:
        corpus_path: Path to the raw corpus (wikitext_train.txt)
        output_path: Where to write the one-sentence-per-line corpus
        max_sentences: Cap on number of sentences (paper uses 10^6)
        min_chars: Minimum sentence length in characters
        max_chars: Maximum sentence length in characters

    Returns:
        Number of sentences written
    """
    print(f"Reading corpus: {corpus_path}")
    text = corpus_path.read_text(encoding="utf-8")

    # WikiText-2 lines that start with "= = Title = =" are headers; drop them
    lines = [
        line for line in text.splitlines() if line.strip() and not line.startswith("= =")
    ]
    text = "\n".join(lines)

    print("Splitting into sentences...")
    # Sentence boundary: . ! ? or " followed by whitespace and capital/quote/digit
    parts = re.split(r"(?<=[.!?\"\'])\s+(?=[A-Z\"\'0-9])", text)

    sentences = []
    seen = set()
    for part in parts:
        sentence = re.sub(r"\s+", " ", part).strip()
        if not (min_chars <= len(sentence) <= max_chars):
            continue
        # Require at least one alphabetic character and a space or two words
        if not re.search(r"[A-Za-z]", sentence):
            continue
        if sentence.count(" ") < 2:
            continue
        # Deduplicate: repeated sentences add no contrastive signal
        if sentence in seen:
            continue
        seen.add(sentence)
        sentences.append(sentence)
        if len(sentences) >= max_sentences:
            break

    print(f"✓ Extracted {len(sentences):,} unique sentences")

    output_path.write_text("\n".join(sentences), encoding="utf-8")
    print(f"✓ Saved to {output_path}")

    return len(sentences)


def download_sts_b():
    """
    Download the STS-B validation/test splits from Hugging Face.

    PAPER REFERENCE:
    ----------------
    Evaluation follows Gao et al. (2021): "we evaluate on several
    semantic textual similarity (STS) tasks... following previous work
    we report Spearman's correlation". We use the dev split during
    training-time evaluation and report both dev and test at the end.

    The dataset is cached by the Hugging Face datasets library; nothing
    is written into the repository's data/ directory.
    """
    from datasets import load_dataset

    for split in ["validation", "test"]:
        dataset = load_dataset("nyu-mll/glue", "stsb", split=split)
        print(
            f"✓ STS-B {split}: {len(dataset):,} pairs "
            f"(cache: {dataset.cache_files[0]['filename'] if dataset.cache_files else 'in-memory'})"
        )


def main():
    parser = argparse.ArgumentParser(description="Download data for unsupervised SimCSE")
    parser.add_argument(
        "--corpus",
        type=str,
        default=str(DATA_DIR / "wikitext_train.txt"),
        help="Source corpus to extract sentences from",
    )
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=config.MAX_SENTENCES,
        help=f"Maximum number of sentences (default: {config.MAX_SENTENCES:,})",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("SIMCSE DATA PREPARATION")
    print("=" * 60)

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        print(f"✗ Corpus not found at {corpus_path}")
        print("  Run Project 3's downloader first:")
        print(
            "  uv run python phase1_foundation/"
            "project3_contextual_embeddings/download_data.py --size small"
        )
        raise SystemExit(1)

    # 1. Sentence corpus
    sentence_path = DATA_DIR / "simcse_sentences.txt"
    extract_sentences(
        corpus_path=corpus_path,
        output_path=sentence_path,
        max_sentences=args.max_sentences,
        min_chars=config.MIN_SENTENCE_CHARS,
        max_chars=config.MAX_SENTENCE_CHARS,
    )

    # 2. STS-B evaluation data
    print("\nDownloading STS-B (glue/stsb)...")
    download_sts_b()

    print("\n" + "=" * 60)
    print("✓ Data preparation complete")
    print("=" * 60)
    print(f"  Sentences: {sentence_path}")
    print("  STS-B: cached by the Hugging Face datasets library")


if __name__ == "__main__":
    main()
