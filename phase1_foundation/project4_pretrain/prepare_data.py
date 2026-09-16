# ruff: noqa
"""
Encode the WikiText corpus into uint16 token files for training.

Pipeline position: train_tokenizer.py → THIS → train.py

Reads the raw split files (wiki.train.raw / wiki.valid.raw /
wiki.test.raw), splits them into documents at the " = Heading = " lines
the WikiText format uses, wraps every document in <BOS> ... <EOS>, and
writes one flat uint16 array per split:

    data/project4/train.bin
    data/project4/val.bin
    data/project4/test.bin

Training then samples random 512-token windows from these files (the
nanoGPT-style memmap layout): no per-step tokenization, no padding, and a
~515 MB corpus becomes a ~230 MB array with 16,384-token vocab.

WHY uint16: our vocabulary is 16,384 tokens, which fits in 16 bits
(config.validate_config asserts this). GPT-2's 50,257 would NOT fit and
would need uint32.

WHY <BOS>/<EOS> AT DOCUMENT BOUNDARIES: the tokenizer's word normalization
collapses newlines, so without explicit boundary tokens the model would see
one endless document and never learn where articles start and end.
"""

import argparse
import pathlib

import numpy as np
from tqdm import tqdm

import config
from tokenizer import BPETokenizer

PAD = config.SPECIAL_TOKENS["<PAD>"]
BOS = config.SPECIAL_TOKENS["<BOS>"]
EOS = config.SPECIAL_TOKENS["<EOS>"]


def split_documents(text: str):
    """
    Yield WikiText documents.

    WikiText marks structure as heading lines: " = Title = ", " = = Section
    = = ", etc. A line starting with " = " (exactly one leading =) begins a
    new article; blank lines are separators. Everything else is paragraph
    text. The heading line itself is kept as document content.
    """
    doc_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if line.startswith(" = ") and doc_lines:
            yield " ".join(doc_lines)
            doc_lines = [stripped]
        else:
            doc_lines.append(stripped)
    if doc_lines:
        yield " ".join(doc_lines)


def encode_split(tokenizer: BPETokenizer, text: str, split_name: str) -> np.ndarray:
    """Encode one split into a uint16 token array with document boundaries."""
    ids: list[int] = []
    docs = list(split_documents(text))
    for doc in tqdm(docs, desc=f"encoding {split_name}", unit="doc"):
        ids.append(BOS)
        ids.extend(tokenizer.encode(doc))
        ids.append(EOS)

    array = np.asarray(ids, dtype=np.uint16)
    unk_rate = ids.count(config.SPECIAL_TOKENS["<UNK>"]) / max(1, len(ids))
    print(f"  {split_name}: {len(docs):,} documents, {len(ids):,} tokens, "
          f"{unk_rate * 100:.3f}% <UNK>")
    return array


def main():
    parser = argparse.ArgumentParser(
        description="Encode WikiText splits to uint16 .bin token files"
    )
    parser.add_argument(
        "--max_docs", type=int, default=0,
        help="Debug: cap documents per split (0 = no limit)",
    )
    args = parser.parse_args()

    tokenizer = BPETokenizer.load()

    out_dir = pathlib.Path(config.TOKENIZED_DATA_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ENCODING CORPUS")
    print("=" * 70)

    splits = {
        "train": "wiki.train.raw",
        "val": "wiki.valid.raw",
        "test": "wiki.test.raw",
    }
    for split_name, filename in splits.items():
        src = pathlib.Path(config.RAW_DATA_DIR) / filename
        if not src.is_file():
            raise FileNotFoundError(f"Corpus file missing: {src}")

        text = src.read_text(encoding="utf-8")
        if args.max_docs > 0:
            text = " ".join(list(split_documents(text))[: args.max_docs])

        array = encode_split(tokenizer, text, split_name)
        out_file = out_dir / f"{split_name}.bin"
        array.tofile(out_file)
        print(f"  wrote {out_file} ({out_file.stat().st_size / (1024 * 1024):.1f} MB)")

    print("\nNext: uv run python phase1_foundation/project4_pretrain/train.py")


if __name__ == "__main__":
    main()
