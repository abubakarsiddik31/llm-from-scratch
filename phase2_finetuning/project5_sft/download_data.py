# ruff: noqa
"""
Download the SFT instruction dataset for Project 5: Stanford Alpaca.

Alpaca (Taori et al., 2023) is 52,002 instruction/input/output triples
generated with Self-Instruct (Wang et al., 2023) from 175 seed tasks,
then used to fine-tune LLaMA-7B. It is the canonical public SFT set and
small enough (a ~23 MB JSON) that fine-tuning a 126M model on a fraction
of it is practical on one RTX 3050.

    --size full    all 52,002 examples  -> alpaca_data.json (the real run)
    --size small   a seeded SMOKE_EXAMPLES-example sample
                   -> alpaca_small.json (pipeline smoke tests)

Files land in data/alpaca/ as JSON lists of
{"instruction": str, "input": str, "output": str}.

SECURITY:
---------
Downloads are restricted to an allow-list of HTTPS hosts (same SSRF guard
as project4_pretrain/download_data.py). The file is JSON - no archive
extraction, so no directory-escape surface.
"""

import argparse
import json
import pathlib
import random
import urllib.parse
import urllib.request

import config

# Only these hosts may be downloaded from (prevents SSRF via CLI-supplied URLs)
ALLOWED_HOSTS = {
    "raw.githubusercontent.com",
}

DATASET = {
    "url": "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json",
    "file_name": "alpaca_data.json",
    "size_mb": 23,
}


def validate_download_url(url: str) -> None:
    """
    Ensure a URL points at an allow-listed HTTPS host before downloading.

    Same guard as project4_pretrain/download_data.py: an attacker-supplied
    URL must not be able to reach internal hosts or non-HTTPS endpoints.
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS:
        raise ValueError(
            f"Refusing to download from {url!r}. "
            f"Allowed hosts: {sorted(ALLOWED_HOSTS)}"
        )


def download_with_progress(url: str, destination: pathlib.Path) -> None:
    """Stream the JSON to disk with a progress readout."""
    validate_download_url(url)
    print(f"Downloading {url}")
    print(f"  → {destination}")

    def progress(block_num, block_size, total_size):
        if total_size > 0:
            percent = min(100.0, block_num * block_size * 100 / total_size)
            mb = block_num * block_size / (1024 * 1024)
            print(f"\r  {percent:5.1f}% ({mb:8.1f} MB)", end="")

    # GitHub raw serves browsers; identify ourselves rather than presenting
    # the default "Python-urllib" user-agent.
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", "llm-from-scratch/0.1 (sft data download)")]
    urllib.request.install_opener(opener)

    destination.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, destination, reporthook=progress)
    print(" ✓")


def load_examples(path: pathlib.Path) -> list:
    """
    Load and validate the Alpaca JSON.

    Every entry must be a dict; missing instruction/input keys become ""
    and entries without a non-empty output are dropped (they would carry
    no supervised signal). Returns the cleaned list.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path} does not contain a non-empty JSON list")

    cleaned = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"entry {i} of {path} is not an object")
        instruction = str(item.get("instruction", ""))
        user_input = str(item.get("input", ""))
        output = str(item.get("output", "")).strip()
        if output:
            cleaned.append({"instruction": instruction, "input": user_input, "output": output})
    dropped = len(data) - len(cleaned)
    if dropped:
        print(f"Dropped {dropped} entries with empty output")
    return cleaned


def main():
    parser = argparse.ArgumentParser(
        description="Download the Stanford Alpaca SFT dataset"
    )
    parser.add_argument(
        "--size", choices=["small", "full"], default="full",
        help="small = seeded 2,000-example sample (smoke tests), "
             "full = all ~52k examples (the real fine-tuning run)",
    )
    args = parser.parse_args()

    raw_dir = pathlib.Path(config.RAW_DATA_DIR)
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Dataset:   Stanford Alpaca (instruction/input/output triples)")
    print(f"Target:    {raw_dir}")
    print("=" * 60)

    full_path = raw_dir / DATASET["file_name"]
    if not full_path.is_file():
        download_with_progress(DATASET["url"], full_path)
    else:
        print(f"\nArchive already downloaded: {full_path}")

    examples = load_examples(full_path)
    print(f"Loaded {len(examples):,} usable examples from {full_path.name}")

    if args.size == "small":
        # Seeded sample: same subset every run, so smoke-test numbers are
        # reproducible across machines and chapters.
        rng = random.Random(config.SEED)
        sample = rng.sample(examples, min(config.SMOKE_EXAMPLES, len(examples)))
        small_path = raw_dir / "alpaca_small.json"
        small_path.write_text(
            json.dumps(sample, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Wrote {len(sample):,} sampled examples to {small_path.name}")

    print("\nDone. Next steps:")
    print("  1. uv run python phase2_finetuning/project5_sft/prepare_data.py"
          + ("" if args.size == "full" else " --size small"))
    print("  2. uv run python phase2_finetuning/project5_sft/test_model.py")
    print("  3. uv run python phase2_finetuning/project5_sft/train.py")


if __name__ == "__main__":
    main()
