# ruff: noqa
"""
Download the pre-training corpus for Project 4.

Downloads the WikiText corpus (Merity et al., 2016, "Pointer Sentinel
Mixture Models") in its raw form from the canonical Zenodo mirror:

    --size small   WikiText-2-raw   (~12 MB text)   smoke tests, CPU runs
    --size full    WikiText-103-raw (~515 MB text)  the real pre-training run

Files land in data/wikitext/ as wiki.train.raw / wiki.valid.raw /
wiki.test.raw (the split files tokenizers and training consume).

SECURITY:
---------
Downloads are restricted to an allow-list of HTTPS hosts (same SSRF guard
as project2_tokenizer/download_data.py), and archives are extracted
without directory-escape protection gaps (each member is checked against
the extraction directory before writing).
"""

import argparse
import pathlib
import urllib.parse
import urllib.request
import zipfile

import config

# Only these hosts may be downloaded from (prevents SSRF via CLI-supplied URLs)
ALLOWED_HOSTS = {
    "wikitext.smerity.com",
    "zenodo.org",
}

CORPORA = {
    "small": {
        "url": "https://wikitext.smerity.com/wikitext-2-raw-v1.zip",
        "zip_name": "wikitext-2-raw-v1.zip",
        "inner_dir": "wikitext-2-raw",
        "text_mb": 12,
    },
    "full": {
        "url": "https://wikitext.smerity.com/wikitext-103-raw-v1.zip",
        "zip_name": "wikitext-103-raw-v1.zip",
        "inner_dir": "wikitext-103-raw",
        "text_mb": 515,
    },
}


def validate_download_url(url: str) -> None:
    """
    Ensure a URL points at an allow-listed HTTPS host before downloading.

    Same guard as project2_tokenizer/download_data.py: an attacker-supplied
    URL must not be able to reach internal hosts or non-HTTPS endpoints.
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS:
        raise ValueError(
            f"Refusing to download from {url!r}. "
            f"Allowed hosts: {sorted(ALLOWED_HOSTS)}"
        )


def download_with_progress(url: str, destination: pathlib.Path) -> None:
    """Stream the archive to disk with a progress readout."""
    validate_download_url(url)
    print(f"Downloading {url}")
    print(f"  → {destination}")

    def progress(block_num, block_size, total_size):
        if total_size > 0:
            percent = min(100.0, block_num * block_size * 100 / total_size)
            mb = block_num * block_size / (1024 * 1024)
            print(f"\r  {percent:5.1f}% ({mb:8.1f} MB)", end="")

    # The mirror 403s the default "Python-urllib" user-agent; identify
    # ourselves instead. Installed opener applies to urlretrieve below.
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", "llm-from-scratch/0.1 (corpus download)")]
    urllib.request.install_opener(opener)

    destination.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, destination, reporthook=progress)
    print(" ✓")


def safe_extract(zip_path: pathlib.Path, target_dir: pathlib.Path) -> None:
    """
    Extract a zip archive, refusing members that would escape target_dir.

    Each member path is resolved and required to stay inside the extraction
    directory, so a malicious archive cannot write outside it.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    target_resolved = target_dir.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            member_path = (target_dir / member).resolve()
            if not member_path.is_relative_to(target_resolved):
                raise ValueError(
                    f"Archive member {member!r} would extract outside "
                    f"{target_resolved}; refusing."
                )
        zf.extractall(target_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Download the WikiText pre-training corpus"
    )
    parser.add_argument(
        "--size", choices=["small", "full"], default="full",
        help="small = WikiText-2-raw (smoke tests), full = WikiText-103-raw "
             "(the real pre-training corpus)",
    )
    args = parser.parse_args()

    corpus = CORPORA[args.size]
    raw_dir = pathlib.Path(config.RAW_DATA_DIR)
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Corpus:      WikiText {'2-raw' if args.size == 'small' else '103-raw'}")
    print(f"Target dir:  {raw_dir}")
    print("=" * 60)

    # The raw split files the downstream scripts expect: wiki.train.raw etc.
    split_files = [
        raw_dir / "wiki.train.raw",
        raw_dir / "wiki.valid.raw",
        raw_dir / "wiki.test.raw",
    ]
    if all(f.is_file() for f in split_files):
        print("\nSplit files already present, nothing to do:")
        for f in split_files:
            print(f"  {f.name}: {f.stat().st_size / (1024 * 1024):.1f} MB")
        return

    zip_path = raw_dir / corpus["zip_name"]
    if not zip_path.is_file():
        download_with_progress(corpus["url"], zip_path)
    else:
        print(f"\nArchive already downloaded: {zip_path}")

    extract_dir = raw_dir / "_archives"
    print(f"Extracting...")
    safe_extract(zip_path, extract_dir)

    # Flatten: move the split files up to raw_dir
    inner = extract_dir / corpus["inner_dir"]
    for f in split_files:
        src = inner / f.name
        if not src.is_file():
            raise FileNotFoundError(f"Expected split file missing after extract: {src}")
        f.write_bytes(src.read_bytes())
        print(f"  {f.name}: {f.stat().st_size / (1024 * 1024):.1f} MB")

    print("\nDone. Next steps:")
    print("  1. uv run python phase1_foundation/project4_pretrain/train_tokenizer.py")
    print("  2. uv run python phase1_foundation/project4_pretrain/prepare_data.py")
    print("  3. uv run python phase1_foundation/project4_pretrain/train.py")


if __name__ == "__main__":
    main()
