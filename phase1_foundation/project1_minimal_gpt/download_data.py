#!/usr/bin/env python3
"""
Download and prepare datasets for training.

Usage:
    python download_data.py

This will download:
- Shakespeare text (for Project 1)
- WikiText-103 (for Project 2 & 3)
"""

import os
import urllib.request
import zipfile
from pathlib import Path


def download_file(url, destination):
    """Download a file with progress indicator."""
    print(f"Downloading {url}")
    print(f"→ {destination}")

    def progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 / total_size) if total_size > 0 else 0
        mb_downloaded = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        print(f"\r  Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='')

    urllib.request.urlretrieve(url, destination, reporthook=progress)
    print(" ✓")


def main():
    # Create data directory
    data_dir = Path(__file__).parent.parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    print(f"Data directory: {data_dir}\n")

    # ============================================================================
    # Shakespeare text (for Project 1: Character-Level GPT)
    # ============================================================================

    shakespeare_path = data_dir / "shakespeare.txt"
    if not shakespeare_path.exists():
        print("1. Downloading Shakespeare text...")
        download_file(
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
            shakespeare_path
        )
        print(f"   Size: {shakespeare_path.stat().st_size / 1024:.1f} KB\n")
    else:
        print(f"1. Shakespeare text already exists ({shakespeare_path.stat().st_size / 1024:.1f} KB)\n")

    # ============================================================================
    # WikiText-103 (for Project 2 & 3)
    # ============================================================================

    # For now, we'll use the HuggingFace datasets library
    # This will be downloaded automatically in the project scripts
    print("2. WikiText-103 will be downloaded automatically using HuggingFace datasets")
    print("   when you run Project 2 or Project 3.\n")

    print("=" * 60)
    print("Data setup complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
