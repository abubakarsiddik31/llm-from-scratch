#!/usr/bin/env python3
"""
Wikipedia Data Downloader for BPE Training

This script downloads and processes Wikipedia dumps for training
the Byte-Pair Encoding tokenizer.

Wikipedia is ideal for BPE training because:
1. Large and diverse vocabulary (~5M articles)
2. Well-formed text with good grammar
3. Multiple domains (science, arts, history, etc.)
4. Standard benchmark for language models

USAGE:
------
python download_data.py [--size SIZE] [--output OUTPUT]

EXAMPLES:
---------
# Download small sample (100MB - good for testing)
python download_data.py --size small

# Download medium dataset (1GB - good for training)
python download_data.py --size medium

# Download large dataset (10GB - production)
python download_data.py --size large

# Specify custom output path
python download_data.py --size medium --output /path/to/corpus.txt
"""

import argparse
import gzip
import os
import re
import sys
from pathlib import Path
from typing import Optional

# Try to import requests, install if not available
try:
    import requests
except ImportError:
    print("Installing required package: requests")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
    import requests

# Try to import datasets library (alternative method)
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


# =============================================================================
# CONFIGURATION
# =============================================================================

# Wikipedia dump URLs (latest dumps as of 2024)
WIKIPEDIA_DUMPS = {
    "en": {
        "url": "https://dumps.wikimedia.org/enwiki/latest/",
        "files": {
            "small": "enwiki-latest-abstract.xml.gz",  # ~500MB
        }
    }
}

# Size presets (in characters, approximately)
SIZE_PRESETS = {
    "small": 100 * 1024 * 1024,      # 100 MB
    "medium": 1024 * 1024 * 1024,    # 1 GB
    "large": 10 * 1024 * 1024 * 1024  # 10 GB
}


# =============================================================================
# DOWNLOAD HELPERS
# =============================================================================


def download_with_progress(url: str, output_path: str) -> int:
    """
    Download file with progress bar.

    Args:
        url: URL to download from
        output_path: Where to save the file

    Returns:
        Number of bytes downloaded

    Raises:
        RuntimeError: If download fails or returns empty file
    """
    print(f"Downloading from: {url}")
    print(f"Saving to: {output_path}")
    print()

    try:
        response = requests.get(url, stream=True, timeout=30, allow_redirects=True)
        response.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to download from {url}: {e}")

    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192
    downloaded = 0

    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)

                # Progress bar
                if total_size > 0:
                    percent = downloaded / total_size * 100
                    mb_downloaded = downloaded / 1024 / 1024
                    mb_total = total_size / 1024 / 1024
                    print(f"\rProgress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='')
                else:
                    mb_downloaded = downloaded / 1024 / 1024
                    print(f"\rDownloaded: {mb_downloaded:.1f} MB", end='')

    print()  # New line after progress bar

    # Verify download was successful
    if downloaded == 0:
        os.remove(output_path)
        raise RuntimeError(f"Download failed - empty file from {url}")

    print(f"Download complete! ({downloaded / 1024 / 1024:.2f} MB)")
    return downloaded


# =============================================================================
# WIKIPEDIA PROCESSING
# =============================================================================


def extract_text_from_wikipedia(xml_content: str, max_chars: Optional[int] = None) -> str:
    """
    Extract clean text from Wikipedia XML dump.

    From Wikipedia dump format:
    <mediawiki>
      <page>
        <title>Article Title</title>
        <revision>
          <text>Article content...</text>
        </revision>
      </page>
    </mediawiki>

    Args:
        xml_content: Raw Wikipedia XML content
        max_chars: Maximum characters to extract (None = all)

    Returns:
        Cleaned text content
    """
    # Extract text between <text> tags
    # Wikipedia uses these tags for article content
    text_pattern = r'<text[^>]*>(.*?)</text>'
    matches = re.findall(text_pattern, xml_content, re.DOTALL)

    # Clean each match
    cleaned_texts = []
    total_chars = 0

    for match in matches:
        # Decode HTML entities
        text = match.replace('&lt;', '<').replace('&gt;', '>')
        text = text.replace('&amp;', '&').replace('&quot;', '"')
        text = text.replace('&#39;', "'")

        # Remove wiki markup
        # Remove templates {{...}}
        text = re.sub(r'\{\{[^}]*\}\}', '', text)
        # Remove links [[...|...]] or [[...]]
        text = re.sub(r'\[\[([^]|]*\|)?([^]]*)\]\]', r'\2', text)
        # Remove headers ==...==
        text = re.sub(r'==+([^=]+)==+', r'\1', text)
        # Remove '''''', '''', '' formatting
        text = re.sub(r"'''''", '', text)
        text = re.sub(r"'''", '', text)
        text = re.sub(r"''", '', text)
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Remove references <ref>...</ref>
        text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
        # Remove file links
        text = re.sub(r'\[\[File:.*?\]\]', '', text, flags=re.DOTALL)
        # Remove image links
        text = re.sub(r'\[\[Image:.*?\]\]', '', text, flags=re.DOTALL)
        # Remove category links
        text = re.sub(r'\[\[Category:.*?\]\]', '', text, flags=re.DOTALL)

        # Clean whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        if text and len(text) > 50:  # Minimum length threshold
            cleaned_texts.append(text)
            total_chars += len(text)

            if max_chars and total_chars >= max_chars:
                break

    return '\n\n'.join(cleaned_texts)


def process_wikipedia_dump(
    dump_path: str,
    output_path: str,
    max_chars: Optional[int] = None
) -> None:
    """
    Process Wikipedia XML dump into clean text file.

    Args:
        dump_path: Path to downloaded Wikipedia dump (.gz)
        output_path: Where to save processed text
        max_chars: Maximum characters to extract
    """
    print(f"Processing Wikipedia dump: {dump_path}")
    print(f"Output: {output_path}")
    if max_chars:
        print(f"Max characters: {max_chars:,}")
    print()

    # Open gzip file
    opener = gzip.open if dump_path.endswith('.gz') else open

    with opener(dump_path, 'rt', encoding='utf-8', errors='ignore') as f:
        # Read in chunks to handle large files
        chunk_size = 1024 * 1024  # 1MB chunks
        buffer = ""
        extracted_texts = []
        total_chars = 0

        print("Extracting and cleaning text...")
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break

            buffer += chunk

            # Process complete articles when possible
            # Wikipedia articles end with </page>
            while '</page>' in buffer:
                # Split at first complete page
                parts = buffer.split('</page>', 1)
                page_content = parts[0] + '</page>'
                buffer = parts[1] if len(parts) > 1 else ""

                # Extract text from this page
                page_text = extract_text_from_wikipedia(page_content)
                if page_text:
                    extracted_texts.append(page_text)
                    total_chars += len(page_text)

                    # Show progress
                    mb_chars = total_chars / 1024 / 1024
                    print(f"\rExtracted: {mb_chars:.1f} MB, {len(extracted_texts):,} articles", end='')

                    if max_chars and total_chars >= max_chars:
                        break

            if max_chars and total_chars >= max_chars:
                break

        # Process remaining buffer
        if buffer and not (max_chars and total_chars >= max_chars):
            remaining_text = extract_text_from_wikipedia(buffer)
            if remaining_text:
                extracted_texts.append(remaining_text)

    print()  # New line after progress

    # Combine and write
    final_text = '\n\n'.join(extracted_texts)

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(final_text)

    mb_size = len(final_text) / 1024 / 1024
    print(f"Saved {len(final_text):,} characters ({mb_size:.1f} MB)")
    print(f"Articles processed: {len(extracted_texts):,}")


# =============================================================================
# ALTERNATIVE: USING WIKITEXT
# =============================================================================


def download_wikitext_huggingface(output_path: str, size: str = "medium") -> None:
    """
    Download WikiText dataset using Hugging Face datasets library.

    This is more reliable than direct URL downloads as the library
    handles authentication, retries, and caching automatically.

    Reference: "The WikiText Long Term Dependency Language Modeling Dataset"
    (Merity et al., 2016)

    Args:
        output_path: Where to save the dataset
        size: "small" (wiki2), "medium" (wiki103)
    """
    if not HAS_DATASETS:
        raise RuntimeError(
            "Hugging Face datasets library not installed. "
            "Install it with: pip install datasets"
        )

    # Map size to dataset name
    dataset_names = {
        "small": "wikitext-2-raw-v1",
        "medium": "wikitext-103-raw-v1",
    }

    if size not in dataset_names:
        raise ValueError(f"Size must be one of: {list(dataset_names.keys())}")

    dataset_name = dataset_names[size]

    print(f"Loading WikiText-{size if size != 'small' else '2'} from Hugging Face...")
    print(f"Dataset: {dataset_name}")
    print()

    try:
        # Load dataset (only training split)
        # Use dataset_name directly as the config name
        dataset = load_dataset("wikitext", dataset_name, split="train")

        # Combine all text
        print("Processing text...")
        text = "\n\n".join(example["text"] for example in dataset)

        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        # Write to output
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)

        mb_size = len(text) / 1024 / 1024
        print(f"Saved {len(text):,} characters ({mb_size:.1f} MB)")
        print(f"Saved to: {output_path}")

    except Exception as e:
        raise RuntimeError(f"Failed to download WikiText from Hugging Face: {e}")


def download_wikitext_direct(output_path: str, size: str = "medium") -> None:
    """
    Download WikiText dataset via direct URL (legacy method).

    This method may not work reliably due to URL changes or access issues.
    Prefer using download_wikitext_huggingface() instead.

    Args:
        output_path: Where to save the dataset
        size: "small" (wiki2-raw), "medium" (wiki103-raw)
    """
    # WikiText-2 and WikiText-103 URLs
    WIKITEXT_URLS = {
        "small": "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip",
        "medium": "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip",
    }

    if size not in WIKITEXT_URLS:
        raise ValueError(f"Size must be one of: {list(WIKITEXT_URLS.keys())}")

    url = WIKITEXT_URLS[size]

    print(f"Downloading WikiText-{size if size != 'small' else '2'} dataset...")
    print(f"URL: {url}")
    print()

    # Download
    temp_zip = output_path + ".zip"
    try:
        download_with_progress(url, temp_zip)
    except RuntimeError as e:
        raise RuntimeError(
            f"{e}\n\n"
            f"Tip: Try using --source huggingface or install the datasets library:\n"
            f"  pip install datasets\n"
            f"Then run: python download_data.py --size {size} --source huggingface"
        )

    # Extract
    print("Extracting...")
    import zipfile
    try:
        with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(temp_zip))
    except zipfile.BadZipFile:
        os.remove(temp_zip)
        raise RuntimeError(
            "Downloaded file is not a valid zip file. "
            "The download may have failed. Try using --source huggingface instead."
        )

    # Find and process the wiki.train.raw file
    extracted_dir = os.path.dirname(temp_zip)
    raw_files = list(Path(extracted_dir).rglob("wiki.train.raw"))

    if not raw_files:
        raise FileNotFoundError("Could not find wiki.train.raw in extracted files")

    raw_file = raw_files[0]

    # Read and combine (using only training data)
    with open(raw_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # Write to output
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)

    # Cleanup
    os.remove(temp_zip)

    mb_size = len(text) / 1024 / 1024
    print(f"Extracted {len(text):,} characters ({mb_size:.1f} MB)")
    print(f"Saved to: {output_path}")


def download_wikitext(output_path: str, size: str = "medium", method: str = "auto") -> None:
    """
    Download WikiText dataset (pre-processed Wikipedia).

    WikiText is a pre-processed Wikipedia dataset commonly used
    for language modeling research.

    Reference: "The WikiText Long Term Dependency Language Modeling Dataset"
    Salesforce research, 2016

    Args:
        output_path: Where to save the dataset
        size: "small" (wiki2-raw), "medium" (wiki103-raw)
        method: "auto", "huggingface", or "direct"
    """
    if method == "auto":
        # Use Hugging Face if available, otherwise direct download
        if HAS_DATASETS:
            method = "huggingface"
        else:
            method = "direct"
            print("Note: Hugging Face datasets library not installed.")
            print("For better reliability, install it with: pip install datasets")
            print()

    if method == "huggingface":
        download_wikitext_huggingface(output_path, size)
    else:
        download_wikitext_direct(output_path, size)


# =============================================================================
# MAIN DOWNLOAD FUNCTION
# =============================================================================


def download_wikipedia_data(
    size: str = "medium",
    output_path: Optional[str] = None,
    source: str = "wikitext",
    method: str = "auto"
) -> str:
    """
    Download Wikipedia data for BPE training.

    PAPER REFERENCE:
    "The WikiText Long Term Dependency Language Modeling Dataset"
    (Merity et al., 2016)

    WikiText is preferred over raw Wikipedia dumps because:
    1. Pre-processed and cleaned
    2. Standard benchmark dataset
    3. Good quality text

    Args:
        size: "small", "medium", or "large"
        output_path: Where to save (default: data/wikipedia_train.txt)
        source: "wikitext" (recommended) or "dump"
        method: "auto", "huggingface", or "direct"

    Returns:
        Path to downloaded data file
    """
    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data",
            "wikipedia_train.txt"
        )

    print("=" * 70)
    print("Wikipedia Data Download for BPE Training")
    print("=" * 70)
    print(f"Size: {size}")
    print(f"Source: {source}")
    print(f"Output: {output_path}")
    print("=" * 70)
    print()

    if source == "wikitext":
        # Use pre-processed WikiText dataset (recommended)
        download_wikitext(output_path, size, method=method)
    else:
        # Use raw Wikipedia dump (more complex, larger)
        raise NotImplementedError(
            "Direct Wikipedia dump download not yet implemented. "
            "Use --source wikitext for pre-processed dataset."
        )

    print()
    print("=" * 70)
    print("Download Complete!")
    print("=" * 70)
    print(f"You can now train the tokenizer with:")
    print(f"  uv run python phase1_foundation/project2_tokenizer/train_tokenizer.py")
    print()

    return output_path


# =============================================================================
# CLI INTERFACE
# =============================================================================


def main():
    """
    Command-line interface for downloading Wikipedia data.

    EXAMPLES:
    ---------
    # Download small WikiText dataset (good for testing)
    python download_data.py --size small

    # Download medium WikiText dataset (good for training)
    python download_data.py --size medium

    # Specify output path
    python download_data.py --size medium --output /path/to/corpus.txt

    # Use raw Wikipedia dump
    python download_data.py --source dump --size medium
    """
    parser = argparse.ArgumentParser(
        description="Download Wikipedia data for BPE tokenizer training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_data.py --size small
  python download_data.py --size medium
  python download_data.py --output /path/to/corpus.txt

Sizes:
  small   - WikiText-2 (~10 MB text, good for testing)
  medium  - WikiText-103 (~500 MB text, good for training)
  large   - Full Wikipedia (not implemented, use --source dump)

Sources:
  wikitext - Pre-processed WikiText dataset (recommended)
  dump     - Raw Wikipedia dump (experimental)
        """
    )

    parser.add_argument(
        "--size", "-s",
        type=str,
        default="medium",
        choices=["small", "medium", "large"],
        help="Dataset size (default: medium)"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path"
    )

    parser.add_argument(
        "--source",
        type=str,
        default="wikitext",
        choices=["wikitext", "dump"],
        help="Data source (default: wikitext)"
    )

    parser.add_argument(
        "--method", "-m",
        type=str,
        default="auto",
        choices=["auto", "huggingface", "direct"],
        help="Download method: auto (try huggingface, fallback to direct), "
             "huggingface (requires datasets library), or direct (HTTP download)"
    )

    args = parser.parse_args()

    download_wikipedia_data(
        size=args.size,
        output_path=args.output,
        source=args.source,
        method=args.method
    )


if __name__ == "__main__":
    main()
