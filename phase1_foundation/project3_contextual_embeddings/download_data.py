# ruff: noqa
"""
Data Download Script for BERT-Style Contextual Embeddings

This script downloads and prepares training data for BERT pre-training.

DATASETS:
---------
- WikiText-2: Wikipedia articles for pre-training
- Project Gutenberg: Books for larger-scale training

USAGE:
------
uv run python phase1_foundation/project3_contextual_embeddings/download_data.py --size small
"""

import argparse
import os
from typing import Optional


# =============================================================================
# DATA DOWNLOAD
# =============================================================================


def download_wikitext(size: str = "small", save_path: Optional[str] = None):
    """
    Download WikiText dataset for BERT pre-training.

    PAPER REFERENCE: "The WikiText Long Term Dependency Language Modeling Dataset"
    (Merity et al., 2016)

    WikiText is a collection of texts from Wikipedia articles.
    - WikiText-2: ~2M tokens
    - WikiText-103: ~100M tokens

    From the paper:
    "WikiText-2 is a smaller collection... suitable for rapid prototyping"

    Args:
        size: 'small' for WikiText-2, 'large' for WikiText-103
        save_path: Path to save the data (default: data/wikitext_train.txt)
    """
    print("=" * 60)
    print(f"DOWNLOADING WIKITEXT-{size.upper() if size == 'small' else '103'}")
    print("=" * 60)

    # Determine dataset config
    if size == "small":
        config_name = "wikitext-2-raw-v1"
    else:
        config_name = "wikitext-103-raw-v1"

    print(f"Dataset: wikitext")
    print(f"Config: {config_name}")

    try:
        from datasets import load_dataset

        # Load dataset
        print("Downloading dataset from Hugging Face...")
        dataset = load_dataset("wikitext", config_name, split="train")

        print(f"✓ Downloaded {len(dataset):,} examples")

        # Extract text
        print("Extracting text...")
        text = "\n".join(example["text"] for example in dataset if example["text"].strip())

        print(f"✓ Extracted {len(text):,} characters")
        print(f"✓ Approximately {len(text.split()):,} tokens (word-level)")

        # Save training data
        if save_path is None:
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            save_path = os.path.join(root_dir, "data", "wikitext_train.txt")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"✓ Saved training data to {save_path}")

        # Download validation data
        print("\nDownloading validation data...")
        val_dataset = load_dataset("wikitext", config_name, split="validation")
        val_text = "\n".join(
            example["text"] for example in val_dataset if example["text"].strip()
        )

        val_path = save_path.replace("_train.txt", "_val.txt")
        with open(val_path, "w", encoding="utf-8") as f:
            f.write(val_text)

        print(f"✓ Saved validation data to {val_path}")

        # Download test data
        print("\nDownloading test data...")
        test_dataset = load_dataset("wikitext", config_name, split="test")
        test_text = "\n".join(
            example["text"] for example in test_dataset if example["text"].strip()
        )

        test_path = save_path.replace("_train.txt", "_test.txt")
        with open(test_path, "w", encoding="utf-8") as f:
            f.write(test_text)

        print(f"✓ Saved test data to {test_path}")

        return save_path, val_path

    except Exception as e:
        print(f"✗ Error downloading WikiText: {e}")
        return None, None


def download_wikipedia_articles(save_path: Optional[str] = None, num_articles: int = 100):
    """
    Download Wikipedia articles for training.

    Uses the Wikipedia API to fetch random articles.

    Args:
        save_path: Path to save the data
        num_articles: Number of articles to download
    """
    print("=" * 60)
    print("DOWNLOADING WIKIPEDIA ARTICLES")
    print("=" * 60)

    try:
        import wikipedia

        wikipedia.set_lang("en")

        print(f"Downloading {num_articles} random Wikipedia articles...")

        all_text = []
        for i in range(num_articles):
            try:
                # Get random article
                title = wikipedia.random(pages=1)
                page = wikipedia.page(title, auto_suggest=False)
                content = page.content

                all_text.append(content)

                if (i + 1) % 10 == 0:
                    print(f"  Downloaded {i + 1}/{num_articles} articles...")

            except Exception as e:
                print(f"  ✗ Error downloading article {i + 1}: {e}")
                continue

        # Combine all text
        text = "\n\n".join(all_text)

        print(f"\n✓ Downloaded {len(all_text)} articles")
        print(f"✓ Total characters: {len(text):,}")

        # Save to file
        if save_path is None:
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            save_path = os.path.join(root_dir, "data", "wikipedia_train.txt")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"✓ Saved to {save_path}")

        return save_path

    except ImportError:
        print("✗ Wikipedia library not installed. Install with: pip install wikipedia")
        return None
    except Exception as e:
        print(f"✗ Error downloading Wikipedia articles: {e}")
        return None


def create_sample_data(save_path: Optional[str] = None):
    """
    Create sample training data for quick testing.

    Uses built-in text for quick experiments without downloading.

    Args:
        save_path: Path to save the data
    """
    print("=" * 60)
    print("CREATING SAMPLE DATA")
    print("=" * 60)

    # Sample text for quick testing
    sample_text = """
    The history of artificial intelligence (AI) began in antiquity, with myths, stories
    and rumors of artificial beings endowed with intelligence or consciousness by master craftsmen.
    The seeds of modern AI were planted by classical philosophers who attempted to describe the
    process of human thinking as the mechanical manipulation of symbols.

    In the 1940s, the study of artificial intelligence emerged as a academic discipline.
    The field was founded on the claim that human intelligence can be described so precisely
    that a machine can be made to simulate it. This raised philosophical arguments about the
    nature of the mind and the ethics of creating artificial beings.

    The concept of machine learning dates back to the 1950s. Early pioneers like Alan Turing
    proposed that if a human could not distinguish between a machine and a human, the machine
    could be considered intelligent. This became known as the Turing Test.

    Deep learning, a subset of machine learning, uses neural networks with multiple layers to
    progressively extract higher-level features from raw input. For example, in image processing,
    lower layers may identify edges, while higher layers may identify the concepts relevant to a
    human such as digits or letters or faces.

    Transformers are a type of deep learning model that use self-attention mechanisms to process
    sequential data. Unlike recurrent neural networks, transformers can process the entire input
    sequence simultaneously, allowing for much better parallelization during training.

    BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based machine
    learning technique for natural language processing (NLP) pre-training developed by Google.
    BERT was created and published in 2018 by Jacob Devlin and his colleagues from Google.

    The transformer architecture has revolutionized natural language processing. It enables
    models to process text more effectively by attending to different parts of the input
    sequence simultaneously, rather than processing words one at a time.

    Large language models like GPT-3 have shown remarkable capabilities in generating human-like
    text, answering questions, and performing a wide range of language tasks. These models are
    typically trained on massive amounts of text data and have billions of parameters.

    """ * 100  # Repeat for more data

    print(f"Created sample data with {len(sample_text):,} characters")

    if save_path is None:
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        save_path = os.path.join(root_dir, "data", "sample_train.txt")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(sample_text)

    print(f"✓ Saved to {save_path}")

    return save_path


# =============================================================================
# MAIN
# =============================================================================


def main():
    """
    Main download function.
    """
    parser = argparse.ArgumentParser(
        description="Download data for BERT pre-training"
    )

    parser.add_argument(
        "--size",
        type=str,
        default="small",
        choices=["small", "large", "wiki", "sample"],
        help="Dataset size: small (WikiText-2), large (WikiText-103), wiki (Wikipedia API), sample (built-in)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: data/wikitext_train.txt)"
    )

    parser.add_argument(
        "--num-articles",
        type=int,
        default=100,
        help="Number of Wikipedia articles to download (default: 100)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("BERT DATA DOWNLOAD")
    print("=" * 60)
    print(f"Size: {args.size}")
    print(f"Output: {args.output or 'default'}")

    if args.size == "sample":
        path = create_sample_data(args.output)
        if path:
            print(f"\n✓ Sample data created successfully!")
            print(f"  Path: {path}")
    elif args.size == "wiki":
        path = download_wikipedia_articles(args.output, args.num_articles)
        if path:
            print(f"\n✓ Wikipedia data downloaded successfully!")
            print(f"  Path: {path}")
        else:
            print("\n✗ Failed to download Wikipedia data")
            print("  Try using --size small for WikiText-2")
    else:
        train_path, val_path = download_wikitext(args.size, args.output)

        if train_path:
            print(f"\n✓ Data downloaded successfully!")
            print(f"  Training: {train_path}")
            if val_path:
                print(f"  Validation: {val_path}")
        else:
            print("\n✗ Failed to download data")
            print("  Try using --size sample for built-in sample data")

            # Fallback to sample
            print("\n  Creating sample data as fallback...")
            path = create_sample_data()
            if path:
                print(f"  Fallback: {path}")


if __name__ == "__main__":
    main()
