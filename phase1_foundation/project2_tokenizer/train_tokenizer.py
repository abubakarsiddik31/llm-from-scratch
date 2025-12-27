#!/usr/bin/env python3
"""
BPE Tokenizer Training Script

This script trains a Byte-Pair Encoding tokenizer on text data.

PAPER REFERENCE:
- "Byte-Pair Encoding: Subword-Based Machine Translation" (Sennrich et al., 2016)
- "Language Models are Unsupervised Multitask Learners" (GPT-2, 2019)

USAGE:
------
python train_tokenizer.py [--vocab_size VOCAB_SIZE] [--min_freq MIN_FREQ]

EXAMPLES:
---------
# Train with default settings (50K vocab)
python train_tokenizer.py

# Train with smaller vocabulary
python train_tokenizer.py --vocab_size 10000

# Train with higher frequency threshold
python train_tokenizer.py --min_freq 5
"""

import argparse
import os
import time

import config
from tokenizer import BPETokenizer, get_tokenizer_stats


# =============================================================================
# DATA LOADING
# =============================================================================


def load_training_data(filepath: str) -> str:
    """
    Load training text from file.

    Args:
        filepath: Path to training text file

    Returns:
        Text content as string

    RAISES:
        FileNotFoundError: If file doesn't exist
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Training data file not found: {filepath}\n"
            f"Please create this file or download training data first."
        )

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    return text


def create_sample_training_data(filepath: str, size_mb: float = 10.0) -> None:
    """
    Create sample training data from Project Gutenberg or similar sources.

    For demonstration, this creates a simple text file.
    In production, you'd download:
    - Wikipedia dump
    - Project Gutenberg corpus
    - WebText sample

    Args:
        filepath: Where to save the training data
        size_mb: Approximate size in MB
    """
    print(f"Creating sample training data at {filepath}...")

    # Sample text (diverse content for better BPE learning)
    sample_texts = [
        # Classic literature (public domain)
        """
        To be, or not to be, that is the question:
        Whether 'tis nobler in the mind to suffer
        The slings and arrows of outrageous fortune,
        Or to take arms against a sea of troubles
        And by opposing end them.
        """,

        # Scientific text
        """
        The theory of evolution by natural selection,
        first formulated in Darwin's book "On the Origin of Species"
        in 1859, is the process by which organisms change over time
        as a result of changes in heritable physical or behavioral traits.
        """,

        # Technical documentation
        """
        A transformer is a deep learning model that adopts the mechanism
        of self-attention, differentially weighting the significance of
        each part of the input data. It is used primarily in the fields of
        natural language processing and computer vision.
        """,

        # Conversational text
        """
        Hello! How are you doing today? I hope you're having a wonderful time.
        Would you like to go for a walk later? The weather is quite nice.
        """,

        # Programming text
        """
        def calculate_fibonacci(n):
            if n <= 1:
                return n
            return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)
        """,
    ]

    # Repeat to create larger corpus
    target_chars = int(size_mb * 1024 * 1024)
    result = []

    while sum(len(t) for t in result) < target_chars:
        result.extend(sample_texts)

    text = "\n\n".join(result)[:target_chars]

    # Create directory if needed
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Created {len(text):,} characters ({len(text) / 1024 / 1024:.2f} MB)")


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================


def train_tokenizer(vocab_size: int = None, min_frequency: int = None,
                    data_path: str = None, checkpoint_path: str = None) -> BPETokenizer:
    """
    Train BPE tokenizer on text data.

    PAPER REFERENCE: Sennrich et al. (2016), Algorithm 1

    TRAINING PROCESS:
    -----------------
    1. Load training text
    2. Initialize BPE tokenizer
    3. Train (learn merge rules)
    4. Save trained tokenizer

    Args:
        vocab_size: Target vocabulary size
        min_frequency: Minimum merge frequency
        data_path: Path to training data
        checkpoint_path: Path to save trained tokenizer

    Returns:
        Trained BPETokenizer instance
    """
    # Use config defaults if not specified
    vocab_size = vocab_size or config.VOCAB_SIZE
    min_frequency = min_frequency or config.MIN_FREQUENCY
    data_path = data_path or config.TRAIN_DATA_PATH
    checkpoint_path = checkpoint_path or os.path.join(
        config.CHECKPOINT_DIR, "tokenizer.pkl"
    )

    print("=" * 70)
    print("BPE Tokenizer Training")
    print("=" * 70)
    print(f"Vocabulary size:     {vocab_size:,}")
    print(f"Min frequency:       {min_frequency}")
    print(f"Training data:       {data_path}")
    print(f"Checkpoint path:     {checkpoint_path}")
    print("=" * 70)
    print()

    # ===========================================================================
    # STEP 1: LOAD TRAINING DATA
    # ===========================================================================

    """
    From BPE paper:
    "We train BPE on a monolingual corpus... The size of the training
    data affects the quality of learned subwords"

    REQUIREMENTS:
    - Large enough to learn meaningful patterns (>1MB recommended)
    - Diverse content for general vocabulary
    - Clean text (remove excessive markup)
    """
    print("Loading training data...")

    if not os.path.exists(data_path):
        print(f"Training data not found at {data_path}")
        print("Creating sample training data...")
        create_sample_training_data(data_path, size_mb=10.0)

    text = load_training_data(data_path)

    print(f"Loaded {len(text):,} characters ({len(text) / 1024 / 1024:.2f} MB)")
    print()

    # ===========================================================================
    # STEP 2: INITIALIZE TOKENIZER
    # ===========================================================================

    """
    Create fresh BPE tokenizer instance.
    Special tokens are added automatically.
    """
    tokenizer = BPETokenizer()

    # ===========================================================================
    # STEP 3: TRAIN
    # ===========================================================================

    """
    From BPE paper, Algorithm 1:
    "Repeat until vocabulary size reached:
       1. Count frequency of all adjacent pairs
       2. Find most frequent pair
       3. Merge into new token
       4. Update all occurrences in corpus"

    This is the core learning step where we discover subword patterns.
    """
    start_time = time.time()

    tokenizer.train(
        text=text,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
    )

    training_time = time.time() - start_time

    print()
    print(f"Training completed in {training_time:.2f} seconds")
    print()

    # ===========================================================================
    # STEP 4: DISPLAY STATISTICS
    # ===========================================================================

    """
    Show statistics about the trained tokenizer.
    """
    print("=" * 70)
    print("Tokenizer Statistics")
    print("=" * 70)

    stats = get_tokenizer_stats(tokenizer)

    print(f"Vocabulary size:        {stats['vocab_size']:,}")
    print(f"Number of merges:       {stats['num_merges']:,}")
    print(f"Average token length:   {stats['avg_token_length']:.2f} characters")
    print(f"Min token length:       {stats['min_token_length']} characters")
    print(f"Max token length:       {stats['max_token_length']} characters")
    print()

    # Show some learned tokens
    print("Sample learned tokens (longest 20):")
    # Sort by length (descending), exclude special tokens
    sorted_tokens = sorted(
        [(token, idx) for token, idx in tokenizer.vocab.items()
         if token not in config.SPECIAL_TOKENS],
        key=lambda x: len(x[0]),
        reverse=True
    )

    for token, idx in sorted_tokens[:20]:
        print(f"  ID {idx:5d}: {repr(token)}")

    print("=" * 70)
    print()

    # ===========================================================================
    # STEP 5: SAVE TOKENIZER
    # ===========================================================================

    """
    Save trained tokenizer to disk for later use.

    SAVED DATA:
    - vocab: Token to ID mapping
    - inverse_vocab: ID to token mapping
    - merges: Ordered list of merge operations

    The saved tokenizer can be loaded for:
    - Training language models (Project 3)
    - Encoding/decoding text
    - Transfer learning
    """
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    tokenizer.save(checkpoint_path)

    print()
    print("Training complete!")
    print(f"Tokenizer saved to: {checkpoint_path}")
    print()
    print("You can now use this tokenizer with:")
    print(f"  python test_tokenizer.py --tokenizer {checkpoint_path}")

    return tokenizer


# =============================================================================
# CLI INTERFACE
# =============================================================================


def main():
    """
    Command-line interface for training BPE tokenizer.

    EXAMPLES:
    ---------
    Train with defaults:
        python train_tokenizer.py

    Train with custom vocab size:
        python train_tokenizer.py --vocab_size 10000

    Train with higher frequency threshold:
        python train_tokenizer.py --min_freq 5

    Use custom training data:
        python train_tokenizer.py --data_path /path/to/corpus.txt

    Specify output path:
        python train_tokenizer.py --output /path/to/tokenizer.pkl
    """
    parser = argparse.ArgumentParser(
        description="Train a Byte-Pair Encoding (BPE) tokenizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_tokenizer.py
  python train_tokenizer.py --vocab_size 10000
  python train_tokenizer.py --min_freq 5
  python train_tokenizer.py --data_path /path/to/corpus.txt
  python train_tokenizer.py --output /path/to/tokenizer.pkl
        """
    )

    parser.add_argument(
        "--vocab_size", "-v",
        type=int,
        default=config.VOCAB_SIZE,
        help=f"Target vocabulary size (default: {config.VOCAB_SIZE})"
    )

    parser.add_argument(
        "--min_freq", "-f",
        type=int,
        default=config.MIN_FREQUENCY,
        help=f"Minimum merge frequency (default: {config.MIN_FREQUENCY})"
    )

    parser.add_argument(
        "--data_path", "-d",
        type=str,
        default=config.TRAIN_DATA_PATH,
        help=f"Path to training data file (default: {config.TRAIN_DATA_PATH})"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output path for trained tokenizer "
             f"(default: {config.CHECKPOINT_DIR}/tokenizer.pkl)"
    )

    args = parser.parse_args()

    # Build checkpoint path
    checkpoint_path = args.output or os.path.join(
        config.CHECKPOINT_DIR, "tokenizer.pkl"
    )

    # Train
    train_tokenizer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_freq,
        data_path=args.data_path,
        checkpoint_path=checkpoint_path,
    )


if __name__ == "__main__":
    main()
