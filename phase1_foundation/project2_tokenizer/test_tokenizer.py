#!/usr/bin/env python3
"""
BPE Tokenizer Testing and Demonstration Script

This script tests a trained BPE tokenizer with various examples.

PAPER REFERENCE:
- "Byte-Pair Encoding: Subword-Based Machine Translation" (Sennrich et al., 2016)

USAGE:
------
python test_tokenizer.py [--tokenizer CHECKPOINT_PATH]

EXAMPLES:
---------
# Test with default tokenizer path
python test_tokenizer.py

# Test with specific tokenizer
python test_tokenizer.py --tokenizer /path/to/tokenizer.pkl

# Interactive mode
python test_tokenizer.py --interactive

# Test specific text
python test_tokenizer.py --text "Hello, world!"
"""

import argparse
import os
import sys

import config
from tokenizer import BPETokenizer


# =============================================================================
# TEST CASES
# =============================================================================


def test_roundtrip(tokenizer: BPETokenizer, text: str) -> bool:
    """
    Test that decode(encode(text)) equals original text.

    From BPE paper: "The encoding and decoding should be lossless"

    Args:
        tokenizer: Trained BPETokenizer instance
        text: Text to test

    Returns:
        True if roundtrip successful, False otherwise
    """
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    success = text == decoded

    print(f"  Original:  '{text[:50]}{'...' if len(text) > 50 else ''}'")
    print(f"  Encoded:   {len(encoded)} tokens")
    print(f"  Decoded:   '{decoded[:50]}{'...' if len(decoded) > 50 else ''}'")
    print(f"  Match:     {'✓' if success else '✗'}")

    return success


def test_common_words(tokenizer: BPETokenizer) -> None:
    """
    Test that common English words are encoded efficiently.

    From BPE paper:
    "Common words should become single tokens, rare words split into subwords"

    EXPECTED BEHAVIOR:
    - Common words: 1-2 tokens (learned as single unit)
    - Rare words: Multiple tokens (split into subwords)
    - Unknown words: Multiple subword tokens

    Args:
        tokenizer: Trained BPETokenizer instance
    """
    print("\n" + "=" * 70)
    print("Common Words Test")
    print("=" * 70)
    print("Testing encoding of common English words...")
    print()

    # Test words of different frequencies
    test_words = {
        "Very common": [
            "the", "be", "to", "of", "and", "a", "in", "that",
            "have", "I", "it", "for", "not", "on", "with"
        ],
        "Common": [
            "hello", "world", "good", "bad", "time", "work",
            "life", "hand", "part", "place", "case", "point"
        ],
        "Less common": [
            "algorithm", "language", "computer", "programming",
            "transformer", "attention", "embedding", "vocabulary"
        ],
        "Rare/Technical": [
            "hyperparameter", "regularization", "backpropagation",
            "convolutional", "reinforcement", "interpretability"
        ],
        "Very rare/Made up": [
            "antidisestablishmentarianism",
            "supercalifragilisticexpialidocious",
            "pneumonoultramicroscopicsilicovolcanoconiosis"
        ],
    }

    for category, words in test_words.items():
        print(f"{category}:")
        for word in words:
            encoded = tokenizer.encode(word)
            # Get token strings
            token_strs = []
            for tid in encoded:
                if tid in tokenizer.inverse_vocab:
                    token = tokenizer.inverse_vocab[tid]
                    if token not in ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]:
                        token_strs.append(token)

            # Show encoding
            if len(encoded) == 1:
                status = "✓ (single token)"
            elif len(encoded) <= 3:
                status = f"({len(encoded)} tokens)"
            else:
                status = f"({len(encoded)} tokens - split)"

            print(f"  {word:40s} → [{', '.join(repr(t) for t in token_strs[:5])}] {status}")
        print()


def test_subword_splitting(tokenizer: BPETokenizer) -> None:
    """
    Test how BPE splits words into subwords.

    From BPE paper:
    "BPE allows the model to handle rare words by splitting them
    into subword units that have been seen in other contexts"

    KEY INSIGHT:
    The same subword can appear in multiple words.
    Example: "help" appears in "helping", "helpful", "helper"
    """
    print("\n" + "=" * 70)
    print("Subword Splitting Test")
    print("=" * 70)
    print("Testing how related words share subwords...")
    print()

    # Test word families
    word_families = {
        "help": ["help", "helps", "helped", "helping", "helper", "helpful"],
        "play": ["play", "plays", "played", "playing", "player", "playful"],
        "happy": ["happy", "happier", "happiest", "happiness", "unhappy"],
        "able": ["able", "unable", "enable", "disabled", "capability"],
    }

    for base, words in word_families.items():
        print(f"Word family: '{base}'")
        encodings = {}

        for word in words:
            encoded = tokenizer.encode(word)
            # Get token strings (excluding special tokens)
            tokens = []
            for tid in encoded:
                if tid in tokenizer.inverse_vocab:
                    token = tokenizer.inverse_vocab[tid]
                    if token not in ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]:
                        tokens.append(token)
            encodings[word] = tokens

        # Show encodings
        for word, tokens in encodings.items():
            token_str = " + ".join(repr(t) for t in tokens)
            print(f"  {word:15s} → {token_str}")

        # Show shared subwords
        all_tokens = set()
        for tokens in encodings.values():
            all_tokens.update(tokens)

        shared = [t for t in all_tokens if any(t in encodings[w] for w in words if w != word)]
        if shared:
            print(f"  Shared subwords: {', '.join(repr(t) for t in shared[:5])}")
        print()


def test_compression_ratio(tokenizer: BPETokenizer) -> None:
    """
    Test the compression ratio of BPE vs character-level encoding.

    From BPE paper:
    "BPE achieves a better compression ratio than character-level models"

    METRICS:
    - Character-level: One token per character
    - BPE: Fewer tokens (subwords merge multiple characters)
    - Compression ratio: char_tokens / bpe_tokens
    """
    print("\n" + "=" * 70)
    print("Compression Ratio Test")
    print("=" * 70)
    print("Comparing BPE vs character-level encoding...")
    print()

    # Test texts
    test_texts = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "To be or not to be, that is the question.",
        """Machine learning is a field of artificial intelligence that
        uses statistical techniques to give computer systems the ability
        to learn from data."""
    ]

    for text in test_texts:
        bpe_encoded = tokenizer.encode(text)
        char_encoded = len(text.replace(" ", ""))  # Character count (no spaces)

        # Calculate compression
        compression = char_encoded / len(bpe_encoded) if bpe_encoded else 1.0

        print(f"Text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        print(f"  Characters:  {char_encoded}")
        print(f"  BPE tokens:  {len(bpe_encoded)}")
        print(f"  Compression: {compression:.2f}x")
        print()


def test_special_cases(tokenizer: BPETokenizer) -> None:
    """
    Test special cases and edge conditions.

    TESTS:
    - Empty string
    - Single character
    - Repeated characters
    - Mixed case
    - Numbers and punctuation
    - Unknown characters (if any)
    """
    print("\n" + "=" * 70)
    print("Special Cases Test")
    print("=" * 70)
    print("Testing edge cases and special conditions...")
    print()

    test_cases = {
        "Empty string": "",
        "Single character": "a",
        "Repeated characters": "aaaaaaa",
        "All caps": "HELLO WORLD",
        "Mixed case": "HeLLo WoRLd",
        "Numbers": "123456789",
        "Punctuation": "!@#$%^&*()",
        "URL": "https://example.com/path/to/resource",
        "Email": "user@example.com",
        "Code": "def function(arg): return arg * 2",
    }

    for name, text in test_cases.items():
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)

        match = text == decoded
        status = "✓" if match else "✗"

        print(f"{name:25s}: '{text[:30]}'")
        print(f"  → {len(encoded)} tokens, roundtrip: {status}")
        if not match:
            print(f"  Decoded: '{decoded}'")
        print()


# =============================================================================
# INTERACTIVE MODE
# =============================================================================


def interactive_mode(tokenizer: BPETokenizer) -> None:
    """
    Interactive mode for testing the tokenizer.

    Commands:
    - Text input: Encode and display the text
    - 'stats': Show tokenizer statistics
    - 'vocab': Show vocabulary samples
    - 'quit' or 'exit': Exit interactive mode
    """
    print("\n" + "=" * 70)
    print("Interactive Mode")
    print("=" * 70)
    print("Enter text to encode/decode, or 'help' for commands.")
    print()

    while True:
        try:
            user_input = input("bpe> ").strip()

            if not user_input:
                continue

            # Commands
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            elif user_input.lower() == "help":
                print("""
Commands:
  <text>        - Encode and decode text
  stats         - Show tokenizer statistics
  vocab         - Show vocabulary samples
  merges        - Show merge rules
  quit/exit     - Exit interactive mode
                """)

            elif user_input.lower() == "stats":
                print(f"\nVocabulary size: {len(tokenizer.vocab):,}")
                print(f"Number of merges: {len(tokenizer.merges):,}")
                print()

            elif user_input.lower() == "vocab":
                print("\nVocabulary samples (longest tokens):")
                sorted_tokens = sorted(
                    [(t, i) for t, i in tokenizer.vocab.items()
                     if t not in config.SPECIAL_TOKENS],
                    key=lambda x: len(x[0]),
                    reverse=True
                )
                for token, idx in sorted_tokens[:20]:
                    print(f"  ID {idx:5d}: {repr(token)}")
                print()

            elif user_input.lower() == "merges":
                print(f"\nFirst 20 merge rules:")
                for i, (a, b) in enumerate(tokenizer.merges[:20], 1):
                    print(f"  {i:3d}. {repr(a)} + {repr(b)} → {repr(a+b)}")
                print(f"  ... ({len(tokenizer.merges)} total merges)")
                print()

            else:
                # Encode and decode
                encoded = tokenizer.encode(user_input)
                decoded = tokenizer.decode(encoded)

                print(f"\nInput:  '{user_input}'")
                print(f"\nEncoded ({len(encoded)} tokens):")
                print(f"  {encoded}")

                print(f"\nTokens:")
                for tid in encoded:
                    if tid in tokenizer.inverse_vocab:
                        token = tokenizer.inverse_vocab[tid]
                        print(f"  ID {tid:5d}: {repr(token)}")

                print(f"\nDecoded: '{decoded}'")
                print(f"Match: {'✓' if user_input == decoded else '✗'}")
                print()

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


# =============================================================================
# MAIN TEST FUNCTION
# =============================================================================


def test_tokenizer(checkpoint_path: str, interactive: bool = False,
                   test_text: str = None) -> None:
    """
    Test a trained BPE tokenizer.

    Args:
        checkpoint_path: Path to saved tokenizer
        interactive: If True, enter interactive mode
        test_text: If provided, test this specific text
    """
    print("=" * 70)
    print("BPE Tokenizer Testing")
    print("=" * 70)
    print(f"Loading tokenizer from: {checkpoint_path}")
    print()

    # Load tokenizer
    if not os.path.exists(checkpoint_path):
        print(f"Error: Tokenizer file not found: {checkpoint_path}")
        print("\nPlease train a tokenizer first:")
        print("  python train_tokenizer.py")
        sys.exit(1)

    tokenizer = BPETokenizer.load(checkpoint_path)

    # Quick info
    print()
    print(f"Vocabulary size: {len(tokenizer.vocab):,}")
    print(f"Merge rules: {len(tokenizer.merges):,}")
    print()

    # Interactive mode
    if interactive:
        interactive_mode(tokenizer)
        return

    # Test specific text
    if test_text:
        print("=" * 70)
        print("Single Text Test")
        print("=" * 70)
        test_roundtrip(tokenizer, test_text)
        return

    # Run all tests
    print("\nRunning all tests...\n")

    # Test 1: Roundtrip
    print("Test 1: Roundtrip (encode → decode)")
    print("-" * 70)
    test_texts = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "Testing 1, 2, 3...",
        """Machine learning is transforming how we solve complex problems
        across various domains from natural language processing to computer
        vision and beyond.""",
    ]

    all_passed = True
    for text in test_texts:
        if not test_roundtrip(tokenizer, text):
            all_passed = False
        print()

    # Test 2: Common words
    test_common_words(tokenizer)

    # Test 3: Subword splitting
    test_subword_splitting(tokenizer)

    # Test 4: Compression ratio
    test_compression_ratio(tokenizer)

    # Test 5: Special cases
    test_special_cases(tokenizer)

    # Summary
    print("=" * 70)
    print("Test Summary")
    print("=" * 70)
    if all_passed:
        print("✓ All roundtrip tests passed!")
    else:
        print("✗ Some roundtrip tests failed!")
    print()


# =============================================================================
# CLI INTERFACE
# =============================================================================


def main():
    """
    Command-line interface for testing BPE tokenizer.
    """
    parser = argparse.ArgumentParser(
        description="Test a trained Byte-Pair Encoding (BPE) tokenizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with default tokenizer
  python test_tokenizer.py

  # Test with specific tokenizer
  python test_tokenizer.py --tokenizer /path/to/tokenizer.pkl

  # Interactive mode
  python test_tokenizer.py --interactive

  # Test specific text
  python test_tokenizer.py --text "Hello, world!"
        """
    )

    parser.add_argument(
        "--tokenizer", "-t",
        type=str,
        default=os.path.join(config.CHECKPOINT_DIR, "tokenizer.pkl"),
        help="Path to trained tokenizer checkpoint"
    )

    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )

    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Test specific text"
    )

    args = parser.parse_args()

    test_tokenizer(
        checkpoint_path=args.tokenizer,
        interactive=args.interactive,
        test_text=args.text,
    )


if __name__ == "__main__":
    main()
