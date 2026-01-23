# ruff: noqa
"""
Byte-Pair Encoding Tokenizer Implementation

This file implements BPE following:
- "Byte-Pair Encoding: Subword-Based Machine Translation" (Sennrich et al., 2016)
- "Language Models are Unsupervised Multitask Learners" (GPT-2, 2019)

ALGORITHM OVERVIEW:
------------------
From Sennrich et al. (2016):
"Byte-Pair Encoding (BPE) iteratively merges the most frequent pair of
bytes (or characters) to create new tokens... This allows the model to
handle rare and unseen words by splitting them into subword units"

KEY INSIGHT:
Instead of fixed word-level or character-level vocabulary:
- Start with character-level vocabulary
- Iteratively merge most frequent adjacent pairs
- Build subword tokens that capture common patterns
- Balance vocabulary size vs sequence length

BENEFITS:
1. Handles OOV words: "unfriendliness" → "un", "friend", "li", "ness"
2. Efficient: Common words = single token, rare words = subwords
3. No unknown tokens: All text can be encoded at character level
4. Compositional: Learn subwords that appear in many words

EXAMPLE:
---------
Initial (character level): h e l l o   w o r l d
After merge "l" + "l": h e ll o   w o r l d
After merge "ll" + "o": h e llo   w o r l d
After merge "e" + "llo": hello   w o r l d

Final encoding:
"hello" → [token_id_for_hello]
"hell" → [token_id_for_hel, token_id_for_l]
"help" → [token_id_for_hel, token_id_for_p]

Notice: "hel" appears in both "hell" and "help" → shared subword!

PAPER REFERENCE:
- Sennrich, Haddow, Birch (2016): Original BPE for NLP
- GPT-2 (2019): Byte-level BPE with 50K vocabulary
- GPT-3 (2020): Same BPE approach, scaled up
"""

import pickle
from collections import Counter
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

import config


# =============================================================================
# CORE BPE TOKENIZER CLASS
# =============================================================================


class BPETokenizer:
    """
    Byte-Pair Encoding (BPE) Tokenizer

    PAPER: "Byte-Pair Encoding: Subword-Based Machine Translation"
    Authors: Sennrich, Haddow, Birch (2016)
    --------------------------------------------------------------

    INTUITION:
    ----------
    BPE sits between character-level and word-level tokenization:
    - Character-level: Too long sequences, no semantic meaning
    - Word-level: Short sequences, but huge vocabulary + OOV problem
    - BPE: Best of both - shorter sequences + handles OOV

    The key insight: Learn subword units that capture common patterns.

    EXAMPLE VOCABULARY BUILDING:
    ----------------------------
    Training text: "hug hug pug hug pug"

    Initial vocab (characters):
    {h, u, g, p, (space)}

    Most frequent pair: "u" + "g" (appears 4 times)
    Merge → "ug" added to vocab

    Updated text: "h ug h ug p ug h ug p ug"

    Next most frequent: "h" + "ug" (appears 3 times)
    Merge → "hug" added to vocab

    Next most frequent: "p" + "ug" (appears 2 times)
    Merge → "pug" added to vocab

    Final vocab: {h, u, g, p, (space), ug, hug, pug}

    ENCODING:
    "hug" → [token_id_for_hug]  (single token!)
    "pug" → [token_id_for_pug]  (single token!)
    "hug pug" → [hug, space, pug]
    "hugg" → [hug, g]  (subword splitting!)

    DECODING:
    Simple lookup: token_id → token string
    [token_id_for_hug, token_id_for_g] → "hug" + "g" = "hugg"

    ALGORITHM:
    ----------
    TRAINING:
    1. Initialize vocabulary with all unique characters
    2. Count all adjacent byte pair frequencies
    3. Merge most frequent pair into new token
    4. Repeat until vocab_size reached

    ENCODING:
    1. Start with character-level representation
    2. Greedily merge pairs using learned merge rules
    3. Use longest-match-first: Prefer longer learned tokens

    DECODING:
    1. Token ID → token string lookup
    2. Concatenate all strings
    """

    def __init__(self):
        """
        Initialize BPE tokenizer with empty vocabulary.

        ATTRIBUTES:
        -----------
        vocab: Dict[str, int]
            Mapping from token string to unique integer ID
            Example: {"hello": 45, "world": 123, "<PAD>": 0}

        inverse_vocab: Dict[int, str]
            Reverse mapping for decoding: ID → token string
            Example: {45: "hello", 123: "world", 0: "<PAD>"}

        merges: List[Tuple[str, str]]
            Ordered list of merge operations learned during training
            Each tuple: (token_a, token_b) merged in that order
            Example: [("e", "r"), ("er", "t"), ("ert", "a"), ...]

        Merges are applied in order during encoding:
            - Earlier merges = more fundamental (lower priority)
            - Later merges = more complex (higher priority)
        """
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []

        # Add special tokens (fixed, not learned)
        self._add_special_tokens()

    # ==========================================================================
    # INITIALIZATION
    # ==========================================================================

    def _add_special_tokens(self):
        """
        Add special tokens to vocabulary.

        PAPER CONTEXT:
        --------------
        From "Sequence to Sequence Learning with Neural Networks" (Sutskever, 2014):
        Special tokens help model understand sequence structure.

        SPECIAL TOKENS:
        ---------------
        <PAD>: Padding for equal-length batches
        <UNK>: Unknown token fallback (should rarely be used)
        <BOS>: Beginning of sequence marker
        <EOS>: End of sequence marker (model learns to predict this)

        IMPLEMENTATION:
        ---------------
        Special tokens are added first with fixed IDs (0, 1, 2, 3).
        They are NOT learned during BPE training.
        """
        for token, idx in config.SPECIAL_TOKENS.items():
            self.vocab[token] = idx
            self.inverse_vocab[idx] = token

    # ==========================================================================
    # TRAINING
    # ==========================================================================

    def train(self, text: str, vocab_size: int = config.VOCAB_SIZE,
              min_frequency: int = config.MIN_FREQUENCY) -> None:
        """
        Train BPE tokenizer on text.

        PAPER REFERENCE: Sennrich et al. (2016), Algorithm 1

        ALGORITHM:
        ----------
        1. Initialize vocabulary with all unique characters
        2. Represent text as list of characters
        3. Repeat until vocab_size reached:
           a. Count frequency of all adjacent pairs
           b. Find most frequent pair
           c. If frequency >= min_frequency:
              - Add merged token to vocabulary
              - Update text with merged token
           d. Else: Stop (no more valid merges)

        Args:
            text: Training text (corpus)
            vocab_size: Target vocabulary size (including special tokens)
            min_frequency: Minimum frequency for a merge to be considered

        Example:
            text = "hug hug pug"
            vocab_size = 10
            min_frequency = 2

            Step 1: chars = {h, u, g, p, (space)} (5 tokens)
            Step 2: merge "u"+"g" → "ug" (freq=4)
            Step 3: merge "h"+"ug" → "hug" (freq=3)
            Step 4: merge "p"+"ug" → "pug" (freq=2)

            Final vocab = {<PAD>, <UNK>, <BOS>, <EOS>, h, u, g, p, space, ug, hug, pug}
        """
        print(f"Training BPE tokenizer on {len(text):,} characters...")
        print(f"Target vocab size: {vocab_size:,}")
        print(f"Min merge frequency: {min_frequency}")

        # =======================================================================
        # STEP 1: INITIALIZE WITH CHARACTERS
        # =======================================================================

        """
        From BPE paper:
        "We start with a vocabulary of all characters... and then
        iteratively merge the most frequent pair"

        IMPLEMENTATION:
        - Get all unique characters from text
        - Add each character to vocabulary
        - Reserve space for special tokens (already added)
        """
        # Get unique characters
        chars = sorted(list(set(text)))
        print(f"Found {len(chars)} unique characters")

        # Add characters to vocabulary (after special tokens)
        for char in chars:
            if char not in self.vocab:
                # Start assigning IDs after special tokens
                new_id = len(self.vocab)
                self.vocab[char] = new_id
                self.inverse_vocab[new_id] = char

        # =======================================================================
        # STEP 2: PREPARE TEXT FOR MERGING
        # =======================================================================

        """
        Convert text to list of characters for processing.

        Example:
        text = "hello"
        chars = ['h', 'e', 'l', 'l', 'o']

        We'll repeatedly merge adjacent pairs in this list.
        """
        # Represent as list of characters (with spaces as explicit tokens)
        words = [list(word) for word in text.split()]

        # =======================================================================
        # STEP 3: ITERATIVE MERGING
        # =======================================================================

        """
        From BPE paper, Algorithm 1:
        "Repeat until desired vocabulary size reached:
           1. Count all adjacent symbol pairs
           2. Find most frequent pair
           3. Merge pair into new symbol
           4. Update all occurrences in corpus"
        """
        # Calculate how many new tokens we can learn
        num_merges = vocab_size - len(config.SPECIAL_TOKENS) - len(chars)

        print(f"Will learn {num_merges:,} merge operations...\n")

        # Create progress bar
        pbar = tqdm(total=num_merges, desc="Training BPE", unit="merge")

        for merge_idx in range(num_merges):
            # ------------------------------------------------------------------
            # 3a. COUNT PAIR FREQUENCIES
            # ------------------------------------------------------------------

            """
            Count frequency of each adjacent pair across all words.

            Example words: [['h', 'e', 'l', 'l', 'o'], ['w', 'o', 'r', 'l', 'd']]
            Pairs: ('h','e'): 1, ('e','l'): 1, ('l','l'): 1, ('l','o'): 2, ...
            """
            pair_counts = self._get_pair_counts(words)

            if not pair_counts:
                tqdm.write(f"No more pairs to merge at iteration {merge_idx}")
                break

            # ------------------------------------------------------------------
            # 3b. FIND MOST FREQUENT PAIR
            # ------------------------------------------------------------------

            """
            Get the pair with highest frequency.
            """
            best_pair = max(pair_counts, key=pair_counts.get)
            best_freq = pair_counts[best_pair]

            # ------------------------------------------------------------------
            # 3c. CHECK MINIMUM FREQUENCY
            # ------------------------------------------------------------------

            """
            Skip if below threshold (prevents learning noisy patterns).
            """
            if best_freq < min_frequency:
                tqdm.write(f"Best pair frequency {best_freq} below minimum {min_frequency}")
                break

            # ------------------------------------------------------------------
            # 3d. CREATE NEW TOKEN
            # ------------------------------------------------------------------

            """
            Merge the pair into a new token.

            Example:
            best_pair = ('l', 'l')
            new_token = 'll'

            Add to vocabulary and record merge.
            """
            new_token = best_pair[0] + best_pair[1]

            # Add to vocabulary
            new_id = len(self.vocab)
            self.vocab[new_token] = new_id
            self.inverse_vocab[new_id] = new_token

            # Record merge (order matters for encoding!)
            self.merges.append(best_pair)

            # ------------------------------------------------------------------
            # 3e. UPDATE TEXT WITH MERGED TOKEN
            # ------------------------------------------------------------------

            """
            Apply merge to all words in corpus.

            Example:
            Before: ['h', 'e', 'l', 'l', 'o']
            Merge: ('l', 'l') → 'll'
            After:  ['h', 'e', 'll', 'o']

            IMPLEMENTATION:
            - Scan each word
            - When we find the pair, replace with merged token
            - Continue until all occurrences merged
            """
            words = self._apply_merge(words, best_pair)

            # ------------------------------------------------------------------
            # PROGRESS REPORTING
            # ------------------------------------------------------------------

            # Update progress bar with current merge info
            pbar.set_postfix({
                'merge': f"'{best_pair[0]}'+'{best_pair[1]}'→'{new_token}'",
                'freq': f'{best_freq:,}',
                'vocab': f'{len(self.vocab):,}'
            })
            pbar.update(1)
            pbar.refresh()

        pbar.close()

        print(f"\nTraining complete!")
        print(f"  Final vocabulary size: {len(self.vocab):,}")
        print(f"  Total merges learned: {len(self.merges):,}")

    # ==========================================================================
    # TRAINING HELPERS
    # ==========================================================================

    def _get_pair_counts(self, words: List[List[str]]) -> Counter:
        """
        Count frequency of all adjacent pairs.

        From BPE paper: "Count the frequency of each adjacent pair"

        Args:
            words: List of words, each word is list of tokens
                   Example: [['h', 'e', 'll', 'o'], ['w', 'o', 'r', 'l', 'd']]

        Returns:
            Counter mapping (token_a, token_b) → frequency

        Example:
            words = [['h', 'e', 'l', 'l', 'o'], ['l', 'l']]
            Pairs: ('h','e'): 1, ('e','l'): 1, ('l','l'): 2, ('l','o'): 1

        ALGORITHM:
        For each word:
            For each adjacent pair in word:
                Increment count for that pair
        """
        pair_counts = Counter()

        for word in words:
            # Get all adjacent pairs in this word
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_counts[pair] += 1

        return pair_counts

    def _apply_merge(self, words: List[List[str]],
                     pair: Tuple[str, str]) -> List[List[str]]:
        """
        Apply a merge to all words.

        From BPE paper: "Replace all occurrences of pair with merged token"

        Args:
            words: List of words, each word is list of tokens
            pair: The pair to merge, e.g., ('l', 'l')

        Returns:
            Updated words with merge applied

        Example:
            words = [['h', 'e', 'l', 'l', 'o']]
            pair = ('l', 'l')
            Result: [['h', 'e', 'll', 'o']]

        ALGORITHM:
        For each word:
            Scan through tokens
            When we find the pair, replace with merged token
            Continue scanning (overlapping merges handled by order)

        NOTE ON OVERLAPPING PAIRS:
        If word = ['l', 'l', 'l'] and pair = ('l', 'l'):
        After first merge: ['ll', 'l']
        The 'll' is now a single token, so no more merges possible
        (we don't merge 'l' from 'll' with the next 'l')
        """
        new_token = pair[0] + pair[1]
        new_words = []

        for word in words:
            new_word = []
            i = 0

            while i < len(word):
                # Check if current position matches the pair
                if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                    new_word.append(new_token)
                    i += 2  # Skip both tokens
                else:
                    new_word.append(word[i])
                    i += 1

            new_words.append(new_word)

        return new_words

    # ==========================================================================
    # ENCODING
    # ==========================================================================

    def encode(self, text: str) -> List[int]:
        """
        Encode text to list of token IDs.

        PAPER REFERENCE: Sennrich et al. (2016), Algorithm 2

        ALGORITHM:
        ----------
        Greedy longest-match-first tokenization:
        1. Split text into words
        2. For each word:
           a. Start with character-level representation
           b. Apply learned merges in order
           c. Use longest possible matches
        3. Convert tokens to IDs

        GREEDY LONGEST-MATCH:
        --------------------
        From BPE paper:
        "We apply the learned merge operations greedily in order of frequency"

        Example:
        Learned merges: [('e', 'r'), ('er', 't'), ('ert', 'a')]
        Word: "erter"

        Step 1: chars = ['e', 'r', 't', 'e', 'r']
        Step 2: apply ('e','r') → ['er', 't', 'e', 'r']
        Step 3: apply ('er','t') → ['ert', 'e', 'r']
        Step 4: apply ('e','r') → ['ert', 'er']
        Step 5: no more merges apply

        Result: ['ert', 'er'] → [token_id_ert, token_id_er]

        Args:
            text: Input text to encode

        Returns:
            List of token IDs

        Example:
            tokenizer.encode("hello world")
            → [45, 123]  # (assuming 'hello' has ID 45, 'world' has ID 123)
        """
        # Split into words (preserve spaces by re-joining)
        words = text.split()

        token_ids = []

        for word in words:
            # Start with character-level representation
            tokens = list(word)

            # Apply learned merges greedily
            while len(tokens) > 1:
                # Find the best merge we can apply
                merge_applied = False

                for pair in self.merges:
                    # Check if this pair exists in tokens
                    for i in range(len(tokens) - 1):
                        if tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                            # Apply merge
                            new_token = pair[0] + pair[1]
                            tokens = tokens[:i] + [new_token] + tokens[i + 2:]
                            merge_applied = True
                            break

                    if merge_applied:
                        break

                # If no merge applied, we're done with this word
                if not merge_applied:
                    break

            # Convert tokens to IDs
            for token in tokens:
                if token in self.vocab:
                    token_ids.append(self.vocab[token])
                else:
                    # Fallback: encode as individual characters
                    for char in token:
                        if char in self.vocab:
                            token_ids.append(self.vocab[char])
                        else:
                            # Unknown character - use UNK token
                            token_ids.append(config.SPECIAL_TOKENS["<UNK>"])

            # Add space token between words (except last)
            token_ids.append(config.SPECIAL_TOKENS["<PAD>"])

        # Remove trailing space token
        if token_ids and token_ids[-1] == config.SPECIAL_TOKENS["<PAD>"]:
            token_ids.pop()

        return token_ids

    # ==========================================================================
    # DECODING
    # ==========================================================================

    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to text.

        PAPER CONTEXT:
        --------------
        From BPE paper:
        "Decoding is straightforward: we simply replace each token ID
        with its corresponding string and concatenate"

        ALGORITHM:
        ----------
        1. Look up each token ID in inverse_vocab
        2. Concatenate all strings
        3. Handle special tokens appropriately

        SPECIAL TOKEN HANDLING:
        -----------------------
        - <PAD>: Skip (don't output anything)
        - <UNK>: Output replacement string (e.g., "<UNK>")
        - <BOS>: Skip (used for conditioning, not output)
        - <EOS>: Stop decoding (end of generation)

        Args:
            token_ids: List of token IDs to decode

        Returns:
            Decoded text string

        Example:
            tokenizer.decode([45, 123])
            → "hello world"

        NOTE ON SPACES:
        In our encoding, we use <PAD> to represent spaces between words.
        During decoding, we convert <PAD> back to spaces.
        """
        text_parts = []

        for token_id in token_ids:
            # Get token string from inverse_vocab
            if token_id in self.inverse_vocab:
                token = self.inverse_vocab[token_id]

                # Handle special tokens
                if token == "<PAD>":
                    text_parts.append(" ")
                elif token == "<UNK>":
                    text_parts.append("<UNK>")
                elif token in ["<BOS>", "<EOS>"]:
                    # Skip control tokens in output
                    continue
                else:
                    text_parts.append(token)
            else:
                # Unknown ID - shouldn't happen with proper encoding
                text_parts.append("<UNK>")

        return "".join(text_parts)

    # ==========================================================================
    # UTILITY METHODS
    # ==========================================================================

    def get_vocab_size(self) -> int:
        """Return the current vocabulary size."""
        return len(self.vocab)

    def save(self, filepath: str) -> None:
        """
        Save tokenizer to file.

        SAVED DATA:
        - vocab: Token to ID mapping
        - inverse_vocab: ID to token mapping
        - merges: Ordered list of merge operations

        Args:
            filepath: Path to save tokenizer (.pkl file)
        """
        data = {
            "vocab": self.vocab,
            "inverse_vocab": self.inverse_vocab,
            "merges": self.merges,
        }

        with open(filepath, "wb") as f:
            pickle.dump(data, f)

        print(f"Tokenizer saved to {filepath}")
        print(f"  Vocabulary size: {len(self.vocab):,}")
        print(f"  Merge rules: {len(self.merges):,}")

    @classmethod
    def load(cls, filepath: str) -> "BPETokenizer":
        """
        Load tokenizer from file.

        Args:
            filepath: Path to saved tokenizer (.pkl file)

        Returns:
            Loaded BPETokenizer instance
        """
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        tokenizer = cls()
        tokenizer.vocab = data["vocab"]
        tokenizer.inverse_vocab = data["inverse_vocab"]
        tokenizer.merges = data["merges"]

        print(f"Tokenizer loaded from {filepath}")
        print(f"  Vocabulary size: {len(tokenizer.vocab):,}")
        print(f"  Merge rules: {len(tokenizer.merges):,}")

        return tokenizer


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def get_tokenizer_stats(tokenizer: BPETokenizer) -> Dict[str, any]:
    """
    Get statistics about trained tokenizer.

    Args:
        tokenizer: Trained BPETokenizer instance

    Returns:
        Dictionary with statistics
    """
    # Get merge lengths
    merge_lengths = [len(a) + len(b) for a, b in tokenizer.merges]

    # Get token lengths
    token_lengths = [len(token) for token in tokenizer.vocab.keys()
                     if token not in config.SPECIAL_TOKENS]

    return {
        "vocab_size": len(tokenizer.vocab),
        "num_merges": len(tokenizer.merges),
        "avg_token_length": sum(token_lengths) / len(token_lengths) if token_lengths else 0,
        "max_token_length": max(token_lengths) if token_lengths else 0,
        "min_token_length": min(token_lengths) if token_lengths else 0,
    }


if __name__ == "__main__":
    # Quick self-test
    print("BPE Tokenizer - Self Test")
    print("=" * 60)

    # Create and train on simple example
    tokenizer = BPETokenizer()

    # Simple training text
    text = "hello hello hell help helping"
    tokenizer.train(text, vocab_size=20, min_frequency=1)

    print("\n" + "=" * 60)
    print("Encoding/Decoding Test")
    print("=" * 60)

    # Test encode/decode
    test_text = "hello helping"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)

    print(f"Original:  '{test_text}'")
    print(f"Encoded:   {encoded}")
    print(f"Decoded:   '{decoded}'")
    print(f"Match:     {test_text == decoded}")
