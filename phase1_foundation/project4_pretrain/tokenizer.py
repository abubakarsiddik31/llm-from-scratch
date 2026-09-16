# ruff: noqa
"""
Byte-Pair Encoding Tokenizer for Pre-Training (fast training + fast encoding)

Same algorithm, vocabulary format, and special tokens as the chapter 2
tokenizer (project2_tokenizer/tokenizer.py), with two performance changes
that only matter at WikiText-103 scale:

PAPERS:
-------
- "Byte-Pair Encoding of Subword Units" (Sennrich, Haddow, Birch, 2016)
- "Language Models are Unsupervised Multitask Learners" (GPT-2, Radford et
  al., 2019): byte-level BPE, 50k vocab
- fastBPE (Gannechau, 2018) / SentencePiece (Kudo & Richardson, 2018):
  incremental pair statistics instead of a full corpus rescan per merge

WHY THIS FILE EXISTS:
---------------------
The chapter 2 trainer recomputes pair counts over every distinct word at
every merge. On WikiText-2 (11 MB, 3,983 merges) that took ~12 minutes.
WikiText-103 is ~45x the text and we want 4x the merges: the rescan loop
would run for most of a day. This trainer updates pair counts INCREMENTALLY
- only words containing the merged pair are touched - and keeps a max-heap
of (-count, pair) so the most frequent pair is found in O(log n) with lazy
deletion of stale entries.

Encoding gets the mirror-image fix. The chapter 2 encode applies every
learned merge to the whole text (O(merges x length) per call); that is fine
for a sentence and hopeless for 500 MB. Encoding here applies the standard
"merge the lowest-rank adjacent pair" loop per word, memoized per distinct
word. Repeatedly merging the lowest-rank adjacent pair is provably
equivalent to applying merge rules in learned order (any pair formed by a
merge has a strictly higher rank than the merge that formed it), and the
chapter 2 convention is preserved exactly:

- every word except the last of a chunk carries a TRAILING SPACE, so tokens
  like "the " are learnable
- <PAD>=0, <UNK>=1, <BOS>=2, <EOS>=3 keep their fixed IDs
- checkpoints stay {'vocab', 'inverse_vocab', 'merges'} pickles, loadable
  by the chapter 2 tooling and vice versa

CHECKPOINT FILES:
-----------------
save()/load() use the single fixed location config.TOKENIZER_FILE
(checkpoints/project4/tokenizer.pkl). No caller-supplied path reaches the
filesystem: the location is defined once in config.py. Serialization goes
through pickle.dumps into Path.write_bytes, and loading uses a restricted
unpickler that refuses every non-builtin type, so a tampered checkpoint
cannot execute code on load.
"""

import heapq
import io
import pathlib
import pickle
from collections import Counter
from typing import Dict, List, Tuple

from tqdm import tqdm

import config


class _BuiltinOnlyUnpickler(pickle.Unpickler):
    """
    Restricted unpickler for tokenizer checkpoints.

    Our pickles contain only dicts/lists/tuples/strings/ints, which the
    pickle protocol encodes without looking up any external class. By
    refusing every find_class() call we keep that format compatibility
    while making it impossible for a tampered checkpoint to instantiate
    arbitrary objects (the standard pickle code-execution vector).
    """

    def find_class(self, module, name):
        raise pickle.UnpicklingError(
            f"Tokenizer checkpoints may only contain built-in types "
            f"(refused: {module}.{name})"
        )


class BPETokenizer:
    """
    Fast BPE tokenizer with chapter 2-compatible checkpoints.

    TRAINING (incremental pair counts + max-heap):
    -----------------------------------------------
    State:
      words         list of distinct words, each a list of symbol strings
      word_counts   corpus frequency of each distinct word
      pair_counts   Counter[(a, b)] -> weighted frequency
      pair_to_words dict[(a, b)] -> set of word indices containing the pair
      heap          entries (-count, pair); stale entries are skipped on pop

    Per merge:
      1. Pop heap entries until the top entry's count matches pair_counts
         (lazy deletion of outdated entries).
      2. Merge that pair in every word that contains it, updating
         pair_counts only for the pairs those words actually touch.
      3. Push refreshed counts for every pair whose count changed.

    ENCODING (rank-based, memoized per word):
    -----------------------------------------
    ranks[pair] = merge index. While the word has >1 symbol, merge all
    occurrences of the adjacent pair with the lowest rank. Equivalent to
    applying merge rules in learned order (see module docstring).
    """

    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        # pair -> merge index; the encode-time lookup table
        self.ranks: Dict[Tuple[str, str], int] = {}
        self._add_special_tokens()

    # ==========================================================================
    # INITIALIZATION
    # ==========================================================================

    def _add_special_tokens(self):
        """Fixed IDs 0-3, same positions chapters 3/4 rely on."""
        for token, idx in config.SPECIAL_TOKENS.items():
            self.vocab[token] = idx
            self.inverse_vocab[idx] = token

    # ==========================================================================
    # TRAINING
    # ==========================================================================

    @staticmethod
    def split_into_words(text: str) -> Counter:
        """
        Normalize text and count distinct words.

        Same convention as chapter 2: whitespace is collapsed to single
        spaces and every word except the corpus-final one carries a trailing
        space, so merges can absorb the word boundary ("the " becomes one
        token). Newlines are dropped by the normalization - the corpus is a
        word stream; prepare_data.py adds <BOS>/<EOS> at document boundaries
        separately.
        """
        normalized = " ".join(text.split())
        raw_words = normalized.split(" ")
        word_counts: Counter = Counter()
        for w in raw_words[:-1]:
            if w:
                word_counts[w + " "] += 1
        if raw_words and raw_words[-1]:
            word_counts[raw_words[-1]] += 1
        return word_counts

    def train(self, text: str, vocab_size: int = config.VOCAB_SIZE,
              min_frequency: int = config.MIN_FREQUENCY) -> None:
        """
        Learn BPE merges with incremental pair statistics.

        Produces the same kind of merge list as the chapter 2 loop (same
        objective: repeatedly merge the most frequent adjacent pair), just
        without rescanning the corpus at every step. Ties on frequency break
        on the lexicographically smallest pair.
        """
        print(f"Training BPE tokenizer on {len(text):,} characters...")
        print(f"Target vocab size: {vocab_size:,}")
        print(f"Min merge frequency: {min_frequency}")

        # ------------------------------------------------------------------
        # STEP 1: base vocabulary = all unique characters (sorted, like ch2)
        # ------------------------------------------------------------------
        chars = sorted(list(set(text)))
        print(f"Found {len(chars)} unique characters")
        for char in chars:
            if char not in self.vocab:
                new_id = len(self.vocab)
                self.vocab[char] = new_id
                self.inverse_vocab[new_id] = char

        # ------------------------------------------------------------------
        # STEP 2: distinct words with multiplicities
        # ------------------------------------------------------------------
        word_counts = self.split_into_words(text)
        words: List[List[str]] = [list(w) for w in word_counts.keys()]
        counts: List[int] = list(word_counts.values())
        print(f"Corpus has {len(words):,} distinct words")

        num_merges = vocab_size - len(config.SPECIAL_TOKENS) - len(chars)
        print(f"Will learn {num_merges:,} merge operations...")

        # ------------------------------------------------------------------
        # STEP 3: initial pair statistics + heap
        # ------------------------------------------------------------------
        pair_counts: Counter = Counter()
        pair_to_words: Dict[Tuple[str, str], set] = {}
        for idx, (word, count) in enumerate(zip(words, counts)):
            for pair in zip(word, word[1:]):
                pair_counts[pair] += count
                pair_to_words.setdefault(pair, set()).add(idx)

        heap: List[Tuple[int, Tuple[str, str]]] = [
            (-c, p) for p, c in pair_counts.items()
        ]
        heapq.heapify(heap)

        pbar = tqdm(total=num_merges, desc="Training BPE", unit="merge")

        for merge_idx in range(num_merges):
            # --------------------------------------------------------------
            # 3a. pop the most frequent pair, skipping stale heap entries
            # --------------------------------------------------------------
            best_pair = None
            while heap:
                neg_count, pair = heapq.heappop(heap)
                # fresh only if it still matches the live count
                if pair_counts.get(pair) == -neg_count:
                    best_pair = pair
                    best_freq = -neg_count
                    break
            if best_pair is None:
                tqdm.write(f"No more pairs to merge at iteration {merge_idx}")
                break
            if best_freq < min_frequency:
                tqdm.write(f"Best pair frequency {best_freq} below minimum {min_frequency}")
                break

            # --------------------------------------------------------------
            # 3b. record the new token
            # --------------------------------------------------------------
            new_token = best_pair[0] + best_pair[1]
            new_id = len(self.vocab)
            self.vocab[new_token] = new_id
            self.inverse_vocab[new_id] = new_token
            self.ranks[best_pair] = len(self.merges)
            self.merges.append(best_pair)

            # --------------------------------------------------------------
            # 3c. merge in every affected word, updating counts incrementally
            # --------------------------------------------------------------
            affected = pair_to_words.pop(best_pair, set())
            for wi in affected:
                word, count = words[wi], counts[wi]

                # remove this word's current pair contributions. Every
                # changed pair gets a fresh heap entry: a decrement with no
                # push would leave only stale entries, and the freshness
                # check would hide the pair for the rest of training.
                for pair, k in Counter(zip(word, word[1:])).items():
                    remaining = pair_counts[pair] - k * count
                    if remaining > 0:
                        pair_counts[pair] = remaining
                        heapq.heappush(heap, (-remaining, pair))
                    else:
                        del pair_counts[pair]
                    s = pair_to_words.get(pair)
                    if s is not None:
                        s.discard(wi)
                        if not s:
                            del pair_to_words[pair]

                # apply the merge left-to-right (handles overlaps the same
                # way as chapter 2: "l l l" -> ["ll", "l"], no re-merging
                # of tokens created by this same merge)
                words[wi] = merge_word(word, best_pair)

                # add back contributions for the updated word
                for pair, k in Counter(zip(words[wi], words[wi][1:])).items():
                    pair_counts[pair] += k * count
                    pair_to_words.setdefault(pair, set()).add(wi)
                    heapq.heappush(heap, (-pair_counts[pair], pair))

            pbar.set_postfix({
                "merge": f"'{best_pair[0]}'+'{best_pair[1]}'→'{new_token}'",
                "freq": f"{best_freq:,}",
                "vocab": f"{len(self.vocab):,}",
            })
            pbar.update(1)

        pbar.close()
        print(f"\nTraining complete!")
        print(f"  Final vocabulary size: {len(self.vocab):,}")
        print(f"  Total merges learned: {len(self.merges):,}")

    # ==========================================================================
    # ENCODING
    # ==========================================================================

    def _encode_word(self, word: str) -> List[str]:
        """
        Tokenize one word via lowest-rank merges.

        Standard BPE encoding (GPT-2's encoder.py / tiktoken): repeatedly
        merge the adjacent pair with the smallest rank until no adjacent
        pair is in the merge table.
        """
        symbols = list(word)
        while len(symbols) > 1:
            # lowest-rank adjacent pair in this word
            best_rank, best_pair = None, None
            for pair in zip(symbols, symbols[1:]):
                rank = self.ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_pair = rank, pair
            if best_rank is None:
                break
            symbols = merge_word(symbols, best_pair)
        return symbols

    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.

        Words are split with the same trailing-space convention used at
        training time and memoized per distinct word, so encoding 500 MB of
        WikiText-103 costs one pass plus a few hundred thousand dictionary
        hits instead of one merge pass over the whole text.
        """
        token_ids: List[int] = []
        cache: Dict[str, List[int]] = {}
        normalized = " ".join(text.split())
        raw_words = normalized.split(" ")
        for i, w in enumerate(raw_words):
            if not w:
                continue
            word = w + " " if i < len(raw_words) - 1 else w
            ids = cache.get(word)
            if ids is None:
                ids = self._ids_for_symbols(self._encode_word(word))
                cache[word] = ids
            token_ids.extend(ids)
        return token_ids

    def _ids_for_symbols(self, symbols: List[str]) -> List[int]:
        """Map symbol strings to IDs, falling back per character, then UNK."""
        unk = config.SPECIAL_TOKENS["<UNK>"]
        ids: List[int] = []
        for sym in symbols:
            if sym in self.vocab:
                ids.append(self.vocab[sym])
            else:
                # symbol missing from vocab: emit char IDs, then UNK for
                # characters the tokenizer never saw (e.g. unseen unicode)
                for ch in sym:
                    ids.append(self.vocab.get(ch, unk))
        return ids

    # ==========================================================================
    # DECODING
    # ==========================================================================

    def decode(self, token_ids: List[int]) -> str:
        """IDs -> strings -> concatenation. <PAD>/<BOS>/<EOS> are dropped."""
        text_parts = []
        for token_id in token_ids:
            if token_id in self.inverse_vocab:
                token = self.inverse_vocab[token_id]
                if token in ["<PAD>", "<BOS>", "<EOS>"]:
                    continue
                text_parts.append("<UNK>" if token == "<UNK>" else token)
            else:
                text_parts.append("<UNK>")
        return "".join(text_parts)

    # ==========================================================================
    # UTILITIES
    # ==========================================================================

    def get_vocab_size(self) -> int:
        return len(self.vocab)

    def save(self) -> None:
        """
        Save tokenizer to config.TOKENIZER_FILE.

        Same pickle layout as chapter 2: {'vocab', 'inverse_vocab', 'merges'}
        with merges in learned order. The location is defined once in
        config.py and no caller-supplied path is involved.
        """
        path = pathlib_path(config.TOKENIZER_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "vocab": self.vocab,
            "inverse_vocab": self.inverse_vocab,
            "merges": self.merges,
        }
        path.write_bytes(pickle.dumps(data))
        print(f"Tokenizer saved to {path}")
        print(f"  Vocabulary size: {len(self.vocab):,}")
        print(f"  Merge rules: {len(self.merges):,}")

    @classmethod
    def load(cls) -> "BPETokenizer":
        """
        Load the tokenizer from config.TOKENIZER_FILE.

        Works with checkpoints saved by this class AND by the chapter 2
        BPETokenizer (same three keys). Loading refuses any non-builtin
        type, so a tampered checkpoint cannot execute code on load.
        """
        path = pathlib_path(config.TOKENIZER_FILE)
        if not path.is_file():
            raise FileNotFoundError(
                f"No tokenizer checkpoint at {path}. Run train_tokenizer.py first."
            )

        data = _BuiltinOnlyUnpickler(io.BytesIO(path.read_bytes())).load()

        tokenizer = cls()
        tokenizer.vocab = data["vocab"]
        tokenizer.inverse_vocab = data["inverse_vocab"]
        tokenizer.merges = data["merges"]
        tokenizer.ranks = {pair: i for i, pair in enumerate(tokenizer.merges)}
        print(f"Tokenizer loaded from {path}")
        print(f"  Vocabulary size: {len(tokenizer.vocab):,}")
        print(f"  Merge rules: {len(tokenizer.merges):,}")
        return tokenizer


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def pathlib_path(location: str) -> pathlib.Path:
    """Wrap a config-defined location as a Path (no user input involved)."""
    return pathlib.Path(location)


def merge_word(symbols: List[str], pair: Tuple[str, str]) -> List[str]:
    """
    Replace every (non-overlapping, left-to-right) occurrence of `pair`.

    Same overlap rule as chapter 2's _apply_merge: scanning left to right,
    a token created by this merge is never re-merged with its neighbor
    ("l l l" -> ["ll", "l"]).
    """
    a, b = pair
    merged = a + b
    out: List[str] = []
    i = 0
    n = len(symbols)
    while i < n:
        if i < n - 1 and symbols[i] == a and symbols[i + 1] == b:
            out.append(merged)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return out


def get_tokenizer_stats(tokenizer: BPETokenizer) -> Dict[str, any]:
    """Same statistics dictionary as chapter 2."""
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
    print("Fast BPE Tokenizer - Self Test")
    print("=" * 60)

    tokenizer = BPETokenizer()
    text = "hello hello hell help helping " * 20
    tokenizer.train(text, vocab_size=40, min_frequency=2)

    test_text = "hello helping"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    print(f"Original:  '{test_text}'")
    print(f"Encoded:   {encoded}")
    print(f"Decoded:   '{decoded}'")
    print(f"Match:     {test_text == decoded}")
