# ruff: noqa
"""
BPE Tokenizer (load-only) for LoRA Fine-Tuning

Same encoding half as project4_pretrain/tokenizer.py, minus the trainer.
LoRA never learns new merges: it loads the checkpoint Project 4 trained
(checkpoints/project4/tokenizer.pkl, 16,384 vocab, chapter 2 checkpoint
format {'vocab', 'inverse_vocab', 'merges'}) and encodes Alpaca examples
with it. Verbatim copy of project 5's load-only tokenizer (same
self-contained-project convention; the class docstring is updated for the
LoRA context).

PAPERS:
-------
- "Byte-Pair Encoding of Subword Units" (Sennrich, Haddow, Birch, 2016)
- "Language Models are Unsupervised Multitask Learners" (GPT-2, Radford
  et al., 2019): byte-level BPE

WHAT IS KEPT, WHAT IS DROPPED:
------------------------------
- encode(): rank-based, memoized per distinct word, with the chapter 2
  trailing-space convention ("the " is one token). Unchanged.
- decode(): drops <PAD>/<BOS>/<EOS>, marks <UNK>. Unchanged.
- train(): DROPPED. Fine-tuning must not touch the vocabulary - a new
  token would resize the embedding table the base checkpoint owns.

ADDED for project 5: encode_words(), the word-level half of encode()
exposed as a list operation (template.py builds response masks from it).
For a word list [w1, ..., wn], every word except the last encodes with a
trailing space - the same treatment words get inside encode() when they
are not the final word of the text. encode_words(prompt_words, False) +
encode_words(response_words, True) == encode(prompt_text + " " + response)
exactly, because encoding is per-word and context-free; test_model.py
checks that identity.

LOADING uses the same restricted unpickler as project 4: the pickle
format contains only built-in types, and find_class() is refused so a
tampered checkpoint cannot execute code on load.
"""

import io
import pathlib
import pickle
from typing import Dict, List, Tuple

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
    Load-only BPE tokenizer, checkpoint-compatible with chapters 2 and 4.

    Encoding is the standard "merge the lowest-rank adjacent pair" loop
    (GPT-2's encoder.py / tiktoken), memoized per distinct word: repeatedly
    merge the adjacent pair with the smallest merge rank until no adjacent
    pair is in the merge table. Equivalent to applying merge rules in
    learned order.
    """

    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        # pair -> merge index; the encode-time lookup table
        self.ranks: Dict[Tuple[str, str], int] = {}

    # ==========================================================================
    # LOADING
    # ==========================================================================

    @classmethod
    def load(cls) -> "BPETokenizer":
        """
        Load the tokenizer Project 4 trained (config.TOKENIZER_FILE).

        Works with checkpoints saved by the chapter 2 and project 4
        trainers (same three keys). Loading refuses any non-builtin type,
        so a tampered checkpoint cannot execute code on load.
        """
        path = pathlib.Path(config.TOKENIZER_FILE)
        if not path.is_file():
            raise FileNotFoundError(
                f"No tokenizer checkpoint at {path}. Train it once with "
                f"phase1_foundation/project4_pretrain/train_tokenizer.py."
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

    def encode_words(self, words: List[str], final: bool) -> List[int]:
        """
        Encode a word list with the trailing-space convention applied
        explicitly.

        final=False: every word gets a trailing space (they all sit before
        more words in the full text). final=True: only the last word does
        not. This is the exact treatment encode() gives words by position;
        template.py needs it piecewise to know where the response begins.
        """
        token_ids: List[int] = []
        for i, w in enumerate(words):
            if not w:
                continue
            word = w + " " if (final is False or i < len(words) - 1) else w
            token_ids.extend(self._ids_for_symbols(self._encode_word(word)))
        return token_ids

    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.

        Same normalization and trailing-space convention as training time
        (chapter 2): whitespace collapses to single spaces, every word
        except the last carries a trailing space, per-word results are
        memoized.
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
