# ruff: noqa
"""
Data Loading and Tokenization for SimCSE

Handles the sentence corpus and tokenization pipeline following:
- "SimCSE: Simple Contrastive Learning of Sentence Embeddings"
  (Gao, Yao & Chen, EMNLP 2021)

TRAINING BATCH SHAPE (paper, section 3):
----------------------------------------
A batch of N sentences is encoded TWICE per step:

    view 1:  [CLS] s_1 [SEP] ... [CLS] s_N [SEP]   (dropout masks z)
    view 2:  [CLS] s_1 [SEP] ... [CLS] s_N [SEP]   (dropout masks z')

The two views share the same token ids but produce different embeddings
because dropout is sampled independently. This module builds ONE batch
of token ids; train.py simply calls the encoder on it twice.

TOKENIZATION:
-------------
Uses the Project 2 BPE tokenizer (the same vocabulary the encoder was
pre-trained with). Sequences are formatted BERT-style:

    [CLS] tokens... [SEP] [PAD]...

and truncated to max_len - 2 so the special tokens fit. The paper uses
max sequence length 32 for SimCSE training.

PADDING: each batch is padded to its own longest sequence (not the
global max) for efficiency. An attention mask marks real vs padding
tokens; the encoder's attention itself has no padding mask (Project 3
limitation) but pooling excludes PAD positions.
"""

import pickle
from pathlib import Path
from typing import List, Tuple

import torch

import config


def load_tokenizer(tokenizer_path: str = None) -> dict:
    """
    Load the Project 2 BPE tokenizer checkpoint.

    Args:
        tokenizer_path: Path to tokenizer.pkl (default: config)

    Returns:
        Dict with 'vocab' (token -> id), 'inverse_vocab', 'merges'
    """
    tokenizer_path = tokenizer_path or config.TOKENIZER_PATH
    if not Path(tokenizer_path).exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {tokenizer_path}. Train it first:\n"
            "  uv run python phase1_foundation/project2_tokenizer/train_tokenizer.py"
        )
    with open(tokenizer_path, "rb") as f:
        return pickle.load(f)


def load_sentences(
    sentence_path: str = None,
    max_sentences: int = None,
) -> List[str]:
    """
    Load the one-sentence-per-line training corpus.

    Args:
        sentence_path: Path to simcse_sentences.txt (default: config)
        max_sentences: Cap on number of sentences

    Returns:
        List of sentences
    """
    sentence_path = sentence_path or config.SENTENCE_DATA_PATH
    path = Path(sentence_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Sentence corpus not found at {sentence_path}. Run:\n"
            "  uv run python phase1_foundation/project3b_simcse/download_data.py"
        )

    sentences = [
        line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if max_sentences:
        sentences = sentences[:max_sentences]
    print(f"Loaded {len(sentences):,} sentences from {path}")
    return sentences


class Tokenizer:
    """
    Thin BERT-style wrapper over the Project 2 BPE tokenizer.

    Adds [CLS]/[SEP] framing and truncation to a fixed budget so that
    every input fits the encoder's window with special tokens included.
    """

    def __init__(self, tokenizer_dict: dict, max_len: int = config.MAX_SEQ_LEN):
        self.vocab = tokenizer_dict["vocab"]
        self.max_len = max_len
        self.cls_id = self.vocab.get("[CLS]", config.CLS_TOKEN_ID)
        self.sep_id = self.vocab.get("[SEP]", config.SEP_TOKEN_ID)
        self.pad_id = self.vocab.get("[PAD]", config.PAD_TOKEN_ID)
        self.unk_id = self.vocab.get("[UNK]", 1)

    def encode(self, sentence: str) -> List[int]:
        """
        Encode a sentence to [CLS] bpe_tokens [SEP], truncated to max_len.

        Falls back to [UNK] for out-of-vocabulary BPE pieces (the BPE
        tokenizer rarely produces them because it includes base
        characters).

        Args:
            sentence: Raw sentence string

        Returns:
            List of token ids, length <= max_len
        """
        ids = self.vocab.get(sentence, None)
        if ids is not None:
            # Whole sentence happened to be a single learned token (rare)
            tokens = [ids]
        else:
            tokens = []
            for piece in sentence.split():
                if piece in self.vocab:
                    tokens.append(self.vocab[piece])
                else:
                    # Character-level fallback: encode word as its chars
                    for ch in piece:
                        tokens.append(self.vocab.get(ch, self.unk_id))
        # Truncate to leave room for [CLS] and [SEP]
        return [self.cls_id] + tokens[: self.max_len - 2] + [self.sep_id]


def tokenize_batch(sentences: List[str], tokenizer=None, max_len: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Tokenize a list of sentences into padded tensors.

    Args:
        sentences: List of raw sentences
        tokenizer: Optional shared Tokenizer (created lazily otherwise)
        max_len: Max sequence length including special tokens

    Returns:
        (idx, attention_mask):
        idx            (B, T) long tensor of token ids
        attention_mask (B, T) long tensor, 1 = real token, 0 = [PAD]
    """
    tokenizer = tokenizer or _get_default_tokenizer(max_len)
    max_len = max_len or config.MAX_SEQ_LEN

    encoded = [tokenizer.encode(s) for s in sentences]
    batch_max = min(max(len(seq) for seq in encoded), max_len)

    idx = torch.full((len(encoded), batch_max), tokenizer.pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(encoded), batch_max), dtype=torch.long)

    for i, seq in enumerate(encoded):
        seq = seq[:batch_max]
        idx[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        attention_mask[i, : len(seq)] = 1

    return idx, attention_mask


_DEFAULT_TOKENIZER = None


def _get_default_tokenizer(max_len: int = None) -> Tokenizer:
    """Lazily create a shared Tokenizer instance."""
    global _DEFAULT_TOKENIZER
    if _DEFAULT_TOKENIZER is None:
        _DEFAULT_TOKENIZER = Tokenizer(load_tokenizer(), max_len or config.MAX_SEQ_LEN)
    return _DEFAULT_TOKENIZER
