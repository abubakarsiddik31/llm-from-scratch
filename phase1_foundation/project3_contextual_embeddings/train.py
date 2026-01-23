# ruff: noqa
"""
Training Script for BERT-Style Contextual Embeddings

This script implements MLM (Masked Language Modeling) pre-training following:
- "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2018)

PRE-TRAINING OBJECTIVE:
-----------------------
From BERT paper:
"We use two pre-training tasks: Masked Language Modeling (MLM) and
Next Sentence Prediction (NSP)"

MLM PROCEDURE:
--------------
1. Take input sequence
2. Randomly mask 15% of tokens (replace with [MASK])
3. For each masked token:
   - 80%: Replace with [MASK] token
   - 10%: Replace with random token
   - 10%: Keep original token
4. Model predicts original tokens at masked positions
5. Cross-entropy loss on masked positions only

NSP PROCEDURE:
--------------
1. Take two sentences (A, B)
2. 50% of the time: B is actual next sentence (positive)
3. 50% of the time: B is random sentence (negative)
4. Model predicts: IsNext (1) or NotNext (0)
5. Binary cross-entropy loss

USAGE:
------
uv run python phase1_foundation/project3_contextual_embeddings/train.py
"""

import os
import pickle
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

import config
from model import BERT


# =============================================================================
# DATA LOADING
# =============================================================================


def load_tokenizer(tokenizer_path: str) -> dict:
    """
    Load trained BPE tokenizer from Project 2.

    Args:
        tokenizer_path: Path to tokenizer checkpoint (.pkl)

    Returns:
        Dictionary with vocab, inverse_vocab, merges
    """
    with open(tokenizer_path, "rb") as f:
        data = pickle.load(f)

    return data


def load_text_data(data_path: str) -> str:
    """
    Load training text data.

    Args:
        data_path: Path to text file

    Returns:
        Text content as string
    """
    print(f"Loading data from {data_path}...")

    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Loaded {len(text):,} characters")
    return text


def text_to_token_ids(text: str, tokenizer: dict, max_length: int = 512) -> List[int]:
    """
    Convert text to token IDs using BPE tokenizer.

    Simplified encoding - uses character-level as fallback.

    Args:
        text: Input text
        tokenizer: Tokenizer dictionary
        max_length: Maximum sequence length

    Returns:
        List of token IDs
    """
    vocab = tokenizer["vocab"]
    unk_id = tokenizer["vocab"].get("[UNK]", 1)

    # Simple word-level tokenization with BPE vocab lookup
    # For simplicity, we use word-based encoding here
    # In production, you'd use the full BPE encoder

    token_ids = []
    for word in text.split():
        # Try to find word in vocab
        if word in vocab:
            token_ids.append(vocab[word])
        else:
            # Fallback: character-level encoding
            for char in word:
                if char in vocab:
                    token_ids.append(vocab[char])
                else:
                    token_ids.append(unk_id)
            token_ids.append(vocab.get(" ", 0))  # Space

    return token_ids[:max_length]


# =============================================================================
# MASKED LANGUAGE MODELING
# =============================================================================


def mask_tokens(
    token_ids: List[int],
    tokenizer_vocab: dict,
    mask_prob: float = 0.15,
    mask_token_id: int = 4,
    pad_token_id: int = 0,
    unk_token_id: int = 1,
    random_prob: float = 0.1,
    keep_prob: float = 0.1,
) -> Tuple[List[int], List[int]]:
    """
    Mask tokens for MLM training following BERT paper.

    PAPER REFERENCE: BERT (2018), Section 3.1

    ALGORITHM:
    -----------
    "We replace 15% of all tokens in the input sequence with a [MASK] token
    in the following way:
    - 80% of the time: Replace with [MASK]
    - 10% of the time: Replace with a random token
    - 10% of the time: Keep the original token"

    Args:
        token_ids: Original token IDs
        tokenizer_vocab: Vocabulary dictionary
        mask_prob: Fraction of tokens to mask (default 0.15)
        mask_token_id: ID of [MASK] token
        pad_token_id: ID of [PAD] token
        unk_token_id: ID of [UNK] token
        random_prob: Probability of random token replacement
        keep_prob: Probability of keeping original token

    Returns:
        masked_token_ids: Token IDs with masking applied
        labels: Original token IDs for masked positions (others = -100)

    Example:
        Input: "The cat sat on the mat"
        After masking: "The [MASK] sat on the [MASK]"
        Labels: [-100, cat, -100, -100, -100, mat, -100]
    """
    # Get vocab size for random token sampling
    vocab_size = len(tokenizer_vocab)

    # Create labels (initialize with -100 for ignore_index)
    labels = [-100] * len(token_ids)

    # Create masked copy
    masked_token_ids = token_ids.copy()

    # Get indices of tokens that can be masked
    # Don't mask special tokens ([CLS], [SEP], [PAD])
    special_tokens = {0, 1, 2, 3, 4}  # [PAD], [UNK], [CLS], [SEP], [MASK]
    valid_indices = [
        i for i, token_id in enumerate(token_ids)
        if token_id not in special_tokens
    ]

    # Calculate number of tokens to mask
    num_to_mask = max(1, int(len(valid_indices) * mask_prob))

    # Randomly select tokens to mask
    if len(valid_indices) > 0:
        mask_indices = np.random.choice(
            valid_indices,
            size=min(num_to_mask, len(valid_indices)),
            replace=False
        )

        for idx in mask_indices:
            original_token = token_ids[idx]

            # Store original token in labels
            labels[idx] = original_token

            # Apply masking strategy
            rand = np.random.random()

            if rand < config.MASK_PROB:
                # 80%: Replace with [MASK]
                masked_token_ids[idx] = mask_token_id
            elif rand < config.MASK_PROB + config.RANDOM_PROB:
                # 10%: Replace with random token
                # Don't replace with special tokens
                random_token = np.random.randint(5, vocab_size)
                masked_token_ids[idx] = random_token
            else:
                # 10%: Keep original token
                masked_token_ids[idx] = original_token

    return masked_token_ids, labels


# =============================================================================
# NEXT SENTENCE PREDICTION
# =============================================================================


def create_nsp_pairs(
    sentences: List[str],
    num_pairs: int,
    tokenizer: dict,
    max_length: int = 128,
) -> Tuple[List[List[int]], List[int], List[List[int]]]:
    """
    Create sentence pairs for NSP training.

    PAPER REFERENCE: BERT (2018), Section 3.1

    ALGORITHM:
    -----------
    "For sentence pairs, we construct binary classification task:
    - 50% of pairs: Sentence B is actual next sentence (positive)
    - 50% of pairs: Sentence B is random sentence (negative)"

    Args:
        sentences: List of sentences
        num_pairs: Number of pairs to create
        tokenizer: Tokenizer dictionary
        max_length: Maximum sequence length

    Returns:
        input_ids: List of token ID pairs
        segment_ids: List of segment ID pairs
        nsp_labels: List of NSP labels (0 or 1)
    """
    input_ids = []
    segment_ids = []
    nsp_labels = []

    # Special token IDs
    cls_id = tokenizer["vocab"].get("[CLS]", 2)
    sep_id = tokenizer["vocab"].get("[SEP]", 3)

    for _ in range(num_pairs):
        # Randomly select two sentences
        idx_a = np.random.randint(0, len(sentences))
        idx_b = np.random.randint(0, len(sentences))

        sent_a = sentences[idx_a]
        sent_b = sentences[idx_b]

        # 50% chance of using actual next sentence (positive)
        # 50% chance of using random sentence (negative)
        is_next = np.random.random() < 0.5

        if is_next and idx_a + 1 < len(sentences):
            # Use actual next sentence
            sent_b = sentences[idx_a + 1]
            label = 1  # IsNext
        else:
            # Use random sentence
            label = 0  # NotNext

        # Tokenize both sentences
        tokens_a = text_to_token_ids(sent_a, tokenizer, max_length // 2 - 2)
        tokens_b = text_to_token_ids(sent_b, tokenizer, max_length // 2 - 2)

        # Truncate if necessary
        total_len = len(tokens_a) + len(tokens_b) + 2  # +2 for [CLS] and [SEP]
        if total_len > max_length:
            # Truncate proportionally
            max_a = (max_length - 2) // 2
            max_b = max_length - 2 - max_a
            tokens_a = tokens_a[:max_a]
            tokens_b = tokens_b[:max_b]

        # Construct input: [CLS] sent_a [SEP] sent_b [SEP]
        input_pair = [cls_id] + tokens_a + [sep_id] + tokens_b + [sep_id]

        # Pad to max_length
        input_pair += [0] * (max_length - len(input_pair))

        # Segment IDs: 0 for sentence A, 1 for sentence B
        seg_a = [0] * (len(tokens_a) + 2)  # +2 for [CLS] and [SEP]
        seg_b = [1] * (len(tokens_b) + 1)  # +1 for final [SEP]
        segment_pair = seg_a + seg_b + [0] * (max_length - len(seg_a) - len(seg_b))

        input_ids.append(input_pair)
        segment_ids.append(segment_pair)
        nsp_labels.append(label)

    return input_ids, segment_ids, nsp_labels


# =============================================================================
# DATASET
# =============================================================================


@dataclass
class BERTDataset:
    """
    Dataset for BERT pre-training.

    Handles both MLM and NSP tasks.
    """
    input_ids: List[List[int]]
    segment_ids: List[List[int]]
    masked_labels: List[List[int]]
    nsp_labels: List[int]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.input_ids[idx]),
            torch.tensor(self.segment_ids[idx]),
            torch.tensor(self.masked_labels[idx]),
            torch.tensor(self.nsp_labels[idx]),
        )


def create_dataloader(dataset: BERTDataset, batch_size: int, shuffle: bool = True):
    """
    Create a simple dataloader (without torch.utils.data.DataLoader).

    Args:
        dataset: BERTDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data

    Returns:
        Generator that yields batches
    """
    indices = list(range(len(dataset)))

    if shuffle:
        np.random.shuffle(indices)

    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i + batch_size]

        batch_input_ids = []
        batch_segment_ids = []
        batch_masked_labels = []
        batch_nsp_labels = []

        for idx in batch_indices:
            input_id, seg_id, mask_label, nsp_label = dataset[idx]
            batch_input_ids.append(input_id)
            batch_segment_ids.append(seg_id)
            batch_masked_labels.append(mask_label)
            batch_nsp_labels.append(nsp_label)

        yield (
            torch.stack(batch_input_ids),
            torch.stack(batch_segment_ids),
            torch.stack(batch_masked_labels),
            torch.stack(batch_nsp_labels),
        )


# =============================================================================
# TRAINING
# =============================================================================


def estimate_loss(model: BERT, dataset: BERTDataset, eval_iters: int = 100):
    """
    Estimate loss on validation set.

    Args:
        model: BERT model
        dataset: Validation dataset
        eval_iters: Number of iterations to evaluate

    Returns:
        Dictionary with MLM and NSP losses
    """
    model.eval()
    losses_mlm = []
    losses_nsp = []

    dataloader = create_dataloader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    with torch.no_grad():
        for i, (input_ids, segment_ids, masked_labels, nsp_labels) in enumerate(dataloader):
            if i >= eval_iters:
                break

            input_ids = input_ids.to(config.DEVICE)
            segment_ids = segment_ids.to(config.DEVICE)
            masked_labels = masked_labels.to(config.DEVICE)
            nsp_labels = nsp_labels.to(config.DEVICE)

            _, _, mlm_loss, nsp_loss = model(
                input_ids, segment_ids, masked_labels, nsp_labels
            )

            losses_mlm.append(mlm_loss.item())
            if nsp_loss is not None:
                losses_nsp.append(nsp_loss.item())

    model.train()

    return {
        "mlm": np.mean(losses_mlm) if losses_mlm else float("inf"),
        "nsp": np.mean(losses_nsp) if losses_nsp else float("inf"),
    }


def train(model: BERT, dataset: BERTDataset, val_dataset: BERTDataset):
    """
    Train BERT model with MLM and NSP objectives.

    PAPER REFERENCE: BERT (2018)

    OPTIMIZATION:
    --------------
    "We use the Adam optimizer with a learning rate of 1e-4...
    We use a linear decay learning rate schedule with warm-up"

    Args:
        model: BERT model
        dataset: Training dataset
        val_dataset: Validation dataset
    """
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    # Training loop
    iter_num = 0
    best_val_loss = float("inf")

    model.train()
    t0 = time.time()

    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    # Create progress bar
    pbar = tqdm(total=config.MAX_ITERS, desc="Training", unit="iter")

    # Training loop - recreate dataloader each epoch
    while iter_num < config.MAX_ITERS:
        # Create new dataloader for each epoch
        dataloader = create_dataloader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

        for input_ids, segment_ids, masked_labels, nsp_labels in dataloader:
            if iter_num >= config.MAX_ITERS:
                break

            # Move to device
            input_ids = input_ids.to(config.DEVICE)
            segment_ids = segment_ids.to(config.DEVICE)
            masked_labels = masked_labels.to(config.DEVICE)
            nsp_labels = nsp_labels.to(config.DEVICE)

            # Forward pass
            _, _, mlm_loss, nsp_loss = model(
                input_ids, segment_ids, masked_labels, nsp_labels
            )

            # Combine losses
            loss = mlm_loss
            if nsp_loss is not None:
                loss = loss + nsp_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update progress bar with current losses
            pbar.set_postfix({
                'MLM': f'{mlm_loss.item():.4f}',
                'NSP': f'{nsp_loss.item() if nsp_loss is not None else 0:.4f}',
                'Best': f'{best_val_loss:.4f}'
            })
            pbar.update(1)
            pbar.refresh()  # Force refresh the progress bar

            # Logging and evaluation at intervals
            if iter_num % config.EVAL_INTERVAL == 0 and iter_num > 0:
                t1 = time.time()

                # Pause progress bar for evaluation output
                pbar.close()

                # Evaluate on validation set
                print(f"\n{'='*60}")
                print(f"Evaluating at iteration {iter_num}...")
                val_losses = estimate_loss(model, val_dataset, eval_iters=config.EVAL_ITERS)
                val_loss = val_losses["mlm"]
                if val_losses["nsp"] != float("inf"):
                    val_loss += val_losses["nsp"]

                print(f"Iter {iter_num:6d}/{config.MAX_ITERS}")
                print(f"  Train loss (MLM): {mlm_loss.item():.4f}")
                if nsp_loss is not None:
                    print(f"  Train loss (NSP): {nsp_loss.item():.4f}")
                print(f"  Val loss (MLM):     {val_losses['mlm']:.4f}")
                if val_losses["nsp"] != float("inf"):
                    print(f"  Val loss (NSP):     {val_losses['nsp']:.4f}")
                print(f"  Time:               {t1 - t0:.2f}s")

                # Save checkpoint
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(model, optimizer, iter_num, val_loss)
                    print(f"  ✓ Saved checkpoint (new best val loss: {val_loss:.4f})")

                print(f"{'='*60}\n")

                # Recreate progress bar
                pbar = tqdm(total=config.MAX_ITERS, desc="Training", unit="iter", initial=iter_num)
                t0 = time.time()

            iter_num += 1

    pbar.close()
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


def save_checkpoint(model, optimizer, iter_num, loss):
    """
    Save model checkpoint.

    Args:
        model: BERT model
        optimizer: Optimizer
        iter_num: Current iteration
        loss: Current loss
    """
    checkpoint_path = os.path.join(
        config.CHECKPOINT_DIR,
        f"bert_iter{iter_num}_loss{loss:.4f}.pt"
    )

    torch.save({
        "iter": iter_num,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "config": {
            "vocab_size": config.VOCAB_SIZE,
            "n_embd": config.N_EMBD,
            "n_head": config.N_HEAD,
            "n_layer": config.N_LAYER,
        }
    }, checkpoint_path)


# =============================================================================
# MAIN
# =============================================================================


def main():
    """
    Main training function.
    """
    print("=" * 60)
    print("BERT-STYLE CONTEXTUAL EMBEDDINGS - TRAINING")
    print("=" * 60)

    # Validate and print config
    config.validate_config()

    # =======================================================================
    # LOAD TOKENIZER
    # =======================================================================

    tokenizer_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "checkpoints",
        "project2",
        "tokenizer.pkl"
    )

    if not os.path.exists(tokenizer_path):
        print(f"\n⚠ Warning: Tokenizer not found at {tokenizer_path}")
        print("Using character-level vocabulary...")

        # Create simple character vocab
        text = load_text_data(config.TRAIN_DATA_PATH)
        vocab_size = len(set(text)) + len(config.SPECIAL_TOKENS)
        config.VOCAB_SIZE = vocab_size

        # Create fake tokenizer for compatibility
        tokenizer = {
            "vocab": {str(i): i for i in range(vocab_size)},
            "inverse_vocab": {i: str(i) for i in range(vocab_size)},
            "merges": []
        }
    else:
        tokenizer = load_tokenizer(tokenizer_path)
        config.VOCAB_SIZE = len(tokenizer["vocab"])

    config.print_config()

    # =======================================================================
    # LOAD DATA
    # =======================================================================

    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    # Load training data
    if not os.path.exists(config.TRAIN_DATA_PATH):
        print(f"\n⚠ Warning: Training data not found at {config.TRAIN_DATA_PATH}")
        print("Using Shakespeare data as fallback...")
        config.TRAIN_DATA_PATH = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data",
            "shakespeare.txt"
        )

    train_text = load_text_data(config.TRAIN_DATA_PATH)

    # Create sentences (simple split by period)
    train_sentences = [s.strip() for s in train_text.split(".") if len(s.strip()) > 10]

    print(f"Created {len(train_sentences)} training sentences")

    # Create validation split (90/10)
    val_sentences = train_sentences[:int(len(train_sentences) * 0.1)]
    train_sentences = train_sentences[int(len(train_sentences) * 0.1):]

    print(f"Training sentences: {len(train_sentences)}")
    print(f"Validation sentences: {len(val_sentences)}")

    # =======================================================================
    # CREATE DATASETS
    # =======================================================================

    print("\n" + "=" * 60)
    print("CREATING DATASETS")
    print("=" * 60)

    num_train_pairs = 10000
    num_val_pairs = 1000

    train_input_ids, train_segment_ids, train_nsp_labels = create_nsp_pairs(
        train_sentences, num_train_pairs, tokenizer, config.BLOCK_SIZE
    )

    val_input_ids, val_segment_ids, val_nsp_labels = create_nsp_pairs(
        val_sentences, num_val_pairs, tokenizer, config.BLOCK_SIZE
    )

    # Apply MLM masking
    train_masked_labels = []
    for input_ids in train_input_ids:
        masked, labels = mask_tokens(
            input_ids,
            tokenizer["vocab"],
            config.MLM_PROB,
            tokenizer["vocab"].get("ে", 4),
        )
        train_input_ids[train_input_ids.index(input_ids)] = masked
        train_masked_labels.append(labels)

    val_masked_labels = []
    for input_ids in val_input_ids:
        masked, labels = mask_tokens(
            input_ids,
            tokenizer["vocab"],
            config.MLM_PROB,
            tokenizer["vocab"].get("ে", 4),
        )
        val_input_ids[val_input_ids.index(input_ids)] = masked
        val_masked_labels.append(labels)

    train_dataset = BERTDataset(
        train_input_ids, train_segment_ids, train_masked_labels, train_nsp_labels
    )
    val_dataset = BERTDataset(
        val_input_ids, val_segment_ids, val_masked_labels, val_nsp_labels
    )

    print(f"Training pairs: {len(train_dataset)}")
    print(f"Validation pairs: {len(val_dataset)}")

    # =======================================================================
    # CREATE MODEL
    # =======================================================================

    print("\n" + "=" * 60)
    print("CREATING MODEL")
    print("=" * 60)

    model = BERT(config)
    model = model.to(config.DEVICE)

    # =======================================================================
    # TRAIN
    # =======================================================================

    train(model, train_dataset, val_dataset)

    print("\n✓ Training complete!")
    print(f"Checkpoints saved to: {config.CHECKPOINT_DIR}")


if __name__ == "__main__":
    main()
