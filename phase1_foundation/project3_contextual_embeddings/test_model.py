# ruff: noqa
"""
Test Script for BERT-Style Contextual Embeddings

This script validates the BERT model implementation with:
1. Model architecture tests
2. MLM prediction tests
3. NSP prediction tests
4. Contextual embedding extraction

USAGE:
------
uv run python phase1_foundation/project3_contextual_embeddings/test_model.py
"""

import os
import pickle
import sys

import torch

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import config
from model import BERT
from train import mask_tokens, text_to_token_ids, load_tokenizer


# =============================================================================
# MODEL ARCHITECTURE TESTS
# =============================================================================


def test_model_creation():
    """
    Test that model can be created and has correct architecture.

    PAPER REFERENCE: BERT (2018), Section 3

    From BERT paper:
    "BERT-base: 12 layers, 768 hidden, 12 heads
     BERT-large: 24 layers, 1024 hidden, 16 heads"
    """
    print("\n" + "=" * 60)
    print("TEST 1: Model Creation")
    print("=" * 60)

    # Set small vocab for testing
    config.VOCAB_SIZE = 100
    config.N_EMBD = 128
    config.N_HEAD = 4
    config.N_LAYER = 2

    model = BERT(config)

    # Check parameter count
    num_params = model.get_num_params()
    print(f"✓ Model created with {num_params:,} parameters")

    # Verify components
    assert hasattr(model, "embeddings"), "Missing embeddings"
    assert hasattr(model, "blocks"), "Missing transformer blocks"
    assert hasattr(model, "mlm_head"), "Missing MLM head"
    assert hasattr(model, "nsp_head"), "Missing NSP head"

    print("✓ All model components present")

    return model


def test_forward_pass(model):
    """
    Test forward pass with various input shapes.

    From BERT paper:
    "Input: [CLS] Sentence A [SEP] Sentence B [SEP]"
    """
    print("\n" + "=" * 60)
    print("TEST 2: Forward Pass")
    print("=" * 60)

    B, T = 4, 32

    # Create random input
    idx = torch.randint(0, config.VOCAB_SIZE, (B, T))
    segment_ids = torch.zeros(B, T, dtype=torch.long)

    # Test without labels
    mlm_logits, nsp_logits, mlm_loss, nsp_loss = model(idx, segment_ids)

    assert mlm_logits.shape == (B, T, config.VOCAB_SIZE), \
        f"Expected MLM logits shape {(B, T, config.VOCAB_SIZE)}, got {mlm_logits.shape}"
    print(f"✓ MLM logits shape correct: {mlm_logits.shape}")

    if nsp_logits is not None:
        assert nsp_logits.shape == (B, 2), \
            f"Expected NSP logits shape {(B, 2)}, got {nsp_logits.shape}"
        print(f"✓ NSP logits shape correct: {nsp_logits.shape}")

    assert mlm_loss is None, "MLM loss should be None without labels"
    assert nsp_loss is None, "NSP loss should be None without labels"
    print("✓ Loss correctly returns None without labels")

    # Test with labels
    masked_labels = torch.randint(0, config.VOCAB_SIZE, (B, T))
    masked_labels[:, :T//2] = -100  # Mask half as non-masked

    nsp_labels = torch.randint(0, 2, (B,))

    mlm_logits, nsp_logits, mlm_loss, nsp_loss = model(
        idx, segment_ids, masked_labels, nsp_labels
    )

    assert mlm_loss is not None, "MLM loss should be computed with labels"
    assert nsp_loss is not None, "NSP loss should be computed with labels"
    print(f"✓ MLM loss computed: {mlm_loss.item():.4f}")
    print(f"✓ NSP loss computed: {nsp_loss.item():.4f}")

    return mlm_logits, nsp_logits


def test_masking():
    """
    Test masking function for MLM.

    PAPER REFERENCE: BERT (2018), Section 3.1

    From BERT paper:
    "80% of the time: replace with enko token
     10% of the time: replace with random token
     10% of the time: keep original token"
    """
    print("\n" + "=" * 60)
    print("TEST 3: MLM Masking")
    print("=" * 60)

    # Create simple vocab
    vocab = {str(i): i for i in range(100)}
    vocab["[PAD]"] = 0
    vocab["[UNK]"] = 1
    vocab["[CLS]"] = 2
    vocab["[SEP]"] = 3
    vocab["ে"] = 4

    # Create test sequence
    token_ids = list(range(5, 25))  # Non-special tokens

    masked_ids, labels = mask_tokens(
        token_ids,
        vocab,
        mask_prob=0.15,
        mask_token_id=4,
    )

    # Count masks
    num_masked = sum(1 for i, l in enumerate(labels) if l != -100)
    print(f"✓ Masked {num_masked} out of {len(token_ids)} tokens " +
          f"({num_masked/len(token_ids)*100:.1f}%)")

    # Verify labels
    for i, (orig, masked, label) in enumerate(zip(token_ids, masked_ids, labels)):
        if label != -100:
            # This position was masked
            assert label == orig, f"Label should be original token at position {i}"

    print("✓ Labels correctly store original tokens")


# =============================================================================
# MLM PREDICTION TESTS
# =============================================================================


def test_mlm_prediction():
    """
    Test MLM prediction on masked sentences.

    Example from BERT paper:
    Input: "The man went to the [MASK] store."
    Predict: "store" → "grocery", "hardware", etc.
    """
    print("\n" + "=" * 60)
    print("TEST 4: MLM Prediction")
    print("=" * 60)

    # Create simple test
    config.VOCAB_SIZE = 100
    config.N_EMBD = 128
    config.N_HEAD = 4
    config.N_LAYER = 2

    model = BERT(config)
    model.eval()

    # Create test input with mask
    B, T = 2, 16
    idx = torch.randint(0, config.VOCAB_SIZE, (B, T))

    # Replace some tokens with mask token
    mask_token_id = 4  # [MASK]
    idx[:, 5] = mask_token_id
    idx[:, 10] = mask_token_id

    segment_ids = torch.zeros(B, T, dtype=torch.long)

    with torch.no_grad():
        mlm_logits, _, _, _ = model(idx, segment_ids)

    # Get predictions for masked positions
    predictions = torch.argmax(mlm_logits, dim=-1)

    print(f"✓ Predictions shape: {predictions.shape}")
    print(f"  Prediction for position 5: {predictions[0, 5].item()}")
    print(f"  Prediction for position 10: {predictions[0, 10].item()}")

    # Verify masked positions have valid predictions
    assert predictions[0, 5].item() < config.VOCAB_SIZE
    assert predictions[0, 10].item() < config.VOCAB_SIZE
    print("✓ MLM predictions are valid token IDs")


# =============================================================================
# NSP PREDICTION TESTS
# =============================================================================


def test_nsp_prediction():
    """
    Test NSP prediction on sentence pairs.

    From BERT paper:
    "Task: Given sentence pair (A, B), predict if B is actual next sentence"
    """
    print("\n" + "=" * 60)
    print("TEST 5: NSP Prediction")
    print("=" * 60)

    config.VOCAB_SIZE = 100
    config.N_EMBD = 128
    config.N_HEAD = 4
    config.N_LAYER = 2

    model = BERT(config)
    model.eval()

    # Create test input
    B, T = 4, 32
    idx = torch.randint(0, config.VOCAB_SIZE, (B, T))
    segment_ids = torch.cat([
        torch.zeros(B, T//2, dtype=torch.long),  # Sentence A
        torch.ones(B, T//2, dtype=torch.long),   # Sentence B
    ], dim=1)

    with torch.no_grad():
        _, nsp_logits, _, _ = model(idx, segment_ids)

    # Get NSP predictions
    predictions = torch.argmax(nsp_logits, dim=-1)
    probs = torch.softmax(nsp_logits, dim=-1)

    print(f"✓ NSP logits shape: {nsp_logits.shape}")
    print(f"✓ Predictions: {predictions.tolist()}")

    for i in range(B):
        print(f"  Sample {i}: NotNext={probs[i, 0].item():.3f}, " +
              f"IsNext={probs[i, 1].item():.3f} → {predictions[i].item()}")

    # Verify predictions are binary
    assert all(p in [0, 1] for p in predictions.tolist())
    print("✓ NSP predictions are binary (0 or 1)")


# =============================================================================
# CONTEXTUAL EMBEDDING EXTRACTION
# =============================================================================


def test_embedding_extraction():
    """
    Test extraction of contextual embeddings.

    From BERT paper:
    "The output of BERT is a contextualized embedding for each token"

    For downstream tasks:
    - Classification: Use [CLS] token output
    - NER: Use token-level outputs
    - QA: Use span prediction on outputs
    """
    print("\n" + "=" * 60)
    print("TEST 6: Contextual Embedding Extraction")
    print("=" * 60)

    config.VOCAB_SIZE = 100
    config.N_EMBD = 128
    config.N_HEAD = 4
    config.N_LAYER = 2

    model = BERT(config)
    model.eval()

    # Create test input
    B, T = 2, 16
    idx = torch.randint(0, config.VOCAB_SIZE, (B, T))
    segment_ids = torch.zeros(B, T, dtype=torch.long)

    # Extract embeddings from the last encoder block
    with torch.no_grad():
        # We need to modify forward to return intermediate outputs
        # For now, we'll use a workaround
        x = model.embeddings.token(idx) + \
            model.embeddings.position(torch.arange(T, device=idx.device)) + \
            model.embeddings.segment(segment_ids)
        x = model.embeddings.drop(x)

        for block in model.blocks:
            x = block(x)

        x = model.ln_f(x)

    print(f"✓ Contextual embeddings shape: {x.shape}")
    print(f"  Batch: {x.shape[0]}")
    print(f"  Sequence: {x.shape[1]}")
    print(f"  Hidden size: {x.shape[2]}")

    # [CLS] token is at position 0
    cls_embeddings = x[:, 0, :]
    print(f"✓ [CLS] embeddings shape: {cls_embeddings.shape}")
    print(f"  (Used for classification tasks)")

    # Token embeddings
    token_embeddings = x[:, 1:T-1, :]  # Exclude [CLS] and [SEP]
    print(f"✓ Token embeddings shape: {token_embeddings.shape}")
    print(f"  (Used for NER, QA, etc.)")


# =============================================================================
# INTERACTIVE TESTS
# =============================================================================


def test_interactive_mlm(model, tokenizer):
    """
    Interactive MLM prediction test.

    Example:
    Input:  "The [MASK] sat on the mat."
    Output: Predictions: "cat" (0.45), "dog" (0.32), "rat" (0.12)...
    """
    print("\n" + "=" * 60)
    print("TEST 7: Interactive MLM (Simulated)")
    print("=" * 60)

    model.eval()

    # Simulate a masked sentence
    test_sentences = [
        [5, 6, 4, 8, 9, 10],  # with mask at position 2
        [10, 11, 12, 4, 14, 15],  # with mask at position 3
    ]

    for sentence in test_sentences:
        idx = torch.tensor([sentence])
        segment_ids = torch.zeros_like(idx)

        with torch.no_grad():
            mlm_logits, _, _, _ = model(idx, segment_ids)

        # Get predictions for masked position
        mask_pos = sentence.index(4)  # [MASK] token
        mask_logits = mlm_logits[0, mask_pos, :]
        probs = torch.softmax(mask_logits, dim=-1)

        top_k = 5
        top_probs, top_ids = torch.topk(probs, top_k)

        print(f"\nSentence: {sentence}")
        print(f"Mask position: {mask_pos}")
        print(f"Top {top_k} predictions:")
        for i in range(top_k):
            print(f"  {i+1}. Token {top_ids[i].item()} " +
                  f"(prob: {top_probs[i].item():.3f})")


# =============================================================================
# MAIN
# =============================================================================


def run_all_tests():
    """
    Run all tests.
    """
    print("=" * 60)
    print("BERT-STYLE CONTEXTUAL EMBEDDINGS - TESTS")
    print("=" * 60)

    try:
        # Test 1: Model creation
        model = test_model_creation()

        # Test 2: Forward pass
        mlm_logits, nsp_logits = test_forward_pass(model)

        # Test 3: Masking
        test_masking()

        # Test 4: MLM prediction
        test_mlm_prediction()

        # Test 5: NSP prediction
        test_nsp_prediction()

        # Test 6: Embedding extraction
        test_embedding_extraction()

        # Test 7: Interactive MLM
        test_interactive_mlm(model, None)

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
