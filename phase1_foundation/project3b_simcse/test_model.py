# ruff: noqa
"""
Sanity Tests for the SimCSE Implementation

Validates the core SimCSE components following:
- "SimCSE: Simple Contrastive Learning of Sentence Embeddings"
  (Gao, Yao & Chen, EMNLP 2021)

Tests are deliberately small and self-contained: they check the
MECHANICS of the objective (positives closer than negatives, dropout
independence, loss behavior in degenerate cases) rather than model
quality. Run before any long training:

uv run python phase1_foundation/project3b_simcse/test_model.py
"""

import torch

import config
from model import MLPHead, SimCSEEncoder, alignment_and_uniformity, simcse_loss


def test_dropout_creates_two_views():
    """
    The heart of unsupervised SimCSE: same input, two dropout masks,
    two DIFFERENT embeddings. If this fails (identical outputs), the
    positive pair is degenerate and training cannot work (paper
    Table 3: "fixed 0.1" / "no dropout" variants collapse).
    """
    tiny = config_like()
    encoder = SimCSEEncoder(tiny).to(config.DEVICE)
    encoder.train()

    idx = torch.randint(5, 100, (4, 10), device=config.DEVICE)
    mask = torch.ones_like(idx)

    z1 = encoder.encode(idx, mask, project=False)
    z2 = encoder.encode(idx, mask, project=False)

    assert z1.shape == z2.shape and z1.shape[-1] == tiny.N_EMBD
    diff = (z1 - z2).abs().max().item()
    assert diff > 1e-6, f"Two forward passes produced identical embeddings ({diff})"
    print(f"✓ Dropout independence: max |z1 - z2| = {diff:.4f} (> 0)")


def test_positive_closer_than_negative():
    """
    The objective must rank the TRUE positive above in-batch negatives:
    a batch whose two views are noisy copies of each other (what
    dropout produces) must yield a lower loss than a batch whose views
    are misaligned (shuffled rows = every pair points at a negative).
    """
    torch.manual_seed(0)
    z1 = torch.randn(16, 32)

    # Positive pairs: view 2 is a noisy copy of view 1 (dropout-like)
    z2_close = z1 + 0.1 * torch.randn(16, 32)
    loss_close = simcse_loss(z1, z2_close, temperature=0.05)

    # Misaligned: each row pairs with a different sentence's embedding
    z2_shuffled = z1[torch.randperm(16)]
    loss_shuffled = simcse_loss(z1, z2_shuffled, temperature=0.05)

    assert loss_close < loss_shuffled, (
        f"Loss ordering wrong: aligned={loss_close:.3f} "
        f"should be < misaligned={loss_shuffled:.3f}"
    )
    print(
        f"✓ Positive ranking: aligned {loss_close:.3f} < misaligned {loss_shuffled:.3f}"
    )


def test_loss_finite_and_gradient_flows():
    """Loss is finite and gradients reach both the head and the encoder."""
    encoder = SimCSEEncoder(config_like()).to(config.DEVICE)
    encoder.train()

    idx = torch.randint(5, 100, (6, 10), device=config.DEVICE)
    mask = torch.ones_like(idx)

    z1 = encoder.encode(idx, mask, project=True)
    z2 = encoder.encode(idx, mask, project=True)
    loss = simcse_loss(z1, z2, temperature=0.05)

    assert torch.isfinite(loss), "Loss is not finite"
    loss.backward()

    assert encoder.bert.embeddings.token.weight.grad is not None, (
        "No gradient reached the token embeddings"
    )
    assert encoder.mlp_head.dense.weight.grad is not None, (
        "No gradient reached the MLP head"
    )
    print(f"✓ Gradients flow (loss={loss.item():.4f})")


def test_mlp_head_changes_projection_only():
    """
    The MLP trick: with project=False the raw CLS embedding is returned;
    with project=True the tanh head transforms it. Evaluation uses the
    former, training the latter (paper section 5).
    """
    head = MLPHead(16)
    x = torch.randn(4, 16)
    out = head(x)
    assert out.shape == x.shape
    assert out.min() >= -1.0 and out.max() <= 1.0, "tanh output out of range"
    print("✓ MLP head: output bounded by tanh, shape preserved")


def test_alignment_uniformity():
    """Alignment/uniformity metrics respond in the right direction."""
    torch.manual_seed(0)
    z1 = torch.randn(64, 32)
    z2 = z1 + 0.01 * torch.randn(64, 32)  # tight pair -> low alignment
    align_tight, uniform_tight = alignment_and_uniformity(z1, z2)

    z3 = torch.randn(64, 32)
    z4 = z3 + torch.randn(64, 32)  # loose pair -> higher alignment
    align_loose, _ = alignment_and_uniformity(z3, z4)

    assert align_tight < align_loose, (
        f"Alignment direction wrong: {align_tight:.3f} vs {align_loose:.3f}"
    )
    print(
        f"✓ Alignment/uniformity: tight {align_tight:.3f} < loose {align_loose:.3f}"
    )


def config_like():
    """
    Build a minimal project-3-style config object for tests.

    The real BERT class reads attributes off the config module passed to
    it; we construct an equivalent namespace with a tiny vocabulary so
    tests run fast on any device.
    """
    import importlib.util
    import os
    import sys

    p3_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "project3_contextual_embeddings",
    )
    spec = importlib.util.spec_from_file_location(
        "_p3_config", os.path.join(p3_dir, "config.py")
    )
    p3_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p3_config)

    # Tiny model for fast tests
    p3_config.VOCAB_SIZE = 128
    p3_config.N_EMBD = 32
    p3_config.N_HEAD = 4
    p3_config.N_LAYER = 2
    p3_config.BLOCK_SIZE = 32
    p3_config.DROPOUT = 0.1
    p3_config.USE_NSP = False

    saved = sys.modules.get("config")
    sys.modules["config"] = p3_config
    try:
        spec = importlib.util.spec_from_file_location(
            "_p3_model", os.path.join(p3_dir, "model.py")
        )
        p3_model = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(p3_model)
    finally:
        if saved is not None:
            sys.modules["config"] = saved
        else:
            sys.modules.pop("config", None)

    return p3_config


if __name__ == "__main__":
    print("=" * 60)
    print("SIMCSE SANITY TESTS")
    print("=" * 60)
    test_dropout_creates_two_views()
    test_positive_closer_than_negative()
    test_loss_finite_and_gradient_flows()
    test_mlp_head_changes_projection_only()
    test_alignment_uniformity()
    print("=" * 60)
    print("✓ All tests passed")
    print("=" * 60)
