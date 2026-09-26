# ruff: noqa
"""
Tests for Project 6: LoRA adapters, freezing, merging, checkpoint wiring.

Run from the repo root:
    uv run python phase2_finetuning/project6_lora/test_model.py

What the tests pin down:

- At init (lora_B = 0) the wrapped model computes EXACTLY what the base
  model computes - the paper's step-0 guarantee.
- Only adapters require gradients; the frozen weights do not move during
  training, and the trainable count matches the r x (in + out) arithmetic
  for every wrapped module (393,216 = 0.311% at the real model's defaults).
- merge() folds the update into the weight with outputs matching to float
  tolerance - the deploy-time story of the paper.
- Checkpoints with a 'lora_config' round-trip through generate.load_model
  to identical logits; plain checkpoints still load.
- The real project 4 base checkpoint loads and wraps at the expected
  adapter count.
- Adapter-only training moves the loss on a synthetic batch (and plateaus
  at the random base's feature limit - the docstring explains).

The tokenizer-dependent identity tests live in project 5's suite; this
file's tests are tokenizer-free except the base-checkpoint test.
"""

import pathlib
import random
from types import SimpleNamespace

import torch
from torch.nn import functional as F

import config
import lora
from lora import LoRALinear, apply_lora, lora_parameter_report
from model import GPT


# =============================================================================
# HELPERS
# =============================================================================


def make_tiny_model(vocab_size: int = 256, block_size: int = 64, seed: int = 0) -> GPT:
    """Small GPT with the same class as the real one; CPU-fast."""
    cfg = SimpleNamespace(
        VOCAB_SIZE=vocab_size, N_EMBD=64, N_HEAD=4, N_LAYER=2,
        BLOCK_SIZE=block_size, DROPOUT=0.0,
    )
    torch.manual_seed(seed)
    return GPT(cfg)


def make_constant_response_batch(batch_size: int = 4, seq_len: int = 32,
                                 vocab_size: int = 256, response_id: int = 7,
                                 seed: int = 1):
    """
    Synthetic SFT rows whose response is a single repeated token: the
    mapping is trivially learnable by a tiny adapter set, unlike random
    response tokens (which only full-capacity memorization can fit).
    Layout matches the training arrays: [BOS, prompt..., response..., EOS,
    PAD...], labels -1 outside response + EOS.
    """
    rng = random.Random(seed)
    bos, eos, pad = (config.SPECIAL_TOKENS[k] for k in ("<BOS>", "<EOS>", "<PAD>"))
    tokens, labels = [], []
    for _ in range(batch_size):
        n_prompt = rng.randint(8, 14)
        n_resp = rng.randint(6, 12)
        row = [bos] + [rng.randint(4, vocab_size - 1) for _ in range(n_prompt)]
        response_start = len(row)
        row += [response_id] * n_resp + [eos]
        row += [pad] * (seq_len - len(row))
        lab = [-1] * len(row)
        for i in range(response_start, seq_len):
            if row[i] != pad:
                lab[i] = row[i]
        tokens.append(row)
        labels.append(lab)
    return (
        torch.tensor(tokens, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


def shift(tokens: torch.Tensor, labels: torch.Tensor):
    """The (x, y) teacher-forcing shift train.py applies; contiguous
    targets required by the fused cross-entropy's .view(-1)."""
    return tokens[:, :-1].contiguous(), labels[:, 1:].contiguous()


def train_steps(model, x, y, steps: int, lr: float = 1e-3):
    """AdamW over LoRA's trainable parameters only; returns first/last loss."""
    params = lora.trainable_parameters(model)
    opt = torch.optim.AdamW(params, lr=lr)
    first = None
    loss = None
    for _ in range(steps):
        _, loss = model(x, y)
        if first is None:
            first = loss.item()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return first, loss.item()


# =============================================================================
# LORA MECHANICS
# =============================================================================


def test_lora_init_is_identity():
    """With lora_B = 0, wrapped and base models must agree exactly."""
    model = make_tiny_model(seed=2)
    x = torch.randint(4, 256, (2, 16))
    torch.manual_seed(123)
    base_logits, _ = model(x)
    torch.manual_seed(123)
    apply_lora(model, ["c_attn"], r=8, alpha=16)
    wrapped_logits, _ = model(x)
    assert torch.equal(base_logits, wrapped_logits), (
        "wrapped model differs from base at init - lora_B is not zero"
    )


def test_trainable_count_and_freeze():
    model = make_tiny_model(seed=3)
    apply_lora(model, ["c_attn"], r=8, alpha=16)

    trainable, total = lora_parameter_report(model)
    # per wrapped c_attn (768->2304 fused? no: tiny model 64->192):
    # A is 8 x 64, B is 192 x 8 -> 2048 per module, 2 layers -> 4096
    n_wrapped = sum(isinstance(m, LoRALinear) for m in model.modules())
    per_module = 8 * 64 + 192 * 8
    assert n_wrapped == 2
    assert trainable == n_wrapped * per_module, (
        f"expected {n_wrapped * per_module} trainable, got {trainable}"
    )
    for name, p in model.named_parameters():
        if "lora_" in name:
            assert p.requires_grad, f"{name} should train"
        else:
            assert not p.requires_grad, f"{name} should be frozen"

    frozen_weight = model.transformer.blocks[0].attn.c_attn.weight.clone()
    tokens, labels = make_constant_response_batch()
    x, y = shift(tokens, labels)
    train_steps(model, x, y, steps=5)
    assert torch.equal(model.transformer.blocks[0].attn.c_attn.weight, frozen_weight), (
        "a frozen weight moved during training"
    )
    assert not torch.equal(model.transformer.blocks[0].attn.c_attn.lora_B,
                           torch.zeros_like(model.transformer.blocks[0].attn.c_attn.lora_B)), (
        "lora_B never moved - no gradient reached the adapters"
    )


def test_lora_target_validation():
    model = make_tiny_model(seed=4)
    try:
        apply_lora(model, ["no_such_module"], r=8, alpha=16)
    except ValueError as exc:
        assert "matched no module" in str(exc)
    else:
        raise AssertionError("bogus LoRA target did not raise")


def test_merge_equivalence():
    """After training (B != 0), merge() must preserve outputs to float
    tolerance while switching the layer to the plain path."""
    torch.manual_seed(6)
    model = make_tiny_model(seed=6)
    apply_lora(model, ["c_attn"], r=8, alpha=16)
    tokens, labels = make_constant_response_batch(seed=9)
    x, y = shift(tokens, labels)
    train_steps(model, x, y, steps=20)
    assert model.transformer.blocks[0].attn.c_attn.merged is False

    model.eval()
    with torch.no_grad():
        before, _ = model(x)
    for block in model.transformer.blocks:
        block.attn.c_attn.merge()
    with torch.no_grad():
        after, _ = model(x)
    assert all(block.attn.c_attn.merged for block in model.transformer.blocks)
    assert torch.allclose(before, after, rtol=1e-5, atol=1e-6), (
        "merged model's outputs diverged from the adapter model's"
    )


# =============================================================================
# OVERFIT (gradients flow through the adapters)
# =============================================================================


def test_overfit_constant_response():
    """
    Adapter-only training must move the loss, on a RANDOM tiny base.

    Measured behavior (and why the threshold is modest): loss falls then
    plateaus around 4.2 from an initial 4.8, stable across target sets
    (c_attn only vs +c_proj +fc) and learning rates (3e-3, 1e-2). A random
    frozen base gives the adapters noise features to steer; LoRA's
    premise is a pre-trained, structured base (the paper's section 2
    argument), which is what the real run supplies. This test pins the
    wiring: gradients reach the adapters and the loss responds.
    """
    model = make_tiny_model(vocab_size=256, block_size=32, seed=11)
    apply_lora(model, ["c_attn"], r=8, alpha=16)
    tokens, labels = make_constant_response_batch(seq_len=32)
    x, y = shift(tokens, labels)

    first, final = train_steps(model, x, y, steps=600, lr=3e-3)
    print(f"    adapter-only training: {first:.3f} -> {final:.3f}")
    assert final < 0.9 * first, (
        f"adapter-only training did not move the loss ({first:.3f} -> {final:.3f})"
    )


# =============================================================================
# CHECKPOINT WIRING
# =============================================================================


def test_lora_checkpoint_roundtrip():
    """five-key + lora_config round-trips through generate.load_model to
    identical logits; a plain checkpoint loads too."""
    from generate import load_model

    torch.manual_seed(13)
    model = make_tiny_model(seed=13)
    apply_lora(model, ["c_attn"], r=8, alpha=16)
    path = pathlib.Path(config.CHECKPOINT_DIR) / "_test_lora.pt"
    torch.save({
        "iter": 0,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": None,
        "train_loss": 1.0,
        "val_loss": 1.0,
        "config": {
            "VOCAB_SIZE": model.config.VOCAB_SIZE,
            "N_EMBD": model.config.N_EMBD,
            "N_HEAD": model.config.N_HEAD,
            "N_LAYER": model.config.N_LAYER,
            "BLOCK_SIZE": model.config.BLOCK_SIZE,
            "DROPOUT": 0.0,
        },
        "lora_config": {"r": 8, "alpha": 16, "targets": ["c_attn"]},
    }, path)
    restored = load_model("_test_lora.pt", "cpu")
    model.eval()
    restored.eval()
    x = torch.randint(4, 256, (2, 16))
    with torch.no_grad():
        a, _ = model(x)
        b, _ = restored(x)
    assert torch.allclose(a, b, atol=0.0), "LoRA checkpoint round-trip changed logits"
    path.unlink()

    # a plain checkpoint (no lora_config) also loads through the same door
    plain = make_tiny_model(seed=14)
    plain_path = pathlib.Path(config.CHECKPOINT_DIR) / "_test_plain.pt"
    torch.save({
        "iter": 0,
        "model_state_dict": plain.state_dict(),
        "optimizer_state_dict": None,
        "train_loss": 1.0,
        "val_loss": 1.0,
        "config": {
            "VOCAB_SIZE": plain.config.VOCAB_SIZE,
            "N_EMBD": plain.config.N_EMBD,
            "N_HEAD": plain.config.N_HEAD,
            "N_LAYER": plain.config.N_LAYER,
            "BLOCK_SIZE": plain.config.BLOCK_SIZE,
            "DROPOUT": 0.0,
        },
    }, plain_path)
    restored_plain = load_model("_test_plain.pt", "cpu")
    restored_plain.eval()
    with torch.no_grad():
        c, _ = restored_plain(x)
    plain.eval()
    with torch.no_grad():
        d, _ = plain(x)
    assert torch.allclose(d, c, atol=0.0)
    plain_path.unlink()


def test_base_checkpoint_loads_and_wraps():
    """The project 4 base checkpoint loads, wraps at 16 modules, and the
    trainable count matches the config-projected 393,216 (0.311%)."""
    if not pathlib.Path(config.BASE_CHECKPOINT).is_file():
        raise AssertionError(
            f"No base checkpoint at {config.BASE_CHECKPOINT}; pre-train "
            f"project 4 first."
        )
    ckpt = torch.load(config.BASE_CHECKPOINT, map_location="cpu", weights_only=True)
    arch = ckpt["config"]
    cfg = SimpleNamespace(
        VOCAB_SIZE=arch["VOCAB_SIZE"], N_EMBD=arch["N_EMBD"],
        N_HEAD=arch["N_HEAD"], N_LAYER=arch["N_LAYER"],
        BLOCK_SIZE=arch["BLOCK_SIZE"], DROPOUT=0.0,
    )
    model = GPT(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    apply_lora(model, config.LORA_TARGETS, config.LORA_R, config.LORA_ALPHA)
    trainable, total = lora_parameter_report(model)
    per_module = config.LORA_R * arch["N_EMBD"] + 3 * arch["N_EMBD"] * config.LORA_R
    expected = arch["N_LAYER"] * per_module
    assert trainable == expected, (
        f"expected {expected} trainable params, got {trainable}"
    )
    print(f"    base checkpoint: iter {ckpt['iter']}, "
          f"val loss {ckpt['val_loss']:.4f}; adapters train "
          f"{trainable:,}/{total:,} params ({trainable / total:.3%})")


def test_eos_stop_sampling():
    """sample_response() stops at <EOS> and respects max_new_tokens
    (same contract as project 5's sampler)."""

    class StubModel:
        class config:
            BLOCK_SIZE = 64

        def __call__(self, idx):
            n = idx.size(1)
            eos = config.SPECIAL_TOKENS["<EOS>"]
            next_token = eos if n >= 3 else 7 + (n - 1)
            logits = torch.full((1, 1, 256), -10.0)
            logits[0, 0, next_token] = 10.0
            return logits, None

    class StubTokenizer:
        def encode_words(self, words, final):
            return [config.SPECIAL_TOKENS["<BOS>"]]

        def decode(self, ids):
            return ",".join(str(i) for i in ids)

    from generate import sample_response

    text = sample_response(StubModel(), StubTokenizer(), "answer me",
                           max_new_tokens=50, temperature=1.0, top_k=None,
                           device="cpu")
    assert text == "7,8", f"expected exactly [7, 8] then stop, got {text!r}"

    class NeverStops(StubModel):
        def __call__(self, idx):
            logits = torch.full((1, 1, 256), -10.0)
            logits[0, 0, 9] = 10.0
            return logits, None

    text = sample_response(NeverStops(), StubTokenizer(), "answer me",
                           max_new_tokens=5, temperature=1.0, top_k=None,
                           device="cpu")
    assert text == ",".join(["9"] * 5), f"max_new_tokens cap broken: {text!r}"


# =============================================================================
# RUNNER
# =============================================================================


def main():
    print("=" * 60)
    print("PROJECT 6 TESTS (LoRA)")
    print("=" * 60)
    tests = [
        test_lora_init_is_identity,
        test_trainable_count_and_freeze,
        test_lora_target_validation,
        test_merge_equivalence,
        test_overfit_constant_response,
        test_lora_checkpoint_roundtrip,
        test_base_checkpoint_loads_and_wraps,
        test_eos_stop_sampling,
    ]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"✓ {test.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"✗ {test.__name__}: {exc}")
    print("=" * 60)
    if failed:
        print(f"{failed} test(s) FAILED")
        raise SystemExit(1)
    print(f"All {len(tests)} tests passed")


if __name__ == "__main__":
    main()
