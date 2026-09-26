# ruff: noqa
"""
Tests for Project 5: SFT template, response masking, and checkpoint wiring.

Run from the repo root:
    uv run python phase2_finetuning/project5_sft/test_model.py

What the tests pin down:

- The template renders Alpaca's exact format, with and without the
  optional input block.
- encode_labeled() produces token lists IDENTICAL to encoding the full
  example text in one call (the trailing-space junction from template.py's
  docstring), masks exactly the prompt/pad positions, and the (x, y)
  shift lands the first response token and <EOS> on taught positions.
- The masked loss is the cross-entropy over unmasked positions only:
  a single-taught-position batch scores exactly that position, changing
  ignored targets changes nothing, and a fully-ignored batch is NaN (the
  sharp edge that makes "no response tokens" a prepare-time error, not a
  training-time surprise).
- The base model checkpoint loads into this project's model class with
  matching shapes (SFT attaches to project 4, literally).
- A tiny model overfits a fixed SFT batch: gradients really flow through
  the mask.
- sample_response() stops at <EOS> - the property generate.py exists for.

The tokenizer tests load the checkpoint Project 4 trained; if it is
missing on this machine, those tests fail with a pointer to
train_tokenizer.py rather than silently passing.
"""

import pathlib
import random
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

import config
from model import GPT
from template import PREAMBLE, RESPONSE_MARKER, encode_labeled, format_prompt
from tokenizer import BPETokenizer


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


def load_tokenizer_or_fail() -> BPETokenizer:
    """Load the project 4 tokenizer with a helpful failure message."""
    if not pathlib.Path(config.TOKENIZER_FILE).is_file():
        raise AssertionError(
            f"No tokenizer at {config.TOKENIZER_FILE}; run "
            f"phase1_foundation/project4_pretrain/train_tokenizer.py once."
        )
    return BPETokenizer.load()


def shift(tokens: torch.Tensor, labels: torch.Tensor):
    """
    The (x, y) shift train.py applies, mirrored for tests.

    .contiguous() matters: model.forward's fused cross-entropy calls
    .view(-1) on the targets, which requires a contiguous tensor. The
    training path guarantees this (numpy .astype returns a fresh
    C-contiguous copy); tests must too.
    """
    return tokens[:, :-1].contiguous(), labels[:, 1:].contiguous()


def make_synthetic_batch(batch_size: int = 4, seq_len: int = 32,
                         vocab_size: int = 256, seed: int = 1):
    """
    Synthetic SFT rows with the real array layout: [BOS, prompt..., 
    response..., EOS, PAD...], labels -1 everywhere except response + EOS.
    No tokenizer involved - the mask arithmetic is what is under test.
    """
    rng = random.Random(seed)
    bos, eos, pad = (config.SPECIAL_TOKENS[k] for k in ("<BOS>", "<EOS>", "<PAD>"))
    tokens, labels = [], []
    for _ in range(batch_size):
        n_prompt = rng.randint(8, 14)
        n_resp = rng.randint(6, 12)
        row = [bos] + [rng.randint(4, vocab_size - 1) for _ in range(n_prompt)]
        response_start = len(row)
        row += [rng.randint(4, vocab_size - 1) for _ in range(n_resp)] + [eos]
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


# =============================================================================
# TEMPLATE + MASK TESTS
# =============================================================================


def test_template_formatting():
    with_input = format_prompt("Fold the laundry.", "Colors separately.")
    without_input = format_prompt("Fold the laundry.")
    assert with_input.startswith(PREAMBLE), "preamble missing"
    assert "### Instruction:\nFold the laundry." in with_input
    assert "### Input:\nColors separately." in with_input
    assert "### Input:" not in without_input, "empty input must not render a block"
    assert with_input.rstrip().endswith(RESPONSE_MARKER)
    assert without_input.rstrip().endswith(RESPONSE_MARKER)


def test_encoding_matches_whole_text_encode():
    """The word-by-word halves must equal one-shot encode of the full text
    (the trailing-space identity template.py promises)."""
    tok = load_tokenizer_or_fail()
    instruction = "Name two primary colors."
    output = "Red and blue are primary colors."
    tokens, labels = encode_labeled(tok, instruction, "", output)

    prompt_text = format_prompt(instruction, "")
    one_shot = tok.encode(prompt_text + " " + output)
    bos = config.SPECIAL_TOKENS["<BOS>"]
    eos = config.SPECIAL_TOKENS["<EOS>"]
    assert tokens[0] == bos
    assert list(tokens[1:1 + len(one_shot)]) == one_shot, (
        "word-by-word encoding diverged from whole-text encode"
    )
    assert tokens[len(one_shot) + 1] == eos


def test_label_mask_alignment():
    """Taught positions = response + <EOS>; the shift lands them right."""
    tok = load_tokenizer_or_fail()
    instruction = "What is 2 + 2?"
    output = "2 + 2 equals 4."
    tokens, labels = encode_labeled(tok, instruction, "", output)
    tokens = np.array(tokens)
    labels = np.array(labels)

    bos = config.SPECIAL_TOKENS["<BOS>"]
    eos = config.SPECIAL_TOKENS["<EOS>"]
    pad = config.SPECIAL_TOKENS["<PAD>"]
    assert tokens[0] == bos

    taught = np.where(labels != -1)[0]
    # every taught position is a real response/EOS token position
    assert np.all(tokens[taught] != pad), "pads must never be taught"
    # the prompt (up to the first taught position) is fully masked
    first_taught = taught[0]
    assert np.all(labels[:first_taught] == -1), "prompt positions must be masked"
    # <EOS> is the final taught token: the model learns where to stop
    assert tokens[taught[-1]] == eos, "<EOS> must be the final taught token"

    # the (x, y) shift train.py applies: x = tokens[:-1], y = labels[1:]
    x, y = tokens[:-1], labels[1:]
    taught_shifted = np.where(y != -1)[0]
    assert np.array_equal(y[taught_shifted], tokens[taught_shifted + 1]), (
        "taught target must equal the token one position after its input"
    )
    # predicting FROM the last prompt token yields the first response token
    assert y[first_taught - 1] == tokens[first_taught]


# =============================================================================
# LOSS TESTS
# =============================================================================


def test_masked_loss_is_unmasked_positions_only():
    model = make_tiny_model()
    tokens, labels = make_synthetic_batch()
    x, y = shift(tokens, labels)

    logits, loss = model(x, y)
    manual = F.cross_entropy(
        logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1
    )
    assert torch.isfinite(loss)
    assert torch.allclose(loss, manual, atol=1e-6), (
        "model loss must equal manual CE over unmasked positions"
    )


def test_single_taught_position_scores_exactly_itself():
    """With exactly ONE taught position in the whole batch, the model loss
    is the cross-entropy at that position and nothing else."""
    model = make_tiny_model()
    tokens, labels = make_synthetic_batch(seed=5)
    taught = torch.where(labels[0] != -1)[0]
    keep = int(taught[1])  # a response-token position in row 0

    slim_labels = torch.full_like(labels, -1)
    slim_labels[0, keep] = labels[0, keep]
    x, y = shift(tokens, slim_labels)
    assert int((y != -1).sum()) == 1, "test premise: exactly one taught position"

    logits, loss = model(x, y)
    target_pos = keep - 1  # in x/y coordinates after the shift
    row = logits[0, target_pos]
    manual = F.cross_entropy(row.unsqueeze(0), y[0, target_pos].unsqueeze(0))
    assert torch.allclose(loss, manual, atol=1e-6)


def test_response_mask_changes_the_objective():
    """Response-only (InstructGPT) and full-sequence (Alpaca) losses are
    different numbers on the same batch - the mask is not cosmetic."""
    model = make_tiny_model()
    tokens, labels = make_synthetic_batch(seed=9)
    x, y_resp = shift(tokens, labels)
    y_full = tokens[:, 1:].contiguous()

    _, loss_resp = model(x, y_resp)
    _, loss_full = model(x, y_full)
    manual_full = F.cross_entropy(
        model(x)[0].view(-1, model.config.VOCAB_SIZE), y_full.view(-1)
    )
    assert loss_resp != loss_full
    assert torch.allclose(loss_full, manual_full, atol=1e-6)


def test_fully_masked_batch_is_nan():
    """A batch with zero taught positions has no gradient signal and
    comes back NaN. prepare_data.py makes this unreachable (every example
    teaches >= 1 response token + <EOS>); the test records why that
    guarantee matters."""
    model = make_tiny_model()
    tokens, _ = make_synthetic_batch(seed=7)
    x, y = shift(tokens, torch.zeros_like(tokens))
    y = torch.full_like(y, -1)
    _, loss = model(x, y)
    assert torch.isnan(loss)


def test_overfit_sft_batch():
    """Gradients flow through the mask: a tiny model driven on one fixed
    batch must drive the response loss toward zero."""
    model = make_tiny_model(vocab_size=256, block_size=32, seed=11)
    tokens, labels = make_synthetic_batch(seq_len=32)
    x, y = shift(tokens, labels)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    first = None
    for step in range(250):
        _, loss = model(x, y)
        if first is None:
            first = loss.item()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    final = loss.item()
    print(f"    overfit: {first:.3f} -> {final:.3f}")
    assert final < 0.5 * first, (
        f"overfit failed: loss {first:.3f} -> {final:.3f} "
        f"(mask or shift is likely wrong)"
    )


# =============================================================================
# CHECKPOINT TESTS
# =============================================================================


def test_checkpoint_roundtrip_with_config():
    """Five-key format + embedded config dict round-trips to identical
    logits - the contract generate.py relies on."""
    model = make_tiny_model(seed=13)
    path = pathlib.Path(config.CHECKPOINT_DIR) / "_test_tiny.pt"
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
    }, path)
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    restored = GPT(SimpleNamespace(
        VOCAB_SIZE=ckpt["config"]["VOCAB_SIZE"],
        N_EMBD=ckpt["config"]["N_EMBD"],
        N_HEAD=ckpt["config"]["N_HEAD"],
        N_LAYER=ckpt["config"]["N_LAYER"],
        BLOCK_SIZE=ckpt["config"]["BLOCK_SIZE"],
        DROPOUT=ckpt["config"]["DROPOUT"],
    ))
    restored.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    restored.eval()
    x = torch.randint(4, model.config.VOCAB_SIZE, (2, 16))
    with torch.no_grad():
        a, _ = model(x)
        b, _ = restored(x)
    assert torch.allclose(a, b, atol=0.0), "checkpoint round-trip changed logits"
    path.unlink()


def test_base_checkpoint_loads():
    """The project 4 base model must load into project 5's model class
    with matching shapes - the literal 'attach to the base model' step."""
    if not pathlib.Path(config.BASE_CHECKPOINT).is_file():
        raise AssertionError(
            f"No base checkpoint at {config.BASE_CHECKPOINT}; pre-train "
            f"project 4 first."
        )
    ckpt = torch.load(config.BASE_CHECKPOINT, map_location="cpu", weights_only=True)
    arch = ckpt["config"]
    model = GPT(SimpleNamespace(
        VOCAB_SIZE=arch["VOCAB_SIZE"], N_EMBD=arch["N_EMBD"],
        N_HEAD=arch["N_HEAD"], N_LAYER=arch["N_LAYER"],
        BLOCK_SIZE=arch["BLOCK_SIZE"], DROPOUT=0.0,
    ))
    model.load_state_dict(ckpt["model_state_dict"])  # shape mismatch would raise
    print(f"    base checkpoint: iter {ckpt['iter']}, "
          f"val loss {ckpt['val_loss']:.4f}, "
          f"{model.get_num_params(non_embedding=False):,} params")


def test_eos_stop_sampling():
    """sample_response() must stop when the model emits <EOS> and must
    not include it in the decoded text."""

    class StubModel:
        """Emits tokens 7, 8, then <EOS>, forever."""
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

    # and the cap still applies when <EOS> never comes
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
    print("PROJECT 5 TESTS (SFT)")
    print("=" * 60)
    tests = [
        test_template_formatting,
        test_encoding_matches_whole_text_encode,
        test_label_mask_alignment,
        test_masked_loss_is_unmasked_positions_only,
        test_single_taught_position_scores_exactly_itself,
        test_response_mask_changes_the_objective,
        test_fully_masked_batch_is_nan,
        test_overfit_sft_batch,
        test_checkpoint_roundtrip_with_config,
        test_base_checkpoint_loads,
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
