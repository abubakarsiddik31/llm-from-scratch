# ruff: noqa
"""
Tests for Project 4: fast BPE tokenizer + ~126M GPT.

Run from the repo root:
    uv run python phase1_foundation/project4_pretrain/test_model.py

The tokenizer tests are the important ones: the incremental (heap-based)
trainer must produce EXACTLY what a naive full-rescan trainer produces
under the same tie-breaking rule, and the rank-based encoder must equal
applying merge rules in learned order. Those two properties are what let
this file differ from the chapter 2 implementation without changing what a
"checkpoint" means.

Model tests cover the ~126M parameter budget, forward/loss shapes, causal
masking (future tokens must not leak into past logits), weight tying,
overfitting a tiny batch end-to-end, checkpoint save/load (the five-key
repository format, weights_only), and generation.
"""

import pathlib
import random
import time
from collections import Counter
from types import SimpleNamespace

import torch

import config
from model import GPT, GPTConfig
from tokenizer import BPETokenizer, merge_word


# =============================================================================
# TOKENIZER TESTS
# =============================================================================


def naive_bpe_train(text: str, vocab_size: int, min_frequency: int) -> BPETokenizer:
    """
    Naive full-rescan BPE trainer (the chapter 2 algorithm, plus this
    project's deterministic tie-break: among pairs with the maximum count,
    merge the lexicographically smallest). Used ONLY to verify that the
    incremental trainer in tokenizer.py produces an identical merge list.
    """
    tok = BPETokenizer()
    chars = sorted(list(set(text)))
    for ch in chars:
        if ch not in tok.vocab:
            new_id = len(tok.vocab)
            tok.vocab[ch] = new_id
            tok.inverse_vocab[new_id] = ch

    word_counts = tok.split_into_words(text)
    words = [list(w) for w in word_counts.keys()]
    counts = list(word_counts.values())

    num_merges = vocab_size - len(config.SPECIAL_TOKENS) - len(chars)
    for _ in range(num_merges):
        pair_counts: Counter = Counter()
        for word, c in zip(words, counts):
            for pair in zip(word, word[1:]):
                pair_counts[pair] += c
        if not pair_counts:
            break
        best_freq = max(pair_counts.values())
        best = min(p for p, c in pair_counts.items() if c == best_freq)
        if best_freq < min_frequency:
            break
        new_id = len(tok.vocab)
        new_token = best[0] + best[1]
        tok.vocab[new_token] = new_id
        tok.inverse_vocab[new_id] = new_token
        tok.merges.append(best)
        words = [merge_word(w, best) for w in words]
    return tok


def make_test_corpus(seed: int = 7) -> str:
    """
    Small redundant corpus: enough structure for ~100 merges to bite, and
    punctuation/digits present so round-trip probes can be reproduced
    (characters outside the vocabulary decode as <UNK> by design).
    """
    rng = random.Random(seed)
    words = ("the quick brown fox jumps over a lazy dog, while engineers: "
             "train transformers with gradient descent and 123 data loaders! "
             "shuffle tokens through attention layers every morning.").split()
    return " ".join(rng.choice(words) for _ in range(3000))


def test_tokenizer_incremental_matches_naive():
    corpus = make_test_corpus()
    incremental = BPETokenizer()
    incremental.train(corpus, vocab_size=250, min_frequency=2)
    naive = naive_bpe_train(corpus, vocab_size=250, min_frequency=2)

    if incremental.merges != naive.merges:
        shared = min(len(incremental.merges), len(naive.merges))
        diff = next(
            (i for i in range(shared) if incremental.merges[i] != naive.merges[i]),
            shared,
        )
        raise AssertionError(
            f"merge lists diverge at index {diff}: "
            f"{incremental.merges[diff] if diff < len(incremental.merges) else 'end'} vs "
            f"{naive.merges[diff] if diff < len(naive.merges) else 'end'}"
        )
    assert incremental.vocab == naive.vocab
    print(f"✓ Incremental trainer matches naive rescan "
          f"({len(incremental.merges)} identical merges)")


def test_tokenizer_roundtrip():
    tok = BPETokenizer()
    corpus = make_test_corpus()
    tok.train(corpus, vocab_size=300, min_frequency=2)

    probes = [
        "the quick brown fox",
        "attention layers, every morning: 123 tokens shuffle!",
    ]
    for probe in probes:
        decoded = tok.decode(tok.encode(probe))
        assert decoded == probe, f"round-trip failed for {probe!r}: {decoded!r}"

    # whitespace normalization is lossy by design (chapter 2 convention:
    # any whitespace run becomes one separator), so compare against the
    # normalized form for the space-run probe
    messy = "    spaces   collapse    here "
    assert tok.decode(tok.encode(messy)) == " ".join(messy.split()), \
        "normalized round-trip failed"
    print("✓ encode/decode round-trips on punctuation, digits; "
          "whitespace normalizes like chapter 2")


def test_tokenizer_rank_encode_equals_ordered_merges():
    """
    The fast encoder (lowest-rank first) must equal the textbook encoding:
    apply every learned merge in order, per word.
    """
    tok = BPETokenizer()
    tok.train(make_test_corpus(), vocab_size=300, min_frequency=2)

    probe = "the quick brown fox jumps"
    fast_ids = tok.encode(probe)

    # textbook: split with the training convention, apply merges in order
    word_counts = tok.split_into_words(probe)
    reference_ids = []
    for word in word_counts.keys():
        symbols = list(word)
        for pair in tok.merges:
            symbols = merge_word(symbols, pair)
        reference_ids.extend(tok._ids_for_symbols(symbols))

    assert fast_ids == reference_ids, "rank-based encode != ordered merges"
    print(f"✓ rank-based encode equals ordered-merge encode ({len(fast_ids)} tokens)")


def test_tokenizer_special_tokens_and_unk():
    assert config.SPECIAL_TOKENS == {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}, (
        "special token IDs must stay at 0-3 (chapters 3/4 depend on them)"
    )
    tok = BPETokenizer()
    tok.train("plain ascii text only", vocab_size=60, min_frequency=2)
    ids = tok.encode("ascii \u00e9")
    assert config.SPECIAL_TOKENS["<UNK>"] in ids, "unseen char should map to UNK"
    assert tok.decode(ids).endswith("<UNK>")
    print("✓ special tokens keep IDs 0-3; unseen characters encode as <UNK>")


# =============================================================================
# MODEL TESTS
# =============================================================================


def full_size_config() -> GPTConfig:
    return GPTConfig(
        VOCAB_SIZE=config.VOCAB_SIZE,
        N_EMBD=config.N_EMBD,
        N_HEAD=config.N_HEAD,
        N_LAYER=config.N_LAYER,
        BLOCK_SIZE=config.BLOCK_SIZE,
        DROPOUT=config.DROPOUT,
    )


def tiny_config(**overrides) -> GPTConfig:
    defaults = dict(VOCAB_SIZE=64, N_EMBD=64, N_HEAD=2, N_LAYER=2, BLOCK_SIZE=32,
                    DROPOUT=0.0)
    defaults.update(overrides)
    return GPTConfig(**defaults)


def test_param_count():
    model = GPT(full_size_config())
    total = model.get_num_params(non_embedding=False)
    non_emb = model.get_num_params(non_embedding=True)
    # 16 layers x 768 x 16,384 vocab: 126,245,376 total (GPT-2 small: 124.4M)
    assert 120_000_000 <= total <= 135_000_000, f"total params {total:,} out of range"
    assert 105_000_000 <= non_emb <= 120_000_000, f"non-embedding {non_emb:,} out of range"
    print(f"✓ parameter count: {total:,} total, {non_emb:,} non-embedding "
          f"(GPT-2 small class)")


def test_forward_shapes_and_loss():
    torch.manual_seed(0)
    cfg = tiny_config()
    model = GPT(cfg)
    x = torch.randint(0, cfg.VOCAB_SIZE, (2, 16))
    y = torch.randint(0, cfg.VOCAB_SIZE, (2, 16))

    logits, loss = model(x, y)
    assert logits.shape == (2, 16, cfg.VOCAB_SIZE)
    assert loss.ndim == 0 and torch.isfinite(loss)

    logits_only, no_loss = model(x)
    assert no_loss is None and torch.equal(logits, logits_only)
    print(f"✓ forward shapes (2,16,{cfg.VOCAB_SIZE}), finite loss {loss.item():.3f}")


def test_causality():
    """Changing tokens AFTER position t must not change logits at <= t."""
    torch.manual_seed(1)
    cfg = tiny_config()
    model = GPT(cfg).eval()

    x = torch.randint(0, cfg.VOCAB_SIZE, (1, 16))
    x_perturbed = x.clone()
    x_perturbed[0, 10:] = torch.randint(0, cfg.VOCAB_SIZE, (6,))

    with torch.no_grad():
        base, _ = model(x)
        pert, _ = model(x_perturbed)

    assert torch.allclose(base[0, :10], pert[0, :10], atol=1e-5), \
        "past logits changed when future tokens changed (causal mask broken)"
    assert not torch.allclose(base[0, 10:], pert[0, 10:]), \
        "future logits did not react to future tokens (model frozen?)"
    print("✓ causal masking: future-token changes leave past logits untouched")


def test_weight_tying():
    model = GPT(tiny_config())
    assert model.lm_head.weight is model.transformer.wte.weight, \
        "lm_head must share wte's matrix"
    print("✓ lm_head and wte share one weight matrix (weight tying)")


def test_overfit_single_batch():
    """One batch, 300 steps: the optimizer must drive loss under 0.5."""
    torch.manual_seed(2)
    cfg = tiny_config()
    model = GPT(cfg)
    x = torch.randint(0, cfg.VOCAB_SIZE, (4, cfg.BLOCK_SIZE))
    y = torch.randint(0, cfg.VOCAB_SIZE, (4, cfg.BLOCK_SIZE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    start = time.time()
    loss_value = None
    for _ in range(300):
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_value = loss.item()
    took = time.time() - start

    assert loss_value < 0.5, f"failed to overfit: loss {loss_value:.3f} after 300 steps"
    print(f"✓ overfit single batch: loss {loss_value:.3f} in 300 steps ({took:.1f}s)")


def test_checkpoint_roundtrip():
    """
    Save in the repository five-key format, reload with weights_only=True
    (the safe torch.load the training scripts use), and confirm identical
    logits.
    """
    torch.manual_seed(3)
    cfg = tiny_config()
    model = GPT(cfg).eval()

    ckpt_path = pathlib.Path(config.CHECKPOINT_DIR) / "test_tmp_checkpoint.pt"
    torch.save({
        "iter": 123,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": None,
        "train_loss": 2.0,
        "val_loss": 1.0,
        "config": {
            "VOCAB_SIZE": cfg.VOCAB_SIZE, "N_EMBD": cfg.N_EMBD,
            "N_HEAD": cfg.N_HEAD, "N_LAYER": cfg.N_LAYER,
            "BLOCK_SIZE": cfg.BLOCK_SIZE, "DROPOUT": cfg.DROPOUT,
        },
    }, ckpt_path)
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        assert ckpt["iter"] == 123 and ckpt["val_loss"] == 1.0
        restored = GPT(GPTConfig(**ckpt["config"]))
        restored.load_state_dict(ckpt["model_state_dict"])
        restored.eval()

        x = torch.randint(0, cfg.VOCAB_SIZE, (1, 8))
        with torch.no_grad():
            original, _ = model(x)
            reloaded, _ = restored(x)
        assert torch.equal(original, reloaded), "logits differ after save/load"
    finally:
        ckpt_path.unlink(missing_ok=True)
    print("✓ checkpoint round-trip: five-key format, weights_only load, exact logits")


def test_generate():
    torch.manual_seed(4)
    cfg = tiny_config()
    model = GPT(cfg).eval()
    context = torch.randint(0, cfg.VOCAB_SIZE, (1, 4))
    out = model.generate(context, max_new_tokens=10, temperature=0.8, top_k=10)
    assert out.shape == (1, 14)
    assert (out >= 0).all() and (out < cfg.VOCAB_SIZE).all()
    print(f"✓ generate: {tuple(out.shape)} with valid token ids")


# =============================================================================
# RUNNER
# =============================================================================


def main():
    print("=" * 60)
    print("PROJECT 4 TESTS")
    print("=" * 60)
    tests = [
        test_tokenizer_incremental_matches_naive,
        test_tokenizer_roundtrip,
        test_tokenizer_rank_encode_equals_ordered_merges,
        test_tokenizer_special_tokens_and_unk,
        test_param_count,
        test_forward_shapes_and_loss,
        test_causality,
        test_weight_tying,
        test_overfit_single_batch,
        test_checkpoint_roundtrip,
        test_generate,
    ]
    failed = 0
    for test in tests:
        try:
            test()
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
