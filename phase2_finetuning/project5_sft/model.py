# ruff: noqa
"""
GPT Model for Supervised Fine-Tuning (~126M parameters)

Same code as project4_pretrain/model.py - same classes, same init, same
forward - because SFT changes the DATA and the LOSS MASK, not the
network. (The class docstrings here add fine-tuning notes; the executable
lines are identical.) Project 5 fine-tunes the checkpoint Project 4
trained; keeping this file in sync (per the repository's
self-contained-project convention) means state dicts load without any key
translation, and any code change here would be a bug to catch in review.

PAPERS (architecture, unchanged):
---------------------------------
- "Attention is All You Need" (Vaswani et al., 2017): transformer blocks
- "Language Models are Unsupervised Multitask Learners" (Radford et al.,
  2019): GPT-2 architecture (pre-LN, learned positions, GELU, tied heads)
- "FlashAttention" (Dao et al., 2022) / PyTorch SDPA: fused attention
- "Using the Output Embedding to Improve Language Models" (Press & Wolf,
  2017): weight tying

WHAT FINE-TUNING CHANGES AROUND THIS FILE (not in it):
------------------------------------------------------
1. The WEIGHTS arrive from checkpoints/project4/model_final.pt instead of
   GPT-2 initialization (train.py loads the state dict; _init_weights
   below only runs before loading, and its output is immediately
   overwritten).
2. The LOSS masks prompt positions: forward(idx, targets) already accepts
   an ignore_index (-1), so response-only fine-tuning passes targets with
   -1 on every prompt/pad position. No architecture change is needed -
   cross-entropy ignoring a label was all the masking SFT requires.
3. DROPOUT stays 0.0 for the same reason as pre-training (see config.py
   for the SFT-specific note).

Interface: GPT(config) where config exposes VOCAB_SIZE, N_EMBD, N_HEAD,
N_LAYER, BLOCK_SIZE, DROPOUT - either the config module or a namespace
(tests build tiny models this way). The module is also the reference for
what "the same model" means when chapters compare pre-training and SFT
runs.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    """Architecture shape, so checkpoints can carry their own config."""

    VOCAB_SIZE: int
    N_EMBD: int
    N_HEAD: int
    N_LAYER: int
    BLOCK_SIZE: int
    DROPOUT: float = 0.0


# =============================================================================
# PART 1: CAUSAL SELF-ATTENTION
# =============================================================================


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.

    PAPER: Vaswani et al. 2017, sections 3.2.1-3.2.2; GPT-2 (2019).

        Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

    Causality: position t may attend only to positions <= t, which is what
    makes next-token prediction well-defined at every offset in one
    parallel forward pass.

    IMPLEMENTATION:
    - Q, K, V come from one fused Linear to 3*n_embd (one GEMM instead of
      three).
    - The masking + softmax + weighted sum run inside
      F.scaled_dot_product_attention(is_causal=True). SDPA picks a fused
      kernel (FlashAttention on CUDA) that never materializes the T x T
      attention matrix - at T=512 that matrix is 512 x 512 x B x heads
      floats per layer, which is exactly what OOMs an 8 GB card.
    """

    def __init__(self, config):
        super().__init__()
        assert config.N_EMBD % config.N_HEAD == 0, (
            f"n_embd ({config.N_EMBD}) must be divisible by n_head ({config.N_HEAD})"
        )
        self.n_head = config.N_HEAD
        self.n_embd = config.N_EMBD
        self.head_size = config.N_EMBD // config.N_HEAD

        # fused QKV projection: one GEMM for all three (see module docstring)
        self.c_attn = nn.Linear(config.N_EMBD, 3 * config.N_EMBD, bias=False)
        # output projection - this is a residual-path weight, initialized
        # scaled down (see GPT._init_weights)
        self.c_proj = nn.Linear(config.N_EMBD, config.N_EMBD, bias=False)

        self.attn_dropout = config.DROPOUT  # passed to SDPA as dropout_p
        self.resid_dropout = nn.Dropout(config.DROPOUT)

    def forward(self, x):
        B, T, C = x.shape

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # (B, T, C) -> (B, n_head, T, head_size): each head is a
        # 64-dimensional subspace of the embedding
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # fused causal attention: softmax(q k^T / sqrt(d_k)) v with the
        # upper triangle masked out. dropout_p regularizes attention
        # weights during training only.
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=True,
        )

        # re-assemble heads: (B, n_head, T, hs) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


# =============================================================================
# PART 2: FEED-FORWARD NETWORK
# =============================================================================


class FeedForward(nn.Module):
    """
    Position-wise MLP: Linear(n_embd -> 4*n_embd) -> GELU -> Linear(back).

    PAPER: Vaswani et al. 2017, section 3.3 (the 4x expansion); GPT-2 for
    GELU instead of ReLU. GELU is x * Phi(x) - a smooth ReLU - and GPT-2/3
    all use it; the smoother corner trains slightly better at depth.
    """

    def __init__(self, config):
        super().__init__()
        self.fc = nn.Linear(config.N_EMBD, 4 * config.N_EMBD, bias=False)
        self.c_proj = nn.Linear(4 * config.N_EMBD, config.N_EMBD, bias=False)
        self.drop = nn.Dropout(config.DROPOUT)

    def forward(self, x):
        return self.drop(self.c_proj(F.gelu(self.fc(x))))


# =============================================================================
# PART 3: TRANSFORMER BLOCK
# =============================================================================


class TransformerBlock(nn.Module):
    """
    Pre-LN block (GPT-2):

        x = x + Attention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))

    LayerNorm BEFORE each sublayer (not after, as in the original
    Transformer) keeps a clean residual path from input to output, which is
    what lets 16-100+ layer stacks train without warm-up tricks. Same
    choice as chapter 1; listed again because it is load-bearing at depth.
    """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.N_EMBD)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.N_EMBD)
        self.ffwd = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# =============================================================================
# PART 4: COMPLETE GPT MODEL
# =============================================================================


class GPT(nn.Module):
    """
    The full decoder-only model (see module docstring for the shape).

    forward(idx, targets) returns (logits, loss) exactly like chapter 1;
    generate() does temperature/top-k sampling exactly like chapter 1.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.VOCAB_SIZE, config.N_EMBD),
                wpe=nn.Embedding(config.BLOCK_SIZE, config.N_EMBD),
                drop=nn.Dropout(config.DROPOUT),
                blocks=nn.ModuleList(
                    [TransformerBlock(config) for _ in range(config.N_LAYER)]
                ),
                ln_f=nn.LayerNorm(config.N_EMBD),
            )
        )
        self.lm_head = nn.Linear(config.N_EMBD, config.VOCAB_SIZE, bias=False)

        # Weight tying (Press & Wolf 2017; GPT-2): project to the vocabulary
        # with the SAME matrix that embeds tokens. Saves VOCAB_SIZE * N_EMBD
        # parameters (12.6M here) and ties input/output token geometry.
        self.lm_head.weight = self.transformer.wte.weight

        # mark residual projections BEFORE initializing: the init below
        # reads this flag to shrink their std
        for name, child in self.named_modules():
            if isinstance(child, nn.Linear) and name.endswith("c_proj"):
                child._is_residual = True

        self.apply(self._init_weights)
        print(f"Model initialized with {self.get_num_params(non_embedding=False):,} "
              f"parameters")

    def _init_weights(self, module):
        """
        GPT-2 initialization.

        - Linear/Embedding weights ~ Normal(0, 0.02); LayerNorm starts as
          identity (gamma=1, beta=0).
        - Residual-path projections (attn c_proj, ffn c_proj) get an extra
          1/sqrt(2*N_LAYER) shrink, from GPT-2's released code: with N
          layers adding to the residual stream, per-layer contributions are
          scaled so the stream's variance stays ~constant at initialization
          instead of growing like N.
        """
        if isinstance(module, nn.Linear):
            std = 0.02
            if getattr(module, "_is_residual", False):
                std /= (2 * self.config.N_LAYER) ** 0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Parameter count. With weight tying, wte IS lm_head, so it is counted
        once. non_embedding=True additionally excludes the (tied) embedding
        table and position table - the convention behind GPT-2's "117M
        non-embedding" vs 124M total.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def forward(self, idx, targets=None):
        """
        idx: (B, T) token ids, T <= BLOCK_SIZE
        targets: (B, T) next-token ids (optional); positions holding -1
            contribute nothing to the loss (ignore_index), which is how
            response-only fine-tuning masks prompts.

        Returns logits (B, T, VOCAB_SIZE) and the mean cross-entropy over
        the unmasked positions (None if no targets).
        """
        device = idx.device
        B, T = idx.shape
        assert T <= self.config.BLOCK_SIZE, (
            f"Sequence length {T} exceeds block size {self.config.BLOCK_SIZE}"
        )

        tok_emb = self.transformer.wte(idx)  # (B, T, C)
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        pos_emb = self.transformer.wpe(pos)  # (T, C), broadcast over batch
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.blocks:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # Fused linear+cross-entropy: never materializes the
            # (B, T, VOCAB_SIZE) fp32 logits tensor, which at vocab 16,384
            # is ~1/3 of the activation memory in the naive path.
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            logits = self.lm_head(x)
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Sample max_new_tokens tokens autoregressively.

        Same algorithm as chapter 1: crop context to BLOCK_SIZE, take the
        last position's logits, divide by temperature, optionally keep only
        the top_k logits, softmax and draw. (Fan et al., 2018 for top-k.)

        Note: this loop does NOT stop at <EOS> - the pre-training model had
        no reason to emit it mid-stream. SFT teaches the model to end its
        turn with <EOS>, so generate.py wraps its own sampler that listens
        for it; keep this method byte-compatible with project 4.
        """
        for _ in range(max_new_tokens):
            idx_crop = (
                idx
                if idx.size(1) <= self.config.BLOCK_SIZE
                else idx[:, -self.config.BLOCK_SIZE:]
            )
            logits, _ = self(idx_crop)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
