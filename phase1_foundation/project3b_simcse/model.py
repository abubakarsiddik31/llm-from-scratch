# ruff: noqa
"""
SimCSE Model: Encoder Wrapper + Contrastive Objective

This file implements unsupervised SimCSE following:
- "SimCSE: Simple Contrastive Learning of Sentence Embeddings"
  (Gao, Yao & Chen, EMNLP 2021)

THE SIMCSE OBJECTIVE (paper, equation 4):
------------------------------------------
    l_i = -log exp(sim(h_i^z, h_i^z')/tau) / sum_j exp(sim(h_i^z, h_j^z')/tau)

where h_i^z = f_theta(x_i, z) is the embedding of sentence x_i under
dropout mask z. The positive pair (i, i) is the SAME sentence encoded
twice with independently sampled dropout masks; all other sentences in
the batch are negatives. sim(., .) is cosine similarity and tau is a
temperature.

PAPER CONTEXT (why it works):
-----------------------------
"dropout acts as minimal data augmentation, and removing it leads to
a representation collapse." If both copies use the same mask (or no
dropout), the two embeddings are identical, the model trivially
succeeds, and alignment degrades drastically. Independent dropout
masks force the model to learn dropout-invariant representations that
also spread out uniformly on the hypersphere.

Two properties matter (paper, section 3, "Why does it work?"):
- ALIGNMENT:    similar sentences map to nearby points (positives close)
- UNIFORMITY:   embeddings spread evenly on the unit hypersphere

IMPLEMENTATION:
---------------
1. ENCODER: the Project 3 BERT (bidirectional transformer, MLM
   pre-trained). We replicate its forward pass up to the final layer
   norm to expose hidden states (its own forward returns MLM/NSP
   logits, not hidden states).
2. POOLING: the [CLS] hidden state at position 0, as in the paper.
3. MLP HEAD: "an MLP layer (with one tanh activation) on top of the
   [CLS] representation" - used ONLY during training. At evaluation
   time the raw pre-projection embedding is used (the "MLP trick";
   improves STS-B by roughly 1-2 points).
4. LOSS: symmetric cross-entropy over the in-batch similarity matrix.

WHY: each piece mirrors the paper exactly so results transfer.
"""

import importlib.util
import os
import sys

import torch
import torch.nn as nn
from torch.nn import functional as F

import config


def load_checkpoint(path: str, map_location: str = "cpu") -> dict:
    """
    Load a checkpoint with weights_only=True (no arbitrary code execution).

    Project 3's checkpoints store loss values as numpy scalars, which are
    not in PyTorch's default weights-only allowlist; we explicitly
    allowlist that single scalar type. All other objects must still be
    plain tensors/primitives, so the loader stays safe.
    """
    try:
        import numpy as _np
        from numpy._core import multiarray as _np_ma
    except ImportError:  # numpy < 2.0
        import numpy as _np
        from numpy import core as _np_ma
    # numpy scalar objects (checkpoint loss values) carry their dtype
    # class and dtype instance; allowlist the plain numeric ones.
    safe = [
        _np.dtype,
        _np_ma.scalar,
        _np_ma._reconstruct,
        _np_ma.ndarray,
        _np.dtypes.Float16DType,
        _np.dtypes.Float32DType,
        _np.dtypes.Float64DType,
        _np.dtypes.Int8DType,
        _np.dtypes.Int16DType,
        _np.dtypes.Int32DType,
        _np.dtypes.Int64DType,
        _np.dtypes.UInt8DType,
        _np.dtypes.UInt16DType,
        _np.dtypes.UInt32DType,
        _np.dtypes.UInt64DType,
        _np.dtypes.BoolDType,
    ]
    torch.serialization.add_safe_globals(safe)
    return torch.load(path, map_location=map_location, weights_only=True)


# =============================================================================
# PART 1: LOAD THE PROJECT 3 ENCODER CLASS
# =============================================================================


def _load_project3_bert():
    """
    Import Project 3's BERT class without polluting this project's namespace.

    Project 3's model.py does a plain `import config`, so we temporarily
    swap sys.modules["config"] with Project 3's config module while
    loading, then restore ours. This keeps both projects' configurations
    independent while reusing the exact encoder implementation.

    Returns:
        (BERT class, project3 config module)
    """
    p3_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "project3_contextual_embeddings",
    )

    spec = importlib.util.spec_from_file_location(
        "_p3_config", os.path.join(p3_dir, "config.py")
    )
    p3_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p3_config)

    saved_config = sys.modules.get("config")
    sys.modules["config"] = p3_config
    try:
        spec = importlib.util.spec_from_file_location(
            "_p3_model", os.path.join(p3_dir, "model.py")
        )
        p3_model = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(p3_model)
    finally:
        if saved_config is not None:
            sys.modules["config"] = saved_config
        else:
            sys.modules.pop("config", None)

    return p3_model.BERT, p3_config


# =============================================================================
# PART 2: MLP PROJECTION HEAD (THE "MLP TRICK")
# =============================================================================


class MLPHead(nn.Module):
    """
    Projection head used ONLY during SimCSE training.

    PAPER: Gao et al. (2021), implementation details:
    ---------------------------------------------------------------------------
    "We use an MLP layer (with one tanh activation) on top of the
    [CLS] representation ... and use it only for training but not for
    evaluation."

    INTUITION:
    ----------
    The contrastive loss shapes the representation space. Applying that
    pressure through a trainable projection lets the underlying [CLS]
    embedding stay closer to the MLM pre-trained manifold (good initial
    alignment). At evaluation we drop the head and use the [CLS]
    embedding directly - this consistently scores better on STS-B.

    WHY: known as the "MLP trick"; the same construction is used by the
    official SimCSE codebase.
    """

    def __init__(self, n_embd: int):
        super().__init__()
        self.dense = nn.Linear(n_embd, n_embd)
        self.activation = nn.Tanh()

    def forward(self, cls_hidden: torch.Tensor) -> torch.Tensor:
        return self.activation(self.dense(cls_hidden))


# =============================================================================
# PART 3: SIMCSE ENCODER
# =============================================================================


class SimCSEEncoder(nn.Module):
    """
    Wraps the Project 3 BERT to produce sentence embeddings.

    PAPER CONTEXT:
    --------------
    The encoder f_theta(x, z) maps a sentence to an embedding; z is the
    random dropout state. We call the encoder TWICE per batch with the
    same inputs: because dropout masks are sampled independently on each
    forward pass, the two calls yield different views of the same
    sentence. No other augmentation is applied.

    IMPLEMENTATION:
    ---------------
    Input format follows Project 3 / BERT convention:
        [CLS] tokens... [SEP] [PAD]...
    We run the encoder stack manually (embeddings -> blocks -> ln_f) to
    expose hidden states, then pool:
        pooler="cls"  -> hidden state at [CLS] (position 0)  [paper default]
        pooler="mean" -> mean of non-padding hidden states

    NOTE ON PADDING: Project 3's attention does not implement padding
    masks, so [PAD] tokens participate in attention. This is consistent
    across both dropout views (identical inputs), so contrastive
    training is unaffected; mean pooling simply excludes PAD positions
    when averaging.
    """

    def __init__(self, p3_config, pooler: str = "cls"):
        super().__init__()
        self.p3_config = p3_config
        self.pooler = pooler
        self.bert = _load_project3_bert()[0](p3_config)
        self.mlp_head = MLPHead(p3_config.N_EMBD)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _hidden_states(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Run the BERT encoder stack and return final hidden states.

        Mirrors Project 3's BERT.forward() steps 1-6 (embeddings,
        transformer blocks, final layer norm) without the MLM/NSP heads.

        Args:
            idx: (B, T) token ids including [CLS]/[SEP]/[PAD]

        Returns:
            (B, T, N_EMBD) final hidden states
        """
        cfg = self.p3_config
        device = idx.device
        B, T = idx.shape
        assert T <= cfg.BLOCK_SIZE, f"T={T} exceeds BLOCK_SIZE={cfg.BLOCK_SIZE}"

        tok_emb = self.bert.embeddings.token(idx)
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        pos_emb = self.bert.embeddings.position(pos)
        seg_emb = self.bert.embeddings.segment(torch.zeros_like(idx))

        x = self.bert.embeddings.drop(tok_emb + pos_emb + seg_emb)
        for block in self.bert.blocks:
            x = block(x)
        x = self.bert.ln_f(x)
        return x

    def encode(
        self,
        idx: torch.Tensor,
        attention_mask: torch.Tensor = None,
        project: bool = False,
    ) -> torch.Tensor:
        """
        Encode sentences into embeddings.

        Args:
            idx: (B, T) token ids ([CLS] s [SEP] [PAD]...)
            attention_mask: (B, T) 1 for real tokens, 0 for [PAD].
                Required for mean pooling; ignored by CLS pooling.
            project: If True, apply the training-only MLP head.
                Training passes use project=True; evaluation uses the
                raw [CLS] embedding (paper's "MLP trick").

        Returns:
            (B, N_EMBD) sentence embeddings
        """
        hidden = self._hidden_states(idx)

        if self.pooler == "cls":
            emb = hidden[:, 0, :]
        elif self.pooler == "mean":
            if attention_mask is None:
                raise ValueError("mean pooling requires attention_mask")
            mask = attention_mask.unsqueeze(-1).float()
            emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            raise ValueError(f"Unknown pooler: {self.pooler}")

        if project:
            emb = self.mlp_head(emb)
        return emb

    # ------------------------------------------------------------------
    # Loading the MLM pre-trained checkpoint
    # ------------------------------------------------------------------

    def load_pretrained(self, checkpoint_path: str) -> None:
        """
        Load MLM pre-trained weights from a Project 3 checkpoint.

        Uses load_checkpoint(): weights-only loading (no arbitrary code
        execution) with a numpy-scalar allowlist for the loss value.

        Args:
            checkpoint_path: Path to a Project 3 checkpoint (.pt) that
                contains 'model_state_dict'
        """
        checkpoint = load_checkpoint(checkpoint_path)
        state_dict = checkpoint["model_state_dict"]
        missing, unexpected = self.bert.load_state_dict(state_dict, strict=False)
        # Only the two heads may be missing if checkpoint predates them;
        # anything else means an architecture mismatch.
        allowed_missing = {
            "mlm_head.weight",
            "mlm_head.bias",
            "nsp_head.weight",
            "nsp_head.bias",
        }
        bad_missing = [k for k in missing if k not in allowed_missing]
        if bad_missing:
            raise RuntimeError(f"Missing encoder parameters: {bad_missing}")
        print(
            f"✓ Loaded encoder from {checkpoint_path} "
            f"(iter={checkpoint.get('iter', '?')}, "
            f"loss={checkpoint.get('loss', checkpoint.get('val_loss', '?'))})"
        )


# =============================================================================
# PART 4: CONTRASTIVE LOSS
# =============================================================================


def simcse_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Unsupervised SimCSE contrastive loss (paper, equation 4).

    PAPER CONTEXT:
    --------------
    For a mini-batch of N sentences, each sentence's embedding h_i^z
    (view 1) should be close to its own re-encoding h_i^z' (view 2) and
    far from every other sentence's second view h_j^z'. The denominator
    sums over ALL j (including the positive i = j), which makes this a
    standard (N-way) cross-entropy over cosine similarities scaled by
    1/tau.

    IMPLEMENTATION:
    ---------------
    1. L2-normalize both views (cosine similarity = dot product).
    2. Similarity matrix S[i, j] = cos(z1_i, z2_j) / tau, shape (N, N).
    3. Labels are the identity: positive of row i is column i.
    4. Symmetrize by also computing the loss with views swapped, so
       both "directions" of the encoder are trained (standard practice).

    Args:
        z1: (N, D) view-1 embeddings (before normalization)
        z2: (N, D) view-2 embeddings (independent dropout masks)
        temperature: tau; lower = harder negatives (paper: 0.05)

    Returns:
        Scalar loss
    """
    z1 = F.normalize(z1, p=2, dim=-1)
    z2 = F.normalize(z2, p=2, dim=-1)

    # Cosine similarities scaled by temperature: (N, N)
    sim_12 = z1 @ z2.t() / temperature
    sim_21 = z2 @ z1.t() / temperature

    # Positive for row i is column i (same sentence, other view)
    labels = torch.arange(z1.size(0), device=z1.device)

    loss_12 = F.cross_entropy(sim_12, labels)
    loss_21 = F.cross_entropy(sim_21, labels)
    return (loss_12 + loss_21) / 2


def alignment_and_uniformity(
    z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.05
) -> tuple:
    """
    Diagnostic metrics from the paper (Wang & Isola, 2020 metrics, as
    used in SimCSE section 3 to explain WHY dropout-noise positives work).

    PAPER CONTEXT:
    --------------
    "starting from pre-trained checkpoints, all models greatly improve
    uniformity. However, the alignment of the two [degenerate] variants
    degrades drastically, while our unsupervised SimCSE keeps a steady
    alignment, thanks to the use of dropout noise."

    Args:
        z1, z2: (N, D) embeddings of the two views
        temperature: temperature for the uniformity metric

    Returns:
        (alignment, uniformity) - both lower is better
    """
    z1 = F.normalize(z1, p=2, dim=-1)
    z2 = F.normalize(z2, p=2, dim=-1)

    # Alignment: expected squared L2 distance between positive pairs
    alignment = (z1 - z2).norm(p=2, dim=-1).pow(2).mean().item()

    # Uniformity: log of expected Gaussian potential over all pairs
    all_z = torch.cat([z1, z2], dim=0)
    sq_pdist = torch.pdist(all_z, p=2).pow(2)
    uniformity = (-sq_pdist.mul(-2.0 * temperature).exp().mean().log()).item()

    return alignment, uniformity
