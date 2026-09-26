# ruff: noqa
"""
LoRA: Low-Rank Adaptation of Linear Layers

PAPER: "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al.,
2021, Microsoft) - freeze the pre-trained weight matrix W and learn a
rank-r update dW = B @ A instead, where A is r x k and B is d x r with
r << min(d, k):

    y = W x + (alpha / r) * B (A x)

INTUITION:
----------
Full fine-tuning changes W by some delta W. The paper's observation is
that this update has a low intrinsic rank for adaptation tasks: the
task-specific "directions" the model needs live in a tiny subspace of
the weight space. Parameterizing dW directly as B @ A (rank r) makes
that assumption structural and drops the trainable count from
d x k to r x (d + k). On GPT-3, matching full fine-tuning quality with
10,000x fewer trainable parameters is the paper's headline result.

WHY BEFORE HOW (what breaks without LoRA):
------------------------------------------
Full fine-tuning stores an AdamW moment pair per parameter, so optimizer
state alone costs 2x the model in fp32 (~1 GB for 126M params), and
every adapted task needs its own full weight copy. A rank-r adapter is
0.3% of the parameters here: moments become negligible, and one frozen
base checkpoint can serve many tasks as small adapter files.

IMPLEMENTATION (this file):
---------------------------
- LoRALinear replaces an nn.Linear(bias=False) IN PLACE: same parameter
  name ("weight"), same state-dict keys plus lora_A / lora_B, so a
  wrapped model loads the base checkpoint's tensors without translation
  and project 4/5 state dicts remain loadable.
- Init follows the paper's released code exactly: B = zeros (so the
  wrapped model computes EXACTLY what the base model computes at step
  0), A ~ kaiming-uniform (nn.Linear's default reset). With B = 0, A's
  distribution affects only the early gradient path, not the function.
- apply_lora(model, targets, r, alpha) walks named modules and swaps in
  LoRALinear wherever the fully-qualified name ENDS WITH a target string
  ("c_attn" here: the fused QKV projection of every block).
- merge() folds (alpha / r) * B @ A into W and flags the layer merged.
  The paper's deploy-time trick: the merged model has the original
  architecture and zero inference latency overhead. test_model.py
  verifies merged outputs match adapter outputs to float tolerance.
"""

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    A frozen Linear plus a trainable low-rank update.

    REPLACES (not wraps): apply_lora swaps this module for the original
    nn.Linear at the same attribute path. Parameter names are chosen so
    state dicts stay compatible:
        weight  - frozen copy of the original weight (requires_grad=False)
        lora_A  - (r, in_features), trainable
        lora_B  - (out_features, r), trainable, initialized to ZERO
    """

    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        assert base.bias is None, "LoRA here targets bias-free linears only"
        assert r > 0 and r <= min(base.in_features, base.out_features)

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.merged = False

        self.weight = nn.Parameter(base.weight.data.clone(), requires_grad=False)
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # lora_B stays zero: dW = B @ A = 0 at init, so the first forward
        # is bit-identical to the base model (Hu et al., 2021, section 4.1)

    def forward(self, x):
        out = F.linear(x, self.weight)
        if not self.merged:
            lora = F.linear(F.linear(x, self.lora_A), self.lora_B)
            out = out + self.scaling * lora
        return out

    @torch.no_grad()
    def merge(self) -> None:
        """
        Fold the trained update into the frozen weight: W += scaling * B@A.

        After this, forward() takes the plain path. The lora parameters
        remain (now redundant) so state-dict keys stay stable; the merged
        flag is what switches the math off.
        """
        if self.merged:
            return
        self.weight.data += self.scaling * (self.lora_B @ self.lora_A)
        self.merged = True


def apply_lora(model: nn.Module, targets: List[str], r: int, alpha: float):
    """
    Freeze the whole model, then replace every nn.Linear whose
    fully-qualified name ends with a target string with a LoRALinear
    wrapping it.

    Freezing happens FIRST so that exactly two parameter sets exist
    afterwards: frozen originals and fresh lora_A / lora_B (nn.Parameter
    defaults to requires_grad=True). This is the paper's setup - "we
    freeze the pre-trained model weights" - and the trainable-count test
    asserts nothing else escaped the freeze.

    Returns the list of replaced module names. Idempotent by construction:
    LoRALinear instances are not nn.Linear, so re-running matches nothing.
    """
    for parameter in model.parameters():
        parameter.requires_grad = False
    replaced = []
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and any(
            name.endswith(t) for t in targets
        ):
            parent_name, _, leaf = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, leaf, LoRALinear(module, r=r, alpha=alpha))
            replaced.append(name)
    for target in targets:
        if not any(name.endswith(target) for name in replaced):
            raise ValueError(f"LoRA target matched no module: {target}")
    return replaced


def trainable_parameters(model: nn.Module):
    """Parameters LoRA will train (lora_A / lora_B only)."""
    return [p for p in model.parameters() if p.requires_grad]


def lora_parameter_report(model: nn.Module):
    """(trainable count, total count) for the frozen-vs-trainable printout."""
    trainable = sum(p.numel() for p in trainable_parameters(model))
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
