"""LoRA adapters for the post-hoc loop (--lora-rank).

SOURCE: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models",
ICLR 2022 (arXiv:2106.09685). A wrapped linear computes
    y = W x + b + scaling * B (A x),   A: (rank, in), B: (out, rank)
with A ~ Kaiming-uniform and B = 0, so AT INIT the wrapped model is
bit-exact to the base. We fix
scaling = 1 (alpha = rank) so the SWEEP KNOB is rank alone — capacity as
the implicit anchor — instead of coupling rank with a rescale.

Injection targets: QIKT (the model the paper runs LoRA on) — the output
linears of its MLP heads and the question-concept fusion linear. The base
weights are frozen; only A/B train.
"""
import math

import torch
import torch.nn as nn


LORA_TARGET_ATTRIBUTES = ("out", "que_c_linear")


class LoRALinear(nn.Module):
    def __init__(self, base_linear, rank):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad_(False)
        in_features = base_linear.in_features
        out_features = base_linear.out_features
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        # LoRA paper init: A Kaiming-uniform, B zero -> identity at init
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + (x @ self.lora_A.T) @ self.lora_B.T


def inject_lora(model, rank):
    """Wrap every target linear in-place; freeze all non-adapter params.
    Returns the list of trainable (adapter) parameters."""
    device = next(model.parameters()).device
    # collect first, replace after — never mutate the tree mid-traversal
    replacements = [(module, attribute, child)
                    for module in model.modules()
                    for attribute in LORA_TARGET_ATTRIBUTES
                    for child in [getattr(module, attribute, None)]
                    if isinstance(child, nn.Linear)]
    assert replacements, ("no LoRA target linears in this model "
                          f"(attributes {LORA_TARGET_ATTRIBUTES})")
    for module, attribute, child in replacements:
        setattr(module, attribute, LoRALinear(child, rank))
    model.to(device)          # new adapter params join the model's device
    for name, p in model.named_parameters():
        p.requires_grad_("lora_A" in name or "lora_B" in name)
    return [p for p in model.parameters() if p.requires_grad]
