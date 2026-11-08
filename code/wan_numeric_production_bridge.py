"""Explicit, side-effect-free Wan numeric-conditioning integration helpers.

Production integration should thread the returned regularizer through
DiTBlock -> WanModel -> training loss.  It must not accumulate hook state,
because Wan uses non-reentrant gradient checkpointing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from wan_numeric_conditioning import (
    HarmonicNumericEncoder,
    NumericBatch,
    inject_numeric_residual,
    numeric_attention_hinge,
)


@dataclass
class NumericConditioningOutput:
    context: Tensor
    text_numeric_mask: Tensor


class WanNumericConditioner(nn.Module):
    """Trainable module saved alongside LoRA; backbone/text encoder stay frozen."""

    def __init__(self, model_dim: int, max_components: int = 4,
                 num_types: int = 256, harmonics: int = 8,
                 hidden_dim: int = 512) -> None:
        super().__init__()
        self.encoder = HarmonicNumericEncoder(
            text_dim=model_dim,
            max_components=max_components,
            num_types=num_types,
            harmonics=harmonics,
            hidden_dim=hidden_dim,
        )

    def forward(self, projected_text: Tensor, batch: NumericBatch) -> NumericConditioningOutput:
        residual = self.encoder(batch).to(projected_text.dtype)
        context = inject_numeric_residual(projected_text, residual, batch.token_indices)
        mask = torch.zeros(
            projected_text.shape[:2], dtype=torch.bool, device=projected_text.device
        )
        valid = batch.token_indices.ge(0) & batch.token_indices.lt(projected_text.shape[1])
        mask.scatter_(1, batch.token_indices.clamp(0, projected_text.shape[1] - 1), valid)
        return NumericConditioningOutput(context=context, text_numeric_mask=mask)


def cross_attention_with_numeric_regularizer(
    module: nn.Module,
    x: Tensor,
    y: Tensor,
    text_numeric_mask: Tensor,
    query_indices: Tensor,
    threshold: float,
) -> tuple[Tensor, Tensor]:
    """Run the unmodified efficient attention and recompute only sampled Q rows.

    The denominator includes every text key used by the real text-attention
    branch. Image keys are a separate Wan attention branch and are therefore
    intentionally excluded from the text-token focus denominator.
    """
    if module.has_image_input:
        image_context, text_context = y[:, :257], y[:, 257:]
    else:
        image_context, text_context = None, y

    q_full = module.norm_q(module.q(x))
    k_text = module.norm_k(module.k(text_context))
    v_text = module.v(text_context)
    output = module.attn(q_full, k_text, v_text)
    if image_context is not None:
        k_image = module.norm_k_img(module.k_img(image_context))
        v_image = module.v_img(image_context)
        output = output + module.attn(q_full, k_image, v_image)
    output = module.o(output)

    bsz, _, width = q_full.shape
    heads = module.num_heads
    head_dim = width // heads
    gather_index = query_indices[:, :, None].expand(-1, -1, width)
    q_sample = q_full.gather(1, gather_index).reshape(bsz, -1, heads, head_dim).transpose(1, 2)
    k_heads = k_text.reshape(bsz, -1, heads, head_dim).transpose(1, 2)
    regularizer = numeric_attention_hinge(
        q_sample,
        k_heads,
        text_numeric_mask,
        valid_key_mask=torch.ones_like(text_numeric_mask),
        threshold=threshold,
        scale=head_dim ** -0.5,
    )
    return output, regularizer


def numeric_checkpoint_payload(conditioner: WanNumericConditioner, optimizer,
                               schema: dict, normalization: dict) -> dict:
    return {
        "numeric_conditioner": conditioner.state_dict(),
        "numeric_optimizer": optimizer.state_dict(),
        "numeric_schema": schema,
        "numeric_normalization": normalization,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
