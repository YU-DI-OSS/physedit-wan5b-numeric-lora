"""Isolated prototype for Wan numeric conditioning.

This module deliberately does not patch the production model.  It defines the
trainable numeric projection and the exact full-key attention regularizer that
must be integrated after frozen text embeddings and before cross-attention.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class NumericBatch:
    values: Tensor          # [B, N, D], normalized scalar/vector components
    type_ids: Tensor        # [B, N]
    token_indices: Tensor   # [B, N], exact tokenizer positions; -1 is invalid
    component_mask: Tensor  # [B, N, D]


class HarmonicNumericEncoder(nn.Module):
    def __init__(self, text_dim: int = 4096, max_components: int = 4,
                 num_types: int = 256, harmonics: int = 8,
                 hidden_dim: int = 512) -> None:
        super().__init__()
        self.harmonics = harmonics
        self.max_components = max_components
        self.type_embedding = nn.Embedding(num_types, hidden_dim)
        per_component = 1 + 2 * harmonics
        self.project = nn.Sequential(
            nn.Linear(max_components * per_component + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, text_dim),
        )

    def forward(self, batch: NumericBatch) -> Tensor:
        x = batch.values * batch.component_mask
        frequencies = (2.0 ** torch.arange(
            self.harmonics, device=x.device, dtype=x.dtype
        )) * math.pi
        phase = x.unsqueeze(-1) * frequencies
        encoded = torch.cat((x.unsqueeze(-1), phase.sin(), phase.cos()), dim=-1)
        encoded = encoded * batch.component_mask.unsqueeze(-1)
        encoded = encoded.flatten(-2)
        typed = self.type_embedding(batch.type_ids)
        return self.project(torch.cat((encoded, typed), dim=-1))


def inject_numeric_residual(text_embeddings: Tensor, residuals: Tensor,
                            token_indices: Tensor) -> Tensor:
    """Out-of-place residual injection; invalid/truncated positions are ignored."""
    out = text_embeddings.clone()
    valid = token_indices.ge(0) & token_indices.lt(text_embeddings.shape[1])
    safe = token_indices.clamp(0, text_embeddings.shape[1] - 1)
    out.scatter_add_(1, safe.unsqueeze(-1).expand_as(residuals), residuals * valid.unsqueeze(-1))
    return out


def vector_direction_magnitude(vector: Tensor, eps: float = 1e-12) -> tuple[Tensor, Tensor]:
    magnitude = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    direction = torch.where(magnitude > eps, vector / magnitude.clamp_min(eps), torch.zeros_like(vector))
    return direction, magnitude


def numeric_attention_hinge(q: Tensor, k: Tensor, numeric_key_mask: Tensor,
                            valid_key_mask: Optional[Tensor] = None,
                            threshold: float = 0.05, scale: Optional[float] = None) -> Tensor:
    """Recompute attention with the *full* valid-key denominator.

    q: [B,H,Q,D], k: [B,H,K,D], masks: [B,K].  This helper is intended only
    for sampled layers/queries; production flash attention remains unchanged.
    """
    scale = scale if scale is not None else q.shape[-1] ** -0.5
    logits = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    if valid_key_mask is not None:
        logits = logits.masked_fill(~valid_key_mask[:, None, None, :], -torch.inf)
    weights = logits.softmax(dim=-1)
    focus = (weights * numeric_key_mask[:, None, None, :]).sum(dim=-1)
    return torch.relu(torch.as_tensor(threshold, device=focus.device) - focus).mean()
