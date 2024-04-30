from __future__ import annotations

import torch
from torch import nn


def prepare_attention_mask(
    attention_mask: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    attention_mask = attention_mask.to(device=device, dtype=dtype)
    return attention_mask.unsqueeze(-1)


class GlobalMaxPooling1D(nn.Module):
    def __init__(self):
        """Applies global max pooling over timesteps dimension"""

        super().__init__()

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None
    ):
        """Forward pass of the global max pooling layer.

        Args:
            x (torch.Tensor): Input tensor of shape
                (batch_size, seq_len, embedding_dim).
            attention_mask (torch.Tensor, optional): Attention mask of shape
                (batch_size, seq_len). Defaults to None.
        """
        if attention_mask is not None:
            attention_mask = prepare_attention_mask(
                attention_mask=attention_mask,
                dtype=x.dtype,
                device=x.device,
            )
            x = x * attention_mask

        x = torch.amax(x, dim=1)
        return x


class GlobalAvgPooling1D(nn.Module):
    def __init__(self):
        """Applies global average pooling over timesteps dimension"""

        super().__init__()

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None
    ):
        """Forward pass of the global max pooling layer.

        Args:
            x (torch.Tensor): Input tensor of shape
                (batch_size, seq_len, embedding_dim).
            attention_mask (torch.Tensor, optional): Attention mask of shape
                (batch_size, seq_len). Defaults to None.
        """
        if attention_mask is not None:
            attention_mask = prepare_attention_mask(
                attention_mask=attention_mask,
                dtype=x.dtype,
                device=x.device,
            )
            x = x * attention_mask
            return torch.sum(x, dim=1) / torch.sum(attention_mask, dim=1)
        else:
            return torch.mean(x, dim=1)
