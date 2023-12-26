from functools import partial
from typing import Optional

import torch
from torch import nn


class GlobalMaxPooling1D(nn.Module):
    def __init__(self):
        """Applies global max pooling over timesteps dimension"""

        super().__init__()
        self.global_max_pool1d = partial(torch.amax, dim=1)

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ):
        """Forward pass of the global max pooling layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embedding_dim).
            attention_mask (torch.Tensor, optional): Attention mask of shape (batch_size, seq_len). Defaults to None.
        """
        if attention_mask is not None:
            attention_mask = attention_mask.bool()
            # fill masked valBinaryClassificationHeadues with -inf so that they are not selected by max pooling
            x = x.masked_fill(~attention_mask.unsqueeze(-1), float("-inf"))
        out = self.global_max_pool1d(x)
        return out


class GlobalAvgPooling1D(nn.Module):
    def __init__(self):
        """Applies global average pooling over timesteps dimension"""

        super().__init__()
        self.global_avg_pool1d = partial(torch.mean, dim=1)
        self.global_avg_pool1d_with_nan = partial(torch.nanmean, dim=1)

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ):
        """Forward pass of the global max pooling layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embedding_dim).
            attention_mask (torch.Tensor, optional): Attention mask of shape (batch_size, seq_len). Defaults to None.
        """
        if attention_mask is not None:
            if torch.isnan(x).any():
                # if there are nan values in x, use mean to propagate the nan forward
                out = self.global_avg_pool1d(x)
            else:
                # fill masked values with nan so that they don't affect torch.nanmean
                attention_mask = attention_mask.bool()
                x = x.masked_fill(~attention_mask.unsqueeze(-1), torch.nan)
                out = self.global_avg_pool1d_with_nan(x)
        else:
            out = self.global_avg_pool1d(x)
        return out
