from typing import Tuple

import torch
import transformers.models.convbert as c_bert
from torch import nn

from protbench.models.pooling import GlobalAvgPooling1D
from protbench.models.pooling import GlobalMaxPooling1D


class ConvBert(nn.Module):
    def __init__(
        self,
        input_dim: int,
        nhead: int,
        hidden_dim: int,
        num_layers: int = 1,
        kernel_size: int = 7,
        dropout: float = 0.2,
        pooling: str | nn.Module = None,
    ):
        """
        Base ConvBert encoder model.

        Args:
            input_dim: Dimension of the input embeddings.
            nhead: Integer specifying the number of heads for the `ConvBert`
                   model.
            hidden_dim: Integer specifying the hidden dimension for the
                        `ConvBert` model.
            num_layers: Integer specifying the number of layers for the
                        `ConvBert` model.
            kernel_size: Integer specifying the filter size for the
                         `ConvBert` model. Default: 7
            dropout: Float specifying the dropout rate for the `ConvBert` model.
                     Default: 0.2
            pooling: String specifying the global pooling function.
                     Accepts "avg" or "max". Default: "max".
        """
        super().__init__()

        config = c_bert.ConvBertConfig(
            hidden_size=input_dim,
            num_attention_heads=nhead,
            intermediate_size=hidden_dim,
            conv_kernel_size=kernel_size,
            num_hidden_layers=num_layers,
            hidden_dropout_prob=dropout,
        )
        self.transformer_encoder = c_bert.ConvBertModel(config).encoder

        if pooling is not None and not isinstance(pooling, nn.Module):
            if pooling in {"avg", "mean"}:
                self.pooling = GlobalAvgPooling1D()
            elif pooling == "max":
                self.pooling = GlobalMaxPooling1D()
            else:
                raise ValueError(
                    "Expected pooling to be [`avg`, `max`]. "
                    f"Recieved: {pooling}."
                )
        elif isinstance(pooling, nn.Module):
            self.pooling = pooling
        else:
            self.pooling = None

    def get_extended_attention_mask(
        self, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """This function is taken from the `ConvBertModel` implementation in
           the transformers library.
        See: https://github.com/huggingface/transformers/blob/fe861e578f50dc9c06de33cd361d2f625017e624/src/transformers/modeling_utils.py#L863 # noqa

        It is used to extend the attention mask to work with ConvBert's
        implementation of self-attention.

        Args:
            attention_mask: Tensor of shape [batch_size, seq_len] containing ones in unmasked
                indices and zeros in masked indices.

        Returns:
            Tensor of extended attention mask that can be fed to the ConvBert model.
        """
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (
            1.0 - extended_attention_mask
        ) * torch.finfo(attention_mask.dtype).min
        return extended_attention_mask

    def forward(
        self, embd: torch.Tensor, attention_mask
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask.to(embd.dtype)
        )
        hidden_states = self.transformer_encoder(
            embd, attention_mask=extended_attention_mask
        )[0]
        if self.pooling is not None:
            hidden_states = self.pooling(hidden_states, attention_mask)
        return hidden_states
