from typing import Callable
from typing import Optional
from typing import Union

import torch
from torch import nn

from protbench.models.pooling import GlobalAvgPooling1D
from protbench.models.pooling import GlobalMaxPooling1D


class DownstreamModelFromEmbedding(nn.Module):
    def __init__(
        self,
        downstream_backbone: nn.Module,
        head: nn.Module,
        pad_token_id: Optional[int] = 0,
    ):
        """Initializes a downstream model using a backbone and a head.

        Args:
            downstream_backbone (nn.Module): downstream backbone. Should
                                             receive the last hidden state of
                                             the embedding model
                                             (batch_size, seq_len, embd_dim),
                                             and a padding mask of shape
                                             (batch_size, seq_len) as input and
                                             return a tensor of the same shape
                                             as the input embedding.
                Note: if pad_token_id is None, the padding mask will be None.
            head (nn.Module): Downstream head. Should receive the output of the
                              backbone as input of shape
                              (batch_size, seq_len, embd_dim).
            pad_token_id (int, optional): padding token id used to pad the
                                          input embeddings. Defaults to 0.
                                          If None, the padding mask passed to
                                          the backbone will be None.
        """
        super(DownstreamModelFromEmbedding, self).__init__()
        self.downstream_backbone = downstream_backbone
        self.head = head
        self.pad_token_id = pad_token_id

    def forward(
        self, embds: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Embed a batch of sequences.

        Args:
            embds (torch.Tensor): embeddings of shape
                                  (batch_size, seq_len, embd_dim).
            labels (Optional[torch.Tensor], optional): labels. Defaults to None.

        Returns:
            torch.Tensor: return the output of the head.
        """
        if self.pad_token_id is not None:
            attention_mask = (embds != self.pad_token_id).all(dim=-1)
        else:
            attention_mask = None
        hidden_states = self.downstream_backbone(embds, attention_mask)
        return self.head(hidden_states, labels=labels)


class DownstreamModelWithPretrainedBackbone(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        pooling: Union[Callable, nn.Module] = None,
    ) -> None:
        """Initializes a Downstream model with a head and
           pooling layer attached to it.

        Args:
            backbone (nn.Module): Backbone pretrained model.
            head (nn.Module): Head that is related to the specified task
            pooling (Union[Callable, nn.Module], optional): Pooling layer.
                                                            Defaults to None.
        """
        super(DownstreamModelWithPretrainedBackbone, self).__init__()
        self.backbone = backbone
        self.head = head

        if pooling == "max":
            self.pooling = GlobalMaxPooling1D()
        elif pooling == "avg":
            self.pooling = GlobalAvgPooling1D()
        else:
            self.pooling = None

    def forward(self, input_ids, attention_mask=None, labels=None):
        embeddings = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state

        if self.pooling is not None:
            embeddings = self.pooling(embeddings, attention_mask)

        return self.head(embeddings, labels)
