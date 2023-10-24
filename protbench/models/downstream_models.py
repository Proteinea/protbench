from typing import Optional

import torch
from protbench.models.pooling import GlobalAvgPooling1D, GlobalMaxPooling1D


class DownstreamModelFromEmbedding(torch.nn.Module):
    def __init__(
        self,
        downstream_backbone: torch.nn.Module,
        head: torch.nn.Module,
        pad_token_id: Optional[int] = 0,
    ) -> None:
        """Constrcut a downstream model using a backbone and a head.

        Args:
            downstream_backbone (torch.nn.Module): downstream backbone. Should receive
                the last hidden state of the embedding model (batch_size, seq_len, embd_dim), and a padding mask of shape (batch_size, seq_len)
                as input and return a tensor of the same shape as the input embedding.
                Note: if pad_token_id is None, the padding mask will be None.
            head (torch.nn.Module): downstream head. Should receive the output of the backbone
                as input of shape (batch_size, seq_len, embd_dim).
            pad_token_id (int, optional): padding token id used to pad the input embeddings. Defaults to 0.
                If None, the padding mask passed to the backbone will be None.
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
            embds (torch.Tensor): embeddings of shape (batch_size, seq_len, embd_dim).
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


class DownstreamModelWithPretrainedBackbone(torch.nn.Module):
    def __init__(self, backbone, head, pooling=None) -> None:
        super(DownstreamModelWithPretrainedBackbone, self).__init__()
        self.backbone = backbone
        self.head = head
        if pooling == 'max':
            self.pooling = GlobalMaxPooling1D()
        elif pooling == 'avg':
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
