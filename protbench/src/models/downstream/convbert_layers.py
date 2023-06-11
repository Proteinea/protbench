import abc

from torch import nn
from functools import partial
import torch
import transformers.models.convbert as c_bert
from transformers.modeling_outputs import (
    TokenClassifierOutput,
    SequenceClassifierOutput,
)


class GlobalMaxPooling1D(nn.Module):
    def __init__(self):
        """
        Applies global max pooling over timesteps dimension
        """

        super().__init__()
        self.global_max_pool1d = partial(torch.max, dim=1)

    def forward(self, x):
        out, _ = self.global_max_pool1d(x)
        return out


class GlobalAvgPooling1D(nn.Module):
    def __init__(self):
        """
        Applies global average pooling over timesteps dimension
        """

        super().__init__()
        self.global_avg_pool1d = partial(torch.mean, dim=1)

    def forward(self, x):
        out = self.global_avg_pool1d(x)
        return out


class BaseConvBert(nn.Module):
    def __init__(
        self,
        input_dim: int,
        nhead: int,
        hidden_dim: int,
        num_layers: int = 1,
        kernel_size: int = 7,
        dropout: float = 0.2,
        pooling: str = None,
    ):
        """
        Base ConvBert encoder model.

        Args:
            input_dim: Dimension of the input embeddings.
            nhead: Integer specifying the number of heads for the `ConvBert` model.
            hidden_dim: Integer specifying the hidden dimension for the `ConvBert` model.
            num_layers: Integer specifying the number of layers for the `ConvBert` model.
            kernel_size: Integer specifying the filter size for the `ConvBert` model. Default: 7
            dropout: Float specifying the dropout rate for the `ConvBert` model. Default: 0.2
            pooling: String specifying the global pooling function. Accepts "avg" or "max". Default: "max"
        """
        super(BaseConvBert, self).__init__()

        config = c_bert.ConvBertConfig(
            hidden_size=input_dim,
            num_attention_heads=nhead,
            intermediate_size=hidden_dim,
            conv_kernel_size=kernel_size,
            num_hidden_layers=num_layers,
            hidden_dropout_prob=dropout,
        )
        self.transformer_encoder = c_bert.ConvBertModel(config).encoder

        if pooling is not None:
            if pooling in {"avg", "mean"}:
                self.pooling = GlobalAvgPooling1D()
            elif pooling == "max":
                self.pooling = GlobalMaxPooling1D()
            else:
                raise ValueError(
                    f"Expected pooling to be [`avg`, `max`]. Recieved: {pooling}"
                )

    def get_extended_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        This function is taken from the `ConvBertModel` implementation in the transformers library.
        See: https://github.com/huggingface/transformers/blob/fe861e578f50dc9c06de33cd361d2f625017e624/src/transformers/modeling_utils.py#L863

        It is used to extend the attention mask to work with ConvBert's implementation of self-attention.

        Args:
            attention_mask: Tensor of shape [batch_size, seq_len] containing ones in unmasked
                indices and zeros in masked indices.

        Returns:
            Tensor of extended attention mask that can be fed to the ConvBert model.
        """
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(
            attention_mask.dtype
        ).min
        return extended_attention_mask

    def convbert_forward(self, embd: torch.Tensor) -> torch.Tensor:
        attention_mask = (embd != 0).all(dim=-1)
        attention_mask = self.get_extended_attention_mask(attention_mask.to(embd.dtype))
        return self.transformer_encoder(embd, attention_mask=attention_mask)[0]


class ConvBertForTokenClassification(BaseConvBert, abc.ABC):
    def __init__(
        self,
        num_tokens: int,
        input_dim: int,
        nhead: int,
        hidden_dim: int,
        num_layers: int = 1,
        kernel_size: int = 7,
        dropout: float = 0.2,
    ):
        """
        ConvBert model for token classification classification task. Should be subclassed
        for multiclass and binary token classification tasks and implementing the proper
        _compute_loss function for each task.

        Args:
            num_tokens: Integer specifying the number of tokens that should be the output of the final layer.
            input_dim: Dimension of the input embeddings.
            nhead: Integer specifying the number of heads for the `ConvBert` model.
            hidden_dim: Integer specifying the hidden dimension for the `ConvBert` model.
            num_layers: Integer specifying the number of `ConvBert` layers.
            kernel_size: Integer specifying the filter size for the `ConvBert` model. Default: 7
            dropout: Float specifying the dropout rate for the `ConvBert` model. Default: 0.2
        """
        super(ConvBertForTokenClassification, self).__init__(
            input_dim=input_dim,
            nhead=nhead,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
            pooling=None,
        )

        self.num_labels = num_tokens
        self.decoder = nn.Linear(input_dim, num_tokens)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    @abc.abstractmethod
    def _compute_loss(self, logits, labels):
        pass

    def forward(self, embd, labels=None):
        hidden_inputs = self.convbert_forward(embd)
        logits = self.decoder(hidden_inputs)
        loss = self._compute_loss(logits, labels)

        return TokenClassifierOutput(
            loss=loss,  # type: ignore
            logits=logits,
            hidden_states=None,
            attentions=None,
        )


class ConvBertForSeqClassification(BaseConvBert, abc.ABC):
    def __init__(
        self,
        num_tokens: int,
        input_dim: int,
        nhead: int,
        hidden_dim: int,
        num_layers: int = 1,
        kernel_size: int = 7,
        dropout: float = 0.2,
        pooling: str = "max",
    ):
        super(ConvBertForSeqClassification, self).__init__(
            input_dim=input_dim,
            nhead=nhead,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
            pooling=pooling,
        )
        """
            ConvBert model for sequence classification classification task. Should be subclassed
            for multiclass and binary sequence classification tasks and implementing the proper
            _compute_loss function for each task.

            Args:
                num_tokens: Integer specifying the number of tokens that should be the output of the final layer.
                input_dim: Dimension of the input embeddings.
                nhead: Integer specifying the number of heads for the `ConvBert` model.
                hidden_dim: Integer specifying the hidden dimension for the `ConvBert` model.
                num_layers: Integer specifying the number of `ConvBert` layers.
                kernel_size: Integer specifying the filter size for the `ConvBert` model. Default: 7
                dropout: Float specifying the dropout rate for the `ConvBert` model. Default: 0.2
                pooling: String specifying the global pooling function. Accepts "avg" or "max". Default: "max"
        """

        self.num_labels = num_tokens
        self.decoder = nn.Linear(input_dim, num_tokens)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    @abc.abstractmethod
    def _compute_loss(self, logits, labels):
        pass

    def forward(self, embd, labels=None):
        hidden_inputs = self.convbert_forward(embd)
        hidden_inputs = self.pooling(hidden_inputs)
        logits = self.decoder(hidden_inputs)
        loss = self._compute_loss(logits, labels)

        return SequenceClassifierOutput(
            loss=loss, logits=logits, hidden_states=None, attentions=None  # type: ignore
        )
