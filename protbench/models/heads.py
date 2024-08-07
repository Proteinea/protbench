from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.modeling_outputs import TokenClassifierOutput


class MultiLabelClassifierOutpu(ModelOutput):
    # just a dummy class to ensure consistency
    pass


class TokenClassificationHead(torch.nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, ignore_index: int = -100
    ):
        """Initializes TokeClassificationHead instance.

        Args:
            input_dim (int): Input dimension
            output_dim (int): Output dimension
            ignore_index (int, optional): Index that will be ignored while
                                          calculating the loss.
                                          Defaults to -100.
        """
        super(TokenClassificationHead, self).__init__()
        self.output_dim = output_dim
        self.loss_ignore_index = ignore_index
        self.decoder = nn.Linear(input_dim, output_dim)
        self.reset_parameters()

    def reset_parameters(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor):
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.output_dim),
                labels.view(-1),
                ignore_index=self.loss_ignore_index,
            )
        else:
            loss = None
        return loss

    def forward(
        self, hidden_states: torch.Tensor, labels: torch.Tensor | None = None
    ):
        logits = self.decoder(hidden_states)
        loss = self.compute_loss(logits, labels)

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )


class BinaryClassificationHead(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim=1):
        super().__init__()
        self.decoder = nn.Linear(input_dim, output_dim)
        self.reset_parameters()

    def reset_parameters(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor):
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(
                logits, labels.to(logits.dtype).reshape(-1, 1)
            )
        else:
            loss = None
        return loss

    def forward(
        self, hidden_states: torch.Tensor, labels: torch.Tensor | None = None
    ):
        logits = self.decoder(hidden_states)
        loss = self.compute_loss(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )


class MultiLabelClassificationHead(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.output_dim = output_dim
        self.decoder = nn.Linear(input_dim, output_dim)
        self.reset_parameters()

    def reset_parameters(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor):
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(
                logits.view(-1, self.output_dim),
                labels.view(-1),
            )
        else:
            loss = None
        return loss

    def forward(
        self, hidden_states: torch.Tensor, labels: torch.Tensor | None = None
    ):
        logits = self.decoder(hidden_states)
        loss = self.compute_loss(logits, labels)

        return MultiLabelClassifierOutpu(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )


class MultiClassClassificationHead(torch.nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.decoder = nn.Linear(input_dim, output_dim)
        self.reset_parameters()

    def reset_parameters(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor):
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        else:
            loss = None
        return loss

    def forward(
        self, hidden_states: torch.Tensor, labels: torch.Tensor | None = None
    ):
        logits = self.decoder(hidden_states)
        loss = self.compute_loss(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )


class RegressionHead(torch.nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.decoder = nn.Linear(input_dim, 1)
        self.reset_parameters()

    def reset_parameters(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor):
        if labels is not None:
            loss = F.mse_loss(logits, labels.reshape(-1, 1))
        else:
            loss = None
        return loss

    def forward(
        self, hidden_states: torch.Tensor, labels: torch.Tensor | None = None
    ):
        logits = self.decoder(hidden_states)
        loss = self.compute_loss(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )


class ContactPredictionHead(torch.nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, ignore_index: int = -1
    ):
        super().__init__()

        self.num_labels = output_dim
        self.decoder = nn.Linear(input_dim * 2, output_dim)
        self.ignore_index = ignore_index
        self.reset_parameters()

    def reset_parameters(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor):
        if labels is None:
            return
        return F.cross_entropy(
            logits.view(-1, 2),
            labels.view(-1),
            ignore_index=self.ignore_index,
        )

    def forward(
        self, hidden_states: torch.Tensor, labels: torch.Tensor | None = None
    ):
        prod = hidden_states[:, :, None, :] * hidden_states[:, None, :, :]
        diff = hidden_states[:, :, None, :] - hidden_states[:, None, :, :]
        pairwise_features = torch.cat((prod, diff), -1)
        logits = self.decoder(pairwise_features)
        logits = (logits + logits.transpose(1, 2)) / 2
        loss = self.compute_loss(logits, labels)

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
