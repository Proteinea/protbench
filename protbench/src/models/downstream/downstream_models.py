from torch import nn
from torch.nn import functional as F
from transformers.modeling_outputs import SequenceClassifierOutput

from protbench.src.models.downstream import convbert_layers

from protbench.src.models.model_registry import DownstreamModelRegistry


@DownstreamModelRegistry.register("convbert_for_multiclass_token_classification")
class ConvBertForMultiClassTokenClassification(
    convbert_layers.ConvBertForTokenClassification
):
    def __init__(
        self,
        input_dim: int,
        num_tokens: int,
        nhead: int,
        hidden_dim: int,
        num_layers: int = 1,
        kernel_size: int = 7,
        dropout: float = 0.2,
        loss_ignore_index: int = -100,
    ):
        """
        ConvBert model for multiclass token classification task.

        Args:
            input_dim: Dimension of the input embeddings.
            num_tokens: Integer specifying the number of tokens that should be the output of the final layer.
            nhead: Integer specifying the number of heads for the `ConvBert` model.
            hidden_dim: Integer specifying the hidden dimension for the `ConvBert` model.
            num_layers: Integer specifying the number of `ConvBert` layers.
            kernel_size: Integer specifying the filter size for the `ConvBert` model. Default: 7
            dropout: Float specifying the dropout rate for the `ConvBert` model. Default: 0.2
            loss_ignore_index: Integer specifying the value of the labels to ignore in the loss function. Default: -100.
        """
        super(ConvBertForMultiClassTokenClassification, self).__init__(
            num_tokens=num_tokens,
            input_dim=input_dim,
            nhead=nhead,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
            loss_ignore_index=loss_ignore_index,
        )

    def _compute_loss(self, logits, labels):
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.num_labels),
                labels.view(-1),
                ignore_index=self.loss_ignore_index,
            )
        else:
            loss = None
        return loss


@DownstreamModelRegistry.register("convbert_for_binary_token_classification")
class ConvBertForBinaryTokenClassification(
    convbert_layers.ConvBertForTokenClassification
):
    def __init__(
        self,
        input_dim: int,
        nhead: int,
        hidden_dim: int,
        num_layers: int = 1,
        kernel_size: int = 7,
        dropout: float = 0.2,
        loss_ignore_index: int = -100,
    ):
        """
        ConvBert model for binary token classification task.

        Args:
            input_dim: Dimension of the input embeddings.
            nhead: Integer specifying the number of heads for the `ConvBert` model.
            hidden_dim: Integer specifying the hidden dimension for the `ConvBert` model.
            num_layers: Integer specifying the number of `ConvBert` layers.
            kernel_size: Integer specifying the filter size for the `ConvBert` model. Default: 7
            dropout: Float specifying the dropout rate for the `ConvBert` model. Default: 0.2
            loss_ignore_index: Integer specifying the value of the labels to ignore in the loss function. Default: -100.
        """
        super(ConvBertForBinaryTokenClassification, self).__init__(
            num_tokens=1,
            input_dim=input_dim,
            nhead=nhead,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
            loss_ignore_index=loss_ignore_index,
        )

    def _compute_loss(self, logits, labels):
        if labels is not None:
            mask = labels != self.loss_ignore_index
            loss = F.binary_cross_entropy_with_logits(
                logits.reshape(labels.shape)[mask], labels[mask].to(logits.dtype)
            )
        else:
            loss = None
        return loss


@DownstreamModelRegistry.register("convbert_for_multiclass_sequence_classification")
class ConvBertForMultiClassSeqClassification(
    convbert_layers.ConvBertForSeqClassification
):
    def __init__(
        self,
        input_dim: int,
        num_tokens: int,
        nhead: int,
        hidden_dim: int,
        num_layers: int = 1,
        kernel_size: int = 7,
        dropout: float = 0.2,
        pooling: str = "max",
    ):
        """
        ConvBert model for multiclass sequence classification task.

        Args:
            input_dim: Dimension of the input embeddings.
            num_tokens: Integer specifying the number of tokens that should be the output of the final layer.
            nhead: Integer specifying the number of heads for the `ConvBert` model.
            hidden_dim: Integer specifying the hidden dimension for the `ConvBert` model.
            num_layers: Integer specifying the number of `ConvBert` layers.
            kernel_size: Integer specifying the filter size for the `ConvBert` model. Default: 7
            dropout: Float specifying the dropout rate for the `ConvBert` model. Default: 0.2
        """
        super(ConvBertForMultiClassSeqClassification, self).__init__(
            num_tokens=num_tokens,
            input_dim=input_dim,
            nhead=nhead,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
            pooling=pooling,
        )

    def _compute_loss(self, logits, labels):
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
        else:
            loss = None
        return loss


@DownstreamModelRegistry.register("convbert_for_binary_sequence_classification")
class ConvBertForBinarySeqClassification(convbert_layers.ConvBertForSeqClassification):
    def __init__(
        self,
        input_dim: int,
        nhead: int,
        hidden_dim: int,
        num_layers: int = 1,
        kernel_size: int = 7,
        dropout: float = 0.2,
        pooling: str = "max",
    ):
        """
        ConvBert model for binary sequence classification task.

        Args:
            input_dim: Dimension of the input embeddings.
            nhead: Integer specifying the number of heads for the `ConvBert` model.
            hidden_dim: Integer specifying the hidden dimension for the `ConvBert` model.
            num_layers: Integer specifying the number of `ConvBert` layers.
            kernel_size: Integer specifying the filter size for the `ConvBert` model. Default: 7
            dropout: Float specifying the dropout rate for the `ConvBert` model. Default: 0.2
        """
        super(ConvBertForBinarySeqClassification, self).__init__(
            num_tokens=1,
            input_dim=input_dim,
            nhead=nhead,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
            pooling=pooling,
        )

    def _compute_loss(self, logits, labels):
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(
                logits.reshape(labels.shape), labels.to(logits.dtype)
            )
        else:
            loss = None
        return loss


@DownstreamModelRegistry.register("convbert_for_regression")
class ConvBertForRegression(convbert_layers.BaseConvBert):
    def __init__(
        self,
        input_dim: int,
        nhead: int,
        hidden_dim: int,
        num_layers: int = 1,
        kernel_size: int = 7,
        dropout: float = 0.2,
        pooling: str = "max",
        training_labels_mean: float = None,
    ):
        if pooling is None:
            raise ValueError(
                '`pooling` cannot be `None` in a regression task. Expected ["avg", "max"].'
            )

        super(ConvBertForRegression, self).__init__(
            input_dim=input_dim,
            nhead=nhead,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
            pooling=pooling,
        )
        """
            ConvBert model for regression task.

            Args:
                input_dim: Dimension of the input embeddings.
                nhead: Integer specifying the number of heads for the `ConvBert` model.
                hidden_dim: Integer specifying the hidden dimension for the `ConvBert` model.
                num_layers: Integer specifying the number of `ConvBert` layers.
                kernel_size: Integer specifying the filter size for the `ConvBert` model. Default: 7
                dropout: Float specifying the dropout rate for the `ConvBert` model. Default: 0.2
                pooling: String specifying the global pooling function. Accepts "avg" or "max". Default: "max"
                training_labels_mean: Float specifying the average of the training labels. Useful for faster and better training. Default: None
        """

        self.training_labels_mean = training_labels_mean
        self.decoder = nn.Linear(input_dim, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        if self.training_labels_mean is not None:
            self.decoder.bias.data.fill_(self.training_labels_mean)
        else:
            self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def _compute_loss(self, logits, labels):
        if labels is not None:
            loss = F.mse_loss(logits, labels.reshape(logits.shape))
        else:
            loss = None
        return loss

    def forward(self, embd, labels=None):
        hidden_inputs, attention_mask = self.convbert_forward(embd)
        hidden_inputs = self.pooling(hidden_inputs, attention_mask=attention_mask)
        logits = self.decoder(hidden_inputs)
        loss = self._compute_loss(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
