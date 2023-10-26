from protbench.models.convbert import ConvBert
from protbench.models.downstream_models import (
    DownstreamModelFromEmbedding,
    DownstreamModelWithPretrainedBackbone,
)


def initialize_convbert_model_from_embedding_with_default_hyperparameters(
    embedding_dim, head, pooling="max"
):
    downstream_model = ConvBert(
        input_dim=embedding_dim,
        nhead=4,
        hidden_dim=int(embedding_dim / 2),
        num_layers=1,
        kernel_size=7,
        dropout=0.1,
        pooling=pooling,
    )
    model = DownstreamModelFromEmbedding(
        downstream_backbone=downstream_model, head=head
    )
    return model


def initialize_convbert_model_with_default_hyperparameters(
    embedding_dim, head, pooling="max"
):
    downstream_model = ConvBert(
        input_dim=embedding_dim,
        nhead=4,
        hidden_dim=int(embedding_dim / 2),
        num_layers=1,
        kernel_size=7,
        dropout=0.1,
        pooling=pooling,
    )

    model = DownstreamModelWithPretrainedBackbone(
        downstream_backbone=downstream_model, head=head
    )
    return model
