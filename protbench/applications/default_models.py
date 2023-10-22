from protbench.models.convbert import ConvBert
from protbench.models.downstream_models import DownstreamModelFromEmbedding
from protbench.models.downstream_models import (
    DownstreamModelWithPretrainedBackbone,
)


def initialize_default_convbert_model_from_embedding(
    embedding_dim, pooling, head
):
    downstream_model = ConvBert(
        input_dim=embedding_dim,
        nhead=4,
        hidden_dim=int(embedding_dim / 2),
        num_layers=1,
        kernel_size=7,
        dropout=0.2,
        pooling=pooling,
    )
    model = DownstreamModelFromEmbedding(
        downstream_backbone=downstream_model, head=head
    )
    return model


def initialize_default_convbert_model(embedding_dim, pooling, head):
    downstream_model = ConvBert(
        input_dim=embedding_dim,
        nhead=4,
        hidden_dim=int(embedding_dim / 2),
        num_layers=1,
        kernel_size=7,
        dropout=0.2,
        pooling=pooling,
    )
    model = DownstreamModelWithPretrainedBackbone(
        downstream_backbone=downstream_model, head=head
    )
    return model


def initialize_default_model_with_pretrained_backbone(backbone, head, pooling):
    return DownstreamModelWithPretrainedBackbone(backbone, head, pooling)
