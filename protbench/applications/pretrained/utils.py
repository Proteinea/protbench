
from protbench.models.downstream_models import DownstreamModelFromEmbedding
from protbench.models.downstream_models import (
    DownstreamModelWithPretrainedBackbone,
)


def initialize_model(
    task,
    embedding_dim,
    from_embeddings,
    backbone=None,
    downstream_model=None,
    pooling=None,
    embedding_postprocessing_fn=None,
):
    head = task.get_task_head(embedding_dim=embedding_dim)
    if from_embeddings:
        model = DownstreamModelFromEmbedding(
            downstream_mpdel=downstream_model,
            head=head,
        )
    else:
        model = DownstreamModelWithPretrainedBackbone(
            backbone=backbone,
            head=head,
            pooling=pooling,
            embedding_postprocessing_fn=embedding_postprocessing_fn,
        )
    return model
