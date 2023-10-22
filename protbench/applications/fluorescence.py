from protbench.applications.benchmarking_task import BenchmarkingTask
from protbench.tasks import HuggingFaceSequenceToValue
from protbench.utils import collate_inputs, collate_sequence_and_labels
from protbench import metrics
from protbench.models.heads import RegressionHead
from protbench.models.downstream_models import DownstreamModelFromEmbedding
from protbench.models.downstream_models import (
    DownstreamModelWithPretrainedBackbone,
)


def get_proteinea_fluorescence_dataset():
    train_data = HuggingFaceSequenceToValue(
        dataset_url="proteinea/fluorosence",
        data_files=None,
        data_key="train",
        seqs_col="primary",
        labels_col="log_fluorescence",
    )
    val_data = HuggingFaceSequenceToValue(
        dataset_url="proteinea/fluorosence",
        data_files=None,
        data_key="test",
        seqs_col="primary",
        labels_col="log_fluorescence",
    )
    return train_data, val_data


supported_datasets = {"fluorescence": get_proteinea_fluorescence_dataset}


def get_metrics():
    return lambda x: {
        "spearman": metrics.compute_spearman(x),
    }


class Fluorescence(BenchmarkingTask):
    def __init__(self, dataset="fluorescence", from_embeddings=False, tokenizer=None):
        train_dataset, eval_dataset = dataset[dataset]()
        metrics_fn = get_metrics()
        if not from_embeddings:
            collate_fn = collate_sequence_and_labels(tokenizer=tokenizer)
        else:
            collate_fn = collate_inputs
        super().__init__(
            train_dataset,
            eval_dataset,
            None,
            collate_fn,
            metrics_fn,
            "spearman",
            from_embeddings,
            tokenizer
        )

    def get_train_data(self):
        return self.train_dataset.data[0], self.train_dataset.data[1]

    def get_eval_data(self):
        return self.eval_dataset.data[0], self.eval_dataset.data[1]

    def get_downstream_model(
        self, backbone_model, embedding_dim, pooling=None
    ):
        head = RegressionHead(input_dim=embedding_dim)
        if self.from_embeddings:
            model = DownstreamModelFromEmbedding(backbone_model, head)
        else:
            model = DownstreamModelWithPretrainedBackbone(
                backbone_model, head, pooling
            )
        return model
