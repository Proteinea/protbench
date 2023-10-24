from protbench.applications.benchmarking_task import BenchmarkingTask
from protbench.tasks import HuggingFaceSequenceToClass
from protbench.utils import preprocess_binary_classification_logits
from protbench.utils import collate_inputs, collate_sequence_and_labels
from protbench import metrics
from protbench.models.heads import BinaryClassificationHead
from protbench.models.downstream_models import DownstreamModelFromEmbedding
from protbench.models.downstream_models import (
    DownstreamModelWithPretrainedBackbone,
)


def get_proteinea_solubility_dataset():
    train_data = HuggingFaceSequenceToClass(
        dataset_url="proteinea/solubility",
        data_files=None,
        data_key="train",
        seqs_col="sequences",
        labels_col="labels",
    )
    test_data = HuggingFaceSequenceToClass(
        class_to_id=train_data.class_to_id,
        dataset_url="proteinea/solubility",
        data_files=None,
        data_key="test",
        seqs_col="sequences",
        labels_col="labels",
    )
    return train_data, test_data


supported_datasets = {"solubility": get_proteinea_solubility_dataset}


def get_metrics():
    return lambda x: {
        "accuracy": metrics.compute_accuracy(x),
        "precision": metrics.compute_precision(x, average="binary"),
        "recall": metrics.compute_recall(x, average="binary"),
        "f1": metrics.compute_f1(x, average="binary"),
    }


class Solubility(BenchmarkingTask):
    def __init__(self, dataset="solubility", from_embeddings=False, tokenizer=None):
        train_dataset, eval_dataset = supported_datasets[dataset]()
        metrics_fn = get_metrics()
        if not from_embeddings:
            collate_fn = collate_sequence_and_labels(tokenizer=tokenizer)
        else:
            collate_fn = collate_inputs

        self.requires_pooling = True

        super().__init__(
            train_dataset,
            eval_dataset,
            preprocess_binary_classification_logits,
            collate_fn,
            metrics_fn,
            "accuracy",
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
        head = BinaryClassificationHead(
            input_dim=embedding_dim, output_dim=self.get_num_classes()
        )
        if self.from_embeddings:
            model = DownstreamModelFromEmbedding(backbone_model, head)
        else:
            model = DownstreamModelWithPretrainedBackbone(
                backbone_model, head, pooling
            )
        return model
