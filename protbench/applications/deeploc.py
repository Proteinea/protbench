from protbench.applications.benchmarking_task import BenchmarkingTask
from protbench.tasks import HuggingFaceSequenceToClass
from protbench.utils import collate_inputs, collate_sequence_and_labels
from protbench.models.heads import MultiClassClassificationHead
from protbench.models.downstream_models import DownstreamModelFromEmbedding
from protbench.models.downstream_models import (
    DownstreamModelWithPretrainedBackbone,
)  # noqa
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score


def get_proteinea_deeploc_dataset():
    train_data = HuggingFaceSequenceToClass(
        dataset_url="proteinea/deeploc",
        seqs_col="input",
        labels_col="loc",
        data_files=None,
        data_key="train",
    )

    val_data = HuggingFaceSequenceToClass(
        class_to_id=train_data.class_to_id,
        dataset_url="proteinea/deeploc",
        seqs_col="input",
        labels_col="loc",
        data_files=None,
        data_key="test",
    )
    return train_data, val_data


supported_datasets = {"deeploc": get_proteinea_deeploc_dataset}


def compute_deep_localization_metrics(p):
    prfs = precision_recall_fscore_support(
        p.label_ids, p.predictions.argmax(axis=1), average="macro"
    )
    return {
        "accuracy": accuracy_score(p.label_ids, p.predictions.argmax(axis=1)),
        "precision": prfs[0],
        "recall": prfs[1],
        "f1": prfs[2],
    }


class DeepLoc(BenchmarkingTask):
    def __init__(self, dataset="deeploc", from_embeddings=False, tokenizer=None):
        train_dataset, eval_dataset = supported_datasets[dataset]()
        metrics_fn = compute_deep_localization_metrics
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
            "eval_accuracy",
            from_embeddings,
            tokenizer=tokenizer,
            requires_pooling=True,
        )

    def get_train_data(self):
        return self.train_dataset.data[0], self.train_dataset.data[1]

    def get_eval_data(self):
        return self.eval_dataset.data[0], self.eval_dataset.data[1]

    def get_downstream_model(
        self, backbone_model, embedding_dim, pooling=None
    ):
        head = MultiClassClassificationHead(
            input_dim=embedding_dim, output_dim=self.get_num_classes()
        )
        if self.from_embeddings:
            model = DownstreamModelFromEmbedding(backbone_model, head)
        else:
            model = DownstreamModelWithPretrainedBackbone(
                backbone=backbone_model, head=head, pooling=pooling
            )
        return model