from protbench.applications.benchmarking_task import BenchmarkingTask
from protbench.tasks import HuggingFaceSequenceToClass
from protbench.utils import collate_inputs, collate_sequence_and_labels
from protbench.models.heads import MultiClassClassificationHead
from protbench.models.downstream_models import DownstreamModelFromEmbedding
from protbench.models.downstream_models import (
    DownstreamModelWithPretrainedBackbone,
)
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import top_k_accuracy_score
import numpy as np
from functools import partial


def get_proteinea_remote_homology_dataset():
    train_data = HuggingFaceSequenceToClass(
        dataset_url="proteinea/remote_homology",
        seqs_col="primary",
        labels_col="fold_label",
        data_files=None,
        data_key="train",
    )

    val_data = HuggingFaceSequenceToClass(
        class_to_id=train_data.class_to_id,
        dataset_url="proteinea/remote_homology",
        seqs_col="primary",
        labels_col="fold_label",
        data_files=None,
        data_key="validation",
    )
    return train_data, val_data


supported_datasets = {"remote_homology": get_proteinea_remote_homology_dataset}


def compute_remote_homology_metrics(p, num_classes):
    prfs = precision_recall_fscore_support(
        p.label_ids, p.predictions.argmax(axis=1), average="macro"
    )

    return {
        "accuracy": accuracy_score(p.label_ids, p.predictions.argmax(axis=1)),
        "precision": prfs[0],
        "recall": prfs[1],
        "f1": prfs[2],
        "hits10": top_k_accuracy_score(
            p.label_ids, p.predictions, k=10, labels=np.arange(num_classes)
        ),
    }


class RemoteHomology(BenchmarkingTask):
    def __init__(self, dataset="remote_homology", from_embeddings=False, tokenizer=None):
        train_dataset, eval_dataset = supported_datasets[dataset]()
        metrics_fn = partial(compute_remote_homology_metrics,
                             num_classes=self.train_dataset.num_classes)
        if not from_embeddings:
            collate_fn = collate_sequence_and_labels(tokenizer=tokenizer)
        else:
            collate_fn = collate_inputs

        self.requires_pooling = True

        super().__init__(
            train_dataset,
            eval_dataset,
            None,
            collate_fn,
            metrics_fn,
            "eval_hits10",
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
                backbone_model, head, pooling
            )
        return model
