from typing import Callable
from typing import Dict
from typing import Tuple

import torch
from peft import TaskType
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from transformers import EvalPrediction

from protbench.applications.benchmarking_task import BenchmarkingTask
from protbench.metrics import compute_accuracies_error_bar
from protbench.metrics import compute_accuracies_std
from protbench.models.heads import MultiClassClassificationHead
from protbench.tasks import HuggingFaceSequenceToClass
from protbench.utils import collate_inputs
from protbench.utils import collate_sequence_and_labels
from protbench.utils.preprocessing_utils import \
    preprocess_multi_classification_logits


def get_deeploc_dataset() -> Tuple:
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


supported_datasets = {
    "deeploc": get_deeploc_dataset,
}


def compute_deep_localization_metrics(p: EvalPrediction) -> Dict:
    prfs = precision_recall_fscore_support(
        p.label_ids, p.predictions, average="macro"
    )

    accuracies_std = compute_accuracies_std(p)
    num_examples = p.label_ids.shape[0]
    error_bar = compute_accuracies_error_bar(
        accuracies_std=accuracies_std, num_examples=num_examples
    )

    return {
        "accuracy": accuracy_score(p.label_ids, p.predictions),
        "precision": prfs[0],
        "recall": prfs[1],
        "f1": prfs[2],
        "accuracies_std": accuracies_std,
        "error_bar": error_bar,
    }


class DeepLoc(BenchmarkingTask):
    def __init__(
        self,
        dataset: str = "deeploc",
        from_embeddings: bool = False,
        tokenizer: Callable = None,
    ):
        train_dataset, eval_dataset = supported_datasets[dataset]()
        collate_fn = (
            collate_inputs
            if from_embeddings
            else collate_sequence_and_labels(tokenizer=tokenizer)
        )
        super().__init__(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            preprocessing_fn=preprocess_multi_classification_logits,
            collate_fn=collate_fn,
            metrics_fn=compute_deep_localization_metrics,
            metric_for_best_model="eval_validation_accuracy",
            from_embeddings=from_embeddings,
            tokenizer=tokenizer,
            requires_pooling=True,
            task_type=TaskType.SEQ_CLS,
        )

    def get_train_data(self) -> Tuple:
        return self.train_dataset.data[0], self.train_dataset.data[1]

    def get_eval_data(self):
        return self.eval_dataset.data[0], self.eval_dataset.data[1]

    def get_task_head(self, embedding_dim: int) -> torch.nn.Module:
        head = MultiClassClassificationHead(
            input_dim=embedding_dim, output_dim=self.get_num_classes()
        )
        return head
