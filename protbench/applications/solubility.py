from transformers import EvalPrediction

from protbench import metrics
from protbench.applications.benchmarking_task import BenchmarkingTask
from protbench.models.heads import BinaryClassificationHead
from protbench.tasks import HuggingFaceSequenceToClass
from protbench.utils import collate_inputs
from protbench.utils import collate_sequence_and_labels
from protbench.utils import preprocess_binary_classification_logits


def get_solubility_dataset():
    train_data = HuggingFaceSequenceToClass(
        dataset_url="proteinea/solubility",
        data_files=None,
        data_key="train",
        seqs_col="sequences",
        labels_col="labels",
    )
    validation_data = HuggingFaceSequenceToClass(
        class_to_id=train_data.class_to_id,
        dataset_url="proteinea/solubility",
        data_files=None,
        data_key="validation",
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
    return train_data, validation_data, test_data


supported_datasets = {
    "solubility": get_solubility_dataset,
}


def compute_solubility_metrics(p: EvalPrediction):
    accuracies_std = metrics.compute_accuracies_std(p)
    num_examples = p.label_ids.shape[0]
    error_bar = metrics.compute_accuracies_error_bar(
        accuracies_std=accuracies_std, num_examples=num_examples
    )
    return {
        "accuracy": metrics.compute_accuracy(p),
        "precision": metrics.compute_precision(p, average="binary"),
        "recall": metrics.compute_recall(p, average="binary"),
        "f1": metrics.compute_f1(p, average="binary"),
        "accuracy_std": accuracies_std,
        "error_bar": error_bar,
    }


class Solubility(BenchmarkingTask):
    def __init__(
        self,
        dataset="solubility",
        from_embeddings=False,
        tokenizer=None,
        task_type=None,
    ):
        train_dataset, eval_dataset, test_dataset = supported_datasets[
            dataset
        ]()
        collate_fn = (
            collate_inputs
            if from_embeddings
            else collate_sequence_and_labels(tokenizer=tokenizer)
        )
        super().__init__(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            test_dataset=test_dataset,
            preprocessing_fn=preprocess_binary_classification_logits,
            collate_fn=collate_fn,
            metrics_fn=compute_solubility_metrics,
            metric_for_best_model="eval_validation_accuracy",
            from_embeddings=from_embeddings,
            tokenizer=tokenizer,
            requires_pooling=True,
            task_type=task_type,
        )

    def get_train_data(self):
        return self.train_dataset.data[0], self.train_dataset.data[1]

    def get_eval_data(self):
        return self.eval_dataset.data[0], self.eval_dataset.data[1]

    def get_test_data(self):
        return self.test_dataset.data[0], self.test_dataset.data[1]

    def get_task_head(self, embedding_dim):
        head = BinaryClassificationHead(input_dim=embedding_dim)
        return head
