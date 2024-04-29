from protbench import metrics
from protbench.applications.benchmarking_task import BenchmarkingTask
from protbench.models.heads import RegressionHead
from protbench.tasks import HuggingFaceSequenceToValue
from protbench.utils import collate_inputs
from protbench.utils import collate_sequence_and_labels


def get_thermostability_dataset():
    train_data = HuggingFaceSequenceToValue(
        dataset_url="proteinea/thermostability",
        data_files="thermo_train.csv",
        data_key="train",
        seqs_col="sequence",
        labels_col="target",
    )
    val_data = HuggingFaceSequenceToValue(
        dataset_url="proteinea/thermostability",
        data_files="thermo_eval.csv",
        data_key="train",
        seqs_col="sequence",
        labels_col="target",
    )
    test_data = HuggingFaceSequenceToValue(
        dataset_url="proteinea/thermostability",
        data_files="thermo_test.csv",
        data_key="train",
        seqs_col="sequence",
        labels_col="target",
    )
    return train_data, val_data, test_data


supported_datasets = {
    "thermostability": get_thermostability_dataset,
}


def compute_thermostability_metrics(p):
    spearman = metrics.compute_spearman(p)
    num_examples = p.label_ids.shape[0]
    error_bar = metrics.compute_error_bar_for_regression(
        spearman_corr=spearman, num_examples=num_examples
    )
    return {
        "spearman": spearman,
        "error_bar": error_bar,
    }


class Thermostability(BenchmarkingTask):
    def __init__(
        self,
        dataset="thermostability",
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
            preprocessing_fn=None,
            collate_fn=collate_fn,
            metrics_fn=compute_thermostability_metrics,
            metric_for_best_model="eval_validation_spearman",
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
        head = RegressionHead(input_dim=embedding_dim)
        return head
