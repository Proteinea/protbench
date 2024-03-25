from protbench import metrics
from protbench.applications.benchmarking_task import BenchmarkingTask
from protbench.models.downstream_models import (
    DownstreamModelFromEmbedding,
    DownstreamModelWithPretrainedBackbone,
)
from protbench.models.heads import RegressionHead
from protbench.tasks import HuggingFaceSequenceToValue
from protbench.utils import collate_inputs, collate_sequence_and_labels
import pandas as pd
from scipy.stats import spearmanr
from transformers import EvalPrediction


def get_pli_dataset():
    train_data = HuggingFaceSequenceToValue(
        dataset_url="proteinea/pdbbind_pli",
        data_files="pdbbind_train.csv",
        data_key="train",
        seqs_col="protein",
        labels_col="affinity (pKd/pKi)",
    )
    val_data = HuggingFaceSequenceToValue(
        dataset_url="proteinea/pdbbind_pli",
        data_files="pdbbind_valid.csv",
        data_key="train",
        seqs_col="protein",
        labels_col="affinity (pKd/pKi)",
    )
    test_data = HuggingFaceSequenceToValue(
        dataset_url="proteinea/pdbbind_pli",
        data_files="casf_test.csv",
        data_key="train",
        seqs_col="protein",
        labels_col="affinity (pKd/pKi)",
    )
    return train_data, val_data, test_data


supported_datasets = {"pli": get_pli_dataset}


def compute_pli_metrics(p: EvalPrediction):
    spearmanr = metrics.compute_spearman(p)
    num_examples = p.label_ids.shape[0]
    error_bar = metrics.compute_error_bar_for_regression(
        spearman_corr=spearmanr, num_examples=num_examples
    )
    rmse = metrics.compute_rmse(p)
    pearson_corr = metrics.compute_pearsonr(p)
    return {
        "spearman": spearmanr,
        "error_bar": error_bar,
        "pearsonr": pearson_corr,
        "rmse": rmse,
    }


class PLI(BenchmarkingTask):
    def __init__(
        self,
        dataset="pli",
        from_embeddings=False,
        tokenizer=None,
        task_type=None,
    ):
        train_dataset, eval_dataset, test_dataset = supported_datasets[dataset]()
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
            metrics_fn=compute_pli_metrics,
            metric_for_best_model="eval_validation_spearman",
            from_embeddings=from_embeddings,
            tokenizer=tokenizer,
            requires_pooling=True,
            task_type=task_type,
        )

    def get_train_data(self):
        xs = []
        ys = []
        for x, y in zip(self.train_dataset.data[0], self.train_dataset.data[1]):
            if x is None or len(x) == 0:
                continue

            xs.append(x)
            ys.append(y)
        return xs, ys

    def get_eval_data(self):
        xs = []
        ys = []
        for x, y in zip(self.eval_dataset.data[0], self.eval_dataset.data[1]):
            if x is None or len(x) == 0:
                continue

            xs.append(x)
            ys.append(y)
        return xs, ys

    def get_test_data(self):
        xs = []
        ys = []
        for x, y in zip(self.test_dataset.data[0], self.test_dataset.data[1]):
            if x is None or len(x) == 0:
                continue
            xs.append(x)
            ys.append(y)
        return xs, ys

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
