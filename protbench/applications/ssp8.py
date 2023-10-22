from protbench.applications.benchmarking_task import BenchmarkingTask
from protbench.models.downstream_models import DownstreamModelFromEmbedding
from protbench.models.downstream_models import (
    DownstreamModelWithPretrainedBackbone,
)
from protbench.models.heads import TokenClassificationHead
from protbench.tasks import HuggingFaceResidueToClass
from protbench.utils.preprocessing_utils import (
    preprocess_multi_classification_logits,
)
from protbench import metrics
from protbench.utils import collate_inputs_and_labels, collate_sequence_and_labels


def preprocess_ssp_rows(seq, label, mask):
    mask = list(map(float, mask.split()))
    return seq, label, mask


def get_ssp8_casp12():
    train_data = HuggingFaceResidueToClass(
        dataset_url="proteinea/secondary_structure_prediction",
        data_files="training_hhblits.csv",
        data_key="train",
        seqs_col="input",
        labels_col="dssp8",
        mask_col="disorder",
        preprocessing_function=preprocess_ssp_rows,
    )

    val_data = HuggingFaceResidueToClass(
        class_to_id=train_data.class_to_id,
        dataset_url="proteinea/secondary_structure_prediction",
        data_files="CASP12.csv",
        data_key="train",
        seqs_col="input",
        labels_col="dssp8",
        mask_col="disorder",
        preprocessing_function=preprocess_ssp_rows,
    )
    return train_data, val_data


def get_ssp8_casp14():
    train_data = HuggingFaceResidueToClass(
        dataset_url="proteinea/secondary_structure_prediction",
        data_files="training_hhblits.csv",
        data_key="train",
        seqs_col="input",
        labels_col="dssp8",
        mask_col="disorder",
        preprocessing_function=preprocess_ssp_rows,
    )
    val_data = HuggingFaceResidueToClass(
        class_to_id=train_data.class_to_id,
        dataset_url="proteinea/secondary_structure_prediction",
        data_files="CASP14.csv",
        data_key="train",
        seqs_col="input",
        labels_col="dssp8",
        mask_col="disorder",
        preprocessing_function=preprocess_ssp_rows,
    )

    return train_data, val_data


def get_ssp8_cb513():
    train_data = HuggingFaceResidueToClass(
        dataset_url="proteinea/secondary_structure_prediction",
        data_files="training_hhblits.csv",
        data_key="train",
        seqs_col="input",
        labels_col="dssp8",
        mask_col="disorder",
        preprocessing_function=preprocess_ssp_rows,
    )
    val_data = HuggingFaceResidueToClass(
        class_to_id=train_data.class_to_id,
        dataset_url="proteinea/secondary_structure_prediction",
        data_files="CB513.csv",
        data_key="train",
        seqs_col="input",
        labels_col="dssp8",
        mask_col="disorder",
        preprocessing_function=preprocess_ssp_rows,
    )
    return train_data, val_data


def get_ssp8_ts115():
    train_data = HuggingFaceResidueToClass(
        dataset_url="proteinea/secondary_structure_prediction",
        data_files="training_hhblits.csv",
        data_key="train",
        seqs_col="input",
        labels_col="dssp8",
        mask_col="disorder",
        preprocessing_function=preprocess_ssp_rows,
    )
    val_data = HuggingFaceResidueToClass(
        class_to_id=train_data.class_to_id,
        dataset_url="proteinea/secondary_structure_prediction",
        data_files="TS115.csv",
        data_key="train",
        seqs_col="input",
        labels_col="dssp8",
        mask_col="disorder",
        preprocessing_function=preprocess_ssp_rows,
    )
    return train_data, val_data


available_datasets = {
    "ssp8_casp12": get_ssp8_casp12,
    "ssp8_casp14": get_ssp8_casp14,
    "ssp8_ts115": get_ssp8_ts115,
    "ssp8_cb513": get_ssp8_cb513,
}


def get_metrics_function():
    return lambda x: {
        "accuracy": metrics.compute_accuracy(x),
        "precision": metrics.compute_precision(x, average="macro"),
        "recall": metrics.compute_recall(x, average="macro"),
        "f1": metrics.compute_f1(x, average="macro"),
    }


class SSP8(BenchmarkingTask):
    def __init__(self, dataset, from_embeddings=False, tokenizer=None):
        train_dataset, eval_dataset = available_datasets[dataset]()
        metrics_fn = get_metrics_function()

        if not from_embeddings:
            collate_fn = collate_sequence_and_labels(tokenizer)
        else:
            collate_fn = collate_inputs_and_labels

        super().__init__(
            train_dataset,
            eval_dataset,
            preprocess_multi_classification_logits,
            collate_fn,
            metrics_fn,
            "accuracy",
            from_embeddings,
            tokenizer=tokenizer
        )

    def get_train_data(self):
        return self.train_dataset.data[0], self.train_dataset.data[1]

    def get_eval_data(self):
        return self.eval_dataset.data[0], self.eval_dataset.data[1]

    def get_downstream_model(self, backbone_model, embedding_dim, pooling=None):
        head = TokenClassificationHead(
            input_dim=embedding_dim, output_dim=self.get_num_classes()
        )
        if self.from_embeddings:
            model = DownstreamModelFromEmbedding(backbone_model, head)
        else:
            model = DownstreamModelWithPretrainedBackbone(backbone_model, head)
        return model
