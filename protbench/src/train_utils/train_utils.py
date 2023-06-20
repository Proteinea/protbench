from functools import partial
from typing import List, Dict, Type, Any, Tuple, Callable

import torch
from torch.utils.data import Dataset
from transformers import EvalPrediction, TrainingArguments

from protbench.src.tasks import Task, TaskRegistry
from protbench.src.metrics import MetricRegistry
from protbench.src.models import DownstreamModelRegistry, PretrainedModelRegistry
from protbench.src.models.pretrained import BasePretrainedModel
from protbench.src.train_utils import EmbeddingsDataset


def collate_inputs(
    features: List[Dict[str, torch.Tensor]], padding_value: int = 0
) -> Dict[str, torch.Tensor]:
    """Collate a list of features into a batch. This function only pads the embeddings.

    Args:
        features (List[Dict[str, torch.Tensor]]): The features are expected to be a list of
            dictionaries with the keys "embd" and "labels"
        padding_value (int, optional): the padding value used. Defaults to 0.

    Returns:
        Dict[str, torch.Tensor]: A dictionary with the keys "embd" and "labels" containing
            the padded embeddings and labels tensors.
    """
    embds = [example["embd"] for example in features]
    labels = [example["labels"] for example in features]
    embds = torch.nn.utils.rnn.pad_sequence(
        embds, batch_first=True, padding_value=padding_value
    )
    return {"embd": embds, "labels": torch.tensor(labels)}


def collate_inputs_and_labels(
    features: List[Dict[str, torch.Tensor]],
    input_padding_value: int = 0,
    label_padding_value: int = -100,
) -> Dict[str, torch.Tensor]:
    """Collate a list of features into a batch. This function pads both the embeddings and the labels.

    Args:
        features (List[Dict[str, torch.Tensor]]): The features are expected to be a list of
            dictionaries with the keys "embd" and "labels"
        input_padding_value (int, optional): the padding value used for the embeddings. Defaults to 0.
        label_padding_value (int, optional): the padding value used for labels. Defaults to -100.

    Returns:
        Dict[str, torch.Tensor]: _description_
    """
    embds = [example["embd"] for example in features]
    labels = [example["labels"] for example in features]
    embds = torch.nn.utils.rnn.pad_sequence(
        embds, batch_first=True, padding_value=input_padding_value
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=label_padding_value
    )
    return {"embd": embds, "labels": labels}


def preprocess_multi_classification_logits(logits: torch.Tensor, _) -> torch.Tensor:
    """
    Preprocess logits for multiclassification tasks to produce predictions.

    Args:
        logits (torch.Tensor): logits from the model (batch_size, seq_len, num_classes)
            for token classification tasks or (batch_size, num_classes) for sequence classification tasks.
    Returns:
        torch.Tensor: predictions with shape (batch_size, seq_len) for token classification
            tasks or (batch_size,) for sequence classification tasks.
    """
    return logits.argmax(dim=-1)


def preprocess_binary_classification_logits(logits: torch.Tensor, _) -> torch.Tensor:
    """
    Preprocess logits for binary classification tasks to produce predictions.

    Args:
        logits (torch.Tensor): logits from the model (batch_size, seq_len, 1)
            for token classification tasks or (batch_size, 1) for sequence classification tasks.
    """
    return (torch.sigmoid_(logits) > 0.5).to(int)


class TrainUtils:
    OPTIMIZERS = {
        "Adadelta": torch.optim.Adadelta,
        "Adagrad": torch.optim.Adagrad,
        "Adam": torch.optim.Adam,
        "AdamW": torch.optim.AdamW,
        "SparseAdam": torch.optim.SparseAdam,
        "Adamax": torch.optim.Adamax,
        "ASGD": torch.optim.ASGD,
        "NAdam": torch.optim.NAdam,
        "RMSprop": torch.optim.RMSprop,
        "Rprop": torch.optim.Rprop,
        "SGD": torch.optim.SGD,
    }
    LR_SCHEDULERS = {
        "StepLR": torch.optim.lr_scheduler.StepLR,
        "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR,
        "LinearLR": torch.optim.lr_scheduler.LinearLR,
        "ConstantLR": torch.optim.lr_scheduler.ConstantLR,
        "MultiStepLR": torch.optim.lr_scheduler.MultiStepLR,
    }

    def __init__(self, config: Dict[str, Any]):
        """Utility class for training."""
        self.config = config
        self.task = self._get_task_from_config()
        self.downstream_model = self._get_downstream_model_from_config()
        self.pretrained_model = self._get_pretrained_model_from_config()
        self.training_args = self._get_training_args_from_config()
        self.collator_fn = self._get_suitable_collator()
        self.logits_preprocessor_fn = self._get_logits_preprocessor_fn()
        self.device = self._get_device()
        self._metrics = self._get_metrics_from_config()

    def get_optimizers(
        self, model: torch.nn.Module
    ) -> Tuple[
        torch.optim.Optimizer | None, torch.optim.lr_scheduler.LRScheduler | None
    ]:
        """Get the optimizer and learning rate scheduler (if set in config) from the config.

        Args:
            model (torch.nn.Module): The model to optimize.

        Returns:
            Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler | None]: The optimizer and
                learning rate scheduler. The learning rate scheduler is None if not set in the config.
        """
        if "optimizer" in self.config:
            optimizer_name = self.config["optimizer"]["name"]
            optimizer_config = (
                self.config["optimizer"]["kwargs"]
                if "kwargs" in self.config["optimizer"]
                else {}
            )
            optimizer = self.OPTIMIZERS[optimizer_name](
                model.parameters(), **optimizer_config
            )
        else:
            if "lr_scheduler" in self.config:
                raise ValueError(
                    "lr_scheduler cannot be set without optimizer in config."
                )
            optimizer = None

        if "lr_scheduler" in self.config:
            lr_name = self.config["lr_scheduler"]["name"]
            lr_config = (
                self.config["lr_scheduler"]["kwargs"]
                if "kwargs" in self.config["lr_scheduler"]
                else {}
            )
            lr_scheduler = self.LR_SCHEDULERS[lr_name](optimizer, **lr_config)
        else:
            lr_scheduler = None
        return optimizer, lr_scheduler

    def compute_metrics(self, p: EvalPrediction) -> Dict[str, float]:
        """Function used with the 🤗 Trainer to compute metrics.

        Args:
            p (EvalPrediction): the prediction object.

        Returns:
            Dict[str, float]: dictionary of computed metrics.
        """
        computed_metrics = {}
        for metric_name, metric_fn in self._metrics.items():
            computed_metrics[metric_name] = metric_fn(p)
        return computed_metrics

    def create_embd_datasets(
        self,
    ) -> Tuple[EmbeddingsDataset, EmbeddingsDataset | None]:
        """Create the datasets of embeddings for training and validation.

        Returns:
            Tuple[EmbeddingsDataset, EmbeddingsDataset | None]: training and validation datasets.
                The validation dataset is None if no validation data is provided.
        """
        train_data = self.task.train_data
        val_data = self.task.val_data
        maxlen = self.config["general"]["sequence_max_length"]

        train_embds = self.pretrained_model.embed_sequences(
            sequences=[sample["sequence"][:maxlen] for sample in train_data],
            device=self.device,
        )

        if self.task.task_description.task_type["entity"] == "token":
            # truncate labels as well if entity is token
            labels = [sample["label"][:maxlen] for sample in train_data]
        else:
            labels = [sample["label"] for sample in train_data]

        train_dataset = EmbeddingsDataset(
            embeddings=train_embds,
            labels=labels,
        )

        if val_data:
            val_embds = self.pretrained_model.embed_sequences(
                sequences=[sample["sequence"][:maxlen] for sample in val_data],
                device=self.device,
            )
            if self.task.task_description.task_type["entity"] == "token":
                labels = [sample["label"][:maxlen] for sample in val_data]
            else:
                labels = [sample["label"] for sample in val_data]
            val_dataset = EmbeddingsDataset(
                embeddings=val_embds,
                labels=labels,
            )
        else:
            val_dataset = None

        return train_dataset, val_dataset

    def log_info(self):
        print("*" * 100)
        print(f"Run Name: {self.training_args.run_name}")
        print("-" * 100)
        print(f"Task Details:\n{self.task.task_description}")
        print("Number of training samples:", len(self.task.train_data))
        print("Number of validation samples:", len(self.task.val_data))
        print("-" * 100)
        print(f"Pretrained Model: {self.config['pretrained_model']['name']}")
        print(
            f"Number of parameters: {self.pretrained_model.get_number_of_parameters():,}"
        )
        print("-" * 100)
        print(f"Downstream Model: {self.config['downstream_model']['name']}")
        print(f"Number of parameters: {self.get_num_params(self.downstream_model):,}")
        print("*" * 100)

    def get_num_params(self, model):
        return sum(p.numel() for p in model.parameters())

    def delete_pretrained_model(self):
        del self.pretrained_model
        torch.cuda.empty_cache()

    def _get_device(self):
        """Get device to use for embeddings extraction."""
        if self.training_args.no_cuda or not torch.cuda.is_available():
            # user deliberately wants to use CPU or no GPU is not available
            device = torch.device("cpu")
        else:  # use first GPU available
            device = torch.device("cuda:0")
        return device

    def _get_task_from_config(self) -> Task:
        """Get the task from the config.

        Args:
            config (Dict[str, str]): The config dictionary.

        Returns:
            Task: The task.
        """
        task_name = self.config["task"]["name"]
        task_config = (
            self.config["task"]["kwargs"] if "kwargs" in self.config["task"] else {}
        )
        return TaskRegistry.task_name_map[task_name](**task_config)

    def _get_pretrained_model_from_config(
        self,
    ) -> BasePretrainedModel | torch.nn.Module:
        """Get the pretrained model from the config.

        Args:
            config (Dict[str, str]): The config dictionary.

        Returns:
            PretrainedModel: The pretrained model.
        """
        pretrained_model_name = self.config["pretrained_model"]["name"]
        pretrained_model_config = (
            self.config["pretrained_model"]["kwargs"]
            if "kwargs" in self.config["pretrained_model"]
            else {}
        )
        return PretrainedModelRegistry.pretrained_model_name_map[pretrained_model_name](
            **pretrained_model_config
        )

    def _get_downstream_model_from_config(self) -> torch.nn.Module:
        """Get the downstream model from the config.

        Returns:
            torch.nn.Module: The downstream model.
        """
        downstream_model_name = self.config["downstream_model"]["name"]
        downstream_model_config = (
            self.config["downstream_model"]["kwargs"]
            if "kwargs" in self.config["downstream_model"]
            else {}
        )
        return DownstreamModelRegistry.downstream_model_name_map[downstream_model_name](
            **downstream_model_config
        )

    def _get_training_args_from_config(self) -> TrainingArguments:
        """Get the training arguments from the config.

        Returns:
            TrainingArguments: The training arguments.
        """
        return TrainingArguments(**self.config["training_args"])

    def _get_metrics_from_config(self) -> Dict[str, Callable[[EvalPrediction], float]]:
        """Get the required metrics from the config.

        Returns:
            Dict[str, Callable[[EvalPrediction], float]]: Dictionary of metrics.
                The keys are the metric names and the values are the metric functions
                (taking an EvalPrediction object as input and returning a float).
        """
        metrics = self.config["metrics"]
        metrics_fns = {}
        for metric in metrics:
            metric_name = metric["name"]
            metric_fn = MetricRegistry.metric_name_map[metric_name]
            metric_config = metric["kwargs"] if "kwargs" in metric else {}
            metric_fn = partial(metric_fn, **metric_config)
            metrics_fns[metric_name] = metric_fn

        return metrics_fns

    def _get_suitable_collator(self) -> Callable:
        """Get the suitable collator for the task. If the task is a sequence task,
        the collator will only collate the inputs. Otherwise, it will collate the inputs
        and the labels.

        Returns:
            Callable: The collator.
        """
        if self.task.task_description.task_type["entity"] == "sequence":
            return collate_inputs
        else:
            if self.task.label_ignore_value == None:
                raise ValueError(
                    f"Task {self.task} does not set label_ignore_value, "
                    "but it is required for collation of tasks operating on token level."
                )
            return partial(
                collate_inputs_and_labels,
                label_padding_value=self.task.label_ignore_value,
            )

    def _get_logits_preprocessor_fn(self) -> Callable | None:
        if self.task.task_description.task_type["operation"] == "binary_classification":
            return preprocess_binary_classification_logits
        elif (
            self.task.task_description.task_type["operation"]
            == "multiclass_classification"
        ):
            return preprocess_multi_classification_logits
        else:
            return None

    def _test_device(self, device: str) -> None:
        """Test if the device is available.

        Args:
            device (torch.device): The device.

        Raises:
            ValueError: If the device is not available.
        """
        try:
            device = torch.device(device)
            random_tensor = torch.rand(1)
            random_tensor = random_tensor.to(device)
        except Exception as e:
            raise RuntimeError(f"Error with chosen device {device}: {e}")
