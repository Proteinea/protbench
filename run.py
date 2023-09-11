import os

os.environ["WANDB_PROJECT"] = "AnkhV2"

import logging
from functools import partial

from protbench.embedder import TorchEmbedder, TorchEmbeddingFunction
from protbench.tasks import HuggingFaceResidueToClass
from protbench.models import ConvBert, TokenClassificationHead, DownstreamModel
from protbench import metrics

from torch.utils.data import Dataset
import torch
import numpy as np
import random
from transformers import (
    T5EncoderModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EvalPrediction,
)

seed = 7
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


class EmbeddingsDataset(Dataset):
    def __init__(self, embeddings, labels):
        """Dataset for embeddings and corresponding labels of a task.

        Args:
            embeddings (list[torch.Tensor]): list of tensors of embeddings (batch_size, seq_len, embd_dim)
                where each tensor may have a different seq_len.
            labels (list[Any]): list of labels.
        """
        if len(embeddings) != len(labels):
            raise ValueError(
                "embeddings and labels must have the same length but got "
                f"{len(embeddings)} and {len(labels)}"
            )
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embds = self.embeddings[idx][0, :-1, :]
        labels = torch.tensor(self.labels[idx])
        return {
            "embds": embds,
            "labels": labels,
        }


def collate_inputs_and_labels(
    features,
    input_padding_value: int = 0,
    label_padding_value: int = -100,
):
    """Collate a list of features into a batch. This function pads both the embeddings and the labels.

    Args:
        features (List[Dict[str, torch.Tensor]]): The features are expected to be a list of
            dictionaries with the keys "embd" and "labels"
        input_padding_value (int, optional): the padding value used for the embeddings. Defaults to 0.
        label_padding_value (int, optional): the padding value used for labels. Defaults to -100.

    Returns:
        Dict[str, torch.Tensor]: _description_
    """
    embds = [example["embds"] for example in features]
    labels = [example["labels"] for example in features]
    embds = torch.nn.utils.rnn.pad_sequence(
        embds, batch_first=True, padding_value=input_padding_value
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=label_padding_value
    )
    return {"embds": embds, "labels": labels}


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


def preprocess_func(seq, label, mask):
    mask = list(map(float, mask.split()))
    return seq, label, mask


def tokenize(batch, tokenizer):
    return tokenizer.batch_encode_plus(
        batch,
        add_special_tokens=True,
        padding=True,
        return_tensors="pt",
    )["input_ids"]


def embeddings_postprocessing_fn(model_outputs):
    return model_outputs[0]


def main():
    logging.basicConfig(level=logging.INFO)

    hf_residu_to_class_train = HuggingFaceResidueToClass(
        dataset_url="proteinea/secondary_structure_prediction",
        data_files="training_hhblits.csv",
        data_key="train",
        seqs_col="input",
        labels_col="dssp3",
        mask_col="disorder",
        preprocessing_function=preprocess_func,
    )
    hf_residu_to_class_val = HuggingFaceResidueToClass(
        dataset_url="proteinea/secondary_structure_prediction",
        data_files="CASP12.csv",
        data_key="train",
        seqs_col="input",
        labels_col="dssp3",
        mask_col="disorder",
        preprocessing_function=preprocess_func,
    )

    checkpoints = [
        "proteinea-ea/ankh-v2-large-41epochs-e4a2c3615ff005e5e7b5bbd33ec0654106b64f1a",
        "proteinea-ea/ankh-v2-large-45epochs-62fe367d20d957efdf6e8afe6ae1c724f5bc6775",
    ]
    epochs = [41, 45]
    for checkpoint, epoch in zip(checkpoints, epochs):
        TASK = "ssp3_casp12"
        run_name = f"ankhv2-{epoch}-{TASK}"

        model = T5EncoderModel.from_pretrained(checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        embedding_fn = TorchEmbeddingFunction(
            model,
            partial(tokenize, tokenizer=tokenizer),
            device=None,
            embeddings_postprocessing_fn=embeddings_postprocessing_fn,
            pad_token_id=tokenizer.pad_token_id,
        )
        embedder = TorchEmbedder(
            embedding_fn,
            low_memory=False,
            # save_path=f"{run_name}-embeddings",
            save_path=None,
            devices=None,
            batch_size=1,
        )
        train_embeddings = embedder.run(hf_residu_to_class_train.data[0])
        val_embeddings = embedder.run(hf_residu_to_class_val.data[0])
        training_dataset = EmbeddingsDataset(
            embeddings=train_embeddings,
            labels=hf_residu_to_class_train.data[1],
        )
        val_dataset = EmbeddingsDataset(
            embeddings=val_embeddings, labels=hf_residu_to_class_val.data[1]
        )
        model = model.to("cpu")
        del model
        torch.cuda.empty_cache()

        embedding_dim = train_embeddings[0].shape[-1]
        convbert = ConvBert(
            input_dim=embedding_dim,
            nhead=4,
            hidden_dim=int(embedding_dim / 2),
            num_layers=1,
            kernel_size=7,
            dropout=0.2,
        )
        head = TokenClassificationHead(
            input_dim=embedding_dim,
            output_dim=hf_residu_to_class_train.num_classes,
        )

        training_args = TrainingArguments(
            output_dir=run_name,
            run_name=run_name,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            warmup_steps=1000,
            learning_rate=1e-03,
            weight_decay=0.0,
            logging_dir=f"./logs_{run_name}",
            logging_steps=50,
            do_train=True,
            do_eval=True,
            evaluation_strategy="epoch",
            gradient_accumulation_steps=16,
            fp16=False,
            fp16_opt_level="02",
            seed=7,
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            greater_is_better=True,
            save_strategy="epoch",
            report_to=None,
        )

        def compute_metrics(predictions: EvalPrediction):
            return {
                "eval_accuracy": metrics.compute_accuracy(predictions),
                "eval_precision": metrics.compute_precision(
                    predictions, average="macro"
                ),
                "eval_recall": metrics.compute_recall(predictions, average="macro"),
                "eval_f1": metrics.compute_f1(predictions, average="macro"),
            }

        NUM_TRIALS = 3
        for _ in range(NUM_TRIALS):
            downstream_model = DownstreamModel(downstream_backbone=convbert, head=head)

            trainer = Trainer(
                model=downstream_model,
                args=training_args,
                train_dataset=training_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_metrics,
                data_collator=collate_inputs_and_labels,
                preprocess_logits_for_metrics=preprocess_multi_classification_logits,
            )
            trainer.train()
        downstream_model = downstream_model.to("cpu")
        del downstream_model


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.multiprocessing.set_start_method("spawn")
    main()
