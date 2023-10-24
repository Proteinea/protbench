from typing import Dict, List, Callable

import torch
from transformers import AutoTokenizer


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
    embds = [example["embds"] for example in features]
    labels = [example["labels"] for example in features]
    embds = torch.nn.utils.rnn.pad_sequence(
        embds, batch_first=True, padding_value=padding_value
    )
    return {"embds": embds, "labels": torch.tensor(labels)}


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
    embds = [example["embds"] for example in features]
    labels = [example["labels"] for example in features]
    embds = torch.nn.utils.rnn.pad_sequence(
        embds, batch_first=True, padding_value=input_padding_value
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=label_padding_value
    )
    return {"embds": embds, "labels": labels}


def collate_sequence_and_align_labels(tokenizer: AutoTokenizer, ignore_index=-100) -> Callable:
    def _collate_sequence_and_align_labels(batch: List[Dict]) -> Dict:
        sequences = [example["sequences"] for example in batch]
        labels = [example["labels"] for example in batch]

        sequences_encoded = tokenizer(sequences, add_special_tokens=True,
                                      padding='longest', return_tensors="pt")
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=ignore_index
        )
        if labels.shape[-1] < sequences_encoded.shape[-1]:
            diff = sequences_encoded.shape[-1] - labels.shape[-1]
            padding_tokens = torch.tensor([[-100] * diff] * sequences_encoded.shape[0])
            labels = torch.cat((labels, padding_tokens), dim=1)
        sequences_encoded['labels'] = labels
        return sequences_encoded
    return _collate_sequence_and_align_labels

def collate_sequence_and_labels(tokenizer: AutoTokenizer) -> Callable:
    def _collate_sequence_and_labels(batch: List[Dict]) -> Dict:
        sequences = [example["sequences"] for example in batch]
        labels = [example["labels"] for example in batch]

        sequences_encoded = tokenizer(sequences, add_special_tokens=True,
                                      padding='longest', return_tensors="pt")
        labels = torch.tensor(labels)
        sequences_encoded['labels'] = labels
        return sequences_encoded
    return _collate_sequence_and_labels