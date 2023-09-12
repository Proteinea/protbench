from typing import List, Dict
import torch


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


def preprocess_multi_classification_logits(
    logits: torch.Tensor, _
) -> torch.Tensor:
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


def preprocess_binary_classification_logits(
    logits: torch.Tensor, _
) -> torch.Tensor:
    """
    Preprocess logits for binary classification tasks to produce predictions.

    Args:
        logits (torch.Tensor): logits from the model (batch_size, seq_len, 1)
            for token classification tasks or (batch_size, 1) for sequence classification tasks.
    """
    return (torch.sigmoid_(logits) > 0.5).to(int)
