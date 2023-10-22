import torch


def preprocess_multi_classification_logits(
    logits: torch.Tensor, *args, **kwargs
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
    logits: torch.Tensor, *args, **kwargs
) -> torch.Tensor:
    """
    Preprocess logits for binary classification tasks to produce predictions.

    Args:
        logits (torch.Tensor): logits from the model (batch_size, seq_len, 1)
            for token classification tasks or (batch_size, 1) for sequence classification tasks.
    """
    threshold = kwargs.pop("threshold", 0.5)
    return (torch.sigmoid_(logits) > threshold).to(int)
