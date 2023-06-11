import torch
import numpy as np
from transformers import EvalPrediction


_TORCH_IGNORE_INDEX = torch.nn.CrossEntropyLoss().ignore_index


def preprocess_classification_predictions(
    p: EvalPrediction, ignore_value: int = _TORCH_IGNORE_INDEX
) -> EvalPrediction:
    """Preprocess predictions for classification tasks by aligning predictions and labels
    and removing indices that should be ignored.

    Args:
        p (EvalPrediction): predictions object from the hugginface trainer.
        ignore_value (int, optional): the value present in the label that should be ignored.
            Defaults to _TORCH_IGNORE_INDEX which is the default ignore index defined in the
            Pytorch cross entropy loss class.

    Returns:
        EvalPrediction: predictions object with predictions and labels aligned and ignore values removed.
    """
    predictions, labels = p.predictions, p.label_ids
    predictions, labels = predictions.reshape(-1), labels.reshape(-1)
    ignore_mask = np.where(labels != ignore_value)
    p.predictions = predictions[ignore_mask]
    p.label_ids = labels[ignore_mask]
    return p
