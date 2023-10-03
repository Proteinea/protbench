from typing import Optional

from scipy.stats import spearmanr
from transformers import EvalPrediction
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from protbench.metrics.utils import remove_ignored_predictions
import numpy as np
import torch


def compute_accuracy(
    p: EvalPrediction, ignore_index: Optional[int] = -100, **kwargs
) -> float:
    """Compute accuracy for classification tasks.

    Args:
        p (EvalPrediction): predictions object from the hugginface trainer.

    Returns:
        float: classification accuracy
    """
    predictions, labels = p.predictions.reshape(-1), p.label_ids.reshape(-1)
    if ignore_index is not None:
        predictions, labels = remove_ignored_predictions(
            predictions, labels, ignore_value=ignore_index
        )
    return float(accuracy_score(predictions, labels, **kwargs))


def compute_precision(
    p: EvalPrediction, ignore_index: Optional[int] = -100, **kwargs
) -> float:
    """Compute precision for classification tasks.

    Args:
        p (EvalPrediction): predictions object from the hugginface trainer.

    Returns:
        float: classification precision
    """
    predictions, labels = p.predictions.reshape(-1), p.label_ids.reshape(-1)
    if ignore_index is not None:
        predictions, labels = remove_ignored_predictions(
            predictions, labels, ignore_value=ignore_index
        )
    return float(precision_score(predictions, labels, **kwargs))


def compute_recall(
    p: EvalPrediction, ignore_index: Optional[int] = -100, **kwargs
) -> float:
    """Compute recall for classification tasks.

    Args:
        p (EvalPrediction): predictions object from the hugginface trainer.

    Returns:
        float: classification recall
    """
    predictions, labels = p.predictions.reshape(-1), p.label_ids.reshape(-1)
    if ignore_index is not None:
        predictions, labels = remove_ignored_predictions(
            predictions, labels, ignore_value=ignore_index
        )
    return float(recall_score(predictions, labels, **kwargs))


def compute_f1(
    p: EvalPrediction, ignore_index: Optional[int] = -100, **kwargs
) -> float:
    """Compute f1 for classification tasks.

    Args:
        p (EvalPrediction): predictions object from the hugginface trainer.

    Returns:
        float: classification f1
    """
    predictions, labels = p.predictions.reshape(-1), p.label_ids.reshape(-1)
    if ignore_index is not None:
        predictions, labels = remove_ignored_predictions(
            predictions, labels, ignore_value=ignore_index
        )
    return float(f1_score(predictions, labels, **kwargs))


def compute_spearman(
    p: EvalPrediction, ignore_index: Optional[int] = -100, **kwargs
) -> float:
    """
    Compute spearmanr correlation for regression tasks.

    Args:
        p (EvalPrediction): predictions object from the hugginface trainer.

    Returns:
        float: spearmanr correlation
    """
    predictions, labels = p.predictions.reshape(-1), p.label_ids.reshape(-1)
    if ignore_index is not None:
        predictions, labels = remove_ignored_predictions(
            predictions, labels, ignore_value=ignore_index
        )
    return spearmanr(predictions, labels, **kwargs).correlation


def compute_error_bar_for_token_classification(p: EvalPrediction,
                                               ignore_index=-100):
    accuracies = []
    for i in range(p.predictions.shape[0]):
        current_pred = torch.argmax(p.predictions[None, i], dim=-1)
        current_labels = p.label_ids[None, i]
        ep = EvalPrediction(current_pred, current_labels)
        accuracies.append(compute_accuracy(ep, ignore_index=ignore_index))
    accuracy_std = np.std(accuracies)
    accs_std_error = accuracy_std / (len(accuracies)**0.5)
    # 1.96 is the z score for 95% confidence interval
    error_bar = 1.96 * accs_std_error
    return error_bar


def compute_error_bar_for_binary_classification(p: EvalPrediction):
    preds = (torch.sigmoid(torch.tensor(p.predictions)) > 0.5).type(torch.float32)
    accuracies = (preds.numpy() == p.label_ids).astype('float32')
    accs_std = np.std(accuracies)
    accs_std_error = accs_std / (len(accs_std)) ** 0.5
    error_bar = 1.96 * accs_std_error
    return error_bar


def compute_error_bar_for_regression(p: EvalPrediction):
    spearman_corr = compute_spearman(p)
    n = p.shape[0]
    error = ((1 - spearman_corr ** 2) ** 2 * (1 + spearman_corr **2 / 2) / (n - 3)) ** 0.5
    return error
