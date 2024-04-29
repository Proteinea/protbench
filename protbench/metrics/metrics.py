from typing import Optional

import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from transformers import EvalPrediction

from protbench.metrics.utils import remove_ignored_predictions


def compute_pearsonr(
    p: EvalPrediction, ignore_index: Optional[int] = -100, **kwargs
):
    return pearsonr(p.predictions.flatten(), p.label_ids.flatten()).statistic


def compute_rmse(
    p: EvalPrediction, ignore_index: Optional[int] = -100, **kwargs
):
    return mean_squared_error(p.label_ids.flatten(), p.predictions.flatten())


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


def compute_accuracies_std(
    p: EvalPrediction, ignore_index: Optional[int] = -100, **kwargs
):
    accuracies = []
    for predictions, labels in zip(p.predictions, p.label_ids):
        predictions, labels = predictions.reshape(-1), labels.reshape(-1)
        if ignore_index is not None:
            predictions, labels = remove_ignored_predictions(
                predictions, labels, ignore_value=ignore_index
            )
        accuracies.append((predictions == labels).mean())
    return np.std(accuracies)


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


def compute_accuracies_error_bar(accuracies_std, num_examples):
    accs_std_error = accuracies_std / num_examples**0.5
    error_bar = 1.96 * accs_std_error
    return error_bar


def compute_error_bar_for_regression(spearman_corr, num_examples):
    error_bar = (
        (1 - spearman_corr**2) ** 2
        * (1 + spearman_corr**2 / 2)
        / (num_examples - 3)
    ) ** 0.5
    return error_bar
