from typing import Optional

from scipy.stats import spearmanr
from transformers import EvalPrediction
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from protbench.src.metrics import MetricRegistry
from protbench.src.metrics.utils import remove_ignored_predictions


@MetricRegistry.register("accuracy")
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


@MetricRegistry.register("precision")
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


@MetricRegistry.register("recall")
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


@MetricRegistry.register("f1")
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


@MetricRegistry.register("spearman")
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
