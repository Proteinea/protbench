from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import EvalPrediction

from protbench.src.tasks import TaskDescription
from protbench.src.metrics import MetricRegistry
from protbench.src.metrics.utils import preprocess_classification_predictions


@MetricRegistry.add_metric("accuracy")
def accuracy(p: EvalPrediction, _) -> float:
    """Compute accuracy for classification tasks.

    Args:
        p (EvalPrediction): predictions object from the hugginface trainer.

    Returns:
        float: classification accuracy
    """
    p = preprocess_classification_predictions(p)
    predictions, labels = p.predictions, p.label_ids
    return accuracy_score(predictions, labels)


@MetricRegistry.add_metric("precision")
def precision(p: EvalPrediction, task_description: TaskDescription) -> float:
    """Compute precision for classification tasks.

    Args:
        p (EvalPrediction): predictions object from the hugginface trainer.

    Returns:
        float: classification precision
    """
    p = preprocess_classification_predictions(p)
    predictions, labels = p.predictions, p.label_ids
    if task_description.task_type["operation"] in [
        "multilabel_classification",
        "binary_classification",
    ]:
        return precision_score(predictions, labels, average="binary")

    return precision_score(predictions, labels, average="macro")


@MetricRegistry.add_metric("recall")
def recall(p: EvalPrediction, task_description: TaskDescription) -> float:
    """Compute recall for classification tasks.

    Args:
        p (EvalPrediction): predictions object from the hugginface trainer.

    Returns:
        float: classification recall
    """
    p = preprocess_classification_predictions(p)
    predictions, labels = p.predictions, p.label_ids
    if task_description.task_type["operation"] in [
        "multilabel_classification",
        "binary_classification",
    ]:
        return recall_score(predictions, labels, average="binary")

    return recall_score(predictions, labels, average="macro")


@MetricRegistry.add_metric("f1")
def f1(p: EvalPrediction, task_description: TaskDescription) -> float:
    """Compute f1 for classification tasks.

    Args:
        p (EvalPrediction): predictions object from the hugginface trainer.

    Returns:
        float: classification f1
    """
    p = preprocess_classification_predictions(p)
    predictions, labels = p.predictions, p.label_ids
    if task_description.task_type["operation"] in [
        "multilabel_classification",
        "binary_classification",
    ]:
        return f1_score(predictions, labels, average="binary")

    return f1_score(predictions, labels, average="macro")


@MetricRegistry.add_metric("spearmanr")
def spearmanr(p: EvalPrediction, _) -> float:
    """
    Compute spearmanr correlation for regression tasks.

    Args:
        p (EvalPrediction): predictions object from the hugginface trainer.

    Returns:
        float: spearmanr correlation
    """
    return spearmanr(p.label_ids, p.predictions).correlation
