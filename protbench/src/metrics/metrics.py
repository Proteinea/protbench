from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import EvalPrediction

from protbench.src.tasks import TaskDescription
from protbench.src.metrics import MetricRegistry
from protbench.src.metrics.utils import preprocess_classification_predictions


@MetricRegistry.register("accuracy")
def compute_accuracy(p: EvalPrediction, *args, **kwargs) -> float:
    """Compute accuracy for classification tasks.

    Args:
        p (EvalPrediction): predictions object from the hugginface trainer.

    Returns:
        float: classification accuracy
    """
    p = preprocess_classification_predictions(p)
    predictions, labels = p.predictions, p.label_ids
    return accuracy_score(predictions, labels)


@MetricRegistry.register("precision")
def compute_precision(p: EvalPrediction, task_description: TaskDescription) -> float:
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


@MetricRegistry.register("recall")
def compute_recall(p: EvalPrediction, task_description: TaskDescription) -> float:
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


@MetricRegistry.register("f1")
def compute_f1(p: EvalPrediction, task_description: TaskDescription) -> float:
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


@MetricRegistry.register("spearman")
def compute_spearman(p: EvalPrediction, *args, **kwargs) -> float:
    """
    Compute spearmanr correlation for regression tasks.

    Args:
        p (EvalPrediction): predictions object from the hugginface trainer.

    Returns:
        float: spearmanr correlation
    """
    return spearmanr(p.label_ids, p.predictions).correlation
