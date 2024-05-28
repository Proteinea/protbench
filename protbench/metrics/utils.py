from typing import Tuple

import numpy as np


def remove_ignored_predictions(
    predictions: np.ndarray, labels: np.ndarray, ignore_value: int = -100
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove ignored predictions from the predictions and labels.

    Args:
        predictions (np.ndarray): numpy array of predictions of shape (N,)
        labels (np.ndarray): numpy array of labels of shape (N,)
        ignore_value (int, optional): value to ignore in the labels. Defaults to -100.

    Returns:
        Tuple[np.ndarray, np.ndarray]: tuple of predictions and labels with ignored values removed
            each of shape (N',) where N' is the number of non-ignored values.
    """
    non_ignored_mask = np.where(labels != ignore_value)
    predictions = predictions[non_ignored_mask]
    labels = labels[non_ignored_mask]
    return predictions, labels
