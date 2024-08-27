# NOT FULLY INTEGERATED YET.

import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

from protbench.tasks.residue_to_class.pickle_residue_to_class import \
    PickleResidueToClass


def preprocess_contact_prediction_labels(seq, label, mask):
    contact_map = np.less(squareform(pdist(label)), 8.0).astype(np.int64)
    yind, xind = np.indices(contact_map.shape)
    invalid_mask = ~(mask[:, None] & mask[None, :])
    invalid_mask |= np.abs(yind - xind) < 6
    contact_map[invalid_mask] = -1
    return seq, contact_map, mask


def get_contact_prediction():
    train_data = PickleResidueToClass(
        dataset_path="contact_prediction/train.pickle",
        seqs_col="primary",
        labels_col="tertiary",
        mask_col="valid_mask",
        preprocessing_function=preprocess_contact_prediction_labels,
        num_classes=2,
    )
    val_data = PickleResidueToClass(
        dataset_path="contact_prediction/valid.pickle",
        seqs_col="primary",
        labels_col="tertiary",
        mask_col="valid_mask",
        preprocessing_function=preprocess_contact_prediction_labels,
        num_classes=2,
    )

    return train_data, val_data
