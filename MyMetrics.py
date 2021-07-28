from sklearn.metrics import make_scorer
import numpy as np


def rmspe_origin(y_true, y_pred):
    """
    y_true:array-like of shape (n_samples,) or (n_samples, n_outputs)
    Ground truth (correct) target values.

    y_pred:array-like of shape (n_samples,) or (n_samples, n_outputs)
    Estimated target values.
    """
    return np.sqrt(np.sum(((y_true - y_pred) / y_true) ** 2) / len(y_true))


def get_rmspe():
    """
    return an object of scoring
    """
    return make_scorer(rmspe_origin, greater_is_better=False, needs_proba=False)
