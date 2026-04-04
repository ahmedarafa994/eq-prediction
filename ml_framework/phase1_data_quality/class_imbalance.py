"""
Class imbalance assessment and cost-sensitive classification.

Provides tools to evaluate class imbalance severity and a classifier wrapper
that applies a cost matrix via sample weights.
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone


def assess_imbalance(y) -> Dict[str, Any]:
    """
    Assess class imbalance in a target variable.

    Parameters
    ----------
    y : array-like
        Target variable (1-D).

    Returns
    -------
    dict
        Keys:
          - ``class_counts`` : dict mapping class label to count.
          - ``total_samples`` : int.
          - ``n_classes`` : int.
          - ``imbalance_ratio`` : float (max_count / min_count).
          - ``severity`` : str, one of ``"balanced"``, ``"mild"``,
            ``"moderate"``, ``"severe"``.
          - ``minority_class`` : the class with the fewest samples.
          - ``majority_class`` : the class with the most samples.
          - ``class_frequencies`` : dict mapping class label to frequency (0-1).
    """
    if isinstance(y, pd.Series):
        y_arr = y.values
    elif isinstance(y, np.ndarray):
        y_arr = y
    else:
        y_arr = np.asarray(y)

    unique, counts = np.unique(y_arr, return_counts=True)
    total = int(counts.sum())

    class_counts = {str(label): int(cnt) for label, cnt in zip(unique, counts)}
    class_frequencies = {
        str(label): round(float(cnt / total), 4)
        for label, cnt in zip(unique, counts)
    }

    sorted_counts = sorted(counts, reverse=True)
    max_count = int(sorted_counts[0])
    min_count = int(sorted_counts[-1])

    imbalance_ratio = float(max_count / min_count) if min_count > 0 else float("inf")

    idx_max = int(np.argmax(counts))
    idx_min = int(np.argmin(counts))
    majority_class = unique[idx_max]
    minority_class = unique[idx_min]

    if imbalance_ratio <= 1.5:
        severity = "balanced"
    elif imbalance_ratio <= 3.0:
        severity = "mild"
    elif imbalance_ratio <= 10.0:
        severity = "moderate"
    else:
        severity = "severe"

    return {
        "class_counts": class_counts,
        "total_samples": total,
        "n_classes": int(len(unique)),
        "imbalance_ratio": round(imbalance_ratio, 4),
        "severity": severity,
        "minority_class": minority_class,
        "majority_class": majority_class,
        "class_frequencies": class_frequencies,
    }


class CostSensitiveClassifier(BaseEstimator, ClassifierMixin):
    """
    Wrapper that applies a cost matrix to a base classifier via sample weights.

    The cost matrix defines the penalty/reward for each (actual, predicted)
    pair.  During ``fit``, per-sample weights are computed so that
    misclassifying a sample incurs a weight proportional to the cost.

    For binary classification (positive = class 1):
      - FP cost: cost_fp (predict 1 when actual is 0)
      - FN cost: cost_fn (predict 0 when actual is 1)
      - TP reward: cost_tp (usually 0 or negative)
      - TN reward: cost_tn (usually 0 or negative)

    Sample weights are set per class based on the cost of misclassifying
    that class relative to correct classification.

    Parameters
    ----------
    base_estimator : estimator object
        An sklearn-compatible classifier that supports ``sample_weight`` in
        its ``fit`` method.
    cost_fp : float, default 1.0
        Cost of a false positive.
    cost_fn : float, default 1.0
        Cost of a false negative.
    cost_tp : float, default 0.0
        Cost (or negative reward) of a true positive.
    cost_tn : float, default 0.0
        Cost (or negative reward) of a true negative.

    Attributes
    ----------
    estimator_ : estimator object
        The fitted base estimator.
    classes_ : ndarray
        The class labels seen during fit.
    """

    def __init__(
        self,
        base_estimator,
        cost_fp: float = 1.0,
        cost_fn: float = 1.0,
        cost_tp: float = 0.0,
        cost_tn: float = 0.0,
    ):
        self.base_estimator = base_estimator
        self.cost_fp = cost_fp
        self.cost_fn = cost_fn
        self.cost_tp = cost_tp
        self.cost_tn = cost_tn

    def _compute_sample_weights(self, y) -> np.ndarray:
        """
        Compute sample weights from the cost matrix.

        For each class, the weight is the cost of misclassifying a sample
        of that class (the cost of predicting the other class).
        """
        y_arr = np.asarray(y)
        classes = np.unique(y_arr)
        weights = np.ones(len(y_arr), dtype=float)

        if len(classes) == 2:
            # Binary: assume classes are [neg, pos] sorted
            neg_class = classes[0]
            pos_class = classes[1]

            for i, label in enumerate(y_arr):
                if label == pos_class:
                    # Cost of misclassifying positive as negative = FN cost
                    # Net weight = cost_fn - cost_tp
                    weights[i] = max(self.cost_fn - self.cost_tp, 0.0)
                else:
                    # Cost of misclassifying negative as positive = FP cost
                    # Net weight = cost_fp - cost_tn
                    weights[i] = max(self.cost_fp - self.cost_tn, 0.0)
        else:
            # Multi-class: use average misclassification cost as weight
            # Simplified: assign cost_fn for minority, cost_fp for majority
            unique, counts = np.unique(y_arr, return_counts=True)
            majority_label = unique[np.argmax(counts)]
            for i, label in enumerate(y_arr):
                if label == majority_label:
                    weights[i] = max(self.cost_fp - self.cost_tn, 0.0)
                else:
                    weights[i] = max(self.cost_fn - self.cost_tp, 0.0)

        # Normalize so weights average to 1.0
        if weights.sum() > 0:
            weights = weights * len(weights) / weights.sum()

        return weights

    def fit(self, X, y, sample_weight=None) -> "CostSensitiveClassifier":
        """
        Fit the base estimator with cost-sensitive sample weights.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target labels.
        sample_weight : array-like, optional
            Additional per-sample weights. Multiplied with the cost-derived
            weights.

        Returns
        -------
        self
        """
        y_arr = np.asarray(y)
        self.classes_ = np.unique(y_arr)

        cost_weights = self._compute_sample_weights(y_arr)

        if sample_weight is not None:
            cost_weights = cost_weights * np.asarray(sample_weight)

        self.estimator_ = clone(self.base_estimator)
        self.estimator_.fit(X, y_arr, sample_weight=cost_weights)

        return self

    def predict(self, X):
        """Predict using the fitted base estimator."""
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        """Predict probabilities using the fitted base estimator."""
        return self.estimator_.predict_proba(X)

    def decision_function(self, X):
        """Decision function of the fitted base estimator."""
        return self.estimator_.decision_function(X)
