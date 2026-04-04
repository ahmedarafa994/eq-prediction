"""
Threshold optimization: F1, cost-sensitive, profit-maximizing, Youden's J,
constraint-based, ensemble thresholding.
"""

import numpy as np
from sklearn.metrics import (
    precision_recall_curve,
    confusion_matrix,
    roc_curve,
)


def find_optimal_threshold_f1(y_true, y_prob):
    """Find threshold maximizing F1 score. Returns (threshold, f1)."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    f1_scores = np.nan_to_num(f1_scores)
    optimal_idx = np.argmax(f1_scores)
    return thresholds[optimal_idx], f1_scores[optimal_idx]


def cost_optimal_threshold(y_true, y_prob, cost_fp, cost_fn, cost_tp=0, cost_tn=0):
    """Find threshold minimizing expected cost. Returns (threshold, min_cost)."""
    thresholds = np.unique(y_prob)
    costs = []
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        costs.append(fp * cost_fp + fn * cost_fn + tp * cost_tp + tn * cost_tn)
    optimal_idx = np.argmin(costs)
    return thresholds[optimal_idx], costs[optimal_idx]


def profit_maximizing_threshold(y_true, y_prob, tp_value, fp_cost, fn_cost, tn_value=0):
    """Find threshold maximizing expected profit. Returns (threshold, max_profit)."""
    thresholds = np.linspace(0, 1, 101)
    profits = []
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        profits.append(tp * tp_value + tn * tn_value - fp * fp_cost - fn * fn_cost)
    optimal_idx = np.argmax(profits)
    return thresholds[optimal_idx], profits[optimal_idx]


def youdens_j_threshold(y_true, y_prob):
    """Find threshold maximizing Youden's J statistic (TPR - FPR). Returns (threshold, j)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    return thresholds[optimal_idx], j_scores[optimal_idx]


def threshold_for_constraint(y_true, y_prob, metric="recall", min_value=0.95):
    """Find threshold achieving minimum recall or precision constraint. Returns (threshold, precision, recall)."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    if metric == "recall":
        valid_idx = np.where(recalls >= min_value)[0]
        if len(valid_idx) > 0:
            best_idx = valid_idx[np.argmax(precisions[valid_idx])]
            return thresholds[best_idx], precisions[best_idx], recalls[best_idx]
    elif metric == "precision":
        valid_idx = np.where(precisions >= min_value)[0]
        if len(valid_idx) > 0:
            best_idx = valid_idx[np.argmax(recalls[valid_idx])]
            return thresholds[best_idx], precisions[best_idx], recalls[best_idx]

    return None, None, None


def ensemble_threshold(models, X_cal, y_cal, X_test, method="soft_voting"):
    """Optimize threshold for ensemble predictions. Returns (predictions, threshold)."""
    if method == "soft_voting":
        cal_probs = np.mean([m.predict_proba(X_cal)[:, 1] for m in models], axis=0)
        test_probs = np.mean([m.predict_proba(X_test)[:, 1] for m in models], axis=0)
    elif method == "hard_voting":
        cal_probs = np.mean([m.predict(X_cal) for m in models], axis=0)
        test_probs = np.mean([m.predict(X_test) for m in models], axis=0)
    else:
        raise ValueError(f"Unknown method: {method}")

    optimal_threshold, _ = find_optimal_threshold_f1(y_cal, cal_probs)
    predictions = (test_probs >= optimal_threshold).astype(int)
    return predictions, optimal_threshold
