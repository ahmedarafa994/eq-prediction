"""
Post-processing fairness: equalized odds, reject option classification,
calibrated equalized odds, fairness audit, fairness-accuracy tradeoff analysis.
"""

import numpy as np
from sklearn.metrics import confusion_matrix


def equalized_odds_postprocessing(y_true, y_prob, sensitive_attr, base_rate=None):
    """Derive group-specific thresholds satisfying equalized odds (simplified)."""
    groups = np.unique(sensitive_attr)
    thresholds = {}

    if base_rate is None:
        base_rate = np.mean(y_true)

    for group in groups:
        group_mask = sensitive_attr == group
        y_prob_group = y_prob[group_mask]

        found = 1.0
        for threshold in np.linspace(0, 1, 101):
            pred_rate = np.mean((y_prob_group >= threshold).astype(int))
            if pred_rate >= base_rate:
                found = threshold
                break
        thresholds[group] = found

    return thresholds


def reject_option_classification(y_prob, sensitive_attr, threshold=0.5, band=0.1):
    """Flip predictions near decision boundary to favor disadvantaged group."""
    predictions = (y_prob >= threshold).astype(int)
    groups = np.unique(sensitive_attr)

    group_rates = {g: np.mean(predictions[sensitive_attr == g]) for g in groups}
    disadvantaged_group = min(group_rates, key=group_rates.get)

    for i in range(len(y_prob)):
        if abs(y_prob[i] - threshold) < band:
            if sensitive_attr[i] == disadvantaged_group:
                predictions[i] = 1 - predictions[i]

    return predictions


def calibrated_equalized_odds(y_true, y_prob, sensitive_attr):
    """Compute group-specific TPR/FPR curves over thresholds for equalized odds analysis."""
    groups = np.unique(sensitive_attr)
    result = {}

    for group in groups:
        group_mask = sensitive_attr == group
        y_true_g = y_true[group_mask]
        y_prob_g = y_prob[group_mask]

        thresholds = np.linspace(0, 1, 101)
        tprs, fprs = [], []

        for t in thresholds:
            y_pred = (y_prob_g >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true_g, y_pred, labels=[0, 1]).ravel()
            tprs.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
            fprs.append(fp / (fp + tn) if (fp + tn) > 0 else 0)

        result[group] = {"thresholds": thresholds, "tprs": tprs, "fprs": fprs}

    return result


def fairness_audit(y_true, y_pred, sensitive_attr):
    """Audit model fairness across groups. Returns dict of per-group metrics."""
    groups = np.unique(sensitive_attr)
    metrics = {}

    for group in groups:
        group_mask = sensitive_attr == group
        y_true_g = y_true[group_mask]
        y_pred_g = y_pred[group_mask]

        cm = confusion_matrix(y_true_g, y_pred_g, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics[group] = {
                "positive_rate": float(np.mean(y_pred_g)),
                "tpr": tp / (tp + fn) if (tp + fn) > 0 else 0,
                "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0,
                "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
            }
        else:
            metrics[group] = {"positive_rate": float(np.mean(y_pred_g))}

    return metrics


def fairness_accuracy_tradeoff(y_true, y_prob, sensitive_attr, thresholds=None):
    """Analyze fairness-accuracy tradeoff across thresholds."""
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 17)

    results = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        accuracy = float(np.mean(y_pred == y_true))

        audit = fairness_audit(y_true, y_pred, sensitive_attr)
        groups = list(audit.keys())

        if len(groups) >= 2:
            positive_rates = [audit[g]["positive_rate"] for g in groups]
            positive_rate_diff = max(positive_rates) - min(positive_rates)
        else:
            positive_rate_diff = 0.0

        results.append({
            "threshold": float(t),
            "accuracy": accuracy,
            "positive_rate_diff": positive_rate_diff,
        })

    return results
