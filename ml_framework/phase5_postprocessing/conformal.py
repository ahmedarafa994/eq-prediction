"""
Conformal prediction: split conformal classification/regression,
cross-conformal, conformalized quantile regression, adaptive conformal inference.
"""

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold


# --- Split conformal classification ---

def split_conformal_classification(model, X_cal, y_cal, X_test, alpha=0.1):
    """Split conformal prediction sets for classification with 1-alpha coverage."""
    n_cal = len(X_cal)
    cal_probs = model.predict_proba(X_cal)
    cal_scores = 1 - cal_probs[np.arange(n_cal), y_cal]

    qhat = np.quantile(cal_scores, np.ceil((n_cal + 1) * (1 - alpha)) / n_cal, method="higher")

    test_probs = model.predict_proba(X_test)
    prediction_sets = [np.where(test_probs[i] >= 1 - qhat)[0] for i in range(len(X_test))]

    return prediction_sets, qhat


# --- Split conformal regression ---

def split_conformal_regression(model, X_cal, y_cal, X_test, alpha=0.1):
    """Split conformal prediction intervals for regression with 1-alpha coverage."""
    n_cal = len(X_cal)
    y_pred_cal = model.predict(X_cal)
    cal_scores = np.abs(y_cal - y_pred_cal)

    qhat = np.quantile(cal_scores, np.ceil((n_cal + 1) * (1 - alpha)) / n_cal, method="higher")

    y_pred_test = model.predict(X_test)
    intervals = [(y_pred_test[i] - qhat, y_pred_test[i] + qhat) for i in range(len(X_test))]

    return intervals, qhat


# --- Conformalized quantile regression (CQR) ---

def conformalized_quantile_regression(X_train, y_train, X_cal, y_cal, X_test, alpha=0.1):
    """CQR: adaptive prediction intervals that vary with input."""
    lower_model = GradientBoostingRegressor(loss="quantile", alpha=alpha / 2, random_state=42)
    upper_model = GradientBoostingRegressor(loss="quantile", alpha=1 - alpha / 2, random_state=42)

    lower_model.fit(X_train, y_train)
    upper_model.fit(X_train, y_train)

    lower_cal = lower_model.predict(X_cal)
    upper_cal = upper_model.predict(X_cal)
    cal_scores = np.maximum(y_cal - upper_cal, lower_cal - y_cal)

    n_cal = len(X_cal)
    qhat = np.quantile(cal_scores, np.ceil((n_cal + 1) * (1 - alpha)) / n_cal, method="higher")

    lower_test = lower_model.predict(X_test)
    upper_test = upper_model.predict(X_test)
    intervals = [(lower_test[i] - qhat, upper_test[i] + qhat) for i in range(len(X_test))]

    return intervals, qhat


# --- Cross-conformal ---

def cross_conformal_classification(model, X, y, alpha=0.1, n_folds=5):
    """Cross-conformal prediction for better data efficiency. Returns qhat."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    all_scores = []

    for train_idx, cal_idx in kf.split(X):
        X_train_fold, X_cal_fold = X[train_idx], X[cal_idx]
        y_train_fold, y_cal_fold = y[train_idx], y[cal_idx]

        model.fit(X_train_fold, y_train_fold)
        cal_probs = model.predict_proba(X_cal_fold)
        cal_scores = 1 - cal_probs[np.arange(len(y_cal_fold)), y_cal_fold]
        all_scores.extend(cal_scores)

    n = len(X)
    qhat = np.quantile(all_scores, np.ceil((n + 1) * (1 - alpha)) / n, method="higher")
    return qhat


# --- Adaptive conformal inference for time series ---

def aci_conformal(model, X_cal, y_cal, X_test_stream, alpha=0.1, gamma=0.01):
    """Adaptive Conformal Inference for time-varying distributions (generator)."""
    qhat = 0.0
    coverage_history = []

    for t, X_t in enumerate(X_test_stream):
        pred = model.predict(X_t.reshape(1, -1))[0]
        interval = (pred - qhat, pred + qhat)

        if t < len(y_cal):
            y_true = y_cal[t]
            covered = interval[0] <= y_true <= interval[1]
            coverage_history.append(covered)

            if len(coverage_history) >= 100:
                recent_coverage = np.mean(coverage_history[-100:])
                if recent_coverage < 1 - alpha:
                    qhat += gamma
                elif recent_coverage > 1 - alpha + gamma:
                    qhat = max(0, qhat - gamma)

        yield interval, qhat
