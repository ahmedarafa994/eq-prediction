"""
Model diagnostics: bias-variance diagnosis, learning curves, validation curves,
model comparison, comprehensive diagnostic reports.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    brier_score_loss,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import learning_curve, validation_curve


def diagnose_model(model, X_train, y_train, X_val, y_val, task="classification"):
    """Comprehensive bias-variance diagnosis. Returns dict with metrics and diagnosis."""
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    if task == "classification":
        train_score = accuracy_score(y_train, train_pred)
        val_score = accuracy_score(y_val, val_pred)
        train_f1 = f1_score(y_train, train_pred, zero_division=0)
        val_f1 = f1_score(y_val, val_pred, zero_division=0)

        if hasattr(model, "predict_proba"):
            train_prob = model.predict_proba(X_train)[:, 1]
            val_prob = model.predict_proba(X_val)[:, 1]
            train_brier = brier_score_loss(y_train, train_prob)
            val_brier = brier_score_loss(y_val, val_prob)
        else:
            train_brier = val_brier = None

        gap = train_score - val_score

        if gap > 0.15:
            diagnosis = "high_variance"
        elif val_score < 0.7:
            diagnosis = "high_bias"
        elif gap < 0.05 and val_score > 0.85:
            diagnosis = "good_fit"
        else:
            diagnosis = "moderate"

        result = {
            "train_accuracy": train_score,
            "val_accuracy": val_score,
            "train_f1": train_f1,
            "val_f1": val_f1,
            "train_brier": train_brier,
            "val_brier": val_brier,
            "gap": gap,
            "diagnosis": diagnosis,
        }
    else:
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        train_r2 = r2_score(y_train, train_pred)
        val_r2 = r2_score(y_val, val_pred)
        gap = train_r2 - val_r2

        if gap > 0.15:
            diagnosis = "high_variance"
        elif val_r2 < 0.5:
            diagnosis = "high_bias"
        elif gap < 0.05 and val_r2 > 0.85:
            diagnosis = "good_fit"
        else:
            diagnosis = "moderate"

        result = {
            "train_rmse": train_rmse,
            "val_rmse": val_rmse,
            "train_r2": train_r2,
            "val_r2": val_r2,
            "gap": gap,
            "diagnosis": diagnosis,
        }

    return result


def get_learning_curve_data(estimator, X, y, cv=5, train_sizes=None, scoring="accuracy"):
    """Get learning curve data. Returns dict with arrays for plotting."""
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)

    ts, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1, train_sizes=train_sizes,
        scoring=scoring, shuffle=True, random_state=42,
    )

    return {
        "train_sizes": ts,
        "train_mean": train_scores.mean(axis=1),
        "train_std": train_scores.std(axis=1),
        "val_mean": val_scores.mean(axis=1),
        "val_std": val_scores.std(axis=1),
    }


def get_validation_curve_data(estimator, X, y, param_name, param_range, cv=5, scoring="accuracy"):
    """Get validation curve data for a single hyperparameter. Returns dict."""
    train_scores, val_scores = validation_curve(
        estimator, X, y, param_name=param_name,
        param_range=param_range, cv=cv, scoring=scoring,
    )

    return {
        "param_range": param_range,
        "train_mean": train_scores.mean(axis=1),
        "train_std": train_scores.std(axis=1),
        "val_mean": val_scores.mean(axis=1),
        "val_std": val_scores.std(axis=1),
    }


def compare_models(models, X, y, cv=5, scoring="accuracy"):
    """Compare multiple models via cross-validation. Returns list of (name, mean, std)."""
    from sklearn.model_selection import cross_val_score

    results = []
    for name, model in models:
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        results.append({"name": name, "mean": scores.mean(), "std": scores.std(), "scores": scores})
    return results
