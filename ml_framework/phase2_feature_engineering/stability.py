"""
Feature stability analysis across splits, time, and subsamples.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold


def analyze_feature_stability(X, y, feature_selector, n_splits=5, random_state=42, task="classification"):
    """Analyze stability of feature selection across data splits.

    Returns dict with jaccard_mean/std, selection_frequency, importance_correlations.
    """
    kf = (StratifiedKFold if task == "classification" else KFold)(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    selected_per_fold, importances_per_fold = [], []
    X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

    for train_idx, _ in kf.split(X_df, y):
        X_train = X_df.iloc[train_idx] if isinstance(X, pd.DataFrame) else X[train_idx]
        y_train = y.iloc[train_idx] if isinstance(y, pd.Series) else y[train_idx]
        selector = clone(feature_selector)
        selector.fit(X_train, y_train)
        support = selector.get_support() if hasattr(selector, "get_support") else selector.support_
        selected_per_fold.append(set(np.where(support)[0]))
        if hasattr(selector, "feature_importances_"):
            importances_per_fold.append(selector.feature_importances_)
        elif hasattr(selector, "estimator") and hasattr(selector.estimator, "coef_"):
            importances_per_fold.append(np.abs(selector.estimator.coef_[0]))

    jaccard_indices = []
    for i in range(n_splits):
        for j in range(i + 1, n_splits):
            inter = len(selected_per_fold[i] & selected_per_fold[j])
            union = len(selected_per_fold[i] | selected_per_fold[j])
            jaccard_indices.append(inter / union if union > 0 else 0)

    n_features = X_df.shape[1]
    selection_freq = np.zeros(n_features)
    for sel in selected_per_fold:
        selection_freq[list(sel)] += 1
    selection_freq /= n_splits

    importance_correlations = None
    if importances_per_fold:
        importance_correlations = []
        for i in range(len(importances_per_fold)):
            for j in range(i + 1, len(importances_per_fold)):
                corr, _ = spearmanr(importances_per_fold[i], importances_per_fold[j], nan_policy="omit")
                if not np.isnan(corr):
                    importance_correlations.append(corr)

    return {
        "jaccard_mean": np.mean(jaccard_indices),
        "jaccard_std": np.std(jaccard_indices),
        "selection_frequency": selection_freq,
        "importance_correlations": importance_correlations,
    }


def time_based_stability_analysis(df, feature_cols, target_col, time_col, n_windows=5, window_size=None):
    """Analyze feature stability across time windows for concept drift detection."""
    df_sorted = df.sort_values(time_col)
    if window_size is None:
        window_size = len(df_sorted) // n_windows
    results = []
    for i in range(n_windows):
        start_idx = i * window_size
        end_idx = min((i + 1) * window_size, len(df_sorted))
        window_data = df_sorted.iloc[start_idx:end_idx]
        stats = {"window": i, "n_samples": len(window_data)}
        for col in feature_cols:
            if pd.api.types.is_numeric_dtype(window_data[col]):
                stats[f"{col}_mean"] = window_data[col].mean()
                stats[f"{col}_std"] = window_data[col].std()
        results.append(stats)
    stability_df = pd.DataFrame(results)
    drift_metrics = {}
    for col in feature_cols:
        if f"{col}_mean" in stability_df.columns:
            mean_col = stability_df[f"{col}_mean"]
            drift_metrics[f"{col}_cv"] = mean_col.std() / (mean_col.mean() + 1e-10)
    return stability_df, drift_metrics


def subsampling_stability(X, y, model, n_subsamples=100, subsample_size=0.8, random_state=42):
    """Assess feature importance stability via random subsampling."""
    from sklearn.utils import resample
    n_features = X.shape[1]
    importance_samples = np.zeros((n_subsamples, n_features))
    for i in range(n_subsamples):
        X_sub, y_sub = resample(X, y, n_samples=int(len(X) * subsample_size), replace=False,
                                 random_state=random_state + i if random_state else None)
        m = clone(model)
        m.fit(X_sub, y_sub)
        if hasattr(m, "feature_importances_"):
            importance_samples[i] = m.feature_importances_
        elif hasattr(m, "coef_"):
            importance_samples[i] = np.abs(m.coef_).flatten()
    mean_imp = importance_samples.mean(axis=0)
    std_imp = importance_samples.std(axis=0)
    threshold = mean_imp.mean()
    selection_freq = (importance_samples > threshold).mean(axis=0)
    return {
        "mean_importance": mean_imp,
        "std_importance": std_imp,
        "cv_importance": std_imp / (mean_imp + 1e-10),
        "selection_frequency": selection_freq,
    }


def correlation_stability_analysis(X, n_splits=5, random_state=42):
    """Analyze stability of feature correlations across data splits."""
    X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    correlation_matrices = []
    for train_idx, _ in kf.split(X_df):
        corr_matrix = X_df.iloc[train_idx].corr().values
        correlation_matrices.append(corr_matrix)
    corr_array = np.array(correlation_matrices)
    mean_corr = corr_array.mean(axis=0)
    std_corr = corr_array.std(axis=0)
    mask = ~np.eye(mean_corr.shape[0], dtype=bool)
    cv_corr = std_corr[mask] / (np.abs(mean_corr[mask]) + 1e-10)
    return {
        "mean_correlation": mean_corr,
        "std_correlation": std_corr,
        "unstable_correlations": cv_corr > 0.5,
        "cv_correlations": cv_corr,
    }
