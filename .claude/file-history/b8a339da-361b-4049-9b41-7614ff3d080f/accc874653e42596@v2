"""
Cross-validation strategies: K-fold, stratified, group, time series, nested, repeated.
"""

import numpy as np
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    GroupKFold,
    TimeSeriesSplit,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    cross_val_score,
)


def kfold_cv(model, X, y, n_splits=5, scoring="accuracy"):
    """Standard K-fold cross-validation. Returns (mean, std, scores)."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kf, scoring=scoring, n_jobs=-1)
    return scores.mean(), scores.std(), scores


def stratified_kfold_cv(model, X, y, n_splits=5, scoring="accuracy"):
    """Stratified K-fold CV preserving class distribution. Returns (mean, std, scores)."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring=scoring, n_jobs=-1)
    return scores.mean(), scores.std(), scores


def group_kfold_cv(model, X, y, groups, n_splits=5, scoring="accuracy"):
    """Group K-fold CV preventing leakage between groups. Returns (mean, std, scores)."""
    gkf = GroupKFold(n_splits=n_splits)
    scores = cross_val_score(model, X, y, cv=gkf, scoring=scoring, n_jobs=-1, groups=groups)
    return scores.mean(), scores.std(), scores


def time_series_cv(model, X, y, n_splits=5, test_size=None, scoring="accuracy"):
    """Time series cross-validation with forward chaining. Returns (mean, std, scores)."""
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    scores = cross_val_score(model, X, y, cv=tscv, scoring=scoring, n_jobs=-1)
    return scores.mean(), scores.std(), scores


def nested_cv(model, X, y, param_grid, outer_splits=5, inner_splits=3, scoring="accuracy"):
    """Nested CV for unbiased hyperparameter evaluation. Returns (mean, std, scores, best_params_list)."""
    from sklearn.model_selection import GridSearchCV

    outer_cv = KFold(n_splits=outer_splits, shuffle=True, random_state=42)
    inner_cv = KFold(n_splits=inner_splits, shuffle=True, random_state=42)

    outer_scores = []
    best_params_list = []

    for train_idx, test_idx in outer_cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        gs = GridSearchCV(model, param_grid, cv=inner_cv, scoring=scoring, n_jobs=-1)
        gs.fit(X_train, y_train)

        outer_scores.append(gs.best_estimator_.score(X_test, y_test))
        best_params_list.append(gs.best_params_)

    return np.mean(outer_scores), np.std(outer_scores), np.array(outer_scores), best_params_list


def repeated_kfold_cv(model, X, y, n_splits=5, n_repeats=3, scoring="accuracy"):
    """Repeated K-fold CV for stable estimates. Returns (mean, std, scores)."""
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    scores = cross_val_score(model, X, y, cv=rkf, scoring=scoring, n_jobs=-1)
    return scores.mean(), scores.std(), scores


def repeated_stratified_kfold_cv(model, X, y, n_splits=5, n_repeats=3, scoring="accuracy"):
    """Repeated stratified K-fold CV. Returns (mean, std, scores)."""
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    scores = cross_val_score(model, X, y, cv=rskf, scoring=scoring, n_jobs=-1)
    return scores.mean(), scores.std(), scores
