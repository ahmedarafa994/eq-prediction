"""
Feature selection methods: filter, wrapper, embedded, and SHAP-based.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    SelectKBest,
    SelectFromModel,
    VarianceThreshold,
    RFE,
    RFECV,
    mutual_info_classif,
    mutual_info_regression,
    chi2,
    f_classif,
    f_regression,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler


def univariate_feature_selection(X, y, task="classification", k=50):
    """Apply multiple univariate feature selection methods.

    Returns dict with 'mutual_info', 'f_classif'/'f_regression', and optionally 'chi_squared'.
    """
    results = {}
    if task == "classification":
        mi_scores = mutual_info_classif(X, y, random_state=42)
    else:
        mi_scores = mutual_info_regression(X, y, random_state=42)
    results["mutual_info"] = sorted(
        zip(X.columns, mi_scores), key=lambda x: x[1], reverse=True
    )

    if task == "classification":
        f_scores, p_values = f_classif(X, y)
        results["f_classif"] = sorted(
            zip(X.columns, f_scores, p_values), key=lambda x: x[1], reverse=True
        )
        X_minmax = MinMaxScaler().fit_transform(X)
        chi_scores, chi_p = chi2(X_minmax, y)
        results["chi_squared"] = sorted(
            zip(X.columns, chi_scores, chi_p), key=lambda x: x[1], reverse=True
        )
    else:
        f_scores, p_values = f_regression(X, y)
        results["f_regression"] = sorted(
            zip(X.columns, f_scores, p_values), key=lambda x: x[1], reverse=True
        )
    return results


def remove_low_variance_features(X, threshold=0.01):
    """Remove features with variance below threshold."""
    selector = VarianceThreshold(threshold=threshold)
    X_filtered = selector.fit_transform(X)
    selected_features = X.columns[selector.get_support()]
    return X_filtered, selected_features


def recursive_feature_elimination(X, y, estimator=None, n_features_to_select=10):
    """RFE wrapper returning (X_selected, selected_features, rfe_object)."""
    if estimator is None:
        estimator = LogisticRegression(max_iter=1000, penalty="l1", solver="saga")
    rfe = RFE(estimator, n_features_to_select=n_features_to_select, step=0.1)
    X_selected = rfe.fit_transform(X, y)
    selected_features = X.columns[rfe.support_]
    return X_selected, selected_features, rfe


def recursive_feature_elimination_cv(X, y, estimator=None, min_features=1, cv=5):
    """RFECV finding optimal number of features via cross-validation."""
    if estimator is None:
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)
    rfecv = RFECV(
        estimator, step=1, cv=cv, scoring="accuracy", min_features_to_select=min_features
    )
    X_selected = rfecv.fit_transform(X, y)
    selected_features = X.columns[rfecv.support_]
    return X_selected, selected_features, rfecv


def l1_based_selection(X, y, C=0.1):
    """L1 regularization feature selection."""
    lasso = LogisticRegression(penalty="l1", C=C, solver="saga", max_iter=1000, random_state=42)
    selector = SelectFromModel(lasso, threshold="mean")
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    return X_selected, selected_features, selector


def tree_based_selection(X, y, n_estimators=100, threshold="median"):
    """Tree-based feature selection using Random Forest."""
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    selector = SelectFromModel(rf, threshold=threshold)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    importances = rf.feature_importances_
    feature_importance = sorted(
        zip(X.columns, importances), key=lambda x: x[1], reverse=True
    )
    return X_selected, selected_features, selector, feature_importance


def shap_feature_selection(X_train, y_train, X_val, top_k=50):
    """Use SHAP values to select features. Returns (selected_features, importance, explainer)."""
    import shap

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    mean_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = sorted(
        zip(X_train.columns, mean_shap), key=lambda x: x[1], reverse=True
    )
    selected_features = [f[0] for f in feature_importance[:top_k]]
    return selected_features, feature_importance, explainer
