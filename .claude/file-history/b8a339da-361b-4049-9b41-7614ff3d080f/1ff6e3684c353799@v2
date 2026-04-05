"""
Ensemble methods: stacking, blending, bagging, boosting, ensemble pruning, diversity metrics.
"""

import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    StackingClassifier,
    AdaBoostClassifier,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_predict, KFold, train_test_split
from sklearn.utils import resample


# --- Stacking ---

def stacking_classifier(X_train, y_train, base_estimators=None, meta_learner=None, cv=5):
    """Fit a stacking classifier with cross-validated out-of-fold predictions."""
    if base_estimators is None:
        base_estimators = [
            ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
            ("gb", GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ]
    if meta_learner is None:
        meta_learner = LogisticRegression()

    stack = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_learner,
        cv=cv,
        stack_method="predict_proba",
        n_jobs=-1,
    )
    stack.fit(X_train, y_train)
    return stack


def manual_stacking(X_train, y_train, X_test, base_models, meta_model, cv=5):
    """Manual stacking with explicit OOF meta-feature generation. Returns test predictions."""
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    meta_features_train = np.zeros((len(X_train), len(base_models)))

    for i, (name, model) in enumerate(base_models):
        oof_preds = cross_val_predict(model, X_train, y_train, cv=kf, method="predict_proba")
        meta_features_train[:, i] = oof_preds[:, 1]
        model.fit(X_train, y_train)

    meta_model.fit(meta_features_train, y_train)

    meta_features_test = np.zeros((len(X_test), len(base_models)))
    for i, (name, model) in enumerate(base_models):
        meta_features_test[:, i] = model.predict_proba(X_test)[:, 1]

    return meta_model.predict(meta_features_test)


# --- Blending ---

def blend_models(X_train, y_train, X_test, base_models, meta_model=None, blend_size=0.2):
    """Blend models using a holdout set for meta-learner training. Returns test predictions."""
    if meta_model is None:
        meta_model = LogisticRegression()

    X_tr, X_blend, y_tr, y_blend = train_test_split(
        X_train, y_train, test_size=blend_size, random_state=42
    )

    blend_features = []
    test_features = []
    for name, model in base_models:
        model.fit(X_tr, y_tr)
        blend_features.append(model.predict_proba(X_blend)[:, 1])
        test_features.append(model.predict_proba(X_test)[:, 1])

    meta_features = np.column_stack(blend_features)
    meta_model.fit(meta_features, y_blend)

    test_meta = np.column_stack(test_features)
    return meta_model.predict(test_meta)


# --- Bagging ---

def bagged_predictions(X_train, y_train, X_test, model_factory, n_bags=10):
    """Bagged predictions from bootstrapped models. Returns averaged predictions."""
    predictions = []
    for i in range(n_bags):
        X_boot, y_boot = resample(X_train, y_train, replace=True, random_state=i)
        model = model_factory()
        model.fit(X_boot, y_boot)
        predictions.append(model.predict_proba(X_test)[:, 1])
    return np.mean(predictions, axis=0)


# --- Boosting defaults ---

def xgboost_classifier(**kwargs):
    """Create an XGBoost classifier with sensible defaults."""
    from xgboost import XGBClassifier
    defaults = dict(
        n_estimators=100, learning_rate=0.1, max_depth=3,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        eval_metric="logloss", random_state=42,
    )
    defaults.update(kwargs)
    return XGBClassifier(**defaults)


def lightgbm_classifier(**kwargs):
    """Create a LightGBM classifier with sensible defaults."""
    from lightgbm import LGBMClassifier
    defaults = dict(
        n_estimators=100, learning_rate=0.1, max_depth=3,
        num_leaves=31, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, verbose=-1, random_state=42,
    )
    defaults.update(kwargs)
    return LGBMClassifier(**defaults)


def catboost_classifier(**kwargs):
    """Create a CatBoost classifier with sensible defaults."""
    from catboost import CatBoostClassifier
    defaults = dict(
        depth=6, learning_rate=0.1, iterations=100,
        l2_leaf_reg=3.0, random_seed=42, verbose=False,
    )
    defaults.update(kwargs)
    return CatBoostClassifier(**defaults)


# --- Ensemble pruning ---

def ensemble_selection(models, X_val, y_val, n_select=None, scoring_fn=None):
    """Caruana's greedy ensemble selection. Returns list of selected models."""
    if n_select is None:
        n_select = max(1, len(models) // 2)
    if scoring_fn is None:
        from sklearn.metrics import accuracy_score as scoring_fn

    predictions = np.array([m.predict_proba(X_val)[:, 1] for m in predictions])

    selected = []
    ensemble_pred = np.zeros(len(y_val))

    for _ in range(n_select):
        best_score = -np.inf
        best_idx = None
        for i in range(len(models)):
            if i in selected:
                continue
            test_pred = (ensemble_pred + predictions[i]) / (len(selected) + 1)
            score = scoring_fn(y_val, (test_pred > 0.5).astype(int))
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx is not None:
            selected.append(best_idx)
            ensemble_pred = (ensemble_pred * len(selected) + predictions[best_idx]) / (len(selected) + 1)

    return [models[i] for i in selected]


# --- Diversity metrics ---

def disagreement_measure(y_true, preds_a, preds_b):
    """Fraction of examples where two models disagree."""
    return np.mean(preds_a != preds_b)


def double_fault_measure(y_true, preds_a, preds_b):
    """Fraction of examples where both models are wrong."""
    return np.mean((preds_a != y_true) & (preds_b != y_true))


def q_statistics(y_true, preds_a, preds_b):
    """Q-statistic: correlation between model errors. Range -1 (diverse) to 1 (similar)."""
    n11 = np.sum((preds_a == y_true) & (preds_b == y_true))
    n00 = np.sum((preds_a != y_true) & (preds_b != y_true))
    n10 = np.sum((preds_a == y_true) & (preds_b != y_true))
    n01 = np.sum((preds_a != y_true) & (preds_b == y_true))
    denom = n11 * n00 + n01 * n10
    if denom == 0:
        return 0.0
    return (n11 * n00 - n01 * n10) / denom


def entropy_diversity(predictions):
    """Average prediction entropy for probabilistic ensemble predictions.

    predictions: (n_models, n_samples, n_classes) array.
    """
    mean_pred = predictions.mean(axis=0)
    entropy = -(mean_pred * np.log(mean_pred + 1e-10)).sum(axis=1)
    return entropy.mean()
