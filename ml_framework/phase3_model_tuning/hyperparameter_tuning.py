"""
Hyperparameter tuning: grid search, random search, Bayesian optimization, multi-phase, nested CV, early stopping.
"""

import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint, uniform


def grid_search_baseline(X_train, y_train, param_grid=None, estimator=None):
    """Grid search baseline. Returns fitted GridSearchCV."""
    if estimator is None:
        estimator = RandomForestClassifier(random_state=42)
    if param_grid is None:
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [5, 10, 15, None],
            "min_samples_split": [2, 5, 10],
        }
    gs = GridSearchCV(estimator, param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)
    return gs


def random_search_baseline(X_train, y_train, n_iter=50, estimator=None):
    """Random search baseline. Returns fitted RandomizedSearchCV."""
    if estimator is None:
        estimator = RandomForestClassifier(random_state=42)
    param_distributions = {
        "n_estimators": randint(50, 500),
        "max_depth": randint(3, 20),
        "min_samples_split": uniform(0.01, 0),
        "min_samples_leaf": uniform(0.01, 0.05),
        "max_features": uniform(0.1, 0.9),
    }
    rs = RandomizedSearchCV(estimator, param_distributions, n_iter=n_iter, cv=5, scoring="accuracy", n_jobs=-1, random_state=42)
    rs.fit(X_train, y_train)
    return rs


def bayesian_optimization_tpe(X_train, y_train, X_val, y_val, n_trials=100, estimator_type="xgboost"):
    """Bayesian optimization with Optuna TPE sampler. Returns Optuna study."""
    import optuna

    def objective(trial):
        if estimator_type == "xgboost":
            import xgboost as xgb
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            }
            model = xgb.XGBClassifier(**params, random_state=42, eval_metric="logloss")
        else:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            }
            model = RandomForestClassifier(**params, random_state=42)
        model.fit(X_train, y_train)
        return model.score(X_val, y_val)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    return study


class MultiPhaseOptimizer:
    """Structured multi-phase hyperparameter optimization."""

    def __init__(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

    def phase1_coarse_search(self):
        """Wide ranges, 50 trials."""
        return bayesian_optimization_tpe(self.X_train, self.y_train, self.X_val, self.y_val, n_trials=50)

    def phase2_refined_search(self, best_params):
        """Narrow ranges around best config, 100 trials."""
        import optuna

        def objective(trial):
            import xgboost as xgb
            params = {
                "n_estimators": trial.suggest_int("n_estimators", max(50, best_params.get("n_estimators", 300) - 100), best_params.get("n_estimators", 300) + 100),
                "max_depth": trial.suggest_int("max_depth", max(3, best_params.get("max_depth", 6) - 3), best_params.get("max_depth", 6) + 3),
                "learning_rate": trial.suggest_float("learning_rate", best_params.get("learning_rate", 0.05) / 3, best_params.get("learning_rate", 0.05) * 3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            }
            model = xgb.XGBClassifier(**params, random_state=42, eval_metric="logloss")
            model.fit(self.X_train, self.y_train)
            return model.score(self.X_val, self.y_val)

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=100)
        return study

    def run_all(self):
        """Execute all 3 phases sequentially."""
        study1 = self.phase1_coarse_search()
        best = study1.best_params
        study2 = self.phase2_refined_search(best)
        return study2, study2.best_params


def cv_objective(trial, X, y, n_folds=5, estimator_type="xgboost"):
    """Objective function with K-fold CV for robust evaluation."""
    import optuna

    if estimator_type == "xgboost":
        import xgboost as xgb
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
        }
        model_fn = lambda: xgb.XGBClassifier(**params, random_state=42, eval_metric="logloss")
    else:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
        }
        model_fn = lambda: RandomForestClassifier(**params, random_state=42)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in kf.split(X):
        model = model_fn()
        model.fit(X[train_idx], y[train_idx])
        scores.append(model.score(X[val_idx], y[val_idx]))
    return np.mean(scores)


def nested_cv_evaluation(X, y, estimator_type="xgboost", n_outer=5, n_inner_trials=50):
    """Nested CV for unbiased hyperparameter evaluation."""
    import optuna

    outer_cv = KFold(n_splits=n_outer, shuffle=True, random_state=42)
    outer_scores = []
    for train_idx, test_idx in outer_cv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: cv_objective(trial, X_train, y_train, estimator_type=estimator_type), n_trials=n_inner_trials)
        if estimator_type == "xgboost":
            import xgboost as xgb
            best_model = xgb.XGBClassifier(**study.best_params, random_state=42)
        else:
            best_model = RandomForestClassifier(**study.best_params, random_state=42)
        best_model.fit(X_train, y_train)
        outer_scores.append(best_model.score(X_test, y_test))
    return np.mean(outer_scores), np.std(outer_scores)


class EarlyStopping:
    """Early stopping that tracks best model weights."""

    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False

    def __call__(self, val_score, model):
        if self.best_score is None:
            self.best_score = val_score
            self.save_checkpoint(model)
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
        else:
            self.best_score = val_score
            self.save_checkpoint(model)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}


class AdaptiveEarlyStopping(EarlyStopping):
    """Early stopping that increases patience over time."""

    def __init__(self, initial_patience=10, max_patience=50, **kwargs):
        super().__init__(patience=initial_patience, **kwargs)
        self.initial_patience = initial_patience
        self.max_patience = max_patience
        self.epoch = 0

    def __call__(self, val_score, model):
        self.epoch += 1
        self.patience = int(min(self.initial_patience + self.epoch // 10, self.max_patience))
        return super().__call__(val_score, model)
