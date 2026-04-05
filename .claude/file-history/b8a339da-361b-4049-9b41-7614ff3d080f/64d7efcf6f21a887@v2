"""
Regularization: classical (L1, L2, ElasticNet), neural network (dropout, batchnorm, weight decay),
advanced (mixup, cutmix, label smoothing), early stopping, data augmentation, bias-variance diagnostics.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# --- Classical regularization ---

def l1_feature_selection(X_train, y_train, C=0.1, task="classification"):
    """L1-regularized feature selection. Returns selected feature indices."""
    from sklearn.linear_model import LogisticRegression, Lasso
    from sklearn.feature_selection import SelectFromModel

    if task == "classification":
        model = LogisticRegression(penalty="l1", solver="saga", C=C, random_state=42, max_iter=5000)
    else:
        model = Lasso(alpha=C, random_state=42)

    selector = SelectFromModel(model)
    selector.fit(X_train, y_train)
    return selector.get_support(indices=True)


def ridge_regression_model(alpha=1.0):
    """Create a Ridge regression model."""
    from sklearn.linear_model import Ridge
    return Ridge(alpha=alpha)


def elastic_net_model(alpha=1.0, l1_ratio=0.5, task="classification"):
    """Create an ElasticNet-regularized model."""
    from sklearn.linear_model import ElasticNet, LogisticRegression
    if task == "classification":
        return LogisticRegression(
            penalty="elasticnet", solver="saga", l1_ratio=l1_ratio, C=1.0 / alpha,
            random_state=42, max_iter=5000,
        )
    return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)


# --- Neural network regularization ---

class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing cross-entropy loss."""

    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_classes = pred.size(-1)
        log_preds = torch.log_softmax(pred, dim=-1)
        with torch.no_grad():
            smooth_target = torch.zeros_like(pred)
            smooth_target.fill_(self.smoothing / (n_classes - 1))
            smooth_target.scatter_(1, target.unsqueeze(1), 1 - self.smoothing)
        return torch.sum(-smooth_target * log_preds, dim=-1).mean()


def mixup_data(x, y, alpha=0.2):
    """Mixup data augmentation: linear interpolation between samples."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss combining two targets."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def cutmix_data(x, y, alpha=1.0):
    """CutMix data augmentation for image data."""
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size)

    W, H = x.size(2), x.size(3)
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    mixed_x = x.clone()
    mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    return mixed_x, y, y[index], lam


# --- Tabular augmentation ---

def add_gaussian_noise(X, noise_level=0.01):
    """Add Gaussian noise to features for regularization."""
    return X + np.random.normal(0, noise_level, X.shape)


def tabular_mixup(X, y, alpha=0.2):
    """Mixup for tabular data."""
    lam = np.random.beta(alpha, alpha)
    index = np.random.permutation(len(X))
    return lam * X + (1 - lam) * X[index], lam * y + (1 - lam) * y[index]


# --- Optimizer builders ---

def adamw_optimizer(model, lr=1e-4, weight_decay=0.01, layer_decay=None):
    """Create AdamW optimizer, optionally with per-layer weight decay."""
    if layer_decay is not None:
        params = [
            {"params": model.base.parameters(), "weight_decay": layer_decay.get("base", weight_decay)},
            {"params": model.head.parameters(), "weight_decay": layer_decay.get("head", weight_decay)},
        ]
    else:
        params = [{"params": model.parameters(), "weight_decay": weight_decay}]
    return optim.AdamW(params, lr=lr, betas=(0.9, 0.999))


# --- Bias-variance diagnostics ---

def diagnose_bias_variance(model, X_train, y_train, X_val, y_val):
    """Quick bias-variance diagnosis from train vs val performance. Returns dict."""
    from sklearn.metrics import accuracy_score

    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    val_acc = accuracy_score(y_val, model.predict(X_val))
    gap = train_acc - val_acc

    if gap > 0.15:
        diagnosis = "high_variance"
    elif val_acc < 0.7:
        diagnosis = "high_bias"
    elif gap < 0.05 and val_acc > 0.85:
        diagnosis = "good_fit"
    else:
        diagnosis = "moderate"

    return {
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "gap": gap,
        "diagnosis": diagnosis,
    }


def learning_curve_data(estimator, X, y, cv=5, train_sizes=None, scoring="accuracy"):
    """Get learning curve data for bias-variance diagnosis. Returns dict of arrays."""
    from sklearn.model_selection import learning_curve

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


def validation_curve_data(estimator, X, y, param_name, param_range, cv=5, scoring="accuracy"):
    """Get validation curve data for a single hyperparameter. Returns dict of arrays."""
    from sklearn.model_selection import validation_curve

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
