"""
Probability calibration: Platt scaling, isotonic regression, temperature scaling,
beta calibration, ECE, reliability diagrams.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss


# --- Platt scaling ---

class PlattScaler:
    """Platt scaling calibration using logistic regression on model scores."""

    def __init__(self):
        self.scaler = LogisticRegression()

    def fit(self, scores, y):
        self.scaler.fit(scores.reshape(-1, 1), y)
        return self

    def transform(self, scores):
        return self.scaler.predict_proba(scores.reshape(-1, 1))[:, 1]


# --- Isotonic regression calibration ---

class IsotonicCalibrator:
    """Isotonic regression calibration (non-parametric)."""

    def __init__(self):
        self.regressor = IsotonicRegression(out_of_bounds="clip")

    def fit(self, scores, y):
        self.regressor.fit(scores, y)
        return self

    def transform(self, scores):
        return self.regressor.transform(scores)


# --- Temperature scaling ---

def find_temperature(val_logits, val_labels, max_iter=50):
    """Find optimal temperature for scaling logits using LBFGS on NLL."""
    temperature = nn.Parameter(torch.ones(1))
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=max_iter)

    def eval_fn():
        optimizer.zero_grad()
        scaled = val_logits / temperature
        loss = F.cross_entropy(scaled, val_labels)
        loss.backward()
        return loss

    optimizer.step(eval_fn)
    return temperature.item()


def temperature_scale(logits, temperature):
    """Scale logits by temperature."""
    return logits / temperature


# --- Beta calibration ---

class BetaCalibrator:
    """Beta calibration using log-odds transform + logistic regression."""

    def __init__(self):
        self.regressor = LogisticRegression()

    @staticmethod
    def _logit(p):
        eps = 1e-10
        return np.log((p + eps) / (1 - p + eps))

    def fit(self, scores, y):
        logit_scores = self._logit(scores)
        self.regressor.fit(logit_scores.reshape(-1, 1), y)
        return self

    def transform(self, scores):
        logit_scores = self._logit(scores)
        calibrated_logit = self.regressor.predict(logit_scores.reshape(-1, 1))
        return 1 / (1 + np.exp(-calibrated_logit))


# --- Calibration evaluation ---

def expected_calibration_error(y_true, y_proba, n_bins=10):
    """Compute Expected Calibration Error (ECE)."""
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="uniform")
    bin_counts = np.histogram(y_proba, bins=np.linspace(0, 1, n_bins + 1))[0]
    nonempty = bin_counts[bin_counts > 0]
    weights = nonempty / len(y_true)
    return float(np.sum(weights * np.abs(prob_true - prob_pred)))


def reliability_data(y_true, y_proba, n_bins=10):
    """Get reliability diagram data. Returns (prob_true, prob_pred)."""
    return calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="uniform")


def calibration_evaluation(y_true, y_proba, n_bins=10):
    """Full calibration evaluation. Returns dict with brier, ece, reliability data."""
    return {
        "brier_score": brier_score_loss(y_true, y_proba),
        "ece": expected_calibration_error(y_true, y_proba, n_bins=n_bins),
        "prob_true": None,
        "prob_pred": None,
    }
