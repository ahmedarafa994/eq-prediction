# Section 5: Post-Processing, Decision Framework & Pitfalls

This section provides a comprehensive guide to post-processing techniques, decision-making frameworks for technique selection, and a catalogue of common pitfalls with diagnostic procedures. By the end of this section, you will understand how to calibrate model outputs, optimize decision thresholds, apply fairness-aware post-processing, select appropriate techniques based on constraints, and detect/avoid common ML pitfalls.

## Part A: Post-Processing & Calibration

### 1. Probability Calibration

**Why Calibration Matters**

A well-calibrated classifier produces probability estimates that reflect true likelihoods. If a model predicts 0.7 for 100 instances, approximately 70 should actually be positive. Modern neural networks, especially deep networks, tend to be overconfident—their predicted probabilities are systematically higher than empirical frequencies. Calibration is crucial when:
- Probabilities inform downstream decisions (e.g., medical triage, risk assessment)
- You need to compare models across different probability thresholds
- Stakeholders interpret probabilities directly (e.g., "70% chance of fraud")

**Platt Scaling**

Platt scaling (Platt, 1999) is a parametric calibration method that fits a logistic regression model to the model's outputs. Given uncalibrated scores $s$, the calibrated probability is:

$$P(y=1|s) = \frac{1}{1 + \exp(A \cdot s + B)}$$

where parameters $A$ and $B$ are optimized on a held-out calibration set.

*Implementation:*
```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Assuming X_train, y_train, X_test available
X_train_main, X_cal, y_train_main, y_cal = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train
)

# Train base model
model.fit(X_train_main, y_train_main)

# Get uncalibrated scores on calibration set
uncalibrated_probs = model.predict_proba(X_cal)[:, 1]

# Fit Platt scaling
platt_scaler = LogisticRegression()
platt_scaler.fit(uncalibrated_probs.reshape(-1, 1), y_cal)

# Apply calibration at inference
test_scores = model.predict_proba(X_test)[:, 1]
calibrated_probs = platt_scaler.predict_proba(test_scores.reshape(-1, 1))[:, 1]
```

*When to use Platt scaling:*
- Small calibration sets (hundreds of samples)
- When you need a simple, interpretable transformation
- As a baseline before trying more complex methods

*Limitations:*
- Assumes sigmoid relationship between scores and true probabilities
- Can be outperformed by non-parametric methods with sufficient calibration data

**Isotonic Regression**

Isotonic regression is a non-parametric calibration method that fits a piecewise-constant, monotonically increasing function to map scores to probabilities. It makes no assumptions about the functional form, requiring only monotonicity.

*Implementation:*
```python
from sklearn.isotonic import IsotonicRegression

# Fit isotonic regression
iso_regressor = IsotonicRegression(out_of_bounds='clip')
iso_regressor.fit(uncalibrated_probs, y_cal)

# Apply calibration
calibrated_probs = iso_regressor.transform(test_scores)
```

*When to use isotonic regression:*
- Large calibration sets (thousands of samples)
- Complex mis-calibration patterns that Platt scaling cannot capture
- When computational overhead at inference is acceptable

*Limitations:*
- Requires more calibration data than Platt scaling
- Can overfit on small calibration sets
- Slightly slower at inference (requires storing piecewise function)

**Temperature Scaling**

Temperature scaling (Guo et al., 2017) is the most popular calibration method for deep neural networks. It introduces a single scalar parameter $T$ (temperature) that softens the output distribution:

$$P(y=i|x) = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}$$

where $z$ are the logits. When $T=1$, no scaling occurs; $T>1$ softens (reduces confidence), while $T<1$ sharpens (increases confidence).

*Implementation:*
```python
import torch
import torch.nn.functional as F

def temperature_scale(logits, temperature):
    return logits / temperature

def find_temperature(val_logits, val_labels):
    # Optimize temperature to minimize NLL on validation set
    temperature = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.LBFGS([temperature], lr=0.01)

    def eval():
        optimizer.zero_grad()
        scaled_logits = temperature_scale(val_logits, temperature)
        loss = F.cross_entropy(scaled_logits, val_labels)
        loss.backward()
        return loss

    optimizer.step(eval)
    return temperature.item()

# Usage
temperature = find_temperature(val_logits, val_labels)
test_logits = model(X_test)
scaled_logits = test_logits / temperature
calibrated_probs = F.softmax(scaled_logits, dim=-1)
```

*Advantages of temperature scaling:*
- Single parameter makes it data-efficient
- Does not change the model's rank ordering (preserves accuracy)
- No significant inference overhead
- Specifically designed for deep neural networks

**Beta Calibration**

Beta calibration (Kull et al., 2017) generalizes Platt scaling by using a beta distribution instead of logistic regression. It can model more complex probability distributions, including U-shaped and multimodal mis-calibration patterns:

$$P(y=1|s) = \frac{1}{1 + \exp(-(\ln(s) - \ln(1-s)) \cdot a - b)}$$

This parameterization works directly on probability inputs and can capture asymmetric behavior.

*Implementation:*
```python
from sklearn.preprocessing import FunctionTransformer
import numpy as np

def logit(p):
    """Convert probability to log-odds"""
    eps = 1e-10
    return np.log((p + eps) / (1 - p + eps))

# Transform probabilities to log-odds space
logit_scores = logit(uncalibrated_probs)

# Fit linear regression in log-odds space
from sklearn.linear_model import LinearRegression
beta_regressor = LinearRegression()
beta_regressor.fit(logit_scores.reshape(-1, 1), y_cal)

# Apply calibration
test_logit = logit(test_scores)
calibrated_logit = beta_regressor.predict(test_logit.reshape(-1, 1))
calibrated_probs = 1 / (1 + np.exp(-calibrated_logit))
```

*When to use beta calibration:*
- When Platt scaling fails to capture mis-calibration patterns
- For models that output probabilities directly (not logits)
- When you need to handle extreme probabilities near 0 or 1

**Practical Calibration Workflow**

1. **Split your data**: training (main model), calibration (post-processor), testing (final evaluation). Typical split: 70/15/15 or 80/10/10.

2. **Select calibration method**:
   - Small calibration set (<500 samples): Platt scaling or temperature scaling
   - Large calibration set (>2000 samples): Isotonic regression
   - Deep neural nets: Temperature scaling (preferred)
   - Simple binary models: Platt scaling (good baseline)

3. **Evaluate calibration quality**:
   - **Reliability diagrams**: Bin predictions and compare mean predicted probability to empirical frequency
   - **Expected Calibration Error (ECE)**: Weighted average of calibration error across bins
   - **Brier score**: Mean squared error of predicted probabilities
   - **Log loss**: Negative log-likelihood

```python
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

def plot_reliability_diagram(y_true, y_prob, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)

    plt.figure(figsize=(10, 6))
    plt.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')
    plt.plot(prob_pred, prob_true, 's-', label='Model')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('True probability in each bin')
    plt.title('Reliability Diagram')
    plt.legend()
    plt.show()

def compute_ece(y_true, y_prob, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    bin_counts = np.histogram(y_prob, bins=np.linspace(0, 1, n_bins + 1))[0]
    ece = np.sum(np.abs(prob_true - prob_pred) * bin_counts) / len(y_true)
    return ece
```

4. **Monitor calibration over time**: Recalibration is often needed as data distributions shift.

### 2. Conformal Prediction

**Introduction to Conformal Prediction**

Conformal prediction is a distribution-free framework for uncertainty quantification that provides statistically valid prediction regions under minimal assumptions. Unlike traditional confidence intervals, conformal prediction guarantees coverage properties without requiring distributional assumptions about the data.

Key advantages:
- **Distribution-free**: Works under the exchangeability assumption only
- **Coverage guarantees**: With probability at least 1-α, the true label falls within the prediction set
- **Model-agnostic**: Can wrap any machine learning model
- **Applicability**: Works for classification, regression, and other tasks

**Split Conformal Prediction**

The simplest form of conformal prediction uses a held-out calibration set to compute nonconformity scores.

*For classification:*
```python
def split_conformal_classification(model, X_cal, y_cal, X_test, alpha=0.1):
    """
    Returns prediction sets with 1-alpha coverage guarantee
    """
    n_classes = len(model.classes_)
    n_cal = len(X_cal)

    # Get calibration scores
    cal_probs = model.predict_proba(X_cal)
    cal_scores = 1 - cal_probs[np.arange(n_cal), y_cal]

    # Compute quantile
    qhat = np.quantile(cal_scores, np.ceil((n_cal + 1) * (1 - alpha)) / n_cal, method='higher')

    # Construct prediction sets for test data
    test_probs = model.predict_proba(X_test)
    prediction_sets = [np.where(test_probs[i] >= 1 - qhat)[0]
                       for i in range(len(X_test))]

    return prediction_sets, qhat
```

*For regression:*
```python
def split_conformal_regression(model, X_cal, y_cal, X_test, alpha=0.1):
    """
    Returns prediction intervals with 1-alpha coverage guarantee
    """
    n_cal = len(X_cal)

    # Compute absolute residuals on calibration set
    y_pred_cal = model.predict(X_cal)
    cal_scores = np.abs(y_cal - y_pred_cal)

    # Compute quantile
    qhat = np.quantile(cal_scores, np.ceil((n_cal + 1) * (1 - alpha)) / n_cal, method='higher')

    # Construct intervals for test data
    y_pred_test = model.predict(X_test)
    intervals = [(y_pred_test[i] - qhat, y_pred_test[i] + qhat)
                 for i in range(len(X_test))]

    return intervals, qhat
```

**Coverage Guarantees**

Under the exchangeability assumption (data points are exchangeable—i.i.d. is a special case), conformal prediction provides marginal coverage:

$$P(Y_{n+1} \in \hat{C}_{n+1}) \geq 1 - \alpha$$

This guarantee holds regardless of the underlying distribution or model quality. Even with a poorly calibrated model, conformal prediction maintains coverage (though prediction sets may become larger/less informative).

**Adaptive Conformal Inference**

Standard conformal prediction produces intervals of constant width, which can be inefficient when uncertainty varies across the input space. Adaptive methods like CQR (Conformalized Quantile Regression) address this:

```python
from sklearn.ensemble import GradientBoostingRegressor

def conformalized_quantile_regression(X_train, y_train, X_cal, y_cal, X_test, alpha=0.1):
    """
    CQR: Adaptive intervals that vary with input
    """
    # Train quantile regression models
    lower_model = GradientBoostingRegressor(loss='quantile', alpha=alpha/2)
    upper_model = GradientBoostingRegressor(loss='quantile', alpha=1-alpha/2)

    lower_model.fit(X_train, y_train)
    upper_model.fit(X_train, y_train)

    # Compute conformity scores on calibration set
    lower_cal = lower_model.predict(X_cal)
    upper_cal = upper_model.predict(X_cal)
    cal_scores = np.maximum(y_cal - upper_cal, lower_cal - y_cal)

    # Compute quantile
    n_cal = len(X_cal)
    qhat = np.quantile(cal_scores, np.ceil((n_cal + 1) * (1 - alpha)) / n_cal, method='higher')

    # Construct intervals for test data
    lower_test = lower_model.predict(X_test)
    upper_test = upper_model.predict(X_test)
    intervals = [(lower_test[i] - qhat, upper_test[i] + qhat)
                 for i in range(len(X_test))]

    return intervals
```

**Practical Implementation Considerations**

1. **Calibration set size**: 500-2000 samples typically sufficient for α=0.1 coverage. Smaller α requires larger calibration sets.

2. **Cross-conformal prediction**: More efficient use of data through cross-validation:
```python
from sklearn.model_selection import KFold

def cross_conformal(model, X, y, alpha=0.1, n_folds=5):
    """
    Cross-conformal prediction for better data efficiency
    """
    kf = KFold(n_splits=n_folds)
    all_scores = []

    for train_idx, cal_idx in kf.split(X):
        X_train_fold, X_cal_fold = X[train_idx], X[cal_idx]
        y_train_fold, y_cal_fold = y[train_idx], y[cal_idx]

        model.fit(X_train_fold, y_train_fold)
        cal_probs = model.predict_proba(X_cal_fold)
        cal_scores = 1 - cal_probs[np.arange(len(y_cal_fold)), y_cal_fold]
        all_scores.extend(cal_scores)

    # Compute global quantile from all folds
    qhat = np.quantile(all_scores, np.ceil((len(X) + 1) * (1 - alpha)) / len(X), method='higher')
    return qhat
```

3. **Coverage vs. efficiency trade-off**: Smaller α (higher coverage) produces larger prediction sets. Choose α based on application requirements:
   - High-stakes (medical diagnosis): α=0.01 or 0.05 (99-95% coverage)
   - Medium-stakes (fraud detection): α=0.1 (90% coverage)
   - Low-stakes (recommendation): α=0.2 (80% coverage)

4. **Time-series data**: Standard conformal prediction fails when exchangeability is violated. Use conformal prediction for time series with adaptive methods:
```python
def aci_conformal(model, X_cal, y_cal, X_test_stream, alpha=0.1, gamma=0.01):
    """
    Adaptive Conformal Inference for time-varying distributions
    """
    qhat = 0
    coverage_history = []

    for t, X_t in enumerate(X_test_stream):
        # Make prediction with current qhat
        pred = model.predict(X_t.reshape(1, -1))[0]
        interval = (pred - qhat, pred + qhat)

        # Get actual value (in streaming scenario, this arrives later)
        if t < len(y_cal):
            y_true = y_cal[t]
            covered = (interval[0] <= y_true <= interval[1])
            coverage_history.append(covered)

            # Update qhat based on recent coverage
            if len(coverage_history) >= 100:
                recent_coverage = np.mean(coverage_history[-100:])
                if recent_coverage < 1 - alpha:
                    qhat += gamma  # Increase interval width
                elif recent_coverage > 1 - alpha + gamma:
                    qhat = max(0, qhat - gamma)  # Decrease interval width

        yield interval, qhat
```

**Python Libraries**

Several libraries provide conformal prediction implementations:

- **MAPIE** (Monotone Adjustment for Prediction Intervals Estimation):
```python
from mapie.classification import MapieClassifier
from mapie.regression import MapieRegressor

# Classification
mapie_clf = MapieClassifier(estimator=model, cv='prefit', method='score')
mapie_clf.fit(X_cal, y_cal)
y_pred, y_set = mapie_clf.predict(X_test, alpha=0.1)

# Regression
mapie_reg = MapieRegressor(estimator=model, cv='prefit')
mapie_reg.fit(X_cal, y_cal)
y_pred, y_pis = mapie_reg.predict(X_test, alpha=0.1)
```

- **Crema**: Simple API for conformal regression
- **Fortuna**: Lightweight conformal prediction for PyTorch models

### 3. Threshold Optimization

Default thresholds (0.5 for binary classification) are rarely optimal. Threshold optimization aligns decision boundaries with business objectives.

**Precision-Recall Trade-off**

The ROC curve shows the trade-off between true positive rate and false positive rate across thresholds. The precision-recall curve is more informative for imbalanced datasets.

*Optimizing for F1 score:*
```python
from sklearn.metrics import precision_recall_curve, f1_score

def find_optimal_threshold_f1(y_true, y_prob):
    """
    Find threshold that maximizes F1 score
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)

    # Handle NaN from division by zero
    f1_scores = np.nan_to_num(f1_scores)

    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]

    return optimal_threshold, optimal_f1
```

*Visualizing precision-recall trade-off:*
```python
def plot_precision_recall_threshold(y_true, y_prob):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(thresholds, precisions[:-1], 'b--', label='Precision')
    plt.plot(thresholds, recalls[:-1], 'g-', label='Recall')
    plt.xlabel('Threshold')
    plt.legend()
    plt.title('Precision vs Recall')

    plt.subplot(1, 3, 2)
    plt.plot(thresholds, f1_scores[:-1], 'r-')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title('F1 vs Threshold')

    plt.subplot(1, 3, 3)
    plt.plot(recalls, precisions, 'b-')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')

    plt.tight_layout()
    plt.show()
```

**Cost-Sensitive Thresholding**

When false positives and false negatives have different costs, optimize for expected cost rather than classification metrics.

```python
def cost_optimal_threshold(y_true, y_prob, cost_fp, cost_fn, cost_tp=0, cost_tn=0):
    """
    Find threshold minimizing expected cost
    """
    thresholds = np.unique(y_prob)
    costs = []

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        total_cost = (fp * cost_fp + fn * fn + tp * cost_tp + tn * cost_tn)
        costs.append(total_cost)

    optimal_idx = np.argmin(costs)
    optimal_threshold = thresholds[optimal_idx]

    return optimal_threshold, costs[optimal_idx]

# Example: Fraud detection where missing fraud is 10x worse than false alarms
optimal_threshold, min_cost = cost_optimal_threshold(
    y_true, y_prob, cost_fp=1, cost_fn=10
)
```

**Profit-Maximizing Thresholds**

For business applications, directly optimize for profit or utility:

```python
def profit_maximizing_threshold(y_true, y_prob, tp_value, fp_cost, fn_cost, tn_value=0):
    """
    Maximize expected profit: TP*value - FP*cost - FN*cost
    """
    thresholds = np.linspace(0, 1, 101)
    profits = []

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        profit = (tp * tp_value + tn * tn_value - fp * fp_cost - fn * fn_cost)
        profits.append(profit)

    optimal_idx = np.argmax(profits)
    return thresholds[optimal_idx], profits[optimal_idx]
```

**Youden's J Statistic**

Youden's J maximizes the difference between true positive rate and false positive rate:

$$J = \text{TPR} - \text{FPR} = \text{Sensitivity} + \text{Specificity} - 1$$

```python
def youdens_j_threshold(y_true, y_prob):
    """
    Find threshold maximizing Youden's J statistic
    """
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr

    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_j = j_scores[optimal_idx]

    return optimal_threshold, optimal_j
```

Youden's J is particularly useful when you want to balance sensitivity and specificity equally, as in medical screening tests.

**Threshold Selection by Constraint**

Sometimes you need to satisfy a specific constraint (e.g., minimum recall):

```python
def threshold_for_constraint(y_true, y_prob, metric='recall', min_value=0.95):
    """
    Find threshold achieving minimum recall/precision
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    if metric == 'recall':
        valid_idx = np.where(recalls >= min_value)[0]
        if len(valid_idx) > 0:
            best_idx = valid_idx[np.argmax(precisions[valid_idx])]
            return thresholds[best_idx], precisions[best_idx], recalls[best_idx]

    elif metric == 'precision':
        valid_idx = np.where(precisions >= min_value)[0]
        if len(valid_idx) > 0:
            best_idx = valid_idx[np.argmax(recalls[valid_idx])]
            return thresholds[best_idx], precisions[best_idx], recalls[best_idx]

    return None, None, None
```

**Ensemble Thresholding**

For ensemble methods, consider thresholding at the ensemble level rather than individual models:

```python
def ensemble_threshold(models, X_cal, y_cal, X_test, method='soft_voting'):
    """
    Optimize threshold for ensemble predictions
    """
    if method == 'soft_voting':
        # Average probabilities
        cal_probs = np.mean([m.predict_proba(X_cal)[:, 1] for m in models], axis=0)
        test_probs = np.mean([m.predict_proba(X_test)[:, 1] for m in models], axis=0)

    elif method == 'hard_voting':
        # Majority vote, then find threshold
        cal_probs = np.mean([m.predict(X_cal) for m in models], axis=0)
        test_probs = np.mean([m.predict(X_test) for m in models], axis=0)

    # Find optimal threshold
    optimal_threshold, _ = find_optimal_threshold_f1(y_cal, cal_probs)
    test_predictions = (test_probs >= optimal_threshold).astype(int)

    return test_predictions, optimal_threshold
```

### 4. Post-Processing for Fairness

Post-processing techniques mitigate bias in model predictions without retraining. These methods are attractive because they're model-agnostic and can be applied after deployment.

**Equalized Odds Post-Processing**

Equalized odds requires that true positive rates and false positive rates are equal across groups. Hardt et al. (2016) derived a post-processing algorithm that finds optimal randomized thresholds.

```python
from scipy.optimize import linprog

def equalized_odds_postprocessing(y_true, y_prob, sensitive_attr, base_rate=None):
    """
    Derive group-specific thresholds satisfying equalized odds
    Simplified version without randomization
    """
    groups = np.unique(sensitive_attr)
    thresholds = {}

    if base_rate is None:
        base_rate = np.mean(y_true)

    for group in groups:
        group_mask = (sensitive_attr == group)
        y_true_group = y_true[group_mask]
        y_prob_group = y_prob[group_mask]

        # Find threshold achieving base rate for this group
        def find_target_rate(target_rate):
            for threshold in np.linspace(0, 1, 101):
                pred_rate = np.mean((y_prob_group >= threshold).astype(int))
                if pred_rate >= target_rate:
                    return threshold
            return 1.0

        thresholds[group] = find_target_rate(base_rate)

    return thresholds

# Usage
thresholds = equalized_odds_postprocessing(y_val, y_prob_val, sensitive_attr_val)
predictions = [(y_prob_test[i] >= thresholds[sensitive_attr_test[i]]).astype(int)
               for i in range(len(y_prob_test))]
```

**Reject Option Classification**

Reject option classification (Kamiran et al., 2012) adjusts predictions near the decision boundary:

```python
def reject_option_classification(y_prob, sensitive_attr, threshold=0.5, band=0.1):
    """
    Flip predictions in uncertainty band to favor disadvantaged group
    """
    predictions = (y_prob >= threshold).astype(int)
    groups = np.unique(sensitive_attr)

    # Determine which group is disadvantaged (lower positive rate)
    group_rates = {g: np.mean(predictions[sensitive_attr == g]) for g in groups}
    disadvantaged_group = min(group_rates, key=group_rates.get)

    # Flip predictions in uncertainty band
    for i in range(len(y_prob)):
        if abs(y_prob[i] - threshold) < band:
            if sensitive_attr[i] == disadvantaged_group:
                predictions[i] = 1 - predictions[i]

    return predictions
```

**Calibrated Equalized Odds**

Pleiss et al. (2017) derived a method for deriving group-specific thresholds that maintain both calibration and equalized odds:

```python
def calibrated_equalized_odds(y_true, y_prob, sensitive_attr, alpha=0.05):
    """
    Find group-specific thresholds satisfying equalized odds while preserving calibration
    """
    groups = np.unique(sensitive_attr)
    result = {}

    for group in groups:
        group_mask = (sensitive_attr == group)
        y_true_g = y_true[group_mask]
        y_prob_g = y_prob[group_mask]

        # For each threshold, compute TPR and FPR
        thresholds = np.linspace(0, 1, 101)
        tprs = []
        fprs = []

        for t in thresholds:
            y_pred = (y_prob_g >= t).astype(int)
            tp = np.sum((y_pred == 1) & (y_true_g == 1))
            fp = np.sum((y_pred == 1) & (y_true_g == 0))
            fn = np.sum((y_pred == 0) & (y_true_g == 1))
            tn = np.sum((y_pred == 0) & (y_true_g == 0))

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            tprs.append(tpr)
            fprs.append(fpr)

        result[group] = {
            'thresholds': thresholds,
            'tprs': tprs,
            'fprs': fprs
        }

    # Find thresholds where TPR and FPR are approximately equal across groups
    # This is a simplification; full solution requires solving optimization
    return result
```

**Practical Fairness Post-Processing Workflow**

1. **Audit current model**: Measure fairness metrics across groups (demographic parity, equalized odds, calibration)
```python
def fairness_audit(y_true, y_pred, sensitive_attr):
    groups = np.unique(sensitive_attr)
    metrics = {}

    for group in groups:
        group_mask = (sensitive_attr == group)
        y_true_g = y_true[group_mask]
        y_pred_g = y_pred[group_mask]

        tn, fp, fn, tp = confusion_matrix(y_true_g, y_pred_g).ravel()

        metrics[group] = {
            'positive_rate': np.mean(y_pred_g),
            'tpr': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0
        }

    return metrics
```

2. **Choose fairness criterion**:
   - Demographic parity: Equal positive rates across groups
   - Equalized odds: Equal TPR and FPR across groups
   - Predictive parity: Equal precision across groups

3. **Apply post-processing**: Use appropriate algorithm for chosen criterion

4. **Trade-off analysis**: Measure fairness-accuracy trade-off
```python
def fairness_accuracy_tradeoff(y_true, y_prob, sensitive_attr, thresholds):
    """
    Analyze how different thresholds affect fairness and accuracy
    """
    results = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        accuracy = np.mean(y_pred == y_true)

        metrics = fairness_audit(y_true, y_pred, sensitive_attr)

        # Compute fairness metrics (e.g., difference in positive rates)
        groups = list(metrics.keys())
        if len(groups) == 2:
            positive_rate_diff = abs(metrics[groups[0]]['positive_rate'] -
                                      metrics[groups[1]]['positive_rate'])
        else:
            positive_rate_diff = 0

        results.append({
            'threshold': t,
            'accuracy': accuracy,
            'fairness_metric': positive_rate_diff
        })

    return results
```

---

## Part B: Decision Criteria Framework

This framework provides structured guidance for selecting ML techniques based on your constraints. Use the tables and decision trees below to prioritize approaches for your specific situation.

### Framework Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Problem Type   │ -> │ Dataset Size     │ -> │ Compute Budget  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        |
                                                        v
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Latency Req.    │ <- │ Interpretability │ <- │ Selected Model  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### 1. Dataset Size Considerations

| Dataset Size | Recommended Techniques | Expected Impact | Secondary Techniques |
|--------------|----------------------|-----------------|---------------------|
| **Small (<1K)** | - Logistic Regression<br>- Regularized Linear Models<br>- Decision Trees (shallow)<br>- Transfer Learning<br>- Data Augmentation | + High interpretability<br>+ Fast training<br>- Limited complexity<br>- Prone to overfitting | - Bayesian Methods<br>- Few-shot Learning<br>- Synthetic Data<br>- Strong Regularization |
| **Medium (1K-100K)** | - Random Forests<br>- Gradient Boosting<br>- SVMs<br>- Neural Nets (1-3 layers)<br>- Feature Engineering | + Good accuracy-complexity balance<br>+ Most algorithms viable<br>+ Cross-validation reliable | - Ensemble Methods<br>- Hyperparameter Tuning<br>- Feature Selection<br>- Calibration |
| **Large (100K-1M)** | - Deep Neural Networks<br>- Gradient Boosting (XGBoost, LightGBM)<br>- Feature Learning<br>- Embedding Layers | + Can model complex patterns<br>+ Deep learning viable<br>+ Regularization effective | - Architecture Search<br>- Distributed Training<br>- Batch Normalization |
| **Very Large (>1M)** | - Deep Learning (CNNs, Transformers)<br>- Distributed Training<br>- Online Learning<br>- Sampling Strategies | + State-of-the-art performance<br>- Significant compute required<br>- Infrastructure critical | - Model Distillation<br>- Pruning/Quantization<br>- Mixed Precision<br>- Incremental Learning |

**Decision Rules:**

- **<1K samples**: Prioritize regularization and transfer learning. Avoid complex models.
- **1K-10K**: Sweet spot for classical ML. Use tree-based models with cross-validation.
- **10K-100K**: Can start using shallow neural networks. Feature engineering still valuable.
- **100K-1M**: Deep learning becomes practical. Gradient boosting remains competitive.
- **>1M**: Deep learning preferred. Consider infrastructure and training time.

### 2. Computational Constraints

| Compute Constraint | Recommended Techniques | Trade-offs | Optimization Tips |
|-------------------|----------------------|------------|-------------------|
| **Limited GPU/CPU** | - Linear/Logistic Regression<br>- Decision Trees<br>- SGD-based training<br>- Feature Selection first<br>- Smaller batch sizes | + Lower memory footprint<br>+ Faster iterations<br>- May sacrifice accuracy | - Use dimensionality reduction<br>- Opt-in features only<br>- Use libraries optimized for CPU (scikit-learn, LightGBM) |
| **Single Machine** | - XGBoost/LightGBM<br>- Random Forests<br>- Medium-sized Neural Nets<br>- Batch training<br>- Mixed precision | + Good for most problems<br>- Limited to models fitting in memory | - Gradient accumulation<br>- Checkpointing<br>- Reduce max_depth for trees<br>- Use DataLoader with workers |
| **Small Cluster** | - Distributed Training<br>- Data Parallelism<br>- Hyperparameter Search<br>- Ensemble Methods | + Can train larger models<br>+ Parallel experiment tracking | - Use Dask/Ray for distribution<br>- Efficient data loading<br>- Gradient compression |
| **Large Cluster** | - Large-scale Deep Learning<br>- Model Parallelism<br>- Architecture Search<br>- Massive Ensembles | + SOTA performance possible<br>- High engineering overhead<br>- Expensive | - Use distributed frameworks (Horovod, DeepSpeed)<br>- Optimize communication<br>- Consider data locality |

### 3. Latency Requirements

| Latency Requirement | Suitable Models | Inference Optimization | Trade-offs |
|--------------------|----------------|----------------------|------------|
| **Real-time (<10ms)** | - Logistic/Linear Regression<br>- Small Decision Trees<br>- Quantized Neural Nets<br>- Tree ensembles (max depth ~10)<br>- FANN/FAISS for retrieval | - Model quantization (INT8)<br>- Pruning<br>- ONNX/TFLite export<br>- Batching at edge | - Limited model capacity<br>- May sacrifice accuracy<br>- Hardware dependency |
| **Near Real-time (<100ms)** | - Gradient Boosting<br>- Medium Neural Nets<br>- SVM (linear kernel)<br>- Random Forests (limited trees) | - Model compilation<br>- Caching predictions<br>- Async inference<br>- Batch processing | - Good accuracy-latency balance<br>- Most business requirements met |
| **Interactive (<1s)** | - Deep Learning (medium)<br>- Large Gradient Boosting<br>- Feature-rich models<br>- Some ensembles | - GPU acceleration<br>- Model parallelism<br>- Lazy evaluation<br>- Result caching | - Viable for many web apps<br>- User experience still good |
| **Batch (no constraint)** | - Any model<br>- Large Ensembles<br>- Deep Learning<br>- Complex pipelines | - Pre-computation<br>- Materialized views<br>- Incremental updates | - Maximum accuracy focus<br>- No latency constraints |

**Inference Optimization Techniques:**

```python
# Model quantization for faster inference
import torch.quantization as quantization

def quantize_model(model):
    # Post-training static quantization
    model.qconfig = quantization.get_default_qconfig('fbgemm')
    quantization.prepare(model, inplace=True)
    # Calibrate with representative data...
    quantization.convert(model, inplace=True)
    return model

# ONNX export for deployment
import torch.onnx

def export_to_onnx(model, dummy_input, onnx_path):
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}}
    )
```

### 4. Interpretability Needs

| Interpretability Level | Recommended Models | Explanation Methods | Use Cases |
|-----------------------|------------------|-------------------|-----------|
| **Fully Interpretable** | - Linear/Logistic Regression<br>- Decision Trees<br>- Rule-based Systems<br>- Generalized Additive Models (GAMs) | - Coefficient inspection<br>- Tree visualization<br>- Rule extraction | - Regulated industries<br>- Medical diagnosis<br>- Credit scoring<br>- Scientific discovery |
| **Partially Interpretable** | - Random Forests<br>- Gradient Boosting<br>- Shallow Neural Nets | - Feature Importance<br>- SHAP values<br>- Partial Dependence<br>- LIME | - Business intelligence<br>- Model debugging<br>- Stakeholder communication |
| **Post-hoc Explanations** | - Deep Learning<br>- Large Ensembles<br>- Complex Models | - SHAP/LIME<br>- Attention visualization<br>- Saliency maps<br>- Counterfactuals | - R&D applications<br>- Black-box acceptable<br>- Performance-critical |
| **Not Required** | - Any model<br>- Focus on accuracy | - None needed | - Ranking/recommendation<br>- Internal tools<br>- Benchmarks |

**Interpretability Techniques:**

```python
# SHAP values for model explanation
import shap

def explain_with_shap(model, X_train, X_test):
    # Create explainer
    explainer = shap.TreeExplainer(model) if hasattr(model, 'feature_importances_') else shap.KernelExplainer(model.predict, X_train)

    # Get SHAP values
    shap_values = explainer.shap_values(X_test)

    # Visualize
    shap.summary_plot(shap_values, X_test, plot_type="bar")
    shap.summary_plot(shap_values, X_test)

    return shap_values

# Partial dependence plots
from sklearn.inspection import PartialDependenceDisplay

def plot_partial_dependence(model, X, features):
    PartialDependenceDisplay.from_estimator(
        model, X, features,
        n_jobs=-1, grid_resolution=50
    )
```

### 5. Problem Type Decision Matrix

| Problem Type | Best Techniques | Evaluation Metrics | Special Considerations |
|--------------|----------------|-------------------|----------------------|
| **Binary Classification** | - Logistic Regression<br>- Tree Ensembles<br>- Neural Networks | - AUC-ROC<br>- Precision/Recall<br>- F1<br>- Calibration | - Class imbalance handling<br>- Threshold selection<br>- Cost-sensitive learning |
| **Multi-class Classification** | - Random Forest<br>- XGBoost<br>- Neural Networks (Softmax) | - Accuracy<br>- Top-K Accuracy<br>- Macro/Micro F1<br>- Confusion Matrix | - Class imbalance (weighted metrics)<br>- Hierarchical labels<br>- Error analysis |
| **Regression** | - Linear Regression<br>- Gradient Boosting<br>- Neural Networks | - RMSE/MAE<br>- R²<br>- MAPE<br>- Residual Analysis | - Heteroscedasticity<br>- Outlier handling<br>- Prediction intervals |
| **Ranking/Recommendation** | - Learning to Rank<br>- Matrix Factorization<br>- Neural Collaborative Filtering | - NDCG<br>- MAP<br>- MRR<br>- Precision@K | - Position bias<br>- Cold start<br>- Diversity |
| **Time Series** | - ARIMA/Prophet<br>- RNN/LSTM/Transformers<br>- Tree-based with lag features | - RMSE on horizon<br>- MAPE<br>- SMAPE<br>- Forecast skill | - Temporal CV<br>- Look-ahead bias<br>- Seasonality |
| **Anomaly Detection** | - Isolation Forest<br>- Autoencoders<br>- One-Class SVM | - Precision@K<br>- Recall@K<br>- AUC (PR curve) | - Highly imbalanced<br>- Novelty vs. outlier<br>- Threshold selection |
| **Structured Prediction** | - CRFs<br>- Structured SVM<br>- Seq2Seq | - Token-level accuracy<br>- BLEU/ROUGE<br>- Exact match | - Sequence length<br>- Beam search<br>- Label consistency |

### Comprehensive Decision Flowchart

```
START
  |
  v
What is your problem type?
  |
  +-- Classification --> Is data >100K?
  |                        |
  |                        +-- Yes --> Deep Learning / Gradient Boosting
  |                        |           |
  |                        |           v
  |                        |          Need real-time (<10ms)?
  |                        |           |
  |                        |           +-- Yes --> Quantized NN / LightGBM
  |                        |           +-- No --> Full model OK
  |                        |
  |                        +-- No (<100K) --> Is data <1K?
  |                                          |
  |                                          +-- Yes --> Logistic Regression + Transfer Learning
  |                                          +-- No --> Random Forest / XGBoost
  |
  +-- Regression --> Is interpretability critical?
  |                    |
  |                    +-- Yes --> Linear Regression / GAM
  |                    +-- No --> Gradient Boosting / Neural Net
  |
  +-- Time Series --> Is forecast horizon short (<7 days)?
  |                    |
  |                    +-- Yes --> ARIMA / Prophet
  |                    +-- No --> LSTM / Transformer / Tree-based with lags
  |
  +-- Ranking --> Is cold start a problem?
                      |
                      +-- Yes --> Content-based / Matrix Factorization
                      +-- No --> Collaborative Filtering / Learning to Rank
```

### Quick Reference: Technique Selection by Constraint Combination

| Scenario | Dataset | Compute | Latency | Interpretability | Recommended Technique |
|----------|---------|---------|---------|------------------|---------------------|
| **Fraud Detection** | Medium | Limited | <100ms | Partial | XGBoost + SHAP |
| **Medical Diagnosis** | Small | Limited | <1s | Full | Logistic Regression / Small Tree |
| **Real-time Bidding** | Large | Cluster | <10ms | Post-hoc | Quantized NN / LightGBM |
| **Customer Churn** | Medium | Single | Batch | Partial | Random Forest / Neural Net |
| **Image Classification** | Large | Cluster | <1s | Post-hoc | ResNet / EfficientNet |
| **NLP Classification** | Large | Cluster | <100ms | Post-hoc | BERT / RoBERTa |
| **Credit Scoring** | Large | Single | <1s | Full | Logistic Regression + Feature Engineering |
| **Recommendation** | Large | Cluster | <100ms | Partial | Collaborative Filtering + Neural |
| **Demand Forecasting** | Medium | Single | Batch | Partial | Prophet / XGBoost with lags |
| **Anomaly Detection** | Medium | Single | <100ms | Partial | Isolation Forest / Autoencoder |

---

## Part C: Common Pitfalls & Diagnostics Catalogue

This section covers the most common ML pitfalls, how to detect them, and how to fix them. Each pitfall includes specific diagnostic procedures and code examples.

### Pitfall 1: Data Leakage

**Description**: Data leakage occurs when information from outside the training dataset is used to create the model. This leads to overly optimistic performance estimates that don't generalize to production.

**Common Types**:
- **Target leakage**: Using features that contain information about the target
- **Train-test contamination**: Preprocessing applied before data splitting
- **Temporal leakage**: Using future data to predict past events

**Detection**:
```python
# 1. Check for target leakage through feature correlation
def detect_target_leakage(X, y, threshold=0.95):
    """
    Identify features highly correlated with target
    """
    correlations = X.corrwith(y).abs()
    suspicious = correlations[correlations > threshold]
    return suspicious

# 2. Check for temporal leakage in time series
def detect_temporal_leakage(df, timestamp_col, target_col, feature_cols):
    """
    Check if features contain future information
    """
    df_sorted = df.sort_values(timestamp_col)

    for feature in feature_cols:
        # Compute cross-correlation with future target
        for lag in [1, 2, 5, 10]:
            df_sorted[f'{feature}_lead_{lag}'] = df_sorted[target_col].shift(-lag)
            correlation = df_sorted[feature].corr(df_sorted[f'{feature}_lead_{lag}'])

            if abs(correlation) > 0.7:
                print(f"WARNING: {feature} has high correlation ({correlation:.2f}) "
                      f"with target {lag} steps ahead")

# 3. Check train-test information sharing
def check_preprocessing_leakage(pipeline, X_train, X_test):
    """
    Verify preprocessing doesn't share statistics across train/test
    """
    # Fit on training data only
    pipeline.fit(X_train)

    # Get preprocessing statistics
    if hasattr(pipeline.named_steps['preprocess'], 'mean_'):
        train_mean = pipeline.named_steps['preprocess'].mean_

        # Check if test statistics match training (indicates leakage)
        X_test_scaled = pipeline.transform(X_test)
        test_mean = X_test_scaled.mean(axis=0)

        if np.allclose(train_mean, test_mean, rtol=0.01):
            print("WARNING: Test data statistics suspiciously close to training")

    return pipeline
```

**Fix**:
```python
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Correct approach: Split first, then preprocess
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Use pipeline to prevent leakage
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', YourModel())
])

pipeline.fit(X_train, y_train)
score = pipeline.score(X_test, y_test)

# For time series, always split by time
def temporal_split(df, test_start_date):
    train = df[df['date'] < test_start_date]
    test = df[df['date'] >= test_start_date]
    return train, test
```

### Pitfall 2: Look-Ahead Bias in Time Series

**Description**: Using future information to train models that will be applied to historical data. Common in feature engineering for time series.

**Detection**:
```python
def detect_lookahead_bias(df, feature_cols, target_col, time_col):
    """
    Test if features have predictive power from future to past
    """
    df = df.sort_values(time_col)
    issues = []

    for feature in feature_cols:
        # Calculate correlation of feature at time t with target at time t+1
        future_correlation = df[feature].corr(df[target_col].shift(-1))

        # Calculate correlation of feature at time t with target at time t-1
        past_correlation = df[feature].corr(df[target_col].shift(1))

        if abs(future_correlation) > abs(past_correlation) * 1.5:
            issues.append({
                'feature': feature,
                'future_corr': future_correlation,
                'past_corr': past_correlation,
                'issue': 'Feature has stronger correlation with future target'
            })

    return issues
```

**Fix**:
```python
# Correct feature engineering for time series
def create_lag_features(df, feature_cols, lags=[1, 2, 3, 7]):
    """
    Create lag features without look-ahead bias
    """
    for feature in feature_cols:
        for lag in lags:
            df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)

    # Rolling statistics exclude current value
    for feature in feature_cols:
        df[f'{feature}_rolling_mean_7'] = df[feature].shift(1).rolling(7).mean()
        df[f'{feature}_rolling_std_7'] = df[feature].shift(1).rolling(7).std()

    return df.dropna()

# Always use time-based cross-validation
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X):
    # Test indices always come after train indices
    X_train, X_test = X[train_idx], X[test_idx]
```

### Pitfall 3: Overfitting to Validation Set

**Description**: Repeatedly tuning hyperparameters on the same validation set causes information leakage and overfitting to the validation set.

**Detection**:
```python
def detect_validation_overfitting(cv_scores, test_scores, threshold=0.05):
    """
    Compare CV scores with hold-out test scores
    """
    cv_mean = np.mean(cv_scores)
    test_score = np.mean(test_scores)

    gap = cv_mean - test_score

    if gap > threshold:
        print(f"WARNING: Large gap between CV ({cv_mean:.4f}) and test ({test_score:.4f})")
        print("This may indicate validation set overfitting")

        # Suggest solutions
        print("\nSolutions:")
        print("1. Use nested cross-validation for hyperparameter tuning")
        print("2. Reduce number of tuning iterations")
        print("3. Keep a true hold-out set that's never used for tuning")

    return gap
```

**Fix**:
```python
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier

# Nested cross-validation
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)

# Outer loop: estimate generalization
outer_scores = []
for train_idx, test_idx in outer_cv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Inner loop: hyperparameter tuning
    param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, None]}
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=inner_cv
    )
    grid_search.fit(X_train, y_train)

    # Evaluate on outer test fold
    outer_score = grid_search.score(X_test, y_test)
    outer_scores.append(outer_score)

print(f"Unbiased estimate: {np.mean(outer_scores):.4f} ± {np.std(outer_scores):.4f}")
```

### Pitfall 4: Distribution Shift Between Train and Production

**Description**: The data distribution in production differs from training data, causing performance degradation.

**Detection**:
```python
from scipy.stats import ks_2samp

def detect_distribution_shift(X_train, X_production, feature_names, threshold=0.05):
    """
    Use Kolmogorov-Smirnov test to detect feature distribution shift
    """
    shifts = []

    for i, feature in enumerate(feature_names):
        train_feature = X_train[:, i]
        prod_feature = X_production[:, i]

        # KS test for distribution difference
        statistic, p_value = ks_2samp(train_feature, prod_feature)

        if p_value < threshold:
            shifts.append({
                'feature': feature,
                'ks_statistic': statistic,
                'p_value': p_value,
                'shift_detected': True
            })

            # Print summary statistics
            print(f"Feature: {feature}")
            print(f"  Train: mean={train_feature.mean():.2f}, std={train_feature.std():.2f}")
            print(f"  Prod:  mean={prod_feature.mean():.2f}, std={prod_feature.std():.2f}")
            print(f"  KS test: statistic={statistic:.4f}, p={p_value:.6f}")

    return shifts

# For target distribution shift
def detect_target_shift(y_train, y_production):
    """
    Detect shift in target distribution
    """
    train_dist = np.bincount(y_train) / len(y_train)
    prod_dist = np.bincount(y_production) / len(y_production)

    kl_divergence = np.sum(train_dist * np.log(train_dist / (prod_dist + 1e-10)))

    print(f"Class distribution train: {train_dist}")
    print(f"Class distribution production: {prod_dist}")
    print(f"KL divergence: {kl_divergence:.4f}")

    return kl_divergence
```

**Fix**:
```python
# 1. Importance weighting for covariate shift
def estimate_importance_weights(X_train, X_production):
    """
    Estimate importance weights to account for covariate shift
    """
    from sklearn.ensemble import RandomForestClassifier

    # Train classifier to distinguish train from production
    y_source = np.concatenate([np.zeros(len(X_train)), np.ones(len(X_production))])
    X_combined = np.vstack([X_train, X_production])

    classifier = RandomForestClassifier(random_state=42)
    classifier.fit(X_combined, y_source)

    # Get probabilities
    train_probs = classifier.predict_proba(X_train)[:, 1]
    prod_probs = classifier.predict_proba(X_production)[:, 1]

    # Compute weights
    train_weights = prod_probs.mean() / (train_probs + 1e-10)
    train_weights = np.clip(train_weights, 0, 10)  # Clip extreme weights

    return train_weights

# 2. Periodic retraining schedule
class PeriodicRetrainer:
    def __init__(self, model, retrain_threshold=0.05):
        self.model = model
        self.retrain_threshold = retrain_threshold
        self.baseline_performance = None

    def check_and_retrain(self, X_new, y_new):
        current_performance = self.model.score(X_new, y_new)

        if self.baseline_performance is None:
            self.baseline_performance = current_performance
            return False

        performance_drop = self.baseline_performance - current_performance

        if performance_drop > self.retrain_threshold:
            print(f"Performance dropped by {performance_drop:.4f}. Retraining...")
            # Retrain on accumulated data
            self.model.fit(X_new, y_new)
            self.baseline_performance = current_performance
            return True

        return False
```

### Pitfall 5: Feature Importance Instability

**Description**: Feature importance rankings vary significantly across different random seeds or data samples, indicating unreliable importance measures.

**Detection**:
```python
def detect_importance_instability(model, X, y, n_iterations=10):
    """
    Test feature importance stability across multiple random seeds
    """
    importances = []

    for seed in range(n_iterations):
        # Train with different random state
        if hasattr(model, 'random_state'):
            model.set_params(random_state=seed)

        model.fit(X, y)

        # Get feature importances
        if hasattr(model, 'feature_importances_'):
            imp = model.feature_importances_
        else:
            # Use permutation importance as fallback
            from sklearn.inspection import permutation_importance
            result = permutation_importance(model, X, y, n_repeats=5, random_state=seed)
            imp = result['importances_mean']

        importances.append(imp)

    importances = np.array(importances)

    # Calculate coefficient of variation for each feature
    cv = importances.std(axis=0) / (importances.mean(axis=0) + 1e-10)

    unstable_features = np.where(cv > 0.5)[0]

    print("Feature importance stability:")
    for i in range(len(cv)):
        status = "UNSTABLE" if i in unstable_features else "stable"
        print(f"  Feature {i}: CV={cv[i]:.2f} ({status})")

    return cv, unstable_features
```

**Fix**:
```python
# 1. Use permutation importance instead of built-in importance
from sklearn.inspection import permutation_importance

def get_stable_importance(model, X, y, n_repeats=10):
    """
    Get permutation importance with multiple repetitions
    """
    result = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        random_state=42
    )

    # Sort features by mean importance
    sorted_idx = result.importances_mean.argsort()[::-1]

    print("Feature ranking:")
    for i, idx in enumerate(sorted_idx):
        print(f"{i+1}. Feature {idx}: "
              f"{result.importances_mean[idx]:.4f} "
              f"(±{result.importances_std[idx]:.4f})")

    return result

# 2. Use stability selection
from sklearn.utils import resample

def stability_selection(model, X, y, n_iterations=100, threshold=0.6):
    """
    Select features that appear consistently across subsamples
    """
    n_features = X.shape[1]
    selection_counts = np.zeros(n_features)

    for _ in range(n_iterations):
        X_sample, y_sample = resample(X, y, random_state=_)
        model.fit(X_sample, y_sample)

        if hasattr(model, 'feature_importances_'):
            # Select top features
            importances = model.feature_importances_
            top_features = np.where(importances > np.percentile(importances, 75))[0]
            selection_counts[top_features] += 1

    # Select features selected in at least threshold% of iterations
    stable_features = np.where(selection_counts / n_iterations >= threshold)[0]

    print(f"Stable features ({len(stable_features)}): {stable_features}")
    print(f"Selection frequencies: {selection_counts[stable_features] / n_iterations}")

    return stable_features
```

### Pitfall 6: Improper Cross-Validation for Grouped/Sequential Data

**Description**: Using standard K-fold CV on data with groups (e.g., multiple samples per user) or temporal structure causes optimistic bias.

**Detection**:
```python
def detect_cv_leakage(y_true, y_pred, group_ids):
    """
    Check if predictions from same group are in both train and validation
    """
    # Check for information leakage across folds
    unique_groups = np.unique(group_ids)

    leakage_detected = False
    for group in unique_groups:
        group_indices = np.where(group_ids == group)[0]

        # If group has both correct and incorrect predictions, may be OK
        # If group has all same predictions, might be memorization
        if len(group_indices) > 5:
            pred_variance = np.var(y_pred[group_indices])
            if pred_variance < 0.01:
                print(f"WARNING: Group {group} has nearly identical predictions")
                leakage_detected = True

    return leakage_detected
```

**Fix**:
```python
# 1. Group-based cross-validation
from sklearn.model_selection import GroupKFold

def grouped_cross_validation(model, X, y, groups, n_splits=5):
    """
    Cross-validation ensuring no group appears in both train and validation
    """
    group_kfold = GroupKFold(n_splits=n_splits)
    scores = []

    for train_idx, val_idx in group_kfold.split(X, y, groups):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Check groups don't overlap
        train_groups = set(groups[train_idx])
        val_groups = set(groups[val_idx])
        assert len(train_groups & val_groups) == 0, "Groups overlap!"

        model.fit(X_train, y_train)
        score = model.score(X_val, y_val)
        scores.append(score)

    return np.array(scores)

# 2. Time-series cross-validation
from sklearn.model_selection import TimeSeriesSplit

def time_series_cross_validation(model, X, y, n_splits=5):
    """
    Cross-validation respecting temporal order
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []

    for train_idx, val_idx in tscv.split(X):
        # Verify temporal ordering
        assert max(train_idx) < min(val_idx), "Temporal ordering violated!"

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model.fit(X_train, y_train)
        score = model.score(X_val, y_val)
        scores.append(score)

    return np.array(scores)
```

### Pitfall 7: Ignoring Class Imbalance in Evaluation

**Description**: Using accuracy on imbalanced datasets gives misleading results. A model that predicts the majority class for all samples can have high accuracy.

**Detection**:
```python
def detect_imbalance_issues(y_true, y_pred):
    """
    Check if evaluation metrics are misleading due to imbalance
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    # Check class distribution
    class_counts = np.bincount(y_true)
    imbalance_ratio = max(class_counts) / min(class_counts)

    if imbalance_ratio > 3:
        print(f"WARNING: Dataset is imbalanced (ratio: {imbalance_ratio:.2f}:1)")
        print("Accuracy alone is not a reliable metric")

    # Calculate multiple metrics
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"\nMetrics:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")

    # If accuracy >> F1, model may be majority-class guessing
    if acc - f1 > 0.2:
        print("\nWARNING: Accuracy significantly higher than F1.")
        print("Model may be predicting majority class.")

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'imbalance_ratio': imbalance_ratio
    }
```

**Fix**:
```python
# 1. Use appropriate metrics
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_imbalanced(y_true, y_pred, class_names=None):
    """
    Comprehensive evaluation for imbalanced datasets
    """
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    # Per-class metrics
    for i in range(cm.shape[0]):
        if cm[i].sum() > 0:
            precision = cm[i, i] / cm[:, i].sum()
            recall = cm[i, i] / cm[i].sum()
            f1 = 2 * (precision * recall) / (precision + recall)
            print(f"Class {i}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")

# 2. Use stratified sampling
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,  # Preserve class distribution
    random_state=42
)

# 3. Use appropriate sampling techniques
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# Create balanced dataset
resampling_pipeline = Pipeline([
    ('oversample', SMOTE(random_state=42)),
    ('undersample', RandomUnderSampler(random_state=42))
])

X_resampled, y_resampled = resampling_pipeline.fit_resample(X_train, y_train)
```

### Pitfall 8: Metric Gaming (Goodhart's Law)

**Description**: When a metric becomes a target, it ceases to be a good measure. Optimizing for a single metric often leads to unintended behavior.

**Detection**:
```python
def detect_metric_gaming(model, X, y, metrics):
    """
    Check if optimization on one metric hurts others
    """
    scores = {}
    for name, metric_func in metrics.items():
        y_pred = model.predict(X)
        score = metric_func(y, y_pred)
        scores[name] = score

    # Check for suspicious patterns
    if 'accuracy' in scores and 'f1' in scores:
        if scores['accuracy'] - scores['f1'] > 0.3:
            print("WARNING: Large accuracy-F1 gap suggests metric gaming")

    # Check calibration vs accuracy trade-off
    if 'accuracy' in scores and 'brier_score' in scores:
        if scores['accuracy'] > 0.9 and scores['brier_score'] > 0.3:
            print("WARNING: High accuracy but poor calibration")
            print("Model may be overconfident")

    return scores
```

**Fix**:
```python
# 1. Use multi-metric optimization
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate

def multi_metric_evaluation(model, X, y, cv=5):
    """
    Evaluate model across multiple metrics simultaneously
    """
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='weighted'),
        'recall': make_scorer(recall_score, average='weighted'),
        'f1': make_scorer(f1_score, average='weighted'),
        'roc_auc': make_scorer(roc_auc_score, needs_proba=True, multi_class='ovr')
    }

    results = cross_validate(model, X, y, cv=cv, scoring=scoring)

    print("Cross-validation results:")
    for metric in scoring.keys():
        scores = results[f'test_{metric}']
        print(f"  {metric}: {scores.mean():.4f} (±{scores.std():.4f})")

    # Check for balanced performance
    metric_means = [results[f'test_{m}'].mean() for m in scoring.keys()]
    metric_std = np.std(metric_means)

    if metric_std > 0.1:
        print(f"\nWARNING: High variance across metrics (std={metric_std:.4f})")
        print("Consider multi-objective optimization")

    return results

# 2. Custom composite metric
def business_metric(y_true, y_pred, fp_cost=1, fn_cost=10, tp_value=5):
    """
    Custom metric aligned with business objectives
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    total_value = tp * tp_value - fp * fp_cost - fn * fn_cost
    max_value = len(y_true) * tp_value

    return total_value / max_value
```

### Pitfall 9: Sampling Bias in Training Data

**Description**: The training data is not representative of the true population, leading to biased models.

**Detection**:
```python
def detect_sampling_bias(X_train, X_population, feature_names):
    """
    Compare training data distribution to known population
    """
    from scipy.stats import ks_2samp, chi2_contingency

    biases = []

    for i, feature in enumerate(feature_names):
        train_feature = X_train[:, i]
        pop_feature = X_population[:, i]

        # KS test for continuous features
        stat, p_value = ks_2samp(train_feature, pop_feature)

        if p_value < 0.05:
            biases.append({
                'feature': feature,
                'test': 'KS',
                'statistic': stat,
                'p_value': p_value,
                'train_mean': train_feature.mean(),
                'pop_mean': pop_feature.mean()
            })

    if biases:
        print("Sampling bias detected in features:")
        for bias in biases:
            print(f"  {bias['feature']}: p={bias['p_value']:.6f}")
            print(f"    Train mean: {bias['train_mean']:.2f}")
            print(f"    Pop mean: {bias['pop_mean']:.2f}")

    return biases
```

**Fix**:
```python
# 1. Importance weighting
def compute_importance_weights(X_train, X_target):
    """
    Compute sample weights to account for sampling bias
    """
    from sklearn.ensemble import RandomForestClassifier

    # Create binary classification problem
    n_train = len(X_train)
    n_target = len(X_target)

    y_source = np.array([0] * n_train + [1] * n_target)
    X_combined = np.vstack([X_train, X_target])

    # Train classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_combined, y_source)

    # Get propensity scores
    propensity = clf.predict_proba(X_train[:n_train])[:, 1]

    # Compute importance weights
    weights = (n_target / n_train) * (1 - propensity) / (propensity + 1e-10)
    weights = np.clip(weights, 0, 10)  # Clip extreme weights

    return weights

# 2. Stratified sampling
from sklearn.model_selection import train_test_split

def stratified_sampling(X, y, strata, test_size=0.2):
    """
    Sample to preserve distribution across strata
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=strata,
        random_state=42
    )

    return X_train, X_test, y_train, y_test
```

### Pitfall 10: Confounding Variables

**Description**: A variable that influences both the independent variable and dependent variable, creating a spurious association.

**Detection**:
```python
def detect_confounders(df, treatment, outcome, covariates, threshold=0.3):
    """
    Detect potential confounding variables
    """
    from scipy.stats import pearsonr

    confounders = []

    for covariate in covariates:
        # Check association with treatment
        corr_treatment, p_treatment = pearsonr(df[covariate], df[treatment])

        # Check association with outcome
        corr_outcome, p_outcome = pearsonr(df[covariate], df[outcome])

        # If significantly associated with both, potential confounder
        if (abs(corr_treatment) > threshold and p_treatment < 0.05 and
            abs(corr_outcome) > threshold and p_outcome < 0.05):

            confounders.append({
                'covariate': covariate,
                'corr_with_treatment': corr_treatment,
                'corr_with_outcome': corr_outcome,
                'p_treatment': p_treatment,
                'p_outcome': p_outcome
            })

    if confounders:
        print("Potential confounders detected:")
        for conf in confounders:
            print(f"  {conf['covariate']}:")
            print(f"    Corr with treatment: {conf['corr_with_treatment']:.3f}")
            print(f"    Corr with outcome: {conf['corr_with_outcome']:.3f}")

    return confounders
```

**Fix**:
```python
# 1. Propensity score matching
from sklearn.linear_model import LogisticRegression

def propensity_score_matching(df, treatment, covariates):
    """
    Match treated and control units with similar propensity scores
    """
    # Estimate propensity scores
    X = df[covariates].values
    y = df[treatment].values

    ps_model = LogisticRegression(random_state=42)
    ps_model.fit(X, y)
    propensity_scores = ps_model.predict_proba(X)[:, 1]

    df['propensity_score'] = propensity_scores

    # Match treated and control units
    treated = df[df[treatment] == 1]
    control = df[df[treatment] == 0]

    matched_control = []
    for _, treated_unit in treated.iterrows():
        # Find control unit with closest propensity score
        distances = np.abs(control['propensity_score'] - treated_unit['propensity_score'])
        closest_idx = distances.idxmin()
        matched_control.append(control.loc[closest_idx])

    matched_control = pd.DataFrame(matched_control)
    matched_df = pd.concat([treated, matched_control])

    return matched_df

# 2. Include confounders in model
def adjust_for_confounders(model, X, y, confounder_indices):
    """
    Ensure model explicitly adjusts for confounders
    """
    # Include confounders in feature set
    X_with_confounders = X

    # Fit model
    model.fit(X_with_confounders, y)

    return model
```

### Summary: Diagnostic Checklist

Use this checklist before deploying any model:

- [ ] **Data integrity**: No target leakage, no temporal leakage, proper train/val/test split
- [ ] **Cross-validation**: Appropriate CV strategy for data structure (grouped, temporal)
- [ ] **Distribution alignment**: Train distribution matches expected production distribution
- [ ] **Class balance**: Appropriate metrics used, sampling strategies applied if needed
- [ ] **Evaluation quality**: Multiple metrics checked, no metric gaming detected
- [ ] **Feature stability**: Feature importances stable across random seeds
- [ ] **Calibration**: Probabilities calibrated if used for decision-making
- [ ] **Fairness**: Model audited for bias across protected groups
- [ ] **Robustness**: Model tested on edge cases and out-of-distribution samples
- [ ] **Reproducibility**: Random seeds set, code versioned, experiments logged

---

## Key Takeaways

1. **Post-processing matters**: Calibration, threshold optimization, and fairness-aware post-processing can significantly improve real-world model performance without retraining.

2. **Context-driven decisions**: There's no universal "best" technique. Use the decision framework to select approaches based on your dataset size, computational constraints, latency requirements, interpretability needs, and problem type.

3. **Pitfalls are common**: Even experienced practitioners fall prey to data leakage, distribution shift, and improper evaluation. Implement systematic checks and diagnostics.

4. **Test in production-like conditions**: Always validate models using data and conditions that match your production environment as closely as possible.

5. **Iterate and monitor**: ML systems require ongoing monitoring and updating as data distributions shift and business requirements evolve.

---

## Further Reading

- Guo et al. (2017). "On Calibration of Modern Neural Networks." ICML.
- Hardt et al. (2016). "Equality of Opportunity in Supervised Learning." NeurIPS.
- Vovk et al. (2005). "Algorithmic Learning in a Random World." (Conformal Prediction)
- Kull et al. (2017). "Beta Calibration: A Well-Founded and Easily Implemented Improvement to Logistic Calibration for Binary Classifiers." AISTATS.
- Pleiss et al. (2017). "On Fairness and Calibration." NeurIPS.
- scikit-learn documentation: https://scikit-learn.org/stable/
- Imbalanced-learn documentation: https://imbalanced-learn.org/
- MAPIE documentation: https://mapie.readthedocs.io/
