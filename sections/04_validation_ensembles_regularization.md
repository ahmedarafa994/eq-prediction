# 4. Validation, Ensembles & Regularization

This section covers robust validation strategies, ensemble methods, and regularization techniques that form the foundation of reliable machine learning systems. Proper validation ensures unbiased performance estimates, ensembles combine models for improved generalization, and regularization controls model complexity to prevent overfitting.

## Part A: Robust Validation & Evaluation

### 4.1 Cross-Validation Strategies

#### 4.1.1 K-Fold Cross-Validation (HIGH-IMPACT)

K-fold cross-validation is the workhorse of model evaluation. The dataset is partitioned into k equal folds, with each fold serving once as validation while the remaining k-1 folds form the training set.

**When to use:** Most standard ML problems with i.i.d. data

**Choosing k:**
- k=5: Good balance between computation and reliability
- k=10: Lower bias, more reliable estimates
- k=n (LOO): High variance, computationally expensive, rarely recommended

```python
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np

X, y = load_data()  # Your data
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Basic 5-fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
print(f"Accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
```

**Common Pitfall:** Not shuffling data when samples are ordered by class or time. Always use `shuffle=True` for i.i.d. data.

#### 4.1.2 Stratified K-Fold (HIGH-IMPACT)

Stratified k-fold preserves class distribution across folds—critical for imbalanced datasets where standard k-fold might create validation folds with no positive samples.

```python
from sklearn.model_selection import StratifiedKFold

# For imbalanced classification (e.g., 95% negative, 5% positive)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf, scoring='f1')

# Check stratification worked
for train_idx, test_idx in skf.split(X, y):
    print(f"Train class ratio: {np.bincount(y[train_idx]) / len(train_idx)}")
    print(f"Test class ratio: {np.bincount(y[test_idx]) / len(test_idx)}")
```

**Diagnosis:** If you see `undefined metric` warnings for ROC-AUC or F1, your validation folds lack positive classes—switch to stratified immediately.

#### 4.1.3 Group K-Fold (HIGH-IMPACT)

Group k-fold prevents data leakage when samples from the same group (patient, user, session) appear in both train and validation. Essential for:

- Medical data: Multiple samples per patient
- User data: Multiple transactions per user
- Sensor data: Multiple readings from same device

```python
from sklearn.model_selection import GroupKFold
import pandas as pd

# Sample data with user_id column
df = pd.DataFrame({'features': X, 'target': y, 'user_id': user_ids})
groups = df['user_id'].values

gkf = GroupKFold(n_splits=5)
for train_idx, test_idx in gkf.split(X, y, groups):
    train_groups = set(groups[train_idx])
    test_groups = set(groups[test_idx])
    assert len(train_groups & test_groups) == 0  # Verify no leakage
```

**Common Pitfall:** Using standard k-fold on grouped data yields overconfident estimates—models learn group-specific patterns that don't generalize.

#### 4.1.4 Time Series Cross-Validation (HIGH-IMPACT)

Time series data requires temporal validation—never randomly shuffle. Use forward chaining where each validation fold comes after its training fold.

```python
from sklearn.model_selection import TimeSeriesSplit

# For temporal data (stock prices, sensor readings, etc.)
tscv = TimeSeriesSplit(n_splits=5, test_size=100)  # Fixed test window
for train_idx, test_idx in tscv.split(X):
    assert test_idx[0] > train_idx[-1]  # Verify temporal order

# Alternative: Expanding window
tscv_expanding = TimeSeriesSplit(n_splits=5, max_train_size=None)

# Alternative: Sliding window (fixed-size training)
tscv_sliding = TimeSeriesSplit(n_splits=5, max_train_size=1000)
```

**Common Pitfall:** Standard k-fold on time series creates look-ahead bias—model learns from future to predict past.

#### 4.1.5 Nested Cross-Validation (HIGH-IMPACT)

Nested CV provides unbiased performance estimates when tuning hyperparameters. Outer loop estimates performance; inner loop selects hyperparameters.

```python
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.svm import SVC

# Outer CV: Performance estimation
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Inner CV: Hyperparameter tuning
inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)

param_grid = {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 0.01, 0.1]}

# Nested CV
outer_scores = []
for train_idx, test_idx in outer_cv.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Inner loop: find best hyperparameters
    grid_search = GridSearchCV(SVC(), param_grid, cv=inner_cv, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Outer loop: evaluate with best params
    best_model = grid_search.best_estimator_
    outer_scores.append(best_model.score(X_test, y_test))

print(f"Nested CV Score: {np.mean(outer_scores):.3f} (+/- {np.std(outer_scores):.3f})")
```

**When to use:**
- Publishing research results
- Limited data (<10k samples)
- Need unbiased hyperparameter selection + performance estimate

**Computational cost:** O(k_outer × k_inner × n_hyperparameters)

#### 4.1.6 Repeated Cross-Validation (INCREMENTAL)

Repeated k-fold reduces variance of performance estimates by repeating k-fold with different random splits.

```python
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold

# Repeat 5-fold CV 3 times = 15 total fits
rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

scores = cross_val_score(model, X, y, cv=rskf, scoring='accuracy')
print(f"Mean: {scores.mean():.3f}, Std: {scores.std():.3f}")
```

**When to use:** Reporting results with confidence intervals where you need stable estimates.

### 4.2 Metric Selection

#### 4.2.1 Classification Metrics (HIGH-IMPACT)

Accuracy is misleading for imbalanced data. Choose metrics aligned with your problem:

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)

y_true, y_pred, y_proba = get_predictions()

# For balanced classes
acc = accuracy_score(y_true, y_pred)

# For imbalanced classes
precision = precision_score(y_true, y_pred)  # TP / (TP + FP)
recall = recall_score(y_true, y_pred)        # TP / (TP + FN)
f1 = f1_score(y_true, y_pred)                # Harmonic mean

# For probability predictions
roc_auc = roc_auc_score(y_true, y_proba)           # Threshold-independent
pr_auc = average_precision_score(y_true, y_proba)  # Better for imbalanced

# Full diagnostic report
print(classification_report(y_true, y_pred))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()
```

**Metric Selection Guide:**
- Spam detection: Optimize precision (minimize false positives)
- Medical diagnosis: Optimize recall (minimize false negatives)
- Imbalanced binary: Use PR-AUC, not ROC-AUC
- Multi-class: Use macro/micro averaged F1 based on class importance

#### 4.2.2 Regression Metrics (HIGH-IMPACT)

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import numpy as np

y_true, y_pred = get_predictions()

# Scale-dependent errors
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
mape = mean_absolute_percentage_error(y_true, y_pred)

# Scale-independent (0-1, lower is better)
def max_error_normalized(y_true, y_pred):
    return np.max(np.abs(y_true - y_pred)) / (np.max(y_true) - np.min(y_true))

# R-squared (explained variance)
from sklearn.metrics import r2_score
r2 = r2_score(y_true, y_pred)
```

**When to use which:**
- RMSE: Penalizes large errors more heavily (sensitive to outliers)
- MAE: Robust to outliers, easier to interpret
- MAPE: Business-friendly percentage, but breaks at y=0
- R²: Communicative, but can be misleading for non-linear relationships

#### 4.2.3 Custom Metrics (INCREMENTAL)

Create domain-specific metrics when standard ones don't capture business value:

```python
from sklearn.metrics import make_scorer

# Example: Cost-sensitive metric for fraud detection
def business_cost(y_true, y_pred, cost_fp=10, cost_fn=1000):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return -(fp * cost_fp + fn * cost_fn)  # Negative for minimization

business_scorer = make_scorer(business_cost, cost_fp=10, cost_fn=1000)
scores = cross_val_score(model, X, y, cv=5, scoring=business_scorer)
```

### 4.3 Calibration-Aware Evaluation

#### 4.3.1 Brier Score & Expected Calibration Error (HIGH-IMPACT)

A well-calibrated model's predicted probabilities match observed frequencies. If model predicts 0.8 for 100 samples, ~80 should be positive.

```python
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
import numpy as np

# Brier score (lower is better, 0 is perfect)
brier = brier_score_loss(y_true, y_proba)

# Expected Calibration Error
def expected_calibration_error(y_true, y_proba, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy='uniform')
    bin_counts = np.histogram(y_proba, bins=n_bins, range=(0, 1))[0]
    bin_weights = bin_counts / len(y_true)

    ece = np.sum(bin_weights * np.abs(prob_true - prob_pred))
    return ece

ece = expected_calibration_error(y_true, y_proba)

print(f"Brier Score: {brier:.4f}")
print(f"ECE: {ece:.4f}")
```

**Diagnosing Calibration Issues:**
- Brier score decomposes into: calibration + refinement + uncertainty
- A lower Brier doesn't guarantee better calibration—model could have better discrimination
- Use ECE for direct calibration assessment

#### 4.3.2 Reliability Diagrams (HIGH-IMPACT)

Visual inspection of calibration:

```python
from sklearn.calibration import CalibrationDisplay
import matplotlib.pyplot as plt

# Single model
CalibrationDisplay.from_estimator(model, X_test, y_test, n_bins=10)
plt.show()

# Multiple models comparison
fig, ax = plt.subplots(figsize=(10, 6))
for name, model in models:
    CalibrationDisplay.from_estimator(model, X_test, y_test, n_bins=10, ax=ax, name=name)
plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
plt.legend()
plt.show()
```

**Common Calibration Patterns:**
- S-shaped curve: Under-confident (pushes probs toward 0.5)
- Inverse S: Over-confident (pushes probs toward 0 and 1)
- Random forest: Typically under-confident near extremes
- Naive Bayes: Typically over-confident

#### 4.3.3 Calibration Methods (HIGH-IMPACT)

```python
from sklearn.calibration import CalibratedClassifierCV

# Sigmoid calibration (Platt scaling) - parametric, works well with small data
calibrated_sigmoid = CalibratedClassifierCV(
    base_estimator=model,
    method='sigmoid',
    cv='prefit'  # Use if model already fitted
)
calibrated_sigmoid.fit(X_cal, y_cal)

# Isotonic regression - non-parametric, needs more data
calibrated_isotonic = CalibratedClassifierCV(
    base_estimator=model,
    method='isotonic',
    cv='prefit'
)
calibrated_isotonic.fit(X_cal, y_cal)

# Temperature scaling - for neural networks, preserves ranking
# Requires external implementation (use torch.nn.CrossEntropyLoss with T parameter)
```

**Choosing a method:**
- Sigmoid: Small datasets (<1000 samples), symmetrical calibration error
- Isotonic: Larger datasets (>1000 samples), arbitrary monotonic distortion
- Temperature scaling: Deep networks, multi-class, preserves ranking

**Important:** Always calibrate on held-out data, not training data. Use a dedicated calibration split or cross-validation.

### 4.4 Statistical Significance Testing

#### 4.4.1 Paired t-tests (INCREMENTAL)

When comparing models, test if differences are statistically significant:

```python
from scipy import stats
import numpy as np

# Get cross-validation scores for both models (same folds!)
scores_model_a = cross_val_score(model_a, X, y, cv=10, scoring='accuracy')
scores_model_b = cross_val_score(model_b, X, y, cv=10, scoring='accuracy')

# Paired t-test (same folds)
t_stat, p_value = stats.ttest_rel(scores_model_a, scores_model_b)

print(f"Model A: {scores_model_a.mean():.4f}")
print(f"Model B: {scores_model_b.mean():.4f}")
print(f"Difference: {scores_model_a.mean() - scores_model_b.mean():.4f}")
print(f"p-value: {p_value:.4f}")

if p_value < 0.05:
    print("Significant difference (p < 0.05)")
else:
    print("Not significant - difference could be due to chance")
```

**Critical:** Use paired tests on same folds. Unpaired tests are anti-conservative for CV comparisons.

#### 4.4.2 Bootstrap Confidence Intervals (INCREMENTAL)

Bootstrap provides non-parametric confidence intervals:

```python
def bootstrap_ci(scores, n_bootstrap=10000, alpha=0.05):
    boot_means = []
    n = len(scores)

    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=n, replace=True)
        boot_means.append(sample.mean())

    lower = np.percentile(boot_means, 100 * alpha / 2)
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return np.mean(scores), lower, upper

mean, lower, upper = bootstrap_ci(scores_model_b)
print(f"Mean: {mean:.4f}, 95% CI: [{lower:.4f}, {upper:.4f}]")
```

#### 4.4.3 Multiple Comparison Correction (INCREMENTAL)

When comparing many models, correct for multiple testing to avoid false positives:

```python
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

# Example: Comparing 10 models
p_values = [compare_model_i_to_baseline(i) for i in range(10)]

# Bonferroni correction (conservative)
rejected, p_corrected, _, _ = multipletests(p_values, method='bonferroni')

# Holm-Bonferroni (less conservative, more powerful)
rejected, p_corrected, _, _ = multipletests(p_values, method='holm')

# Benjamini-Hochberg FDR (controls false discovery rate)
rejected, p_corrected, _, _ = multipletests(p_values, method='fdr_bh')
```

**Common Pitfall:** Cherry-picking the best CV score without significance testing leads to inflated performance claims.

---

## Part B: Ensemble & Stacking Techniques

### 4.5 Bagging (Bootstrap Aggregating)

#### 4.5.1 Random Forests (HIGH-IMPACT)

Random forests combine bagging with random feature selection to decorrelate trees:

```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Classification
rf_clf = RandomForestClassifier(
    n_estimators=200,        # More trees = better, diminishing returns
    max_depth=None,          # Grow trees fully (low bias)
    min_samples_split=2,     # Regularization
    min_samples_leaf=1,
    max_features='sqrt',     # Random subset at each split
    bootstrap=True,          # Bagging
    oob_score=True,          # Out-of-bag estimation
    n_jobs=-1,
    random_state=42,
    class_weight='balanced'  # For imbalanced data
)

# Regression
rf_reg = RandomForestRegressor(
    n_estimators=200,
    max_features=1.0/3,      # Recommendation for regression
    oob_score=True
)

# OOB score is unbiased estimate (free validation set!)
rf_clf.fit(X_train, y_train)
print(f"OOB Score: {rf_clf.oob_score_:.4f}")
```

**Variance Reduction Mechanism:**
- Bagging: Trains on bootstrap samples (sampling with replacement)
- Each tree sees ~63% unique samples, ~37% out-of-bag
- Averaging uncorrelated predictions reduces variance: Var(average) = Var/ntrees

**Diagnosing Need for More Trees:** Plot OOB score vs n_estimators—should plateau.

#### 4.5.2 Bagging Neural Networks (INCREMENTAL)

```python
from sklearn.utils import resample
import numpy as np

def bagged_nn_predictions(X_train, y_train, X_test, n_bags=10):
    predictions = []

    for i in range(n_bags):
        # Bootstrap sample
        X_boot, y_boot = resample(X_train, y_train, replace=True, random_state=i)

        # Train neural network
        model = create_nn_model()
        model.fit(X_boot, y_boot, epochs=100, verbose=0)

        # Predict
        pred = model.predict(X_test)
        predictions.append(pred)

    return np.mean(predictions, axis=0)

# Reduces variance from neural network instability
```

**When bagging helps:**
- High-variance models (deep trees, neural nets)
- Unstable learners (small data changes cause large prediction changes)
- Doesn't help high-bias models (linear models)

### 4.6 Boosting

#### 4.6.1 Gradient Boosting (HIGH-IMPACT)

Boosting sequentially trains weak learners to correct previous errors:

```python
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

# Scikit-learn implementation
gb_clf = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,       # Shrinkage - smaller = more robust, needs more trees
    max_depth=3,             # Shallow trees for weak learners
    subsample=0.8,           # Stochastic gradient boosting (row sampling)
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)

# XGBoost (faster, regularized)
xgb_clf = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,    # Feature sampling
    reg_alpha=0.1,           # L1 regularization
    reg_lambda=1.0,          # L2 regularization
    eval_metric='logloss',
    early_stopping_rounds=10,
    random_state=42
)

# LightGBM (efficient with large data)
lgbm_clf = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    num_leaves=31,           # LightGBM-specific: controls tree complexity
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    verbose=-1
)

# CatBoost (handles categorical features natively)
cat_clf = CatBoostClassifier(
    depth=6,
    learning_rate=0.1,
    iterations=100,
    l2_leaf_reg=3.0,
    random_seed=42,
    verbose=False
)
```

**When boosting helps vs hurts:**

| Scenario | Boosting | Bagging |
|----------|----------|---------|
| High bias (underfitting) | Helps | Limited help |
| High variance (overfitting) | Can overfit | Helps |
| Noisy data | Can overfit noise | More robust |
| Outliers | Sensitive | Less sensitive |

**Common Pitfall:** Too high learning rate with too few trees—boosting overfits to noise. Use learning_rate=0.01-0.1 with n_estimators=200-1000.

#### 4.6.2 AdaBoost (INCREMENTAL)

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

ada_clf = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=1),  # Decision stumps
    n_estimators=50,
    learning_rate=1.0,
    algorithm='SAMME'  # For discrete AdaBoost
)
```

**Use when:** Simple, interpretable weak learners (decision stumps) are sufficient. Generally outperformed by gradient boosting.

### 4.7 Stacking & Blending

#### 4.7.1 Stacking with Cross-Validated Predictions (HIGH-IMPACT)

Stacking uses a meta-learner to combine base models. Critical: use out-of-fold predictions for training meta-learner to avoid data leakage.

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict
import numpy as np

# Define base models
estimators = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('svm', SVC(probability=True, random_state=42))
]

# Stacking with CV
stacking_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),  # Meta-learner
    cv=5,                                  # K-fold for OOF predictions
    stack_method='predict_proba',          # Use probabilities
    n_jobs=-1
)

stacking_clf.fit(X_train, y_train)

# Manual stacking for more control
def manual_stacking(X_train, y_train, X_test, base_models, meta_model, cv=5):
    from sklearn.model_selection import KFold

    # Get OOF predictions for training meta-features
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    meta_features_train = np.zeros((len(X_train), len(base_models)))

    for i, (name, model) in enumerate(base_models):
        # OOF predictions via cross_val_predict
        oof_preds = cross_val_predict(model, X_train, y_train, cv=kf, method='predict_proba')
        meta_features_train[:, i] = oof_preds[:, 1]  # Positive class prob

        # Retrain on full data for test predictions
        model.fit(X_train, y_train)

    # Train meta-learner
    meta_model.fit(meta_features_train, y_train)

    # Create meta-features for test set
    meta_features_test = np.zeros((len(X_test), len(base_models)))
    for i, (name, model) in enumerate(base_models):
        meta_features_test[:, i] = model.predict_proba(X_test)[:, 1]

    return meta_model.predict(meta_features_test)
```

**Meta-learner design:**
- Simple models (LogisticRegression, RidgeRegression) often work best
- Regularized meta-learner prevents overfitting to base model predictions
- For regression: use linear/ridge regression
- For classification: use logistic regression or simple neural net

#### 4.7.2 Blending (INCREMENTAL)

Blending uses a holdout set for meta-learner training (simpler but wastes data):

```python
def blend_models(X_train, y_train, X_test, base_models, meta_model, blend_size=0.2):
    from sklearn.model_selection import train_test_split

    # Split training data
    X_tr, X_blend, y_tr, y_blend = train_test_split(
        X_train, y_train, test_size=blend_size, random_state=42
    )

    # Train base models on main portion
    blend_features = []
    test_features = []

    for name, model in base_models:
        model.fit(X_tr, y_tr)
        blend_features.append(model.predict_proba(X_blend)[:, 1])
        test_features.append(model.predict_proba(X_test)[:, 1])

    # Train meta-learner on blend set
    meta_features = np.column_stack(blend_features)
    meta_model.fit(meta_features, y_blend)

    # Predict on test
    test_meta = np.column_stack(test_features)
    return meta_model.predict(test_meta)
```

#### 4.7.3 Practical Stacking Architecture (INCREMENTAL)

```python
# Multi-level stacking
level_0 = [
    ('rf', RandomForestClassifier(n_estimators=100)),
    ('gb', GradientBoostingClassifier(n_estimators=100)),
    ('et', ExtraTreesClassifier(n_estimators=100))
]

level_1 = [
    ('xgb', XGBClassifier(n_estimators=50)),
    ('lgbm', LGBMClassifier(n_estimators=50))
]

# Level 0 predictions -> Level 1 -> Final prediction
from sklearn.ensemble import StackingClassifier

stacking = StackingClassifier(
    estimators=level_0,
    final_estimator=StackingClassifier(
        estimators=level_1,
        final_estimator=LogisticRegression()
    ),
    cv=5
)
```

**Diversity requirement:** Base models should make different errors. Correlated models provide little benefit over a single model.

### 4.8 Ensemble Pruning

#### 4.8.1 Ensemble Selection (INCREMENTAL)

When ensembles are too expensive, select optimal subset:

```python
from itertools import combinations
import numpy as np

def ensemble_selection(models, X_val, y_val, n_select=None, metric='accuracy'):
    """Caruana's ensemble selection algorithm"""
    if n_select is None:
        n_select = len(models) // 2

    # Get predictions from all models
    predictions = []
    for model in models:
        pred = model.predict_proba(X_val)[:, 1]
        predictions.append(pred)

    predictions = np.array(predictions)

    # Normalize predictions
    predictions = (predictions - predictions.mean()) / predictions.std()

    # Greedy selection
    selected = []
    ensemble_pred = np.zeros(len(y_val))

    for _ in range(n_select):
        best_score = -np.inf
        best_idx = None

        for i in range(len(models)):
            if i in selected:
                continue

            # Try adding this model
            test_pred = (ensemble_pred + predictions[i]) / (len(selected) + 1)
            score = compute_score(y_val, (test_pred > 0.5).astype(int), metric)

            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx is not None:
            selected.append(best_idx)
            ensemble_pred += predictions[best_idx]
            ensemble_pred /= len(selected)

    return [models[i] for i in selected]
```

**When to prune:**
- Model size constraints (mobile deployment)
- Latency requirements (real-time prediction)
- Diminishing returns from additional models

### 4.9 Diversity Metrics

#### 4.9.1 Measuring Ensemble Diversity (INCREMENTAL)

Diverse ensembles generalize better. Key metrics:

```python
def disagreement_measure(y_true, preds_a, preds_b):
    """Fraction of examples where models disagree"""
    disagree = np.sum(preds_a != preds_b)
    return disagree / len(y_true)

def double_fault_measure(y_true, preds_a, preds_b):
    """Fraction of examples where both models are wrong"""
    both_wrong = np.sum((preds_a != y_true) & (preds_b != y_true))
    return both_wrong / len(y_true)

def q_statistics(y_true, preds_a, preds_b):
    """Correlation between model errors"""
    n00 = n01 = n10 = n11 = 0

    for yt, pa, pb in zip(y_true, preds_a, preds_b):
        if pa == pb:
            if pa == yt:
                n11 += 1  # Both correct
            else:
                n00 += 1  # Both wrong
        else:
            if pa == yt:
                n10 += 1  # A correct, B wrong
            else:
                n01 += 1  # B correct, A wrong

    if (n11 * n00 + n01 * n10) == 0:
        return 0

    q = (n11 * n00 - n01 * n10) / (n11 * n00 + n01 * n10)
    return q  # Range: -1 (diverse) to 1 (similar)

def entropy_diversity(predictions):
    """Average prediction entropy (for probabilistic predictions)"""
    # predictions: (n_models, n_samples, n_classes)
    mean_pred = predictions.mean(axis=0)  # Average prediction per sample
    entropy = -(mean_pred * np.log(mean_pred + 1e-10)).sum(axis=1)
    return entropy.mean()

# Usage example
preds_a = model_a.predict(X_val)
preds_b = model_b.predict(X_val)

disagreement = disagreement_measure(y_val, preds_a, preds_b)
double_fault = double_fault_measure(y_val, preds_a, preds_b)
q_stat = q_statistics(y_val, preds_a, preds_b)

print(f"Disagreement: {disagreement:.3f}")
print(f"Double Fault: {double_fault:.3f}")
print(f"Q-statistic: {q_stat:.3f}")
```

**Interpretation:**
- High disagreement: Models make different errors (good for ensembling)
- High double fault: Models fail on same examples (poor ensemble candidates)
- Q near 0: Independent errors (ideal)
- Q near 1: Correlated errors (little benefit from ensembling)

---

## Part C: Regularization & Bias-Variance

### 4.10 Classical Regularization

#### 4.10.1 L1 Regularization (Lasso) (HIGH-IMPACT)

L1 regularization encourages sparsity—useful for feature selection:

```python
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel

# Regression with L1
lasso = Lasso(alpha=0.1)  # alpha controls regularization strength
lasso.fit(X_train, y_train)

# Check which features were selected
selected_features = np.where(lasso.coef_ != 0)[0]
print(f"Selected {len(selected_features)} out of {X.shape[1]} features")

# Feature selection using L1
selector = SelectFromModel(LogisticRegression(penalty='l1', solver='saga', C=0.1))
selector.fit(X_train, y_train)
X_selected = selector.transform(X)
```

**When L1 helps:**
- High-dimensional data with few relevant features
- Need interpretable models (sparse solutions)
- Feature selection as preprocessing step

#### 4.10.2 L2 Regularization (Ridge) (HIGH-IMPACT)

L2 regularization shrinks coefficients smoothly:

```python
from sklearn.linear_model import Ridge, LogisticRegression

# Regression with L2
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

# Classification with L2
log_reg_l2 = LogisticRegression(penalty='l2', C=1.0)  # C = 1/alpha
log_reg_l2.fit(X_train, y_train)
```

**When L2 helps:**
- Multicollinear features (correlated predictors)
- Want all features but reduce overfitting
- Prefer stable coefficients over exact zeros

#### 4.10.3 Elastic Net (INCREMENTAL)

Combines L1 and L2 for balanced sparsity and stability:

```python
from sklearn.linear_model import ElasticNet, LogisticRegression

# ElasticNet: alpha controls total strength, l1_ratio controls L1/L2 mix
# l1_ratio=0: pure L2, l1_ratio=1: pure L1, l1_ratio=0.5: equal mix
en = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)
en.fit(X_train, y_train)

# For classification (use saga solver)
log_reg_en = LogisticRegression(
    penalty='elasticnet',
    solver='saga',
    l1_ratio=0.5,
    C=1.0
)
log_reg_en.fit(X_train, y_train)
```

**When Elastic Net helps:**
- Correlated features where L1 arbitrarily selects one
- Want some sparsity but not as extreme as pure L1
- p >> n (more features than samples)

### 4.11 Neural Network Regularization

#### 4.11.1 Dropout (HIGH-IMPACT)

Dropout randomly deactivates neurons during training, preventing co-adaptation:

```python
import torch
import torch.nn as nn

class RegularizedNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_rate=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)  # Only active during training
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# Dropout rate tuning
# - 0.2-0.3: Mild regularization
# - 0.5: Standard (original paper)
# - 0.7-0.8: Heavy regularization
model = RegularizedNet(input_size=784, hidden_size=256, num_classes=10, dropout_rate=0.5)

# Important: Set model to eval mode during inference
model.eval()  # Disables dropout
with torch.no_grad():
    predictions = model(X_test)

model.train()  # Re-enables dropout for training
```

**Common Pitfall:** Forgetting to call `model.eval()` during inference—dropout stays active, causing stochastic predictions.

#### 4.11.2 Batch Normalization (HIGH-IMPACT)

Batch norm reduces internal covariate shift and provides regularization:

```python
class BatchNormNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x

# Batch norm tips:
# - Use before activation function
# - Can reduce/eliminate need for dropout
# - Adds noise during training (regularization effect)
# - Small batch sizes (<8) can make it unstable
```

**Training vs Inference:**
- Training: Uses batch statistics (mean/var), adds noise
- Inference: Uses running statistics, deterministic
- Always call `model.eval()` for inference!

#### 4.11.3 Weight Decay (HIGH-IMPACT)

Weight decay is L2 regularization applied to network weights:

```python
import torch.optim as optim

# Standard weight decay
optimizer = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4  # L2 regularization
)

# AdamW: Decoupled weight decay (recommended for transformers)
optimizer_adamw = optim.AdamW(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.01  # True weight decay, not L2
)

# Different weight decay for different layers
optimizer = optim.AdamW([
    {'params': model.base.parameters(), 'weight_decay': 1e-2},
    {'params': model.head.parameters(), 'weight_decay': 1e-4}
], lr=1e-4)
```

**Weight decay values:**
- 1e-4: Standard for CNNs
- 1e-2 to 1e-1: For transformers and large networks
- 0: No regularization (risky for large models)

#### 4.11.4 Advanced Regularization Techniques (INCREMENTAL)

```python
# Mixup: Linear interpolation between samples
def mixup_data(x, y, alpha=0.2):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# Mixup loss
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# CutMix: Cut and paste between images
def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    # Get random bbox
    W, H = x.size()[2], x.size()[3]
    cut_rat = np.sqrt(1. - lam)
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

    # Adjust lambda based on actual cut area
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam

# Label smoothing
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_classes = pred.size(-1)
        log_preds = torch.log_softmax(pred, dim=-1)

        # Smooth labels: (1 - eps) * one_hot + eps / n_classes
        with torch.no_grad():
            smooth_target = torch.zeros_like(pred)
            smooth_target.fill_(self.smoothing / (n_classes - 1))
            smooth_target.scatter_(1, target.unsqueeze(1), 1 - self.smoothing)

        loss = torch.sum(-smooth_target * log_preds, dim=-1).mean()
        return loss

# Usage
criterion_smooth = LabelSmoothingCrossEntropy(smoothing=0.1)
```

**When to use advanced augmentation:**
- Mixup: General purpose, works for tabular too
- CutMix: Image-specific, preserves spatial structure
- Label smoothing: Prevents overconfident predictions, helpful for calibration

### 4.12 Early Stopping

#### 4.12.1 Proper Implementation (HIGH-IMPACT)

Early stopping prevents overfitting by halting training when validation performance plateaus:

```python
import torch
import numpy as np

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        """
        Args:
            patience: How many epochs to wait after last improvement
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore model from best epoch
        """
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
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
        else:
            self.best_score = val_score
            self.save_checkpoint(model)
            self.counter = 0

        return self.early_stop

    def save_checkpoint(self, model):
        """Save model weights when we get a new best score"""
        self.best_weights = model.state_dict().copy()

# Training loop with early stopping
def train_with_early_stopping(model, train_loader, val_loader, n_epochs=100):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    early_stopping = EarlyStopping(patience=10, min_delta=0.001)

    train_losses = []
    val_scores = []

    for epoch in range(n_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_score = evaluate_model(model, val_loader)  # Higher is better
        val_scores.append(val_score)

        print(f'Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.4f}, Val Score: {val_score:.4f}')

        # Check early stopping
        if early_stopping(val_score, model):
            print(f'Early stopping at epoch {epoch+1}')
            break

    return train_losses, val_scores

def evaluate_model(model, data_loader):
    """Return validation metric (higher is better)"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    return correct / total
```

#### 4.12.2 Patience Tuning (INCREMENTAL)

```python
# Patience guidelines:
# - Small datasets (n < 1000): patience 5-10
# - Medium datasets (1000 < n < 100k): patience 10-20
# - Large datasets (n > 100k): patience 20-50

# Adaptive patience (increase during training)
class AdaptiveEarlyStopping(EarlyStopping):
    def __init__(self, initial_patience=10, max_patience=50, **kwargs):
        super().__init__(patience=initial_patience, **kwargs)
        self.initial_patience = initial_patience
        self.max_patience = max_patience
        self.epoch = 0

    def __call__(self, val_score, model):
        self.epoch += 1

        # Increase patience over time
        new_patience = min(
            self.initial_patience + self.epoch // 10,
            self.max_patience
        )
        self.patience = int(new_patience)

        return super().__call__(val_score, model)
```

**Common Pitfalls:**
- Too small patience: Stops before convergence
- Too large patience: Wastes computation, may overfit
- Forgetting to restore best weights: Returns overfit model

### 4.13 Data Augmentation

#### 4.13.1 Image Augmentation (HIGH-IMPACT)

```python
import torchvision.transforms as transforms
from PIL import Image

# Training augmentation pipeline
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),  # Random crop + resize
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.5)  # Cutout regularization
])

# Validation: NO augmentation (only normalization)
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Advanced: AutoAugment / RandAugment
from torchvision.transforms import AutoAugment, AutoAugmentPolicy, RandAugment

# AutoAugment: Learned augmentation policies
autoaugment_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.AutoAugment(policy=AutoAugmentPolicy.IMAGENET),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# RandAugment: Randomly sampled augmentations
randaugment_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

#### 4.13.2 Tabular Augmentation (INCREMENTAL)

```python
import numpy as np

# SMOTE for imbalanced classification
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN

# Standard SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Borderline SMOTE (only synthesize near decision boundary)
smote_bl = BorderlineSMOTE(kind='borderline-1', random_state=42)
X_resampled, y_resampled = smote_bl.fit_resample(X_train, y_train)

# ADASYN (adaptive synthesis based on density)
adasyn = ADASYN(random_state=42)
X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)

# Gaussian noise injection for regression
def add_gaussian_noise(X, noise_level=0.01):
    """Add small Gaussian noise to features"""
    noise = np.random.normal(0, noise_level, X.shape)
    return X + noise

# Mixup for tabular
def tabular_mixup(X, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    batch_size = X.shape[0]
    index = np.random.permutation(batch_size)

    mixed_X = lam * X + (1 - lam) * X[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_X, mixed_y
```

### 4.14 Diagnosing Bias-Variance

#### 4.14.1 Learning Curves (HIGH-IMPACT)

Learning curves show how performance changes with training size—key diagnostic tool:

```python
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, X, y, cv=5, train_sizes=None):
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)

    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1,
        train_sizes=train_sizes,
        scoring='accuracy',
        shuffle=True,
        random_state=42
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', label='Training score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    plt.plot(train_sizes, val_mean, 'o-', label='Validation score')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)

    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy')
    plt.title('Learning Curve')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

    # Diagnosis
    final_train = train_mean[-1]
    final_val = val_mean[-1]
    gap = final_train - final_val

    print(f"Final Training Score: {final_train:.4f}")
    print(f"Final Validation Score: {final_val:.4f}")
    print(f"Gap: {gap:.4f}")

    if gap > 0.1:
        if final_val < 0.8:
            print("Diagnosis: High variance (overfitting) - gap is large")
        else:
            print("Diagnosis: Acceptable variance - model performs well")
    elif final_val < 0.8:
        print("Diagnosis: High bias (underfitting) - both scores low")
    else:
        print("Diagnosis: Good fit - model generalizes well")

# Usage
plot_learning_curve(RandomForestClassifier(n_estimators=100), X, y)
```

**Learning curve patterns:**
- **High variance (overfitting):** Large train-val gap, validation plateaus
  - Solution: More data, more regularization, simpler model
- **High bias (underfitting):** Both scores low, close together
  - Solution: More complex model, feature engineering, less regularization
- **Good fit:** High scores, small gap, both converging

#### 4.14.2 Validation Curves (INCREMENTAL)

```python
from sklearn.model_selection import validation_curve

def plot_validation_curve(estimator, X, y, param_name, param_range, cv=5):
    train_scores, val_scores = validation_curve(
        estimator, X, y, param_name=param_name,
        param_range=param_range, cv=cv, scoring='accuracy'
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(param_range, train_mean, 'o-', label='Training score')
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1)
    plt.plot(param_range, val_mean, 'o-', label='Validation score')
    plt.fill_between(param_range, val_mean - val_std, val_mean + val_std, alpha=0.1)

    plt.xlabel(param_name)
    plt.ylabel('Accuracy')
    plt.title(f'Validation Curve ({param_name})')
    plt.legend(loc='best')
    plt.xscale('log')
    plt.grid(True)
    plt.show()

# Example: Find optimal max_depth for RandomForest
plot_validation_curve(
    RandomForestClassifier(n_estimators=100),
    X, y,
    param_name='max_depth',
    param_range=[3, 5, 10, 15, 20, None]
)
```

#### 4.14.3 Diagnostic Flowchart (INCREMENTAL)

```python
def diagnose_model(X_train, y_train, X_val, y_val, model):
    """Comprehensive bias-variance diagnosis"""

    # Train model
    model.fit(X_train, y_train)

    # Get predictions
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    train_prob = model.predict_proba(X_train)[:, 1]
    val_prob = model.predict_proba(X_val)[:, 1]

    # Calculate metrics
    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)
    train_f1 = f1_score(y_train, train_pred)
    val_f1 = f1_score(y_val, val_pred)
    train_brier = brier_score_loss(y_train, train_prob)
    val_brier = brier_score_loss(y_val, val_prob)

    # Print diagnosis
    print("=" * 50)
    print("MODEL DIAGNOSIS")
    print("=" * 50)

    print(f"\n1. ACCURACY")
    print(f"   Training:   {train_acc:.4f}")
    print(f"   Validation: {val_acc:.4f}")
    print(f"   Gap:        {train_acc - val_acc:.4f}")

    print(f"\n2. F1 SCORE")
    print(f"   Training:   {train_f1:.4f}")
    print(f"   Validation: {val_f1:.4f}")

    print(f"\n3. CALIBRATION (Brier, lower is better)")
    print(f"   Training:   {train_brier:.4f}")
    print(f"   Validation: {val_brier:.4f}")

    # Diagnosis
    gap = train_acc - val_acc

    print(f"\n4. DIAGNOSIS")

    if gap > 0.15:
        print("   HIGH VARIANCE (Overfitting)")
        print("   - Train >> Val gap")
        print("   Solutions:")
        print("     * Add more training data")
        print("     * Increase regularization")
        print("     * Reduce model complexity")
        print("     * Use ensemble methods (bagging)")

    elif val_acc < 0.7:
        print("   HIGH BIAS (Underfitting)")
        print("   - Both train and val scores low")
        print("   Solutions:")
        print("     * Increase model complexity")
        print("     * Add more features")
        print("     * Reduce regularization")
        print("     * Try different model architecture")

    elif gap < 0.05 and val_acc > 0.85:
        print("   GOOD FIT")
        print("   - Model generalizes well")
        print("   Consider:")
        print("     * Model is production-ready")
        print("     * Monitor calibration for probability outputs")

    else:
        print("   MODERATE FIT")
        print("   - Some room for improvement")
        print("   Consider:")
        print("     * Hyperparameter tuning")
        print("     * Feature engineering")
        print("     * Ensemble methods")

    # Additional checks
    print(f"\n5. ADDITIONAL CHECKS")

    # Check for overfitting to calibration
    if val_brier > train_brier * 1.5:
        print("   - Calibration degrades on validation")
        print("     Consider probability calibration")

    # Check class balance issues
    unique, counts = np.unique(y_train, return_counts=True)
    if counts.min() / counts.max() < 0.1:
        print("   - Severe class imbalance detected")
        print("     Use stratified sampling, class weights, or appropriate metrics")

    return {
        'train_acc': train_acc,
        'val_acc': val_acc,
        'gap': gap,
        'diagnosis': 'high_variance' if gap > 0.15 else 'high_bias' if val_acc < 0.7 else 'good_fit'
    }

# Usage
diagnosis = diagnose_model(X_train, y_train, X_val, y_val, RandomForestClassifier())
```

---

## Summary Checklist

### Validation & Evaluation
- [ ] Use stratified k-fold for imbalanced classification
- [ ] Use group k-fold when samples share group IDs
- [ ] Use time series split for temporal data
- [ ] Apply nested CV for unbiased hyperparameter selection
- [ ] Choose metrics aligned with business objectives
- [ ] Check calibration for probability predictions
- [ ] Use statistical tests for model comparisons

### Ensembles
- [ ] Try bagging (Random Forest) for high-variance models
- [ ] Try boosting (XGBoost/LightGBM) for high-bias problems
- [ ] Use stacking with diverse base models
- [ ] Verify base models have diverse errors
- [ ] Consider ensemble pruning for deployment constraints

### Regularization
- [ ] Apply L1 for feature selection, L2 for stability
- [ ] Use dropout (0.3-0.5) for neural networks
- [ ] Apply batch normalization before activations
- [ ] Implement early stopping with patience tuning
- [ ] Use data augmentation appropriate to your domain
- [ ] Diagnose bias-variance with learning curves
