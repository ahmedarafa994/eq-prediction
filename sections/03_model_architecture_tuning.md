# 3. Model Architecture & Hyperparameter Tuning

## Introduction

Selecting the right model architecture and systematically tuning hyperparameters are two of the most impactful decisions in any machine learning project. A poor choice here cannot be recovered by clever feature engineering or extensive hyperparameter tuning. Conversely, starting with an appropriate architecture matched to your data characteristics and constraints makes tuning far more effective and efficient.

This section provides a systematic framework for:

1. **Part A**: Selecting model architectures based on data characteristics, constraints, and requirements
2. **Part B**: Executing systematic hyperparameter tuning using modern optimization methods

Throughout, we mark topics as **HIGH-IMPACT** or **INCREMENTAL** to help prioritize effort.

---

## Part A: Model Architecture Matching

### 3.1 Decision Framework for Model Selection

The choice of model family should be driven by a structured decision process, not by fashion or familiarity. Below is a systematic framework for architecture selection.

#### 3.1.1 The Architecture Decision Flowchart

```
START
  |
  v
What is your primary data modality?
  |
  +-- Tabular (structured, heterogeneous features) --------> GO TO 3.1.2
  |
  +-- Images (spatial, visual patterns) -------------------> GO TO 3.1.3
  |
  +-- Text/Sequences (temporal, language) ----------------> GO TO 3.1.4
  |
  +-- Time-Series (temporal, potentially multivariate) ---> GO TO 3.1.5
  |
  +-- Graph/Relational -----------------------------------> GO TO 3.1.6
  |
  v
FOR ANY MODALITY: Apply constraint filters
  |
  v
Interpretability required? -------------------------------> GO TO 3.1.7
  |
  v
Latency budget critical? --------------------------------> GO TO 3.1.8
  |
  v
Training compute budget? --------------------------------> GO TO 3.1.9
  |
  v
Hardware constraints (edge, memory)? -------------------> GO TO 3.1.8
  |
  v
MAKE FINAL SELECTION
```

#### 3.1.2 Tabular Data Decision Tree

**HIGH-IMPACT** - Tabular data remains the most common modality in industry applications.

```
Tabular Data
  |
  v
Sample size (rows)?
  |
  +-- < 1,000 samples ----------------------------------> Simple models: Linear/Logistic Regression,
  |                                                      Regularized GLM, shallow decision trees
  |                                                      Avoid: Deep learning, ensembles
  |
  +-- 1K - 100K samples -------------------------------> Tree-based ensembles (XGBoost, LightGBM, CatBoost)
  |                                                     Random Forests as baseline
  |                                                     Consider: Regularized linear models as baseline
  |
  +-- 100K - 1M samples ------------------------------> Gradient boosting (primary choice)
  |                                                     Deep learning if high-cardinality embeddings helpful
  |
  +-- > 1M samples ------------------------------------> Gradient boosting (still excellent)
                                                        Deep learning (TabNet, FT-Transformer) if:
                                                          - Extreme high-cardinality categorical features
                                                          - Transfer learning opportunities
                                                          - Integration with other modalities
  |
  v
Feature characteristics?
  |
  +-- Mostly numeric, linear relationships -----------> Start: Linear/Regularized models
  |                                                    Try: Gradient boosting for non-linear gains
  |
  +-- Mixed categorical/numeric ----------------------> Gradient boosting (CatBoost best for categoricals)
  |                                                    Tree-based ensembles handle natively
  |
  +-- High-cardinality categoricals ------------------> Target encoding + Gradient boosting
  |                                                    Entity embeddings (neural) if >100K samples
  |
  +-- Sparse features ---------------------------------> Linear models (SGD, liblinear)
                                                        FFM (Field-aware Factorization Machines)
  |
  v
Interpretability critical?
  |
  +-- Yes ---------------------------------------------> Glass-box models: GLMs, GAMs, EBM
  |                                                    Single decision trees (for visualization)
  |                                                    Shapley values for complex models
  |
  +-- No ---------------------------------------------> Gradient boosting (max accuracy)
                                                        Deep learning (if data size warrants)
```

**Practical Recommendations for Tabular Data:**

1. **Default starting point (2025 consensus)**: XGBoost, LightGBM, or CatBoost
2. **Baseline first**: Always run a simple linear model to establish a performance floor
3. **Categorical handling**: CatBoost handles categoricals natively; XGBoost/LightGBM require encoding
4. **Small data (<5K rows)**: Regularized linear models often outperform due to lower variance

**Example: Gradient Boosting Baseline (XGBoost)**

```python
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def xgboost_baseline(X, y, task_type='classification'):
    """Establish a strong baseline with XGBoost."""
    if task_type == 'classification':
        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss',
            early_stopping_rounds=20
        )
    else:
        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            early_stopping_rounds=20
        )

    # Use cross-validation for robust evaluation
    scores = cross_val_score(
        model, X, y,
        cv=5,
        scoring='accuracy' if task_type == 'classification' else 'neg_root_mean_squared_error',
        n_jobs=-1
    )

    return {
        'mean_score': scores.mean(),
        'std_score': scores.std(),
        'model': model
    }

# Usage
# results = xgboost_baseline(X_train, y_train, task_type='classification')
# print(f"Baseline CV accuracy: {results['mean_score']:.4f} (+/- {results['std_score']:.4f})")
```

#### 3.1.3 Image Data Decision Tree

**HIGH-IMPACT** - Computer vision has clear architectural defaults.

```
Image Data
  |
  v
Task type?
  |
  +-- Classification -----------------------------------> ConvNext, ResNet, EfficientNet (transfer learning)
  |                                                    Vision Transformer (ViT) for very large datasets
  |
  +-- Detection/Segmentation ---------------------------> Mask R-CNN, YOLO, DETR
  |                                                      U-Net for segmentation
  |
  +-- Generation ---------------------------------------> GANs (StyleGAN), Diffusion models, VAEs
  |
  +-- Style Transfer -----------------------------------> Style transfer networks, diffusion-based
  |
  v
Data availability?
  |
  +-- < 1K images -------------------------------------> Transfer learning critical (freeze early layers)
  |                                                      Data augmentation essential
  |                                                      Consider few-shot methods
  |
  +-- 1K - 100K images --------------------------------> Transfer learning (fine-tune all layers)
  |
  +-- 100K - 10M images ------------------------------> Train from scratch (if domain differs from ImageNet)
  |                                                    Pre-training still helps
  |
  +-- > 10M images -------------------------------------> Train from scratch with modern architectures
  |
  v
Latency constraints?
  |
  +-- Edge deployment (<10ms) -------------------------> MobileNet, EfficientNet-Lite, TinyML approaches
  |                                                    Quantization, pruning after training
  |
  +-- Server deployment (10-100ms) --------------------> Standard architectures acceptable
  |                                                    Consider model distillation
  |
  +-- Batch processing (no latency concern) ---------> Use largest model within budget
```

**2025 Update: CNN vs Vision Transformer**

- **CNNs** (ResNet, ConvNext): Still strong for small-to-medium datasets, better sample efficiency
- **Vision Transformers** (ViT, Swin): Excel with large datasets (>1M images), pre-training critical
- **Hybrid** (ConvNeXt): CNN architecture designed with ViT principles, best of both

#### 3.1.4 Text/Sequence Data Decision Tree

**HIGH-IMPACT** - NLP has seen rapid evolution.

```
Text/Sequences
  |
  v
Task complexity?
  |
  +-- Simple classification (sentiment, topic) -------> Fine-tuned encoder (BERT, RoBERTa)
  |                                                    SetFit for few-shot (<50 examples/label)
  |
  +-- Sequence labeling (NER, POS) -------------------> Token classification with encoder models
  |                                                    CRF layer for structured output
  |
  +-- Generation (summarization, translation) -------> Decoder models (GPT, T5, LLaMA)
  |                                                    Encoder-decoder for translation
  |
  +-- Question Answering -----------------------------> Reader models (extractive) or generator models
  |
  +-- RAG/Retrieval -----------------------------------> Embedding models + Vector database + Generator
  |
  v
Data size?
  |
  +-- < 100 examples per label -----------------------> Few-shot: SetFit, GPT-NER, prompt engineering
  |                                                    Zero-shot: Pre-trained models with prompting
  |
  +-- 100 - 10K examples per label -------------------> Fine-tune pre-trained models
  |
  +-- > 10K examples per label -----------------------> Fine-tune larger models or train from scratch
  |
  v
Domain specificity?
  |
  +-- General domain ----------------------------------> Use pre-trained models (BERT, RoBERTa)
  |
  +-- Specialized domain (medical, legal) -----------> Domain-adapted pre-trained models
  |                                                    Continue pre-training on domain corpus
  |
  +-- Highly specialized/no public models ------------> Consider training from scratch if sufficient data
```

#### 3.1.5 Time-Series Decision Tree

**HIGH-IMPACT** - Time-series requires specialized handling of temporal dependencies.

```
Time-Series
  |
  v
Forecasting horizon?
  |
  +-- Single-step (1-step ahead) ---------------------> ARIMA, Prophet, tree-based on lag features
  |
  +-- Multi-step (<100 steps) ------------------------> Direct recursive strategy with tree models
  |                                                    TCN, LSTM/GRU
  |
  +-- Long-horizon (>100 steps) ----------------------> Direct multi-output, N-BEATS, TFT
  |                                                    Global models across many series
  |
  v
Number of series?
  |
  +-- Single series -----------------------------------> Classical: ARIMA, ETS, Prophet
  |                                                    ML: Tree-based on lag features
  |
  +-- Few related series (<100) ----------------------> Hierarchical models, vector models (VAR)
  |
  +-- Many series (>100) ----------------------------> Global models (train across all series)
  |                                                    Deep learning: TFT, N-BEATS, DeepAR
  |
  v
Exogenous variables available?
  |
  +-- Yes ---------------------------------------------> Tree models with exogenous features
  |                                                    TFT (handles exogenous + covariates)
  |                                                    Prophet (with regressors)
  |
  +-- No ---------------------------------------------> Univariate methods: ARIMA, ETS, Prophet
                                                        Pure temporal models
```

**Temporal Architecture Choices:**

- **TCN (Temporal Convolutional Networks)**: Parallel training, flexible receptive field, good for long sequences
- **LSTM/GRU**: Classic choice, handles variable length, but slower training
- **Transformers for Time-Series**: Informer, Autoformer, PatchTST - excel with long sequences and many series
- **Classical + ML hybrid**: Use ML models on features extracted from classical methods

**Example: TCN for Time-Series Classification**

```python
import torch
import torch.nn as nn

class TemporalBlock(nn.Module):
    """TCN temporal block with residual connections."""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.relu1, self.dropout1,
            self.conv2, self.relu2, self.dropout2
        )
        self.downsample = nn.Conv1d(
            n_inputs, n_outputs, 1
        ) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNModel(nn.Module):
    """Temporal Convolutional Network for sequence modeling."""
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TCNModel, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size

            layers += [
                TemporalBlock(
                    in_channels, out_channels, kernel_size, stride=1,
                    dilation=dilation_size, padding=padding, dropout=dropout
                )
            ]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
```

#### 3.1.6 Graph/Relational Data

**🟡 INCREMENTAL** - Specialized but increasingly important.

```
Graph Data
  |
  v
Task type?
  |
  +-- Node classification -----------------------------> GraphSAGE, GAT (Graph Attention Network)
  |
  +-- Link prediction --------------------------------> GraphSAGE, SEAL, variational autoencoders
  |
  +-- Graph classification ---------------------------> Graph Isomorphism Network (GIN), DiffPool
  |
  v
Graph size?
  |
  +-- Small (<10K nodes) -----------------------------> Full-batch GNNs
  |
  +-- Medium (10K-1M nodes) -------------------------> GraphSAGE (sampling-based)
  |
  +-- Large (>1M nodes) -----------------------------> Cluster-GCN, GraphSAINT
```

#### 3.1.7 Interpretability-Accuracy Tradeoffs

**HIGH-IMPACT** - Critical for regulated industries and debugging.

**Glass-Box Models (Inherently Interpretable):**

- **Linear/Logistic Regression**: Coefficients directly show feature impact
- **Decision Trees**: Visualizable decision rules
- **GAMs (Generalized Additive Models)**: Partial dependence plots for each feature
- **EBMs (Explainable Boosting Machines)**: Interpretable boosted trees

```python
import interpret.glassbox
from interpret import show

def train_ebm(X_train, y_train, X_test, y_test):
    """Train an Explainable Boosting Machine."""
    ebm = interpret.glassbox.ExplainableBoostingClassifier(
        interactions=10,  # Number of pairwise interactions
        outer_bags=8,      # Ensemble bags
    )
    ebm.fit(X_train, y_train)

    # Global explanations
    ebm_global = ebm.explain_global()
    show(ebm_global)

    # Local explanations
    ebm_local = ebm.explain_local(X_test[:5], y_test[:5])
    show(ebm_local)

    return ebm
```

**Post-Hoc Explanation for Black-Box Models:**

- **SHAP (SHapley Additive exPlanations)**: Game-theoretic, consistent, local explanations
- **LIME**: Local linear approximations
- **Attention visualization**: For transformers and attention-based models

```python
import shap
from xgboost import XGBClassifier

def explain_with_shap(X_train, X_test, model):
    """Generate SHAP explanations for any model."""
    # Create explainer
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)

    # Summary plot
    shap.summary_plot(shap_values, X_test, plot_type="bar")

    # Dependence plot for most important feature
    shap.dependence_plot(
        shap_values.abs.mean(0).argmax(),
        shap_values, X_test
    )

    # Force plot for individual prediction
    shap.force_plot(
        explainer.expected_value,
        shap_values[0,:],
        X_test.iloc[0,:]
    )

    return explainer, shap_values
```

**When Interpretability is Non-Negotiable:**

1. **Healthcare**: Patient diagnosis, treatment recommendations
2. **Finance**: Credit scoring, fraud detection (regulatory requirements)
3. **Legal**: Risk assessment, sentencing recommendations
4. **Debugging**: Understanding model failures, data drift detection

#### 3.1.8 Computational & Latency Constraints

**HIGH-IMPACT** - Production deployment often requires specific performance characteristics.

**Model Size vs Accuracy Pareto Frontier:**

```
Accuracy
  ^
  |        _______________ Large models
  |       /
  |      /
  |     /________________ Medium models
  |    /
  |   /
  |  /___________________ Small models
  | /
  |/_____________________ Tiny models
  +---------------------------------> Model Size (FLOPs/params)
```

**Optimization Techniques:**

1. **Quantization**: FP32 -> INT8 (4x reduction, minimal accuracy loss)
2. **Pruning**: Remove less important weights/neurons
3. **Knowledge Distillation**: Train smaller model to mimic larger model
4. **Architecture Search**: NAS for latency-constrained architectures

**Example: Knowledge Distillation**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    """Knowledge distillation loss combining student loss with teacher logits."""
    def __init__(self, alpha=0.5, temperature=4.0):
        super().__init__()
        self.alpha = alpha  # Balance between hard and soft targets
        self.temperature = temperature

    def forward(self, student_logits, teacher_logits, targets):
        # Hard label loss
        hard_loss = F.cross_entropy(student_logits, targets)

        # Soft label loss (KL divergence)
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            F.softmax(teacher_logits / self.temperature, dim=-1),
            reduction='batchmean'
        ) * (self.temperature ** 2)

        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss


def distill(teacher_model, student_model, train_loader, epochs=10):
    """Train student model using teacher's knowledge."""
    criterion = DistillationLoss(alpha=0.5, temperature=4.0)
    optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-3)

    teacher_model.eval()
    student_model.train()

    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            with torch.no_grad():
                teacher_output = teacher_model(data)

            student_output = student_model(data)
            loss = criterion(student_output, teacher_output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return student_model
```

**Latency Budgeting Rules of Thumb:**

- **Real-time edge (<10ms)**: TinyML approaches, extreme quantization, hardware-specific optimization
- **Interactive (10-100ms)**: Model distillation, pruning, efficient architectures
- **Near real-time (100ms-1s)**: Standard models, mild optimization
- **Batch processing**: Accuracy over latency, largest feasible model

#### 3.1.9 Training Compute Budget

**🟡 INCREMENTAL** - Important for resource-constrained teams.

```
Compute Budget (GPU-hours)
  |
  v
< 10 GPU-hours ------------------------------------> Transfer learning, fine-tuning only
                                                      Small models (distilBERT, MobileNet)
                                                      Pre-computed features

10-100 GPU-hours -----------------------------------> Fine-tune medium models
                                                      Limited hyperparameter search
                                                      Pre-trained embeddings

100-1000 GPU-hours ---------------------------------> Train medium models from scratch
                                                      Extensive hyperparameter tuning
                                                      Data augmentation

> 1000 GPU-hours ------------------------------------> Train large models
                                                      Architecture search
                                                      Extensive data synthesis
```

---

### 3.2 Architecture Selection Common Pitfalls

**HIGH-IMPACT** - Avoid these common mistakes.

1. **Overfitting to test set via architecture selection**: Repeatedly evaluating architectures on test set leaks information. Always use a validation set or nested cross-validation.

2. **Ignoring data size**: Using transformers on small datasets or deep learning on tabular data with few samples usually underperforms simpler methods.

3. **Premature optimization**: Starting with complex architectures before establishing baselines wastes time.

4. **Ignoring production constraints**: Selecting models that cannot meet latency or memory requirements for deployment.

5. **Falling for SOTA claims**: State-of-the-art reported benchmarks often don't translate to your specific problem or data distribution.

6. **Ignoring the no-free-lunch theorem**: No single architecture is best for all problems. Match to your data characteristics.

---

## Part B: Systematic Hyperparameter Tuning

### 3.3 Search Strategies

**HIGH-IMPACT** - The choice of search strategy dramatically impacts efficiency.

#### 3.3.1 Grid Search

**🟡 INCREMENTAL** - Systematic but inefficient.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

def grid_search_baseline(X_train, y_train):
    """Baseline grid search - not recommended for large spaces."""
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)
    return grid_search
```

**When to use:**
- Very small search spaces (<10 total combinations)
- Need to understand the relationship between parameters and performance
- Computational budget is extremely limited and predictability matters

**When to avoid:**
- More than 3-4 hyperparameters
- Continuous parameters with many possible values
- Large search spaces where efficiency matters

#### 3.3.2 Random Search

**HIGH-IMPACT** - Often more efficient than grid search.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

def random_search_baseline(X_train, y_train, n_iter=50):
    """Random search - samples parameter space uniformly."""
    param_distributions = {
        'n_estimators': randint(50, 500),
        'max_depth': randint(3, 20),
        'min_samples_split': uniform(0.01, 0.1),
        'min_samples_leaf': uniform(0.01, 0.05),
        'max_features': uniform(0.1, 0.9)
    }

    random_search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )

    random_search.fit(X_train, y_train)
    return random_search
```

**Why random search beats grid search:**
- In high-dimensional spaces, grid search wastes points on redundant dimensions
- Random search explores more distinct values per hyperparameter
- Better coverage of the search space for the same computational budget

#### 3.3.3 Bayesian Optimization

**HIGH-IMPACT** - The most popular choice for expensive hyperparameter optimization.

**Core Concepts:**
- Builds a probabilistic model (surrogate) of the objective function
- Uses acquisition functions to decide where to sample next
- Balances exploration (uncertain regions) vs exploitation (promising regions)

**TPE (Tree-structured Parzen Estimator)** - Most popular:

```python
import optuna

def objective_tpe(trial, X_train, y_train, X_val, y_val):
    """Optuna objective with TPE sampler for XGBoost."""
    # Define search space
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5)
    }

    model = xgb.XGBClassifier(**params, random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=20, verbose=False)

    # Return metric to maximize (use negative for minimization)
    return model.score(X_val, y_val)


def bayesian_optimization_tpe(X_train, y_train, X_val, y_val, n_trials=100):
    """Bayesian optimization with TPE sampler."""
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    study.optimize(
        lambda trial: objective_tpe(trial, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        n_jobs=1  # TPESampler is sequential by default
    )

    print(f"Best trial: {study.best_trial.params}")
    print(f"Best value: {study.best_value:.4f}")

    return study
```

**Gaussian Process (GP) Optimization:**

```python
def bayesian_optimization_gp(X_train, y_train, X_val, y_val, n_trials=50):
    """Bayesian optimization with Gaussian Process sampler."""
    # GP sampler works well for smaller dimensional spaces (<20 params)
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.GPSampler(seed=42)
    )

    study.optimize(
        lambda trial: objective_tpe(trial, X_train, y_train, X_val, y_val),
        n_trials=n_trials
    )

    return study
```

**TPE vs GP:**
- **TPE**: Better for high-dimensional spaces, conditional parameters, faster
- **GP**: More theoretically sound for small spaces, provides uncertainty estimates

#### 3.3.4 Multi-Fidelity Methods

**HIGH-IMPACT** - Leverage partial training to evaluate configurations faster.

**Key Insight**: We can often identify bad hyperparameter configurations after only a few training iterations, without waiting for full convergence.

**Successive Halving:**

```python
def successive_halving(objective, config_space, R=81, eta=3):
    """
    Successive halving algorithm.

    Args:
        objective: Function that returns score given config and budget
        config_space: Configuration space (list of dicts)
        R: Maximum budget (e.g., training iterations)
        eta: Reduction factor (typically 3 or 4)

    Returns:
        Best configuration found
    """
    import math

    n = len(config_space)
    s_max = int(math.log(R, eta))

    for s in reversed(range(s_max + 1)):
        n_configs = n * (eta ** s) / (R)
        r = R * (eta ** (-s))

        # Get configurations
        if s == s_max:
            configs = config_space[:n_configs]
        else:
            # Select top configs from previous round
            pass

        # Evaluate all configs with budget r
        results = [(c, objective(c, r)) for c in configs]

        # Sort by score and keep top 1/eta
        results.sort(key=lambda x: x[1], reverse=True)
        configs = [c for c, _ in results[:int(n_configs / eta)]]

    return configs[0]  # Return best config
```

**ASHA (Asynchronous Successive Halving Algorithm):**

```python
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

def asha_optimization(config):
    """ASHA-optimized training with Ray Tune."""
    # Access hyperparameters
    lr = config['lr']
    batch_size = config['batch_size']
    hidden_size = config['hidden_size']

    # Initialize model, optimizer, data loaders
    # ...

    for epoch in range(config['max_epochs']):
        # Training code
        train_one_epoch(model, optimizer, train_loader, epoch)

        # Validation
        val_acc = validate(model, val_loader)

        # Report metric for ASHA
        tune.report(mean_accuracy=val_acc, epoch=epoch)


def run_asha_tune():
    """Configure and run ASHA hyperparameter search."""
    scheduler = ASHAScheduler(
        time_attr='epoch',  # Metric to track resource usage
        metric='mean_accuracy',
        mode='max',
        max_t=100,  # Maximum epochs
        grace_period=10,  # Minimum epochs before pruning
        reduction_factor=3,  # Fraction of configs to keep each round
        brackets=1  # Number of simultaneous SHAs
    )

    tuner = tune.Tuner(
        asha_optimization,
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=50,  # Number of trials
        ),
        run_config=tune.RunConfig(
            name='asha_experiment',
            resources_per_trial={'cpu': 2, 'gpu': 0.5}
        ),
        param_space={
            'lr': tune.loguniform(1e-4, 1e-1),
            'batch_size': tune.choice([32, 64, 128]),
            'hidden_size': tune.choice([64, 128, 256, 512]),
            'max_epochs': 100
        }
    )

    results = tuner.fit()
    return results
```

**HyperBand:**

```python
from ray.tune.schedulers import HyperBandScheduler
from ray.tune.schedulers.impl.hyperband import HyperBandScheduler

def hyperband_optimization():
    """HyperBand scheduler - multi-bracket successive halving."""
    scheduler = HyperBandScheduler(
        time_attr='training_iteration',
        metric='mean_accuracy',
        mode='max',
        max_t=100,  # Maximum iterations per trial
        reduction_factor=4
    )

    tuner = tune.Tuner(
        trainable,
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=100
        ),
        run_config=tune.RunConfig(
            name='hyperband_experiment'
        )
    )

    results = tuner.fit()
    return results
```

**Optuna Hyperband Implementation:**

```python
def hyperband_optuna_pruning():
    """HyperBand pruning with Optuna."""
    import optuna

    def objective_with_pruning(trial):
        # Define hyperparameters
        lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
        hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256])

        model = create_model(hidden_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Training loop with pruning
        for epoch in range(100):
            train_one_epoch(model, optimizer, train_loader)
            val_loss = validate(model, val_loader)

            # Report intermediate value for pruning
            trial.report(val_loss, epoch)

            # Prune trial if it's unpromising
            if trial.should_prune():
                raise optuna.TrialPruned()

        return val_loss

    # Create study with HyperBand pruner
    study = optuna.create_study(
        direction='minimize',
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=1,
            max_resource=100,
            reduction_factor=3
        )
    )

    study.optimize(objective_with_pruning, n_trials=50)
    return study
```

#### 3.3.5 Advanced Methods

**🟡 INCREMENTAL** - Advanced techniques for specific scenarios.

**BOHB (Bayesian Optimization HyperBand):**

Combines TPE with HyperBand for efficient multi-fidelity optimization.

```python
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB

def bohb_optimization():
    """BOHB: TPE search algorithm + HyperBand scheduler."""
    # BOHB search algorithm
    bohb = TuneBOHB()

    # HyperBand scheduler tuned for BOHB
    scheduler = HyperBandForBOHB(
        time_attr='training_iteration',
        max_t=100,
        reduction_factor=4,
        stop_last_trials=False
    )

    # Limit concurrency for BOHB
    bohb = tune.search.ConcurrencyLimiter(bohb, max_concurrent=4)

    tuner = tune.Tuner(
        trainable,
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            search_alg=bohb,
            num_samples=50
        ),
        param_space={
            'lr': tune.loguniform(1e-5, 1e-1),
            'momentum': tune.uniform(0.1, 0.9),
            'hidden_size': tune.choice([32, 64, 128, 256])
        }
    )

    results = tuner.fit()
    return results
```

**Population Based Training (PBT):**

PBT trains a population of models in parallel, periodically:
1. Evaluating performance
2. Exploiting by copying weights from better models
3. Exploring by perturbing hyperparameters

```python
from ray.tune.schedulers import PopulationBasedTraining

def pbt_optimization():
    """Population Based Training for hyperparameter optimization."""
    pbt_scheduler = PopulationBasedTraining(
        time_attr='training_iteration',
        metric='mean_accuracy',
        mode='max',
        perturbation_interval=10,  # Steps between exploitation/exploration
        hyperparam_mutations={
            # Hyperparameters to explore and their mutation ranges
            'lr': lambda: tune.loguniform(1e-5, 1e-1),
            'momentum': lambda: tune.uniform(0.1, 0.9),
            'batch_size': [32, 64, 128, 256]
        }
    )

    tuner = tune.Tuner(
        trainable,
        tune_config=tune.TuneConfig(
            scheduler=pbt_scheduler,
            num_samples=8,  # Population size
        ),
        param_space={
            'lr': 0.01,  # Initial learning rate
            'momentum': 0.9,  # Initial momentum
            'batch_size': 64  # Initial batch size
        }
    )

    results = tuner.fit()
    return results
```

**Multi-Objective Optimization:**

Optimize multiple competing objectives (e.g., accuracy vs latency).

```python
def multi_objective_optimization():
    """Pareto optimization for accuracy vs model size."""
    def objective(trial):
        # Hyperparameters
        hidden_size = trial.suggest_int('hidden_size', 32, 512)
        num_layers = trial.suggest_int('num_layers', 1, 5)
        dropout = trial.suggest_float('dropout', 0.0, 0.5)

        # Create and train model
        model = create_model(hidden_size, num_layers, dropout)
        train_model(model)

        # Calculate metrics
        accuracy = evaluate_accuracy(model)
        num_params = count_parameters(model)
        latency = measure_inference_latency(model)

        # Return multiple objectives
        return accuracy, num_params, latency

    # Create multi-objective study
    study = optuna.create_study(
        directions=['maximize', 'minimize', 'minimize']  # max accuracy, min params, min latency
    )

    study.optimize(objective, n_trials=100)

    # Analyze Pareto front
    print("Number of trials on Pareto front: ", len(study.best_trials))

    # Plot Pareto front
    fig = optuna.visualization.plot_pareto_front(study)
    fig.show()

    return study
```

---

### 3.4 Practical Workflows

**HIGH-IMPACT** - Structured approaches to efficient tuning.

#### 3.4.1 What to Tune First

Prioritize hyperparameters by their impact:

```
High Impact (Tune First):
  |
  +-- Learning rate (most critical for deep learning)
  +-- Number of estimators/trees (ensemble methods)
  +-- Max depth (tree-based models)
  +-- Regularization strength (L1/L2)
  +-- Batch size (deep learning)
  +-- Number of layers / hidden size (architecture)

Medium Impact:
  |
  +-- Subsample ratio (bagging)
  +-- Learning rate schedule
  +-- Optimizer type
  +-- Activation function

Low Impact (Tune Last):
  |
  +-- Random seed
  +-- Specific initialization method
  +-- Minor optimizer parameters (beta1, beta2, epsilon)
```

#### 3.4.2 Budget Allocation

**Rule of thumb: 60/40 split**

- **60%** of budget on architecture selection
- **40%** of budget on hyperparameter tuning

Or, if using transfer learning:
- **30%** model selection
- **50%** hyperparameter tuning
- **20%** architecture adaptation (head design, layer freezing)

#### 3.4.3 Warm-Starting

Leverage previous runs to speed up optimization:

```python
def warm_start_optuna(previous_study_path):
    """Warm-start optimization from previous study."""
    # Load previous study
    previous_study = joblib.load(previous_study_path)

    # Create new study with trials from previous study
    study = optuna.create_study(
        study_name='warm_start_study',
        direction='maximize',
        load_if_exists=True
    )

    # Add previous trials
    for trial in previous_study.trials:
        study.add_trial(trial)

    # Continue optimization
    study.optimize(objective, n_trials=50)

    return study
```

#### 3.4.4 Multi-Phase Optimization Strategy

```python
class MultiPhaseOptimizer:
    """Structured multi-phase hyperparameter optimization."""

    def __init__(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

    def phase1_coarse_search(self):
        """Phase 1: Coarse random search over wide ranges."""
        study = optuna.create_study(direction='maximize')

        # Wide ranges
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1.0, log=True),
            }
            model = xgb.XGBClassifier(**params)
            model.fit(self.X_train, self.y_train)
            return model.score(self.X_val, self.y_val)

        study.optimize(objective, n_trials=50, n_jobs=4)
        return study

    def phase2_refined_search(self, best_params):
        """Phase 2: Refined search around best config."""
        study = optuna.create_study(direction='maximize')

        # Narrower ranges around best params
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int(
                    'n_estimators',
                    max(50, best_params['n_estimators'] - 100),
                    best_params['n_estimators'] + 100
                ),
                'max_depth': trial.suggest_int(
                    'max_depth',
                    max(3, best_params['max_depth'] - 3),
                    best_params['max_depth'] + 3
                ),
                'learning_rate': trial.suggest_float(
                    'learning_rate',
                    best_params['learning_rate'] / 3,
                    best_params['learning_rate'] * 3,
                    log=True
                ),
                # Add new hyperparameters
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            }
            model = xgb.XGBClassifier(**params)
            model.fit(self.X_train, self.y_train)
            return model.score(self.X_val, self.y_val)

        study.optimize(objective, n_trials=100)
        return study

    def phase3_final_polish(self, best_params):
        """Phase 3: Fine-grained tuning with pruning."""
        def objective(trial):
            params = {
                'n_estimators': best_params['n_estimators'],
                'max_depth': best_params['max_depth'],
                'learning_rate': best_params['learning_rate'],
                'min_child_weight': best_params['min_child_weight'],
                'subsample': best_params['subsample'],
                # Fine-tune regularization
                'reg_alpha': trial.suggest_float('reg_alpha', 0, best_params['reg_alpha'] * 2),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, best_params['reg_lambda'] * 2),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            }

            model = xgb.XGBClassifier(**params, early_stopping_rounds=20)
            model.fit(
                self.X_train, self.y_train,
                eval_set=[(self.X_val, self.y_val)],
                verbose=False
            )
            return model.score(self.X_val, self.y_val)

        study = optuna.create_study(
            direction='maximize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        study.optimize(objective, n_trials=50)
        return study
```

#### 3.4.5 Distributed Hyperparameter Tuning

```python
def distributed_optuna_optimization():
    """Distributed hyperparameter optimization with Optuna."""
    import optuna

    # Create study with persistent storage for distributed optimization
    study = optuna.create_study(
        study_name='distributed_study',
        storage='postgresql://user:password@host:port/dbname',  # Or sqlite:///db.sqlite3
        direction='maximize',
        load_if_exists=True
    )

    # Define objective
    def objective(trial):
        # Define search space
        params = {...}
        # Training code
        score = train_and_evaluate(params)
        return score

    # Run optimization (can be run in parallel on multiple machines)
    study.optimize(objective, n_trials=1000)

    return study
```

---

### 3.5 Tooling Guide

**HIGH-IMPACT** - Choose the right tool for your needs.

#### 3.5.1 Optuna

**Strengths:**
- Define-by-run API (dynamic search spaces)
- Efficient samplers (TPE, CMA-ES, GP, Random)
- Advanced pruning (MedianPruner, HyperbandPruner, SuccessiveHalvingPruner)
- Multi-objective optimization
- Excellent visualization
- Easy distributed optimization

**Best for:**
- Python projects
- Complex search spaces with conditional parameters
- Multi-objective optimization
- Research and experimentation

**Advanced Optuna Features:**

```python
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler, CmaEsSampler

# Custom pruning callback
class CustomPruner(optuna.pruners.BasePruner):
    def __init__(self, threshold=0.1):
        self.threshold = threshold

    def prune(self, study, trial, trial_id):
        # Custom pruning logic
        values = trial.intermediate_values
        if len(values) > 10:
            recent_values = list(values.values())[-10:]
            if max(recent_values) < self.threshold:
                return True
        return False

# Conditional parameter spaces
def objective_with_conditionals(trial):
    classifier_name = trial.suggest_categorical('classifier', ['SVM', 'RandomForest'])

    if classifier_name == 'SVM':
        svm_c = trial.suggest_float('svm_c', 1e-10, 1e10, log=True)
        classifier = sklearn.svm.SVC(C=svm_c)
    else:
        rf_max_depth = trial.suggest_int('rf_max_depth', 2, 32)
        classifier = sklearn.ensemble.RandomForestClassifier(max_depth=rf_max_depth)

    # ... rest of training

# Visualization
def analyze_study(study):
    # Optimization history
    fig = optuna.visualization.plot_optimization_history(study)
    fig.show()

    # Parameter importance
    fig = optuna.visualization.plot_param_importances(study)
    fig.show()

    # Parameter relationships
    fig = optuna.visualization.plot_parallel_coordinate(study)
    fig.show()

    # Slice plot for single parameter
    fig = optuna.visualization.plot_slice(study, params=['learning_rate'])
    fig.show()
```

#### 3.5.2 Ray Tune

**Strengths:**
- Excellent for distributed training
- Built-in schedulers (ASHA, HyperBand, PBT)
- Seamless integration with ML frameworks
- Scalable to large clusters
- Rich result analysis tools

**Best for:**
- Large-scale hyperparameter searches
- Distributed training across multiple machines
- Deep learning projects
- Projects already using Ray

```python
from ray import tune
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCallback
import pytorch_lightning as pl

# Ray Tune with PyTorch Lightning
def lightning_trainable(config):
    """Trainable function for PyTorch Lightning with Ray Tune."""
    model = LightningModel(config)
    trainer = pl.Trainer(
        max_epochs=config['max_epochs'],
        callbacks=[
            TuneReportCallback(
                metrics={'val_loss': 'val_loss', 'val_acc': 'val_acc'},
                on='validation_end'
            )
        ]
    )
    trainer.fit(model)
    return trainer.callback_metrics

def run_ray_tune_lightning():
    """Configure Ray Tune for PyTorch Lightning."""
    reporter = CLIReporter(
        metric_columns=['val_loss', 'val_acc', 'training_iteration'],
        max_report_frequency=30  # Report every 30s
    )

    scheduler = ASHAScheduler(
        max_t=100,
        grace_period=10,
        reduction_factor=3
    )

    tuner = tune.Tuner(
        lightning_trainable,
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=100
        ),
        run_config=tune.RunConfig(
            progress_reporter=reporter,
            local_dir='./ray_results'
        ),
        param_space={
            'learning_rate': tune.loguniform(1e-4, 1e-1),
            'batch_size': tune.choice([32, 64, 128]),
            'hidden_dim': tune.choice([64, 128, 256]),
            'max_epochs': 100
        }
    )

    results = tuner.fit()
    return results
```

#### 3.5.3 Weights & Biases Sweeps

**Strengths:**
- Cloud-based UI for experiment tracking
- Easy hyperparameter sweeps
- Integration with many frameworks
- Excellent visualization and comparison

**Best for:**
- Teams needing experiment tracking
- Cloud-based optimization
- Collaborative ML projects

```python
import wandb

# Define sweep configuration
sweep_config = {
    'method': 'bayes',  # Can be 'grid', 'random', 'bayes'
    'metric': {'name': 'val_acc', 'goal': 'maximize'},
    'parameters': {
        'learning_rate': {'min': 1e-5, 'max': 1e-1},
        'batch_size': {'values': [32, 64, 128]},
        'hidden_dim': {'values': [64, 128, 256]},
        'dropout': {'min': 0.0, 'max': 0.5}
    }
}

# Initialize sweep
sweep_id = wandb.sweep(sweep_config, project='my_project')

# Define training function
def train():
    with wandb.init() as run:
        config = wandb.config
        # Access hyperparameters
        lr = config.learning_rate
        batch_size = config.batch_size

        # Training loop
        for epoch in range(epochs):
            train_one_epoch(model, optimizer, train_loader)
            val_acc = validate(model, val_loader)

            # Log metrics
            wandb.log({'epoch': epoch, 'val_acc': val_acc})

# Run sweep
wandb.agent(sweep_id, train, count=50)
```

#### 3.5.4 Tool Selection Guide

```
Small project, single machine
  |
  +-- Python -----------------------------------------> Optuna (simplest, powerful)
  +-- Non-Python ------------------------------------> Hyperopt, SMAC3

Large project, distributed
  |
  +-- Deep learning -----------------------------------> Ray Tune (best scalability)
  +-- General ML ------------------------------------> Optuna with distributed storage

Team collaboration needed
  |
  +-- Experiment tracking priority --------------------> W&B Sweeps
  +-- Custom infrastructure --------------------------> Optuna + custom dashboard

Research and experimentation
  |
  +-- Flexible search spaces -------------------------> Optuna (define-by-run)
  +-- Multi-objective --------------------------------> Optuna (best support)
```

---

### 3.6 Common Pitfalls in Hyperparameter Tuning

**HIGH-IMPACT** - Avoid these mistakes.

1. **Tuning on test set**: Never use test data for hyperparameter selection. Always use validation set or cross-validation.

2. **Data leakage in preprocessing**: Fitting preprocessors (scalers, encoders) on full dataset before splitting leaks information.

3. **Overfitting validation set**: With extensive hyperparameter search, you effectively train on the validation set. Use nested cross-validation or hold-out test set.

4. **Ignoring reproducibility**: Not setting random seeds leads to irreproducible results.

5. **Insufficient budget per configuration**: Running too few epochs prevents proper evaluation.

6. **Tuning too many hyperparameters**: Focus on high-impact parameters first; tuning >10 parameters is rarely effective.

7. **Not monitoring optimization**: Without proper logging and visualization, you can't detect problems early.

8. **Early stopping too aggressive**: Stops optimization before convergence; set appropriate patience.

9. **Ignoring computational cost**: Tuning without considering training time leads to impractical solutions.

10. **Forgetting baseline comparison**: Always compare tuned model against a simple baseline.

**Example: Proper Cross-Validation with Hyperparameter Tuning**

```python
from sklearn.model_selection import KFold
import optuna

def cv_objective(trial, X, y, n_folds=5):
    """Objective function with proper cross-validation."""
    # Define hyperparameters
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
    }

    # K-fold cross-validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_scores = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = xgb.XGBClassifier(**params, random_state=42)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=20,
            verbose=False
        )
        score = model.score(X_val, y_val)
        cv_scores.append(score)

    return np.mean(cv_scores)


# Nested CV for unbiased evaluation
def nested_cv_evaluation(X, y):
    """Nested cross-validation for unbiased hyperparameter tuning."""
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    outer_scores = []

    for train_idx, test_idx in outer_cv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Inner CV for hyperparameter tuning
        study = optuna.create_study(direction='maximize')
        study.optimize(
            lambda trial: cv_objective(trial, X_train, y_train),
            n_trials=50
        )

        # Train final model on full training data
        best_model = xgb.XGBClassifier(**study.best_params, random_state=42)
        best_model.fit(X_train, y_train)

        # Evaluate on held-out test fold
        test_score = best_model.score(X_test, y_test)
        outer_scores.append(test_score)

    return np.mean(outer_scores), np.std(outer_scores)
```

---

## Summary Checklist

### Model Architecture Selection

- [ ] Identify primary data modality (tabular, image, text, time-series, graph)
- [ ] Determine sample size and feature characteristics
- [ ] Assess interpretability requirements
- [ ] Define latency and computational constraints
- [ ] Establish simple baseline (linear model or simple ensemble)
- [ ] Select appropriate model family using decision framework
- [ ] Validate choice against production constraints

### Hyperparameter Tuning

- [ ] Set up proper train/validation/test splits
- [ ] Implement cross-validation for robust evaluation
- [ ] Choose search strategy based on budget and problem size
- [ ] Prioritize high-impact hyperparameters
- [ ] Configure multi-fidelity optimization if applicable
- [ ] Set appropriate stopping criteria (early stopping, timeout)
- [ ] Monitor optimization progress with visualization
- [ ] Document best hyperparameters and rationale

### Tooling

- [ ] Select tool based on project scale and requirements
- [ ] Configure distributed optimization if needed
- [ ] Set up experiment tracking and logging
- [ ] Implement reproducibility (random seeds, version control)

---

## Key Takeaways

1. **Match architecture to data characteristics**: Gradient boosting excels on tabular data, CNNs on images, transformers on text, and specialized architectures for time-series and graphs.

2. **Consider constraints early**: Interpretability, latency, and computational requirements significantly impact viable architectures.

3. **Use systematic search strategies**: Move from grid/random search to Bayesian optimization for expensive functions.

4. **Leverage multi-fidelity methods**: ASHA, HyperBand, and pruning dramatically reduce tuning time for deep learning.

5. **Avoid common pitfalls**: Never tune on test set, establish proper validation, and compare against baselines.

6. **Choose tools wisely**: Optuna for flexibility, Ray Tune for scalability, W&B for collaboration.

7. **Iterate intelligently**: Start with coarse search, refine around promising configurations, and finalize with fine-grained tuning.
