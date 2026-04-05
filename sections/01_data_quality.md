# Data Quality & Preprocessing

> "Garbage in, garbage out" remains the most fundamental truth in machine learning. No amount of sophisticated architecture or clever hyperparameter tuning can compensate for poor data quality or improper preprocessing. This section provides a systematic, actionable guide to ensuring your data is model-ready.

---

## 1. Data Profiling & Assessment

**Impact:** 🔴 HIGH-IMPACT

Before building models, you must understand your data. Data profiling is the systematic process of examining data source content and structure, collecting statistical information, and identifying quality issues.

### 1.1 Essential Profiling Components

| Component | What to Check | Why It Matters |
|-----------|---------------|----------------|
| **Data Types** | Incorrect dtypes (numeric stored as string) | Prevents proper feature engineering |
| **Missing Values** | Patterns, percentages, mechanisms | Determines imputation strategy |
| **Cardinality** | Unique values per column | High cardinality impacts encoding choice |
| **Distributions** | Skewness, kurtosis, outliers | Affects model assumptions and scaling |
| **Correlations** | Feature-feature, feature-target | Redundancy detection, leakage risks |
| **Value Ranges** | Min/max, impossible values | Domain validation (e.g., negative prices) |
| **Temporal Patterns** | Date ranges, gaps, seasonality | Critical for time series integrity |

### 1.2 Automated Profiling Pipeline

```python
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any

def comprehensive_profile(df: pd.DataFrame, target: str = None) -> Dict[str, Any]:
    """
    Generate comprehensive data quality report.
    Returns dictionary with quality metrics and issues.
    """
    profile = {
        'shape': df.shape,
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
        'columns': {},
        'issues': []
    }
    
    for col in df.columns:
        col_info = {
            'dtype': str(df[col].dtype),
            'non_null_count': df[col].notna().sum(),
            'null_count': df[col].isna().sum(),
            'null_pct': df[col].isna().mean() * 100,
            'unique_count': df[col].nunique(),
            'unique_pct': df[col].nunique() / len(df) * 100,
        }
        
        # Numeric-specific metrics
        if pd.api.types.is_numeric_dtype(df[col]):
            col_info.update({
                'min': df[col].min(),
                'max': df[col].max(),
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'skewness': stats.skew(df[col].dropna()),
                'kurtosis': stats.kurtosis(df[col].dropna()),
                'zeros': (df[col] == 0).sum(),
                'negatives': (df[col] < 0).sum() if df[col].min() >= 0 else None
            })
            
            # Detect potential outliers using IQR method
            Q1, Q3 = df[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            outliers = ((df[col] < lower) | (df[col] > upper)).sum()
            col_info['outlier_count'] = outliers
            col_info['outlier_pct'] = outliers / len(df) * 100
            
        # Categorical-specific metrics
        else:
            col_info.update({
                'most_common': df[col].value_counts().head(5).to_dict(),
                'rare_categories': (df[col].value_counts() / len(df) < 0.01).sum()
            })
            
            # Check for potential ID/constant columns
            if col_info['unique_pct'] > 95:
                profile['issues'].append(f"{col}: High cardinality (>95%) - potential ID column")
            if col_info['unique_count'] == 1:
                profile['issues'].append(f"{col}: Constant column (zero variance)")
        
        # Missing value patterns
        if col_info['null_pct'] > 50:
            profile['issues'].append(f"{col}: High missingness (>50%)")
        elif 0 < col_info['null_pct'] < 5:
            profile['issues'].append(f"{col}: Low missingness (<5%) - review imputation")
            
        profile['columns'][col] = col_info
    
    # Target-specific checks
    if target and target in df.columns:
        if pd.api.types.is_numeric_dtype(df[target]):
            target_skew = stats.skew(df[target].dropna())
            if abs(target_skew) > 1:
                profile['issues'].append(f"{target}: Highly skewed ({target_skew:.2f}) - consider transformation")
        
        # Classification: check class imbalance
        elif df[target].nunique() < 20:  # Reasonable threshold for classification
            class_dist = df[target].value_counts(normalize=True)
            min_class_pct = class_dist.min() * 100
            if min_class_pct < 5:
                profile['issues'].append(f"{target}: Severe class imbalance (<5% in minority class)")
            elif min_class_pct < 15:
                profile['issues'].append(f"{target}: Moderate class imbalance (<15% in minority class)")
    
    return profile


# Usage example:
# profile = comprehensive_profile(train_df, target='churn_flag')
# for issue in profile['issues']:
#     print(f"⚠️  {issue}")
```

### 1.3 Missing Value Mechanism Assessment

Understanding *why* data is missing determines the appropriate handling strategy:

| Mechanism | Definition | Detection | Recommended Approach |
|-----------|------------|-----------|---------------------|
| **MCAR** (Missing Completely At Random) | Missingness unrelated to any variable | Little's MCAR test (p > 0.05) | Any imputation method works |
| **MAR** (Missing At Random) | Missingness related to observed variables | Patterns in other columns | Include missing indicators, advanced imputation |
| **MNAR** (Missing Not At Random) | Missingness related to unobserved values itself | Domain knowledge, sensitivity analysis | Model missingness mechanism, create indicators |

```python
from scipy import stats

def littles_mcar_test(df: pd.DataFrame) -> tuple:
    """
    Perform Little's MCAR test for missingness pattern.
    Returns: (chi2_statistic, p_value)
    Note: Simplified implementation - use `missingno` or `pyampute` for full version.
    """
    # Count missing patterns
    missing_patterns = df.isna().value_counts(normalize=True)
    n_patterns = len(missing_patterns)
    
    # If only one pattern (all present or all missing), trivially MCAR
    if n_patterns == 1:
        return 0, 1.0
    
    # Simplified heuristic: check if missingness correlates across variables
    missing_matrix = df.isna().astype(int)
    correlation_matrix = missing_matrix.corr()
    mean_abs_corr = correlation_matrix.abs().mean().mean()
    
    # Low correlation suggests MCAR
    # This is a heuristic - proper test requires EM algorithm
    return mean_abs_corr, "Low correlation suggests MCAR" if mean_abs_corr < 0.1 else "Non-random missingness"
```

### Common Pitfalls

- **Pitfall:** Skipping profiling on "clean" datasets. Even well-maintained data has hidden issues.
  - **Diagnostic:** Always run automated profiling, even on trusted sources.
  - **Frequency:** Re-profile after any join or transformation operation.

- **Pitfall:** Ignoring data drift between train and production.
  - **Diagnostic:** Compare distributions using KS test, PSI (Population Stability Index).
  - **Action:** Monitor drift continuously; set up alerts for significant shifts.

```python
def compute_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """
    Compute Population Stability Index for drift detection.
    PSI < 0.1: No significant change
    PSI 0.1-0.2: Moderate change
    PSI > 0.2: Significant change - investigate
    """
    breakpoints = np.linspace(0, 1, buckets + 1)
    expected_percents = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    
    # Avoid division by zero
    expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
    actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
    
    psi = np.sum((expected_percents - actual_percents) * np.log(expected_percents / actual_percents))
    return psi
```

---

## 2. Missing Value Strategies

**Impact:** 🔴 HIGH-IMPACT

Missing data is inevitable. The key is selecting an appropriate strategy based on the missingness mechanism, data size, and problem context.

### 2.1 Decision Framework: Delete vs Impute vs Flag

```
Should I delete missing data?
├── Missingness > 60% for a feature?
│   └── YES → Delete feature (unless MNAR and informative)
├── Missingness < 5% overall?
│   └── MAYBE → Consider deletion if MCAR and sufficient data
└── Missingness 5-60%?
    └── NO → Impute + Flag

Should I flag missingness?
├── Is missingness potentially informative (MAR/MNAR)?
│   └── YES → Always add missing indicator
└── MCAR confirmed?
    └── MAYBE → Add indicator if dataset is small or missingness >10%
```

### 2.2 Imputation Methods Comparison

| Method | Best For | Pros | Cons | Computational Cost |
|--------|----------|------|------|-------------------|
| **Mean/Median/Mode** | MCAR, small missingness, fast baseline | Simple, fast, works on any size | Distorts distributions, ignores correlations | ⚡ Low |
| **KNN Imputation** | Small-medium datasets, MAR | Preserves local structure, handles mixed types | Slow on large data, sensitive to outliers | ⚡⚡ Medium |
| **Iterative (MICE)** | MAR, medium datasets, feature relationships | Uses feature correlations, realistic estimates | Slower, assumes linear relationships | ⚡⚡⚡ High |
| **Deep Learning (GAIN/DataWig)** | Large datasets, complex patterns | Captures non-linear relationships | Requires tuning, computationally expensive | ⚡⚡⚡⚡ Very High |
| **Domain-Specific** | Time series, geospatial, etc. | Respects data structure | Requires custom implementation | Variable |

### 2.3 Implementation: Scikit-learn Imputers

```python
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

def create_imputation_pipeline(
    numeric_strategy: str = 'median',
    numeric_indicator: bool = True,
    k_neighbors: int = 5,
    max_iter: int = 10
) -> ColumnTransformer:
    """
    Create comprehensive imputation pipeline with missing indicators.
    
    Args:
        numeric_strategy: 'mean', 'median', 'constant', or 'knn' or 'iterative'
        numeric_indicator: Add binary columns indicating missingness
        k_neighbors: For KNN imputation
        max_iter: For iterative imputation
    """
    
    if numeric_strategy == 'knn':
        numeric_imputer = KNNImputer(n_neighbors=k_neighbors, add_indicator=numeric_indicator)
    elif numeric_strategy == 'iterative':
        numeric_imputer = IterativeImputer(
            max_iter=max_iter,
            random_state=42,
            add_indicator=numeric_indicator
        )
    else:
        numeric_imputer = SimpleImputer(
            strategy=numeric_strategy,
            add_indicator=numeric_indicator
        )
    
    # Categorical: most frequent with indicator
    categorical_imputer = SimpleImputer(
        strategy='most_frequent',
        add_indicator=True
    )
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', numeric_imputer, selector(dtype_include='number')),
            ('categorical', categorical_imputer, selector(dtype_include='category')),
        ],
        remainder='passthrough'
    )
    
    return preprocessor


def selector(dtype_include):
    """Helper to select columns by dtype."""
    from sklearn.compose import make_column_selector
    return make_column_selector(dtype_include=dtype_include)


# Full pipeline with imputation + scaling
full_pipeline = Pipeline([
    ('imputation', create_imputation_pipeline(numeric_strategy='iterative')),
    ('scaler', StandardScaler())
])

# Usage
# X_train_imputed = full_pipeline.fit_transform(X_train)
# X_test_imputed = full_pipeline.transform(X_test)
```

### 2.4 Advanced: Custom Imputation by Pattern

When missingness follows patterns (e.g., surveys where certain questions are skipped conditionally), custom imputation outperforms generic methods.

```python
import pandas as pd
from typing import Dict, List

class PatternBasedImputer:
    """
    Impute missing values based on conditional patterns.
    Example: If 'employment_status' = 'student', impute 'income' as student_median.
    """
    
    def __init__(self, imputation_rules: Dict[str, List[str]]):
        """
        Args:
            imputation_rules: {target_col: [condition_cols]}
                e.g., {'income': ['employment_status', 'age_group']}
        """
        self.rules = imputation_rules
        self.reference_values = {}
    
    def fit(self, X: pd.DataFrame, y=None):
        """Store reference values for each pattern combination."""
        self.reference_values = {}
        
        for target, conditions in self.rules.items():
            if target not in X.columns:
                continue
                
            # Get non-null values for each pattern
            valid_data = X[conditions + [target]].dropna(subset=[target])
            
            # Group by conditions and compute median
            grouped = valid_data.groupby(conditions)[target].median()
            self.reference_values[target] = grouped
            
            # Store overall median as fallback
            self.reference_values[f'{target}_overall'] = X[target].median()
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply pattern-based imputation."""
        X = X.copy()
        
        for target, conditions in self.rules.items():
            if target not in X.columns:
                continue
                
            # Find rows with missing target
            missing_mask = X[target].isna()
            
            if not missing_mask.any():
                continue
            
            # Get conditions for missing rows
            missing_rows = X.loc[missing_mask, conditions]
            
            # Map to reference values
            for idx, row in missing_rows.iterrows():
                key = tuple(row.values)
                if len(conditions) == 1:
                    key = key[0]
                
                imputed_value = self.reference_values[target].get(
                    key,
                    self.reference_values[f'{target}_overall']
                )
                X.loc[idx, target] = imputed_value
        
        return X


# Usage example:
# imputation_rules = {
#     'income': ['employment_status', 'region'],
#     'purchase_amount': ['customer_segment']
# }
# imputer = PatternBasedImputer(imputation_rules)
# imputer.fit(train_df)
# train_df_imputed = imputer.transform(train_df)
```

### Common Pitfalls

- **Pitfall:** Imputing before train-test split.
  - **Consequence:** Test information leaks into training via global statistics.
  - **Solution:** Always fit imputer on training data only.

- **Pitfall:** Using mean imputation on skewed distributions.
  - **Consequence:** Creates unrealistic values, distorts relationships.
  - **Solution:** Use median for skewed data, or transform first.

- **Pitfall:** Not adding missing indicators for MAR/MNAR.
  - **Consequence:** Loses predictive signal from missingness pattern itself.
  - **Solution:** Always add indicators when missingness >5% or potentially informative.

```python
# Diagnostic: Check if missingness is predictive
def test_missingness_predictiveness(df: pd.DataFrame, col: str, target: str) -> float:
    """
    Test if missingness in `col` correlates with `target`.
    Returns correlation coefficient (for binary classification: point-biserial).
    """
    missing_indicator = df[col].isna().astype(int)
    
    if df[target].dtype in [int, float]:
        return df[target].corr(missing_indicator)
    else:
        # For classification: compare target distribution
        target_mean_when_missing = df.loc[df[col].isna(), target].mean()
        target_mean_when_present = df.loc[df[col].notna(), target].mean()
        return abs(target_mean_when_missing - target_mean_when_present)
```

---

## 3. Outlier Detection & Treatment

**Impact:** 🟡 INCREMENTAL (High for specific domains like fraud detection, manufacturing)

Outliers can represent rare but valid events (fraud, network intrusions) or data errors. Treatment depends on domain context and model type.

### 3.1 Detection Methods Comparison

| Method | Best For | Sensitivity | Robustness to Noise | Scalability |
|--------|----------|-------------|---------------------|-------------|
| **Z-Score** | Univariate, Gaussian data | Low-medium | Low | ⚡⚡⚡ High |
| **IQR Method** | Univariate, any distribution | Medium | High | ⚡⚡⚡ High |
| **Isolation Forest** | Multivariate, high-dimensional | High (tunable) | Medium | ⚡⚡ Medium |
| **Local Outlier Factor (LOF)** | Local density outliers | High | Medium | ⚡ Low |
| **DBSCAN** | Clustering-based, spatial | Medium-high | High | ⚡ Low |
| **One-Class SVM** | Novelty detection, small datasets | High | Low | ⚡⚡ Medium |
| **Autoencoder** | High-dimensional, complex patterns | High | Medium | ⚡ Low |

### 3.2 Implementation: Ensemble Outlier Detection

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from scipy import stats

class EnsembleOutlierDetector:
    """
    Combine multiple outlier detection methods for robust identification.
    Considers a point an outlier if majority of methods agree.
    """
    
    def __init__(
        self,
        contamination: float = 0.05,
        use_isolation_forest: bool = True,
        use_lof: bool = True,
        use_zscore: bool = True,
        use_iqr: bool = True,
        zscore_threshold: float = 3.0,
        iqr_multiplier: float = 1.5
    ):
        self.contamination = contamination
        self.use_isolation_forest = use_isolation_forest
        self.use_lof = use_lof
        self.use_zscore = use_zscore
        self.use_iqr = use_iqr
        self.zscore_threshold = zscore_threshold
        self.iqr_multiplier = iqr_multiplier
        self.scaler = StandardScaler()
    
    def fit_detect(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit detectors and return outlier scores and flags."""
        X_scaled = self.scaler.fit_transform(X.select_dtypes(include=[np.number]))
        results = pd.DataFrame(index=X.index)
        outlier_votes = np.zeros(len(X))
        
        # Isolation Forest
        if self.use_isolation_forest:
            iso_forest = IsolationForest(
                contamination=self.contamination,
                random_state=42
            )
            iso_pred = iso_forest.fit_predict(X_scaled)
            results['isolation_forest_score'] = iso_forest.score_samples(X_scaled)
            results['isolation_forest_outlier'] = iso_pred == -1
            outlier_votes += (iso_pred == -1).astype(int)
        
        # Local Outlier Factor
        if self.use_lof:
            lof = LocalOutlierFactor(
                contamination=self.contamination,
                novelty=True
            )
            lof.fit(X_scaled)
            lof_pred = lof.predict(X_scaled)
            results['lof_score'] = lof.score_samples(X_scaled)
            results['lof_outlier'] = lof_pred == -1
            outlier_votes += (lof_pred == -1).astype(int)
        
        # Z-Score (univariate, per column)
        if self.use_zscore:
            z_scores = np.abs(stats.zscore(X_scaled, nan_policy='omit'))
            z_outliers = (z_scores > self.zscore_threshold).any(axis=1)
            results['max_zscore'] = z_scores.max(axis=1)
            results['zscore_outlier'] = z_outliers
            outlier_votes += z_outliers.astype(int)
        
        # IQR Method (univariate)
        if self.use_iqr:
            Q1 = np.percentile(X_scaled, 25, axis=0)
            Q3 = np.percentile(X_scaled, 75, axis=0)
            IQR = Q3 - Q1
            lower = Q1 - self.iqr_multiplier * IQR
            upper = Q3 + self.iqr_multiplier * IQR
            
            iqr_outliers = ((X_scaled < lower) | (X_scaled > upper)).any(axis=1)
            results['iqr_outlier'] = iqr_outliers
            outlier_votes += iqr_outliers.astype(int)
        
        # Ensemble decision: majority vote
        n_methods = sum([
            self.use_isolation_forest,
            self.use_lof,
            self.use_zscore,
            self.use_iqr
        ])
        results['outlier_votes'] = outlier_votes
        results['is_outlier'] = outlier_votes > (n_methods / 2)
        results['outlier_probability'] = outlier_votes / n_methods
        
        return results


# Usage:
# detector = EnsembleOutlierDetector(contamination=0.05)
# outlier_results = detector.fit_detect(train_df)
# print(f"Detected {outlier_results['is_outlier'].sum()} outliers out of {len(train_df)}")
# outlier_results.sort_values('outlier_votes', ascending=False).head(10)
```

### 3.3 Treatment Strategies

| Strategy | When to Use | How | Impact |
|----------|-------------|-----|--------|
| **Keep** | Outliers are genuine, rare events | No action | Preserves information |
| **Winsorize** | Moderate outliers, maintain distribution | Cap at percentiles (e.g., 1st, 99th) | Reduces extreme impact |
| **Remove** | Clear errors, sufficient data | Delete rows | Cleaner but loses data |
| **Transform** | Skewed distribution producing outliers | Log, Box-Cox, Yeo-Johnson | Normalizes distribution |
| **Separate Model** | Rare but important class | Build dedicated model | Best for fraud/rare events |
| **Flag** | Outliers have special meaning | Add indicator column | Preserves signal |

```python
from sklearn.preprocessing import PowerTransformer
from typing import Literal

def treat_outliers(
    X: pd.DataFrame,
    method: Literal['keep', 'winsorize', 'remove', 'transform'] = 'winsorize',
    outlier_mask: np.ndarray = None,
    winsor_limits: tuple = (0.01, 0.99),
    transform_method: str = 'yeo-johnson'
) -> pd.DataFrame:
    """
    Apply outlier treatment strategy.
    
    Args:
        X: Input DataFrame
        method: Treatment strategy
        outlier_mask: Boolean array indicating outliers (for remove method)
        winsor_limits: Lower/upper percentiles for winsorization
        transform_method: 'yeo-johnson' or 'box-cox'
    """
    X_treated = X.copy()
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    
    if method == 'keep':
        return X_treated
    
    elif method == 'winsorize':
        for col in numeric_cols:
            lower_limit = X[col].quantile(winsor_limits[0])
            upper_limit = X[col].quantile(winsor_limits[1])
            X_treated[col] = X[col].clip(lower=lower_limit, upper=upper_limit)
    
    elif method == 'remove':
        if outlier_mask is None:
            raise ValueError("outlier_mask required for remove method")
        X_treated = X[~outlier_mask]
    
    elif method == 'transform':
        transformer = PowerTransformer(method=transform_method, standardize=True)
        X_treated[numeric_cols] = transformer.fit_transform(X[numeric_cols])
    
    return X_treated


# Winsorization with automatic bounds per feature
def smart_winsorize(
    X: pd.DataFrame,
    iqr_multiplier: float = 3.0,  # More conservative than 1.5
    per_feature: bool = True
) -> pd.DataFrame:
    """
    Winsorize using IQR-based bounds per feature.
    Uses 3xIQR by default (less aggressive than 1.5xIQR).
    """
    X_wins = X.copy()
    
    for col in X.select_dtypes(include=[np.number]).columns:
        Q1, Q3 = X[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower = Q1 - iqr_multiplier * IQR
        upper = Q3 + iqr_multiplier * IQR
        X_wins[col] = X[col].clip(lower=lower, upper=upper)
    
    return X_wins
```

### 3.4 Domain-Specific Outlier Detection

**Time Series Outliers**

```python
from statsmodels.tsa.seasonal import seasonal_decompose

def detect_timeseries_outliers(
    series: pd.Series,
    method: str = 'residual',
    threshold: float = 3.0
) -> pd.Series:
    """
    Detect outliers in time series data.
    
    Methods:
    - 'residual': Based on decomposition residuals
    - 'rolling': Based on rolling statistics
    """
    if method == 'residual':
        # Decompose and check residuals
        decomposition = seasonal_decompose(series, model='additive', period=7)
        residuals = decomposition.resid.dropna()
        z_scores = np.abs(stats.zscore(residuals))
        outliers = z_scores > threshold
        
        # Align with original index
        outlier_series = pd.Series(False, index=series.index)
        outlier_series[residuals.index] = outliers
        return outlier_series
    
    elif method == 'rolling':
        # Compare to rolling window
        rolling_mean = series.rolling(window=7, center=True).mean()
        rolling_std = series.rolling(window=7, center=True).std()
        z_scores = np.abs((series - rolling_mean) / rolling_std)
        return z_scores > threshold
```

**Multivariate Outliers with Mahalanobis Distance**

```python
def mahalanobis_outliers(X: pd.DataFrame, threshold: float = 3.0) -> np.ndarray:
    """
    Detect multivariate outliers using Mahalanobis distance.
    Accounts for correlation between features.
    """
    X_clean = X.dropna()
    cov = X_clean.cov()
    inv_cov = np.linalg.inv(cov)
    mean = X_clean.mean()
    
    # Compute Mahalanobis distance
    diff = X_clean - mean
    mahal_dist = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))
    
    # Threshold: chi-square distribution
    # p=0.975 for 3 degrees of freedom gives threshold ~3.0
    return mahal_dist > threshold
```

### Common Pitfalls

- **Pitfall:** Removing outliers without domain investigation.
  - **Consequence:** Discarding valid rare events (e.g., fraud cases).
  - **Solution:** Always inspect flagged outliers with domain experts.

- **Pitfall:** Using Z-score on non-Gaussian data.
  - **Consequence:** Many false positives/negatives.
  - **Solution:** Use IQR method or transform data first.

- **Pitfall:** Treating outliers before train-test split.
  - **Consequence:** Test set information leaks through outlier calculations.
  - **Solution:** Compute outlier bounds on training set only.

```python
# Diagnostic: Visual outlier inspection
import matplotlib.pyplot as plt
import seaborn as sns

def plot_outlier_diagnostics(df: pd.DataFrame, col: str):
    """Create diagnostic plots for outlier assessment."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Histogram with KDE
    df[col].hist(bins=50, ax=axes[0, 0], density=True, alpha=0.7)
    df[col].plot.kde(ax=axes[0, 0], color='red')
    axes[0, 0].set_title(f'Distribution of {col}')
    
    # Box plot
    df[col].plot.box(ax=axes[0, 1])
    axes[0, 1].set_title(f'Box Plot of {col}')
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(df[col].dropna(), dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot (Normality Check)')
    
    # Time series (if datetime index)
    if isinstance(df.index, pd.DatetimeIndex):
        df[col].plot(ax=axes[1, 1])
        axes[1, 1].set_title(f'{col} Over Time')
        axes[1, 1].axhline(df[col].mean(), color='red', linestyle='--')
    else:
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    return fig
```

---

## 4. Data Validation Pipelines

**Impact:** 🔴 HIGH-IMPACT

Data validation catches issues before they corrupt models or make it to production. Automated validation pipelines are essential for reliable ML systems.

### 4.1 Schema Validation with Pandera

```python
import pandas as pd
import pandera as pa
from pandera.typing import Series, DataFrame

# Define schema using DataFrameModel (class-based API)
class TransactionSchema(pa.DataFrameModel):
    """Schema for transaction data validation."""
    
    # Column definitions with type and value constraints
    transaction_id: Series[str] = pa.Field(
        unique=True,
        nullable=False,
        str_matches=r'^TXN\d{8}$'  # Pattern: TXN followed by 8 digits
    )
    
    user_id: Series[str] = pa.Field(
        str_matches=r'^USER\d{6}$',
        nullable=False
    )
    
    amount: Series[float] = pa.Field(
        ge=0.01,  # Minimum transaction amount
        le=1000000,  # Maximum reasonable amount
        coerce=True  # Automatically convert to float
    )
    
    currency: Series[str] = pa.Field(
        isin=['USD', 'EUR', 'GBP', 'JPY'],
        nullable=False
    )
    
    timestamp: Series[pd.DatetimeTZDtype] = pa.Field(
        nullable=False,
        coerce=True
    )
    
    status: Series[str] = pa.Field(
        isin=['pending', 'completed', 'failed', 'refunded'],
        nullable=False
    )
    
    # Multi-column checks
    @pa.check('amount')
    def amount_not_zero(cls, amount: Series[float]) -> Series[bool]:
        """Ensure no zero-amount transactions."""
        return amount > 0
    
    @pa.check('timestamp')
    def timestamp_not_future(cls, timestamp: Series[pd.Timestamp]) -> Series[bool]:
        """Ensure transactions are not in the future."""
        return timestamp <= pd.Timestamp.now(tz=timestamp.dt.tz)
    
    @pa.dataframe_check
    def positive_amount_for_completed(cls, df: DataFrame) -> Series[bool]:
        """Completed transactions must have positive amounts."""
        return df[df['status'] == 'completed']['amount'] > 0


# Usage:
# try:
#     validated_df = TransactionSchema.validate(raw_df)
#     print("✅ Data validation passed")
# except pa.errors.SchemaErrors as exc:
#     print("❌ Validation failed:")
#     print(exc.failure_cases)
```

### 4.2 Distribution Validation with Statistical Tests

```python
from scipy import stats
import numpy as np
from typing import Dict, Tuple

class DistributionValidator:
    """
    Validate that data distributions match expected patterns.
    Useful for detecting drift and data quality issues.
    """
    
    def __init__(self, reference_df: pd.DataFrame):
        """Store reference distributions."""
        self.reference_stats = {}
        for col in reference_df.select_dtypes(include=[np.number]).columns:
            self.reference_stats[col] = {
                'mean': reference_df[col].mean(),
                'std': reference_df[col].std(),
                'min': reference_df[col].min(),
                'max': reference_df[col].max(),
                'q25': reference_df[col].quantile(0.25),
                'q50': reference_df[col].quantile(0.50),
                'q75': reference_df[col].quantile(0.75),
            }
    
    def validate_ks_test(
        self,
        new_data: pd.DataFrame,
        alpha: float = 0.05
    ) -> Dict[str, Tuple[bool, float]]:
        """
        Kolmogorov-Smirnov test for distribution equality.
        Returns dict of {(column): (is_different, p_value)}
        """
        results = {}
        
        for col in new_data.select_dtypes(include=[np.number]).columns:
            if col not in self.reference_stats:
                continue
            
            # Compare distributions using KS test
            # Note: Need original reference data, not just stats
            # This is simplified - in practice, store reference data
            pass
        
        return results
    
    def validate_range(
        self,
        new_data: pd.DataFrame,
        tolerance: float = 0.1
    ) -> Dict[str, bool]:
        """
        Check if new data values are within expected range (with tolerance).
        """
        violations = {}
        
        for col in new_data.select_dtypes(include=[np.number]).columns:
            if col not in self.reference_stats:
                continue
            
            ref_min = self.reference_stats[col]['min']
            ref_max = self.reference_stats[col]['max']
            
            # Allow some tolerance
            tolerance_amount = (ref_max - ref_min) * tolerance
            lower_bound = ref_min - tolerance_amount
            upper_bound = ref_max + tolerance_amount
            
            violations[col] = (
                new_data[col].min() < lower_bound or
                new_data[col].max() > upper_bound
            )
        
        return violations
    
    def validate_statistical_shift(
        self,
        new_data: pd.DataFrame,
        threshold_std: float = 2.0
    ) -> Dict[str, bool]:
        """
        Check if mean/median shifted significantly (> threshold_std standard deviations).
        """
        shifts = {}
        
        for col in new_data.select_dtypes(include=[np.number]).columns:
            if col not in self.reference_stats:
                continue
            
            ref_mean = self.reference_stats[col]['mean']
            ref_std = self.reference_stats[col]['std']
            new_mean = new_data[col].mean()
            
            # Z-score of the mean shift
            shift_z = abs(new_mean - ref_mean) / (ref_std / np.sqrt(len(new_data)))
            shifts[col] = shift_z > threshold_std
        
        return shifts
```

### 4.3 Pipeline Integration with Scikit-learn

```python
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

class DataValidator(BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn transformer for data validation.
    Raises ValueError if validation fails during fit/transform.
    """
    
    def __init__(self, schema: pa.DataFrameModel, strict: bool = True):
        """
        Args:
            schema: Pandera schema to validate against
            strict: If True, raise exception on validation failure
        """
        self.schema = schema
        self.strict = strict
        self.validation_errors_ = []
    
    def fit(self, X: pd.DataFrame, y=None):
        """Validate training data and learn schema parameters."""
        try:
            self.schema.validate(X)
        except pa.errors.SchemaErrors as exc:
            self.validation_errors_.append(exc.failure_cases)
            if self.strict:
                raise ValueError(f"Training data validation failed:\n{exc}")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Validate new data."""
        try:
            return self.schema.validate(X)
        except pa.errors.SchemaErrors as exc:
            self.validation_errors_.append(exc.failure_cases)
            if self.strict:
                raise ValueError(f"Data validation failed:\n{exc}")
            return X  # Return as-is if not strict
    
    def get_validation_report(self) -> pd.DataFrame:
        """Return summary of all validation errors."""
        if not self.validation_errors_:
            return pd.DataFrame()
        return pd.concat(self.validation_errors_, ignore_index=True)


# Usage in pipeline:
# pipeline = Pipeline([
#     ('validator', DataValidator(TransactionSchema, strict=True)),
#     ('preprocessor', preprocessor),
#     ('model', model)
# ])
```

### 4.4 Continuous Monitoring for Production

```python
from datetime import datetime, timedelta
import json

class DataQualityMonitor:
    """
    Monitor data quality in production with alerting.
    Stores metrics over time for trend analysis.
    """
    
    def __init__(self, schema: pa.DataFrameModel, alert_threshold: float = 0.05):
        self.schema = schema
        self.alert_threshold = alert_threshold  # Alert if >5% rows fail
        self.history = []
    
    def check_batch(
        self,
        data: pd.DataFrame,
        batch_id: str = None,
        timestamp: datetime = None
    ) -> Dict[str, any]:
        """
        Validate a batch of data and record results.
        Returns quality report.
        """
        if timestamp is None:
            timestamp = datetime.now()
        if batch_id is None:
            batch_id = timestamp.isoformat()
        
        report = {
            'batch_id': batch_id,
            'timestamp': timestamp.isoformat(),
            'total_rows': len(data),
            'failed_rows': 0,
            'failure_rate': 0.0,
            'checks': {}
        }
        
        try:
            validated = self.schema.validate(data)
            report['status'] = 'passed'
        except pa.errors.SchemaErrors as exc:
            failures = exc.failure_cases
            report['status'] = 'failed'
            report['failed_rows'] = failures['failure_case'].nunique()
            report['failure_rate'] = report['failed_rows'] / len(data)
            
            # Group by check
            for check, group in failures.groupby('check'):
                report['checks'][check] = {
                    'failed_count': len(group),
                    'failed_rows': group['failure_case'].nunique(),
                    'sample_failures': group['failure_case'].head(5).tolist()
                }
        
        # Store history
        self.history.append(report)
        
        # Alert if threshold exceeded
        if report['failure_rate'] > self.alert_threshold:
            self._send_alert(report)
        
        return report
    
    def _send_alert(self, report: Dict):
        """Send alert (implement with your notification system)."""
        alert_msg = f"""
        🔴 Data Quality Alert - Batch {report['batch_id']}
        Failure Rate: {report['failure_rate']:.2%}
        Failed Checks: {list(report['checks'].keys())}
        Timestamp: {report['timestamp']}
        """
        print(alert_msg)  # Replace with actual alert system
    
    def get_trend_report(self, window_hours: int = 24) -> Dict:
        """Analyze quality trends over time window."""
        cutoff = datetime.now() - timedelta(hours=window_hours)
        recent = [r for r in self.history 
                  if datetime.fromisoformat(r['timestamp']) > cutoff]
        
        if not recent:
            return {'message': 'No data in time window'}
        
        return {
            'window_hours': window_hours,
            'total_batches': len(recent),
            'avg_failure_rate': np.mean([r['failure_rate'] for r in recent]),
            'max_failure_rate': max([r['failure_rate'] for r in recent]),
            'failed_batches': sum(1 for r in recent if r['status'] == 'failed')
        }
```

### Common Pitfalls

- **Pitfall:** Only validating schema, not distributions.
  - **Consequence:** Silent drift in feature distributions degrades model performance.
  - **Solution:** Add statistical distribution checks.

- **Pitfall:** Validating after preprocessing.
  - **Consequence:** Issues masked by transformations (e.g., NaN becomes 0 after fillna).
  - **Solution:** Validate at data ingestion point.

- **Pitfall:** Too strict validation blocks all data.
  - **Consequence:** Pipeline fails, data starvation.
  - **Solution:** Implement quarantine/quarantine review for flagged data.

---

## 5. Class Imbalance Handling

**Impact:** 🔴 HIGH-IMPACT (for classification with imbalanced classes)

Class imbalance is common in fraud detection, medical diagnosis, and fault detection. Standard accuracy metrics are misleading, and models may ignore minority classes.

### 5.1 Assessing Imbalance Severity

| Imbalance Level | Minority Class % | Approach |
|-----------------|------------------|----------|
| **Mild** | 20-40% | Standard techniques, class weights usually sufficient |
| **Moderate** | 5-20% | Resampling + ensemble methods recommended |
| **Severe** | 1-5% | Requires careful handling, consider anomaly detection |
| **Extreme** | <1% | May need reformulation as anomaly detection |

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def assess_imbalance(y: pd.Series, min_samples: int = 100) -> Dict[str, any]:
    """
    Comprehensive assessment of class imbalance.
    """
    class_counts = Counter(y)
    total = len(y)
    n_classes = len(class_counts)
    
    # Sort by count (ascending)
    sorted_counts = sorted(class_counts.items(), key=lambda x: x[1])
    
    report = {
        'n_classes': n_classes,
        'n_samples': total,
        'class_distribution': {k: v / total for k, v in class_counts.items()},
        'minority_class': sorted_counts[0][0],
        'minority_pct': sorted_counts[0][1] / total * 100,
        'majority_class': sorted_counts[-1][0],
        'majority_pct': sorted_counts[-1][1] / total * 100,
        'imbalance_ratio': sorted_counts[-1][1] / sorted_counts[0][1],
        'recommended_approach': None
    }
    
    # Determine severity
    minority_pct = report['minority_pct']
    if minority_pct >= 20:
        severity = 'Mild'
        recommendation = 'Use class weights; standard techniques likely sufficient'
    elif minority_pct >= 5:
        severity = 'Moderate'
        recommendation = 'Use resampling (SMOTE/ADASYN) + ensemble methods'
    elif minority_pct >= 1:
        severity = 'Severe'
        recommendation = 'Combine resampling, ensembles, and threshold tuning'
    else:
        severity = 'Extreme'
        recommendation = 'Consider reformulating as anomaly detection or one-class classification'
    
    report['severity'] = severity
    report['recommended_approach'] = recommendation
    
    # Check if minority class has enough samples
    if sorted_counts[0][1] < min_samples:
        report['warning'] = f"Minority class has < {min_samples} samples - consider data augmentation or few-shot learning"
    
    return report


def plot_class_distribution(y: pd.Series, title: str = "Class Distribution"):
    """Visualize class imbalance."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Bar plot
    class_counts = pd.Series(Counter(y)).sort_values(ascending=False)
    class_counts.plot(kind='bar', ax=ax1)
    ax1.set_title(title)
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Count')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
    
    # Pie chart
    class_counts.plot(kind='pie', ax=ax2, autopct='%1.1f%%')
    ax2.set_title('Class Proportion')
    ax2.set_ylabel('')
    
    plt.tight_layout()
    return fig
```

### 5.2 Resampling Techniques

```python
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, EditedNearestNeighbours
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def create_resampling_pipeline(
    method: str = 'smote',
    sampling_strategy: float = 0.5,  # Target ratio for minority class
    random_state: int = 42
) -> ImbPipeline:
    """
    Create pipeline with resampling.
    
    Args:
        method: 'smote', 'adasyn', 'borderline', 'svmsmote', 
                'undersample', 'smoteenn', 'smotetomek'
        sampling_strategy: Float for ratio, 'auto' for balanced, 
                          or dict for per-class ratios
    """
    
    resampler_map = {
        'smote': SMOTE(sampling_strategy=sampling_strategy, random_state=random_state),
        'adasyn': ADASYN(sampling_strategy=sampling_strategy, random_state=random_state),
        'borderline': BorderlineSMOTE(sampling_strategy=sampling_strategy, 
                                      random_state=random_state),
        'svmsmote': SVMSMOTE(sampling_strategy=sampling_strategy, 
                             random_state=random_state),
        'undersample': RandomUnderSampler(sampling_strategy=sampling_strategy, 
                                          random_state=random_state),
        'smoteenn': SMOTEENN(sampling_strategy=sampling_strategy, 
                             random_state=random_state),
        'smotetomek': SMOTETomek(sampling_strategy=sampling_strategy, 
                                 random_state=random_state),
    }
    
    resampler = resampler_map.get(method.lower(), SMOTE())
    
    pipeline = ImbPipeline([
        ('resampler', resampler),
        ('classifier', RandomForestClassifier(random_state=random_state, 
                                              class_weight='balanced'))
    ])
    
    return pipeline


def compare_resampling_methods(
    X_train, y_train, X_test, y_test,
    methods: list = ['smote', 'adasyn', 'undersample', 'smoteenn']
) -> pd.DataFrame:
    """
    Compare multiple resampling methods.
    Returns DataFrame with metrics for each method.
    """
    results = []
    
    for method in methods:
        pipeline = create_resampling_pipeline(method=method)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        results.append({
            'method': method,
            'accuracy': report['accuracy'],
            'precision_macro': report['macro avg']['precision'],
            'recall_macro': report['macro avg']['recall'],
            'f1_macro': report['macro avg']['f1-score'],
            'precision_minority': report[str(min(y_test))]['precision'],
            'recall_minority': report[str(min(y_test))]['recall'],
            'f1_minority': report[str(min(y_test))]['f1-score'],
        })
    
    return pd.DataFrame(results).set_index('method')
```

### 5.3 Advanced: Threshold Tuning for Imbalanced Data

```python
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_curve, roc_curve
import numpy as np

def optimize_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = 'f1',
    min_precision: float = None,
    min_recall: float = None
) -> float:
    """
    Find optimal threshold for imbalanced classification.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities for positive class
        metric: 'f1', 'precision', 'recall', or 'custom'
        min_precision: Minimum acceptable precision
        min_recall: Minimum acceptable recall
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    
    if metric == 'f1':
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
    elif metric == 'precision':
        best_threshold = thresholds[np.argmax(precisions[:-1])]
    elif metric == 'recall':
        best_threshold = thresholds[np.argmax(recalls[:-1])]
    else:
        # Custom: maximize F1 subject to constraints
        valid_mask = np.ones_like(thresholds, dtype=bool)
        if min_precision is not None:
            valid_mask &= (precisions[:-1] >= min_precision)
        if min_recall is not None:
            valid_mask &= (recalls[:-1] >= min_recall)
        
        if valid_mask.any():
            best_idx = np.where(valid_mask)[0][np.argmax(f1_scores[valid_mask])]
            best_threshold = thresholds[best_idx]
        else:
            best_threshold = 0.5  # Fallback
    
    return best_threshold


class ThresholdTuner:
    """
    Tune classification threshold for imbalanced data.
    Supports cross-validation based threshold selection.
    """
    
    def __init__(
        self,
        metric: str = 'f1',
        cv: int = 5,
        min_precision: float = None,
        min_recall: float = None
    ):
        self.metric = metric
        self.cv = cv
        self.min_precision = min_precision
        self.min_recall = min_recall
        self.best_threshold_ = 0.5
    
    def fit(self, X, y, estimator):
        """
        Find optimal threshold using cross-validation.
        """
        from sklearn.model_selection import StratifiedKFold
        
        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42)
        thresholds = []
        
        for train_idx, val_idx in skf.split(X, y):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            estimator.fit(X_train_fold, y_train_fold)
            y_proba = estimator.predict_proba(X_val_fold)[:, 1]
            
            threshold = optimize_threshold(
                y_val_fold, y_proba,
                metric=self.metric,
                min_precision=self.min_precision,
                min_recall=self.min_recall
            )
            thresholds.append(threshold)
        
        self.best_threshold_ = np.mean(thresholds)
        return self
    
    def predict(self, estimator, X):
        """Predict with tuned threshold."""
        y_proba = estimator.predict_proba(X)[:, 1]
        return (y_proba >= self.best_threshold_).astype(int)
```

### 5.4 Cost-Sensitive Learning

When misclassification costs are asymmetric (e.g., false negative more expensive than false positive), use cost-sensitive learning.

```python
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class CostSensitiveClassifier(BaseEstimator, ClassifierMixin):
    """
    Wrap any classifier to be cost-sensitive.
    Adjusts predictions based on cost matrix.
    """
    
    def __init__(
        self,
        base_estimator,
        cost_fp: float = 1.0,
        cost_fn: float = 1.0,
        cost_tp: float = 0.0,
        cost_tn: float = 0.0
    ):
        """
        Args:
            base_estimator: Any sklearn classifier with predict_proba
            cost_fp: Cost of false positive
            cost_fn: Cost of false negative (usually higher)
            cost_tp: Cost of true positive (often negative = benefit)
            cost_tn: Cost of true negative
        """
        self.base_estimator = base_estimator
        self.cost_fp = cost_fp
        self.cost_fn = cost_fn
        self.cost_tp = cost_tp
        self.cost_tn = cost_tn
    
    def fit(self, X, y):
        self.base_estimator.fit(X, y)
        return self
    
    def predict(self, X):
        """
        Predict based on minimizing expected cost.
        """
        y_proba = self.base_estimator.predict_proba(X)[:, 1]  # P(y=1|x)
        
        # Expected cost of predicting 1:
        # E[cost|pred=1] = P(y=0)*cost_fp + P(y=1)*cost_tp
        # Expected cost of predicting 0:
        # E[cost|pred=0] = P(y=0)*cost_tn + P(y=1)*cost_fn
        
        p0 = 1 - y_proba
        p1 = y_proba
        
        cost_pred_1 = p0 * self.cost_fp + p1 * self.cost_tp
        cost_pred_0 = p0 * self.cost_tn + p1 * self.cost_fn
        
        return (cost_pred_1 < cost_pred_0).astype(int)


# Usage example:
# For fraud detection where missing fraud is 100x worse than false alarm:
# cost_sensitive = CostSensitiveClassifier(
#     base_estimator=RandomForestClassifier(),
#     cost_fp=1.0,    # False alarm: investigate valid transaction
#     cost_fn=100.0,  # Missed fraud: lose transaction amount
#     cost_tp=0.0,
#     cost_tn=0.0
# )
```

### Common Pitfalls

- **Pitfall:** Resampling before train-test split.
  - **Consequence:** Test information leaks into training.
  - **Solution:** Always resample only training data.

- **Pitfall:** Using accuracy as the metric for imbalanced data.
  - **Consequence:** 99% accuracy by predicting majority class.
  - **Solution:** Use precision, recall, F1, AUC-PR, or cost-based metrics.

- **Pitfall:** Over-sampling creating synthetic samples near decision boundary.
  - **Consequence:** Overfitting, especially with SMOTE variants.
  - **Solution:** Use borderline variants carefully, validate thoroughly.

```python
# Diagnostic: Check if resampling introduced overfitting
def check_overfitting_from_resampling(
    X_train, y_train, X_test, y_test,
    with_resampling: bool = True
) -> Dict[str, float]:
    """
    Compare train vs test performance to detect overfitting.
    """
    pipeline = create_resampling_pipeline('smote')
    pipeline.fit(X_train, y_train)
    
    train_score = pipeline.score(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)
    
    return {
        'train_accuracy': train_score,
        'test_accuracy': test_score,
        'overfitting_gap': train_score - test_score,
        'is_overfitting': (train_score - test_score) > 0.15
    }
```

---

## 6. Data Leakage Prevention

**Impact:** 🔴 HIGH-IMPACT (The #1 cause of inflated ML performance)

Data leakage occurs when information from outside the training dataset influences the model, leading to overly optimistic performance estimates that don't generalize.

### 6.1 Types of Data Leakage

| Leakage Type | Description | Example | Detection |
|--------------|-------------|---------|-----------|
| **Target Leakage** | Feature contains target information | "Days until default" in loan default model | Check feature-target correlation |
| **Train-Test Contamination** | Preprocessing uses test data | Fitting imputer/scaler on full dataset | Ensure fit on train only |
| **Temporal Leakage** | Future data leaks to past | Using 2024 data to predict 2023 | Strict temporal splits |
| **Feature Selection Leakage** | Feature selection on all data | Selecting features using full dataset | Include selection in CV loop |
| **Sample Leakage** | Duplicate rows across splits | Same customer in train and test | Check for duplicates |
| **Data Snooping** | Human decisions leak info | Removing outliers after seeing test | Blind holdout sets |

### 6.2 Detection: Leakage Audit Checklist

```python
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import cross_val_score
from typing import List, Dict, Any

def leakage_audit(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_names: List[str] = None
) -> Dict[str, Any]:
    """
    Comprehensive audit for potential data leakage.
    """
    audit_results = {
        'issues_found': [],
        'warnings': [],
        'passed_checks': []
    }
    
    # 1. Check for duplicate rows across train and test
    train_hash = pd.util.hash_pandas_object(X_train).values
    test_hash = pd.util.hash_pandas_object(X_test).values
    duplicate_samples = len(set(train_hash) & set(test_hash))
    
    if duplicate_samples > 0:
        audit_results['issues_found'].append(
            f"Sample Leakage: {duplicate_samples} duplicate samples found across train and test"
        )
    else:
        audit_results['passed_checks'].append("No duplicate samples across train/test")
    
    # 2. Check for target leakage via suspicious feature names
    if feature_names:
        suspicious_keywords = [
            'target', 'label', 'outcome', 'result', 'future', 
            ' upcoming', 'pending', 'final', 'post_', 'after'
        ]
        for feature in feature_names:
            if any(kw in feature.lower() for kw in suspicious_keywords):
                audit_results['warnings'].append(
                    f"Suspicious feature name: '{feature}' may indicate target leakage"
                )
    
    # 3. Check for near-perfect predictors (suspicious for target leakage)
    if len(X_train) > 0 and len(y_train) > 0:
        combined = pd.concat([X_train, y_train], axis=1)
        correlations = combined.corr()[y_train.name].abs().sort_values(ascending=False)
        
        perfect_predictors = correlations[correlations > 0.95].index.tolist()
        perfect_predictors = [c for c in perfect_predictors if c != y_train.name]
        
        if perfect_predictors:
            audit_results['issues_found'].append(
                f"Target Leakage: Features with >0.95 correlation to target: {perfect_predictors}"
            )
    
    # 4. Check distribution similarity between train and test
    for col in X_train.select_dtypes(include=[np.number]).columns:
        train_mean, test_mean = X_train[col].mean(), X_test[col].mean()
        train_std, test_std = X_train[col].std(), X_test[col].std()
        
        # Z-test for difference in means
        pooled_std = np.sqrt(train_std**2/len(X_train) + test_std**2/len(X_test))
        if pooled_std > 0:
            z_stat = abs(train_mean - test_mean) / pooled_std
            if z_stat > 5:  # Highly significant difference
                audit_results['warnings'].append(
                    f"Distribution Shift: Feature '{col}' has very different means (z={z_stat:.2f})"
                )
    
    # 5. Check for temporal leakage (if datetime columns exist)
    datetime_cols = X_train.select_dtypes(include=['datetime64']).columns
    if len(datetime_cols) > 0:
        train_time_range = (X_train[datetime_cols].min(), X_train[datetime_cols].max())
        test_time_range = (X_test[datetime_cols].min(), X_test[datetime_cols].max())
        
        if train_time_range[1].max() > test_time_range[0].min():
            audit_results['issues_found'].append(
                "Temporal Leakage: Training data extends into test period"
            )
    
    # 6. Check for leakage via unique identifiers
    for col in X_train.columns:
        unique_ratio = X_train[col].nunique() / len(X_train)
        if unique_ratio > 0.95 and X_train[col].dtype == 'object':
            audit_results['warnings'].append(
                f"Potential ID column: '{col}' has {unique_ratio:.1%} unique values"
            )
    
    return audit_results


# Usage:
# audit = leakage_audit(X_train, X_test, y_train, y_test)
# print(f"Issues: {len(audit['issues_found'])}")
# print(f"Warnings: {len(audit['warnings'])}")
# for issue in audit['issues_found']:
#     print(f"  ❌ {issue}")
```

### 6.3 Prevention: Proper Pipeline Design

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split

def create_safe_pipeline() -> Pipeline:
    """
    Create a pipeline that prevents data leakage by design.
    All preprocessing happens within CV folds.
    """
    
    # Preprocessing steps
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, selector_numeric()),
        ('cat', categorical_transformer, selector_categorical())
    ])
    
    # Full pipeline with feature selection AND model
    # Feature selection happens INSIDE CV, preventing leakage
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('feature_selection', SelectKBest(f_classif, k=20)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    return full_pipeline


def selector_numeric():
    from sklearn.compose import make_column_selector
    return make_column_selector(dtype_include=np.number)

def selector_categorical():
    from sklearn.compose import make_column_selector
    return make_column_selector(dtype_include=object)


# SAFE: Correct usage with pipeline
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
safe_pipeline = create_safe_pipeline()

# Cross-validation safely applies all steps within each fold
cv_scores = cross_val_score(safe_pipeline, X_train, y_train, cv=5)
print(f"CV Scores (no leakage): {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

# Final evaluation
safe_pipeline.fit(X_train, y_train)
test_score = safe_pipeline.score(X_test, y_test)
print(f"Test Score (realistic): {test_score:.3f}")


# UNSAFE: Leaky approach (DON'T DO THIS)
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)  # ❌ Uses all data!
# X_train, X_test = train_test_split(X_scaled, y)  # ❌ Test data already seen
```

### 6.4 Temporal Data: Time-Based Splitting

For time series or temporally-ordered data, random splitting causes leakage.

```python
from datetime import datetime
import pandas as pd

def temporal_train_test_split(
    df: pd.DataFrame,
    date_column: str,
    test_size: float = 0.2,
    validation_size: float = 0.1
) -> tuple:
    """
    Split data temporally to prevent leakage.
    
    Returns: (train, val, test) DataFrames
    """
    df = df.sort_values(date_column)
    n = len(df)
    
    test_start = int(n * (1 - test_size))
    val_start = int(test_start * (1 - validation_size / (1 - test_size)))
    
    train = df.iloc[:val_start].copy()
    val = df.iloc[val_start:test_start].copy()
    test = df.iloc[test_start:].copy()
    
    print(f"Train: {train[date_column].min()} to {train[date_column].max()}")
    print(f"Val: {val[date_column].min()} to {val[date_column].max()}")
    print(f"Test: {test[date_column].min()} to {test[date_column].max()}")
    
    return train, val, test


def create_temporal_cv(
    df: pd.DataFrame,
    date_column: str,
    n_splits: int = 5
):
    """
    Create temporal cross-validation splits.
    Each fold uses only past data to predict future.
    """
    from sklearn.model_selection import TimeSeriesSplit
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    df_sorted = df.sort_values(date_column)
    
    splits = []
    for train_idx, val_idx in tscv.split(df_sorted):
        splits.append((
            df_sorted.iloc[train_idx],
            df_sorted.iloc[val_idx]
        ))
    
    return splits


# Usage:
# train, val, test = temporal_train_test_split(df, date_column='transaction_date')
# cv_splits = create_temporal_cv(train, date_column='transaction_date')
```

### 6.5 Target Leakage: Redundant Feature Detection

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import numpy as np

def detect_target_leakage(
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.9
) -> List[str]:
    """
    Detect features that may leak target information.
    Returns list of suspicious features.
    """
    suspicious = []
    
    # Method 1: High mutual information with target
    mi_scores = mutual_info_classif(X, y, random_state=42)
    max_mi = mi_scores.max()
    
    for i, score in enumerate(mi_scores):
        if score > threshold * max_mi and score > 0.5:
            suspicious.append(X.columns[i])
    
    # Method 2: Train a simple model and check for perfect prediction
    rf = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
    rf.fit(X, y)
    
    if rf.score(X, y) > 0.95:
        suspicious.append("MODEL_PERFECT: Model achieves >95% accuracy - check for leakage")
    
    # Method 3: Permutation importance check
    # If removing a feature drastically drops performance, it might leak target
    perm_importance = permutation_importance(rf, X, y, n_repeats=5, random_state=42)
    
    for i, importance in enumerate(perm_importance.importances_mean):
        if importance > 0.3:  # Very important feature
            if X.columns[i] not in suspicious:
                suspicious.append(f"HIGH_IMPORTANCE: {X.columns[i]} (importance={importance:.3f})")
    
    return suspicious


# Diagnostic: Remove suspicious features and retrain
def test_leakage_impact(
    X: pd.DataFrame,
    y: pd.Series,
    suspicious_features: List[str]
) -> Dict[str, float]:
    """
    Compare model performance with and without suspicious features.
    Large drop suggests legitimate features; small drop suggests leakage.
    """
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LogisticRegression
    
    X_clean = X.drop(columns=[f for f in suspicious_features if f in X.columns])
    
    model = LogisticRegression(max_iter=1000)
    
    score_with_all = cross_val_score(model, X, y, cv=5).mean()
    score_clean = cross_val_score(model, X_clean, y, cv=5).mean()
    
    return {
        'score_with_all_features': score_with_all,
        'score_without_suspicious': score_clean,
        'performance_drop': score_with_all - score_clean,
        'likely_leakage': (score_with_all - score_clean) < 0.05
    }
```

### Common Pitfalls

- **Pitfall:** Using `.fillna()` before splitting.
  - **Solution:** Always include imputer in pipeline.

- **Pitfall:** Feature selection on entire dataset.
  - **Solution:** Include feature selection in cross-validation loop.

- **Pitfall:** Removing outliers after seeing test performance.
  - **Solution:** Use holdout set that is NEVER touched during development.

- **Pitfall:** Using future data for imputation.
  - **Solution:** For time series, use only past data for imputation.

```python
# Anti-pattern checklist
def detect_leaky_patterns(code_str: str) -> List[str]:
    """
    Detect common leaky patterns in code (simplified).
    Real implementation would use AST parsing.
    """
    leaky_patterns = [
        ('fit_transform on full data', 
         r'\.fit_transform\(X[^_]'),  # Should be fit_transform(X_train) only
        ('fit on full data before split',
         r'\.fit\(X,\s*y\)[^}]'),  # .fit(X, y) before train_test_split
        ('StandardScaler on full data',
         r'StandardScaler\(\)\.fit\(X'),
        ('imputer on full data',
         r'SimpleImputer\(\)\.fit\(X'),
        ('feature selection before split',
         r'SelectKBest.*\.fit\(X'),
    ]
    
    import re
    issues = []
    for pattern_name, pattern in leaky_patterns:
        if re.search(pattern, code_str):
            issues.append(pattern_name)
    
    return issues
```

---

## 7. Encoding Categorical Variables

**Impact:** 🟡 INCREMENTAL (High for tree-based models with high-cardinality features)

Categorical encoding is crucial for ML algorithms that require numerical input. The choice depends on cardinality, model type, and whether the feature is ordinal.

### 7.1 Encoding Method Selection Guide

| Method | Cardinality | Model Type | Ordinal? | Pros | Cons |
|--------|-------------|------------|----------|------|------|
| **One-Hot** | Low (<10) | Linear, Tree | No | No false ordinality, interpretable | Curse of dimensionality |
| **Ordinal** | Any | Tree-based | Yes | Preserves order, efficient | False ordinality if not ordered |
| **Target Encoding** | High | Any | No | Handles high cardinality | Risk of target leakage |
| **Frequency Encoding** | High | Any | No | No leakage, simple | Loses information |
| **Embedding** | Very High | Deep Learning | No | Learns relationships | Requires training, complex |
| **Hashing** | Very High | Any | No | Fixed dimension, fast | Collisions, not interpretable |

### 7.2 Implementation: Comprehensive Encoding Pipeline

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    OneHotEncoder, OrdinalEncoder, LabelEncoder
)
from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders import TargetEncoder, LeaveOneOutEncoder, MEstimateEncoder
import warnings

class SmartCategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Automatically choose encoding based on cardinality and data characteristics.
    """
    
    def __init__(
        self,
        low_cardinality_threshold: int = 10,
        high_cardinality_threshold: int = 100,
        encoding_strategy: str = 'auto',  # 'auto', 'onehot', 'target', 'ordinal'
        target_col: str = None,
        smoothing: float = 1.0,
        handle_unknown: str = 'ignore'
    ):
        self.low_threshold = low_cardinality_threshold
        self.high_threshold = high_cardinality_threshold
        self.strategy = encoding_strategy
        self.target_col = target_col
        self.smoothing = smoothing
        self.handle_unknown = handle_unknown
        
        self.encoders_ = {}
        self.feature_categories_ = {}
    
    def fit(self, X: pd.DataFrame, y=None):
        """Learn categorical levels and fit encoders."""
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if not categorical_cols:
            return self
        
        # Analyze cardinality
        for col in categorical_cols:
            n_unique = X[col].nunique()
            self.feature_categories_[col] = {
                'cardinality': n_unique,
                'is_ordinal': self._detect_ordinal(X[col]),
                'ratio': n_unique / len(X)
            }
        
        # Choose encoders based on strategy and characteristics
        for col in categorical_cols:
            info = self.feature_categories_[col]
            cardinality = info['cardinality']
            
            if self.strategy == 'auto':
                # Auto-select based on cardinality and properties
                if cardinality <= self.low_threshold:
                    encoder = OneHotEncoder(
                        sparse_output=False,
                        handle_unknown=self.handle_unknown
                    )
                elif cardinality >= self.high_threshold:
                    # Use target encoding for high cardinality
                    if y is not None and self.target_col:
                        encoder = TargetEncoder(
                            smoothing=self.smoothing,
                            handle_unknown=self.handle_unknown
                        )
                    else:
                        # Fallback to frequency encoding
                        encoder = FrequencyEncoder()
                else:
                    # Medium cardinality: use ordinal
                    encoder = OrdinalEncoder(
                        handle_unknown='use_encoded_value',
                        unknown_value=-1
                    )
            
            elif self.strategy == 'onehot':
                encoder = OneHotEncoder(
                    sparse_output=False,
                    handle_unknown=self.handle_unknown
                )
            
            elif self.strategy == 'target':
                if y is None:
                    warnings.warn("Target encoding requires y; falling back to ordinal")
                    encoder = OrdinalEncoder(
                        handle_unknown='use_encoded_value',
                        unknown_value=-1
                    )
                else:
                    encoder = TargetEncoder(
                        smoothing=self.smoothing,
                        handle_unknown=self.handle_unknown
                    )
            
            elif self.strategy == 'ordinal':
                encoder = OrdinalEncoder(
                    handle_unknown='use_encoded_value',
                    unknown_value=-1
                )
            
            # Fit encoder
            if isinstance(encoder, (OneHotEncoder, OrdinalEncoder)):
                encoder.fit(X[[col]])
            elif isinstance(encoder, TargetEncoder):
                encoder.fit(X[[col]], y)
            elif isinstance(encoder, FrequencyEncoder):
                encoder.fit(X[[col]])
            
            self.encoders_[col] = encoder
        
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Apply encoding."""
        result_parts = []
        feature_names = []
        
        for col, encoder in self.encoders_.items():
            if col not in X.columns:
                continue
            
            if isinstance(encoder, OneHotEncoder):
                transformed = encoder.transform(X[[col]])
                feature_names.extend([
                    f"{col}_{cat}" for cat in encoder.categories_[0]
                ])
            elif isinstance(encoder, OrdinalEncoder):
                transformed = encoder.transform(X[[col]])
                feature_names.append(col)
            elif isinstance(encoder, (TargetEncoder, FrequencyEncoder)):
                transformed = encoder.transform(X[[col]])
                if transformed.ndim == 1:
                    transformed = transformed.reshape(-1, 1)
                feature_names.append(col)
            
            result_parts.append(transformed)
        
        if result_parts:
            result = np.hstack(result_parts)
        else:
            result = np.empty((len(X), 0))
        
        self.feature_names_out_ = feature_names
        return result
    
    def _detect_ordinal(self, series: pd.Series) -> bool:
        """
        Attempt to detect if a categorical feature is ordinal.
        Heuristic: check if values suggest order (1st, 2nd, low, medium, high).
        """
        ordinal_keywords = [
            'low', 'medium', 'high', 'very',
            '1st', '2nd', '3rd', '4th', '5th',
            'first', 'second', 'third', 'fourth', 'fifth',
            'beginner', 'intermediate', 'advanced',
            'small', 'medium', 'large'
        ]
        
        values = series.str.lower().unique()
        matches = sum(any(kw in str(v) for kw in ordinal_keywords) for v in values)
        
        # If majority of values match ordinal patterns
        return matches / len(values) > 0.5 if len(values) > 0 else False
    
    def get_feature_names_out(self) -> list:
        """Return feature names after encoding."""
        return getattr(self, 'feature_names_out_', [])


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Encode categories by their frequency (count).
    No target leakage risk.
    """
    
    def __init__(self):
        self.frequency_map_ = {}
    
    def fit(self, X: pd.DataFrame, y=None):
        """Learn frequency of each category."""
        for col in X.columns:
            self.frequency_map_[col] = X[col].value_counts(normalize=True).to_dict()
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Replace categories with frequencies."""
        result = X.copy()
        for col in X.columns:
            result[col] = result[col].map(self.frequency_map_.get(col, 0))
        return result.values


# Usage:
# encoder = SmartCategoricalEncoder(encoding_strategy='auto')
# X_encoded = encoder.fit_transform(X_train, y_train)
# print(f"Feature names: {encoder.get_feature_names_out()}")
```

### 7.3 Target Encoding with Proper Regularization

Target encoding is powerful but prone to overfitting. Proper regularization is essential.

```python
from sklearn.model_selection import KFold
import numpy as np

class RegularizedTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Target encoding with K-fold regularization to prevent leakage.
    Only uses out-of-fold target statistics for encoding.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        min_samples: int = 1,
        smoothing: float = 1.0
    ):
        """
        Args:
            n_splits: Number of folds for out-of-fold encoding
            min_samples: Minimum samples per category to trust encoding
            smoothing: Smoothing effect (higher = more global mean used)
        """
        self.n_splits = n_splits
        self.min_samples = min_samples
        self.smoothing = smoothing
        self.global_mean_ = None
        self.category_means_ = {}
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Compute global mean and category statistics."""
        self.global_mean_ = y.mean()
        
        for col in X.columns:
            stats = X.groupby(col)[y.name].agg(['mean', 'count'])
            self.category_means_[col] = stats
        
        return self
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """
        Fit and transform using K-fold to prevent leakage.
        Training data uses out-of-fold encoding.
        """
        result = np.empty((len(X), len(X.columns)))
        
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        
        for col_idx, col in enumerate(X.columns):
            result[:, col_idx] = np.nan
            
            for train_idx, val_idx in kf.split(X):
                # Compute encoding on training fold only
                X_train_fold = X.iloc[train_idx]
                y_train_fold = y.iloc[train_idx]
                
                fold_stats = X_train_fold.groupby(col)[y_train_fold.name].agg(['mean', 'count'])
                
                # Apply encoding to validation fold
                X_val_fold = X.iloc[val_idx]
                encoded = self._encode_column(X_val_fold[col], fold_stats)
                result[val_idx, col_idx] = encoded
        
        return result
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using fitted statistics.
        Uses global mean for unseen categories.
        """
        result = np.empty((len(X), len(X.columns)))
        
        for col_idx, col in enumerate(X.columns):
            if col in self.category_means_:
                result[:, col_idx] = self._encode_column(
                    X[col], 
                    self.category_means_[col]
                )
            else:
                result[:, col_idx] = self.global_mean_
        
        return result
    
    def _encode_column(
        self, 
        series: pd.Series, 
        stats: pd.DataFrame
    ) -> np.ndarray:
        """
        Encode with smoothing regularization.
        
        Formula: encoding = (count * mean + smoothing * global_mean) / (count + smoothing)
        Higher smoothing pulls encoding toward global mean.
        """
        encoded = series.map(stats['mean'])
        counts = series.map(stats['count'])
        
        # Apply smoothing
        smoothing_factor = counts / (counts + self.smoothing)
        encoded = (
            encoded * smoothing_factor + 
            self.global_mean_ * (1 - smoothing_factor)
        )
        
        # Fill unseen categories with global mean
        encoded.fillna(self.global_mean_, inplace=True)
        
        return encoded.values


# Usage:
# encoder = RegularizedTargetEncoder(n_splits=5, smoothing=10)
# X_train_encoded = encoder.fit_transform(X_train[categorical_cols], y_train)
# X_test_encoded = encoder.transform(X_test[categorical_cols])
```

### 7.4 Learned Embeddings for High Cardinality

For very high cardinality features (e.g., user IDs, product SKUs), learned embeddings are effective.

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class CategoricalEmbedding(nn.Module):
    """
    Learn dense embeddings for categorical features.
    Suitable for high-cardinality features.
    """
    
    def __init__(
        self,
        cardinalities: list,  # List of unique counts for each categorical feature
        embedding_dim: int = None,  # If None, uses min(50, cardinality // 2)
        interaction_layers: list = [64, 32]
    ):
        super().__init__()
        
        # Determine embedding dimensions
        if embedding_dim is None:
            embedding_dims = [
                min(50, (card // 2) + 1) for card in cardinalities
            ]
        else:
            embedding_dims = [embedding_dim] * len(cardinalities)
        
        # Create embedding layers
        self.embeddings = nn.ModuleList([
            nn.Embedding(card, dim) 
            for card, dim in zip(cardinalities, embedding_dims)
        ])
        
        # Interaction layers
        total_embed_dim = sum(embedding_dims)
        
        layers = []
        input_dim = total_embed_dim
        for hidden_dim in interaction_layers:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        self.interaction = nn.Sequential(*layers)
        self.output_dim = input_dim
    
    def forward(self, categorical_inputs):
        """
        Args:
            categorical_inputs: List of tensors, one per categorical feature
        Returns:
            Combined embedding vector
        """
        # Embed each categorical feature
        embedded = [
            emb(cat_input) 
            for emb, cat_input in zip(self.embeddings, categorical_inputs)
        ]
        
        # Concatenate embeddings
        combined = torch.cat(embedded, dim=1)
        
        # Apply interaction layers
        output = self.interaction(combined)
        
        return output


# Usage example:
# cardinalities = [len(X_train[col].unique()) for col in categorical_cols]
# embedding_model = CategoricalEmbedding(cardinalities)
# 
# # Convert to tensors
# cat_tensors = [
#     torch.tensor(X_train[col].astype('category').cat.codes.values, dtype=torch.long)
#     for col in categorical_cols
# ]
# 
# # Get embeddings
# with torch.no_grad():
#     embeddings = embedding_model(cat_tensors)
```

### 7.5 Encoding Best Practices by Model Type

```python
def get_encoding_recommendation(
    cardinality: int,
    model_type: str,
    is_ordinal: bool
) -> dict:
    """
    Get encoding recommendation based on data characteristics.
    """
    
    recommendations = {
        'low_cardinality_onehot': {
            'cardinality': (2, 10),
            'models': ['linear', 'tree', 'neural'],
            'encoding': 'OneHotEncoder',
            'notes': 'Safe choice, interpretable'
        },
        'medium_cardinality_ordinal': {
            'cardinality': (10, 50),
            'models': ['tree'],
            'encoding': 'OrdinalEncoder',
            'notes': 'Tree models handle ordinal encoding well'
        },
        'medium_cardinality_onehot': {
            'cardinality': (10, 50),
            'models': ['linear'],
            'encoding': 'OneHotEncoder with max_categories',
            'notes': 'Consider grouping rare categories'
        },
        'high_cardinality_target': {
            'cardinality': (50, float('inf')),
            'models': ['any'],
            'encoding': 'TargetEncoder (with regularization)',
            'notes': 'Must use out-of-fold encoding'
        },
        'high_cardinality_frequency': {
            'cardinality': (50, float('inf')),
            'models': ['tree'],
            'encoding': 'FrequencyEncoder',
            'notes': 'No leakage, simple but less informative'
        },
        'ordinal_features': {
            'cardinality': 'any',
            'models': ['any'],
            'encoding': 'OrdinalEncoder (preserve order)',
            'notes': 'Maintains semantic ordering'
        }
    }
    
    # Select based on inputs
    if is_ordinal:
        return recommendations['ordinal_features']
    elif cardinality < 10:
        return recommendations['low_cardinality_onehot']
    elif cardinality < 50:
        if model_type == 'linear':
            return recommendations['medium_cardinality_onehot']
        else:
            return recommendations['medium_cardinality_ordinal']
    else:
        if model_type == 'linear':
            return recommendations['high_cardinality_target']
        else:
            return recommendations['high_cardinality_frequency']
```

### Common Pitfalls

- **Pitfall:** Target encoding without proper regularization.
  - **Consequence:** Severe overfitting, model memorizes target.
  - **Solution:** Always use smoothing, K-fold encoding, or leave-one-out.

- **Pitfall:** One-hot encoding with linear models on high-cardinality features.
  - **Consequence:** Curse of dimensionality, overfitting, slow training.
  - **Solution:** Use target encoding or hashing trick.

- **Pitfall:** Label encoding for nominal features with linear models.
  - **Consequence:** Model learns false ordinal relationships.
  - **Solution:** Use one-hot for nominal features with linear models.

```python
# Diagnostic: Check if encoding introduced false relationships
def detect_false_ordinals(X: pd.DataFrame, col: str, y: pd.Series) -> dict:
    """
    Check if ordinal encoding may have introduced false relationships.
    """
    from scipy.stats import spearmanr
    
    # Check if encoded values correlate with target
    correlation, p_value = spearmanr(X[col], y)
    
    return {
        'correlation': correlation,
        'is_suspicious': abs(correlation) > 0.8 and p_value < 0.05,
        'recommendation': 'Review if ordinal relationship is meaningful'
    }
```

---

## Quick Reference Checklist

### Data Profiling
- [ ] Ran automated profiling on all features
- [ ] Checked missing value patterns and mechanisms
- [ ] Verified data types and ranges
- [ ] Analyzed distributions and correlations
- [ ] Computed drift metrics vs production

### Missing Values
- [ ] Assessed missingness mechanism (MCAR/MAR/MNAR)
- [ ] Chose appropriate imputation strategy
- [ ] Added missing indicators where appropriate
- [ ] Fit imputer on training data only

### Outliers
- [ ] Detected outliers using multiple methods
- [ ] Investigated outliers with domain experts
- [ ] Chose appropriate treatment (keep/winsorize/remove)
- [ ] Applied treatment after train-test split

### Data Validation
- [ ] Defined schema with type and value constraints
- [ ] Added distribution checks
- [ ] Implemented validation pipeline
- [ ] Set up monitoring and alerting

### Class Imbalance
- [ ] Assessed imbalance severity
- [ ] Selected appropriate resampling method
- [ ] Used proper metrics (F1, AUC-PR, not accuracy)
- [ ] Considered cost-sensitive learning

### Data Leakage
- [ ] Audited for target leakage
- [ ] Ensured preprocessing uses train data only
- [ ] Used proper temporal splits for time series
- [ ] Included feature selection in CV pipeline

### Categorical Encoding
- [ ] Assessed cardinality and ordinality
- [ ] Selected encoding method for each feature
- [ ] Used target encoding with regularization
- [ ] Verified encoding doesn't introduce false relationships

---

## Summary

Data quality and preprocessing form the foundation of any successful ML system. The techniques covered in this section, when applied systematically, prevent the most common causes of model failure and ensure reliable, production-ready models.

**High-Impact Priorities** (start here for quick wins):
1. Implement automated data profiling and validation
2. Set up proper train-test splits with pipelines to prevent leakage
3. Add missing indicators for MAR/MNAR data
4. Use appropriate metrics for imbalanced classification
5. Select encoding based on cardinality and model type

**Incremental Improvements** (for advanced optimization):
1. Advanced imputation (KNN, MICE) for complex missingness
2. Ensemble outlier detection for nuanced anomaly identification
3. Cost-sensitive learning for asymmetric misclassification costs
4. Learned embeddings for very high-cardinality features
5. Continuous monitoring with automated alerting

Remember: **no amount of model sophistication can compensate for poor data quality**. Invest time upfront in understanding and preparing your data—the downstream benefits are substantial and compounding.
