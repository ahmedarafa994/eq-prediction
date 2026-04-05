# Feature Engineering & Selection

## Overview

Feature engineering is the process of using domain knowledge and statistical techniques to create features that make machine learning algorithms work more effectively. This section covers automated feature synthesis, domain-specific transformations, encoding strategies, temporal features, dimensionality reduction, and feature selection methods.

## 1. Automated Feature Synthesis

### 1.1 Deep Feature Synthesis with Featuretools

**Impact: HIGH-IMPACT** - Automated feature engineering can dramatically reduce time-to-model and discover patterns humans might miss.

Featuretools is a Python library for automated feature engineering that implements Deep Feature Synthesis (DFS), which automatically generates features by recursively applying transformation and aggregation functions across related datasets.

```python
import featuretools as ft
import pandas as pd

# Prepare data with EntitySet
es = ft.EntitySet(id="customer_data")

# Add dataframes with their index and time index columns
es = es.add_dataframe(
    dataframe_name="customers",
    dataframe=customers_df,
    index="customer_id"
)
es = es.add_dataframe(
    dataframe_name="sessions", 
    dataframe=sessions_df,
    index="session_id",
    time_index="session_start"
)
es = es.add_dataframe(
    dataframe_name="transactions",
    dataframe=transactions_df,
    index="transaction_id",
    time_index="transaction_time"
)

# Define relationships
es = es.add_relationship("customers", "customer_id", "sessions", "customer_id")
es = es.add_relationship("sessions", "session_id", "transactions", "session_id")

# Run Deep Feature Synthesis
feature_matrix, features = ft.dfs(
    entityset=es,
    target_dataframe_name="customers",
    trans_primitives=["add", "subtract", "multiply", "divide", "percentile"],
    agg_primitives=["mean", "sum", "std", "max", "min", "count", "mode"],
    max_depth=2,
    cutoff_time=pd.Timestamp("2024-01-01")
)

print(f"Generated {len(features)} features")
print(feature_matrix.head())
```

**Best Practices:**

1. **Control Feature Explosion** - Use `max_depth` parameter to limit recursion depth
2. **Specify Primitives** - Choose only relevant transformation and aggregation primitives
3. **Use Cutoff Times** - For temporal data, ensure no data leakage by using cutoff times
4. **Seed Features** - Provide meaningful seed features to guide DFS

```python
# Feature selection from generated features
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(f_classif, k=50)
X_selected = selector.fit_transform(feature_matrix, y)
```

**Common Pitfalls:**

- **Overfitting**: Generating thousands of features increases overfitting risk
  - *Diagnostic*: Use cross-validation to check if selected features generalize
- **Data Leakage**: Temporal features using future information
  - *Diagnostic*: Verify cutoff times and temporal ordering
- **Computational Cost**: DFS on large datasets can be slow
  - *Diagnostic*: Monitor memory usage and generation time

**Computational Cost**: O(n × d × m) where n=samples, d=depth, m=primitives

### 1.2 Feature Stores

**Impact: INCREMENTAL** - Feature stores primarily improve operational efficiency, not model quality.

Feature stores like Feast provide centralized management for features, ensuring consistency between training and inference.

```python
from feast import FeatureStore, FeatureView, Field
from feast.types import Float32, Int32
from datetime import timedelta

# Define feature store
store = FeatureStore(repo_path=".")

# Define a feature view
customer_features = FeatureView(
    name="customer_features",
    entities=["customer_id"],
    schema=[
        Field(name="total_transactions", dtype=Int32),
        Field(name="avg_transaction_amount", dtype=Float32),
        Field(name="days_since_last_transaction", dtype=Int32),
    ],
    source=your_data_source,
    ttl=timedelta(days=30)
)

# Retrieve features for training
training_df = store.get_historical_features(
    entity_df=customer_ids_with_timestamps,
    feature_views=[customer_features]
).to_df()

# Retrieve features for inference
online_features = store.get_online_features(
    features=["customer_features:total_transactions"],
    entity_rows=[{"customer_id": 12345}]
)
```

## 2. Domain-Specific Transformations

### 2.1 Signal Processing Features

**Impact: HIGH-IMPACT** for audio, sensor, and time-series data.

```python
import numpy as np
from scipy import signal
from scipy.stats import entropy
import librosa

def extract_signal_features(audio_waveform, sr=22050):
    """Extract domain-specific features from audio/signal data."""
    features = {}
    
    # Time-domain features
    features['rms_energy'] = np.sqrt(np.mean(audio_waveform**2))
    features['zero_crossing_rate'] = np.mean(np.diff(np.sign(audio_waveform)) != 0)
    features['peak_to_peak'] = np.max(audio_waveform) - np.min(audio_waveform)
    features['crest_factor'] = np.max(np.abs(audio_waveform)) / features['rms_energy']
    
    # Frequency-domain features
    freqs, psd = signal.welch(audio_waveform, sr)
    features['spectral_centroid'] = np.sum(freqs * psd) / np.sum(psd)
    features['spectral_bandwidth'] = np.sqrt(np.sum((freqs - features['spectral_centroid'])**2 * psd) / np.sum(psd))
    features['spectral_entropy'] = entropy(psd + 1e-10)
    features['dominant_frequency'] = freqs[np.argmax(psd)]
    
    # Cepstral features (MFCCs)
    mfccs = librosa.feature.mfcc(y=audio_waveform, sr=sr, n_mfcc=13)
    features['mfcc_mean'] = np.mean(mfccs, axis=1).tolist()
    features['mfcc_std'] = np.std(mfccs, axis=1).tolist()
    
    return features
```

### 2.2 Text Features

**Impact: HIGH-IMPACT** for NLP applications.

```python
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sentence_transformers import SentenceTransformer
import numpy as np

def extract_text_features(texts, use_embeddings=True):
    """Extract features from text data."""
    
    # Traditional bag-of-words
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    tfidf_features = tfidf.fit_transform(texts)
    
    # Character n-grams (useful for language detection, style)
    char_tfidf = TfidfVectorizer(
        analyzer='char',
        ngram_range=(3, 5),
        max_features=1000
    )
    char_features = char_tfidf.fit_transform(texts)
    
    features = {
        'tfidf': tfidf_features,
        'char_tfidf': char_features
    }
    
    # Semantic embeddings (optional)
    if use_embeddings:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(texts, show_progress_bar=False)
        features['embeddings'] = embeddings
    
    # Text statistics
    stats = []
    for text in texts:
        words = text.split()
        stats.append([
            len(text),                    # character count
            len(words),                   # word count
            len(set(words)) / len(words), # unique word ratio
            np.mean([len(w) for w in words]),  # avg word length
            text.count('!'),              # exclamation count
            text.count('?'),              # question count
        ])
    features['stats'] = np.array(stats)
    
    return features
```

### 2.3 Time Series Features

**Impact: HIGH-IMPACT** for forecasting and anomaly detection.

Using tsfresh for automated time series feature extraction:

```python
from tsfresh import extract_features, select_features
from tsfresh.feature_extraction import EfficientFCParameters
import pandas as pd

def extract_timeseries_features(df, column_id, column_sort):
    """
    Extract comprehensive time series features using tsfresh.
    
    Args:
        df: DataFrame with time series data
        column_id: Column name for entity ID
        column_sort: Column name for time ordering
    """
    # Use efficient feature set (computationally reasonable)
    extraction_settings = EfficientFCParameters()
    
    # Extract features
    features = extract_features(
        df,
        column_id=column_id,
        column_sort=column_sort,
        default_fc_parameters=extraction_settings,
        n_jobs=4  # parallel processing
    )
    
    return features

# Custom time series features
def custom_ts_features(series):
    """Extract custom time series features."""
    features = {}
    
    # Trend features
    from scipy import stats
    x = np.arange(len(series))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, series)
    features['trend_slope'] = slope
    features['trend_r2'] = r_value**2
    
    # Seasonality (via autocorrelation)
    autocorr = np.correlate(series, series, mode='full')
    features['autocorr_max'] = np.max(autocorr[len(series):])
    
    # Change point detection
    diff = np.diff(series)
    features['change_points'] = np.sum(np.abs(diff) > 2 * np.std(diff))
    
    # Rolling statistics (multiple windows)
    for window in [7, 14, 30]:
        rolling = series.rolling(window=window)
        features[f'rolling_mean_{window}'] = rolling.mean().iloc[-1]
        features[f'rolling_std_{window}'] = rolling.std().iloc[-1]
        features[f'rolling_min_{window}'] = rolling.min().iloc[-1]
        features[f'rolling_max_{window}'] = rolling.max().iloc[-1]
    
    return features
```

### 2.4 Image Features

**Impact: HIGH-IMPACT** for computer vision applications.

```python
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from skimage.color import rgb2gray

def extract_image_features(image):
    """Extract hand-crafted features from images."""
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = rgb2gray(image)
    else:
        gray = image
    
    features = {}
    
    # Histogram of Oriented Gradients (HOG)
    hog_features, hog_image = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=True
    )
    features['hog'] = hog_features
    
    # Local Binary Patterns (texture)
    lbp = local_binary_pattern(gray, P=24, R=3, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 27), range=(0, 26))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    features['lbp_hist'] = hist
    
    # Color histograms
    if len(image.shape) == 3:
        for i, color in enumerate(['b', 'g', 'r']):
            hist = cv2.calcHist([image], [i], None, [64], [0, 256])
            hist = hist.flatten() / hist.sum()
            features[f'color_hist_{color}'] = hist
    
    # Statistical features
    features['mean'] = np.mean(gray)
    features['std'] = np.std(gray)
    features['skewness'] = stats.skew(gray.ravel())
    features['kurtosis'] = stats.kurtosis(gray.ravel())
    
    # Edge features
    edges = cv2.Canny((gray * 255).astype(np.uint8), 50, 150)
    features['edge_density'] = np.sum(edges > 0) / edges.size
    
    return features
```

## 3. Encoding Strategies

### 3.1 Target Encoding with Regularization

**Impact: HIGH-IMPACT** for high-cardinality categorical features.

Target encoding replaces categories with the target mean for that category. It requires regularization to prevent overfitting.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, TransformerMixin

class RegularizedTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Target encoding with smoothing and cross-validation to prevent overfitting.
    
    Smoothing formula: encoded_value = (mean_target * n_samples + global_mean * alpha) / (n_samples + alpha)
    """
    
    def __init__(self, cols=None, smoothing=10, min_samples=1):
        self.cols = cols
        self.smoothing = smoothing
        self.min_samples = min_samples
        self.target_means = {}
        self.global_mean = None
        
    def fit(self, X, y):
        self.global_mean = y.mean()
        
        if self.cols is None:
            self.cols = X.select_dtypes(include=['object', 'category']).columns
            
        for col in self.cols:
            # Calculate target mean for each category
            target_stats = y.groupby(X[col]).agg(['mean', 'count'])
            self.target_means[col] = target_stats
            
        return self
    
    def transform(self, X):
        X_encoded = X.copy()
        
        for col in self.cols:
            stats = self.target_means[col]
            
            # Apply smoothing
            def encode_category(val):
                if val in stats.index:
                    mean, count = stats.loc[val]
                    # More smoothing for categories with few samples
                    encoded = (mean * count + self.global_mean * self.smoothing) / (count + self.smoothing)
                    return encoded
                return self.global_mean
            
            X_encoded[f'{col}_encoded'] = X[col].apply(encode_category)
            
        return X_encoded
    
    def fit_transform(self, X, y):
        # Use K-fold to prevent data leakage
        X_encoded = X.copy()
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train = y.iloc[train_idx]
            
            self.fit(X_train, y_train)
            
            # Transform validation set
            for col in self.cols if self.cols is None else self.cols:
                if col not in X_encoded.columns:
                    X_encoded[f'{col}_encoded'] = np.nan
                
                stats = self.target_means[col]
                
                for idx in val_idx:
                    val = X.iloc[idx][col]
                    if val in stats.index:
                        mean, count = stats.loc[val]
                        encoded = (mean * count + self.global_mean * self.smoothing) / (count + self.smoothing)
                        X_encoded.loc[X.index[idx], f'{col}_encoded'] = encoded
                    else:
                        X_encoded.loc[X.index[idx], f'{col}_encoded'] = self.global_mean
        
        return X_encoded
```

**Common Pitfalls:**

- **Target Leakage**: Using the full dataset to compute encoding
  - *Diagnostic*: Check if validation performance significantly differs from training
  - *Solution*: Use cross-fit or K-fold encoding
- **Overfitting**: Categories with few samples have unstable encoding
  - *Diagnostic*: High variance in encoded values across folds
  - *Solution*: Increase smoothing parameter or minimum sample threshold

### 3.2 Hashing Trick

**Impact: INCREMENTAL** - Useful for very high cardinality when other methods fail.

```python
from sklearn.feature_extraction import FeatureHasher
import numpy as np

class HashingEncoder:
    """
    Use the hashing trick to encode categorical variables.
    Fixed memory footprint regardless of unique values.
    """
    
    def __init__(self, n_features=2**18, input_type='string'):
        self.hasher = FeatureHasher(
            n_features=n_features,
            input_type=input_type,
            alternate_sign=True
        )
        self.n_features = n_features
        
    def fit_transform(self, X, cols=None):
        if cols is None:
            cols = X.select_dtypes(include=['object', 'category']).columns
        
        # Convert to list of dicts for FeatureHasher
        X_list = X[cols].to_dict('records')
        
        hashed = self.hasher.transform(X_list)
        return hashed.toarray()
```

**Trade-offs:**

- **Pros**: Constant memory, handles new categories, no fitting needed
- **Cons**: Not interpretable, potential collisions, no inverse transform

### 3.3 Learned Entity Embeddings

**Impact: HIGH-IMPACT** for deep learning models with high-cardinality categorical features.

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class EntityEmbedding(nn.Module):
    """
    Learn continuous representations for categorical entities.
    Similar to word embeddings but for categorical variables.
    """
    
    def __init__(self, cardinalities, embedding_dims):
        """
        Args:
            cardinalities: List of unique values per categorical feature
            embedding_dims: List of embedding dimensions (typically min(50, cardinality//2))
        """
        super().__init__()
        
        self.embeddings = nn.ModuleList([
            nn.Embedding(card, emb_dim)
            for card, emb_dim in zip(cardinalities, embedding_dims)
        ])
        
        # Calculate output dimension
        self.output_dim = sum(embedding_dims)
        
    def forward(self, categorical_inputs):
        """
        Args:
            categorical_inputs: List of tensors, one per categorical feature
        """
        embedded = []
        for i, (embed_layer, input_tensor) in enumerate(zip(self.embeddings, categorical_inputs)):
            embedded.append(embed_layer(input_tensor))
        
        # Concatenate all embeddings
        return torch.cat(embedded, dim=1)
    
    def get_embeddings(self, feature_idx, category_idx):
        """Extract learned embedding for a specific category."""
        return self.embeddings[feature_idx].weight[category_idx].detach().numpy()

# Usage example
class TabularDatasetWithEmbeddings(Dataset):
    def __init__(self, X_num, X_cat, y):
        self.X_num = torch.FloatTensor(X_num.values)
        self.X_cat = [torch.LongTensor(X_cat[:, i].values) for i in range(X_cat.shape[1])]
        self.y = torch.FloatTensor(y.values)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return (
            self.X_num[idx],
            [cat[idx] for cat in self.X_cat],
            self.y[idx]
        )

# Example: Use embeddings in a neural network
class ModelWithEntityEmbeddings(nn.Module):
    def __init__(self, n_numerical, cardinalities, embedding_dims, hidden_dims=[64, 32]):
        super().__init__()
        
        self.embedding_layer = EntityEmbedding(cardinalities, embedding_dims)
        
        # Input dimension: numerical + concatenated embeddings
        input_dim = n_numerical + self.embedding_layer.output_dim
        
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.BatchNorm1d(dim),
                nn.Dropout(0.1)
            ])
            prev_dim = dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x_num, x_cat):
        embeddings = self.embedding_layer(x_cat)
        combined = torch.cat([x_num, embeddings], dim=1)
        return self.network(combined)
```

### 3.4 Frequency Encoding

**Impact: INCREMENTAL** - Simple and often effective baseline.

```python
def frequency_encoding(train_df, test_df, cols):
    """
    Encode categorical variables by their frequency (count).
    Useful for tree-based models.
    """
    result_train = train_df.copy()
    result_test = test_df.copy()
    
    for col in cols:
        # Calculate frequencies from training data only
        freq_map = train_df[col].value_counts(normalize=True).to_dict()
        
        # Apply to both train and test
        result_train[f'{col}_freq'] = train_df[col].map(freq_map).fillna(0)
        result_test[f'{col}_freq'] = test_df[col].map(freq_map).fillna(0)
        
        # Optional: Add missing indicator
        result_train[f'{col}_missing'] = train_df[col].isna().astype(int)
        result_test[f'{col}_missing'] = test_df[col].isna().astype(int)
    
    return result_train, result_test
```

## 4. Temporal & Interaction Features

### 4.1 Lag Features

**Impact: HIGH-IMPACT** for time series forecasting.

```python
def create_lag_features(df, columns, lags=[1, 7, 14, 30]):
    """
    Create lagged features for time series data.
    
    Args:
        df: DataFrame with datetime index
        columns: List of columns to create lags for
        lags: List of lag periods
    """
    result = df.copy()
    
    for col in columns:
        for lag in lags:
            result[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    return result

def create_rolling_features(df, columns, windows=[7, 14, 30], functions=['mean', 'std', 'min', 'max']):
    """
    Create rolling window features.
    """
    result = df.copy()
    
    for col in columns:
        for window in windows:
            rolling = df[col].rolling(window=window)
            for func in functions:
                result[f'{col}_rolling_{window}_{func}'] = getattr(rolling, func)()
    
    return result

# Exponential weighted features (captures recency better than simple rolling)
def create_ewm_features(df, columns, spans=[3, 7, 14]):
    """
    Create exponentially weighted moving average features.
    """
    result = df.copy()
    
    for col in columns:
        for span in spans:
            result[f'{col}_ewm_{span}'] = df[col].ewm(span=span, adjust=False).mean()
    
    return result
```

### 4.2 Time Since Events

**Impact: HIGH-IMPACT** for customer behavior, failure prediction, and churn modeling.

```python
def create_time_since_features(df, event_columns, reference_date_col):
    """
    Calculate time since specific events.
    
    Args:
        df: DataFrame with event columns
        event_columns: List of column names containing event dates
        reference_date_col: Column name for reference date
    """
    result = df.copy()
    
    for col in event_columns:
        # Convert to datetime if needed
        result[col] = pd.to_datetime(result[col])
        result[reference_date_col] = pd.to_datetime(result[reference_date_col])
        
        # Calculate time difference
        result[f'{col}_time_since'] = (
            result[reference_date_col] - result[col]
        ).dt.total_seconds() / 3600  # hours
        
        # Add binary indicator for missing events
        result[f'{col}_is_missing'] = result[col].isna().astype(int)
    
    return result

# Example: Time since last purchase
df['days_since_last_purchase'] = (
    df['current_date'] - df.groupby('customer_id')['purchase_date'].transform('max')
).dt.days
```

### 4.3 Polynomial Interactions

**Impact: INCREMENTAL** - Can capture non-linear relationships.

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

def create_polynomial_features(X, degree=2, interaction_only=True):
    """
    Create polynomial and interaction features.
    
    Use interaction_only=True to avoid pure powers (x^2, x^3)
    which can increase overfitting.
    """
    poly = PolynomialFeatures(
        degree=degree,
        interaction_only=interaction_only,
        include_bias=False
    )
    
    X_poly = poly.fit_transform(X)
    
    # Get feature names
    feature_names = poly.get_feature_names_out()
    
    return X_poly, feature_names

# Selective interaction features (more interpretable)
def create_ratio_features(df, numerator_cols, denominator_cols, epsilon=1e-6):
    """
    Create ratio features between columns.
    Useful for rates, efficiency metrics, etc.
    """
    result = df.copy()
    
    for num_col in numerator_cols:
        for den_col in denominator_cols:
            if num_col != den_col:
                ratio_name = f'{num_col}_per_{den_col}'
                result[ratio_name] = df[num_col] / (df[den_col] + epsilon)
                result[f'{ratio_name}_log'] = np.log1p(result[ratio_name])
    
    return result
```

### 4.4 Difference Features

**Impact: HIGH-IMPACT** for time series and trend detection.

```python
def create_difference_features(df, columns, periods=[1], seasonal_periods=None):
    """
    Create differenced features to remove trends and seasonality.
    """
    result = df.copy()
    
    for col in columns:
        for period in periods:
            result[f'{col}_diff_{period}'] = df[col].diff(periods=period)
            result[f'{col}_pct_change_{period}'] = df[col].pct_change(periods=period)
        
        # Seasonal differencing
        if seasonal_periods:
            for period in seasonal_periods:
                result[f'{col}_seasonal_diff_{period}'] = (
                    df[col] - df[col].shift(period)
                )
    
    return result
```

## 5. Dimensionality Reduction

### 5.1 PCA (Principal Component Analysis)

**Impact: HIGH-IMPACT** for high-dimensional data with linear relationships.

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

def apply_pca(X, variance_threshold=0.95, plot_explained=True):
    """
    Apply PCA with automatic component selection based on variance explained.
    """
    # Standardize first
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    # Find number of components for threshold
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumsum >= variance_threshold) + 1
    
    # Refit with optimal components
    pca_optimal = PCA(n_components=n_components)
    X_reduced = pca_optimal.fit_transform(X_scaled)
    
    if plot_explained:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), 
                pca.explained_variance_ratio_)
        plt.xlabel('Principal Component')
        plt.ylabel('Variance Explained')
        
        plt.subplot(1, 2, 2)
        plt.plot(cumsum)
        plt.axhline(y=variance_threshold, color='r', linestyle='--')
        plt.axvline(x=n_components, color='r', linestyle='--')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Variance Explained')
        plt.tight_layout()
        plt.show()
    
    return {
        'X_reduced': X_reduced,
        'pca': pca_optimal,
        'scaler': scaler,
        'n_components': n_components,
        'explained_variance': pca_optimal.explained_variance_ratio_.sum()
    }

# Incremental PCA for large datasets
from sklearn.decomposition import IncrementalPCA

def batch_pca(X, batch_size=1000, n_components=50):
    """
    Apply IncrementalPCA for datasets that don't fit in memory.
    """
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    
    # Process in batches
    X_reduced = ipca.fit_transform(X)
    
    return X_reduced, ipca
```

**When to Use PCA:**
- Features are highly correlated (multicollinearity)
- Need to reduce dimensionality for visualization or computational efficiency
- Linear relationships between features
- As preprocessing for algorithms sensitive to feature scale

**When NOT to Use PCA:**
- Features are already interpretable and meaningful
- Non-linear relationships exist
- Feature independence is important for the model

### 5.2 UMAP (Uniform Manifold Approximation and Projection)

**Impact: HIGH-IMPACT** for non-linear dimensionality reduction and visualization.

```python
import umap
from sklearn.preprocessing import StandardScaler

def apply_umap(X, n_components=2, n_neighbors=15, min_dist=0.1):
    """
    Apply UMAP for non-linear dimensionality reduction.
    
    Parameters:
    - n_neighbors: Controls local vs global structure (lower = more local)
    - min_dist: Controls how tightly points are packed (lower = tighter clusters)
    """
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit UMAP
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42,
        n_jobs=-1
    )
    X_umap = reducer.fit_transform(X_scaled)
    
    return X_umap, reducer

# UMAP for feature engineering (not just visualization)
def umap_feature_embedding(X_train, X_test, n_components=10):
    """
    Use UMAP to create new feature embeddings.
    """
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=30,
        min_dist=0.0,
        random_state=42
    )
    
    # Fit on training data only
    X_train_umap = reducer.fit_transform(X_train)
    X_test_umap = reducer.transform(X_test)
    
    # Create feature names
    feature_names = [f'umap_{i}' for i in range(n_components)]
    
    return X_train_umap, X_test_umap, feature_names
```

**Computational Cost**: O(n^1.14 * d) approximately, where n=samples, d=dimensions

### 5.3 Autoencoders

**Impact: HIGH-IMPACT** for complex, non-linear data with large sample sizes.

```python
import torch
import torch.nn as nn
import numpy as np

class Autoencoder(nn.Module):
    """
    Basic autoencoder for dimensionality reduction.
    Can be extended to variational autoencoder (VAE) for probabilistic encoding.
    """
    
    def __init__(self, input_dim, encoding_dims):
        """
        Args:
            input_dim: Original feature dimension
            encoding_dims: List of dimensions for encoding layers [hidden1, hidden2, ..., latent]
        """
        super().__init__()
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for dim in encoding_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.BatchNorm1d(dim)
            ])
            prev_dim = dim
        
        # Decoder (reverse of encoder)
        decoder_dims = encoding_dims[::-1][1:] + [input_dim]
        decoder_layers = []
        prev_dim = encoding_dims[-1]
        for dim in decoder_dims:
            decoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU() if dim != input_dim else nn.Sigmoid()
            ])
            prev_dim = dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    
    def encode(self, x):
        return self.encoder(x)

class DenoisingAutoencoder(nn.Module):
    """
    Denoising autoencoder learns robust representations by reconstructing
    from corrupted input.
    """
    
    def __init__(self, input_dim, hidden_dims, noise_factor=0.2):
        super().__init__()
        
        self.noise_factor = noise_factor
        
        # Encoder
        encoder = []
        prev_dim = input_dim
        for dim in hidden_dims:
            encoder.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = dim
        
        # Decoder
        decoder = []
        prev_dim = hidden_dims[-1]
        for dim in reversed(hidden_dims[:-1]):
            decoder.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU()
            ])
            prev_dim = dim
        decoder.append(nn.Linear(prev_dim, input_dim))
        decoder.append(nn.Sigmoid())
        
        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)
    
    def forward(self, x):
        # Add noise during training
        if self.training:
            noisy = x + self.noise_factor * torch.randn_like(x)
            noisy = torch.clamp(noisy, 0, 1)
        else:
            noisy = x
        
        encoded = self.encoder(noisy)
        decoded = self.decoder(encoded)
        return encoded, decoded

# Training function
def train_autoencoder(X_train, X_val, input_dim, encoding_dims, 
                      epochs=100, batch_size=256, lr=0.001):
    """
    Train an autoencoder and return the encoder.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = Autoencoder(input_dim, encoding_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    X_train_t = torch.FloatTensor(X_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Training
        _, decoded = model(X_train_t)
        train_loss = criterion(decoded, X_train_t)
        train_loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            _, val_decoded = model(X_val_t)
            val_loss = criterion(val_decoded, X_val_t)
        
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
    
    return model, train_losses, val_losses
```

### 5.4 t-SNE

**Impact: INCREMENTAL** - Primarily for visualization, not feature engineering.

```python
from sklearn.manifold import TSNE

def apply_tsne(X, n_components=2, perplexity=30):
    """
    Apply t-SNE for visualization.
    Note: t-SNE is stochastic - results may vary between runs.
    
    Important parameters:
    - perplexity: Related to number of nearest neighbors (5-50 typical)
    - learning_rate: Typically 10-1000
    """
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate='auto',
        init='pca',
        random_state=42,
        n_jobs=-1
    )
    
    X_tsne = tsne.fit_transform(X)
    
    return X_tsne, tsne
```

**When to Use:**
- Visualization of high-dimensional data
- Exploratory data analysis
- Understanding cluster structure

**When NOT to Use:**
- As features for downstream models (no transform method for new data)
- When reproducibility is critical (stochastic)
- Large datasets (computationally expensive)

### 5.5 Randomized Projections

**Impact: INCREMENTAL** - Fast approximation for very high-dimensional data.

```python
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection

def apply_random_projection(X, n_components='auto', eps=0.1):
    """
    Apply random projection for fast dimensionality reduction.
    
    Johnson-Lindenstrauss lemma: random projections preserve distances
    with high probability when n_components >= 4 * log(n_samples) / (eps^2 / 2 - eps^3 / 3)
    
    Parameters:
    - eps: Parameter controlling quality of embedding (smaller = better)
    """
    # Sparse projection is faster and more memory-efficient
    transformer = SparseRandomProjection(
        n_components=n_components,
        eps=eps,
        random_state=42
    )
    
    X_projected = transformer.fit_transform(X)
    
    return X_projected, transformer

# Verify distortion is within acceptable bounds
def check_projection_quality(X_original, X_projected):
    """
    Check if random projection preserves pairwise distances.
    """
    from sklearn.metrics import pairwise_distances
    
    # Original distances
    D_original = pairwise_distances(X_original)
    
    # Projected distances
    D_projected = pairwise_distances(X_projected)
    
    # Calculate distortion
    distortion = np.abs(D_original - D_projected) / D_original
    mean_distortion = np.mean(distortion)
    
    print(f"Mean distance distortion: {mean_distortion:.4f}")
    return mean_distortion
```

## 6. Feature Selection Methods

### 6.1 Filter Methods

**Impact: HIGH-IMPACT** - Fast baseline selection independent of model.

```python
from sklearn.feature_selection import (
    mutual_info_classif, mutual_info_regression,
    chi2, f_classif, f_regression,
    SelectKBest, SelectPercentile, SelectFpr, SelectFdr
)
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

def univariate_feature_selection(X, y, task='classification', k=50):
    """
    Apply multiple univariate feature selection methods.
    """
    results = {}
    
    # Mutual Information (non-linear relationships)
    if task == 'classification':
        mi_scores = mutual_info_classif(X, y, random_state=42)
    else:
        mi_scores = mutual_info_regression(X, y, random_state=42)
    results['mutual_info'] = sorted(
        zip(X.columns, mi_scores), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    # ANOVA F-test (classification)
    if task == 'classification':
        f_scores, p_values = f_classif(X, y)
        results['f_classif'] = sorted(
            zip(X.columns, f_scores, p_values),
            key=lambda x: x[1],
            reverse=True
        )
    
    # F-regression (regression)
    if task == 'regression':
        f_scores, p_values = f_regression(X, y)
        results['f_regression'] = sorted(
            zip(X.columns, f_scores, p_values),
            key=lambda x: x[1],
            reverse=True
        )
    
    # Chi-squared (requires non-negative features)
    if task == 'classification':
        X_minmax = MinMaxScaler().fit_transform(X)
        chi_scores, p_values = chi2(X_minmax, y)
        results['chi_squared'] = sorted(
            zip(X.columns, chi_scores, p_values),
            key=lambda x: x[1],
            reverse=True
        )
    
    return results

# Variance threshold (remove constant features)
def remove_low_variance_features(X, threshold=0.01):
    """
    Remove features with variance below threshold.
    For binary features, threshold = p * (1-p) where p is the allowed frequency.
    """
    from sklearn.feature_selection import VarianceThreshold
    
    selector = VarianceThreshold(threshold=threshold)
    X_filtered = selector.fit_transform(X)
    
    selected_features = X.columns[selector.get_support()]
    print(f"Removed {X.shape[1] - len(selected_features)} low-variance features")
    
    return X_filtered, selected_features
```

**Common Pitfalls:**

- **Ignoring Feature Dependencies**: Univariate methods miss feature interactions
  - *Solution*: Use as baseline, then apply wrapper/embedded methods
- **Different Scales**: Some tests assume specific distributions
  - *Solution*: Standardize before applying

### 6.2 Wrapper Methods

**Impact: HIGH-IMPACT** - Model-specific selection with better performance.

```python
from sklearn.feature_selection import RFE, RFECV, SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np

def recursive_feature_elimination(X, y, estimator=None, n_features_to_select=10):
    """
    RFE recursively removes least important features based on model weights.
    """
    if estimator is None:
        # Use linear model for stable coefficients
        estimator = LogisticRegression(max_iter=1000, penalty='l1', solver='saga')
    
    rfe = RFE(
        estimator=estimator,
        n_features_to_select=n_features_to_select,
        step=0.1  # Remove 10% of features each iteration
    )
    
    X_selected = rfe.fit_transform(X, y)
    selected_features = X.columns[rfe.support_]
    
    return X_selected, selected_features, rfe

def recursive_feature_elimination_cv(X, y, estimator=None, min_features=1, cv=5):
    """
    RFECV finds optimal number of features via cross-validation.
    """
    if estimator is None:
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)
    
    rfecv = RFECV(
        estimator=estimator,
        step=1,
        cv=cv,
        scoring='accuracy',
        min_features_to_select=min_features
    )
    
    X_selected = rfecv.fit_transform(X, y)
    selected_features = X.columns[rfecv.support_]
    
    print(f"Optimal number of features: {rfecv.n_features_}")
    
    return X_selected, selected_features, rfecv

def sequential_feature_selection(X, y, estimator=None, direction='forward', n_features=10, cv=5):
    """
    Sequential Feature Selection (SFS) - greedy forward/backward selection.
    
    Forward: Start with 0 features, add one at a time
    Backward: Start with all features, remove one at a time
    """
    if estimator is None:
        estimator = LogisticRegression(max_iter=1000)
    
    sfs = SequentialFeatureSelector(
        estimator=estimator,
        n_features_to_select=n_features,
        direction=direction,
        cv=cv,
        n_jobs=-1
    )
    
    X_selected = sfs.fit_transform(X, y)
    selected_features = X.columns[sfs.support_]
    
    return X_selected, selected_features, sfs
```

**Computational Cost**: 
- RFE: O(n_features × n_iter × fit_time)
- SFS: O(n_features^2 × cv × fit_time) - can be very expensive

### 6.3 Embedded Methods

**Impact: HIGH-IMPACT** - Most efficient as selection happens during training.

```python
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
import numpy as np

def l1_based_selection(X, y, C=0.1):
    """
    L1 regularization (Lasso) produces sparse solutions.
    Features with non-zero coefficients are selected.
    """
    # For classification
    lasso = LogisticRegression(
        penalty='l1',
        C=C,  # Inverse of regularization strength
        solver='saga',
        max_iter=1000,
        random_state=42
    )
    
    selector = SelectFromModel(
        estimator=lasso,
        threshold='mean'  # Select features with importance > mean
    )
    
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    
    return X_selected, selected_features, selector

def tree_based_selection(X, y, n_estimators=100, threshold='median'):
    """
    Tree-based models provide feature importances.
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1
    )
    
    selector = SelectFromModel(
        estimator=rf,
        threshold=threshold
    )
    
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    
    # Get feature importances
    importances = rf.feature_importances_
    feature_importance = sorted(
        zip(X.columns, importances),
        key=lambda x: x[1],
        reverse=True
    )
    
    return X_selected, selected_features, selector, feature_importance

def elastic_net_selection(X, y, alpha=1.0, l1_ratio=0.5):
    """
    Elastic Net combines L1 and L2 regularization.
    
    - l1_ratio=0: Ridge (L2 only)
    - l1_ratio=1: Lasso (L1 only)
    - 0 < l1_ratio < 1: Elastic Net (combination)
    """
    enet = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        random_state=42,
        max_iter=10000
    )
    
    selector = SelectFromModel(
        estimator=enet,
        threshold='mean'
    )
    
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    
    return X_selected, selected_features, selector
```

### 6.4 SHAP-Based Selection

**Impact: HIGH-IMPACT** - Model-agnostic, captures non-linear relationships.

```python
import shap
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

def shap_feature_selection(X_train, y_train, X_val, top_k=50):
    """
    Use SHAP values to select features.
    SHAP provides consistent, locally accurate feature importance.
    """
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)
    
    # For binary classification, shap_values is a list
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Use positive class
    
    # Calculate mean absolute SHAP value for each feature
    mean_shap = np.abs(shap_values).mean(axis=0)
    
    # Rank features
    feature_importance = sorted(
        zip(X_train.columns, mean_shap),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Select top features
    selected_features = [f[0] for f in feature_importance[:top_k]]
    
    return selected_features, feature_importance, explainer

# SHAP interaction values for detecting feature interactions
def shap_interaction_selection(X_train, y_train, X_val, top_k=30):
    """
    Use SHAP interaction values to identify important feature pairs.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    explainer = shap.TreeExplainer(model)
    shap_interaction = explainer.shap_interaction_values(X_val)
    
    if isinstance(shap_interaction, list):
        shap_interaction = shap_interaction[1]
    
    # Mean absolute interaction for each pair
    mean_interaction = np.abs(shap_interaction).mean(axis=0)
    
    # Get top interaction pairs (excluding self-interactions)
    n_features = X_train.shape[1]
    interactions = []
    
    for i in range(n_features):
        for j in range(i+1, n_features):
            interactions.append((
                X_train.columns[i],
                X_train.columns[j],
                mean_interaction[i, j]
            ))
    
    interactions = sorted(interactions, key=lambda x: x[2], reverse=True)
    
    return interactions[:top_k]
```

### 6.5 Boruta

**Impact: HIGH-IMPACT** - All-relevant feature selection using shadow features.

```python
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class Boruta:
    """
    Boruta algorithm implementation.
    Compares original features with "shadow" features (randomized copies)
    to identify statistically significant features.
    """
    
    def __init__(self, estimator=None, n_estimators=1000, max_iter=100, 
                 p_value=0.01, random_state=None):
        """
        Args:
            estimator: Sklearn estimator with feature_importances_ attribute
            n_estimators: Number of trees in random forest
            max_iter: Maximum number of iterations
            p_value: Significance level for feature selection
        """
        if estimator is None:
            estimator = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=random_state,
                n_jobs=-1
            )
        
        self.estimator = estimator
        self.max_iter = max_iter
        self.p_value = p_value
        self.random_state = random_state
        self.n_features_ = None
        self.support_ = None
        self.selected_features_ = None
        
    def fit(self, X, y):
        """
        Find all relevant features.
        """
        if isinstance(X, np.ndarray):
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=feature_names)
        
        n_features = X.shape[1]
        self.n_features_ = n_features
        
        # Initialize feature status: 0=tentative, 1=confirmed, -1=rejected
        feature_status = np.zeros(n_features, dtype=int)
        
        # Track importance history
        importance_history = []
        shadow_max_history = []
        
        for iteration in range(self.max_iter):
            # Create shadow features
            X_shadow = X.copy()
            np.random.seed(self.random_state + iteration if self.random_state else None)
            
            for col in X_shadow.columns:
                X_shadow[col] = np.random.permutation(X_shadow[col].values)
            
            # Rename shadow columns
            X_shadow.columns = [f'shadow_{i}' for i in range(n_features)]
            
            # Combine original and shadow
            X_combined = pd.concat([X, X_shadow], axis=1)
            
            # Fit model
            self.estimator.fit(X_combined, y)
            importances = self.estimator.feature_importances_
            
            # Separate original and shadow importances
            original_imp = importances[:n_features]
            shadow_imp = importances[n_features:]
            
            # Record history
            importance_history.append(original_imp)
            shadow_max_history.append(np.max(shadow_imp))
            
            # Perform Bonferroni-corrected test
            if iteration >= 10:  # Need some iterations for stability
                median_original = np.median(importance_history, axis=0)
                max_shadow = np.max(shadow_max_history)
                
                # Two-sided test
                z_scores = median_original / (np.std(importance_history, axis=0) + 1e-10)
                p_values = 2 * (1 - norm.cdf(np.abs(z_scores)))
                
                # Update status
                tentative = feature_status == 0
                confirmed = (p_values < self.p_value) & (median_original > max_shadow)
                rejected = (p_values >= self.p_value) & (median_original <= max_shadow)
                
                feature_status[tentative & confirmed] = 1
                feature_status[tentative & rejected] = -1
            
            # Check for convergence
            if np.all(feature_status != 0):
                break
        
        self.support_ = feature_status >= 0
        self.selected_features_ = X.columns[feature_status > 0].tolist()
        
        return self
    
    def transform(self, X):
        if isinstance(X, np.ndarray):
            return X[:, self.support_]
        return X.iloc[:, self.support_]
    
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

from scipy.stats import norm
```

**Common Pitfalls:**

- **Computationally Expensive**: Requires multiple full model fits
  - *Solution*: Use smaller n_estimators for initial exploration
- **Conservative**: May select more features than necessary
  - *Solution*: Follow with SelectFromModel for final selection

### 6.6 Stability Selection

**Impact: HIGH-IMPACT** - Reduces false discovery rate.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
import numpy as np

class StabilitySelection:
    """
    Stability Selection: Run feature selection on multiple subsamples
    and select features that appear consistently.
    
    Based on Meinshausen & Buhlmann (2010).
    """
    
    def __init__(self, base_estimator=None, n_iterations=100, 
                 sample_size=0.5, threshold=0.6, random_state=None):
        """
        Args:
            base_estimator: Must have coef_ attribute after fitting
            n_iterations: Number of subsample iterations
            sample_size: Fraction of data to use in each subsample
            threshold: Minimum selection probability to include feature
        """
        if base_estimator is None:
            base_estimator = LogisticRegression(
                penalty='l1',
                C=0.1,
                solver='saga',
                max_iter=1000
            )
        
        self.base_estimator = base_estimator
        self.n_iterations = n_iterations
        self.sample_size = sample_size
        self.threshold = threshold
        self.random_state = random_state
        self.selection_frequencies_ = None
        self.support_ = None
        
    def fit(self, X, y):
        """
        Run stability selection.
        """
        n_samples, n_features = X.shape
        self.selection_frequencies_ = np.zeros(n_features)
        
        for i in range(self.n_iterations):
            # Create subsample
            X_sub, y_sub = resample(
                X, y,
                n_samples=int(n_samples * self.sample_size),
                replace=False,
                random_state=self.random_state + i if self.random_state else None
            )
            
            # Fit model
            self.base_estimator.fit(X_sub, y_sub)
            
            # Count selected features (non-zero coefficients)
            selected = np.abs(self.base_estimator.coef_) > 1e-10
            if selected.ndim > 1:
                selected = selected.any(axis=0)
            
            self.selection_frequencies_ += selected
        
        # Normalize to probabilities
        self.selection_frequencies_ /= self.n_iterations
        
        # Select features above threshold
        self.support_ = self.selection_frequencies_ >= self.threshold
        
        return self
    
    def transform(self, X):
        return X[:, self.support_]
    
    def get_feature_names(self, feature_names):
        """Return selected feature names."""
        return [name for name, selected in zip(feature_names, self.support_) if selected]
```

## 7. Feature Stability Analysis

### 7.1 Measuring Feature Importance Consistency

**Impact: HIGH-IMPACT** - Ensures selected features are robust.

```python
from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

def analyze_feature_stability(X, y, feature_selector, n_splits=5, 
                               random_state=42, task='classification'):
    """
    Analyze stability of feature selection across different data splits.
    
    Returns:
    - Jaccard indices between folds
    - Selection frequency for each feature
    - Mean and std of feature importances
    """
    if task == 'classification':
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, 
                             random_state=random_state)
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, 
                   random_state=random_state)
    
    selected_features_per_fold = []
    importances_per_fold = []
    
    for fold_idx, (train_idx, _) in enumerate(kf.split(X, y)):
        X_train = X.iloc[train_idx] if isinstance(X, pd.DataFrame) else X[train_idx]
        y_train = y.iloc[train_idx] if isinstance(y, pd.Series) else y[train_idx]
        
        # Clone selector to avoid refitting issues
        from sklearn.base import clone
        selector = clone(feature_selector)
        
        # Fit and get selected features
        selector.fit(X_train, y_train)
        
        if hasattr(selector, 'get_support'):
            support = selector.get_support()
        elif hasattr(selector, 'support_'):
            support = selector.support_
        else:
            # For model-based selection, use feature importances
            importances = selector.feature_importances_
            threshold = np.mean(importances)
            support = importances > threshold
        
        selected_features_per_fold.append(set(np.where(support)[0]))
        
        # Store importances if available
        if hasattr(selector, 'feature_importances_'):
            importances_per_fold.append(selector.feature_importances_)
        elif hasattr(selector, 'estimator') and hasattr(selector.estimator, 'coef_'):
            importances_per_fold.append(np.abs(selector.estimator.coef_[0]))
    
    # Calculate Jaccard indices
    jaccard_indices = []
    for i in range(n_splits):
        for j in range(i+1, n_splits):
            intersection = len(selected_features_per_fold[i] & 
                               selected_features_per_fold[j])
            union = len(selected_features_per_fold[i] | 
                       selected_features_per_fold[j])
            jaccard = intersection / union if union > 0 else 0
            jaccard_indices.append(jaccard)
    
    # Calculate selection frequency
    n_features = X.shape[1]
    selection_freq = np.zeros(n_features)
    
    for selected in selected_features_per_fold:
        selection_freq[list(selected)] += 1
    
    selection_freq /= n_splits
    
    # Calculate correlation of importances across folds
    if importances_per_fold:
        importance_correlations = []
        for i in range(len(importances_per_fold)):
            for j in range(i+1, len(importances_per_fold)):
                corr, _ = spearmanr(importances_per_fold[i], 
                                   importances_per_fold[j], 
                                   nan_policy='omit')
                if not np.isnan(corr):
                    importance_correlations.append(corr)
    else:
        importance_correlations = None
    
    return {
        'jaccard_mean': np.mean(jaccard_indices),
        'jaccard_std': np.std(jaccard_indices),
        'selection_frequency': selection_freq,
        'importance_correlations': importance_correlations,
        'selected_per_fold': selected_features_per_fold
    }

def time_based_stability_analysis(df, feature_cols, target_col, time_col,
                                    n_windows=5, window_size=None):
    """
    Analyze feature stability across time windows.
    Useful for detecting concept drift in production.
    """
    df_sorted = df.sort_values(time_col)
    
    if window_size is None:
        window_size = len(df_sorted) // n_windows
    
    results = []
    
    for i in range(n_windows):
        start_idx = i * window_size
        end_idx = min((i + 1) * window_size, len(df_sorted))
        
        window_data = df_sorted.iloc[start_idx:end_idx]
        
        # Calculate feature statistics for this window
        stats = {}
        for col in feature_cols:
            if pd.api.types.is_numeric_dtype(window_data[col]):
                stats[f'{col}_mean'] = window_data[col].mean()
                stats[f'{col}_std'] = window_data[col].std()
                stats[f'{col}_min'] = window_data[col].min()
                stats[f'{col}_max'] = window_data[col].max()
        
        stats['window'] = i
        stats['start_time'] = window_data[time_col].min()
        stats['end_time'] = window_data[time_col].max()
        stats['n_samples'] = len(window_data)
        
        results.append(stats)
    
    stability_df = pd.DataFrame(results)
    
    # Calculate drift metrics
    drift_metrics = {}
    for col in feature_cols:
        if f'{col}_mean' in stability_df.columns:
            mean_col = f'{col}_mean'
            # Coefficient of variation across windows
            cv = stability_df[mean_col].std() / (stability_df[mean_col].mean() + 1e-10)
            drift_metrics[f'{col}_cv'] = cv
    
    return stability_df, drift_metrics
```

### 7.2 Subsampling Stability

```python
def subsampling_stability(X, y, model, n_subsamples=100, 
                          subsample_size=0.8, random_state=42):
    """
    Assess feature importance stability via random subsampling.
    
    Returns selection frequency and variance of importance.
    """
    from sklearn.utils import resample
    
    n_features = X.shape[1]
    importance_samples = np.zeros((n_subsamples, n_features))
    
    for i in range(n_subsamples):
        X_sub, y_sub = resample(
            X, y,
            n_samples=int(len(X) * subsample_size),
            replace=False,
            random_state=random_state + i if random_state else None
        )
        
        model_clone = clone(model)
        model_clone.fit(X_sub, y_sub)
        
        if hasattr(model_clone, 'feature_importances_'):
            importance_samples[i] = model_clone.feature_importances_
        elif hasattr(model_clone, 'coef_'):
            importance_samples[i] = np.abs(model_clone.coef_).flatten()
    
    # Calculate statistics
    mean_importance = importance_samples.mean(axis=0)
    std_importance = importance_samples.std(axis=0)
    cv_importance = std_importance / (mean_importance + 1e-10)
    
    # Selection frequency (times feature had above-median importance)
    threshold = mean_importance.mean()
    selection_freq = (importance_samples > threshold).mean(axis=0)
    
    return {
        'mean_importance': mean_importance,
        'std_importance': std_importance,
        'cv_importance': cv_importance,
        'selection_frequency': selection_freq,
        'importance_samples': importance_samples
    }
```

### 7.3 Cross-Feature Correlation Stability

```python
def correlation_stability_analysis(X, n_splits=5, random_state=42):
    """
    Analyze stability of feature correlations across data splits.
    Unstable correlations may indicate data quality issues.
    """
    from sklearn.model_selection import KFold
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    correlation_matrices = []
    
    for train_idx, _ in kf.split(X):
        X_fold = X.iloc[train_idx] if isinstance(X, pd.DataFrame) else X[train_idx]
        
        # Calculate correlation matrix
        if isinstance(X_fold, np.ndarray):
            X_fold_df = pd.DataFrame(X_fold)
        else:
            X_fold_df = X_fold
        
        corr_matrix = X_fold_df.corr().values
        correlation_matrices.append(corr_matrix)
    
    # Calculate stability of each correlation coefficient
    corr_array = np.array(correlation_matrices)
    
    mean_corr = corr_array.mean(axis=0)
    std_corr = corr_array.std(axis=0)
    
    # Find unstable correlations (high coefficient of variation)
    mask = ~np.eye(mean_corr.shape[0], dtype=bool)
    cv_corr = std_corr[mask] / (np.abs(mean_corr[mask]) + 1e-10)
    
    return {
        'mean_correlation': mean_corr,
        'std_correlation': std_corr,
        'unstable_correlations': cv_corr > 0.5,  # Threshold for unstable
        'cv_correlations': cv_corr
    }
```

## Summary Table

| Method | Impact | Computational Cost | Best For |
|--------|--------|-------------------|----------|
| Deep Feature Synthesis | HIGH | High | Relational/temporal data |
| Target Encoding | HIGH | Low | High-cardinality categories |
| Entity Embeddings | HIGH | Medium | Deep learning, many categories |
| PCA | HIGH | Low | Linear relationships, multicollinearity |
| UMAP | HIGH | Medium | Non-linear structure |
| Autoencoders | HIGH | High | Complex non-linear patterns |
| RFE/RFECV | HIGH | High | Model-specific optimization |
| SHAP Selection | HIGH | Medium | Model interpretation |
| Boruta | HIGH | High | All-relevant features |
| Stability Selection | HIGH | High | Robust feature sets |
| Hashing Trick | INCREMENTAL | Low | Very high cardinality |
| t-SNE | INCREMENTAL | High | Visualization only |

## Best Practices Summary

1. **Start Simple**: Begin with filter methods (variance threshold, mutual information)
2. **Domain Knowledge**: Incorporate domain-specific features before automated methods
3. **Validation**: Always validate feature selection on held-out data
4. **Stability**: Check if selected features are stable across resamples
5. **Computational Budget**: Balance feature complexity with available compute
6. **Monitoring**: Track feature distributions in production for drift
7. **Documentation**: Maintain lineage of feature transformations for reproducibility
