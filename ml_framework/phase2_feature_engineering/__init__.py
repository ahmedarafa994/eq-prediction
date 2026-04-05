"""Phase 2: Feature Engineering & Selection."""

from .automated_features import auto_feature_synthesis, custom_ts_features
from .domain_features import extract_signal_features, extract_text_features, extract_image_features
from .encoding_strategies import HashingEncoder, EntityEmbedding, frequency_encoding
from .temporal_features import (
    create_lag_features, create_rolling_features, create_ewm_features,
    create_difference_features, create_polynomial_features, create_ratio_features,
)
from .dimensionality_reduction import apply_pca, batch_pca, Autoencoder, DenoisingAutoencoder, train_autoencoder
from .feature_selection import (
    univariate_feature_selection, remove_low_variance_features,
    recursive_feature_elimination, recursive_feature_elimination_cv,
    l1_based_selection, tree_based_selection, shap_feature_selection,
)
from .stability import (
    analyze_feature_stability, time_based_stability_analysis,
    subsampling_stability, correlation_stability_analysis,
)
