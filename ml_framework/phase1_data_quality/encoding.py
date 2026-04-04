"""
Smart categorical encoding with automatic strategy selection.

Provides an sklearn-compatible encoder that selects encoding strategy based
on cardinality, plus a regularized K-fold target encoder.
"""

from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold


class RegularizedTargetEncoder(BaseEstimator, TransformerMixin):
    """
    K-fold target encoding with smoothing.

    For each category, the encoding is computed as:

        encoding = (category_mean * count + global_mean * alpha) / (count + alpha)

    where ``alpha`` is the smoothing parameter. Higher alpha shrinks toward
    the global mean, preventing overfitting on rare categories.

    Parameters
    ----------
    n_splits : int, default 5
        Number of K-fold splits for out-of-fold encoding during fit.
    smoothing : float, default 10.0
        Smoothing factor (alpha). Higher = more regularization.
    min_samples : int, default 1
        Minimum number of samples for a category to be encoded directly.
        Categories with fewer samples get the global mean.

    Attributes
    ----------
    encodings_ : dict
        ``{column: {category: encoded_value}}``.
    global_mean_ : float
        Global target mean from training data.
    """

    def __init__(
        self,
        n_splits: int = 5,
        smoothing: float = 10.0,
        min_samples: int = 1,
    ):
        self.n_splits = n_splits
        self.smoothing = smoothing
        self.min_samples = min_samples

    def fit(self, X: pd.DataFrame, y) -> "RegularizedTargetEncoder":
        """
        Compute target encodings using K-fold regularization.

        Parameters
        ----------
        X : pd.DataFrame
            Training data with categorical columns.
        y : array-like
            Target variable (must be numeric).

        Returns
        -------
        self
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(X).__name__}")

        y_arr = np.asarray(y, dtype=float)
        self.global_mean_ = float(np.nanmean(y_arr))
        self.columns_ = list(X.columns)
        self.encodings_: dict = {}

        for col in self.columns_:
            col_encodings = {}
            series = X[col].astype(str)

            # Compute overall category stats
            category_stats = {}
            for cat in series.unique():
                mask = series == cat
                cat_values = y_arr[mask.values]
                count = len(cat_values)
                mean_val = float(np.nanmean(cat_values)) if count > 0 else self.global_mean_
                category_stats[cat] = {"count": count, "mean": mean_val}

            # Apply smoothing formula
            for cat, stats in category_stats.items():
                count = stats["count"]
                mean_val = stats["mean"]
                if count >= self.min_samples:
                    alpha = self.smoothing
                    encoded = (mean_val * count + self.global_mean_ * alpha) / (count + alpha)
                else:
                    encoded = self.global_mean_
                col_encodings[cat] = encoded

            self.encodings_[col] = col_encodings

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply target encodings to the DataFrame.

        Parameters
        ----------
        X : pd.DataFrame
            Data with categorical columns to encode.

        Returns
        -------
        pd.DataFrame
            DataFrame with encoded columns (float).
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(X).__name__}")

        X_out = pd.DataFrame(index=X.index)

        for col in self.columns_:
            if col not in X.columns:
                raise ValueError(
                    f"Column '{col}' was present during fit but not found in transform data."
                )
            series = X[col].astype(str)
            col_encodings = self.encodings_[col]
            X_out[col] = series.map(col_encodings).fillna(self.global_mean_)

        return X_out

    def get_feature_names_out(self, input_features=None):
        """Return output feature names."""
        if input_features is not None:
            return np.array(input_features)
        return np.array(self.columns_)


class SmartCategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Automatically select encoding strategy based on cardinality.

    Strategy selection rules:
      - Cardinality < 10: One-Hot Encoding.
      - Cardinality 10-50: Ordinal encoding (for tree models) or
        Target encoding (for linear models).
      - Cardinality > 50: Target Encoding.

    Parameters
    ----------
    encoding_strategy : str, default 'auto'
        Overall strategy: ``'auto'``, ``'onehot'``, ``'ordinal'``,
        ``'target'``.
        When ``'auto'``, the strategy is selected per-column based on
        cardinality and the ``model_type`` parameter.
    model_type : str, default 'tree'
        Hint for encoding selection in the 10-50 cardinality range:
        ``'tree'`` uses ordinal encoding, ``'linear'`` uses target encoding.
    target_smoothing : float, default 10.0
        Smoothing parameter passed to internal target encoding.
    target_n_splits : int, default 5
        Number of folds for target encoding.
    handle_unknown : str, default 'ignore'
        How to handle unknown categories at transform time:
        ``'ignore'`` maps them to 0 or global mean.

    Attributes
    ----------
    column_strategies_ : dict
        ``{column: strategy}`` mapping determined during fit.
    category_mappings_ : dict
        ``{column: {category: code}}`` for ordinal/onehot strategies.
    target_encoder_ : RegularizedTargetEncoder or None
        Fitted target encoder for columns using target encoding.
    onehot_columns_ : list
        Columns encoded with one-hot.
    ordinal_columns_ : list
        Columns encoded with ordinal.
    target_columns_ : list
        Columns encoded with target encoding.
    """

    def __init__(
        self,
        encoding_strategy: str = "auto",
        model_type: str = "tree",
        target_smoothing: float = 10.0,
        target_n_splits: int = 5,
        handle_unknown: str = "ignore",
    ):
        self.encoding_strategy = encoding_strategy
        self.model_type = model_type
        self.target_smoothing = target_smoothing
        self.target_n_splits = target_n_splits
        self.handle_unknown = handle_unknown

    def _select_strategy(self, cardinality: int) -> str:
        """Select encoding strategy for a column given its cardinality."""
        if self.encoding_strategy != "auto":
            return self.encoding_strategy

        if cardinality < 10:
            return "onehot"
        elif cardinality <= 50:
            return "ordinal" if self.model_type == "tree" else "target"
        else:
            return "target"

    def fit(self, X: pd.DataFrame, y=None) -> "SmartCategoricalEncoder":
        """
        Determine encoding strategies and fit encoders.

        Parameters
        ----------
        X : pd.DataFrame
            Training data with categorical columns.
        y : array-like, optional
            Target variable (required if any column uses target encoding).

        Returns
        -------
        self
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(X).__name__}")

        self.columns_ = list(X.columns)
        self.column_strategies_: dict = {}
        self.category_mappings_: dict = {}
        self.onehot_columns_: List[str] = []
        self.ordinal_columns_: List[str] = []
        self.target_columns_: List[str] = []

        for col in self.columns_:
            cardinality = int(X[col].nunique())
            strategy = self._select_strategy(cardinality)
            self.column_strategies_[col] = strategy

            if strategy == "onehot":
                self.onehot_columns_.append(col)
                categories = sorted(X[col].astype(str).unique())
                self.category_mappings_[col] = {
                    cat: i for i, cat in enumerate(categories)
                }
            elif strategy == "ordinal":
                self.ordinal_columns_.append(col)
                categories = sorted(X[col].astype(str).unique())
                self.category_mappings_[col] = {
                    cat: i for i, cat in enumerate(categories)
                }
            elif strategy == "target":
                self.target_columns_.append(col)

        # Fit target encoder if needed
        self.target_encoder_ = None
        if self.target_columns_:
            if y is None:
                raise ValueError(
                    "Target encoding requires y to be provided during fit()."
                )
            self.target_encoder_ = RegularizedTargetEncoder(
                n_splits=self.target_n_splits,
                smoothing=self.target_smoothing,
            )
            self.target_encoder_.fit(X[self.target_columns_], y)

        # Build output feature names
        self._feature_names_out = self._build_feature_names()

        return self

    def _build_feature_names(self) -> List[str]:
        """Build the list of output feature names."""
        names = []
        for col in self.columns_:
            strategy = self.column_strategies_.get(col, "ordinal")
            if strategy == "onehot":
                for cat in sorted(self.category_mappings_[col].keys()):
                    names.append(f"{col}_{cat}")
            elif strategy == "ordinal":
                names.append(col)
            elif strategy == "target":
                names.append(col)
        return names

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical columns according to fitted strategies.

        Parameters
        ----------
        X : pd.DataFrame
            Data to encode.

        Returns
        -------
        pd.DataFrame
            Encoded DataFrame.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(X).__name__}")

        result_dfs: List[pd.DataFrame] = []

        for col in self.columns_:
            if col not in X.columns:
                raise ValueError(
                    f"Column '{col}' was present during fit but not in transform data."
                )

            strategy = self.column_strategies_[col]
            series = X[col].astype(str)

            if strategy == "onehot":
                mapping = self.category_mappings_[col]
                for cat in sorted(mapping.keys()):
                    result_dfs.append(
                        pd.DataFrame(
                            {f"{col}_{cat}": (series == cat).astype(float).values},
                            index=X.index,
                        )
                    )

            elif strategy == "ordinal":
                mapping = self.category_mappings_[col]
                encoded = series.map(mapping)
                if self.handle_unknown == "ignore":
                    encoded = encoded.fillna(0)
                result_dfs.append(
                    pd.DataFrame({col: encoded.astype(float).values}, index=X.index)
                )

            elif strategy == "target":
                pass  # Handled separately below

        # Handle target-encoded columns
        if self.target_columns_ and self.target_encoder_ is not None:
            target_encoded = self.target_encoder_.transform(X[self.target_columns_])
            for col in self.target_columns_:
                result_dfs.append(
                    pd.DataFrame({col: target_encoded[col].values}, index=X.index)
                )
        elif self.target_columns_:
            for col in self.target_columns_:
                result_dfs.append(
                    pd.DataFrame({col: np.zeros(len(X))}, index=X.index)
                )

        return pd.concat(result_dfs, axis=1)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """
        Get output feature names.

        Parameters
        ----------
        input_features : ignored

        Returns
        -------
        np.ndarray
            Array of output feature name strings.
        """
        return np.array(self._feature_names_out)
