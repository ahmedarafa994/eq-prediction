"""
Pattern-based missing value imputation.

Provides an sklearn-compatible imputer that fills missing values using
group-level medians with a global fallback.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class PatternBasedImputer(BaseEstimator, TransformerMixin):
    """
    Impute missing values using group-level medians.

    For each column specified in ``imputation_rules``, the imputer computes
    the median of that column grouped by the specified grouping columns.
    During transform, missing values are filled with the corresponding
    group median.  If a group is unseen or the group median itself is NaN,
    the global column median (computed at fit time) is used as a fallback.

    Parameters
    ----------
    imputation_rules : dict
        Mapping of column_name -> list of group column names.
        Example: ``{"income": ["education", "region"], "age": ["gender"]}``
        Only the keys of this dict will be imputed; other columns pass
        through unchanged.

    Attributes
    ----------
    group_medians_ : dict
        Nested dict: ``{column: {group_tuple: median_value}}``.
    global_medians_ : dict
        ``{column: global_median_value}`` for fallback.
    """

    def __init__(self, imputation_rules: Dict[str, List[str]]):
        self.imputation_rules = imputation_rules

    def fit(self, X: pd.DataFrame, y=None) -> "PatternBasedImputer":
        """
        Compute group-level and global medians for each column in rules.

        Parameters
        ----------
        X : pd.DataFrame
            Training data.
        y : ignored

        Returns
        -------
        self
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(X).__name__}")

        self.group_medians_: Dict[str, Dict[tuple, float]] = {}
        self.global_medians_: Dict[str, float] = {}
        self._feature_names_out = list(X.columns)

        for col, group_cols in self.imputation_rules.items():
            if col not in X.columns:
                raise ValueError(
                    f"Column '{col}' from imputation_rules not found in DataFrame. "
                    f"Available columns: {list(X.columns)}"
                )

            for gc in group_cols:
                if gc not in X.columns:
                    raise ValueError(
                        f"Group column '{gc}' for target column '{col}' not found "
                        f"in DataFrame. Available columns: {list(X.columns)}"
                    )

            # Global median of the column (ignoring NaNs)
            self.global_medians_[col] = float(X[col].median())

            # Group-level medians
            grouped = X.groupby(group_cols, dropna=False)[col].median()
            self.group_medians_[col] = {}
            for key in grouped.index:
                # Convert scalar keys to tuples for uniform handling
                group_key = key if isinstance(key, tuple) else (key,)
                self.group_medians_[col][group_key] = float(grouped.loc[key])

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing values using group medians with global fallback.

        Parameters
        ----------
        X : pd.DataFrame
            Data with missing values.

        Returns
        -------
        pd.DataFrame
            Copy of X with missing values imputed.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(X).__name__}")

        X_out = X.copy()

        for col, group_cols in self.imputation_rules.items():
            if col not in X_out.columns:
                continue

            missing_mask = X_out[col].isnull()
            if not missing_mask.any():
                continue

            for idx in X_out.index[missing_mask]:
                # Build the group key from the group columns for this row
                group_values = tuple(X_out.loc[idx, gc] for gc in group_cols)

                # Try group median first
                group_median = self.group_medians_[col].get(group_values, None)

                if group_median is not None and not np.isnan(group_median):
                    X_out.loc[idx, col] = group_median
                else:
                    # Fall back to global median
                    X_out.loc[idx, col] = self.global_medians_[col]

        return X_out

    def get_feature_names_out(self, input_features=None):
        """Return feature names (passthrough, columns unchanged)."""
        if input_features is not None:
            return np.array(input_features)
        return np.array(self._feature_names_out)
