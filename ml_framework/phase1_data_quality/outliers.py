"""
Ensemble outlier detection and winsorization.

Combines IQR, Z-score, and Isolation Forest methods for robust outlier
detection, plus automatic winsorization.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import IsolationForest


class EnsembleOutlierDetector(BaseEstimator):
    """
    Detect outliers using an ensemble of three methods.

    Methods used:
      1. **IQR** - Values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
      2. **Z-score** - Absolute z-score > 3.
      3. **Isolation Forest** - sklearn IsolationForest with the given
         contamination.

    Parameters
    ----------
    contamination : float, default 0.05
        Expected fraction of outliers (used by Isolation Forest).

    Attributes
    ----------
    iqr_bounds_ : dict
        ``{column: (lower, upper)}`` IQR bounds from the training data.
    zscore_means_ : dict
        ``{column: mean}`` column means from training data.
    zscore_stds_ : dict
        ``{column: std}`` column stds from training data.
    iso_forest_ : IsolationForest or None
        Fitted Isolation Forest model (only fitted when numeric columns exist).
    numeric_columns_ : list of str
        Numeric columns identified at fit time.
    """

    def __init__(self, contamination: float = 0.05):
        self.contamination = contamination

    def fit(self, X: pd.DataFrame, y=None) -> "EnsembleOutlierDetector":
        """
        Fit the detector: compute IQR bounds, z-score statistics, and
        train Isolation Forest on numeric columns.

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

        self.numeric_columns_ = [
            col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])
        ]

        self.iqr_bounds_: Dict[str, Tuple[float, float]] = {}
        self.zscore_means_: Dict[str, float] = {}
        self.zscore_stds_: Dict[str, float] = {}

        for col in self.numeric_columns_:
            series = X[col].dropna()
            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1
            self.iqr_bounds_[col] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
            self.zscore_means_[col] = float(series.mean())
            self.zscore_stds_[col] = float(series.std()) if float(series.std()) > 0 else 1.0

        # Fit Isolation Forest on numeric columns
        if len(self.numeric_columns_) > 0:
            X_numeric = X[self.numeric_columns_].copy()
            # Fill NaNs with column medians for Isolation Forest fitting
            for col in self.numeric_columns_:
                X_numeric[col].fillna(X_numeric[col].median(), inplace=True)
            self.iso_forest_ = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_jobs=-1,
            )
            self.iso_forest_.fit(X_numeric.values)
        else:
            self.iso_forest_ = None

        return self

    def fit_detect(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and return outlier detection results.

        Parameters
        ----------
        X : pd.DataFrame
            Data to analyze.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
              - ``is_outlier`` (bool): True if flagged by 2+ methods.
              - ``outlier_votes`` (int): Number of methods flagging the row.
              - ``score_iqr`` (float): Fraction of numeric cols with IQR outliers.
              - ``score_zscore`` (float): Fraction of numeric cols with z-score outliers.
              - ``score_isolation`` (float): 1.0 if Isolation Forest flags the row, else 0.0.
        """
        self.fit(X)
        return self._detect(X)

    def detect(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Detect outliers using a previously fitted detector.

        Parameters
        ----------
        X : pd.DataFrame
            Data to analyze (must have same numeric columns as training).

        Returns
        -------
        pd.DataFrame
            Same schema as ``fit_detect``.
        """
        if not hasattr(self, "iqr_bounds_"):
            raise RuntimeError("Detector has not been fitted yet. Call fit() first.")
        return self._detect(X)

    def _detect(self, X: pd.DataFrame) -> pd.DataFrame:
        """Internal detection logic shared by fit_detect and detect."""
        n_rows = len(X)
        result = pd.DataFrame(index=X.index)
        result["is_outlier"] = False
        result["outlier_votes"] = 0
        result["score_iqr"] = 0.0
        result["score_zscore"] = 0.0
        result["score_isolation"] = 0.0

        if len(self.numeric_columns_) == 0:
            return result

        X_numeric = X[self.numeric_columns_].copy()

        # Fill NaNs for scoring purposes
        for col in self.numeric_columns_:
            X_numeric[col].fillna(self.zscore_means_.get(col, 0), inplace=True)

        # --- IQR scores ---
        iqr_outlier_counts = np.zeros(n_rows)
        for col in self.numeric_columns_:
            lower, upper = self.iqr_bounds_[col]
            col_outlier = ((X_numeric[col] < lower) | (X_numeric[col] > upper)).astype(float)
            iqr_outlier_counts += col_outlier.values
        result["score_iqr"] = iqr_outlier_counts / len(self.numeric_columns_)

        # --- Z-score scores ---
        zscore_outlier_counts = np.zeros(n_rows)
        for col in self.numeric_columns_:
            mean = self.zscore_means_[col]
            std = self.zscore_stds_[col]
            z_scores = np.abs((X_numeric[col].values - mean) / std)
            zscore_outlier_counts += (z_scores > 3).astype(float)
        result["score_zscore"] = zscore_outlier_counts / len(self.numeric_columns_)

        # --- Isolation Forest scores ---
        if self.iso_forest_ is not None:
            iso_preds = self.iso_forest_.predict(X_numeric.values)
            # IsolationForest: 1 = inlier, -1 = outlier
            result["score_isolation"] = (iso_preds == -1).astype(float)

        # --- Voting ---
        iqr_flag = (result["score_iqr"] > 0).astype(int)
        zscore_flag = (result["score_zscore"] > 0).astype(int)
        iso_flag = (result["score_isolation"] > 0).astype(int)
        result["outlier_votes"] = iqr_flag + zscore_flag + iso_flag
        result["is_outlier"] = result["outlier_votes"] >= 2

        return result

    def winsorize(
        self,
        X: pd.DataFrame,
        lower_percentile: float = 0.01,
        upper_percentile: float = 0.99,
    ) -> pd.DataFrame:
        """
        Clip numeric columns to the given percentile bounds (winsorization).

        Parameters
        ----------
        X : pd.DataFrame
            Data to winsorize.
        lower_percentile : float
            Lower quantile (0-1). Default 0.01 (1st percentile).
        upper_percentile : float
            Upper quantile (0-1). Default 0.99 (99th percentile).

        Returns
        -------
        pd.DataFrame
            Copy of X with clipped numeric columns.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(X).__name__}")

        X_out = X.copy()
        for col in self.numeric_columns_:
            if col not in X_out.columns:
                continue
            series = X_out[col].dropna()
            if len(series) == 0:
                continue
            lower = float(series.quantile(lower_percentile))
            upper = float(series.quantile(upper_percentile))
            X_out[col] = X_out[col].clip(lower=lower, upper=upper)

        return X_out

    def winsorize_auto(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Winsorize using automatically computed IQR-based bounds.

        Uses the bounds computed during ``fit()`` (or fits first if needed).

        Parameters
        ----------
        X : pd.DataFrame
            Data to winsorize.

        Returns
        -------
        pd.DataFrame
            Copy of X with clipped numeric columns.
        """
        if not hasattr(self, "iqr_bounds_"):
            self.fit(X)

        X_out = X.copy()
        for col in self.numeric_columns_:
            if col not in X_out.columns:
                continue
            lower, upper = self.iqr_bounds_[col]
            X_out[col] = X_out[col].clip(lower=lower, upper=upper)

        return X_out
