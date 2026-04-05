"""
Temporal and interaction feature engineering.

Provides functions for creating lag, rolling, EWM, difference,
polynomial interaction, and ratio features.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures


def create_lag_features(df, columns, lags=None):
    """Create lagged features for time series data."""
    if lags is None:
        lags = [1, 7, 14, 30]
    result = df.copy()
    for col in columns:
        for lag in lags:
            result[f"{col}_lag_{lag}"] = df[col].shift(lag)
    return result


def create_rolling_features(df, columns, windows=None, functions=None):
    """Create rolling window features."""
    if windows is None:
        windows = [7, 14, 30]
    if functions is None:
        functions = ["mean", "std", "min", "max"]
    result = df.copy()
    for col in columns:
        for window in windows:
            rolling = df[col].rolling(window=window)
            for func in functions:
                result[f"{col}_rolling_{window}_{func}"] = getattr(rolling, func)()
    return result


def create_ewm_features(df, columns, spans=None):
    """Create exponentially weighted moving average features."""
    if spans is None:
        spans = [3, 7, 14]
    result = df.copy()
    for col in columns:
        for span in spans:
            result[f"{col}_ewm_{span}"] = df[col].ewm(span=span, adjust=False).mean()
    return result


def create_difference_features(df, columns, periods=None, seasonal_periods=None):
    """Create differenced features to remove trends and seasonality."""
    if periods is None:
        periods = [1]
    result = df.copy()
    for col in columns:
        for period in periods:
            result[f"{col}_diff_{period}"] = df[col].diff(periods=period)
            result[f"{col}_pct_change_{period}"] = df[col].pct_change(periods=period)
        if seasonal_periods:
            for period in seasonal_periods:
                result[f"{col}_seasonal_diff_{period}"] = df[col] - df[col].shift(period)
    return result


def create_polynomial_features(X, degree=2, interaction_only=True):
    """Create polynomial and interaction features.

    Returns (X_poly, feature_names).
    """
    poly = PolynomialFeatures(
        degree=degree, interaction_only=interaction_only, include_bias=False
    )
    X_poly = poly.fit_transform(X)
    feature_names = poly.get_feature_names_out()
    return X_poly, feature_names


def create_ratio_features(df, numerator_cols, denominator_cols, epsilon=1e-6):
    """Create ratio features between columns with log transform."""
    result = df.copy()
    for num_col in numerator_cols:
        for den_col in denominator_cols:
            if num_col != den_col:
                ratio_name = f"{num_col}_per_{den_col}"
                result[ratio_name] = df[num_col] / (df[den_col] + epsilon)
                result[f"{ratio_name}_log"] = np.log1p(result[ratio_name])
    return result
