"""
Comprehensive data profiling module.

Provides detailed statistical profiles of DataFrames including per-column
statistics, outlier detection, and automated issue detection.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


def _profile_numeric_column(series: pd.Series) -> Dict[str, Any]:
    """Compute detailed statistics for a numeric column."""
    non_null = series.dropna()
    n_null = series.isnull().sum()
    n_total = len(series)

    result: Dict[str, Any] = {
        "dtype": str(series.dtype),
        "null_count": int(n_null),
        "null_pct": round(float(n_null / n_total) * 100, 2) if n_total > 0 else 0.0,
        "unique": int(series.nunique()),
    }

    if len(non_null) == 0:
        result.update({
            "min": None, "max": None, "mean": None, "median": None,
            "std": None, "skewness": None, "kurtosis": None,
            "zeros": 0, "zeros_pct": 0.0,
            "negatives": 0, "negatives_pct": 0.0,
            "outlier_count_iqr": 0, "outlier_pct_iqr": 0.0,
            "iqr_lower": None, "iqr_upper": None,
        })
        return result

    q1 = float(non_null.quantile(0.25))
    q3 = float(non_null.quantile(0.75))
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outlier_mask = (non_null < lower_bound) | (non_null > upper_bound)
    outlier_count = int(outlier_mask.sum())

    zeros = int((non_null == 0).sum())
    negatives = int((non_null < 0).sum())

    skewness = float(sp_stats.skew(non_null.values, nan_policy="omit"))
    kurtosis = float(sp_stats.kurtosis(non_null.values, nan_policy="omit", fisher=True))

    result.update({
        "min": float(non_null.min()),
        "max": float(non_null.max()),
        "mean": float(non_null.mean()),
        "median": float(non_null.median()),
        "std": float(non_null.std()),
        "skewness": round(skewness, 4),
        "kurtosis": round(kurtosis, 4),
        "zeros": zeros,
        "zeros_pct": round(float(zeros / n_total) * 100, 2),
        "negatives": negatives,
        "negatives_pct": round(float(negatives / n_total) * 100, 2),
        "outlier_count_iqr": outlier_count,
        "outlier_pct_iqr": round(float(outlier_count / n_total) * 100, 2),
        "iqr_lower": round(lower_bound, 4),
        "iqr_upper": round(upper_bound, 4),
    })

    return result


def _profile_categorical_column(series: pd.Series) -> Dict[str, Any]:
    """Compute detailed statistics for a categorical/object column."""
    n_null = series.isnull().sum()
    n_total = len(series)
    non_null = series.dropna()
    unique_count = int(series.nunique())

    result: Dict[str, Any] = {
        "dtype": str(series.dtype),
        "null_count": int(n_null),
        "null_pct": round(float(n_null / n_total) * 100, 2) if n_total > 0 else 0.0,
        "unique": unique_count,
    }

    if len(non_null) == 0:
        result.update({"most_common": None, "rare_categories": []})
        return result

    value_counts = non_null.value_counts()
    most_common_value = str(value_counts.index[0])
    most_common_count = int(value_counts.iloc[0])
    result["most_common"] = {
        "value": most_common_value,
        "count": most_common_count,
        "frequency": round(float(most_common_count / len(non_null)) * 100, 2),
    }

    # Rare categories: those appearing in less than 1% of rows
    threshold = max(1, int(len(non_null) * 0.01))
    rare = value_counts[value_counts < threshold]
    result["rare_categories"] = [
        {"value": str(cat), "count": int(cnt)}
        for cat, cnt in rare.items()
    ]

    return result


def _detect_issues(
    profile: Dict[str, Any],
    target: Optional[str],
) -> List[Dict[str, Any]]:
    """Detect data quality issues from the computed profile."""
    issues: List[Dict[str, Any]] = []

    for col_name, col_profile in profile["columns"].items():
        null_pct = col_profile.get("null_pct", 0)
        unique = col_profile.get("unique", 0)
        n_rows = profile["shape"][0]

        # High missingness
        if null_pct > 50:
            issues.append({
                "column": col_name,
                "type": "high_missingness",
                "severity": "high",
                "detail": f"{null_pct}% missing values",
            })

        # Low missingness (may still need attention)
        if 0 < null_pct < 5:
            issues.append({
                "column": col_name,
                "type": "low_missingness",
                "severity": "low",
                "detail": f"{null_pct}% missing values (small but present)",
            })

        # Constant column
        if unique <= 1:
            issues.append({
                "column": col_name,
                "type": "constant_column",
                "severity": "medium",
                "detail": f"Only {unique} unique value(s)",
            })

        # High cardinality (for categorical-like columns)
        if col_profile.get("dtype", "").startswith(("object", "category", "string")):
            if n_rows > 0 and unique / n_rows > 0.95:
                issues.append({
                    "column": col_name,
                    "type": "high_cardinality",
                    "severity": "medium",
                    "detail": f"{unique} unique values out of {n_rows} rows ({round(unique / n_rows * 100, 1)}%)",
                })

        # Skewness issues (numeric only)
        if "skewness" in col_profile and col_profile["skewness"] is not None:
            abs_skew = abs(col_profile["skewness"])
            if abs_skew > 3:
                issues.append({
                    "column": col_name,
                    "type": "high_skewness",
                    "severity": "medium",
                    "detail": f"Skewness = {col_profile['skewness']} (|skew| > 3)",
                })
            elif abs_skew > 1:
                issues.append({
                    "column": col_name,
                    "type": "moderate_skewness",
                    "severity": "low",
                    "detail": f"Skewness = {col_profile['skewness']} (1 < |skew| <= 3)",
                })

        # Outlier issues (numeric only)
        if col_profile.get("outlier_pct_iqr", 0) > 10:
            issues.append({
                "column": col_name,
                "type": "high_outlier_rate",
                "severity": "medium",
                "detail": f"{col_profile['outlier_pct_iqr']}% outliers by IQR method",
            })

    # Class imbalance (if target provided and is categorical-ish)
    if target is not None and target in profile["columns"]:
        col_profile = profile["columns"][target]
        if col_profile.get("dtype", "").startswith(("object", "category", "string", "int")):
            n_unique = col_profile.get("unique", 0)
            if 2 <= n_unique <= 20:
                # Check class imbalance
                # We need to re-examine the raw data; use the profile info
                most_common_freq = col_profile.get("most_common", {}).get("frequency", 0)
                if most_common_freq > 0:
                    n_classes = n_unique
                    expected_freq = 100.0 / n_classes
                    if most_common_freq > expected_freq * 3:
                        issues.append({
                            "column": target,
                            "type": "class_imbalance",
                            "severity": "high",
                            "detail": f"Dominant class at {most_common_freq}% across {n_classes} classes",
                        })
                    elif most_common_freq > expected_freq * 2:
                        issues.append({
                            "column": target,
                            "type": "class_imbalance",
                            "severity": "medium",
                            "detail": f"Dominant class at {most_common_freq}% across {n_classes} classes",
                        })

    return issues


def comprehensive_profile(
    df: pd.DataFrame,
    target: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Produce a comprehensive profile of a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The data to profile.
    target : str, optional
        Name of the target column. Enables class-imbalance detection.

    Returns
    -------
    dict
        Keys: shape, memory_usage_mb, columns (per-column stats), issues.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__}")

    profile: Dict[str, Any] = {
        "shape": df.shape,
        "memory_usage_mb": round(float(df.memory_usage(deep=True).sum() / 1024 / 1024), 3),
        "columns": {},
    }

    numeric_dtypes = {"int8", "int16", "int32", "int64", "uint8", "uint16",
                      "uint32", "uint64", "float16", "float32", "float64",
                      "float128"}
    for col in df.columns:
        series = df[col]
        dtype_str = str(series.dtype)

        # Determine if numeric
        is_numeric = (
            pd.api.types.is_numeric_dtype(series)
            or dtype_str in numeric_dtypes
        )

        if is_numeric:
            profile["columns"][col] = _profile_numeric_column(series)
            profile["columns"][col]["type"] = "numeric"
        else:
            profile["columns"][col] = _profile_categorical_column(series)
            profile["columns"][col]["type"] = "categorical"

    profile["issues"] = _detect_issues(profile, target)

    return profile
