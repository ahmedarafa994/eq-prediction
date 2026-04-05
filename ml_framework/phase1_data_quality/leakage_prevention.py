"""
Data leakage prevention utilities.

Provides tools to audit train/test splits for data leakage, perform
temporal splits, and create temporal cross-validation folds.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


def leakage_audit(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Dict[str, object]:
    """
    Audit train/test splits for common data leakage patterns.

    Checks performed:
      1. **Duplicate rows** between train and test.
      2. **Target distribution shift** using the Kolmogorov-Smirnov test
         (for numeric targets) or chi-squared test (for categorical).
      3. **Feature-target leakage**: features with absolute correlation
         > 0.95 with the target (numeric targets only).

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    X_test : pd.DataFrame
        Test features.
    y_train : pd.Series
        Training target.
    y_test : pd.Series
        Test target.

    Returns
    -------
    dict
        Keys:
          - ``issues_found`` (list of dict): Each issue has keys
            ``type``, ``severity``, ``detail``.
          - ``warnings`` (list of str): Non-critical observations.
          - ``summary`` (str): Human-readable summary.
    """
    issues: List[Dict[str, str]] = []
    warnings: List[str] = []

    # --- 1. Duplicate rows between train and test ---
    train_records = X_train.to_dict(orient="records")
    test_records = X_test.to_dict(orient="records")

    # Use a hash-based approach for efficiency
    train_hashes = set()
    for rec in train_records:
        try:
            h = hash(tuple(sorted(rec.items())))
            train_hashes.add(h)
        except TypeError:
            # Unhashable values; skip this record
            pass

    duplicate_count = 0
    for rec in test_records:
        try:
            h = hash(tuple(sorted(rec.items())))
            if h in train_hashes:
                duplicate_count += 1
        except TypeError:
            pass

    if duplicate_count > 0:
        pct = round(duplicate_count / len(X_test) * 100, 2)
        issues.append({
            "type": "duplicate_rows",
            "severity": "high",
            "detail": (
                f"{duplicate_count} test rows ({pct}%) appear in training set. "
                f"This is a direct data leak."
            ),
        })
    else:
        warnings.append("No duplicate rows found between train and test sets.")

    # --- 2. Target distribution shift ---
    y_tr = np.asarray(y_train, dtype=float)
    y_te = np.asarray(y_test, dtype=float)

    # Try numeric KS test first
    try:
        ks_stat, ks_pvalue = sp_stats.ks_2samp(y_tr, y_te)
        if ks_pvalue < 0.01:
            issues.append({
                "type": "target_distribution_shift",
                "severity": "medium",
                "detail": (
                    f"KS test statistic={ks_stat:.4f}, p-value={ks_pvalue:.6f}. "
                    f"Target distributions differ significantly between "
                    f"train and test (p < 0.01)."
                ),
            })
        elif ks_pvalue < 0.05:
            warnings.append(
                f"Marginal target distribution shift detected: "
                f"KS stat={ks_stat:.4f}, p-value={ks_pvalue:.4f}."
            )
        else:
            warnings.append(
                f"Target distributions appear similar "
                f"(KS stat={ks_stat:.4f}, p-value={ks_pvalue:.4f})."
            )
    except (ValueError, TypeError):
        # Categorical target: use chi-squared test
        try:
            y_tr_cat = np.asarray(y_train)
            y_te_cat = np.asarray(y_test)
            all_classes = np.unique(np.concatenate([y_tr_cat, y_te_cat]))
            train_counts = np.array([np.sum(y_tr_cat == c) for c in all_classes])
            test_counts = np.array([np.sum(y_te_cat == c) for c in all_classes])
            # Normalize to expected frequencies
            test_expected = test_counts.sum() * train_counts / train_counts.sum()
            chi2_stat, chi2_pvalue = sp_stats.chisquare(test_counts, f_exp=test_expected)
            if chi2_pvalue < 0.01:
                issues.append({
                    "type": "target_distribution_shift",
                    "severity": "medium",
                    "detail": (
                        f"Chi-squared test: stat={chi2_stat:.4f}, "
                        f"p-value={chi2_pvalue:.6f}. "
                        f"Class proportions differ significantly."
                    ),
                })
            else:
                warnings.append(
                    f"Class distributions appear similar "
                    f"(chi2 stat={chi2_stat:.4f}, p={chi2_pvalue:.4f})."
                )
        except Exception:
            warnings.append("Could not assess target distribution shift.")

    # --- 3. Feature-target leakage ---
    y_tr_series = pd.Series(y_train, index=X_train.index)
    numeric_cols = [
        col for col in X_train.columns
        if pd.api.types.is_numeric_dtype(X_train[col])
    ]

    if pd.api.types.is_numeric_dtype(y_tr_series) and len(numeric_cols) > 0:
        for col in numeric_cols:
            series = X_train[col]
            # Need enough non-null values and some variance
            valid = series.notna() & y_tr_series.notna()
            if valid.sum() < 10:
                continue
            if series[valid].std() == 0:
                continue
            corr = np.abs(np.corrcoef(
                series[valid].values,
                y_tr_series[valid].values,
            )[0, 1])
            if not np.isnan(corr) and corr > 0.95:
                issues.append({
                    "type": "feature_target_leakage",
                    "severity": "high",
                    "detail": (
                        f"Feature '{col}' has absolute correlation {corr:.4f} "
                        f"with target (>0.95). Possible target leak."
                    ),
                })

    # --- Summary ---
    n_high = sum(1 for i in issues if i["severity"] == "high")
    n_medium = sum(1 for i in issues if i["severity"] == "medium")
    summary = (
        f"Leakage audit found {len(issues)} issue(s): "
        f"{n_high} high severity, {n_medium} medium severity. "
        f"{len(warnings)} warning(s)."
    )

    return {
        "issues_found": issues,
        "warnings": warnings,
        "summary": summary,
    }


def temporal_train_test_split(
    df: pd.DataFrame,
    date_column: str,
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame temporally into train, validation, and test sets.

    The data is sorted by ``date_column`` and split chronologically:
      - Last ``test_size`` fraction -> test
      - The portion before that is split 80/20 into train/val.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``date_column``.
    date_column : str
        Column with sortable date/timestamp values.
    test_size : float, default 0.2
        Fraction of data to reserve for the test set.

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame, pd.DataFrame)
        ``(train_df, val_df, test_df)`` in chronological order.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__}")
    if date_column not in df.columns:
        raise ValueError(f"Column '{date_column}' not found in DataFrame.")

    df_sorted = df.sort_values(date_column).reset_index(drop=True)
    n = len(df_sorted)

    test_start = int(n * (1 - test_size))
    val_start = int(test_start * 0.8)

    train_df = df_sorted.iloc[:val_start].copy()
    val_df = df_sorted.iloc[val_start:test_start].copy()
    test_df = df_sorted.iloc[test_start:].copy()

    return train_df, val_df, test_df


def create_temporal_cv(
    train_df: pd.DataFrame,
    date_column: str,
    n_splits: int = 5,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create temporal cross-validation folds from a training DataFrame.

    Each fold expands the training window forward in time and uses the
    next chunk for validation (expanding window approach).

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data sorted by ``date_column``.
    date_column : str
        Column with sortable date/timestamp values.
    n_splits : int, default 5
        Number of CV folds.

    Returns
    -------
    list of (train_idx, val_idx) tuples
        Each tuple contains numpy arrays of integer indices into
        ``train_df``.
    """
    if not isinstance(train_df, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(train_df).__name__}")
    if date_column not in train_df.columns:
        raise ValueError(f"Column '{date_column}' not found in DataFrame.")

    df_sorted = train_df.sort_values(date_column).reset_index(drop=True)
    n = len(df_sorted)
    folds: List[Tuple[np.ndarray, np.ndarray]] = []

    # Reserve the first chunk as minimum training size
    min_train = max(n // (n_splits + 1), 1)
    remaining = n - min_train
    fold_size = remaining // n_splits

    for i in range(n_splits):
        train_end = min_train + i * fold_size
        val_end = train_end + fold_size

        if i == n_splits - 1:
            val_end = n  # last fold uses all remaining data

        if val_end > n:
            break

        train_idx = np.arange(0, train_end)
        val_idx = np.arange(train_end, val_end)

        folds.append((train_idx, val_idx))

    return folds
