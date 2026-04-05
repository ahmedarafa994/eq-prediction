"""
DataFrame validation against pandera schemas.

Provides an sklearn-compatible transformer that validates DataFrames
before passing them through the pipeline.
"""

from typing import Optional

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

try:
    import pandera as pa
    from pandera import Check, Column, DataFrameSchema
    HAS_PANDERA = True
except ImportError:
    HAS_PANDERA = False


class DataValidator(BaseEstimator, TransformerMixin):
    """
    Validate DataFrames against a pandera schema, then pass them through.

    This sits at the start of an sklearn pipeline to catch data quality
    issues before they propagate downstream.

    Parameters
    ----------
    schema : pandera.DataFrameSchema or dict
        A pandera schema or a dict that can be converted into one.
        If a dict is provided, it should follow pandera's schema dict format.
    strict : bool, default False
        If True, raise an error on validation failure.
        If False, issue a warning and pass the data through anyway.

    Attributes
    ----------
    schema_ : DataFrameSchema
        The resolved pandera schema used for validation.

    Examples
    --------
    >>> import pandera as pa
    >>> schema = pa.DataFrameSchema({
    ...     "age": pa.Column(int, pa.Check.ge(0)),
    ...     "income": pa.Column(float, nullable=True),
    ... })
    >>> validator = DataValidator(schema=schema, strict=False)
    >>> validated = validator.fit_transform(df)
    """

    def __init__(self, schema, strict: bool = False):
        self.schema = schema
        self.strict = strict

    def _resolve_schema(self):
        """Convert the schema parameter to a DataFrameSchema if needed."""
        if HAS_PANDERA:
            if isinstance(self.schema, pa.DataFrameSchema):
                return self.schema
            if isinstance(self.schema, dict):
                return pa.DataFrameSchema(self.schema)
        else:
            # Without pandera, only allow dict-based minimal validation
            if isinstance(self.schema, dict):
                return self.schema
        return self.schema

    def fit(self, X: pd.DataFrame, y=None) -> "DataValidator":
        """
        Store the schema and feature names; no fitting needed.

        Parameters
        ----------
        X : pd.DataFrame
            Training data (used to record column names).
        y : ignored

        Returns
        -------
        self
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(X).__name__}")

        self.schema_ = self._resolve_schema()
        self._feature_names_out = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Validate X against the schema and return it.

        Parameters
        ----------
        X : pd.DataFrame
            Data to validate.

        Returns
        -------
        pd.DataFrame
            The input DataFrame (unchanged if validation passes).

        Raises
        ------
        ValueError
            If ``strict=True`` and validation fails.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(X).__name__}")

        self.schema_ = self._resolve_schema()

        if HAS_PANDERA and isinstance(self.schema_, pa.DataFrameSchema):
            try:
                self.schema_.validate(X, lazy=True)
            except pa.errors.SchemaErrors as err:
                failure_messages = []
                for failure in err.failure_cases.itertuples():
                    failure_messages.append(
                        f"Column '{failure.column}': check '{failure.check}' "
                        f"failed for value {failure.failure_case} "
                        f"at index {failure.index}"
                    )
                msg = (
                    f"Data validation failed with {len(failure_messages)} issue(s):\n"
                    + "\n".join(failure_messages[:20])  # cap at 20 messages
                )
                if len(failure_messages) > 20:
                    msg += f"\n... and {len(failure_messages) - 20} more issues."

                if self.strict:
                    raise ValueError(msg)
                else:
                    import warnings
                    warnings.warn(msg, UserWarning, stacklevel=2)

        elif isinstance(self.schema_, dict):
            # Minimal validation without pandera
            self._validate_dict_schema(X)
        else:
            import warnings
            warnings.warn(
                "No valid schema provided and pandera is not installed. "
                "Skipping validation.",
                UserWarning,
                stacklevel=2,
            )

        return X

    def _validate_dict_schema(self, X: pd.DataFrame) -> None:
        """Perform basic validation using a plain dict schema.

        The dict schema format is:
        ``{column_name: {"dtype": dtype, "nullable": bool, "min": number, "max": number, "values": list}}``
        """
        import warnings

        issues = []
        for col, rules in self.schema_.items():
            if col not in X.columns:
                issues.append(f"Missing column: '{col}'")
                continue

            series = X[col]

            if "dtype" in rules:
                expected = rules["dtype"]
                if not pd.api.types.is_dtype_equal(series.dtype, expected):
                    # Try string comparison as fallback
                    if str(series.dtype) != str(expected):
                        issues.append(
                            f"Column '{col}': expected dtype {expected}, "
                            f"got {series.dtype}"
                        )

            if rules.get("nullable") is False:
                null_count = series.isnull().sum()
                if null_count > 0:
                    issues.append(
                        f"Column '{col}': {null_count} null values "
                        f"(nullable=False)"
                    )

            if "min" in rules:
                if pd.api.types.is_numeric_dtype(series):
                    below_min = (series.dropna() < rules["min"]).sum()
                    if below_min > 0:
                        issues.append(
                            f"Column '{col}': {below_min} values below "
                            f"minimum {rules['min']}"
                        )

            if "max" in rules:
                if pd.api.types.is_numeric_dtype(series):
                    above_max = (series.dropna() > rules["max"]).sum()
                    if above_max > 0:
                        issues.append(
                            f"Column '{col}': {above_max} values above "
                            f"maximum {rules['max']}"
                        )

            if "values" in rules:
                allowed = set(rules["values"])
                actual = set(series.dropna().unique())
                invalid = actual - allowed
                if invalid:
                    issues.append(
                        f"Column '{col}': {len(invalid)} values not in "
                        f"allowed set: {invalid}"
                    )

        if issues:
            msg = "Data validation issues:\n" + "\n".join(issues)
            if self.strict:
                raise ValueError(msg)
            else:
                warnings.warn(msg, UserWarning, stacklevel=2)

    def get_feature_names_out(self, input_features=None):
        """Return feature names (passthrough, columns unchanged)."""
        if input_features is not None:
            return list(input_features)
        return self._feature_names_out
