"""
Phase 1: Data Quality & Preprocessing

Modules:
    profiling          - Comprehensive DataFrame profiling
    missing_values     - Pattern-based imputation
    outliers           - Ensemble outlier detection and winsorization
    validation         - DataFrame validation against pandera schemas
    class_imbalance    - Class imbalance assessment and cost-sensitive classification
    leakage_prevention - Data leakage auditing and temporal splitting
    encoding           - Smart categorical encoding with auto strategy selection
"""

from .profiling import comprehensive_profile
from .missing_values import PatternBasedImputer
from .outliers import EnsembleOutlierDetector
from .validation import DataValidator
from .class_imbalance import assess_imbalance, CostSensitiveClassifier
from .leakage_prevention import (
    leakage_audit,
    temporal_train_test_split,
    create_temporal_cv,
)
from .encoding import SmartCategoricalEncoder, RegularizedTargetEncoder

__all__ = [
    "comprehensive_profile",
    "PatternBasedImputer",
    "EnsembleOutlierDetector",
    "DataValidator",
    "assess_imbalance",
    "CostSensitiveClassifier",
    "leakage_audit",
    "temporal_train_test_split",
    "create_temporal_cv",
    "SmartCategoricalEncoder",
    "RegularizedTargetEncoder",
]
