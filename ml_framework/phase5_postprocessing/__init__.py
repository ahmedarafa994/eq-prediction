"""Phase 5: Post-Processing & Diagnostics."""

from .calibration import (
    PlattScaler, IsotonicCalibrator, BetaCalibrator,
    find_temperature, temperature_scale,
    expected_calibration_error, reliability_data, calibration_evaluation,
)
from .conformal import (
    split_conformal_classification, split_conformal_regression,
    conformalized_quantile_regression, cross_conformal_classification, aci_conformal,
)
from .threshold import (
    find_optimal_threshold_f1, cost_optimal_threshold,
    profit_maximizing_threshold, youdens_j_threshold,
    threshold_for_constraint, ensemble_threshold,
)
from .fairness import (
    equalized_odds_postprocessing, reject_option_classification,
    calibrated_equalized_odds, fairness_audit, fairness_accuracy_tradeoff,
)
from .diagnostics import (
    diagnose_model, get_learning_curve_data,
    get_validation_curve_data, compare_models,
)
