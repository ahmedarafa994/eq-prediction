"""Phase 4: Validation, Ensembles & Regularization."""

from .validation import (
    kfold_cv, stratified_kfold_cv, group_kfold_cv,
    time_series_cv, nested_cv, repeated_kfold_cv, repeated_stratified_kfold_cv,
)
from .metrics import (
    classification_metrics, classification_report_dict, confusion_matrix_metrics,
    regression_metrics, max_error_normalized,
    business_cost_metric, make_business_scorer,
    expected_calibration_error, brier_score,
    paired_ttest, bootstrap_ci, multiple_comparison_correction,
)
from .ensembles import (
    stacking_classifier, manual_stacking, blend_models, bagged_predictions,
    xgboost_classifier, lightgbm_classifier, catboost_classifier,
    ensemble_selection,
    disagreement_measure, double_fault_measure, q_statistics, entropy_diversity,
)
from .regularization import (
    l1_feature_selection, ridge_regression_model, elastic_net_model,
    LabelSmoothingCrossEntropy, mixup_data, mixup_criterion, cutmix_data,
    add_gaussian_noise, tabular_mixup, adamw_optimizer,
    diagnose_bias_variance, learning_curve_data, validation_curve_data,
)
