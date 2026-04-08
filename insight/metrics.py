"""Shared matched/raw evaluation metrics for the multi-type EQ estimator."""

import torch

try:
    from .differentiable_eq import FILTER_NAMES
    from .loss_multitype import HungarianBandMatcher
except ImportError:
    from differentiable_eq import FILTER_NAMES
    from loss_multitype import HungarianBandMatcher


def compute_eq_metrics(
    gain_pred,
    freq_pred,
    q_pred,
    type_pred,
    type_logits,
    gain_gt,
    freq_gt,
    q_gt,
    type_gt,
    lambda_type_match=0.0,
):
    """Compute Hungarian-matched and raw metrics from concatenated tensors."""
    matcher = HungarianBandMatcher(
        lambda_gain=1.0,
        lambda_freq=1.0,
        lambda_q=1.0,
        lambda_type_match=lambda_type_match,
    )
    matched_gain, matched_freq, matched_q, matched_type = matcher(
        gain_pred,
        freq_pred,
        q_pred,
        gain_gt,
        freq_gt,
        q_gt,
        target_filter_type=type_gt,
        pred_type_logits=type_logits,
    )

    metrics = {
        "gain_mae_db_matched": (gain_pred - matched_gain).abs().mean().item(),
        "gain_mae_db_raw": (gain_pred - gain_gt).abs().mean().item(),
        "gain_median_ae_db_matched": (gain_pred - matched_gain).abs().median().item(),
        "gain_median_ae_db_raw": (gain_pred - gain_gt).abs().median().item(),
    }

    freq_oct_matched = (torch.log2(freq_pred / (matched_freq + 1e-8))).abs()
    freq_oct_raw = (torch.log2(freq_pred / (freq_gt + 1e-8))).abs()
    metrics["freq_mae_oct_matched"] = freq_oct_matched.mean().item()
    metrics["freq_mae_oct_raw"] = freq_oct_raw.mean().item()
    metrics["freq_median_ae_oct_matched"] = freq_oct_matched.median().item()
    metrics["freq_median_ae_oct_raw"] = freq_oct_raw.median().item()

    q_dec_matched = (torch.log10(q_pred / (matched_q + 1e-8))).abs()
    q_dec_raw = (torch.log10(q_pred / (q_gt + 1e-8))).abs()
    metrics["q_mae_dec_matched"] = q_dec_matched.mean().item()
    metrics["q_mae_dec_raw"] = q_dec_raw.mean().item()
    metrics["q_median_ae_dec_matched"] = q_dec_matched.median().item()
    metrics["q_median_ae_dec_raw"] = q_dec_raw.median().item()

    metrics["type_accuracy_matched"] = (
        (type_pred == matched_type).float().mean().item()
    )
    metrics["type_accuracy_raw"] = (type_pred == type_gt).float().mean().item()

    zero_gain_mask = (
        (matched_type == 3)
        | (matched_type == 4)
        | (matched_gain.abs() < 0.5)
    )
    if zero_gain_mask.any():
        zero_gain_abs = gain_pred[zero_gain_mask].abs()
        metrics["zero_gain_abs_mean_db_matched"] = zero_gain_abs.mean().item()
        metrics["zero_gain_abs_p95_db_matched"] = torch.quantile(
            zero_gain_abs, 0.95
        ).item()
        metrics["zero_gain_count_matched"] = int(zero_gain_mask.sum().item())
    else:
        metrics["zero_gain_abs_mean_db_matched"] = 0.0
        metrics["zero_gain_abs_p95_db_matched"] = 0.0
        metrics["zero_gain_count_matched"] = 0

    for idx, name in enumerate(FILTER_NAMES):
        matched_mask = matched_type == idx
        raw_mask = type_gt == idx
        if matched_mask.any():
            metrics[f"type_accuracy_{name}_matched"] = (
                type_pred[matched_mask] == matched_type[matched_mask]
            ).float().mean().item()
        if raw_mask.any():
            metrics[f"type_accuracy_{name}_raw"] = (
                type_pred[raw_mask] == type_gt[raw_mask]
            ).float().mean().item()

    num_types = len(FILTER_NAMES)
    confusion_matched = torch.zeros(num_types, num_types)
    confusion_raw = torch.zeros(num_types, num_types)
    for pred_value, matched_value, raw_value in zip(
        type_pred.flatten(),
        matched_type.flatten(),
        type_gt.flatten(),
    ):
        confusion_matched[matched_value.item(), pred_value.item()] += 1
        confusion_raw[raw_value.item(), pred_value.item()] += 1
    metrics["confusion_matrix_matched"] = confusion_matched.tolist()
    metrics["confusion_matrix_raw"] = confusion_raw.tolist()
    return metrics
