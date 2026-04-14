"""
Multi-type EQ estimation loss functions.

Key components:
- Hungarian matching for permutation-invariant band assignment (DETR-style)
- Combined regression + classification loss
- Spectral consistency loss for blind estimation
- Auxiliary losses (band activity, frequency spread)
- Log-cosh gain loss (Phase 3)
- Curriculum warmup with hybrid gate (Phase 3)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import numpy as np

from loss import MultiResolutionSTFTLoss
from differentiable_eq import FILTER_HIGHPASS, FILTER_LOWPASS, FILTER_LOWSHELF, FILTER_HIGHSHELF
from differentiable_eq import FILTER_PEAKING, BROAD_PASS, BROAD_SHELF, BROAD_PEAKING, FINE_TO_BROAD


def log_cosh_loss(pred, target):
    """
    Log-cosh loss: log(cosh(pred - target)).
    - Near zero: ~0.5 * x^2 (smooth quadratic, better gradients than Huber)
    - Far from zero: ~|x| - log(2) (linear, robust to outliers)
    - C2 continuous everywhere (unlike Huber's C1 at delta)

    Numerically stable formulation:
        log(cosh(x)) = |x| + log(1 + exp(-2|x|)) - log(2)

    H-12: Compute in float32 for numerical stability, cast back at the end.
    AUDIT: HIGH-02 (V-11) — For small |x|, use Taylor expansion directly:
        log(cosh(x)) ≈ x^2/2 for |x| < 1e-6
        This avoids log1p(exp(-2|x|)) ≈ log(1 + very_small) precision loss.
        Ensures C2 continuity at transition point (gradient and second derivative match).
    """
    diff = pred - target
    # Compute in float32 for numerical stability
    abs_diff = diff.float().abs()

    # AUDIT: V-11 — Use 1e-6 threshold for near-zero stability
    # At |x| = 1e-6:
    #   Taylor: x^2/2 = 0.5e-12
    #   Full: |x| + log(1+exp(-2|x|)) - log(2) ≈ 0.5000001e-12
    # The difference is < 0.01% so C2 continuity is preserved
    small_mask = abs_diff < 1e-6

    # Taylor expansion for small values: x^2/2
    # This is C2 continuous with the full formula at the threshold
    result_small = 0.5 * abs_diff * abs_diff

    # Full numerically-stable formula for larger values
    # log(1 + exp(-2|x|)) is stable because argument is in (-inf, 0]
    # Apply mask to preserve shape when computing large values
    abs_diff_masked = torch.where(small_mask, torch.ones_like(abs_diff), abs_diff)
    result_large = abs_diff_masked + torch.log1p(torch.exp(-2.0 * abs_diff_masked)) - math.log(2)

    # Combine results using torch.where for C2 continuity
    result = torch.where(small_mask, result_small, result_large)

    return result.to(diff.dtype)


class UncertaintyWeightedLoss(nn.Module):
    """
    Learned multi-task loss weighting (Kendall et al. 2018).

    Each loss term gets a learnable log_sigma parameter. The effective weight
    is exp(-2*log_sigma)/2 and a regularization term +log_sigma prevents all
    weights from collapsing to zero.
    """

    def __init__(self, n_losses=7, initial_log_sigmas=None):
        super().__init__()
        if initial_log_sigmas is not None:
            self.log_sigma = nn.Parameter(
                torch.tensor(initial_log_sigmas, dtype=torch.float32)
            )
        else:
            self.log_sigma = nn.Parameter(torch.zeros(n_losses))

    def forward(self, losses):
        """
        Args:
            losses: list of scalar loss tensors
        Returns:
            weighted total loss (scalar)
        """
        total = torch.zeros((), device=self.log_sigma.device, dtype=self.log_sigma.dtype)
        for i, loss in enumerate(losses):
            precision = torch.exp(-2 * self.log_sigma[i])
            total = total + precision * loss + self.log_sigma[i]
        return total


def multi_scale_spectral_loss(pred_gain, pred_freq, pred_q, type_probs,
                               target_H_mag, dsp_cascade,
                               fft_sizes=(256, 512, 1024)):
    """
    Soft-render multi-scale spectral loss.

    Uses type probabilities to ensure gradients flow to the type classifier
    via the frequency response.
    """
    total = torch.tensor(0.0, device=pred_gain.device)
    for n_fft in fft_sizes:
        H_mag_pred = dsp_cascade.forward_soft(pred_gain, pred_freq, pred_q, type_probs=type_probs, n_fft=n_fft)
        target_resampled = F.interpolate(target_H_mag.unsqueeze(1), size=n_fft // 2 + 1, mode='linear', align_corners=False).squeeze(1)
        # Compute in fp32 for numerical stability (H-12 fix)
        pred_db = 20 * torch.log10(H_mag_pred.float().clamp(min=1e-6))
        tgt_db = 20 * torch.log10(target_resampled.float().clamp(min=1e-6))
        total = total + F.l1_loss(pred_db, tgt_db)
    return total / len(fft_sizes)


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al. 2020).

    Pulls embeddings of the same filter type together while pushing
    different filter types apart. Applied to the type_input features
    (the representation fed into the linear type classifier) so that
    the classification layer receives well-separated clusters.

    Args:
        temperature: scaling for similarity. Lower = tighter clusters.
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Args:
            features: (B*N, D) — type_input embeddings (before linear type_head)
            labels: (B*N,) — filter type indices (after Hungarian matching)
        Returns:
            scalar loss
        """
        device = features.device
        B_N = features.shape[0]
        if B_N < 2:
            return torch.tensor(0.0, device=device)

        # Normalize features to unit sphere
        feats = F.normalize(features, dim=1)

        # Similarity matrix: (B*N, B*N)
        sim = torch.matmul(feats, feats.T) / self.temperature

        # Mask: positives are same-type pairs (excluding self)
        labels = labels.unsqueeze(0)  # (1, B*N)
        mask_pos = (labels == labels.T).float()  # (B*N, B*N)
        mask_pos.fill_diagonal_(0.0)  # exclude self-pairs

        # For numerical stability: subtract max per row
        logits_max, _ = sim.max(dim=1, keepdim=True)
        logits = sim - logits_max.detach()

        # Log-sum-exp of all negatives + positives (denominator)
        # Mask out self-pair from denominator too
        mask_all = ~torch.eye(B_N, dtype=torch.bool, device=device)
        exp_logits = torch.exp(logits) * mask_all.float()
        log_sum_exp = torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        # Mean log-likelihood over positive pairs
        log_prob = logits - log_sum_exp

        # Only compute for anchors that have at least one positive
        num_positives = mask_pos.sum(dim=1)
        has_positives = num_positives > 0

        if not has_positives.any():
            return torch.tensor(0.0, device=device)

        mean_log_prob = (mask_pos * log_prob).sum(dim=1) / (num_positives + 1e-8)
        loss = -mean_log_prob[has_positives].mean()
        return loss


def compute_physics_soft_labels(
    gain, freq, q, target_H_mag, dsp_cascade, n_fft, temperature=2.0
):
    """Compute physics-informed soft type labels from DSP analysis.

    For each band, renders the frequency response with all 5 filter types
    and measures how well each type fits the target curve.  Types that
    produce similar curves (e.g. lowshelf vs peaking at 20 Hz) get high
    probability, reflecting the physical ambiguity.

    Args:
        gain: (B, N) target gains
        freq: (B, N) target frequencies
        q: (B, N) target Q values
        target_H_mag: (B, F) target frequency response magnitude
        dsp_cascade: DifferentiableBiquadCascade instance
        n_fft: FFT size
        temperature: softmax temperature (higher = softer labels)
    Returns:
        soft_labels: (B, N, 5) probability distribution over filter types
    """
    B, N = gain.shape
    device = gain.device
    n_fft_bins = n_fft // 2 + 1
    n_types = 5

    # Resample target to match n_fft if needed
    target_len = target_H_mag.shape[-1]

    mse_per_type = []
    with torch.no_grad():
        for t in range(n_types):
            type_idx = torch.full((B, N), t, dtype=torch.long, device=device)
            H_mag_t = dsp_cascade(gain, freq, q, filter_type=type_idx, n_fft=n_fft)
            # Resample target to match predicted length if different
            if H_mag_t.shape[-1] != target_len:
                tgt = F.interpolate(
                    target_H_mag.unsqueeze(1),
                    size=H_mag_t.shape[-1],
                    mode='linear',
                    align_corners=False,
                ).squeeze(1)
            else:
                tgt = target_H_mag
            # MSE in log-magnitude space (dB-like)
            pred_db = 20 * torch.log10(H_mag_t.float().clamp(min=1e-6))
            tgt_db = 20 * torch.log10(tgt.float().clamp(min=1e-6))
            mse = (pred_db - tgt_db.unsqueeze(1)).pow(2).mean(dim=-1)  # (B, N)
            mse_per_type.append(mse)

    # Stack: (B, N, 5)
    mse_stack = torch.stack(mse_per_type, dim=-1)
    # Softmax over types: lower MSE → higher probability
    soft_labels = F.softmax(-mse_stack / temperature, dim=-1)
    return soft_labels


class HungarianBandMatcher:
    """
    Solves the band permutation problem using Hungarian (bipartite) matching.

    When predicting N EQ bands, the model's band ordering is arbitrary.
    This finds the optimal assignment between predicted and ground-truth bands
    to minimize parameter distance, following Carion et al. (DETR, 2020).
    """

    def __init__(
        self,
        lambda_freq=1.0,
        lambda_q=0.5,
        lambda_gain=2.0,
        lambda_type_match=0.5,
    ):
        self.lambda_freq = lambda_freq
        self.lambda_q = lambda_q
        self.lambda_gain = lambda_gain
        self.lambda_type_match = lambda_type_match

    def compute_cost_matrix(
        self,
        pred_gain,
        pred_freq,
        pred_q,
        target_gain,
        target_freq,
        target_q,
        pred_type_logits=None,
        target_filter_type=None,
    ):
        """
        Compute pairwise cost matrix between predicted and target bands.

        All inputs: (Batch, Num_Bands)
        pred_type_logits: optional (B, N, 5) predicted type logits
        target_filter_type: optional (B, N) ground truth type indices
        Returns: cost matrix (Batch, Num_Pred, Num_Target)
        """
        B, N = pred_gain.shape

        # Clamp inputs to prevent NaN/inf from propagating into the cost matrix
        # (early training instability can produce non-finite predictions)
        pg = torch.clamp(pred_gain.unsqueeze(-1), min=-100.0, max=100.0)  # (B, N, 1)
        tg = target_gain.unsqueeze(-2)  # (B, 1, N)
        pf = torch.clamp(pred_freq.unsqueeze(-1), min=1.0, max=40000.0)
        tf = torch.clamp(target_freq.unsqueeze(-2), min=1.0, max=40000.0)
        pq = torch.clamp(pred_q.unsqueeze(-1), min=1e-4, max=100.0)
        tq = torch.clamp(target_q.unsqueeze(-2), min=1e-4, max=100.0)

        # Gain cost: L1 in dB (weighted to balance against octave-scale freq cost)
        cost_gain = self.lambda_gain * (pg - tg).abs()

        # Frequency cost: L1 in log-space (octaves)
        cost_freq = (
            self.lambda_freq * (torch.log(pf + 1e-8) - torch.log(tf + 1e-8)).abs()
        )

        # Q cost: L1 in log-space (decades)
        cost_q = self.lambda_q * (torch.log(pq + 1e-8) - torch.log(tq + 1e-8)).abs()

        cost = cost_gain + cost_freq + cost_q  # (B, N, N)

        # Filter type cost: if available, add a penalty for type mismatch
        if pred_type_logits is not None and target_filter_type is not None:
            pred_probs = pred_type_logits.softmax(-1)  # (B, N, 5)
            target_one_hot = F.one_hot(target_filter_type, 5).float()  # (B, N, 5)
            # Pairwise: (B, N_pred, 1, 5) vs (B, 1, N_target, 5)
            p_correct_type = (
                pred_probs.unsqueeze(2) * target_one_hot.unsqueeze(1)
            ).sum(-1)  # (B, N, N)
            # Increase type mismatch penalty from 0.5 to 2.0 to force type-aware matching
            # This makes matching "stubborn" about type alignment.
            type_cost = 1.0 - p_correct_type
            cost = cost + 2.0 * type_cost

        return cost

    def match(self, cost_matrix, pred_type_logits=None, target_filter_type=None):
        """
        Solve the assignment problem for each batch element.

        Args:
            cost_matrix: (Batch, N_pred, N_target)
            pred_type_logits: optional (B, N, 5) predicted type logits (for cost matrix computation)
            target_filter_type: optional (B, N) ground truth type indices (for cost matrix computation)

        Returns:
            List of (row_indices, col_indices) tuples per batch element
        """
        B = cost_matrix.shape[0]
        assignments = []
        for b in range(B):
            cost_np = cost_matrix[b].detach().cpu().numpy()
            # Triple-guard against NaN/inf while preserving type-aware structure.
            cost_np = np.nan_to_num(cost_np, nan=1e6, posinf=1e6, neginf=0.0)
            cost_np = np.minimum(cost_np, 1e6)
            if not np.isfinite(cost_np).all():
                # Fallback: identity assignment with zero cost
                row_ind = np.arange(cost_np.shape[0])
                col_ind = np.arange(cost_np.shape[1])
                assignments.append((row_ind, col_ind))
                continue
            row_ind, col_ind = linear_sum_assignment(cost_np)
            assignments.append((row_ind, col_ind))
        return assignments

    def __call__(
        self,
        pred_gain,
        pred_freq,
        pred_q,
        target_gain,
        target_freq,
        target_q,
        target_filter_type=None,
        pred_type_logits=None,
    ):
        """
        Find optimal matching and reorder targets.

        Args:
            target_filter_type: optional (B, N) long tensor of ground-truth filter types.
                When provided, returns a 4th matched tensor using the same permutation.
            pred_type_logits: optional (B, N, 5) predicted type logits for type-aware matching.

        Returns:
            3-tuple (matched_gain, matched_freq, matched_q) when target_filter_type is None,
            4-tuple (matched_gain, matched_freq, matched_q, matched_filter_type) otherwise.
        """
        cost = self.compute_cost_matrix(
            pred_gain,
            pred_freq,
            pred_q,
            target_gain,
            target_freq,
            target_q,
            pred_type_logits=pred_type_logits,
            target_filter_type=target_filter_type,
        )
        assignments = self.match(
            cost,
            pred_type_logits=pred_type_logits,
            target_filter_type=target_filter_type,
        )

        B, N = pred_gain.shape
        device = pred_gain.device

        matched_gain = torch.zeros_like(target_gain)
        matched_freq = torch.zeros_like(target_freq)
        matched_q = torch.zeros_like(target_q)
        if target_filter_type is not None:
            matched_filter_type = torch.zeros_like(target_filter_type)

        for b in range(B):
            row_ind, col_ind = assignments[b]
            perm = torch.zeros(N, dtype=torch.long, device=device)
            for r, c in zip(row_ind, col_ind):
                perm[r] = c
            matched_gain[b] = target_gain[b, perm]
            matched_freq[b] = target_freq[b, perm]
            matched_q[b] = target_q[b, perm]
            if target_filter_type is not None:
                matched_filter_type[b] = target_filter_type[b, perm]

        if target_filter_type is not None:
            return matched_gain, matched_freq, matched_q, matched_filter_type
        return matched_gain, matched_freq, matched_q


class PermutationInvariantParamLoss(nn.Module):
    """
    Parameter regression loss with Hungarian matching for band ordering.
    Uses Huber loss for robustness to outliers.
    """

    def __init__(self, lambda_gain=2.0, lambda_freq=1.0, lambda_q=0.5, lambda_type_match=0.5):
        super().__init__()
        self.matcher = HungarianBandMatcher(
            lambda_freq, lambda_q, lambda_gain=lambda_gain, lambda_type_match=lambda_type_match
        )
        self.huber = nn.HuberLoss(delta=5.0)
        self.lambda_gain = lambda_gain
        self.lambda_freq = lambda_freq
        self.lambda_q = lambda_q

    def forward(
        self,
        pred_gain,
        pred_freq,
        pred_q,
        target_gain,
        target_freq,
        target_q,
        pred_type_logits=None,
        target_filter_type=None,
    ):
        """
        All inputs: (Batch, Num_Bands)
        pred_type_logits: optional (B, N, 5) predicted type logits
        target_filter_type: optional (B, N) ground truth type indices
        Returns: scalar loss
        """
        matched_gain, matched_freq, matched_q = self.matcher(
            pred_gain,
            pred_freq,
            pred_q,
            target_gain,
            target_freq,
            target_q,
            pred_type_logits=pred_type_logits,
            target_filter_type=target_filter_type,
        )

        loss_gain = self.huber(pred_gain, matched_gain)
        loss_freq = self.huber(
            torch.log(pred_freq + 1e-8), torch.log(matched_freq + 1e-8)
        )
        loss_q = self.huber(torch.log(pred_q + 1e-8), torch.log(matched_q + 1e-8))

        return loss_gain, loss_freq, loss_q


class MultiTypeEQLoss(nn.Module):
    """
    Combined loss for multi-type blind EQ estimation.

    Components:
    - L_param: Permutation-invariant parameter regression (Huber)
    - L_type: Filter type classification (focal / balanced-softmax / soft-label KL)
    - L_spectral: Spectral magnitude L1 (LOSS-05)
    - L_hmag: Frequency response magnitude L1 (hard types)
    - L_activity: Band activity regularization
    - L_spread: Frequency spread regularization
    - L_embed_var: Embedding variance regularization (anti-collapse)
    - L_contrastive: Spectral contrastive diagnostic (anti-collapse)
    - L_supcon: Supervised contrastive loss on per-band embeddings

    Anti-collapse mechanism:
    The TCN encoder can collapse to producing near-identical embeddings for
    all inputs (cosine distance ~0.006). This makes the parameter head's job
    impossible — it receives no discriminative information to decode.
    L_embed_var directly penalizes low variance across batch dimensions,
    forcing the encoder to maintain diverse representations. L_contrastive
    goes further by encouraging embeddings from different mel profiles to
    occupy distinct regions of the embedding space.
    """

    def __init__(
        self,
        n_fft=2048,
        sample_rate=44100,
        lambda_param=1.0,
        lambda_gain=2.0,
        lambda_freq=1.0,
        lambda_q=0.5,
        lambda_type=0.5,
        lambda_spectral=1.0,
        lambda_slope=0.0,
        lambda_hmag=0.3,
        lambda_activity=0.1,
        lambda_spread=0.05,
        lambda_embed_var=0.5,
        lambda_contrastive=0.1,
        lambda_type_match=0.5,
        embed_var_threshold=0.1,
        warmup_epochs=5,
        class_weight_multipliers=None,
        type_class_priors=None,
        type_loss_mode="focal",
        soft_label_temperature=0.35,
        label_smoothing=0.05,
        focal_gamma=2.0,
        lambda_supcon=0.0,
        supcon_temperature=0.07,
        supcon_max_samples=256,
        lambda_type_entropy=0.0,
        lambda_type_prior=0.0,
        sign_penalty_weight=0.0,
        lambda_hdb=2.0,
        lambda_gain_zero=0.25,
        lambda_typed_spectral=2.0,
        lambda_film_diversity=0.0,
        dsp_cascade=None,
        **kwargs,  # absorb extra params from train.py
    ):
        super().__init__()
        self.lambda_param = lambda_param
        self.lambda_gain = lambda_gain
        self.lambda_freq = lambda_freq
        self.lambda_q = lambda_q
        self.lambda_type = lambda_type
        self.lambda_spectral = lambda_spectral
        self.lambda_slope = lambda_slope
        self.lambda_hmag = lambda_hmag
        self.lambda_hdb = lambda_hdb
        self.lambda_gain_zero = lambda_gain_zero
        self.lambda_typed_spectral = lambda_typed_spectral
        self.lambda_film_diversity = lambda_film_diversity
        self.lambda_activity = lambda_activity
        self.lambda_spread = lambda_spread
        self.lambda_embed_var = lambda_embed_var
        self.lambda_contrastive = lambda_contrastive
        self.lambda_type_match = lambda_type_match
        self.lambda_type_entropy = lambda_type_entropy
        self.lambda_type_prior = lambda_type_prior
        self.lambda_multi_scale = kwargs.get("lambda_multi_scale", 1.0)
        self.matcher_lambda_gain = kwargs.get("matcher_lambda_gain", lambda_gain)
        self.matcher_lambda_freq = kwargs.get("matcher_lambda_freq", lambda_freq)
        self.matcher_lambda_q = kwargs.get("matcher_lambda_q", lambda_q)
        self.matcher_lambda_type_match = kwargs.get("matcher_lambda_type_match", lambda_type_match)
        self.embed_var_threshold = embed_var_threshold
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        self.sign_penalty_weight = sign_penalty_weight
        self.type_loss_mode = str(type_loss_mode).lower()
        self.soft_label_temperature = float(soft_label_temperature)
        self.class_weight_multipliers = (
            list(class_weight_multipliers) if class_weight_multipliers is not None else None
        )
        self.lambda_supcon = float(lambda_supcon)
        self.supcon_temperature = float(supcon_temperature)
        self.supcon_max_samples = int(supcon_max_samples)

        # AUDIT: MEDIUM-03 — Single source of truth for warmup tracking.
        # The warmup system has ONE mechanism with TWO components (consolidated):
        # 1. Epoch counter (self.current_epoch) - incremented by train.py each epoch
        # 2. Gain MAE EMA (self.gain_mae_ema) - updated via update_gain_mae() each batch
        # The hybrid warmup gate (in forward()) uses BOTH to determine warmup state.
        # Warmup ends when BOTH conditions are met: epoch >= warmup_epochs AND gain_mae_ema <= 2.5 dB.
        # Hard cap at 15 epochs prevents infinite warmup if gain never converges.
        # This consolidates the previous dual-mechanism approach where H_db ramp
        # was handled separately in train.py (now all warmup is here).
        self.gain_mae_ema = 10.0  # EMA of per-batch gain MAE (dB), alpha=0.1 — start high (realistic)

        self.param_loss = PermutationInvariantParamLoss(
            lambda_gain=lambda_gain,
            lambda_freq=lambda_freq,
            lambda_q=lambda_q,
            lambda_type_match=lambda_type_match
        )
        self.huber = nn.HuberLoss(delta=5.0)
        # Class prior-aware type losses.
        # If priors are provided from config/data, use them; otherwise fallback to
        # a balanced default prior.
        if type_class_priors is None:
            type_priors = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2], dtype=torch.float32)
        else:
            type_priors = torch.tensor(type_class_priors, dtype=torch.float32)
        type_priors = torch.clamp(type_priors, min=1e-6)
        type_priors = type_priors / type_priors.sum()

        self.register_buffer("type_class_priors", type_priors)
        self.register_buffer("type_class_weights", torch.ones_like(type_priors))
        self._refresh_type_class_weights()
        self.focal_gamma = float(focal_gamma)
        self.type_label_smoothing = float(label_smoothing)
        self.mr_stft = MultiResolutionSTFTLoss()

        # Fix 2+3: Learned uncertainty weighting + multi-scale render loss
        # Uniform initial precisions — lambda values provide the static balance.
        self.uncertainty_loss = UncertaintyWeightedLoss(
            n_losses=7,
            initial_log_sigmas=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        self._dsp_cascade = dsp_cascade  # stored as plain attr, NOT nn.Module

    def _compute_physics_soft_labels(
        self,
        matched_gain,
        matched_freq,
        matched_q,
        matched_filter_type,
        num_fft_bins,
        num_types,
    ):
        """
        Build soft type targets from response similarity across all filter types.

        For each band, render a single-band response with the matched parameters
        and each candidate filter type, then convert per-type MSE scores into a
        probability distribution via temperature-scaled softmax.
        """
        if self._dsp_cascade is None:
            return F.one_hot(matched_filter_type, num_classes=num_types).float()

        n_fft = max(2, int((num_fft_bins - 1) * 2))
        temperature = max(self.soft_label_temperature, 1e-4)

        with torch.no_grad():
            tgt_b0, tgt_b1, tgt_b2, tgt_a1, tgt_a2 = (
                self._dsp_cascade.compute_biquad_coeffs_multitype(
                    matched_gain,
                    matched_freq,
                    matched_q,
                    matched_filter_type,
                )
            )
            target_band_mag = self._dsp_cascade.freq_response(
                tgt_b0,
                tgt_b1,
                tgt_b2,
                tgt_a1,
                tgt_a2,
                n_fft=n_fft,
            )
            target_band_log = torch.log(target_band_mag.float().clamp(min=1e-6))

            mse_per_type = []
            for type_idx in range(num_types):
                candidate_type = torch.full_like(matched_filter_type, type_idx)
                c_b0, c_b1, c_b2, c_a1, c_a2 = (
                    self._dsp_cascade.compute_biquad_coeffs_multitype(
                        matched_gain,
                        matched_freq,
                        matched_q,
                        candidate_type,
                    )
                )
                candidate_mag = self._dsp_cascade.freq_response(
                    c_b0,
                    c_b1,
                    c_b2,
                    c_a1,
                    c_a2,
                    n_fft=n_fft,
                )
                candidate_log = torch.log(candidate_mag.float().clamp(min=1e-6))
                # Per-band spectral mismatch (B, N)
                mse = F.mse_loss(candidate_log, target_band_log, reduction="none").mean(dim=-1)
                mse_per_type.append(mse)

            mse_stack = torch.stack(mse_per_type, dim=-1)  # (B, N, C)
            soft_targets = torch.softmax(-mse_stack / temperature, dim=-1)

        return soft_targets

    def _supervised_contrastive_loss(self, band_embedding, labels, sample_weights=None):
        """
        Supervised contrastive loss over per-band embeddings.

        Uses matched filter-type labels to pull same-type embeddings together
        and push different-type embeddings apart.
        """
        if band_embedding is None:
            return None

        if band_embedding.dim() != 3:
            return None

        z = band_embedding.reshape(-1, band_embedding.shape[-1])
        y = labels.reshape(-1)

        if sample_weights is None:
            w = torch.ones(z.shape[0], device=z.device, dtype=z.dtype)
        else:
            w = sample_weights.reshape(-1).to(dtype=z.dtype, device=z.device)

        valid_idx = torch.where(w > 1e-4)[0]
        if valid_idx.numel() < 3:
            return None

        if valid_idx.numel() > self.supcon_max_samples:
            perm = torch.randperm(valid_idx.numel(), device=valid_idx.device)
            valid_idx = valid_idx[perm[: self.supcon_max_samples]]

        z = F.normalize(z[valid_idx], dim=-1)
        y = y[valid_idx]
        w = w[valid_idx]

        m = z.shape[0]
        if m < 3:
            return None

        temperature = max(self.supcon_temperature, 1e-4)
        sim = torch.matmul(z, z.t()) / temperature
        logits = sim - sim.max(dim=1, keepdim=True).values.detach()

        eye_mask = torch.eye(m, device=z.device, dtype=torch.bool)
        non_self = ~eye_mask
        exp_logits = torch.exp(logits) * non_self
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        pos_mask = (y.unsqueeze(0) == y.unsqueeze(1)) & non_self
        pos_count = pos_mask.sum(dim=1)
        valid_anchor = pos_count > 0
        if not valid_anchor.any():
            return None

        mean_log_prob_pos = (
            (pos_mask.float() * log_prob).sum(dim=1) / pos_count.clamp(min=1)
        )

        anchor_weights = w / (w.mean() + 1e-8)
        loss = -(
            anchor_weights[valid_anchor] * mean_log_prob_pos[valid_anchor]
        ).sum() / (anchor_weights[valid_anchor].sum() + 1e-8)
        return loss

    def _refresh_type_class_weights(self):
        inv_w = 1.0 / self.type_class_priors.clamp(min=1e-6)
        inv_w = inv_w / inv_w.mean()
        if self.class_weight_multipliers is not None:
            mult_tensor = torch.tensor(
                self.class_weight_multipliers,
                dtype=inv_w.dtype,
                device=inv_w.device,
            )
            inv_w = inv_w * mult_tensor
            inv_w = inv_w / inv_w.mean()
        self.type_class_weights.copy_(inv_w)

    def update_type_priors(self, type_class_priors):
        if isinstance(type_class_priors, torch.Tensor):
            priors = type_class_priors.to(
                dtype=self.type_class_priors.dtype,
                device=self.type_class_priors.device,
            )
        else:
            priors = torch.tensor(
                type_class_priors,
                dtype=self.type_class_priors.dtype,
                device=self.type_class_priors.device,
            )
        priors = torch.clamp(priors, min=1e-6)
        priors = priors / priors.sum()
        self.type_class_priors.copy_(priors)
        self._refresh_type_class_weights()

    def update_gain_mae(self, batch_gain_mae: float, alpha: float = 0.1) -> None:
        """Update the exponential moving average of per-batch gain MAE.

        AUDIT: MEDIUM-03 — This is part of the consolidated warmup system.
        The warmup state is determined by BOTH current_epoch and gain_mae_ema.
        Called by train.py after each training batch. The updated EMA is used
        in the hybrid warmup gate to determine when freq/Q/type losses should activate.

        To check warmup state externally:
            is_warmup = (criterion.current_epoch < criterion.warmup_epochs) or \
                       (criterion.gain_mae_ema > 2.5)

        Args:
            batch_gain_mae: Mean absolute gain error for the current batch (dB).
            alpha: EMA smoothing factor (default 0.1 -- slow smoothing for stability).
        """
        self.gain_mae_ema = alpha * batch_gain_mae + (1 - alpha) * self.gain_mae_ema

    def forward(
        self,
        pred_gain,
        pred_freq,
        pred_q,
        pred_type_logits,
        pred_H_mag_soft,
        pred_H_mag_hard,
        target_gain,
        target_freq,
        target_q,
        target_filter_type,
        target_H_mag,
        pred_audio=None,
        target_audio=None,
        active_band_mask=None,
        embedding=None,
        band_embedding=None,
        h_db_pred=None,
        h_db_target=None,
        H_mag_typed=None,
        type_probs=None,
        hier_aux=None,
    ):
        """
        Compute total loss.

        Args:
            pred_gain, pred_freq, pred_q: (B, N) predicted parameters
            pred_type_logits: (B, N, 5) predicted type logits
            pred_H_mag_soft: (B, F) soft Gumbel-Softmax path — for spectral_loss
            pred_H_mag_hard: (B, F) argmax path — for hmag_loss (detached inside)
            target_gain, target_freq, target_q: (B, N) ground truth params
            target_filter_type: (B, N) ground truth type indices
            target_H_mag: (B, F) ground truth frequency response
            pred_audio: (B, T) optional reconstructed audio (deprecated, use H_mag_soft)
            target_audio: (B, T) optional target audio (deprecated, use target_H_mag)
            active_band_mask: (B, N) optional mask for active bands
            embedding: (B, D) optional encoder embedding for anti-collapse losses

        Returns:
            total_loss: scalar
            components: dict of individual loss values
        """
        components = {}

        # --- Curriculum warmup gating (D-02, D-03) ---
        # D-03/D-04: Hybrid warmup gate.
        # Warmup ENDS when BOTH conditions are true:
        #   (a) current_epoch >= warmup_epochs (minimum epoch threshold)
        #   (b) gain_mae_ema <= 2.5 dB (gain has converged sufficiently)
        # Hard cap: if epoch >= 15, warmup ends regardless of gain_mae_ema
        # (prevents infinite warmup if gain MAE never converges below 2.5 dB).
        past_epoch_threshold = self.current_epoch >= self.warmup_epochs
        gain_converged = self.gain_mae_ema <= 2.5
        past_hard_cap = self.current_epoch >= 15
        is_warmup = not ((past_epoch_threshold and gain_converged) or past_hard_cap)

        # FIX: Remove warmup gating for spectral and type losses.
        # The hybrid warmup kept spectral_loss at 0.0 for entire epoch 1 (see log),
        # starving the model of its primary supervision signal. With 50K samples
        # and proper regularization, all losses can be active from step 1.
        is_freq_q_active = True
        is_type_active = True
        is_spectral_active = True

        # --- Anti-collapse losses (computed first so they appear early in logs) ---

        # 7. Embedding variance regularization (L_embed_var)
        # FIX: Replace threshold-based loss with continuous penalty.
        # Prior code: F.relu(threshold - var) — zero gradient when var > threshold,
        # allowing slow drift toward collapse (log showed var: 0.93 → 0.59 with
        # no gradient to stop it). New formulation: loss = max(0, target - var)^2
        # provides gradient at ALL variance levels below the target, and a smooth
        # quadratic penalty above zero that never abruptly turns on.
        # LOGGED ONLY — not in core_losses, wrap in no_grad to skip gradient graph
        loss_embed_var = torch.tensor(0.0, device=pred_gain.device)
        if embedding is not None and embedding.shape[0] > 1:
            with torch.no_grad():
                embed_var = embedding.var(dim=0).mean()
                target_var = embedding.shape[-1] * 0.005
                deficit = torch.clamp(target_var - embed_var, min=0.0)
                loss_embed_var = deficit * deficit / (target_var + 1e-8)
                loss_embed_var = torch.clamp(loss_embed_var, max=5.0)
        components["embed_var_loss"] = loss_embed_var

        # 8. Spectral contrastive loss (L_contrastive)
        # Encourages embeddings from different mel profiles to be dissimilar.
        # This complements L_embed_var: variance regularization prevents a trivial
        # collapse to a single point, but the encoder could still produce a narrow
        # manifold. Contrastive loss explicitly pushes embeddings apart by penalizing
        # high pairwise cosine similarity. The formulation is simple: minimize the
        # mean pairwise cosine similarity, which ensures the encoder uses the full
        # embedding space rather than a thin subspace.
        # LOGGED ONLY — not in core_losses, wrap in no_grad to skip gradient graph
        loss_contrastive = torch.tensor(0.0, device=pred_gain.device)
        if embedding is not None and embedding.shape[0] > 1:
            with torch.no_grad():
                if torch.isfinite(embedding).all():
                    embed_norm = F.normalize(embedding, dim=1)
                    sim_matrix = torch.mm(embed_norm, embed_norm.t())
                    B = sim_matrix.shape[0]
                    mask = ~torch.eye(B, dtype=torch.bool, device=sim_matrix.device)
                    mean_sim = sim_matrix[mask].mean()
                    mean_sim_clamped = torch.clamp(mean_sim.float(), max=0.95)
                    loss_contrastive = -torch.log(1.0 - mean_sim_clamped + 1e-3)
                    loss_contrastive = torch.clamp(loss_contrastive, max=5.0)
        components["contrastive_loss"] = loss_contrastive

        # AUDIT: CRITICAL-06 — Re-enable type gradient flow during warmup.
        # The previous implementation detached type_logits during warmup (D-05),
        # preventing the type classifier from learning. With proper regularization
        # (focal loss, class weights, entropy penalty), type gradients can be
        # active from step 1 without destabilizing gain learning. The hybrid warmup
        # gate now only controls freq/Q losses, not type loss.
        pred_type_logits_for_match = pred_type_logits

        # Ensure matcher uses the current dynamic weights from curriculum
        self.param_loss.matcher.lambda_gain = self.matcher_lambda_gain
        self.param_loss.matcher.lambda_freq = self.matcher_lambda_freq
        self.param_loss.matcher.lambda_q = self.matcher_lambda_q
        self.param_loss.matcher.lambda_type_match = self.matcher_lambda_type_match

        # 1. Parameter regression (permutation-invariant) + type matching in one pass.
        # Call the inner matcher directly so we get the same permutation applied to
        # filter types — param_loss.forward() only returns scalars and discards the
        # permutation, leaving type targets in their original unmatched order.
        matched_gain, matched_freq, matched_q, matched_filter_type = (
            self.param_loss.matcher(
                pred_gain,
                pred_freq,
                pred_q,
                target_gain,
                target_freq,
                target_q,
                target_filter_type=target_filter_type,
                pred_type_logits=pred_type_logits_for_match,
            )
        )
        # LOSS-03: Log-cosh for gain (smoother gradients than Huber)
        loss_gain = log_cosh_loss(pred_gain, matched_gain).mean()

        # Sign penalty: penalize predictions where sign(pred) != sign(gt).
        # Targets the pattern where model predicts +X dB when GT is -Y dB.
        # Penalty is proportional to the absolute error so large sign flips
        # cost more than small ones near 0 dB.
        if self.sign_penalty_weight > 0.0:
            sign_mismatch = (pred_gain * matched_gain) < 0.0  # True where signs differ
            sign_penalty = (sign_mismatch.float() * (pred_gain - matched_gain).abs()).mean()
            loss_gain = loss_gain + self.sign_penalty_weight * sign_penalty
            components["sign_penalty"] = sign_penalty.detach()
        else:
            components["sign_penalty"] = torch.tensor(0.0, device=pred_gain.device)

        # Freq and Q still use Huber on log-space
        loss_freq = self.huber(
            torch.log(pred_freq + 1e-8), torch.log(matched_freq + 1e-8)
        )
        loss_q = self.huber(
            torch.log(pred_q + 1e-8), torch.log(matched_q + 1e-8)
        )

        # Warmup gating: zero out non-gain losses during warmup
        if not is_freq_q_active:
            loss_freq = loss_freq * 0.0
            loss_q = loss_q * 0.0

        components["loss_gain"] = loss_gain
        components["loss_freq"] = loss_freq
        components["loss_q"] = loss_q

        # Apply dynamic component weights from curriculum stage
        loss_gain = loss_gain * self.lambda_gain
        loss_freq = loss_freq * self.lambda_freq
        loss_q = loss_q * self.lambda_q

        # 2. Filter type classification against Hungarian-matched targets.
        # D-04: During warmup, type loss is zero (gain-only training)
        # Class-balanced focal loss to prevent peaking filter collapse
        B, N, C = pred_type_logits.shape
        valid_type_mask = torch.ones(
            B * N, device=pred_gain.device, dtype=pred_type_logits.dtype
        )

        if is_type_active:
            logits_flat = pred_type_logits.reshape(B * N, C)  # (B*N, 5)
            targets_flat = matched_filter_type.reshape(B * N)  # (B*N,)

            # Apply gain-gated masking to zero out type gradients for ambiguous (flat) filters
            # EXEMPT HP/LP from gain-gating since they always have 0 dB gain by definition
            target_gain_flat = matched_gain.reshape(-1)
            target_type_flat = matched_filter_type.reshape(-1)
            is_hplp = (target_type_flat == FILTER_HIGHPASS) | (target_type_flat == FILTER_LOWPASS)
            # Soft gain weighting keeps gradients for near-flat bands while still
            # de-emphasizing ambiguous low-gain peaking/shelf samples.
            gain_weight = torch.clamp(torch.abs(target_gain_flat) / 0.5, min=0.2, max=1.0)
            valid_type_mask = torch.where(
                is_hplp,
                torch.ones_like(gain_weight),
                gain_weight,
            ).to(pred_type_logits.dtype)

            if self.type_loss_mode == "soft_kl":
                soft_targets = self._compute_physics_soft_labels(
                    matched_gain,
                    matched_freq,
                    matched_q,
                    matched_filter_type,
                    num_fft_bins=target_H_mag.shape[-1],
                    num_types=C,
                )
                soft_targets_flat = soft_targets.reshape(B * N, C).to(logits_flat.dtype)
                per_sample_loss = F.kl_div(
                    F.log_softmax(logits_flat, dim=1),
                    soft_targets_flat,
                    reduction="none",
                ).sum(dim=1)
            elif self.type_loss_mode == "balanced_softmax":
                # Balanced Softmax: logit correction with class priors.
                # This counteracts class-frequency bias without oversampling.
                prior_log = torch.log(self.type_class_priors + 1e-8).to(logits_flat.dtype)
                logits_bal = logits_flat + prior_log.unsqueeze(0)
                per_sample_loss = F.cross_entropy(
                    logits_bal,
                    targets_flat,
                    label_smoothing=self.type_label_smoothing,
                    reduction="none",
                )
            else:
                # Default: class-balanced focal loss.
                log_probs = F.log_softmax(logits_flat, dim=1)
                probs = log_probs.exp()
                n_classes = C
                smoothed_targets = (1.0 - self.type_label_smoothing) * F.one_hot(
                    targets_flat, n_classes
                ).float() + self.type_label_smoothing / n_classes

                p_t = (probs * smoothed_targets).sum(dim=1)  # (B*N,)
                # Focal loss: focus on hard examples
                focal_weight = (1.0 - p_t).pow(self.focal_gamma)  # (B*N,)

                # Class-balanced alpha: alpha_t for each sample
                alpha_t = self.type_class_weights[targets_flat]  # (B*N,)
                alpha_t = alpha_t / alpha_t.mean()  # renormalize so mean(alpha)=1

                # Per-sample loss: alpha_t * (1-p_t)^gamma * CE
                per_sample_loss = alpha_t * focal_weight * (
                    -(log_probs * smoothed_targets).sum(dim=1)
                )

            masked_per_sample_loss = per_sample_loss * valid_type_mask
            # Avoid division by zero if mask is entirely 0
            num_valid = valid_type_mask.sum()
            loss_type = masked_per_sample_loss.sum() / (num_valid + 1e-8)

            # Apply dynamic weight to type loss
            loss_type = loss_type * self.lambda_type

            # Batch-level type collapse regularization.
            probs_batch = pred_type_logits.softmax(dim=-1)
            mean_probs = probs_batch.reshape(-1, C).mean(dim=0)

            entropy = -(mean_probs * torch.log(mean_probs + 1e-8)).sum()
            max_entropy = math.log(float(C))
            loss_type_entropy = 1.0 - entropy / max_entropy

            prior = self.type_class_priors.to(mean_probs.dtype)
            loss_type_prior = torch.sum(
                prior * (torch.log(prior + 1e-8) - torch.log(mean_probs + 1e-8))
            )
        else:
            loss_type = torch.tensor(0.0, device=pred_gain.device)
            loss_type_entropy = torch.tensor(0.0, device=pred_gain.device)
            loss_type_prior = torch.tensor(0.0, device=pred_gain.device)
        components["type_loss"] = loss_type
        components["type_entropy_loss"] = loss_type_entropy
        components["type_prior_loss"] = loss_type_prior

        # 2b. Hierarchical type losses (if hierarchical head is active)
        loss_hier_broad = torch.tensor(0.0, device=pred_gain.device)
        loss_hier_pass = torch.tensor(0.0, device=pred_gain.device)
        loss_hier_shelf = torch.tensor(0.0, device=pred_gain.device)
        if hier_aux is not None:
            broad_logits, pass_logits, shelf_logits = hier_aux
            B_h, N_h, _ = broad_logits.shape
            target_type_flat = matched_filter_type.reshape(B_h * N_h)

            # Map 5-class targets to 3-class broad targets
            broad_target = torch.tensor(
                [FINE_TO_BROAD[i] for i in range(5)],
                device=target_type_flat.device, dtype=target_type_flat.dtype
            )
            broad_target_flat = broad_target[target_type_flat]  # (B*N,)

            # Gain-gated mask (reuse same logic)
            target_gain_flat = matched_gain.reshape(-1)
            is_hplp = (target_type_flat == FILTER_HIGHPASS) | (target_type_flat == FILTER_LOWPASS)
            gain_weight = torch.clamp(torch.abs(target_gain_flat) / 0.5, min=0.2, max=1.0)
            valid_mask = torch.where(
                is_hplp,
                torch.ones_like(gain_weight),
                gain_weight,
            ).float()

            # Broad class focal loss (3-class)
            broad_flat = broad_logits.reshape(B_h * N_h, 3)
            broad_log_probs = F.log_softmax(broad_flat, dim=1)
            broad_probs = broad_log_probs.exp()
            broad_onehot = F.one_hot(broad_target_flat, 3).float()
            broad_p_t = (broad_probs * broad_onehot).sum(dim=1)
            broad_focal = (1.0 - broad_p_t).pow(2.0)
            broad_ce = -(broad_log_probs * broad_onehot).sum(dim=1)
            loss_hier_broad = (broad_focal * broad_ce * valid_mask).sum() / (valid_mask.sum() + 1e-8)
            loss_hier_broad = loss_hier_broad * 3.0  # weight for broad classification

            # Pass subclass: binary CE for HP vs LP (only on pass bands)
            pass_mask = (broad_target_flat == BROAD_PASS)
            if pass_mask.any():
                pass_flat = pass_logits.reshape(B_h * N_h, 2)
                # HP=0, LP=1 for the binary target
                pass_target = (target_type_flat == FILTER_LOWPASS).long()  # HP→0, LP→1
                pass_loss_per = F.cross_entropy(pass_flat[pass_mask], pass_target[pass_mask], reduction='none')
                # Also apply gain gate
                pass_valid = valid_mask[pass_mask]
                loss_hier_pass = (pass_loss_per * pass_valid).sum() / (pass_valid.sum() + 1e-8)
                loss_hier_pass = loss_hier_pass * 1.0

            # Shelf subclass: binary CE for LS vs HS (only on shelf bands)
            shelf_mask = (broad_target_flat == BROAD_SHELF)
            if shelf_mask.any():
                shelf_flat = shelf_logits.reshape(B_h * N_h, 2)
                # LS=0, HS=1
                shelf_target = (target_type_flat == FILTER_HIGHSHELF).long()  # LS→0, HS→1
                shelf_loss_per = F.cross_entropy(shelf_flat[shelf_mask], shelf_target[shelf_mask], reduction='none')
                shelf_valid = valid_mask[shelf_mask]
                loss_hier_shelf = (shelf_loss_per * shelf_valid).sum() / (shelf_valid.sum() + 1e-8)
                loss_hier_shelf = loss_hier_shelf * 2.0  # Higher weight — hardest distinction

        components["hier_broad_loss"] = loss_hier_broad
        components["hier_pass_loss"] = loss_hier_pass
        components["hier_shelf_loss"] = loss_hier_shelf

        # 2c. Supervised contrastive loss on per-band embeddings.
        loss_supcon = torch.tensor(0.0, device=pred_gain.device)
        if self.lambda_supcon > 0.0 and band_embedding is not None:
            supcon_value = self._supervised_contrastive_loss(
                band_embedding,
                matched_filter_type,
                sample_weights=valid_type_mask.reshape(B, N),
            )
            if supcon_value is not None and torch.isfinite(supcon_value):
                loss_supcon = supcon_value
        components["supcon_loss"] = loss_supcon

        # 3. Frequency response magnitude L1 — LOGGED ONLY (not in core_losses).
        # Computed under no_grad to avoid wasting GPU on unused gradient graphs.
        with torch.no_grad():
            pred_H_mag_hard_safe = torch.clamp(pred_H_mag_hard.float().detach(), min=1e-6, max=1e6)
            target_H_mag_safe = torch.clamp(target_H_mag.float(), min=1e-6, max=1e6)
            loss_hmag = F.l1_loss(torch.log(pred_H_mag_hard_safe), torch.log(target_H_mag_safe))
        components["hmag_loss"] = loss_hmag

        # 4. Spectral reconstruction loss (LOSS-05, D-06)
        # Uses H_mag_soft (Gumbel-Softmax path) for differentiable type gradients.
        # Direct L1 on log-magnitude: log(pred_H_mag_soft) vs log(target_H_mag).
        # Replaces MR-STFT which requires time-domain audio (unavailable from precomputed cache).
        if is_spectral_active:
            pred_spec_safe = torch.clamp(pred_H_mag_soft.float(), min=1e-6, max=1e6)
            target_spec_safe = torch.clamp(target_H_mag.float(), min=1e-6, max=1e6)
            pred_log_spec = torch.log(pred_spec_safe)
            target_log_spec = torch.log(target_spec_safe)
            loss_spectral = F.l1_loss(pred_log_spec, target_log_spec)

            # Spectral slope consistency: matches local shape, not just absolute magnitude.
            pred_slope = pred_log_spec[..., 1:] - pred_log_spec[..., :-1]
            target_slope = target_log_spec[..., 1:] - target_log_spec[..., :-1]
            loss_slope = F.l1_loss(pred_slope, target_slope)
        else:
            loss_spectral = torch.tensor(0.0, device=pred_gain.device)
            loss_slope = torch.tensor(0.0, device=pred_gain.device)
        components["spectral_loss"] = loss_spectral
        components["slope_loss"] = loss_slope

        # 5. Band activity regularization — LOGGED ONLY
        with torch.no_grad():
            if active_band_mask is not None:
                inactive_mask = ~active_band_mask
                loss_activity = (pred_gain * inactive_mask.float()).abs().mean()
            else:
                loss_activity = torch.tensor(0.0, device=pred_gain.device)
        components["activity_loss"] = loss_activity

        # 6. Frequency spread regularization — LOGGED ONLY
        # AUDIT: CRITICAL-06 — Fixed sign inversion: spread_loss was negative,
        # rewarding concentrated frequency predictions instead of penalizing them.
        # The spread metric measures mean pairwise log-frequency distance; higher
        # values mean bands are more spread out. We penalize LOW spread (concentrated
        # predictions) by using: loss = max_spread - spread, which is always >= 0.
        with torch.no_grad():
            if pred_freq.shape[1] > 1:
                log_freq = torch.log(pred_freq + 1e-8)
                diff = log_freq.unsqueeze(-1) - log_freq.unsqueeze(-2)  # (B, N, N)
                spread = diff.abs().mean()
                max_spread = math.log(20000.0 / 20.0)  # ~6.9
                spread = torch.clamp(spread, max=max_spread)
                # AUDIT FIX: Reward spread (diversity), penalize concentration
                # Old (inverted): loss_spread = -spread  (rewarded concentration)
                # New (correct):   loss_spread = max_spread - spread  (penalizes concentration)
                loss_spread = max_spread - spread
            else:
                loss_spread = torch.tensor(0.0, device=pred_gain.device)
        components["spread_loss"] = loss_spread

        # 9. HP/LP zero-gain supervision — LOGGED ONLY
        with torch.no_grad():
            loss_gain_zero = torch.tensor(0.0, device=pred_gain.device)
            zero_gain_mask = (
                (matched_filter_type == FILTER_HIGHPASS)
                | (matched_filter_type == FILTER_LOWPASS)
                | (matched_gain.abs() < 0.5)
            )
            if zero_gain_mask.any():
                loss_gain_zero = F.smooth_l1_loss(
                    pred_gain[zero_gain_mask],
                    torch.zeros_like(pred_gain[zero_gain_mask]),
                )
        components["loss_gain_zero"] = loss_gain_zero

        # 10. H_db direct prediction loss — LOGGED ONLY
        with torch.no_grad():
            if h_db_pred is not None and h_db_target is not None:
                loss_hdb = F.l1_loss(h_db_pred, h_db_target)
            else:
                loss_hdb = torch.tensor(0.0, device=pred_gain.device)
        components["hdb_loss"] = loss_hdb

        # 11. Teacher-forced typed spectral loss
        # Render predicted params with GT filter types — breaks spectral shortcut
        # because wrong params + correct types ≠ target spectrum.
        if H_mag_typed is not None and target_H_mag is not None:
            typed_safe = torch.clamp(H_mag_typed.float(), min=1e-6, max=1e6)
            tgt_safe = torch.clamp(target_H_mag.float(), min=1e-6, max=1e6)
            loss_typed_spectral = F.l1_loss(torch.log(typed_safe), torch.log(tgt_safe))
        else:
            loss_typed_spectral = torch.tensor(0.0, device=pred_gain.device)
        components["typed_spectral_loss"] = loss_typed_spectral

        # 12. FiLM diversity loss (penalize small gamma → near-identity FiLM)
        # AUDIT: P2-14 — This component was consistently zero in training logs.
        # Disabled to save compute. Re-enable if FiLM gamma analysis shows it's needed.
        loss_film_diversity = torch.tensor(0.0, device=pred_gain.device)
        components["film_diversity_loss"] = loss_film_diversity

        # 13. Multi-scale render-domain loss (Fix 3)
        # CRITICAL: detach type_probs to prevent spectral→type gradient conflict
        # (same reasoning as H_mag_soft detach in model_tcn.py)
        loss_multi_scale = torch.tensor(0.0, device=pred_gain.device)
        if self._dsp_cascade is not None and type_probs is not None:
            loss_multi_scale = multi_scale_spectral_loss(
                pred_gain, pred_freq, pred_q, type_probs.detach(),
                target_H_mag, self._dsp_cascade,
            )
        components["multi_scale_loss"] = loss_multi_scale

        # Fix 2: 7 core terms through learned uncertainty weighting
        # Dropped terms (hmag, activity, spread, embed_var, contrastive, hdb,
        # gain_zero, film_diversity) are still
        # computed above for logging but excluded from gradient signal.
        #
        # Apply lambda scaling to spectral losses (type/gain/freq/q already scaled above).
        loss_spectral_scaled = loss_spectral * self.lambda_spectral
        loss_typed_spectral_scaled = loss_typed_spectral * self.lambda_typed_spectral
        loss_multi_scale_scaled = loss_multi_scale * self.lambda_multi_scale
        core_losses = [
            loss_spectral_scaled,        # 0: soft-type spectral L1
            loss_typed_spectral_scaled,  # 1: teacher-forced spectral
            loss_type,                   # 2: type classification
            loss_gain,                   # 3: gain supervision
            loss_freq,                   # 4: freq supervision
            loss_q,                      # 5: Q supervision
            loss_multi_scale_scaled,     # 6: multi-scale render-domain
        ]
        total_loss = self.uncertainty_loss(core_losses)
        total_loss = total_loss + self.lambda_slope * loss_slope
        total_loss = total_loss + self.lambda_supcon * loss_supcon

        # Type-distribution regularization to reduce single-class collapse.
        total_loss = total_loss + self.lambda_type_entropy * loss_type_entropy
        total_loss = total_loss + self.lambda_type_prior * loss_type_prior

        # Add hierarchical type losses (outside uncertainty weighting for stability)
        total_loss = total_loss + loss_hier_broad + loss_hier_pass + loss_hier_shelf

        # Clamp total loss to prevent NaN propagation from any single component
        if total_loss > 5e3:
            print(f"  [loss] WARNING: total_loss={total_loss.item():.1f} approaching clamp boundary")
        total_loss = torch.clamp(total_loss, max=1e4)

        # Log learned weights for monitoring
        with torch.no_grad():
            for i, name in enumerate(["spectral", "typed_spectral", "type",
                                       "gain", "freq", "q", "multi_scale"]):
                components[f"uw_{name}"] = torch.exp(-2 * self.uncertainty_loss.log_sigma[i])

        # AUDIT: P2-14 — Validate loss components for sanity (zero/negative detection)
        self._validate_loss_components(components, total_loss)

        return total_loss, components

    def _validate_loss_components(self, components: dict, total_loss: torch.Tensor) -> None:
        """
        Audit loss components for common issues:
        - Negative values (inverted regularization)
        - Zero contributions (wasted compute)
        - Extreme dominance (>50% of total)

        AUDIT: P2-14 — Periodic validation catches loss component drift.
        """
        if not hasattr(self, '_validation_step'):
            self._validation_step = 0
        self._validation_step += 1

        # Only validate every 100 calls to avoid log spam
        if self._validation_step % 100 != 0:
            return

        # Check for negative logged-only components
        for name, val in components.items():
            if isinstance(val, torch.Tensor) and val.numel() == 1:
                v = val.item()
                if v < -1e-6:
                    # AUDIT: CRITICAL-06 fix — spread_loss should now be positive
                    if name == "spread_loss":
                        print(f"  [loss] WARNING: {name}={v:.4f} is still negative after fix! "
                              f"Check the sign correction in loss_multitype.py")

        # Check component contribution balance (logged-only components)
        finite_components = {
            k: abs(v.item()) if isinstance(v, torch.Tensor) else abs(v)
            for k, v in components.items()
            if isinstance(v, (torch.Tensor, float, int)) and abs(
                v.item() if isinstance(v, torch.Tensor) else v
            ) > 1e-8
        }
        if finite_components:
            total_component = sum(finite_components.values())
            dominant = {
                k: v / total_component
                for k, v in finite_components.items()
                if v / total_component > 0.5
            }
            if dominant:
                for k, pct in dominant.items():
                    if self._validation_step % 500 == 0:  # Less frequent for dominance warnings
                        print(f"  [loss] Component '{k}' dominates at {pct:.0%} of total. "
                              f"Consider rebalancing lambda weights.")
