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


def log_cosh_loss(pred, target):
    """
    Log-cosh loss: log(cosh(pred - target)).
    - Near zero: ~0.5 * x^2 (smooth quadratic, better gradients than Huber)
    - Far from zero: ~|x| - log(2) (linear, robust to outliers)
    - C2 continuous everywhere (unlike Huber's C1 at delta)

    Numerically stable formulation:
        log(cosh(x)) = |x| + log(1 + exp(-2|x|)) - log(2)
    """
    diff = pred - target
    abs_diff = diff.abs()
    return abs_diff + torch.log1p(torch.exp(-2 * abs_diff)) - math.log(2)


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
        lambda_q=1.0,       # D-04: equalized (was 0.5)
        lambda_gain=1.0,     # D-04: equalized (was 2.0)
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

        # Expand for pairwise computation: (B, N, 1) vs (B, 1, N)
        pg = pred_gain.unsqueeze(-1)  # (B, N, 1)
        tg = target_gain.unsqueeze(-2)  # (B, 1, N)
        pf = pred_freq.unsqueeze(-1)
        tf = target_freq.unsqueeze(-2)
        pq = pred_q.unsqueeze(-1)
        tq = target_q.unsqueeze(-2)

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
            type_match = (pred_probs.unsqueeze(2) * target_one_hot.unsqueeze(1)).sum(
                -1
            )  # (B, N, N)
            type_cost = -type_match
            cost = cost + self.lambda_type_match * type_cost

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
            # Guard against NaN/inf from early training instability
            cost_np = np.nan_to_num(cost_np, nan=0.0, posinf=1e6, neginf=-1e6)
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

    def __init__(self, lambda_freq=1.0, lambda_q=1.0, lambda_type_match=0.5):
        super().__init__()
        self.matcher = HungarianBandMatcher(
            lambda_freq, lambda_q, lambda_gain=1.0, lambda_type_match=lambda_type_match
        )
        self.huber = nn.HuberLoss(delta=5.0)

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
    - L_type: Filter type classification (cross-entropy)
    - L_spectral: Spectral magnitude L1 (LOSS-05)
    - L_hmag: Frequency response magnitude L1 (hard types)
    - L_activity: Band activity regularization
    - L_spread: Frequency spread regularization
    - L_embed_var: Embedding variance regularization (anti-collapse)
    - L_contrastive: Spectral contrastive loss (anti-collapse)

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
        lambda_hmag=0.3,
        lambda_activity=0.1,
        lambda_spread=0.05,
        lambda_embed_var=0.5,
        lambda_contrastive=0.1,
        lambda_type_match=0.5,
        embed_var_threshold=0.1,
        warmup_epochs=5,
    ):
        super().__init__()
        self.lambda_param = lambda_param
        self.lambda_gain = lambda_gain
        self.lambda_freq = lambda_freq
        self.lambda_q = lambda_q
        self.lambda_type = lambda_type
        self.lambda_spectral = lambda_spectral
        self.lambda_hmag = lambda_hmag
        self.lambda_activity = lambda_activity
        self.lambda_spread = lambda_spread
        self.lambda_embed_var = lambda_embed_var
        self.lambda_contrastive = lambda_contrastive
        self.embed_var_threshold = embed_var_threshold
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0

        # D-03/D-04: Hybrid warmup gate — warmup ends when BOTH epoch threshold AND
        # gain MAE threshold are met (hard cap at 15 epochs prevents infinite warmup).
        self.gain_mae_ema = 2.5  # EMA of per-batch gain MAE (dB), alpha=0.1

        self.param_loss = PermutationInvariantParamLoss(
            lambda_type_match=lambda_type_match
        )
        # Class-balanced focal loss for type classification (per D-09, D-10)
        # Replaces nn.CrossEntropyLoss. Inverse-frequency weighting from type_weights
        # handles 5:1 class imbalance (peaking 50% vs HP/LP 10%).
        type_weights_cfg = [0.5, 0.15, 0.15, 0.1, 0.1]  # from config data.type_weights
        inv_w = 1.0 / torch.tensor(type_weights_cfg, dtype=torch.float32)
        inv_w = inv_w / inv_w.sum() * len(type_weights_cfg)
        self.register_buffer('type_class_weights', inv_w)
        self.focal_gamma = 2.0  # D-09: focal loss focusing parameter
        self.type_label_smoothing = 0.05
        self.mr_stft = MultiResolutionSTFTLoss()

    def update_gain_mae(self, batch_gain_mae: float, alpha: float = 0.1) -> None:
        """Update the exponential moving average of per-batch gain MAE.

        Called by train.py after each training batch. The updated EMA is used
        in the hybrid warmup gate (D-03/D-04) to determine when freq/Q/type
        losses should activate.

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

        is_freq_q_active = not is_warmup
        is_type_active = not is_warmup and self.current_epoch >= self.warmup_epochs + 1
        is_spectral_active = not is_warmup and self.current_epoch >= self.warmup_epochs + 2

        # --- Anti-collapse losses (computed first so they appear early in logs) ---

        # 7. Embedding variance regularization (L_embed_var)
        # Prevents encoder collapse by penalizing embeddings that have low variance
        # across the batch. When all embeddings are identical (collapse), per-dimension
        # variance drops to zero and this loss fires. The threshold allows variance
        # above it to pass for free — it only activates when the encoder starts to
        # collapse. This is a direct, per-batch diagnostic that requires no auxiliary
        # network or negative sampling.
        loss_embed_var = torch.tensor(0.0, device=pred_gain.device)
        if embedding is not None:
            # Variance of each embedding dimension across the batch, then average
            embed_var = embedding.var(dim=0).mean()
            loss_embed_var = F.relu(self.embed_var_threshold - embed_var)
        components["embed_var_loss"] = loss_embed_var

        # 8. Spectral contrastive loss (L_contrastive)
        # Encourages embeddings from different mel profiles to be dissimilar.
        # This complements L_embed_var: variance regularization prevents a trivial
        # collapse to a single point, but the encoder could still produce a narrow
        # manifold. Contrastive loss explicitly pushes embeddings apart by penalizing
        # high pairwise cosine similarity. The formulation is simple: minimize the
        # mean pairwise cosine similarity, which ensures the encoder uses the full
        # embedding space rather than a thin subspace.
        loss_contrastive = torch.tensor(0.0, device=pred_gain.device)
        if embedding is not None and embedding.shape[0] > 1:
            # Skip if embedding contains NaN/inf (from numerical instability)
            if torch.isfinite(embedding).all():
                # Normalize embeddings to unit sphere for cosine similarity
                embed_norm = F.normalize(embedding, dim=1)
                # Pairwise cosine similarity matrix (B, B)
                sim_matrix = torch.mm(embed_norm, embed_norm.t())
                # Exclude self-similarity (diagonal is always 1.0)
                B = sim_matrix.shape[0]
                mask = ~torch.eye(B, dtype=torch.bool, device=sim_matrix.device)
                mean_sim = sim_matrix[mask].mean()
                # Clamp similarity to prevent log(negative) = NaN and gradient
                # explosion when sim approaches 1.0 (gradient ~1/(1-sim) → huge)
                mean_sim_clamped = torch.clamp(mean_sim.float(), max=0.95)
                loss_contrastive = -torch.log(1.0 - mean_sim_clamped + 1e-3)
                loss_contrastive = torch.clamp(loss_contrastive, max=5.0)
        components["contrastive_loss"] = loss_contrastive

        # DATA-02: Detach type logits from gain gradient path during warmup.
        # During gain-only warmup, the type classification head should NOT
        # receive gradient signal from gain regression. If type_logits are not
        # detached, noisy type gradients (random at init) contaminate gain learning.
        # After warmup, joint gradients are allowed — type and gain learn together.
        if is_warmup:
            pred_type_logits_for_match = pred_type_logits.detach()
        else:
            pred_type_logits_for_match = pred_type_logits

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
        # Freq and Q still use Huber on log-space
        loss_freq = self.param_loss.huber(
            torch.log(pred_freq + 1e-8), torch.log(matched_freq + 1e-8)
        )
        loss_q = self.param_loss.huber(
            torch.log(pred_q + 1e-8), torch.log(matched_q + 1e-8)
        )

        # Warmup gating: zero out non-gain losses during warmup
        if not is_freq_q_active:
            loss_freq = loss_freq * 0.0
            loss_q = loss_q * 0.0

        components["loss_gain"] = loss_gain
        components["loss_freq"] = loss_freq
        components["loss_q"] = loss_q

        # Use independent per-parameter weights when lambda_param is disabled (0.0)
        if self.lambda_param > 0:
            loss_param = (
                self.lambda_param * (loss_freq + loss_q) + self.lambda_gain * loss_gain
            )
        else:
            loss_param = (
                self.lambda_gain * loss_gain
                + self.lambda_freq * loss_freq
                + self.lambda_q * loss_q
            )
        components["param_loss"] = loss_param

        # 2. Filter type classification against Hungarian-matched targets.
        # D-04: During warmup, type loss is zero (gain-only training)
        # Class-balanced focal loss (per D-09, D-10)
        if is_type_active:
            B, N, C = pred_type_logits.shape
            type_logits_flat = pred_type_logits.reshape(B * N, C)
            type_targets_flat = matched_filter_type.reshape(B * N)
            ce_loss = F.cross_entropy(
                type_logits_flat, type_targets_flat,
                weight=self.type_class_weights, reduction='none',
                label_smoothing=self.type_label_smoothing,
            )
            pt = torch.exp(-ce_loss)
            focal_weight = (1.0 - pt) ** self.focal_gamma
            loss_type = (focal_weight * ce_loss).mean()
        else:
            loss_type = torch.tensor(0.0, device=pred_gain.device)
        components["type_loss"] = loss_type

        # 3. Frequency response magnitude L1 — uses H_mag_hard (argmax types)
        # CRITICAL: detach to prevent NaN backward through torch.where argmax branch.
        # (0 * NaN = NaN in gradient accumulation from non-selected filter branches)
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
            loss_spectral = F.l1_loss(torch.log(pred_spec_safe), torch.log(target_spec_safe))
        else:
            loss_spectral = torch.tensor(0.0, device=pred_gain.device)
        components["spectral_loss"] = loss_spectral

        # 5. Band activity regularization
        if active_band_mask is not None:
            inactive_mask = ~active_band_mask
            loss_activity = (pred_gain * inactive_mask.float()).abs().mean()
        else:
            loss_activity = torch.tensor(0.0, device=pred_gain.device)
        components["activity_loss"] = loss_activity

        # 6. Frequency spread regularization (bounded to prevent pushing frequencies to extremes)
        if pred_freq.shape[1] > 1:
            log_freq = torch.log(pred_freq + 1e-8)
            # Pairwise log-frequency differences
            diff = log_freq.unsqueeze(-1) - log_freq.unsqueeze(-2)  # (B, N, N)
            spread = diff.abs().mean()
            # The maximum achievable mean pairwise spread is bounded by the
            # log-frequency range (~6.9 octaves for 20-20000 Hz).  Clamp so
            # the repulsive force saturates instead of growing without bound.
            max_spread = math.log(20000.0 / 20.0)  # ~6.9
            spread = torch.clamp(spread, max=max_spread)
            loss_spread = -spread  # Maximize spread
        else:
            loss_spread = torch.tensor(0.0, device=pred_gain.device)
        components["spread_loss"] = loss_spread

        # Total: independent per-parameter weights (no combined lambda_param wrapper)
        total_loss = (
            self.lambda_gain * loss_gain
            + self.lambda_freq * loss_freq
            + self.lambda_q * loss_q
            + self.lambda_type * loss_type
            + self.lambda_spectral * loss_spectral
            + self.lambda_hmag * loss_hmag
            + self.lambda_activity * loss_activity
            + self.lambda_spread * loss_spread
            + self.lambda_embed_var * loss_embed_var
            + self.lambda_contrastive * loss_contrastive
        )

        # Clamp total loss to prevent NaN propagation from any single component
        total_loss = torch.clamp(total_loss, max=1e4)

        return total_loss, components
