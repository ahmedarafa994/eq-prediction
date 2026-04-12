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
from differentiable_eq import FILTER_HIGHPASS, FILTER_LOWPASS


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
        total = torch.tensor(0.0, device=self.log_sigma.device)
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
        # Render with soft types so gradients flow to classifier
        H_mag_pred = dsp_cascade.forward_soft(
            pred_gain, pred_freq, pred_q,
            type_probs=type_probs, n_fft=n_fft,
        )  # (B, n_fft//2+1)

        # Resample target to match resolution
        target_resampled = F.interpolate(
            target_H_mag.unsqueeze(1), size=n_fft // 2 + 1,
            mode='linear', align_corners=False,
        ).squeeze(1)

        # Log-domain L1 (perceptually meaningful dB comparison)
        pred_db = 20 * torch.log10(H_mag_pred.clamp(min=1e-6))
        tgt_db = 20 * torch.log10(target_resampled.clamp(min=1e-6))
        total = total + F.l1_loss(pred_db, tgt_db)

    return total / len(fft_sizes)


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
            type_cost = 1.0 - p_correct_type
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
        class_weight_multipliers=None,
        type_class_priors=None,
        type_loss_mode="focal",
        label_smoothing=0.05,
        focal_gamma=2.0,
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
        self.matcher_lambda_gain = kwargs.get("matcher_lambda_gain", lambda_gain)
        self.matcher_lambda_freq = kwargs.get("matcher_lambda_freq", lambda_freq)
        self.matcher_lambda_q = kwargs.get("matcher_lambda_q", lambda_q)
        self.matcher_lambda_type_match = kwargs.get("matcher_lambda_type_match", lambda_type_match)
        self.embed_var_threshold = embed_var_threshold
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        self.sign_penalty_weight = sign_penalty_weight
        self.type_loss_mode = str(type_loss_mode).lower()
        self.class_weight_multipliers = (
            list(class_weight_multipliers) if class_weight_multipliers is not None else None
        )

        # D-03/D-04: Hybrid warmup gate — warmup ends when BOTH epoch threshold AND
        # gain MAE threshold are met (hard cap at 15 epochs prevents infinite warmup).
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
        # Initial log_sigmas set so effective weights ≈ design spec static lambdas
        self.uncertainty_loss = UncertaintyWeightedLoss(
            n_losses=7,
            initial_log_sigmas=[-1.1, -0.7, -1.1, -0.4, 0.0, 0.7, -0.7],
        )
        self._dsp_cascade = dsp_cascade  # stored as plain attr, NOT nn.Module

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
        h_db_pred=None,
        h_db_target=None,
        H_mag_typed=None,
        type_probs=None,
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
        loss_embed_var = torch.tensor(0.0, device=pred_gain.device)
        if embedding is not None and embedding.shape[0] > 1:
            embed_var = embedding.var(dim=0).mean()
            # Target variance: embedding_dim * 0.01 gives a reasonable floor
            # for 128-dim embeddings (~1.28). Scale with dimension so the target
            # is invariant to embedding size changes.
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

        # D-05 (DATA-02): Detach type_probs from gain gradient path during warmup.
        # During gain-only warmup, the type classification head should NOT
        # receive gradient signal from gain regression. If type_probs are not detached,
        # noisy type gradients (random at epoch 0) contaminate the gain learning signal.
        # After warmup, joint gradients are allowed -- type and gain learn together.
        if is_warmup:
            pred_type_logits_for_match = pred_type_logits.detach()
        else:
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
        if is_type_active:
            B, N, C = pred_type_logits.shape
            logits_flat = pred_type_logits.reshape(B * N, C)  # (B*N, 5)
            targets_flat = matched_filter_type.reshape(B * N)  # (B*N,)

            # Apply gain-gated masking to zero out type gradients for ambiguous (flat) filters
            target_gain_flat = matched_gain.reshape(-1)
            valid_type_mask = (torch.abs(target_gain_flat) >= 1.0).to(pred_type_logits.dtype)

            if self.type_loss_mode == "balanced_softmax":
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

        # 9. HP/LP zero-gain supervision
        # HP and LP filters should have gain ≈ 0 dB. This explicit loss pushes
        # predicted gain toward zero for bands where the matched target is HP/LP.
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

        # 10. H_db direct prediction loss (hybrid spectral-parametric)
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
        loss_film_diversity = torch.tensor(0.0, device=pred_gain.device)
        components["film_diversity_loss"] = loss_film_diversity

        # 13. Multi-scale render-domain loss (Fix 3)
        loss_multi_scale = torch.tensor(0.0, device=pred_gain.device)
        if self._dsp_cascade is not None and type_probs is not None:
            loss_multi_scale = multi_scale_spectral_loss(
                pred_gain, pred_freq, pred_q, type_probs,
                target_H_mag, self._dsp_cascade,
            )
        components["multi_scale_loss"] = loss_multi_scale

        # Fix 2: 7 core terms through learned uncertainty weighting
        # Dropped terms (hmag, activity, spread, embed_var, contrastive, hdb,
        # gain_zero, type_entropy, type_prior, film_diversity) are still
        # computed above for logging but excluded from gradient signal.
        core_losses = [
            loss_spectral,        # 0: soft-type spectral L1
            loss_typed_spectral,  # 1: teacher-forced spectral
            loss_type,            # 2: type classification
            loss_gain,            # 3: gain supervision
            loss_freq,            # 4: freq supervision
            loss_q,               # 5: Q supervision
            loss_multi_scale,     # 6: multi-scale render-domain
        ]
        total_loss = self.uncertainty_loss(core_losses)

        # Clamp total loss to prevent NaN propagation from any single component
        total_loss = torch.clamp(total_loss, max=1e4)

        # Log learned weights for monitoring
        with torch.no_grad():
            for i, name in enumerate(["spectral", "typed_spectral", "type",
                                       "gain", "freq", "q", "multi_scale"]):
                components[f"uw_{name}"] = torch.exp(-2 * self.uncertainty_loss.log_sigma[i])

        return total_loss, components
