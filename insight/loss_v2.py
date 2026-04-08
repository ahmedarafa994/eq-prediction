"""
Simplified multi-type EQ loss used by `insight/train.py`.

L_total =
    lambda_spectral * L_spectral
    + lambda_param * L_param
    + lambda_gain_zero * L_gain_zero
    + lambda_type * L_type
    + lambda_contrastive * L_contrastive

Where:
  L_spectral  : log-magnitude L1 on the predicted transfer function
  L_param     : Hungarian-matched gain / frequency / Q regression
  L_gain_zero : explicit zero-gain penalty for HP/LP and near-zero targets
  L_type      : class-balanced focal CE with label smoothing
  L_contrastive: cosine anti-collapse penalty on encoder embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from differentiable_eq import FILTER_HIGHPASS, FILTER_LOWPASS
from loss_multitype import HungarianBandMatcher, log_cosh_loss


class SimplifiedEQLoss(nn.Module):
    """Spectral-dominant EQ estimation loss with matched parameter supervision."""

    def __init__(
        self,
        lambda_spectral: float = 5.0,
        lambda_param: float = 1.0,
        lambda_type: float = 2.0,
        lambda_gain: float = 1.0,
        lambda_freq: float = 1.0,
        lambda_q: float = 1.0,
        lambda_gain_zero: float = 0.0,
        label_smoothing: float = 0.02,
        type_class_priors=None,
        focal_gamma: float = 2.0,
        lambda_type_match: float = 0.0,
        lambda_type_prior: float = 0.0,
        lambda_embed_var: float = 0.5,
        lambda_contrastive: float = 0.0,
        embed_var_threshold: float = 0.1,
        class_weight_multipliers=None,
        sign_penalty_weight: float = 0.0,
        lambda_hdb: float = 0.0,
    ):
        super().__init__()
        self.sign_penalty_weight = sign_penalty_weight
        self.lambda_hdb = lambda_hdb
        if lambda_spectral <= 0.0 and lambda_param <= 0.0:
            raise ValueError(
                "SimplifiedEQLoss requires at least one of "
                "`lambda_spectral` or `lambda_param` to be > 0."
            )

        self.lambda_spectral = lambda_spectral
        self.lambda_param = lambda_param
        self.lambda_type = lambda_type
        self.lambda_gain = lambda_gain
        self.lambda_freq = lambda_freq
        self.lambda_q = lambda_q
        self.lambda_gain_zero = lambda_gain_zero
        self.label_smoothing = label_smoothing
        self.focal_gamma = focal_gamma
        self.lambda_type_prior = lambda_type_prior
        self.lambda_embed_var = lambda_embed_var
        self.lambda_contrastive = lambda_contrastive
        self.embed_var_threshold = embed_var_threshold
        self.zero_gain_threshold_db = 0.5

        self.matcher = HungarianBandMatcher(
            lambda_gain=2.0,
            lambda_freq=1.0,
            lambda_q=0.5,
            lambda_type_match=lambda_type_match,
        )
        self.huber = nn.HuberLoss(delta=5.0)

        if type_class_priors is None:
            type_class_priors = [0.5, 0.15, 0.15, 0.1, 0.1]
        priors = torch.as_tensor(type_class_priors, dtype=torch.float32)
        priors = priors / priors.sum()
        inv_priors = 1.0 / torch.clamp(priors, min=1e-6)
        inv_priors = inv_priors / inv_priors.mean()
        if class_weight_multipliers is not None:
            mults = torch.as_tensor(class_weight_multipliers, dtype=torch.float32)
            inv_priors = inv_priors * mults
            inv_priors = inv_priors / inv_priors.mean()
        self.register_buffer("type_class_priors", priors)
        self.register_buffer("type_class_weights", inv_priors)

    def compose_total_loss(self, components, overrides=None):
        overrides = overrides or {}
        total_loss = (
            overrides.get("lambda_spectral", self.lambda_spectral)
            * components["spectral_loss"]
            + overrides.get("lambda_param", self.lambda_param)
            * components["param_loss"]
            + overrides.get("lambda_gain_zero", self.lambda_gain_zero)
            * components["loss_gain_zero"]
            + overrides.get("lambda_type", self.lambda_type)
            * components["type_loss"]
            + overrides.get("lambda_type_prior", self.lambda_type_prior)
            * components["type_prior_loss"]
            + overrides.get("lambda_embed_var", self.lambda_embed_var)
            * components.get("embed_var_loss", components["spectral_loss"].new_zeros(()))
            + overrides.get("lambda_contrastive", self.lambda_contrastive)
            * components.get(
                "contrastive_loss", components["spectral_loss"].new_zeros(())
            )
            + overrides.get("lambda_hdb", self.lambda_hdb)
            * components.get("hdb_loss", components["spectral_loss"].new_zeros(()))
        )
        return torch.clamp(total_loss, max=1e4)

    def forward(
        self,
        pred_gain: torch.Tensor,
        pred_freq: torch.Tensor,
        pred_q: torch.Tensor,
        pred_type_logits: torch.Tensor,
        pred_H_mag: torch.Tensor,
        target_gain: torch.Tensor,
        target_freq: torch.Tensor,
        target_q: torch.Tensor,
        target_filter_type: torch.Tensor,
        target_H_mag: torch.Tensor,
        embedding: torch.Tensor = None,
        h_db_pred: torch.Tensor = None,
        h_db_target: torch.Tensor = None,
    ):
        components = {}

        pred_safe = torch.clamp(pred_H_mag.float(), min=1e-6, max=1e6)
        target_safe = torch.clamp(target_H_mag.float(), min=1e-6, max=1e6)
        loss_spectral = F.l1_loss(torch.log(pred_safe), torch.log(target_safe))
        components["spectral_loss"] = loss_spectral

        matched_gain, matched_freq, matched_q, matched_filter_type = self.matcher(
            pred_gain,
            pred_freq,
            pred_q,
            target_gain,
            target_freq,
            target_q,
            pred_type_logits=pred_type_logits,
            target_filter_type=target_filter_type,
        )

        loss_gain = log_cosh_loss(pred_gain, matched_gain).mean()
        if self.sign_penalty_weight > 0.0:
            sign_mismatch = (pred_gain * matched_gain) < 0
            if sign_mismatch.any():
                sign_penalty = (pred_gain - matched_gain).abs()[sign_mismatch].mean()
            else:
                sign_penalty = pred_gain.new_zeros(())
            loss_gain = loss_gain + self.sign_penalty_weight * sign_penalty
        loss_freq = self.huber(
            torch.log(pred_freq + 1e-8), torch.log(matched_freq + 1e-8)
        )
        loss_q = self.huber(torch.log(pred_q + 1e-8), torch.log(matched_q + 1e-8))
        loss_param = (
            self.lambda_gain * loss_gain
            + self.lambda_freq * loss_freq
            + self.lambda_q * loss_q
        )
        components["loss_gain"] = loss_gain
        components["loss_freq"] = loss_freq
        components["loss_q"] = loss_q
        components["param_loss"] = loss_param

        zero_gain_mask = (
            (matched_filter_type == FILTER_HIGHPASS)
            | (matched_filter_type == FILTER_LOWPASS)
            | (matched_gain.abs() < self.zero_gain_threshold_db)
        )
        if zero_gain_mask.any():
            zero_target = torch.zeros_like(pred_gain[zero_gain_mask])
            loss_gain_zero = F.smooth_l1_loss(
                pred_gain[zero_gain_mask],
                zero_target,
            )
        else:
            loss_gain_zero = pred_gain.new_zeros(())
        components["loss_gain_zero"] = loss_gain_zero

        batch_size, num_bands, num_types = pred_type_logits.shape
        logits_flat = pred_type_logits.reshape(batch_size * num_bands, num_types)
        targets_flat = matched_filter_type.reshape(batch_size * num_bands)

        log_probs = F.log_softmax(logits_flat, dim=1)
        probs = log_probs.exp()
        smoothed_targets = (
            (1.0 - self.label_smoothing)
            * F.one_hot(targets_flat, num_types).float()
            + self.label_smoothing / num_types
        )
        # FIX: focal weight derived from TRUE-class probability only, with mean
        # normalization (not weight-sum normalization).
        # Prior code divided by focal_weight.sum(), which cancels the up-weighting
        # effect for hard examples. Using .mean() preserves the focal loss property
        # that uncertain samples get stronger gradients.
        p_t = (probs * smoothed_targets).sum(dim=1)  # (B*N,)
        focal_weight = (1.0 - p_t).pow(self.focal_gamma)  # (B*N,) scalar per sample
        ce_per_sample = -(log_probs * smoothed_targets).sum(dim=1)  # (B*N,)
        alpha_t = self.type_class_weights[targets_flat]  # (B*N,)
        alpha_t = alpha_t / alpha_t.mean()
        loss_type = (alpha_t * focal_weight * ce_per_sample).mean()
        components["type_loss"] = loss_type
        batch_type_probs = pred_type_logits.softmax(dim=-1).mean(dim=(0, 1))
        loss_type_prior = F.kl_div(
            torch.log(torch.clamp(batch_type_probs, min=1e-8)),
            self.type_class_priors.to(batch_type_probs.dtype),
            reduction="sum",
        )
        components["type_prior_loss"] = loss_type_prior

        # Embedding variance regularization: penalize encoder collapse.
        # FIX: Replace threshold-based loss with continuous penalty.
        # Prior code: F.relu(threshold - embed_var) — zero gradient when var > threshold,
        # allowing slow drift toward collapse. New: quadratic deficit below target.
        loss_embed_var = pred_gain.new_zeros(())
        if embedding is not None and embedding.shape[0] > 1:
            embed_var = embedding.var(dim=0).mean()
            target_var = embedding.shape[-1] * 0.005
            deficit = torch.clamp(target_var - embed_var, min=0.0)
            loss_embed_var = deficit * deficit / (target_var + 1e-8)
            loss_embed_var = torch.clamp(loss_embed_var, max=5.0)
        components["embed_var_loss"] = loss_embed_var

        loss_contrastive = pred_gain.new_zeros(())
        if embedding is not None and embedding.shape[0] > 1 and torch.isfinite(embedding).all():
            embed_norm = F.normalize(embedding.float(), dim=1)
            sim_matrix = torch.mm(embed_norm, embed_norm.t())
            mask = ~torch.eye(
                sim_matrix.shape[0], dtype=torch.bool, device=sim_matrix.device
            )
            if mask.any():
                mean_sim = sim_matrix[mask].mean()
                mean_sim_clamped = torch.clamp(mean_sim, max=0.95)
                loss_contrastive = -torch.log(1.0 - mean_sim_clamped + 1e-3)
                loss_contrastive = torch.clamp(loss_contrastive, max=5.0)
        components["contrastive_loss"] = loss_contrastive

        # H_db direct prediction loss (hybrid spectral-parametric)
        loss_hdb = pred_gain.new_zeros(())
        if h_db_pred is not None and h_db_target is not None:
            loss_hdb = F.l1_loss(h_db_pred, h_db_target)
        components["hdb_loss"] = loss_hdb

        total_loss = self.compose_total_loss(components)
        return total_loss, components
