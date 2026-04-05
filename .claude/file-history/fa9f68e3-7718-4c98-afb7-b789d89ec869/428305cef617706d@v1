"""
Multi-type EQ estimation loss functions.

Key components:
- Hungarian matching for permutation-invariant band assignment (DETR-style)
- Combined regression + classification loss
- Spectral consistency loss for blind estimation
- Auxiliary losses (band activity, frequency spread)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import numpy as np

from loss import MultiResolutionSTFTLoss


class HungarianBandMatcher:
    """
    Solves the band permutation problem using Hungarian (bipartite) matching.

    When predicting N EQ bands, the model's band ordering is arbitrary.
    This finds the optimal assignment between predicted and ground-truth bands
    to minimize parameter distance, following Carion et al. (DETR, 2020).
    """

    def __init__(self, lambda_freq=1.0, lambda_q=0.5):
        self.lambda_freq = lambda_freq
        self.lambda_q = lambda_q

    def compute_cost_matrix(self, pred_gain, pred_freq, pred_q,
                            target_gain, target_freq, target_q):
        """
        Compute pairwise cost matrix between predicted and target bands.

        All inputs: (Batch, Num_Bands)
        Returns: cost matrix (Batch, Num_Pred, Num_Target)
        """
        B, N = pred_gain.shape

        # Expand for pairwise computation: (B, N, 1) vs (B, 1, N)
        pg = pred_gain.unsqueeze(-1)      # (B, N, 1)
        tg = target_gain.unsqueeze(-2)    # (B, 1, N)
        pf = pred_freq.unsqueeze(-1)
        tf = target_freq.unsqueeze(-2)
        pq = pred_q.unsqueeze(-1)
        tq = target_q.unsqueeze(-2)

        # Gain cost: L1 in dB
        cost_gain = (pg - tg).abs()

        # Frequency cost: L1 in log-space (octaves)
        cost_freq = self.lambda_freq * (torch.log(pf + 1e-8) - torch.log(tf + 1e-8)).abs()

        # Q cost: L1 in log-space (decades)
        cost_q = self.lambda_q * (torch.log(pq + 1e-8) - torch.log(tq + 1e-8)).abs()

        # Filter type cost: if available, add a penalty for type mismatch
        cost = cost_gain + cost_freq + cost_q  # (B, N, N)

        return cost

    def match(self, cost_matrix):
        """
        Solve the assignment problem for each batch element.

        Args:
            cost_matrix: (Batch, N_pred, N_target)

        Returns:
            List of (row_indices, col_indices) tuples per batch element
        """
        B = cost_matrix.shape[0]
        assignments = []
        for b in range(B):
            cost_np = cost_matrix[b].detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_np)
            assignments.append((row_ind, col_ind))
        return assignments

    def __call__(self, pred_gain, pred_freq, pred_q,
                 target_gain, target_freq, target_q):
        """
        Find optimal matching and reorder targets.

        Returns:
            target_gain_matched, target_freq_matched, target_q_matched
            — ground truth reordered to match predicted band ordering
        """
        cost = self.compute_cost_matrix(
            pred_gain, pred_freq, pred_q,
            target_gain, target_freq, target_q
        )
        assignments = self.match(cost)

        B, N = pred_gain.shape
        device = pred_gain.device

        matched_gain = torch.zeros_like(target_gain)
        matched_freq = torch.zeros_like(target_freq)
        matched_q = torch.zeros_like(target_q)

        for b in range(B):
            row_ind, col_ind = assignments[b]
            # col_ind[i] is the ground-truth band index that matches predicted band row_ind[i]
            # We want: matched_target[predicted_i] = ground_truth[col for predicted_i]
            perm = torch.zeros(N, dtype=torch.long, device=device)
            for r, c in zip(row_ind, col_ind):
                perm[r] = c
            matched_gain[b] = target_gain[b, perm]
            matched_freq[b] = target_freq[b, perm]
            matched_q[b] = target_q[b, perm]

        return matched_gain, matched_freq, matched_q


class PermutationInvariantParamLoss(nn.Module):
    """
    Parameter regression loss with Hungarian matching for band ordering.
    Uses Huber loss for robustness to outliers.
    """

    def __init__(self, lambda_freq=1.0, lambda_q=0.5):
        super().__init__()
        self.matcher = HungarianBandMatcher(lambda_freq, lambda_q)
        self.huber = nn.HuberLoss(delta=1.0)

    def forward(self, pred_gain, pred_freq, pred_q,
                target_gain, target_freq, target_q):
        """
        All inputs: (Batch, Num_Bands)
        Returns: scalar loss
        """
        matched_gain, matched_freq, matched_q = self.matcher(
            pred_gain, pred_freq, pred_q,
            target_gain, target_freq, target_q
        )

        loss_gain = self.huber(pred_gain, matched_gain)
        loss_freq = self.huber(
            torch.log(pred_freq + 1e-8),
            torch.log(matched_freq + 1e-8)
        )
        loss_q = self.huber(
            torch.log(pred_q + 1e-8),
            torch.log(matched_q + 1e-8)
        )

        return loss_gain + loss_freq + loss_q


class MultiTypeEQLoss(nn.Module):
    """
    Combined loss for multi-type blind EQ estimation.

    Components:
    - L_param: Permutation-invariant parameter regression (Huber)
    - L_type: Filter type classification (cross-entropy)
    - L_spectral: MR-STFT spectral consistency
    - L_hmag: Frequency response magnitude L1
    - L_activity: Band activity regularization
    - L_spread: Frequency spread regularization
    """

    def __init__(self, n_fft=2048, sample_rate=44100,
                 lambda_param=1.0, lambda_type=0.5,
                 lambda_spectral=1.0, lambda_hmag=0.3,
                 lambda_activity=0.1, lambda_spread=0.05):
        super().__init__()
        self.lambda_param = lambda_param
        self.lambda_type = lambda_type
        self.lambda_spectral = lambda_spectral
        self.lambda_hmag = lambda_hmag
        self.lambda_activity = lambda_activity
        self.lambda_spread = lambda_spread

        self.param_loss = PermutationInvariantParamLoss()
        self.type_loss = nn.CrossEntropyLoss()
        self.mr_stft = MultiResolutionSTFTLoss()
        self.huber = nn.HuberLoss()

    def forward(self, pred_gain, pred_freq, pred_q,
                pred_type_logits, pred_H_mag,
                target_gain, target_freq, target_q,
                target_filter_type, target_H_mag,
                pred_audio=None, target_audio=None,
                active_band_mask=None):
        """
        Compute total loss.

        Args:
            pred_gain, pred_freq, pred_q: (B, N) predicted parameters
            pred_type_logits: (B, N, 5) predicted type logits
            pred_H_mag: (B, F) predicted frequency response
            target_gain, target_freq, target_q: (B, N) ground truth params
            target_filter_type: (B, N) ground truth type indices
            target_H_mag: (B, F) ground truth frequency response
            pred_audio: (B, T) optional reconstructed audio
            target_audio: (B, T) optional target audio
            active_band_mask: (B, N) optional mask for active bands

        Returns:
            total_loss: scalar
            components: dict of individual loss values
        """
        components = {}

        # 1. Parameter regression (permutation-invariant)
        loss_param = self.param_loss(
            pred_gain, pred_freq, pred_q,
            target_gain, target_freq, target_q
        )
        components["param_loss"] = loss_param

        # 2. Filter type classification
        # Reshape for cross-entropy: (B*N, 5) and (B*N,)
        B, N, C = pred_type_logits.shape
        loss_type = self.type_loss(
            pred_type_logits.reshape(B * N, C),
            target_filter_type.reshape(B * N)
        )
        components["type_loss"] = loss_type

        # 3. Frequency response magnitude L1
        loss_hmag = F.l1_loss(
            torch.log(pred_H_mag + 1e-8),
            torch.log(target_H_mag + 1e-8)
        )
        components["hmag_loss"] = loss_hmag

        # 4. Spectral consistency (if audio provided)
        if pred_audio is not None and target_audio is not None:
            sc_loss, log_loss = self.mr_stft(pred_audio, target_audio)
            loss_spectral = sc_loss + log_loss
            components["spectral_loss"] = loss_spectral
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

        # 6. Frequency spread regularization
        if pred_freq.shape[1] > 1:
            log_freq = torch.log(pred_freq + 1e-8)
            # Pairwise log-frequency differences
            diff = log_freq.unsqueeze(-1) - log_freq.unsqueeze(-2)  # (B, N, N)
            spread = diff.abs().mean()
            loss_spread = -spread  # Maximize spread
        else:
            loss_spread = torch.tensor(0.0, device=pred_gain.device)
        components["spread_loss"] = loss_spread

        # Total
        total_loss = (
            self.lambda_param * loss_param
            + self.lambda_type * loss_type
            + self.lambda_spectral * loss_spectral
            + self.lambda_hmag * loss_hmag
            + self.lambda_activity * loss_activity
            + self.lambda_spread * loss_spread
        )

        return total_loss, components
