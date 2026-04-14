"""
Validation loop for the IDSP EQ Estimator training pipeline.

AUDIT: P2-13 — Extracted from the monolithic Trainer class (train.py, 2700+ lines).
Handles validation metric computation, Hungarian matching, and result aggregation.
"""
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from metrics import compute_eq_metrics


class ValidationLoop:
    """Runs validation pass and computes metrics."""

    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        frontend: torch.nn.Module,
        matcher=None,
        device: str = "cuda",
        num_bands: int = 5,
    ):
        self.model = model
        self.criterion = criterion
        self.frontend = frontend
        self.matcher = matcher
        self.device = device
        self.num_bands = num_bands

    @torch.no_grad()
    def run(self, val_loader, epoch: int = 0, log_components: bool = True) -> Dict[str, Any]:
        """
        Run validation pass over the entire validation set.

        Args:
            val_loader: Validation DataLoader
            epoch: Current epoch number (for logging)
            log_components: Whether to aggregate and return per-component losses

        Returns:
            Dict with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        component_accum: Dict[str, float] = {}

        # Accumulators for Hungarian-matched metrics
        all_matched_gains_pred = []
        all_matched_gains_target = []
        all_matched_freqs_pred = []
        all_matched_freqs_target = []
        all_matched_qs_pred = []
        all_matched_qs_target = []
        all_matched_types_pred = []
        all_matched_types_target = []

        for batch_idx, batch in enumerate(val_loader):
            wet = batch["wet_audio"].to(self.device)
            target_gain = batch["gain"].to(self.device)
            target_freq = batch["freq"].to(self.device)
            target_q = batch["q"].to(self.device)
            target_type = batch["filter_type"].to(self.device)

            # Forward pass
            mel = self.frontend.mel_spectrogram(wet).squeeze(1)
            out = self.model(mel, wet_audio=wet)
            pred_gain, pred_freq, pred_q = out["params"]

            # Ground truth frequency response
            target_H_mag = self.model.dsp_cascade(
                target_gain, target_freq, target_q, filter_type=target_type
            )

            # Compute loss
            loss, components = self.criterion(
                pred_gain, pred_freq, pred_q,
                out["type_logits"],
                out["H_mag_soft"],
                out["H_mag"],
                target_gain, target_freq, target_q, target_type,
                target_H_mag,
                embedding=out["embedding"],
            )

            total_loss += loss.item()
            n_batches += 1

            # Accumulate components
            if log_components:
                for k, v in components.items():
                    val = v.item() if isinstance(v, torch.Tensor) else v
                    component_accum[k] = component_accum.get(k, 0.0) + val

            # Accumulate matched metrics
            with torch.no_grad():
                pred_types = pred_gain.new_tensor(
                    out["type_logits"].argmax(dim=-1).cpu(), dtype=torch.long
                )

                # Hungarian matching for this batch
                if self.matcher is not None:
                    matched_gain, matched_freq, matched_q, matched_type = self.matcher(
                        pred_gain.cpu(), pred_freq.cpu(), pred_q.cpu(),
                        target_gain.cpu(), target_freq.cpu(), target_q.cpu(),
                        pred_type_logits=out["type_logits"].cpu(),
                        target_filter_type=target_type.cpu(),
                    )
                    all_matched_gains_pred.append(matched_gain)
                    all_matched_gains_target.append(target_gain.cpu())
                    all_matched_freqs_pred.append(matched_freq)
                    all_matched_freqs_target.append(target_freq.cpu())
                    all_matched_qs_pred.append(matched_q)
                    all_matched_qs_target.append(target_q.cpu())
                    all_matched_types_pred.append(pred_types)
                    all_matched_types_target.append(matched_type)

        # Compute aggregated metrics
        avg_loss = total_loss / max(n_batches, 1)
        components_avg = {k: v / n_batches for k, v in component_accum.items()}

        # Compute Hungarian-matched metrics
        metrics = {}
        if all_matched_gains_pred:
            metrics = compute_eq_metrics(
                torch.cat(all_matched_gains_pred, dim=0),
                torch.cat(all_matched_freqs_pred, dim=0),
                torch.cat(all_matched_qs_pred, dim=0),
                torch.cat(all_matched_types_pred, dim=0),
                torch.cat(all_matched_gains_target, dim=0),
                torch.cat(all_matched_freqs_target, dim=0),
                torch.cat(all_matched_qs_target, dim=0),
                torch.cat(all_matched_types_target, dim=0),
            )

        # Add loss components to metrics
        metrics["val_loss"] = avg_loss
        metrics["val_loss_hard"] = avg_loss  # deprecated, same as val_loss
        metrics["val_loss_soft"] = avg_loss  # deprecated, same as val_loss
        metrics.update(components_avg)

        self.model.train()
        return metrics
