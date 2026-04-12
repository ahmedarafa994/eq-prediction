"""
Lightning module for multi-type blind EQ parameter estimation.
Uses the streaming TCN model with multi-type DSP and permutation-invariant loss.

⚠️  DEPRECATION NOTICE (AUDIT: HIGH-09):
    This is a secondary training pipeline. Primary: `insight/train.py`.
    NOT actively maintained. Verify against `train.py` before production use.
"""
import sys
import torch
import pytorch_lightning as pl
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from model_tcn import StreamingTCNModel
from dsp_frontend import STFTFrontend
from loss_multitype import MultiTypeEQLoss


class EQEstimatorLightning(pl.LightningModule):
    """
    Lightning Module for training the IDSP multi-type EQ parameter estimator.
    """

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        data_cfg = config["data"]
        model_cfg = config["model"]
        loss_cfg = config["loss"]

        # Model
        self.model = StreamingTCNModel(
            n_mels=data_cfg.get("n_mels", 128),
            embedding_dim=model_cfg["encoder"].get("embedding_dim", 128),
            num_bands=data_cfg.get("num_bands", 5),
            channels=model_cfg["encoder"].get("channels", 128),
            num_blocks=model_cfg["encoder"].get("num_blocks", 4),
            num_stacks=model_cfg["encoder"].get("num_stacks", 2),
            sample_rate=data_cfg.get("sample_rate", 44100),
            n_fft=data_cfg.get("n_fft", 2048),
            detach_type_for_params=model_cfg.get("detach_type_for_params", True),
            detach_type_for_render=model_cfg.get("detach_type_for_render", True),
        )

        # STFT frontend for audio→mel conversion
        self.frontend = STFTFrontend(
            n_fft=data_cfg.get("n_fft", 2048),
            hop_length=data_cfg.get("hop_length", 256),
            win_length=data_cfg.get("n_fft", 2048),
            mel_bins=data_cfg.get("n_mels", 128),
            sample_rate=data_cfg.get("sample_rate", 44100),
        )

        # Loss
        self.criterion = MultiTypeEQLoss(
            n_fft=data_cfg.get("n_fft", 2048),
            sample_rate=data_cfg.get("sample_rate", 44100),
            n_mels=data_cfg.get("n_mels", 128),
            lambda_param=loss_cfg.get("lambda_param", 1.0),
            lambda_gain=loss_cfg.get("lambda_gain", 1.0),
            lambda_freq=loss_cfg.get("lambda_freq", 1.0),
            lambda_q=loss_cfg.get("lambda_q", 0.5),
            lambda_type=loss_cfg.get("lambda_type", 0.5),
            lambda_spectral=loss_cfg.get("lambda_spectral", 1.0),
            lambda_hmag=loss_cfg.get("lambda_hmag", 0.3),
            lambda_typed_hmag=loss_cfg.get("lambda_typed_hmag", 0.0),
            lambda_activity=loss_cfg.get("lambda_activity", 0.1),
            lambda_spread=loss_cfg.get("lambda_spread", 0.05),
            lambda_type_match=loss_cfg.get("lambda_type_match", 0.5),
            lambda_perceptual=loss_cfg.get("lambda_perceptual", 0.0),
            lambda_group_delay=loss_cfg.get("lambda_group_delay", 0.0),
            lambda_phase=loss_cfg.get("lambda_phase", 0.0),
            lambda_type_diversity=loss_cfg.get("lambda_type_diversity", 0.0),
            class_weights=loss_cfg.get("class_weights"),
            type_prior=loss_cfg.get("type_prior", data_cfg.get("type_weights")),
            use_focal_loss=loss_cfg.get("use_focal_loss", False),
            focal_gamma=loss_cfg.get("focal_gamma", 2.0),
        )

    def forward(self, mel_frames):
        return self.model(mel_frames)

    def _common_step(self, batch, batch_idx, mode="train"):
        wet_audio = batch["wet_audio"]
        dry_audio = batch["dry_audio"]
        target_gain = batch["gain"]
        target_freq = batch["freq"]
        target_q = batch["q"]
        target_ft = batch["filter_type"]

        # Convert wet audio to mel-spectrogram
        mel_spec = self.frontend.mel_spectrogram(wet_audio)  # (B, 1, n_mels, T)
        mel_frames = mel_spec.squeeze(1)  # (B, n_mels, T)

        # Forward pass through TCN model
        output = self.model(mel_frames)

        pred_gain, pred_freq, pred_q = output["params"]
        pred_type_logits = output["type_logits"]
        pred_H_mag = output["H_mag"]

        # Ground truth frequency response
        target_H_mag = self.model.dsp_cascade(
            target_gain, target_freq, target_q,
            n_fft=self.config["data"].get("n_fft", 2048),
            filter_type=target_ft
        )
        _, target_gd, target_phase = self.model.dsp_cascade.forward_full(
            target_gain, target_freq, target_q,
            n_fft=self.config["data"].get("n_fft", 2048),
            filter_type=target_ft
        )

        # Synthesize pred_audio using the predicted parameters
        pred_audio = self.model.dsp_cascade.process_audio(
            dry_audio, pred_gain, pred_freq, pred_q, filter_type=pred_type_logits.argmax(dim=-1)
        )
        # Ensure audio length matches for STFT
        target_audio = wet_audio

        # Compute loss
        total_loss, components = self.criterion(
            pred_gain, pred_freq, pred_q,
            pred_type_logits, pred_H_mag,
            target_gain, target_freq, target_q,
            target_ft, target_H_mag,
            pred_audio=pred_audio, target_audio=target_audio,
            pred_H_mag_hard=output.get("H_mag_hard"),
            pred_group_delay=output.get("group_delay"),
            target_group_delay=target_gd,
            pred_phase=output.get("phase"),
            target_phase=target_phase,
            pred_type_probs=output.get("type_probs"),
        )

        # Logging
        self.log(f"{mode}/total_loss", total_loss, on_step=(mode == "train"),
                 on_epoch=True, prog_bar=True, sync_dist=True)
        for k, v in components.items():
            self.log(f"{mode}/{k}", v, on_step=(mode == "train"),
                     on_epoch=True, sync_dist=True)

        # Log parameter accuracy metrics
        with torch.no_grad():
            gain_mae = (pred_gain - target_gain).abs().mean()
            freq_oct = (torch.log2(pred_freq / (target_freq + 1e-8))).abs().mean()
            q_dec = (torch.log10(pred_q / (target_q + 1e-8))).abs().mean()
            type_acc = (output["filter_type"] == target_ft).float().mean()

            self.log(f"{mode}/gain_mae_db", gain_mae, on_epoch=True)
            self.log(f"{mode}/freq_mae_oct", freq_oct, on_epoch=True)
            self.log(f"{mode}/q_mae_dec", q_dec, on_epoch=True)
            self.log(f"{mode}/type_acc", type_acc, on_epoch=True)

        return total_loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, mode="train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, mode="val")

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, mode="test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config["model"]["learning_rate"],
            weight_decay=self.config["model"]["weight_decay"]
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config["trainer"]["max_epochs"]
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }
