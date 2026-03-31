import torch
import torch.nn as nn
from differentiable_eq import EQParameterHead, DifferentiableBiquadCascade
from dsp_frontend import STFTFrontend, apply_eq_to_complex_stft


class EQEstimatorCNN(nn.Module):
    """
    Blind EQ parameter estimator.
    Takes raw wet audio, extracts mel-spectrogram for CNN encoder,
    estimates (gain, freq, q) per band. Produces both the forward
    magnitude response and the inverse-filtered dry estimate.
    """

    def __init__(self, num_bands=5, sample_rate=22050, n_fft=2048, mel_bins=128):
        super().__init__()
        self.num_bands = num_bands
        self.n_fft = n_fft
        self.sample_rate = sample_rate

        self.frontend = STFTFrontend(
            n_fft=n_fft,
            hop_length=n_fft // 4,
            win_length=n_fft,
            mel_bins=mel_bins,
            sample_rate=sample_rate,
        )

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.param_head = EQParameterHead(embedding_dim=128, num_bands=num_bands)
        self.dsp_cascade = DifferentiableBiquadCascade(
            num_bands=num_bands, sample_rate=sample_rate
        )

    def forward(self, wet_audio):
        """
        Args:
            wet_audio: (Batch, Time) raw audio waveform

        Returns dict with:
            params: tuple of (gain_db, freq, q) each (Batch, NumBands)
            H_mag: forward magnitude response (Batch, FreqBins)
            dry_mag_est: inverse-filtered magnitude (Batch, FreqBins, TimeFrames)
            wet_stft: original complex STFT (Batch, FreqBins, TimeFrames)
            mel_spec: mel spectrogram fed to CNN (Batch, 1, MelBins, TimeFrames)
            audio_length: int, original audio length
        """
        mel_spec, wet_stft, audio_length = self.frontend(wet_audio)

        x = self.features(mel_spec)
        x = self.pool(x)
        embed = x.view(x.size(0), -1)

        gain_db, freq, q = self.param_head(embed)

        H_mag = self.dsp_cascade(gain_db, freq, q, n_fft=self.n_fft)

        wet_mag = torch.abs(wet_stft)
        dry_mag_est = self.dsp_cascade.apply_inverse_to_spectrum(
            wet_mag, gain_db, freq, q
        )

        return {
            "params": (gain_db, freq, q),
            "H_mag": H_mag,
            "dry_mag_est": dry_mag_est,
            "wet_stft": wet_stft,
            "mel_spec": mel_spec,
            "audio_length": audio_length,
        }

    def estimate_params(self, wet_audio):
        out = self.forward(wet_audio)
        return out["params"]

    def reconstruct_roundtrip(self, wet_audio):
        out = self.forward(wet_audio)
        gain_db, freq, q = out["params"]
        dry_mag_est = out["dry_mag_est"]

        wet_mag = torch.abs(out["wet_stft"])
        roundtrip_mag = self.dsp_cascade.apply_to_spectrum(
            dry_mag_est, gain_db, freq, q
        )

        return out, roundtrip_mag
