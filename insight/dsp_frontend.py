import torch
import torch.nn as nn
import torch.nn.functional as F


class STFTFrontend(nn.Module):
    """
    Differentiable STFT/iSTFT frontend.
    Extracts mel-spectrograms for the CNN encoder and provides
    complex STFT for the differentiable EQ filtering path.
    """

    def __init__(
        self,
        n_fft=2048,
        hop_length=512,
        win_length=2048,
        mel_bins=128,
        sample_rate=22050,
        f_min=20.0,
        f_max=None,
        causal=False,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.mel_bins = mel_bins
        self.sample_rate = sample_rate
        self.f_min = f_min
        self.f_max = f_max if f_max is not None else sample_rate // 2
        self.causal = causal
        self.register_buffer("window", torch.hann_window(win_length))
        self.register_buffer("mel_fb", self._build_mel_filterbank())

    def _build_mel_filterbank(self):
        n_freqs = self.n_fft // 2 + 1
        f_min_mel = self._hz_to_mel(torch.tensor(self.f_min))
        f_max_mel = self._hz_to_mel(torch.tensor(self.f_max))
        mel_points = torch.linspace(f_min_mel, f_max_mel, self.mel_bins + 2)
        hz_points = self._mel_to_hz(mel_points)
        bin_points = ((self.n_fft + 1) * hz_points / self.sample_rate).long()
        fb = torch.zeros(self.mel_bins, n_freqs)
        for i in range(self.mel_bins):
            f_left = bin_points[i]
            f_center = bin_points[i + 1]
            f_right = bin_points[i + 2]
            for j in range(f_left, f_center):
                if j < n_freqs and f_center > f_left:
                    fb[i, j] = (j - f_left) / (f_center - f_left)
            for j in range(f_center, f_right):
                if j < n_freqs and f_right > f_center:
                    fb[i, j] = (f_right - j) / (f_right - f_center)
        return fb

    @staticmethod
    def _hz_to_mel(hz):
        return 2595.0 * torch.log10(1.0 + hz / 700.0)

    @staticmethod
    def _mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def stft(self, audio):
        """
        audio: (Batch, Time)
        Returns complex STFT: (Batch, FreqBins, TimeFrames)
        """
        if self.causal:
            # Left-only padding for causal operation
            pad_amount = self.win_length - self.hop_length
            audio = F.pad(audio, (pad_amount, 0))
            return torch.stft(
                audio,
                self.n_fft,
                self.hop_length,
                self.win_length,
                window=self.window,
                return_complex=True,
                center=False,
            )
        return torch.stft(
            audio,
            self.n_fft,
            self.hop_length,
            self.win_length,
            window=self.window,
            return_complex=True,
            center=True,
        )

    def istft(self, complex_stft, length=None):
        """
        complex_stft: (Batch, FreqBins, TimeFrames)
        Returns audio: (Batch, Time)
        """
        return torch.istft(
            complex_stft,
            self.n_fft,
            self.hop_length,
            self.win_length,
            window=self.window,
            length=length,
            center=not self.causal,
        )

    def mel_spectrogram(self, audio):
        """
        audio: (Batch, Time)
        Returns log-mel spectrogram: (Batch, 1, MelBins, TimeFrames)
        """
        complex_spec = self.stft(audio)
        mag = torch.abs(complex_spec)
        mel_spec = torch.matmul(self.mel_fb, mag)
        mel_spec = torch.clamp(mel_spec, min=1e-8)
        log_mel = torch.log(mel_spec)
        return log_mel.unsqueeze(1)

    def get_magnitude(self, audio):
        """
        audio: (Batch, Time)
        Returns magnitude STFT: (Batch, FreqBins, TimeFrames)
        """
        return torch.abs(self.stft(audio))

    def get_complex(self, audio):
        """
        audio: (Batch, Time)
        Returns complex STFT: (Batch, FreqBins, TimeFrames)
        """
        return self.stft(audio)

    def forward(self, audio):
        """
        Full frontend: returns both mel-spectrogram for CNN
        and complex STFT for differentiable filtering.
        audio: (Batch, Time)
        Returns:
            mel_spec: (Batch, 1, MelBins, TimeFrames)
            complex_stft: (Batch, FreqBins, TimeFrames)
            audio_length: int
        """
        mel_spec = self.mel_spectrogram(audio)
        complex_stft = self.stft(audio)
        return mel_spec, complex_stft, audio.shape[-1]


def apply_eq_to_complex_stft(complex_stft, H_mag):
    """
    Apply an EQ magnitude response to a complex STFT.
    Only modifies magnitudes; preserves phase.
    complex_stft: (Batch, FreqBins, TimeFrames) complex
    H_mag: (Batch, FreqBins) or (Batch, FreqBins, 1)
    Returns: modified complex STFT (same shape)
    """
    original_mag = torch.abs(complex_stft)
    phase = torch.angle(complex_stft)
    if H_mag.dim() == 2:
        H_mag = H_mag.unsqueeze(-1)
    new_mag = original_mag * H_mag
    return new_mag * torch.exp(1j * phase)
