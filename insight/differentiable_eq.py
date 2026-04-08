import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class StraightThroughClamp(torch.autograd.Function):
    """
    Straight-through estimator clamp: forward uses torch.clamp (hard bounds),
    backward passes gradients through unmodified (no tanh attenuation).

    Forces float32 internally to avoid bf16 gradient corruption under AMP.
    Custom autograd functions bypass PyTorch's AMP gradient coordination,
    so bf16 forward outputs can produce mismatched/zero gradients.

    The output dtype matches the input dtype (cast back after internal fp32).
    """

    @staticmethod
    def forward(ctx, x, min_val, max_val):
        ctx.input_dtype = x.dtype
        x_fp32 = x.float()
        return torch.clamp(x_fp32, min_val, max_val).to(ctx.input_dtype)

    @staticmethod
    def backward(ctx, grad_output):
        # Ensure gradient is fp32 for numerical stability, cast back to match input
        grad = grad_output.float()
        return grad.to(ctx.input_dtype), None, None


def ste_clamp(x, min_val, max_val):
    return StraightThroughClamp.apply(x, min_val, max_val)


# Filter type constants
FILTER_PEAKING = 0
FILTER_LOWSHELF = 1
FILTER_HIGHSHELF = 2
FILTER_HIGHPASS = 3
FILTER_LOWPASS = 4
NUM_FILTER_TYPES = 5

FILTER_NAMES = ["peaking", "lowshelf", "highshelf", "highpass", "lowpass"]
DEFAULT_TYPE_PRIORS = [0.5, 0.15, 0.15, 0.1, 0.1]
FILTER_GAINFUL_MASK = [1.0, 1.0, 1.0, 0.0, 0.0]
FILTER_LOG_FREQ_BOUNDS = [
    (math.log(20.0), math.log(20000.0)),
    (math.log(20.0), math.log(5000.0)),
    (math.log(1000.0), math.log(20000.0)),
    (math.log(20.0), math.log(500.0)),
    (math.log(2000.0), math.log(20000.0)),
]


class DifferentiableBiquadCascade(nn.Module):
    """
    A differentiable biquad filter bank supporting multiple filter types.
    Translates [Gain, Freq, Q, FilterType] predictions into biquad coefficients
    and computes the frequency response, enabling exact gradient flow in PyTorch.

    Coefficient formulas from the Robert Bristow-Johnson Audio EQ Cookbook.
    """

    def __init__(self, num_bands=5, sample_rate=44100):
        super().__init__()
        self.num_bands = num_bands
        self.sample_rate = sample_rate

    def compute_biquad_coeffs(self, gain_db, freq, q):
        """
        Computes biquad coefficients for Peaking/Bell EQ filters.
        Inputs: tensors of shape (Batch, Num_Bands).
        Returns: b0, b1, b2, a1, a2 (each shape (Batch, Num_Bands))
        """
        A = 10.0 ** (gain_db / 40.0)
        w0 = 2.0 * torch.pi * freq / self.sample_rate
        alpha = torch.sin(w0) / (2.0 * q + 1e-8)

        b0 = 1.0 + alpha * A
        b1 = -2.0 * torch.cos(w0)
        b2 = 1.0 - alpha * A
        a0 = 1.0 + alpha / A
        a1 = -2.0 * torch.cos(w0)
        a2 = 1.0 - alpha / A

        b0 = b0 / a0
        b1 = b1 / a0
        b2 = b2 / a0
        a1 = a1 / a0
        a2 = a2 / a0

        return b0, b1, b2, a1, a2

    def compute_biquad_coeffs_multitype(self, gain_db, freq, q, filter_type):
        """
        Computes biquad coefficients for multiple filter types using the
        Robert Bristow-Johnson Audio EQ Cookbook formulas.

        Inputs:
            gain_db: (Batch, Num_Bands) — gain in dB (ignored for HP/LP)
            freq: (Batch, Num_Bands) — center/cutoff frequency in Hz
            q: (Batch, Num_Bands) — quality factor
            filter_type: (Batch, Num_Bands) — integer tensor in {0,1,2,3,4}

        Returns: b0, b1, b2, a1, a2 (each shape (Batch, Num_Bands))
        """
        A = 10.0 ** (gain_db / 40.0)
        w0 = 2.0 * torch.pi * freq / self.sample_rate
        cos_w0 = torch.cos(w0)
        sin_w0 = torch.sin(w0)
        alpha = sin_w0 / (2.0 * q + 1e-8)
        sqrt_A = torch.sqrt(torch.clamp(A, min=1e-8))
        two_sqrt_A_alpha = 2.0 * sqrt_A * alpha

        # --- Type 0: Peaking/Bell ---
        peak_b0 = 1.0 + alpha * A
        peak_b1 = -2.0 * cos_w0
        peak_b2 = 1.0 - alpha * A
        peak_a0 = 1.0 + alpha / A
        peak_a1 = -2.0 * cos_w0
        peak_a2 = 1.0 - alpha / A

        # --- Type 1: Low-Shelf ---
        ls_b0 = A * ((A + 1.0) - (A - 1.0) * cos_w0 + two_sqrt_A_alpha)
        ls_b1 = 2.0 * A * ((A - 1.0) - (A + 1.0) * cos_w0)
        ls_b2 = A * ((A + 1.0) - (A - 1.0) * cos_w0 - two_sqrt_A_alpha)
        ls_a0 = (A + 1.0) + (A - 1.0) * cos_w0 + two_sqrt_A_alpha
        ls_a1 = -2.0 * ((A - 1.0) + (A + 1.0) * cos_w0)
        ls_a2 = (A + 1.0) + (A - 1.0) * cos_w0 - two_sqrt_A_alpha

        # --- Type 2: High-Shelf ---
        hs_b0 = A * ((A + 1.0) + (A - 1.0) * cos_w0 + two_sqrt_A_alpha)
        hs_b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * cos_w0)
        hs_b2 = A * ((A + 1.0) + (A - 1.0) * cos_w0 - two_sqrt_A_alpha)
        hs_a0 = (A + 1.0) - (A - 1.0) * cos_w0 + two_sqrt_A_alpha
        hs_a1 = 2.0 * ((A - 1.0) - (A + 1.0) * cos_w0)
        hs_a2 = (A + 1.0) - (A - 1.0) * cos_w0 - two_sqrt_A_alpha

        # --- Type 3: High-Pass (gain_db ignored) ---
        hp_b0 = (1.0 + cos_w0) / 2.0
        hp_b1 = -(1.0 + cos_w0)
        hp_b2 = (1.0 + cos_w0) / 2.0
        hp_a0 = 1.0 + alpha
        hp_a1 = -2.0 * cos_w0
        hp_a2 = 1.0 - alpha

        # --- Type 4: Low-Pass (gain_db ignored) ---
        lp_b0 = (1.0 - cos_w0) / 2.0
        lp_b1 = 1.0 - cos_w0
        lp_b2 = (1.0 - cos_w0) / 2.0
        lp_a0 = 1.0 + alpha
        lp_a1 = -2.0 * cos_w0
        lp_a2 = 1.0 - alpha

        # Select coefficients based on filter type using torch.where
        ft = filter_type
        b0 = torch.where(
            ft == FILTER_PEAKING,
            peak_b0,
            torch.where(
                ft == FILTER_LOWSHELF,
                ls_b0,
                torch.where(
                    ft == FILTER_HIGHSHELF,
                    hs_b0,
                    torch.where(ft == FILTER_HIGHPASS, hp_b0, lp_b0),
                ),
            ),
        )
        b1 = torch.where(
            ft == FILTER_PEAKING,
            peak_b1,
            torch.where(
                ft == FILTER_LOWSHELF,
                ls_b1,
                torch.where(
                    ft == FILTER_HIGHSHELF,
                    hs_b1,
                    torch.where(ft == FILTER_HIGHPASS, hp_b1, lp_b1),
                ),
            ),
        )
        b2 = torch.where(
            ft == FILTER_PEAKING,
            peak_b2,
            torch.where(
                ft == FILTER_LOWSHELF,
                ls_b2,
                torch.where(
                    ft == FILTER_HIGHSHELF,
                    hs_b2,
                    torch.where(ft == FILTER_HIGHPASS, hp_b2, lp_b2),
                ),
            ),
        )
        a0 = torch.where(
            ft == FILTER_PEAKING,
            peak_a0,
            torch.where(
                ft == FILTER_LOWSHELF,
                ls_a0,
                torch.where(
                    ft == FILTER_HIGHSHELF,
                    hs_a0,
                    torch.where(ft == FILTER_HIGHPASS, hp_a0, lp_a0),
                ),
            ),
        )
        a1 = torch.where(
            ft == FILTER_PEAKING,
            peak_a1,
            torch.where(
                ft == FILTER_LOWSHELF,
                ls_a1,
                torch.where(
                    ft == FILTER_HIGHSHELF,
                    hs_a1,
                    torch.where(ft == FILTER_HIGHPASS, hp_a1, lp_a1),
                ),
            ),
        )
        a2 = torch.where(
            ft == FILTER_PEAKING,
            peak_a2,
            torch.where(
                ft == FILTER_LOWSHELF,
                ls_a2,
                torch.where(
                    ft == FILTER_HIGHSHELF,
                    hs_a2,
                    torch.where(ft == FILTER_HIGHPASS, hp_a2, lp_a2),
                ),
            ),
        )

        # Normalize by a0 (clamp to prevent near-zero division)
        a0 = torch.clamp(a0, min=1e-4)
        b0 = b0 / a0
        b1 = b1 / a0
        b2 = b2 / a0
        a1 = a1 / a0
        a2 = a2 / a0

        return b0, b1, b2, a1, a2

    def freq_response(self, b0, b1, b2, a1, a2, n_fft=2048):
        """
        Compute the frequency response of the biquad filter.
        (Batch, Num_Bands) -> (Batch, Num_Bands, N_FFT // 2 + 1)
        """
        w = torch.linspace(0, torch.pi, n_fft // 2 + 1, device=b0.device)

        cos_w = torch.cos(w).view(1, 1, -1)
        sin_w = torch.sin(w).view(1, 1, -1)
        cos_2w = torch.cos(2 * w).view(1, 1, -1)
        sin_2w = torch.sin(2 * w).view(1, 1, -1)

        b0_v = b0.unsqueeze(-1)
        b1_v = b1.unsqueeze(-1)
        b2_v = b2.unsqueeze(-1)
        a1_v = a1.unsqueeze(-1)
        a2_v = a2.unsqueeze(-1)

        num_re = b0_v + b1_v * cos_w + b2_v * cos_2w
        num_im = -(b1_v * sin_w + b2_v * sin_2w)

        den_re = 1.0 + a1_v * cos_w + a2_v * cos_2w
        den_im = -(a1_v * sin_w + a2_v * sin_2w)

        num_mag2 = num_re**2 + num_im**2
        den_mag2 = den_re**2 + den_im**2

        H_mag = torch.sqrt(torch.clamp(num_mag2 / (den_mag2 + 1e-6), min=1e-10))

        return H_mag

    def forward(self, gain_db, freq, q, n_fft=2048, filter_type=None):
        """
        Forward pass calculates the overall magnitude response of the cascade.
        If filter_type is None, uses peaking/bell for all bands (backward compat).

        Args:
            gain_db: (Batch, Num_Bands)
            freq: (Batch, Num_Bands)
            q: (Batch, Num_Bands)
            n_fft: FFT size
            filter_type: (Batch, Num_Bands) int tensor, or None for peaking-only

        Returns: (Batch, N_FFT // 2 + 1)
        """
        if filter_type is None:
            b0, b1, b2, a1, a2 = self.compute_biquad_coeffs(gain_db, freq, q)
        else:
            b0, b1, b2, a1, a2 = self.compute_biquad_coeffs_multitype(
                gain_db, freq, q, filter_type
            )
        H_mag_bands = self.freq_response(b0, b1, b2, a1, a2, n_fft)
        H_mag_total = torch.prod(H_mag_bands, dim=1)
        # Clamp to prevent inf/NaN from product of 5 bands
        H_mag_total = torch.clamp(H_mag_total, min=1e-6, max=1e4)
        return H_mag_total

    def process_audio(
        self, audio, gain_db, freq, q, filter_type=None, n_fft=2048, hop_length=512
    ):
        """
        Apply the differentiable EQ cascade to a time-domain audio signal.
        Because time-domain recursive biquad filtering limits parallel gradient flow,
        we apply the complex frequency response via STFT multiplication and iSTFT.

        Args:
            audio: (Batch, Time)
            gain_db, freq, q: (Batch, Num_Bands)
            filter_type: (Batch, Num_Bands)
        Returns:
            filtered_audio: (Batch, Time)
        """
        window = torch.hann_window(n_fft, device=audio.device)
        # 1. Take STFT
        # (Batch, FreqBins, TimeFrames)
        X = torch.stft(
            audio,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            return_complex=True,
            pad_mode="constant",
        )

        # 2. Get magnitude response of the EQ curve
        H_mag = self.forward(gain_db, freq, q, n_fft=n_fft, filter_type=filter_type)

        # 3. Multiply standard complex spectrum by the magnitude response
        # H_mag shape: (Batch, FreqBins). Broadcast over time frames.
        Y = X * H_mag.unsqueeze(-1)

        # 4. Inverse STFT to get the filtered time-domain signal
        y_time = torch.istft(
            Y, n_fft=n_fft, hop_length=hop_length, window=window, length=audio.shape[-1]
        )
        return y_time

    def inverse_freq_response(
        self, gain_db, freq, q, n_fft=2048, max_gain=4.0, filter_type=None
    ):
        """
        Compute the inverse magnitude response of the cascade (1/H).
        Clamps per-band inverse to avoid instability at resonance peaks.
        """
        H_mag_total = self.forward(gain_db, freq, q, n_fft, filter_type)
        H_inv_bands = 1.0 / (H_mag_total + 1e-8)
        H_inv = torch.clamp(H_inv_bands, max=max_gain)
        return H_inv

    def apply_to_spectrum(self, stft_mag, gain_db, freq, q, filter_type=None):
        """
        Apply the EQ cascade to an existing STFT magnitude tensor.
        stft_mag: (Batch, FreqBins) or (Batch, FreqBins, Time)
        """
        n_fft = (stft_mag.shape[-2 if stft_mag.dim() == 3 else 1] - 1) * 2
        H_mag = self.forward(gain_db, freq, q, n_fft=n_fft, filter_type=filter_type)
        if stft_mag.dim() == 3:
            H_mag = H_mag.unsqueeze(-1)
        return stft_mag * H_mag

    def apply_inverse_to_spectrum(
        self, stft_mag, gain_db, freq, q, max_gain=4.0, filter_type=None
    ):
        """
        Apply the inverse EQ cascade to an existing STFT magnitude tensor.
        """
        n_fft = (stft_mag.shape[-2 if stft_mag.dim() == 3 else 1] - 1) * 2
        H_inv = self.inverse_freq_response(
            gain_db, freq, q, n_fft=n_fft, max_gain=max_gain, filter_type=filter_type
        )
        if stft_mag.dim() == 3:
            H_inv = H_inv.unsqueeze(-1)
        return stft_mag * H_inv


class EQParameterHead(nn.Module):
    """
    Maps raw network embeddings to constrained [Gain, Freq, Q] parameters.
    Peaking/bell only — kept for backward compatibility.
    """

    def __init__(self, embedding_dim, num_bands=5):
        super().__init__()
        self.num_bands = num_bands
        self.fc = nn.Linear(embedding_dim, num_bands * 3)

    def forward(self, embedding):
        raw = self.fc(embedding)
        raw = raw.view(-1, self.num_bands, 3)

        gain_db = ste_clamp(raw[:, :, 0] * 24.0, -24.0, 24.0)
        freq = torch.sigmoid(raw[:, :, 1]) * (20000.0 - 20.0) + 20.0
        q = torch.sigmoid(raw[:, :, 2]) * (10.0 - 0.1) + 0.1

        return gain_db, freq, q


class MultiTypeEQParameterHead(nn.Module):
    """
    Parameter head for multi-type EQ estimation.

    Key behaviors:
    - Gain is type-aware: HP/LP gains are suppressed via type probabilities.
    - Frequency is mapped through type-specific log-frequency bounds.
    - Type prediction uses both global and per-band mel evidence.
    """

    def __init__(
        self,
        embedding_dim,
        num_bands=5,
        num_filter_types=NUM_FILTER_TYPES,
        n_mels=0,
        type_conditioned_frequency=True,
        n_shelf_bands=16,
    ):
        super().__init__()
        self.num_bands = num_bands
        self.num_filter_types = num_filter_types
        self.n_mels = n_mels
        self.type_conditioned_frequency = type_conditioned_frequency
        self.n_shelf_bands = n_shelf_bands
        self.log_f_min = FILTER_LOG_FREQ_BOUNDS[FILTER_PEAKING][0]
        self.log_f_max = FILTER_LOG_FREQ_BOUNDS[FILTER_PEAKING][1]
        hidden_dim = 64
        mel_hidden_dim = 64

        # Shared trunk: embedding -> per-band features
        self.trunk = nn.Sequential(
            nn.Linear(embedding_dim, num_bands * hidden_dim),
            nn.ReLU(),
        )

        # Gain head: single linear + ste_clamp
        self.gain_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.gain_head.bias)
        nn.init.xavier_uniform_(self.gain_head.weight, gain=0.1)

        self.freq_context_proj = (
            nn.Sequential(
                nn.Linear(hidden_dim + mel_hidden_dim, hidden_dim),
                nn.GELU(),
            )
            if n_mels > 0
            else None
        )
        self.freq_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.freq_head.bias)
        nn.init.xavier_uniform_(self.freq_head.weight, gain=0.1)

        # Q head: single linear + log-space ste_clamp → exp
        self.q_head = nn.Linear(hidden_dim, 1)

        # Type path: centered spectral-shape evidence plus the learned trunk.
        self.type_mel_proj = (
            nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=15, stride=4, padding=7),
                nn.GELU(),
                nn.Conv1d(32, mel_hidden_dim, kernel_size=7, stride=2, padding=3),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(mel_hidden_dim, mel_hidden_dim),
                nn.GELU(),
            )
            if n_mels > 0
            else None
        )
        self.band_mel_encoder = (
            nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=15, padding=7),
                nn.GELU(),
                nn.Conv1d(32, mel_hidden_dim, kernel_size=7, padding=3),
                nn.GELU(),
            )
            if n_mels > 0
            else None
        )
        self.band_mel_query = (
            nn.Linear(hidden_dim, mel_hidden_dim) if n_mels > 0 else None
        )
        self.band_mel_value = (
            nn.Conv1d(mel_hidden_dim, mel_hidden_dim, kernel_size=1)
            if n_mels > 0
            else None
        )
        self.band_mel_norm = (
            nn.LayerNorm(mel_hidden_dim) if n_mels > 0 else None
        )
        self.type_fusion_proj = (
            nn.Sequential(
                nn.Linear(hidden_dim + mel_hidden_dim, hidden_dim),
                nn.GELU(),
            )
            if n_mels > 0
            else None
        )

        # Gain path: separate auxiliary spectral branch so gain gradients do not
        # need to repurpose the type branch.
        self.gain_mel_proj = (
            nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=15, stride=4, padding=7),
                nn.GELU(),
                nn.Conv1d(32, mel_hidden_dim, kernel_size=7, stride=2, padding=3),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(mel_hidden_dim, mel_hidden_dim),
                nn.GELU(),
            )
            if n_mels > 0
            else None
        )
        self.gain_band_mel_encoder = (
            nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=15, padding=7),
                nn.GELU(),
                nn.Conv1d(32, mel_hidden_dim, kernel_size=7, padding=3),
                nn.GELU(),
            )
            if n_mels > 0
            else None
        )
        self.gain_band_mel_query = (
            nn.Linear(hidden_dim, mel_hidden_dim) if n_mels > 0 else None
        )
        self.gain_band_mel_value = (
            nn.Conv1d(mel_hidden_dim, mel_hidden_dim, kernel_size=1)
            if n_mels > 0
            else None
        )
        self.gain_band_mel_norm = (
            nn.LayerNorm(mel_hidden_dim) if n_mels > 0 else None
        )
        self.gain_context_proj = (
            nn.Sequential(
                nn.Linear(hidden_dim + mel_hidden_dim, hidden_dim),
                nn.GELU(),
            )
            if n_mels > 0
            else None
        )

        type_input_dim = hidden_dim + 3
        self.type_head = nn.Sequential(
            nn.Linear(type_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_filter_types),
        )
        with torch.no_grad():
            self.type_head[-1].bias.copy_(
                torch.log(torch.tensor(DEFAULT_TYPE_PRIORS[:num_filter_types]))
            )
        self.register_buffer(
            "gainful_type_mask",
            torch.tensor(FILTER_GAINFUL_MASK, dtype=torch.float32),
        )
        self.register_buffer(
            "type_log_f_min",
            torch.tensor(
                [bounds[0] for bounds in FILTER_LOG_FREQ_BOUNDS], dtype=torch.float32
            ),
        )
        self.register_buffer(
            "type_log_f_max",
            torch.tensor(
                [bounds[1] for bounds in FILTER_LOG_FREQ_BOUNDS], dtype=torch.float32
            ),
        )

    def _center_mel_profile(self, mel_profile):
        if mel_profile is None:
            return None
        return mel_profile - mel_profile.mean(dim=-1, keepdim=True)

    def _build_band_context(
        self,
        trunk_out,
        mel_profile,
        global_proj,
        local_encoder,
        band_query,
        band_value,
        band_norm,
    ):
        if global_proj is None or mel_profile is None:
            return None

        global_mel_feat = global_proj(mel_profile.unsqueeze(1))

        if local_encoder is None:
            return global_mel_feat.unsqueeze(1).expand(-1, self.num_bands, -1)

        local_mel = local_encoder(mel_profile.unsqueeze(1))
        local_values = band_value(local_mel).transpose(1, 2)
        band_query = band_query(trunk_out)
        attn_scores = torch.einsum("bnc,bmc->bnm", band_query, local_values)
        attn_scores = attn_scores / math.sqrt(local_values.shape[-1])
        attn = torch.softmax(attn_scores, dim=-1)
        local_mel_feat = torch.einsum("bnm,bmc->bnc", attn, local_values)
        local_mel_feat = band_norm(local_mel_feat)
        return local_mel_feat + global_mel_feat.unsqueeze(1)

    def _build_type_mel_context(self, trunk_out, mel_profile):
        return self._build_band_context(
            trunk_out,
            mel_profile,
            self.type_mel_proj,
            self.band_mel_encoder,
            self.band_mel_query,
            self.band_mel_value,
            self.band_mel_norm,
        )

    def _build_gain_aux(self, trunk_out, mel_profile):
        return self._build_band_context(
            trunk_out,
            mel_profile,
            self.gain_mel_proj,
            self.gain_band_mel_encoder,
            self.gain_band_mel_query,
            self.gain_band_mel_value,
            self.gain_band_mel_norm,
        )

    def _compute_shelf_features(self, mel_profile: torch.Tensor) -> torch.Tensor:
        """
        Compute fixed (non-learned) scalar features that discriminate shelf
        filter types from peaking filters.

        A lowshelf boosts/cuts all frequencies below fc monotonically — the
        low-band mean is systematically higher/lower than the high-band mean.
        A peaking filter creates a localized bump with similar energy on both
        sides. These three statistics make that distinction directly observable
        without relying on learned feature extraction.

        Args:
            mel_profile: (B, n_mels) — raw (not centered) mel energy profile

        Returns:
            shelf_feats: (B, 3) — [lo_ratio, hi_ratio, tilt_slope]
                lo_ratio: log(mean(mel[0:N]) / mean(mel[N:])) — bass tilt
                hi_ratio: log(mean(mel[-N:]) / mean(mel[:-N])) — treble tilt
                tilt_slope: linear regression slope across mel bands (normalized)
        """
        B, n_mels = mel_profile.shape
        n = min(self.n_shelf_bands, n_mels // 4)  # guard: never exceed 25% of bands

        # Use absolute value so negative mel values (e.g. log-mel can be negative)
        # do not produce NaN in the log ratio. Energy magnitude is what matters.
        mel_abs = mel_profile.abs()

        # Low-frequency energy ratio: log(E_low / E_high)
        lo_mean = mel_abs[:, :n].mean(dim=-1, keepdim=True)           # (B, 1)
        hi_mean = mel_abs[:, n:].mean(dim=-1, keepdim=True)           # (B, 1)
        lo_ratio = torch.log((lo_mean + 1e-6) / (hi_mean + 1e-6))     # (B, 1)

        # High-frequency energy ratio: log(E_high_tail / E_rest)
        hi_tail = mel_abs[:, -n:].mean(dim=-1, keepdim=True)          # (B, 1)
        rest = mel_abs[:, :-n].mean(dim=-1, keepdim=True)             # (B, 1)
        hi_ratio = torch.log((hi_tail + 1e-6) / (rest + 1e-6))        # (B, 1)

        # Spectral tilt slope: linear regression coefficient across mel bands
        # x = normalized band indices in [-1, 1]; y = mel_profile
        x = torch.linspace(-1.0, 1.0, n_mels, device=mel_profile.device,
                           dtype=mel_profile.dtype).unsqueeze(0)       # (1, n_mels)
        x_centered = x - x.mean()
        denom = (x_centered ** 2).sum()
        y_centered = mel_profile - mel_profile.mean(dim=-1, keepdim=True)
        slope = (x_centered * y_centered).sum(dim=-1, keepdim=True) / (denom + 1e-8)  # (B, 1)

        # Clamp all features to reasonable range to avoid exploding values
        lo_ratio = torch.clamp(lo_ratio, -5.0, 5.0)
        hi_ratio = torch.clamp(hi_ratio, -5.0, 5.0)
        slope = torch.clamp(slope, -5.0, 5.0)

        return torch.cat([lo_ratio, hi_ratio, slope], dim=-1)  # (B, 3)

    def summarize_gain_aux_features(self, mel_profile):
        if self.gain_mel_proj is None or mel_profile is None:
            return None
        centered = self._center_mel_profile(mel_profile)
        return self.gain_mel_proj(centered.unsqueeze(1))

    def forward(
        self,
        embedding,
        mel_profile=None,
        hard_types=False,
        return_aux=False,
    ):
        trunk_out = self.trunk(embedding)  # (B, num_bands * 64)
        trunk_out = trunk_out.view(-1, self.num_bands, 64)
        mel_profile_centered = self._center_mel_profile(mel_profile)

        type_mel_context = self._build_type_mel_context(trunk_out, mel_profile_centered)
        gain_aux = self._build_gain_aux(trunk_out, mel_profile_centered)

        if type_mel_context is not None and self.type_fusion_proj is not None:
            type_input = self.type_fusion_proj(
                torch.cat([trunk_out, type_mel_context], dim=-1)
            )
        else:
            type_input = trunk_out
        if mel_profile is not None and self.n_shelf_bands > 0:
            shelf_feats = self._compute_shelf_features(mel_profile)  # (B, 3)
            # Expand to per-band: (B, 3) -> (B, num_bands, 3)
            shelf_feats_expanded = shelf_feats.unsqueeze(1).expand(-1, self.num_bands, -1)
            type_input_with_shelf = torch.cat([type_input, shelf_feats_expanded], dim=-1)
        else:
            # Pad with zeros if mel_profile unavailable (keeps type_head input dim consistent)
            shelf_feats_expanded = torch.zeros(
                type_input.shape[0], self.num_bands, 3,
                device=type_input.device, dtype=type_input.dtype
            )
            type_input_with_shelf = torch.cat([type_input, shelf_feats_expanded], dim=-1)
        type_logits = self.type_head(type_input_with_shelf)
        type_probs = F.softmax(type_logits, dim=-1)
        filter_type = type_logits.argmax(dim=-1)
        type_probs_for_params = type_probs

        if gain_aux is not None and self.gain_context_proj is not None:
            gain_hidden = self.gain_context_proj(torch.cat([trunk_out, gain_aux], dim=-1))
        else:
            gain_hidden = trunk_out
        gain_raw = self.gain_head(gain_hidden).squeeze(-1)
        gain_db_raw = ste_clamp(gain_raw * 24.0, -24.0, 24.0)
        if hard_types:
            # Inference: hard-zero HP/LP gain via argmax type mask.
            gain_db = gain_db_raw * self.gainful_type_mask[filter_type]
        else:
            # Training: do NOT apply soft gain mask.
            # Soft mask = gain_db_raw * (sum of gainful type probs) chains
            # ∂L/∂gain_db_raw = gain_mask * ∂L/∂gain_db.  When the type head
            # is wrong (e.g. predicts HP for a peaking band), gain_mask≈0 and
            # gain_head gets near-zero gradient — it cannot learn sign or scale.
            # The zero-gain constraint for HP/LP is handled explicitly by
            # lambda_gain_zero in SimplifiedEQLoss instead.
            gain_db = gain_db_raw

        if type_mel_context is not None and self.freq_context_proj is not None:
            freq_hidden = self.freq_context_proj(
                torch.cat([trunk_out, type_mel_context], dim=-1)
            )
        else:
            freq_hidden = trunk_out
        freq_unit = torch.sigmoid(self.freq_head(freq_hidden).squeeze(-1))
        if self.type_conditioned_frequency:
            if hard_types:
                log_f_min = self.type_log_f_min[filter_type]
                log_f_max = self.type_log_f_max[filter_type]
            else:
                log_f_min = (type_probs_for_params * self.type_log_f_min.view(1, 1, -1)).sum(dim=-1)
                log_f_max = (type_probs_for_params * self.type_log_f_max.view(1, 1, -1)).sum(dim=-1)
        else:
            log_f_min = torch.full_like(freq_unit, self.log_f_min)
            log_f_max = torch.full_like(freq_unit, self.log_f_max)
        log_freq = log_f_min + freq_unit * (log_f_max - log_f_min)
        freq = torch.exp(log_freq)

        q_log = self.q_head(trunk_out).squeeze(-1)
        q_log = ste_clamp(q_log, math.log(0.1), math.log(10.0))
        q = torch.exp(q_log)

        if return_aux:
            aux = {
                "mel_profile_centered": mel_profile_centered,
                "gain_aux_summary": (
                    gain_aux.mean(dim=1) if gain_aux is not None else None
                ),
            }
            return gain_db, freq, q, type_logits, type_probs, filter_type, aux

        return gain_db, freq, q, type_logits, type_probs, filter_type
