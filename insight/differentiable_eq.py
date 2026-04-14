import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# AUDIT: LOW-29 — Math references for biquad coefficient formulas
# ---------------------------------------------------------------------------
# All biquad coefficient formulas below are derived from the Robert Bristow-Johnson
# "Audio EQ Cookbook" (https://webaudio.github.io/Audio-EQ-Cookbook/):
#
#   General biquad transfer function:
#       H(z) = (b0 + b1*z^-1 + b2*z^-2) / (1 + a1*z^-1 + a2*z^-2)
#
#   Key substitutions:
#       A   = 10^(gain_dB / 40)          — amplitude ratio (sqrt of power ratio)
#       w0  = 2*pi*f0 / Fs               — normalized center frequency
#       α   = sin(w0) / (2*Q)            — bandwidth parameter
#       cos = cos(w0)
#
#   Peaking EQ (Bristow-Johnson, eq. 15-20):
#       b0 = 1 + α*A         b1 = -2*cos       b2 = 1 - α*A
#       a0 = 1 + α/A         a1 = -2*cos       a2 = 1 - α/A
#
#   Low-Shelf (Bristow-Johnson, eq. 7-12):
#       b0 = A*((A+1)-(A-1)*cos + 2*sqrt(A)*α)
#       b1 = 2*A*((A-1)-(A+1)*cos)
#       b2 = A*((A+1)-(A-1)*cos - 2*sqrt(A)*α)
#       a0 = (A+1)+(A-1)*cos + 2*sqrt(A)*α
#       a1 = -2*((A-1)+(A+1)*cos)
#       a2 = (A+1)+(A-1)*cos - 2*sqrt(A)*α
#
#   High-Shelf: mirror of Low-Shelf (cos sign flipped)
#   High-Pass / Low-Pass: standard forms with α for resonance control
# ---------------------------------------------------------------------------


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
        # C-08: track how often alpha hits the 10.0 clamp ceiling
        self._alpha_clamp_count = 0
        # AUDIT: CRITICAL-01 V-02 — track H_mag clamping for diagnostics
        self._hmag_clamp_count = 0

    def compute_biquad_coeffs(self, gain_db, freq, q):
        """
        Computes biquad coefficients for Peaking/Bell EQ filters.
        Inputs: tensors of shape (Batch, Num_Bands).
        Returns: b0, b1, b2, a1, a2 (each shape (Batch, Num_Bands))
        """
        A = 10.0 ** (gain_db / 40.0)
        w0 = 2.0 * torch.pi * freq / self.sample_rate
        # AUDIT: CRITICAL-01 V-01 — Add epsilon to denominator to prevent division by zero
        # when Q is near zero. Use 1e-4 for numerical stability.
        alpha = torch.sin(w0) / (2.0 * q + 1e-4)
        alpha_raw = alpha
        # V-02: Clamp alpha to prevent extreme values at low Q — add both min and max bounds
        # because alpha can also be too small, causing numerical issues in coefficient formulas
        alpha = torch.clamp(alpha, min=1e-6, max=10.0)
        # C-08: track alpha clamp hits (shared counter with multitype path)
        self._alpha_clamp_count += int((alpha_raw > 10.0).any().item())
        if self._alpha_clamp_count % 1000 == 0 and self._alpha_clamp_count > 0:
            print(f"[C-08 WARN] DifferentiableBiquadCascade: alpha hit 10.0 clamp "
                  f"{self._alpha_clamp_count} times total. Consider tightening Q floor.")

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
        # AUDIT: CRITICAL-01 V-01 — Add epsilon to denominator to prevent division by zero
        # when Q is near zero. Use 1e-4 for numerical stability.
        alpha_raw = sin_w0 / (2.0 * q + 1e-4)
        # V-02: Clamp alpha to prevent extreme values at low Q — add both min and max bounds
        alpha = torch.clamp(alpha_raw, min=1e-6, max=10.0)
        # C-08: track alpha clamp hits (shared counter with peaking path)
        self._alpha_clamp_count += int((alpha_raw > 10.0).any().item())
        if self._alpha_clamp_count % 1000 == 0 and self._alpha_clamp_count > 0:
            print(f"[C-08 WARN] DifferentiableBiquadCascade: alpha hit 10.0 clamp "
                  f"{self._alpha_clamp_count} times total. Consider tightening Q floor.")
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

        # Select coefficients based on filter type using gather (single op vs 5 nested where)
        b0_all = torch.stack([peak_b0, ls_b0, hs_b0, hp_b0, lp_b0], dim=-1)  # (B, N, 5)
        b1_all = torch.stack([peak_b1, ls_b1, hs_b1, hp_b1, lp_b1], dim=-1)
        b2_all = torch.stack([peak_b2, ls_b2, hs_b2, hp_b2, lp_b2], dim=-1)
        a0_all = torch.stack([peak_a0, ls_a0, hs_a0, hp_a0, lp_a0], dim=-1)
        a1_all = torch.stack([peak_a1, ls_a1, hs_a1, hp_a1, lp_a1], dim=-1)
        a2_all = torch.stack([peak_a2, ls_a2, hs_a2, hp_a2, lp_a2], dim=-1)
        ft_idx = filter_type.unsqueeze(-1)  # (B, N, 1)
        b0 = b0_all.gather(-1, ft_idx).squeeze(-1)
        b1 = b1_all.gather(-1, ft_idx).squeeze(-1)
        b2 = b2_all.gather(-1, ft_idx).squeeze(-1)
        a0 = a0_all.gather(-1, ft_idx).squeeze(-1)
        a1 = a1_all.gather(-1, ft_idx).squeeze(-1)
        a2 = a2_all.gather(-1, ft_idx).squeeze(-1)

        # Normalize by a0 (clamp to prevent near-zero division)
        # M-05: floor raised from 1e-4 to 1e-3 for better numerical stability
        a0 = torch.clamp(a0, min=1e-3)
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

        # AUDIT: CRITICAL-01 V-02 — Increased epsilon to 1e-3 for headroom against high-Q resonance
        # Added upper-bound clamping to prevent inf propagation when den_mag2 is near zero
        ratio = num_mag2 / (den_mag2 + 1e-3)
        # Track and log clamping diagnostics
        clamp_min_mask = ratio < 1e-8
        clamp_max_mask = ratio > 1e6
        if clamp_min_mask.any() or clamp_max_mask.any():
            self._hmag_clamp_count += int((clamp_min_mask | clamp_max_mask).sum().item())
            if self._hmag_clamp_count % 1000 == 0 and self._hmag_clamp_count > 0:
                print(f"[V-02 WARN] DifferentiableBiquadCascade: H_mag hit clamping bounds "
                      f"{self._hmag_clamp_count} times total (min=1e-8, max=1e6). "
                      f"Consider checking for extreme gain/freq/Q values.")
        H_mag = torch.sqrt(torch.clamp(ratio, min=1e-8, max=1e6))

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
        # M-10: log-space product prevents overflow from multiplying 5 bands
        log_H = torch.log(H_mag_bands.clamp(min=1e-8))
        log_H_total = log_H.sum(dim=1)
        H_mag_total = torch.exp(log_H_total)
        H_mag_total = torch.clamp(H_mag_total, min=1e-6, max=1e4)
        return H_mag_total

    def forward_soft(self, gain_db, freq, q, type_probs, n_fft=2048):
        """
        Differentiable soft-type frequency response.

        Each filter type's band response is evaluated exactly, then mixed by
        the predicted type probabilities before cascading the bands.
        """
        if type_probs is None:
            raise ValueError("`type_probs` must be provided for soft response.")

        H_per_type = []
        for filter_idx in range(NUM_FILTER_TYPES):
            filter_type = torch.full(
                type_probs.shape[:2],
                filter_idx,
                dtype=torch.long,
                device=type_probs.device,
            )
            b0, b1, b2, a1, a2 = self.compute_biquad_coeffs_multitype(
                gain_db, freq, q, filter_type
            )
            H_per_type.append(self.freq_response(b0, b1, b2, a1, a2, n_fft))

        H_stack = torch.stack(H_per_type, dim=2)
        H_soft_bands = (H_stack * type_probs.unsqueeze(-1)).sum(dim=2)
        # M-10: log-space product prevents overflow from multiplying 5 bands
        log_H = torch.log(H_soft_bands.clamp(min=1e-8))
        log_H_total = log_H.sum(dim=1)
        H_mag_total = torch.exp(log_H_total)
        H_mag_total = torch.clamp(H_mag_total, min=1e-6, max=1e4)
        return H_mag_total

    def process_audio(
        self,
        audio,
        gain_db,
        freq,
        q,
        filter_type=None,
        type_probs=None,
        n_fft=2048,
        hop_length=512,
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
        if type_probs is not None:
            H_mag = self.forward_soft(gain_db, freq, q, type_probs, n_fft=n_fft)
        else:
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


def compute_per_type_shape_features(pred_gain, pred_freq, pred_q, dsp_cascade,
                                     n_fft=512):
    """
    Fix 5: Per-type "what-if" spectral shape features for type classifier.

    For each band, compute frequency response for ALL 5 filter types using
    the current gain/freq/Q predictions. Extract 4 discriminative features
    from each → 20 features per band total.

    Features:
    1. Energy asymmetry (low vs high frequency) — shelves have strong asymmetry
    2. Roll-off slope — HP/LP have steep slopes, peaking ≈ 0
    3. Peak-to-plateau ratio — peaking = high, shelf = low
    4. Edge energy ratio — HP/LP have extreme values (near-zero on one side)

    Args:
        pred_gain: (B, N) gain in dB
        pred_freq: (B, N) frequency in Hz
        pred_q: (B, N) Q factor
        dsp_cascade: DifferentiableBiquadCascade instance
        n_fft: FFT size for evaluation (512 = fast + sufficient resolution)

    Returns:
        (B, N, 20) — 4 features × 5 types, all differentiable
    """
    B, N = pred_gain.shape
    all_features = []

    for type_idx in range(5):  # peaking, lowshelf, highshelf, hp, lp
        type_tensor = torch.full((B, N), type_idx, device=pred_gain.device, dtype=torch.long)

        # Compute per-band frequency response for this type
        b0, b1, b2, a1, a2 = dsp_cascade.compute_biquad_coeffs_multitype(
            pred_gain, pred_freq, pred_q, type_tensor)
        H_mag = dsp_cascade.freq_response(b0, b1, b2, a1, a2, n_fft=n_fft)  # (B, N, n_fft_bins)
        H_db = 20 * torch.log10(H_mag.clamp(min=1e-6))

        n_bins = H_mag.shape[-1]
        mid = n_bins // 2
        q25 = n_bins // 4
        q75 = 3 * n_bins // 4

        # 1. Energy asymmetry — low vs high frequency
        low_e = H_mag[..., :mid].pow(2).mean(-1)
        high_e = H_mag[..., mid:].pow(2).mean(-1)
        asymmetry = (low_e - high_e) / (low_e + high_e).clamp(min=1e-8)

        # 2. Roll-off slope
        slope = (H_db[..., q75:].mean(-1) - H_db[..., :q25].mean(-1)) / 40.0

        # 3. Peak-to-plateau ratio (peaking = high, shelf = low)
        peak_val = H_db.max(dim=-1).values
        mean_val = H_db.mean(dim=-1)
        peak_ratio = (peak_val - mean_val) / 12.0

        # 4. Edge energy ratio (HP/LP have near-zero energy on one side)
        edge_ratio = H_mag[..., :q25].pow(2).mean(-1) / H_mag[..., q75:].pow(2).mean(-1).clamp(min=1e-8)
        edge_ratio = torch.log10(edge_ratio.clamp(min=1e-4, max=1e4)) / 4.0

        all_features.append(torch.stack([asymmetry, slope, peak_ratio, edge_ratio], dim=-1))

    return torch.cat(all_features, dim=-1)  # (B, N, 20)


# ─── Broad class indices for hierarchical type head ─────────────────────
BROAD_PASS = 0    # HP + LP
BROAD_SHELF = 1   # LS + HS
BROAD_PEAKING = 2  # Peaking
NUM_BROAD_CLASSES = 3

# Mapping from 5-class fine type to 3-class broad type
FINE_TO_BROAD = {
    FILTER_PEAKING: BROAD_PEAKING,
    FILTER_LOWSHELF: BROAD_SHELF,
    FILTER_HIGHSHELF: BROAD_SHELF,
    FILTER_HIGHPASS: BROAD_PASS,
    FILTER_LOWPASS: BROAD_PASS,
}

# Mapping from broad class + fine class to binary subclass index
# Pass: HP=0, LP=1.  Shelf: LS=0, HS=1.
PASS_HP_IDX = 0
PASS_LP_IDX = 1
SHELF_LS_IDX = 0
SHELF_HS_IDX = 1


class HierarchicalTypeHead(nn.Module):
    """Two-stage hierarchical type classifier.

    Stage 1: 3-class broad classification (pass / shelf / peaking).
    Stage 2a: Binary HP vs LP (only for pass bands).
    Stage 2b: Binary LS vs HS (only for shelf bands).

    The shelf subhead receives extra asymmetry features (DC gain, Nyquist gain,
    low/high energy ratio, spectral tilt) to resolve the hardest confusion.
    """

    def __init__(self, input_dim, num_filter_types=5):
        super().__init__()
        self.num_filter_types = num_filter_types

        # Stage 1: Broad class classifier (3 classes)
        self.broad_head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_BROAD_CLASSES),
        )
        nn.init.xavier_uniform_(self.broad_head[0].weight, gain=0.5)
        nn.init.zeros_(self.broad_head[-1].bias)

        # Stage 2a: Pass subclassifier (HP vs LP) — binary
        self.pass_head = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # [HP_logit, LP_logit]
        )
        nn.init.xavier_uniform_(self.pass_head[0].weight, gain=0.3)
        nn.init.zeros_(self.pass_head[-1].bias)

        # Stage 2b: Shelf subclassifier (LS vs HS) — binary with extra features
        # Extra features: dc_gain, nyquist_gain, low_high_ratio, spectral_tilt = 4
        self.shelf_head = nn.Sequential(
            nn.Linear(input_dim + 4, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # [LS_logit, HS_logit]
        )
        nn.init.xavier_uniform_(self.shelf_head[0].weight, gain=0.3)
        nn.init.zeros_(self.shelf_head[-1].bias)

    def forward(self, type_input, shelf_extra_features=None):
        """
        Args:
            type_input: (B, N, input_dim) — trunk + shape + spectral features
            shelf_extra_features: (B, N, 4) — [dc_gain, nyquist_gain, lo_hi_ratio, tilt]
                                  or None (computed internally if needed)

        Returns:
            type_logits: (B, N, 5) — mapped back to 5-class logits
            broad_logits: (B, N, 3) — Stage 1 output
            pass_logits: (B, N, 2) — Stage 2a output
            shelf_logits: (B, N, 2) — Stage 2b output
        """
        # Stage 1: Broad class
        broad_logits = self.broad_head(type_input)  # (B, N, 3)

        # Stage 2a: Pass subclass
        pass_logits = self.pass_head(type_input)  # (B, N, 2)

        # Stage 2b: Shelf subclass with extra features
        if shelf_extra_features is not None:
            shelf_input = torch.cat([type_input, shelf_extra_features], dim=-1)
        else:
            # Zero-pad if no extra features (shouldn't happen in practice)
            shelf_input = torch.cat([type_input, type_input.new_zeros(type_input.shape[0], type_input.shape[1], 4)], dim=-1)
        shelf_logits = self.shelf_head(shelf_input)  # (B, N, 2)

        # ── Map hierarchical outputs to 5-class logits ──
        # During training: use soft (differentiable) mapping via broad probabilities
        # During inference: hard routing via argmax
        B, N, _ = broad_logits.shape
        type_logits = broad_logits.new_zeros(B, N, self.num_filter_types)

        # Soft routing: weight Stage 2 outputs by broad probabilities
        broad_probs = F.softmax(broad_logits, dim=-1)  # (B, N, 3)
        pass_probs = F.softmax(pass_logits, dim=-1)    # (B, N, 2)
        shelf_probs = F.softmax(shelf_logits, dim=-1)  # (B, N, 2)

        # peaking logit = broad_peaking_logit
        type_logits[..., FILTER_PEAKING] = broad_logits[..., BROAD_PEAKING]

        # HP = broad_pass * pass_HP
        type_logits[..., FILTER_HIGHPASS] = (
            broad_logits[..., BROAD_PASS] + pass_logits[..., PASS_HP_IDX]
        )
        # LP = broad_pass * pass_LP
        type_logits[..., FILTER_LOWPASS] = (
            broad_logits[..., BROAD_PASS] + pass_logits[..., PASS_LP_IDX]
        )
        # LS = broad_shelf + shelf_LS
        type_logits[..., FILTER_LOWSHELF] = (
            broad_logits[..., BROAD_SHELF] + shelf_logits[..., SHELF_LS_IDX]
        )
        # HS = broad_shelf + shelf_HS
        type_logits[..., FILTER_HIGHSHELF] = (
            broad_logits[..., BROAD_SHELF] + shelf_logits[..., SHELF_HS_IDX]
        )

        return type_logits, broad_logits, pass_logits, shelf_logits


class MultiTypeEQParameterHead(nn.Module):
    """
    Parameter head for multi-type EQ estimation.

    Key behaviors:
    - Gain is type-aware: HP/LP gains are suppressed via type probabilities.
    - Frequency is mapped through type-specific log-frequency bounds.
    - Type prediction uses both global and per-band mel evidence.

    ⚠️  TYPE COLLAPSE WARNING AND MITIGATION STRATEGY ⚠️
    ─────────────────────────────────────────────────────
    The type classification head is prone to "type collapse" — where the
    model predicts a single filter type (typically peaking) for all bands,
    achieving ~20% accuracy (random baseline for 5 types).

    Root causes:
    1. Gumbel temperature too high (tau > 1.0): makes softmax nearly
       uniform, providing no discriminative signal to the parameter head.
    2. lambda_type too low relative to parameter regression losses:
       the optimizer prioritizes gain/freq/Q accuracy over type learning.
    3. Detached type gradients during warmup: if type_logits are detached
       from the gain gradient path during warmup, the encoder never learns
       type-discriminative features.

    Mitigations (see conf/config.yaml):
    - gumbel.start_tau: 0.5 (NOT 2.0 — tau=2.0 is effectively uniform)
    - curriculum.stages[0].lambda_type: 8.0 (strong type signal in stage 1)
    - loss.lambda_type_entropy: 0.5 (penalizes peaked type distributions)
    - loss.class_weight_multipliers: [1.0, 2.0, 2.0, 2.0, 2.0]
      (boosts non-peaking type weights to counter peaking dominance)

    If type accuracy stays below 25% after 10 epochs, the training run
    should be terminated and the above parameters reviewed.
    See tests/test_type_collapse.py for automated detection.
    """

    def __init__(
        self,
        embedding_dim,
        num_bands=5,
        num_filter_types=NUM_FILTER_TYPES,
        n_mels=0,
        type_conditioned_frequency=True,
        n_shelf_bands=16,
        n_fft=2048,
        sample_rate=44100,
        dsp_cascade=None,
        hierarchical_type_head=False,
    ):
        super().__init__()
        self.num_bands = num_bands
        self.num_filter_types = num_filter_types
        self.n_mels = n_mels
        self.type_conditioned_frequency = type_conditioned_frequency
        self.n_shelf_bands = n_shelf_bands
        self.hierarchical_type_head = hierarchical_type_head
        self.n_fft_bins = n_fft // 2 + 1
        print(f"  [model] hierarchical_type_head={hierarchical_type_head}")
        self.sample_rate = sample_rate
        self._dsp_cascade = dsp_cascade  # for Fix 5 shape features (plain attr, not submodule)
        self.log_f_min = FILTER_LOG_FREQ_BOUNDS[FILTER_PEAKING][0]
        self.log_f_max = FILTER_LOG_FREQ_BOUNDS[FILTER_PEAKING][1]
        hidden_dim = 64
        mel_hidden_dim = 64

        # Shared trunk: embedding -> per-band features
        self.trunk = nn.Sequential(
            nn.Linear(embedding_dim, num_bands * hidden_dim),
            nn.ReLU(),
        )

        # Per-band H_db prediction from trunk features — expanded MLP decoder.
        # Three-layer 64→256→512→1025 with GELU. Rank 512 allows complex
        # spectral shapes (multi-peak shelves, asymmetric Q bumps).
        self.h_db_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, self.n_fft_bins),
        )
        nn.init.xavier_uniform_(self.h_db_head[0].weight, gain=0.1)
        nn.init.xavier_uniform_(self.h_db_head[2].weight, gain=0.1)
        nn.init.xavier_uniform_(self.h_db_head[4].weight, gain=0.1)
        nn.init.zeros_(self.h_db_head[4].bias)

        # Learned mixing weight for gain: sigmoid(alpha) * gain_hdb + (1-sigmoid(alpha)) * gain_raw
        # Start at 0.0 (all H_db) since spectral path is proven better
        self.register_buffer('gain_mix_value', torch.tensor(0.1))

        # FiLM: type probabilities → affine transform on trunk features
        self.type_film_gamma = nn.Parameter(torch.ones(num_filter_types, hidden_dim) * 0.1)
        self.type_film_beta = nn.Parameter(torch.zeros(num_filter_types, hidden_dim))

        # Gumbel-Softmax temperature for differentiable type selection
        self.register_buffer('gumbel_tau', torch.tensor(1.0))

        # DC/Nyquist gain scale for shelf type evidence from per-band H_db.
        # DC gain is the analytically exact discriminant: lowshelf has gain at DC,
        # highshelf has gain at Nyquist, peaking has ~0 dB at both.
        self.dc_shelf_scale = nn.Parameter(torch.tensor(1.0))

        # Gain head: 2-layer MLP + ste_clamp (complements H_db interpolation path)
        self.gain_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        nn.init.xavier_uniform_(self.gain_head[0].weight, gain=0.1)
        nn.init.zeros_(self.gain_head[0].bias)
        nn.init.xavier_uniform_(self.gain_head[2].weight, gain=0.1)
        nn.init.zeros_(self.gain_head[2].bias)

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

        self.shelf_feature_encoder = (
            nn.Sequential(
                nn.Conv1d(3, 32, kernel_size=5, padding=2),
                nn.GELU(),
                nn.Conv1d(32, mel_hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
            )
            if n_mels > 0
            else None
        )
        self.shelf_band_query = (
            nn.Linear(hidden_dim, mel_hidden_dim) if n_mels > 0 else None
        )
        self.shelf_context_norm = (
            nn.LayerNorm(mel_hidden_dim) if n_mels > 0 else None
        )
        self.shelf_bias_head = (
            nn.Sequential(
                nn.Linear(mel_hidden_dim, 32),
                nn.GELU(),
                nn.Linear(32, 2),
            )
            if n_mels > 0
            else None
        )
        if self.shelf_bias_head is not None:
            # Small random init (not zeros) so gradient flows through the
            # entire attention chain from the very first training step.
            # Zero-init creates a dead gradient: ∂(0·x)/∂x = 0 for the
            # preceding Conv1d encoder and attention layers.
            nn.init.normal_(self.shelf_bias_head[-1].weight, std=0.01)
            nn.init.zeros_(self.shelf_bias_head[-1].bias)
        # Fixed-scale direct logit bias from prefix_suffix_ratio at the
        # first and last mel-band positions.  These are non-learned scalar
        # evidence signals that bypass the attention mechanism entirely.
        # prefix_suffix_ratio[0] = log(E_below_fc_min / E_above_fc_min)
        #   → strongly positive for lowshelf boost
        # prefix_suffix_ratio[-1] = log(E_below_fc_max / E_above_fc_max)
        #   → strongly positive for highshelf boost (treble-heavy)
        self.direct_shelf_scale = nn.Parameter(torch.tensor(2.0))

        # Number of hand-crafted spectral shape features for type discrimination
        self.n_shape_features = 7  # split_ratio, skewness, centroid_offset, rolloff, stopband, gain_mag, freq_pos

        type_input_dim = (
            hidden_dim
            + (20 if dsp_cascade is not None else 0)
            + (self.n_shape_features if n_mels > 0 else 0)
        )
        self.type_head = nn.Sequential(
            nn.Linear(type_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_filter_types),
        )
        with torch.no_grad():
            # Small peaking bias to counteract shelf_bias_head which adds
            # logits to lowshelf/highshelf during forward. Without this,
            # peaking gets systematically outcompeted. Bias: [peaking=0.5,
            # lowshelf=0, highshelf=0, highpass=0, lowpass=0]
            self.type_head[-1].bias.zero_()
            self.type_head[-1].bias[FILTER_PEAKING] = 0.5

        # Hierarchical two-stage type head (optional)
        if hierarchical_type_head:
            self.hier_type_head = HierarchicalTypeHead(type_input_dim, num_filter_types)
        else:
            self.hier_type_head = None

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

    def _mel_log_to_normalized_energy(self, mel_profile: torch.Tensor) -> torch.Tensor:
        """Convert log-mel to scale-invariant linear energy for shelf features."""
        mel_energy = torch.exp(mel_profile.clamp(min=-12.0, max=12.0))
        return mel_energy / mel_energy.mean(dim=-1, keepdim=True).clamp(min=1e-6)

    def _compute_shelf_feature_map(self, mel_profile: torch.Tensor) -> torch.Tensor:
        """
        Build per-cutoff shelf evidence curves in normalized linear-energy space.

        Returns:
            (B, 3, n_mels) ordered as:
              prefix_suffix_ratio, boundary_step, local_peakness
        """
        _, n_mels = mel_profile.shape
        if n_mels <= 1:
            return mel_profile.new_zeros(mel_profile.shape[0], 3, n_mels)

        mel_energy = self._mel_log_to_normalized_energy(mel_profile)
        boundary_idx = torch.arange(n_mels, device=mel_profile.device).clamp(
            1, n_mels - 1
        )
        boundary_idx = boundary_idx.to(torch.long)
        cumsum = F.pad(mel_energy.cumsum(dim=-1), (1, 0))

        left_counts = boundary_idx.to(mel_profile.dtype).unsqueeze(0)
        right_counts = (n_mels - boundary_idx).to(mel_profile.dtype).unsqueeze(0)
        left_means = cumsum[:, boundary_idx] / left_counts
        right_means = (cumsum[:, -1:] - cumsum[:, boundary_idx]) / right_counts
        prefix_suffix_ratio = torch.log(
            (left_means + 1e-6) / (right_means + 1e-6)
        )

        window = min(max(1, self.n_shelf_bands), max(1, n_mels // 4))
        energy = mel_energy.unsqueeze(1)
        padded = F.pad(energy, (2 * window, 2 * window), mode="replicate")
        short_windows = padded.unfold(-1, window, 1).squeeze(1)
        long_windows = padded.unfold(-1, 2 * window, 1).squeeze(1)

        left_windows = short_windows[:, window : window + n_mels, :]
        right_windows = short_windows[:, 2 * window : 2 * window + n_mels, :]
        boundary_step = left_windows.mean(dim=-1) - right_windows.mean(dim=-1)

        left_outer = short_windows[:, :n_mels, :]
        center_windows = long_windows[:, window : window + n_mels, :]
        right_outer = short_windows[:, 3 * window : 3 * window + n_mels, :]
        local_peakness = center_windows.mean(dim=-1) - 0.5 * (
            left_outer.mean(dim=-1) + right_outer.mean(dim=-1)
        )

        feature_map = torch.stack(
            [prefix_suffix_ratio, boundary_step, local_peakness],
            dim=1,
        )
        return torch.clamp(feature_map, min=-6.0, max=6.0)

    def _build_shelf_context(self, trunk_out, mel_profile):
        if self.shelf_feature_encoder is None or mel_profile is None:
            return None, None

        shelf_feature_map = self._compute_shelf_feature_map(mel_profile)
        shelf_values = self.shelf_feature_encoder(shelf_feature_map)
        shelf_queries = self.shelf_band_query(trunk_out)
        attn_scores = torch.einsum("bnc,bcm->bnm", shelf_queries, shelf_values)
        attn_scores = attn_scores / math.sqrt(shelf_values.shape[1])
        shelf_attention = torch.softmax(attn_scores, dim=-1)
        shelf_context = torch.einsum("bnm,bcm->bnc", shelf_attention, shelf_values)
        shelf_context = self.shelf_context_norm(shelf_context)
        return shelf_context, shelf_attention

    def _compute_spectral_shape_features(
        self, mel_profile, pred_gain_db, pred_freq_hz
    ):
        """
        Hand-crafted spectral shape features for type discrimination.

        These encode domain knowledge about EQ spectral morphologies:
        - Pass filters: steep rolloff, near-zero stopband energy
        - Shelf filters: asymmetric energy split, one-sided shift
        - Peaking: symmetric bell, flat at both extremes

        Args:
            mel_profile: (B, n_mels) log-mel spectrogram (centered)
            pred_gain_db: (B, N) predicted gain in dB
            pred_freq_hz: (B, N) predicted frequency in Hz

        Returns:
            (B, N, 7) feature tensor:
              [split_ratio, skewness, centroid_offset,
               rolloff_steepness, stopband_ratio, gain_magnitude, freq_position]
        """
        B, n_mels = mel_profile.shape
        N = pred_gain_db.shape[1]
        device = mel_profile.device
        dtype = mel_profile.dtype

        # Normalize mel to linear energy for ratio computations
        mel_energy = torch.exp(mel_profile.clamp(min=-12.0, max=12.0))
        mel_energy = mel_energy / mel_energy.mean(dim=-1, keepdim=True).clamp(min=1e-6)

        # Map predicted frequency to mel-bin index
        freq_axis = torch.linspace(0, 1, n_mels, device=device, dtype=dtype)
        freq_normalized = (pred_freq_hz / self.sample_rate).clamp(0.0, 0.5)
        center_idx = (freq_normalized * 2.0 * (n_mels - 1)).long().clamp(1, n_mels - 2)

        # Expand mel for per-band computation
        mel_expanded = mel_energy.unsqueeze(1).expand(B, N, n_mels)  # (B, N, n_mels)
        center_idx_exp = center_idx.unsqueeze(-1)  # (B, N, 1)

        # 1. Energy split ratio at predicted center frequency
        # Peaking: ~1.0 (symmetric). Lowshelf: >>1.0 (bass-heavy). Highshelf: <<1.0.
        cumsum = torch.cumsum(mel_expanded, dim=-1)
        arange = torch.arange(1, n_mels + 1, device=device, dtype=dtype)
        left_counts = center_idx.float().clamp(min=1)  # (B, N)
        right_counts = (n_mels - center_idx).float().clamp(min=1)

        # Gather cumsum at center_idx for each band
        idx = center_idx.clamp(0, n_mels - 1)  # (B, N)
        left_energy = torch.gather(cumsum, 2, idx.unsqueeze(-1)).squeeze(-1)  # (B, N)
        total_energy = cumsum[:, :, -1].squeeze(-1)  # (B, N)
        right_energy = (total_energy - left_energy).clamp(min=1e-6)  # (B, N)

        split_ratio = torch.log((left_energy / left_counts + 1e-6) /
                                (right_energy / right_counts + 1e-6))

        # 2. Spectral skewness (3rd moment of delta-spectrum)
        # Peaking: ~0. Lowshelf: positive. Highshelf: negative.
        delta = mel_expanded - mel_expanded.mean(dim=-1, keepdim=True)
        delta_std = delta.std(dim=-1).clamp(min=1e-6)
        skewness = ((delta ** 3).mean(dim=-1)) / (delta_std ** 3 + 1e-6)
        skewness = skewness.clamp(-5.0, 5.0)

        # 3. Centroid offset from predicted center
        freq_axis_exp = freq_axis.unsqueeze(0).unsqueeze(0).expand(B, N, n_mels)
        centroid = (delta.abs() * freq_axis_exp).sum(dim=-1) / (delta.abs().sum(dim=-1) + 1e-6)
        center_normalized = freq_normalized.squeeze(-1) if freq_normalized.dim() == 3 else freq_normalized
        centroid_offset = centroid - center_normalized

        # 4. Rolloff steepness at predicted center (local gradient)
        idx_clamped = center_idx.clamp(1, n_mels - 2)
        mel_left = torch.gather(mel_profile.unsqueeze(1).expand(B, N, n_mels), 2,
                                (idx_clamped - 1).unsqueeze(-1)).squeeze(-1)
        mel_right = torch.gather(mel_profile.unsqueeze(1).expand(B, N, n_mels), 2,
                                 (idx_clamped + 1).unsqueeze(-1)).squeeze(-1)
        rolloff_steepness = (mel_right - mel_left).abs()  # HP/LP: large; peaking/shelf: small

        # 5. Stopband energy ratio
        # For HP: energy below cutoff should be near-zero
        # For LP: energy above cutoff should be near-zero
        # For peaking/shelf: roughly equal on both sides
        stopband_below = left_energy / (total_energy.squeeze(-1) + 1e-6)
        stopband_above = right_energy / (total_energy.squeeze(-1) + 1e-6)
        stopband_ratio = torch.log(stopband_below.clamp(min=1e-6) /
                                   stopband_above.clamp(min=1e-6) + 1e-6)

        # 6. Gain magnitude (high gain = easier type discrimination)
        gain_magnitude = pred_gain_db.abs() / 12.0  # normalized to [0, 2]

        # 7. Frequency position (normalized log-frequency)
        freq_position = (torch.log(pred_freq_hz.clamp(min=20.0, max=20000.0)) -
                         math.log(20.0)) / (math.log(20000.0) - math.log(20.0))

        features = torch.stack([
            split_ratio.clamp(-3.0, 3.0),
            skewness / 5.0,
            centroid_offset.clamp(-1.0, 1.0),
            rolloff_steepness / 20.0,
            stopband_ratio.clamp(-3.0, 3.0),
            gain_magnitude.clamp(0.0, 2.0),
            freq_position,
        ], dim=-1)  # (B, N, 7)

        return features

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

        if gain_aux is not None and self.gain_context_proj is not None:
            gain_hidden = self.gain_context_proj(torch.cat([trunk_out, gain_aux], dim=-1))
        else:
            gain_hidden = trunk_out
        gain_raw = self.gain_head(gain_hidden).squeeze(-1)
        gain_db_raw = ste_clamp(gain_raw * 24.0, -24.0, 24.0)

        # ── Per-band H_db prediction from trunk features ────────────────
        h_db_pred = self.h_db_head(trunk_out)  # (B, num_bands, n_fft_bins)
        shelf_context, shelf_attention = self._build_shelf_context(trunk_out, mel_profile)

        # Fix 5: Compute per-type shape features using preliminary params
        # (before FiLM conditioning, since type_probs aren't available yet)
        if self._dsp_cascade is not None:
            # Preliminary freq: use peaking bounds (full range), no type conditioning
            freq_unit_prelim = torch.sigmoid(self.freq_head(trunk_out).squeeze(-1))
            log_freq_prelim = self.log_f_min + freq_unit_prelim * (self.log_f_max - self.log_f_min)
            freq_prelim = torch.exp(log_freq_prelim)
            # Preliminary Q from un-FiLMed trunk
            q_log_prelim = self.q_head(trunk_out).squeeze(-1)
            q_log_prelim = ste_clamp(q_log_prelim, math.log(0.1), math.log(10.0))
            q_prelim = torch.exp(q_log_prelim)
            shape_features = compute_per_type_shape_features(
                gain_db_raw, freq_prelim, q_prelim, self._dsp_cascade, n_fft=512)
            type_input = torch.cat([type_input, shape_features], dim=-1)

        # Hand-crafted spectral shape features for type discrimination
        if mel_profile is not None and self.n_mels > 0:
            freq_unit_prelim = torch.sigmoid(self.freq_head(trunk_out).squeeze(-1))
            log_freq_prelim = self.log_f_min + freq_unit_prelim * (self.log_f_max - self.log_f_min)
            freq_prelim_hz = torch.exp(log_freq_prelim)
            spectral_shape = self._compute_spectral_shape_features(
                mel_profile_centered if mel_profile_centered is not None else mel_profile,
                gain_db_raw, freq_prelim_hz)
            type_input = torch.cat([type_input, spectral_shape], dim=-1)

        # ── Compute DC/Nyquist gain for shelf evidence (used by both paths) ──
        dc_gain = h_db_pred[:, :, 0]         # (B, N) gain at DC per band
        nyquist_gain = h_db_pred[:, :, -1]   # (B, N) gain at Nyquist per band

        # ── Type logits: hierarchical or flat ──
        hier_aux = None  # Will hold (broad_logits, pass_logits, shelf_logits) if hierarchical

        if self.hierarchical_type_head:
            # Hierarchical two-stage type classification
            # Compute shelf extra features for Stage 2b
            shelf_extra = torch.stack([
                dc_gain,           # DC gain — strong lowshelf evidence
                nyquist_gain,      # Nyquist gain — strong highshelf evidence
                dc_gain - nyquist_gain,  # Asymmetry: positive→lowshelf, negative→highshelf
                (dc_gain + nyquist_gain).abs(),  # Total edge gain: high for shelves, low for peaking
            ], dim=-1)  # (B, N, 4)

            type_logits, broad_logits, pass_logits, shelf_logits = self.hier_type_head(
                type_input, shelf_extra_features=shelf_extra
            )
            hier_aux = (broad_logits, pass_logits, shelf_logits)

            # Add shelf bias from attention (same as flat path)
            shelf_bias = type_logits.new_zeros(type_logits.shape[0], self.num_bands, 2)
            if shelf_context is not None and self.shelf_bias_head is not None:
                shelf_bias = self.shelf_bias_head(shelf_context)
            # Apply DC/Nyquist gated evidence to the 5-class logits
            dc_gate = torch.sigmoid((dc_gain.abs() - 0.5) * 2.0)
            nyq_gate = torch.sigmoid((nyquist_gain.abs() - 0.5) * 2.0)
            type_logits = type_logits.clone()
            type_logits[..., FILTER_LOWSHELF] += dc_gain * dc_gate * self.dc_shelf_scale
            type_logits[..., FILTER_HIGHSHELF] += nyquist_gain * nyq_gate * self.dc_shelf_scale
            peaking_gate = (1.0 - dc_gate) * (1.0 - nyq_gate)
            type_logits[..., FILTER_PEAKING] += peaking_gate * self.dc_shelf_scale
        else:
            # Original flat 5-class type head
            type_logits = self.type_head(type_input)
            shelf_bias = type_logits.new_zeros(type_logits.shape[0], self.num_bands, 2)
            if shelf_context is not None and self.shelf_bias_head is not None:
                shelf_bias = self.shelf_bias_head(shelf_context)
            if shelf_attention is None:
                shelf_attention = type_logits.new_zeros(
                    type_logits.shape[0], self.num_bands, self.n_mels
                )
            low_shelf_bias = shelf_bias[..., 0] * 0.1
            high_shelf_bias = shelf_bias[..., 1] * 0.1
            type_logits = type_logits.clone()
            type_logits[..., FILTER_LOWSHELF] += low_shelf_bias
            type_logits[..., FILTER_HIGHSHELF] += high_shelf_bias
            type_logits[..., FILTER_PEAKING] -= 0.15 * (
                shelf_bias[..., 0] + shelf_bias[..., 1]
            )

            # Direct non-learned shelf evidence from the feature map.
            if mel_profile is not None and self.n_shelf_bands > 0:
                feature_map = self._compute_shelf_feature_map(mel_profile)  # (B, 3, n_mels)
                prefix_suffix = feature_map[:, 0, :]  # (B, n_mels)
                direct_lo = prefix_suffix[:, 0:1] * self.direct_shelf_scale  # (B, 1)
                direct_hi = -prefix_suffix[:, -1:] * self.direct_shelf_scale  # (B, 1)
                type_logits[..., FILTER_LOWSHELF] += direct_lo  # broadcast over bands
                type_logits[..., FILTER_HIGHSHELF] += direct_hi
                type_logits[..., FILTER_PEAKING] -= 0.5 * (direct_lo + direct_hi)

            # DC/Nyquist gain — gated to prevent h_db_head gaming
            dc_gate = torch.sigmoid((dc_gain.abs() - 0.5) * 2.0)
            nyq_gate = torch.sigmoid((nyquist_gain.abs() - 0.5) * 2.0)
            type_logits[..., FILTER_LOWSHELF] += dc_gain * dc_gate * self.dc_shelf_scale
            type_logits[..., FILTER_HIGHSHELF] += nyquist_gain * nyq_gate * self.dc_shelf_scale
            peaking_gate = (1.0 - dc_gate) * (1.0 - nyq_gate)
            type_logits[..., FILTER_PEAKING] += peaking_gate * self.dc_shelf_scale

        type_probs = F.gumbel_softmax(type_logits, tau=self.gumbel_tau.clamp(min=0.05), hard=False, dim=-1)
        filter_type = type_logits.argmax(dim=-1)
        type_probs_for_params = type_probs

        # FiLM conditioning: modulate trunk features by predicted type
        film_gamma = torch.einsum('bnt,td->bnd', type_probs, self.type_film_gamma)
        film_beta = torch.einsum('bnt,td->bnd', type_probs, self.type_film_beta)
        trunk_filmed = trunk_out * (1.0 + film_gamma) + film_beta

        # ── Final frequency computation ───────────────────────────────────
        if type_mel_context is not None and self.freq_context_proj is not None:
            freq_hidden = self.freq_context_proj(
                torch.cat([trunk_filmed, type_mel_context], dim=-1)
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

        # ── Q computation ─────────────────────────────────────────────────
        q_log = self.q_head(trunk_filmed).squeeze(-1)
        q_log = ste_clamp(q_log, math.log(0.1), math.log(10.0))
        q = torch.exp(q_log)

        # ── Hybrid gain: interpolate H_db at predicted frequency ──────────
        # Convert predicted freq (Hz) to continuous bin index
        freq_clamped = freq.clamp(min=1.0, max=self.sample_rate / 2.0 - 1.0)
        bin_continuous = freq_clamped / (self.sample_rate / 2.0) * (self.n_fft_bins - 1)
        bin_floor = bin_continuous.long().clamp(0, self.n_fft_bins - 2)
        bin_ceil = bin_floor + 1
        weight = (bin_continuous - bin_floor.float()).clamp(0.0, 1.0)  # (B, N)

        # Gather H_db values at floor and ceil bins for each band (per-band dim=2)
        B = h_db_pred.shape[0]
        idx_floor = bin_floor.unsqueeze(-1)  # (B, N, 1)
        idx_ceil = bin_ceil.unsqueeze(-1)    # (B, N, 1)
        hdb_floor = h_db_pred.gather(2, idx_floor).squeeze(-1)  # (B, N)
        hdb_ceil = h_db_pred.gather(2, idx_ceil).squeeze(-1)    # (B, N)
        # Differentiable interpolation
        gain_from_hdb = (1.0 - weight) * hdb_floor + weight * hdb_ceil  # (B, N)

        # Learned mix: sigmoid(alpha) * gain_hdb + (1-sigmoid(alpha)) * gain_raw
        mix = self.gain_mix_value.clamp(0.0, 1.0)
        gain_db_mixed = mix * gain_from_hdb + (1.0 - mix) * gain_db_raw

        if hard_types:
            # Inference: hard-zero HP/LP gain via argmax type mask.
            gain_db = gain_db_mixed * self.gainful_type_mask[filter_type]
        else:
            # Training: no soft mask — zero-gain for HP/LP handled by lambda_gain_zero
            gain_db = gain_db_mixed

        if return_aux:
            aux = {
                "mel_profile_centered": mel_profile_centered,
                "gain_aux_summary": (
                    gain_aux.mean(dim=1) if gain_aux is not None else None
                ),
                "shelf_bias": shelf_bias,
                "shelf_attention": shelf_attention,
                "h_db_pred": h_db_pred,
                "band_embedding": trunk_out,
                "hier_aux": hier_aux,
            }
            return gain_db, freq, q, type_logits, type_probs, filter_type, aux

        return gain_db, freq, q, type_logits, type_probs, filter_type
class ParameterSubHead(nn.Module):
    """
    Sub-head predicting gain, freq, Q for a specific group of filter types.
    """
    def __init__(self, input_dim, num_bands, gain_enabled=True, q_min=0.05, q_max=15.0):
        super().__init__()
        self.num_bands = num_bands
        self.gain_enabled = gain_enabled
        self.q_min = q_min
        self.q_max = q_max

        if gain_enabled:
            self.gain_head = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Linear(64, 1)
            )

        self.freq_head = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        self.q_head = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        """
        x: (B, num_bands, input_dim)
        Returns: gain_db, freq_unit, q
        """
        if self.gain_enabled:
            gain_raw = self.gain_head(x).squeeze(-1)
            gain_db_raw = ste_clamp(gain_raw * 24.0, -24.0, 24.0)
        else:
            gain_db_raw = x.new_zeros(x.shape[0], self.num_bands)

        freq_unit = torch.sigmoid(self.freq_head(x).squeeze(-1))

        q_log = self.q_head(x).squeeze(-1)
        q_log = ste_clamp(q_log, math.log(self.q_min), math.log(self.q_max))
        q = torch.exp(q_log)

        return gain_db_raw, freq_unit, q


class TypeGroupedParameterHead(nn.Module):
    """
    Two-stage parameter head that classifies type first, then routes to
    specialized sub-heads (peaking, shelf, pass) for parameter prediction.
    """
    def __init__(
        self,
        embedding_dim,
        num_bands=5,
        num_filter_types=NUM_FILTER_TYPES,
        n_mels=0,
        type_conditioned_frequency=True,
        n_shelf_bands=16,
        n_fft=2048,
        sample_rate=44100,
        dsp_cascade=None,
    ):
        super().__init__()
        self.num_bands = num_bands
        self.num_filter_types = num_filter_types
        self.n_mels = n_mels
        self.type_conditioned_frequency = type_conditioned_frequency
        self.n_shelf_bands = n_shelf_bands
        self.n_fft_bins = n_fft // 2 + 1
        self.sample_rate = sample_rate
        self._dsp_cascade = dsp_cascade

        self.log_f_min = math.log(20.0)
        self.log_f_max = math.log(20000.0)

        self.register_buffer("gumbel_tau", torch.tensor(1.0))

        # Trunk for base band embeddings
        self.trunk = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, num_bands * 64),
        )

        # 1. Type Classification Stage (Stage 1)
        # -----------------------------------------------------
        type_input_dim = 64

        # Optional mel context for type
        self.type_mel_proj = None
        if n_mels > 0:
            self.type_mel_proj = nn.Linear(n_mels, 32)
            type_input_dim += 32

        self.type_classifier = nn.Sequential(
            nn.Linear(type_input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, num_filter_types)
        )

        # 2. Parameter Stage (Stage 2)
        # -----------------------------------------------------
        # FiLM conditioning
        self.type_film_gamma = nn.Parameter(torch.randn(num_filter_types, 64) * 0.02)
        self.type_film_beta = nn.Parameter(torch.randn(num_filter_types, 64) * 0.02)

        # Sub-heads for specific parameter groups
        self.peaking_head = ParameterSubHead(64, num_bands, gain_enabled=True, q_min=0.05, q_max=15.0)
        self.shelf_head = ParameterSubHead(64, num_bands, gain_enabled=True, q_min=0.3, q_max=3.0)
        self.pass_head = ParameterSubHead(64, num_bands, gain_enabled=False, q_min=0.1, q_max=1.0)

        # Still keep h_db_pred for spectral loss, predicting per-band magnitude responses
        self.h_db_head = nn.Sequential(
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, self.n_fft_bins)
        )

        self.register_buffer("gain_mix_value", torch.tensor(0.5))

        self.register_buffer(
            "type_log_f_min",
            torch.tensor([bounds[0] for bounds in FILTER_LOG_FREQ_BOUNDS], dtype=torch.float32)
        )
        self.register_buffer(
            "type_log_f_max",
            torch.tensor([bounds[1] for bounds in FILTER_LOG_FREQ_BOUNDS], dtype=torch.float32)
        )
        self.register_buffer(
            "gainful_type_mask",
            torch.tensor(FILTER_GAINFUL_MASK, dtype=torch.float32)
        )

    def _center_mel_profile(self, mel_profile):
        if mel_profile is None:
            return None
        return mel_profile - mel_profile.mean(dim=-1, keepdim=True)

    def forward(
        self,
        embedding,
        mel_profile=None,
        hard_types=False,
        return_aux=False,
    ):
        trunk_out = self.trunk(embedding)  # (B, num_bands * 64)
        trunk_out = trunk_out.view(-1, self.num_bands, 64)

        # ── STAGE 1: Type Classification ──────────────────────────────────────
        type_input = trunk_out
        mel_profile_centered = self._center_mel_profile(mel_profile)

        if self.type_mel_proj is not None and mel_profile_centered is not None:
            # Simple global context
            global_mel = self.type_mel_proj(mel_profile_centered.unsqueeze(1))
            global_mel = global_mel.expand(-1, self.num_bands, -1)
            type_input = torch.cat([trunk_out, global_mel], dim=-1)

        type_logits = self.type_classifier(type_input)
        type_probs = F.gumbel_softmax(type_logits, tau=self.gumbel_tau.clamp(min=0.05), hard=False, dim=-1)
        filter_type = type_logits.argmax(dim=-1)

        # ── STAGE 2: Parameter Prediction ─────────────────────────────────────
        # Modulate trunk features by predicted type (soft for training, hard for inference)
        probs_for_film = F.one_hot(filter_type, self.num_filter_types).float() if hard_types else type_probs

        film_gamma = torch.einsum('bnt,td->bnd', probs_for_film, self.type_film_gamma)
        film_beta = torch.einsum('bnt,td->bnd', probs_for_film, self.type_film_beta)
        trunk_filmed = trunk_out * (1.0 + film_gamma) + film_beta

        # Predict from each sub-head
        gain_peaking, freq_unit_peaking, q_peaking = self.peaking_head(trunk_filmed)
        gain_shelf, freq_unit_shelf, q_shelf = self.shelf_head(trunk_filmed)
        gain_pass, freq_unit_pass, q_pass = self.pass_head(trunk_filmed)

        # Group probabilities
        prob_peaking = probs_for_film[..., FILTER_PEAKING:FILTER_PEAKING+1]  # (B, N, 1)
        prob_shelf = probs_for_film[..., FILTER_LOWSHELF:FILTER_HIGHSHELF+1].sum(dim=-1, keepdim=True)  # (B, N, 1)
        prob_pass = probs_for_film[..., FILTER_HIGHPASS:FILTER_LOWPASS+1].sum(dim=-1, keepdim=True)  # (B, N, 1)

        # Soft mixture of outputs
        gain_db_raw = (gain_peaking * prob_peaking.squeeze(-1) +
                       gain_shelf * prob_shelf.squeeze(-1) +
                       gain_pass * prob_pass.squeeze(-1))

        freq_unit = (freq_unit_peaking * prob_peaking.squeeze(-1) +
                     freq_unit_shelf * prob_shelf.squeeze(-1) +
                     freq_unit_pass * prob_pass.squeeze(-1))

        q = (q_peaking * prob_peaking.squeeze(-1) +
             q_shelf * prob_shelf.squeeze(-1) +
             q_pass * prob_pass.squeeze(-1))

        # ── Final Conversions & Output ────────────────────────────────────────
        # H_db prediction for spectral loss
        h_db_pred = self.h_db_head(trunk_out)  # (B, num_bands, n_fft_bins)

        if self.type_conditioned_frequency:
            if hard_types:
                log_f_min = self.type_log_f_min[filter_type]
                log_f_max = self.type_log_f_max[filter_type]
            else:
                log_f_min = (type_probs * self.type_log_f_min.view(1, 1, -1)).sum(dim=-1)
                log_f_max = (type_probs * self.type_log_f_max.view(1, 1, -1)).sum(dim=-1)
        else:
            log_f_min = torch.full_like(freq_unit, self.log_f_min)
            log_f_max = torch.full_like(freq_unit, self.log_f_max)

        log_freq = log_f_min + freq_unit * (log_f_max - log_f_min)
        freq = torch.exp(log_freq)

        # Hybrid gain interpolation
        freq_clamped = freq.clamp(min=1.0, max=self.sample_rate / 2.0 - 1.0)
        bin_continuous = freq_clamped / (self.sample_rate / 2.0) * (self.n_fft_bins - 1)
        bin_floor = bin_continuous.long().clamp(0, self.n_fft_bins - 2)
        bin_ceil = bin_floor + 1
        weight = (bin_continuous - bin_floor.float()).clamp(0.0, 1.0)

        B = h_db_pred.shape[0]
        idx_floor = bin_floor.unsqueeze(-1)
        idx_ceil = bin_ceil.unsqueeze(-1)
        hdb_floor = h_db_pred.gather(2, idx_floor).squeeze(-1)
        hdb_ceil = h_db_pred.gather(2, idx_ceil).squeeze(-1)
        gain_from_hdb = (1.0 - weight) * hdb_floor + weight * hdb_ceil

        mix = self.gain_mix_value.clamp(0.0, 1.0)
        gain_db_mixed = mix * gain_from_hdb + (1.0 - mix) * gain_db_raw

        if hard_types:
            gain_db = gain_db_mixed * self.gainful_type_mask[filter_type]
        else:
            gain_db = gain_db_mixed

        if return_aux:
            aux = {
                "mel_profile_centered": mel_profile_centered,
                "h_db_pred": h_db_pred,
                "band_embedding": trunk_out,
                "gain_aux_summary": None,
                "shelf_bias": type_logits.new_zeros(type_logits.shape[0], self.num_bands, 2),
                "shelf_attention": type_logits.new_zeros(type_logits.shape[0], self.num_bands, self.n_mels)
            }
            return gain_db, freq, q, type_logits, type_probs, filter_type, aux

        return gain_db, freq, q, type_logits, type_probs, filter_type
