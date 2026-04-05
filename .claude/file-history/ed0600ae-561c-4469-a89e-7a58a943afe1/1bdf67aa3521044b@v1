import torch
import torch.nn as nn
import torch.nn.functional as F


class STFTLoss(nn.Module):
    """
    Calculates the Spectral Convergence and Log-Magnitude STFT distances
    for a single FFT size.
    """

    def __init__(self, fft_size, hop_size, win_length):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length))

    def forward(self, x, y):
        # x, y: (Batch, Time)
        x_stft = torch.stft(
            x,
            self.fft_size,
            self.hop_size,
            self.win_length,
            window=self.window,
            return_complex=True,
        )
        y_stft = torch.stft(
            y,
            self.fft_size,
            self.hop_size,
            self.win_length,
            window=self.window,
            return_complex=True,
        )

        x_mag = torch.abs(x_stft)
        y_mag = torch.abs(y_stft)

        # Spectral Convergence
        sc_loss = torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")

        # Log STFT Magnitude Error
        log_loss = F.l1_loss(torch.log(x_mag + 1e-7), torch.log(y_mag + 1e-7))

        return sc_loss, log_loss


class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-Resolution STFT Loss module. Computes STFT loss across various
    frequency window resolutions to capture both transient and steady-state
    spectral discrepancies.
    """

    def __init__(
        self,
        fft_sizes=[1024, 2048, 512],
        hop_sizes=[120, 240, 50],
        win_lengths=[600, 1200, 240],
    ):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = nn.ModuleList()
        for fs, hs, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses.append(STFTLoss(fs, hs, wl))

    def forward(self, x, y):
        sc_loss = 0.0
        log_loss = 0.0
        for f in self.stft_losses:
            sc_l, log_l = f(x, y)
            sc_loss += sc_l
            log_loss += log_l

        return sc_loss / len(self.stft_losses), log_loss / len(self.stft_losses)


class EQLoss(nn.Module):
    """
    Combined loss for the EQ model.
    Weights standard MR-STFT alongside Parameter MSE (if ground truth params exist).
    """

    def __init__(self, lambda_audio=1.0, lambda_param=1.0):
        super().__init__()
        self.mr_stft = MultiResolutionSTFTLoss()
        self.lambda_audio = lambda_audio
        self.lambda_param = lambda_param
        self.param_loss_fn = nn.HuberLoss()

    def forward(self, pred_audio, target_audio, pred_params=None, target_params=None):
        sc_loss, log_loss = self.mr_stft(pred_audio, target_audio)
        audio_loss = sc_loss + log_loss

        total_loss = self.lambda_audio * audio_loss

        param_loss = torch.tensor(0.0, device=pred_audio.device)
        if pred_params is not None and target_params is not None:
            param_loss = self.param_loss_fn(pred_params, target_params)
            total_loss += self.lambda_param * param_loss

        return total_loss, audio_loss, param_loss


class FreqResponseLoss(nn.Module):
    """
    L1 distance between log-magnitude frequency responses in log-frequency domain.
    Compares predicted vs. ground-truth EQ magnitude response curves.
    """

    def __init__(self, n_fft=2048):
        super().__init__()
        self.n_fft = n_fft

    def forward(self, H_pred, H_target):
        log_pred = torch.log(H_pred + 1e-8)
        log_target = torch.log(H_target + 1e-8)
        return F.l1_loss(log_pred, log_target)


class EQParameterPriorLoss(nn.Module):
    """
    Regularization priors for EQ parameter estimates.
    - Gain: Laplace prior centered at 0 dB
    - Frequency: Spread penalty (encourages bands apart)
    - Q: Log-normal prior (penalize extreme Q values)
    """

    def __init__(self, num_bands=5):
        super().__init__()
        self.num_bands = num_bands

    def forward(self, gain_db, freq, q):
        gain_prior = torch.abs(gain_db).mean()

        freq_sorted, _ = torch.sort(freq, dim=-1)
        min_log_gap = torch.log(
            torch.clamp(freq_sorted[:, 1:] / (freq_sorted[:, :-1] + 1e-6), min=1.0)
        )
        freq_spread_penalty = -min_log_gap.mean()

        q_log = torch.log(torch.clamp(q, min=1e-6))
        q_prior = (q_log - torch.log(torch.tensor(1.0, device=q.device))).pow(2).mean()

        return gain_prior + 0.1 * freq_spread_penalty + 0.05 * q_prior


class CycleConsistencyLoss(nn.Module):
    """
    Self-supervised loss: apply estimated EQ to wet signal via inverse,
    then re-apply estimated EQ forward. Round-trip should preserve wet signal.
    L_cycle = MR-STFT(wet, apply_eq(remove_eq(wet, θ̂), θ̂))
    Operates on complex STFT magnitudes.
    """

    def __init__(self, fft_sizes=None, hop_sizes=None, win_lengths=None):
        super().__init__()
        if fft_sizes is None:
            fft_sizes = [1024, 2048, 512]
        if hop_sizes is None:
            hop_sizes = [120, 240, 50]
        if win_lengths is None:
            win_lengths = [600, 1200, 240]
        self.mr_stft = MultiResolutionSTFTLoss(fft_sizes, hop_sizes, win_lengths)

    def forward(self, wet_audio, roundtrip_audio):
        sc_loss, log_loss = self.mr_stft(wet_audio, roundtrip_audio)
        return sc_loss + log_loss


class CombinedIDSPLoss(nn.Module):
    """
    Full combined loss for the IDSP EQ Estimator.

    Components:
        L_freq     - Frequency response matching (supervised)
        L_param    - Parameter regression (supervised, Huber)
        L_cycle    - Cycle-consistency reconstruction (self-supervised)
        L_prior    - Parameter prior regularization

    L_total = λ_freq·L_freq + λ_param·L_param + λ_cycle·L_cycle + λ_prior·L_prior
    """

    def __init__(
        self,
        num_bands=5,
        n_fft=2048,
        lambda_freq=0.3,
        lambda_param=0.5,
        lambda_cycle=1.0,
        lambda_prior=0.1,
    ):
        super().__init__()
        self.freq_loss = FreqResponseLoss(n_fft=n_fft)
        self.param_loss = nn.HuberLoss()
        self.cycle_loss = CycleConsistencyLoss()
        self.prior_loss = EQParameterPriorLoss(num_bands=num_bands)
        self.lambda_freq = lambda_freq
        self.lambda_param = lambda_param
        self.lambda_cycle = lambda_cycle
        self.lambda_prior = lambda_prior

    def forward(
        self,
        H_pred=None,
        H_target=None,
        pred_params=None,
        target_params=None,
        wet_audio=None,
        roundtrip_audio=None,
        gain_db=None,
        freq=None,
        q=None,
    ):
        total_loss = torch.tensor(0.0)
        components = {}

        if H_pred is not None and H_target is not None:
            l_freq = self.lambda_freq * self.freq_loss(H_pred, H_target)
            total_loss = total_loss + l_freq
            components["freq"] = l_freq.detach()

        if pred_params is not None and target_params is not None:
            l_param = self.lambda_param * self.param_loss(pred_params, target_params)
            total_loss = total_loss + l_param
            components["param"] = l_param.detach()

        if wet_audio is not None and roundtrip_audio is not None:
            l_cycle = self.lambda_cycle * self.cycle_loss(wet_audio, roundtrip_audio)
            total_loss = total_loss + l_cycle
            components["cycle"] = l_cycle.detach()

        if gain_db is not None and freq is not None and q is not None:
            l_prior = self.lambda_prior * self.prior_loss(gain_db, freq, q)
            total_loss = total_loss + l_prior
            components["prior"] = l_prior.detach()

        return total_loss, components
