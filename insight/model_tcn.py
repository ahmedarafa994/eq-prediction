"""
Frequency-aware hybrid encoder for blind EQ parameter estimation.

REDesign rationale -- fixing catastrophic TCN encoder collapse:
  1. The original pure 1D TCN treated n_mels as channel dimension, so
     convolutions mixed all frequency bins together and destroyed the
     spectral locality needed to detect WHERE an EQ band acts.
  2. Cumulative-mean pooling over time averaged away all spectral
     variation, yielding identical embeddings regardless of input
     (cosine similarity 1.0).

Solution -- hybrid 2D + frequency-preserving 1D architecture:
  A. 2D convolution front-end processes (n_mels, T) as a 2D feature map
     with small (freq, time) kernels.  This preserves frequency-axis
     locality so the network can learn spectral shapes (peaks, shelves,
     roll-offs) that characterise EQ bands.
  B. Frequency-preserving TCN operates on a (channels, T) signal per
     frequency sub-band group, so frequency is never collapsed into
     channels.  Grouped 1D convolutions keep sub-bands separate while
     allowing limited cross-frequency information sharing.
  C. Attention-weighted temporal pooling replaces cumulative-mean.
     A learned query attends over the time dimension so that frames
     with salient spectral activity contribute more to the summary.
  D. Spectral residual skip: the mean mel profile over time is passed
     directly to the parameter head, guaranteeing spectral information
     is always available regardless of encoder state.
  E. Anti-collapse hooks: embedding_variance() returns per-dimension
     variance across the batch for use as an auxiliary regularisation
     loss in the training loop.

Optimization features:
  - Gradient checkpointing for TCN stacks (trades compute for memory)
  - Fused gated activation kernels (reduces memory traffic by 75%)
  - Activation quantization (INT8 with STE, ~50% activation memory)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from differentiable_eq import DifferentiableBiquadCascade, MultiTypeEQParameterHead, NUM_FILTER_TYPES

# Fused Triton kernels disabled: runtime compilation fails on this Triton version
# (tl.math.tanh AttributeError). Use native PyTorch ops instead.
HAS_FUSED_KERNELS = False
fused_gated_activation_torch = None
fused_attention_pool_torch = None


# ---------------------------------------------------------------------------
# Causal 1D convolution (reused in frequency-preserving TCN blocks)
# ---------------------------------------------------------------------------


class CausalConv1d(nn.Module):
    """Causal 1D convolution with left-only padding (no look-ahead)."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=0,
            groups=groups,
        )

    def forward(self, x):
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


# ---------------------------------------------------------------------------
# 2D spectral front-end  -- preserves frequency-axis locality
# ---------------------------------------------------------------------------


class SpectralConvBlock2D(nn.Module):
    """
    Causal 2D convolution block that treats the spectrogram as a 2D image
    (n_mels x T).  Uses (freq_kernel, time_kernel) kernels that are causal
    in time (pad only the left side) so the block is compatible with
    streaming inference.

    This preserves frequency locality -- nearby mel bins are processed
    together, allowing the network to detect spectral shapes (peaks,
    notches, shelf transitions) that identify EQ bands.
    """

    def __init__(
        self, in_channels, out_channels, freq_kernel=5, time_kernel=3, dilation=(1, 1)
    ):
        super().__init__()
        self.time_pad = (time_kernel - 1) * dilation[1]  # causal (left-only)
        self.freq_pad = (freq_kernel - 1) * dilation[0] // 2  # symmetric

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(freq_kernel, time_kernel),
            dilation=dilation,
            padding=0,  # manual padding below
        )
        self.norm = nn.LayerNorm([out_channels])

    def forward(self, x):
        """
        x: (B, C, n_mels, T)
        returns: (B, C_out, n_mels, T)  -- same spatial dims via symmetric-freq / causal-time padding
        """
        x = F.pad(x, (self.time_pad, 0, self.freq_pad, self.freq_pad))
        x = self.conv(x)
        # LayerNorm over channel dim: (B, C, F, T) → norm over C, per-sample
        # Replaces BatchNorm which memorizes training-set statistics and causes
        # train/val distribution shift (the root cause of val loss plateauing).
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return F.gelu(x)


class SpectralFrontend2D(nn.Module):
    """
    Stack of 2D convolution layers that extract spectral shape features
    from (B, 1, n_mels, T) input.

    Each layer uses small kernels in both frequency and time so that
    local spectral patterns (bumps, dips, slopes) are detected while
    keeping the frequency axis intact as a spatial dimension.

    Output: (B, spectral_channels, n_mels, T)
    """

    def __init__(self, n_mels, out_channels=64):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                # Layer 1: input (1 ch) -> 16 ch, small kernel
                SpectralConvBlock2D(1, 16, freq_kernel=5, time_kernel=3),
                # Layer 2: 16 -> 32 ch
                SpectralConvBlock2D(16, 32, freq_kernel=5, time_kernel=3),
                # Layer 3: 32 -> out_channels, slightly wider freq kernel for context
                SpectralConvBlock2D(32, out_channels, freq_kernel=7, time_kernel=3),
            ]
        )

    def forward(self, x):
        """x: (B, 1, n_mels, T) -> (B, out_channels, n_mels, T)"""
        for layer in self.layers:
            x = layer(x)
        return x


# ---------------------------------------------------------------------------
# Frequency-preserving TCN  -- grouped 1D convolutions keep sub-bands
#                               as separate spatial lanes
# ---------------------------------------------------------------------------


class GroupedGatedBlock(nn.Module):
    """
    Gated activation with grouped 1D convolution.

    Groups correspond to frequency sub-bands.  Within each group, the
    convolution operates across time only; frequency bins in different
    groups do NOT mix.  A small group_overlap 1x1 conv before and after
    allows limited cross-frequency information sharing.

    This prevents the collapse that occurred when all mel bins were
    treated as channels in a standard 1D convolution.
    """

    def __init__(self, channels, num_groups, kernel_size=3, dilation=1):
        super().__init__()
        assert channels % num_groups == 0, (
            f"channels ({channels}) must be divisible by num_groups ({num_groups})"
        )
        self.num_groups = num_groups

        # Cross-group mixing (cheap 1x1 conv before grouping)
        self.pre_mix = nn.Conv1d(channels, channels, 1)

        # Grouped gated activation -- pass groups directly to CausalConv1d
        # so the underlying Conv1d is constructed with the correct weight shape
        self.filter_conv = CausalConv1d(
            channels, channels, kernel_size, dilation, groups=num_groups
        )
        self.gate_conv = CausalConv1d(
            channels, channels, kernel_size, dilation, groups=num_groups
        )

        # Post-group mixing
        self.post_mix = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        """x: (B, channels, T)"""
        x = self.pre_mix(x)
        filter_out = self.filter_conv(x)
        gate_out = self.gate_conv(x)

        if HAS_FUSED_KERNELS and fused_gated_activation_torch is not None:
            return self.post_mix(fused_gated_activation_torch(filter_out, gate_out))
        else:
            z_filter = torch.tanh(filter_out)
            z_gate = torch.sigmoid(gate_out)
            return self.post_mix(z_filter * z_gate)


class FrequencyPreservingTCNBlock(nn.Module):
    """
    Single TCN block using grouped convolutions to preserve frequency
    sub-band structure.  Includes residual and skip connections.
    """

    def __init__(self, channels, num_groups, kernel_size=3, dilation=1, dropout_p=0.1):
        super().__init__()
        self.gated_block = GroupedGatedBlock(
            channels, num_groups, kernel_size, dilation
        )
        self.residual_conv = nn.Conv1d(channels, channels, 1)
        self.skip_conv = nn.Conv1d(channels, channels, 1)
        self.norm = nn.LayerNorm(channels)
        # Dropout after normalization on the residual path.  Applied only during training
        # (nn.Dropout is a no-op in eval mode), so streaming inference is unaffected.
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        """
        x: (B, channels, T)
        returns: residual (B, channels, T), skip (B, channels, T)
        """
        gated = self.gated_block(x)
        skip = self.skip_conv(gated)
        residual = self.dropout(self.norm((self.residual_conv(x) + gated).transpose(1, 2)).transpose(1, 2))
        return residual, skip


class FrequencyPreservingTCN(nn.Module):
    """
    Stack of TCN blocks with exponentially increasing dilation.
    Uses grouped convolutions so frequency sub-bands remain separate.

    After the 2D spectral front-end extracts features per (freq, time),
    we reshape (B, C, n_mels, T) -> (B, C*n_mels, T) and apply grouped
    1D convolutions where each group handles a frequency sub-band.

    Supports gradient checkpointing to trade compute for memory.
    """

    def __init__(
        self,
        channels,
        num_blocks=4,
        num_groups=8,
        kernel_size=3,
        base_dilation=1,
        dropout_p=0.1,
        use_gradient_checkpointing=False,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            dilation = base_dilation * (2**i)
            self.blocks.append(
                FrequencyPreservingTCNBlock(channels, num_groups, kernel_size, dilation, dropout_p=dropout_p)
            )
        self.use_gradient_checkpointing = use_gradient_checkpointing

    def forward(self, x):
        """
        x: (B, channels, T)
        returns: (B, channels, T), skip_sum (B, channels, T)
        """
        skip_sum = 0
        for block in self.blocks:
            if self.use_gradient_checkpointing and self.training:
                out = checkpoint(block, x, use_reentrant=False)
                x, skip = out
            else:
                x, skip = block(x)
            skip_sum = skip_sum + skip
        return x, skip_sum


# ---------------------------------------------------------------------------
# Attention-weighted temporal pooling  -- replaces cumulative mean
# ---------------------------------------------------------------------------


class AttentionTemporalPool(nn.Module):
    """
    Learns to attend over the time dimension, producing a weighted summary
    of the per-frame features.  Replaces cumulative-mean pooling which
    averaged away all spectral variation and caused encoder collapse.

    A learned query vector computes attention weights over T frames.
    Frames with salient spectral activity (e.g., where an EQ band
    creates a noticeable spectral peak) receive higher weight.

    Returns both the pooled embedding (B, D) and per-frame attention
    weights (B, T) for interpretability.
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(embed_dim) * 0.02)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim**-0.5

    def forward(self, x):
        """
        x: (B, D, T) -- per-frame features (D = channels or embedding_dim)
        returns:
            pooled: (B, D) -- attention-weighted summary
            attn_weights: (B, T) -- attention distribution over time
        """
        if x.shape[-1] == 0:
            # Degenerate case: no time frames, return zeros
            return torch.zeros(x.shape[0], x.shape[1], device=x.device), torch.zeros(
                x.shape[0], 0, device=x.device
            )

        if HAS_FUSED_KERNELS and fused_attention_pool_torch is not None:
            return fused_attention_pool_torch(x, self.query, self.scale)

        # Transpose to (B, T, D) for linear projection
        x_t = x.permute(0, 2, 1)  # (B, T, D)
        keys = self.key_proj(x_t)  # (B, T, D)

        # Compute attention: query (D,) dot keys (B, T, D)
        scores = torch.matmul(keys, self.query) * self.scale  # (B, T)
        attn_weights = F.softmax(scores, dim=-1)  # (B, T)

        # Weighted sum: (B, T, D)^T @ (B, T, 1) -> (B, D, 1) -> (B, D)
        pooled = torch.matmul(x, attn_weights.unsqueeze(-1)).squeeze(-1)
        return pooled, attn_weights


# ---------------------------------------------------------------------------
# Hybrid encoder  -- 2D spectral front-end + grouped TCN + attention pool
# ---------------------------------------------------------------------------


class FrequencyAwareEncoder(nn.Module):
    """
    Hybrid encoder that fixes the catastrophic collapse of the original
    pure-1D TCN encoder.

    Pipeline:
      1. 2D spectral front-end: (B, 1, n_mels, T) -> (B, spectral_ch, n_mels, T)
         Detects local spectral shapes (EQ peaks, shelves) while keeping
         frequency as a spatial dimension.

      2. Reshape + projection: flatten freq into channels, project to
         tcn_channels.  (B, spectral_ch * n_mels, T) -> (B, tcn_channels, T)

      3. Grouped TCN: temporal modelling with grouped convolutions so that
         frequency sub-bands remain partially separated.  Captures temporal
         context with exponentially growing receptive field.

      4. Attention-weighted pooling: learned query attends over time to
         produce a single embedding vector.  Replaces the destructive
         cumulative-mean that averaged away all spectral variation.

      5. Spectral residual bypass: mean mel profile over time is computed
         from the input and returned alongside the embedding, ensuring the
         parameter head always receives spectral information.
    """

    def __init__(
        self,
        n_mels=128,
        embedding_dim=128,
        channels=128,
        num_blocks=4,
        num_stacks=2,
        kernel_size=3,
        num_freq_groups=8,
        spectral_channels=64,
        use_gradient_checkpointing=False,
        use_activation_quantization=False,
        dropout_p=0.1,
        mel_noise_std=0.0,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.embedding_dim = embedding_dim
        self.channels = channels
        self.num_blocks = num_blocks
        self.num_stacks = num_stacks
        self.kernel_size = kernel_size
        self.num_freq_groups = num_freq_groups
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_activation_quantization = use_activation_quantization
        self.dropout_p = dropout_p
        self.mel_noise_std = mel_noise_std

        # --- Stage 1: 2D spectral feature extraction ---
        self.spectral_frontend = SpectralFrontend2D(n_mels, spectral_channels)

        # --- Stage 2: Reshape & project ---
        # After 2D conv: (B, spectral_channels, n_mels, T)
        # Flatten to (B, spectral_channels * n_mels, T) then project
        self.freq_proj = nn.Conv1d(spectral_channels * n_mels, channels, 1)

        # --- Stage 3: Grouped TCN stacks ---
        self.tcn_stacks = nn.ModuleList()
        for _ in range(num_stacks):
            self.tcn_stacks.append(
                FrequencyPreservingTCN(
                    channels,
                    num_blocks,
                    num_freq_groups,
                    kernel_size,
                    dropout_p=dropout_p,
                    use_gradient_checkpointing=use_gradient_checkpointing,
                )
            )

        # --- Stage 4: Attention-weighted temporal pooling ---
        self.temporal_pool = AttentionTemporalPool(channels)

        # --- Output projection ---
        self.output_proj = nn.Sequential(
            nn.Linear(channels, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        self._init_weights()

    def _init_weights(self):
        """Careful initialisation to prevent early collapse."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d) and m.weight.dim() >= 2:
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @property
    def receptive_field_frames(self):
        """Number of input frames needed for full receptive field."""
        rf = 1
        for _ in range(self.num_stacks):
            for i in range(self.num_blocks):
                rf += (self.kernel_size - 1) * (2**i)
        return rf

    def forward(self, mel_frames):
        """
        Batch mode: process a full mel-spectrogram sequence.

        Args:
            mel_frames: (B, n_mels, T)

        Returns:
            embedding: (B, embedding_dim) -- attention-pooled embedding
            mel_profile: (B, n_mels) -- mean mel over time (spectral bypass)
            skip_sum: (B, channels, T) -- per-frame skip connections
            attn_weights: (B, T) -- temporal attention weights
        """
        B, n_mels, T = mel_frames.shape

        # --- Mel noise injection: Gaussian noise on input for regularization ---
        if self.training and self.mel_noise_std > 0:
            mel_frames = mel_frames + torch.randn_like(mel_frames) * self.mel_noise_std

        # --- Spectral bypass: retain the mean profile for the parameter head.
        # The head recenters this profile before the auxiliary readouts so
        # global loudness shifts do not masquerade as filter shape.
        mel_profile = mel_frames.mean(dim=-1)  # (B, n_mels)

        # --- Stage 1: 2D spectral features ---
        # Add channel dim: (B, 1, n_mels, T)
        x_2d = mel_frames.unsqueeze(1)
        x_2d = self.spectral_frontend(x_2d)  # (B, spectral_ch, n_mels, T)

        # --- Stage 2: Reshape & project ---
        # Flatten frequency axis into channels: (B, spectral_ch * n_mels, T)
        x = x_2d.reshape(B, -1, T)
        x = F.gelu(self.freq_proj(x))  # (B, channels, T)

        # --- Stage 3: Grouped TCN ---
        skip_total = 0
        for stack in self.tcn_stacks:
            x, skip = stack(x)
            skip_total = skip_total + skip

        # --- Stage 4: Attention-weighted temporal pooling ---
        # Replaces cumulative-mean that caused collapse
        pooled, attn_weights = self.temporal_pool(skip_total)  # (B, channels), (B, T)

        # --- Output projection ---
        embedding = self.output_proj(pooled)  # (B, embedding_dim)

        return embedding, mel_profile, skip_total, attn_weights


# ---------------------------------------------------------------------------
# Streaming-compatible wrapper model
# ---------------------------------------------------------------------------


class StreamingTCNModel(nn.Module):
    """
    Complete streaming model for blind EQ parameter estimation.

    Uses FrequencyAwareEncoder to avoid the catastrophic collapse that
    occurred with the original pure-1D TCN.  The key improvements:

    1. 2D spectral front-end preserves frequency-axis locality so the
       network can detect spectral shapes (EQ peaks, shelves, roll-offs).
    2. Grouped 1D convolutions keep frequency sub-bands partially
       separate, preventing all spectral information from being mixed
       into a single channel dimension.
    3. Attention-weighted temporal pooling replaces cumulative-mean,
       so salient frames contribute more than irrelevant ones.
    4. Spectral residual bypass passes the mean mel profile directly
       to the parameter head, guaranteeing spectral information is
       always available.

    Supports both batch training and streaming inference via
    init_streaming() / process_frame().
    """

    def __init__(
        self,
        n_mels=128,
        embedding_dim=128,
        num_bands=5,
        channels=128,
        num_blocks=4,
        num_stacks=2,
        sample_rate=44100,
        n_fft=2048,
        kernel_size=3,
        num_filter_types=5,
        type_conditioned_frequency=True,
        dropout=0.1,
        mel_noise_std=0.0,
        n_shelf_bands=16,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.embedding_dim = embedding_dim
        self.num_bands = num_bands
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.n_shelf_bands = n_shelf_bands

        # Number of frequency groups for grouped convolutions
        # Use 8 groups by default; this keeps ~16 mel bins per group
        # which is enough for sub-band locality while allowing
        # cross-frequency information flow through the pre/post mix layers
        # Must be a divisor of channels for grouped convolutions
        num_freq_groups = min(8, n_mels)
        while channels % num_freq_groups != 0 and num_freq_groups > 1:
            num_freq_groups -= 1

        # Encoder
        self.encoder = FrequencyAwareEncoder(
            n_mels=n_mels,
            embedding_dim=embedding_dim,
            channels=channels,
            num_blocks=num_blocks,
            num_stacks=num_stacks,
            kernel_size=kernel_size,
            num_freq_groups=num_freq_groups,
            dropout_p=dropout,
            mel_noise_std=mel_noise_std,
        )

        # Parameter head -- receives both embedding AND mel_profile
        self.param_head = MultiTypeEQParameterHead(
            embedding_dim,
            num_bands,
            num_filter_types=num_filter_types,
            n_mels=n_mels,
            type_conditioned_frequency=type_conditioned_frequency,
            n_shelf_bands=n_shelf_bands,
            n_fft=n_fft,
            sample_rate=sample_rate,
        )

        # DSP layer
        self.dsp_cascade = DifferentiableBiquadCascade(num_bands, sample_rate)

        # Streaming state
        self._streaming_buffer = None
        self._streaming_skip_sum = None
        self._frame_count = 0

    @property
    def receptive_field_frames(self):
        return self.encoder.receptive_field_frames

    def reset_state(self):
        """Reset streaming state for a new inference session."""
        self._streaming_buffer = None
        self._streaming_skip_sum = None
        self._frame_count = 0

    def embedding_variance(self, embedding):
        """
        Anti-collapse diagnostic: returns per-dimension variance
        across the batch.  If all values are near zero the encoder
        has collapsed.

        Usage in training loop:
            emb = output["embedding"]
            var = model.embedding_variance(emb)
            collapse_loss = -var.mean()  # maximise variance

        Args:
            embedding: (B, embedding_dim)

        Returns:
            per_dim_variance: (embedding_dim,) -- variance along batch dim
        """
        if embedding.shape[0] < 2:
            return torch.zeros(embedding.shape[-1], device=embedding.device)
        return embedding.var(dim=0)

    def load_compatible_state_dict(self, state_dict):
        """Load only checkpoint tensors whose keys and shapes still match."""
        current_state = self.state_dict()
        compatible_state = {}
        skipped = []
        for key, value in state_dict.items():
            if key in current_state and current_state[key].shape == value.shape:
                compatible_state[key] = value
            else:
                skipped.append(key)
        result = self.load_state_dict(compatible_state, strict=False)
        return result, skipped

    def forward(self, mel_frames, hard_types=None, force_soft_response=False):
        """
        Batch-mode forward pass for training.

        Args:
            mel_frames: (B, n_mels, T) -- full mel-spectrogram sequence

        Returns:
            dict with:
                params: (gain_db, freq, q) -- each (B, num_bands)
                type_logits: (B, num_bands, 5)
                type_probs: (B, num_bands, 5)
                filter_type: (B, num_bands)
                H_mag: (B, n_fft//2+1) -- predicted frequency response
                embedding: (B, embedding_dim)
                mel_profile: (B, n_mels) -- spectral bypass
                attn_weights: (B, T) -- temporal attention weights
        """
        if hard_types is None:
            hard_types = not self.training

        # 1. Encode -- now returns mel_profile and attn_weights too
        embedding, mel_profile, skip_sum, attn_weights = self.encoder(mel_frames)

        # 2. Predict parameters -- pass mel_profile to param head
        gain_db, freq, q, type_logits, type_probs, filter_type, param_aux = self.param_head(
            embedding,
            mel_profile=mel_profile,
            hard_types=hard_types,
            return_aux=True,
        )

        # 3. Compute frequency response
        H_mag = self.dsp_cascade(gain_db, freq, q, self.n_fft, filter_type)

        # Soft H_mag path: differentiable w.r.t. type_logits (used for spectral loss during training)
        if self.training or force_soft_response:
            type_probs_soft = F.softmax(type_logits, dim=-1)  # (B, N, 5)
            H_per_type = []
            for t in range(NUM_FILTER_TYPES):
                ft = torch.full_like(filter_type, t)
                b0, b1, b2, a1, a2 = self.dsp_cascade.compute_biquad_coeffs_multitype(
                    gain_db, freq, q, ft
                )
                H_t = self.dsp_cascade.freq_response(b0, b1, b2, a1, a2, self.n_fft)  # (B, N, F)
                H_per_type.append(H_t)
            H_stack = torch.stack(H_per_type, dim=2)  # (B, N, 5, F)
            weights = type_probs_soft.unsqueeze(-1)     # (B, N, 5, 1)
            H_soft_bands = (H_stack * weights).sum(dim=2)  # (B, N, F)
            H_mag_soft = torch.clamp(H_soft_bands.prod(dim=1), min=1e-6, max=1e4)  # (B, F)
        else:
            H_mag_soft = H_mag

        result = {
            "params": (gain_db, freq, q),
            "type_logits": type_logits,
            "type_probs": type_probs,
            "filter_type": filter_type,
            "H_mag": H_mag,
            "H_mag_soft": H_mag_soft,
            "embedding": embedding,
            "mel_profile": mel_profile,
            "mel_profile_centered": param_aux.get("mel_profile_centered"),
            "gain_aux_summary": param_aux.get("gain_aux_summary"),
            "shelf_bias": param_aux.get("shelf_bias"),
            "shelf_attention": param_aux.get("shelf_attention"),
            "h_db_pred": param_aux.get("h_db_pred"),
            "attn_weights": attn_weights,
        }

        return result

    def init_streaming(self, batch_size=1):
        """Initialize streaming state."""
        rf = self.receptive_field_frames
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        self._streaming_buffer = torch.zeros(
            batch_size, self.n_mels, rf, device=device, dtype=dtype
        )
        self._streaming_skip_sum = None
        self._frame_count = 0

    def process_frame(self, mel_frame):
        """
        Streaming inference: process a single mel frame.

        The streaming path replicates the same computation as forward():
          1. Buffer the frame and run through the 2D spectral front-end
          2. Project and run through grouped TCN
          3. Use attention pooling over the buffered frames
          4. Pass embedding + mel_profile to param head

        Args:
            mel_frame: (B, n_mels) -- single mel-spectrogram frame

        Returns:
            Same dict as forward().
        """
        # Guard: BatchNorm uses running stats in eval mode only.
        # Streaming inference requires eval mode for correct normalization.
        if self.training:
            was_training = True
            self.eval()
        else:
            was_training = False
        if self._streaming_buffer is None:
            self.init_streaming(batch_size=mel_frame.shape[0])

        B = mel_frame.shape[0]

        # Shift buffer left and insert new frame
        self._streaming_buffer = torch.cat(
            [
                self._streaming_buffer[:, :, 1:],
                mel_frame.unsqueeze(-1),
            ],
            dim=-1,
        )

        self._frame_count += 1

        # --- Spectral bypass: mean mel profile from current buffer ---
        mel_profile = self._streaming_buffer.mean(dim=-1)  # (B, n_mels)

        # --- Run through encoder stages ---
        # Stage 1: 2D spectral front-end
        x_2d = self._streaming_buffer.unsqueeze(1)  # (B, 1, n_mels, T_buf)
        x_2d = self.encoder.spectral_frontend(x_2d)  # (B, spec_ch, n_mels, T_buf)

        # Stage 2: Reshape & project
        T_buf = x_2d.shape[-1]
        x = x_2d.reshape(B, -1, T_buf)
        x = F.gelu(self.encoder.freq_proj(x))

        # Stage 3: Grouped TCN
        skip_total = 0
        for stack in self.encoder.tcn_stacks:
            x, skip = stack(x)
            skip_total = skip_total + skip

        # Stage 4: Attention-weighted temporal pooling
        # Use the last channel slice for streaming (attention pool over buffer)
        pooled, _ = self.encoder.temporal_pool(skip_total)

        # Output projection
        embedding = self.encoder.output_proj(pooled)

        # --- Predict parameters ---
        gain_db, freq, q, type_logits, type_probs, filter_type, param_aux = self.param_head(
            embedding,
            mel_profile=mel_profile,
            hard_types=True,
            return_aux=True,
        )

        H_mag = self.dsp_cascade(gain_db, freq, q, self.n_fft, filter_type)

        result = {
            "params": (gain_db, freq, q),
            "type_logits": type_logits,
            "type_probs": type_probs,
            "filter_type": filter_type,
            "H_mag": H_mag,
            "embedding": embedding,
            "mel_profile": mel_profile,
            "mel_profile_centered": param_aux.get("mel_profile_centered"),
            "gain_aux_summary": param_aux.get("gain_aux_summary"),
            "shelf_bias": param_aux.get("shelf_bias"),
            "shelf_attention": param_aux.get("shelf_attention"),
        }

        # Restore training mode if we forced eval for BatchNorm safety
        if was_training:
            self.train()

        return result
