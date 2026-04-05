"""
Streaming-compatible TCN encoder for blind EQ parameter estimation.

Uses causal 1D convolutions with dilated temporal convolutional network (TCN)
architecture for real-time audio processing. Based on WaveNet-style gated
activations with residual and skip connections.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from differentiable_eq import DifferentiableBiquadCascade, MultiTypeEQParameterHead


class CausalConv1d(nn.Module):
    """Causal 1D convolution with left-only padding (no look-ahead)."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=0  # We handle padding manually
        )

    def forward(self, x):
        # Left-only padding for causality
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class GatedResidualBlock(nn.Module):
    """
    Gated activation block with residual and skip connections.
    Uses tanh * sigmoid gating (WaveNet-style).
    """

    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        self.filter_conv = CausalConv1d(channels, channels, kernel_size, dilation)
        self.gate_conv = CausalConv1d(channels, channels, kernel_size, dilation)

    def forward(self, x):
        z_filter = torch.tanh(self.filter_conv(x))
        z_gate = torch.sigmoid(self.gate_conv(x))
        return z_filter * z_gate


class TCNBlock(nn.Module):
    """
    Single TCN block: gated residual + 1x1 conv for skip/residual.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.gated_block = GatedResidualBlock(in_channels, kernel_size, dilation)

        # 1x1 convolutions for residual and skip connections
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()
        self.skip_conv = nn.Conv1d(in_channels, out_channels, 1)

        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        gated = self.gated_block(x)
        skip = self.skip_conv(gated)
        residual = self.residual_conv(x) + gated
        residual = self.bn(residual)
        return residual, skip


class TCNStack(nn.Module):
    """
    Stack of TCN blocks with exponentially increasing dilation.
    """

    def __init__(self, channels, num_blocks, kernel_size=3, base_dilation=1):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            dilation = base_dilation * (2 ** i)
            self.blocks.append(TCNBlock(channels, channels, kernel_size, dilation))

    def forward(self, x):
        skip_sum = 0
        for block in self.blocks:
            x, skip = block(x)
            skip_sum = skip_sum + skip
        return x, skip_sum


class CausalTCNEncoder(nn.Module):
    """
    Causal TCN encoder that processes mel-spectrogram frames sequentially.

    Architecture: input projection → 2 TCN stacks → cumulative mean of
    skip connections → fixed-size embedding.

    Supports both batch mode (full sequence) and streaming mode (frame-by-frame).
    """

    def __init__(self, n_mels=128, embedding_dim=128, channels=128,
                 num_blocks=4, num_stacks=2, kernel_size=3):
        super().__init__()
        self.n_mels = n_mels
        self.embedding_dim = embedding_dim
        self.channels = channels
        self.num_blocks = num_blocks
        self.num_stacks = num_stacks

        # Input projection: mel bins → channels
        self.input_proj = nn.Conv1d(n_mels, channels, 1)

        # TCN stacks
        self.stacks = nn.ModuleList()
        for s in range(num_stacks):
            self.stacks.append(TCNStack(channels, num_blocks, kernel_size))

        # Output projection
        self.output_proj = nn.Linear(channels, embedding_dim)

    @property
    def receptive_field_frames(self):
        """Number of input frames needed for full receptive field."""
        # Each block has kernel_size=3, dilation doubles
        # Receptive field per block = (kernel_size - 1) * dilation
        # Total = sum over all blocks and stacks
        rf = 1
        for _ in range(self.num_stacks):
            for i in range(self.num_blocks):
                rf += (3 - 1) * (2 ** i)
        return rf

    def forward(self, mel_frames):
        """
        Batch mode: process a sequence of mel frames.

        Args:
            mel_frames: (Batch, n_mels, T) — sequence of mel-spectrogram frames

        Returns:
            embedding: (Batch, embedding_dim) — aggregated embedding
            skip_sum: (Batch, channels, T) — per-frame skip connections
        """
        x = self.input_proj(mel_frames)

        skip_total = 0
        for stack in self.stacks:
            x, skip = stack(x)
            skip_total = skip_total + skip

        # Cumulative mean of skip connections over time
        # This gives a stable embedding that improves as more context arrives
        cumsum = skip_total.cumsum(dim=-1)
        counts = torch.arange(1, skip_total.shape[-1] + 1, device=skip_total.device).float()
        cummean = cumsum / counts.unsqueeze(0).unsqueeze(0)

        # Use the last frame's cumulative mean as the embedding
        last_frame = cummean[:, :, -1]  # (Batch, channels)
        embedding = self.output_proj(last_frame)  # (Batch, embedding_dim)

        return embedding, skip_total


class StreamingTCNModel(nn.Module):
    """
    Complete streaming model for blind EQ parameter estimation.

    Combines a causal TCN encoder with multi-type EQ parameter head and
    differentiable biquad cascade. Supports both batch training and
    streaming inference modes.
    """

    def __init__(self, n_mels=128, embedding_dim=128, num_bands=5,
                 channels=128, num_blocks=4, num_stacks=2,
                 sample_rate=44100, n_fft=2048):
        super().__init__()
        self.n_mels = n_mels
        self.embedding_dim = embedding_dim
        self.num_bands = num_bands
        self.sample_rate = sample_rate
        self.n_fft = n_fft

        # Encoder
        self.encoder = CausalTCNEncoder(
            n_mels=n_mels,
            embedding_dim=embedding_dim,
            channels=channels,
            num_blocks=num_blocks,
            num_stacks=num_stacks,
        )

        # Parameter head
        self.param_head = MultiTypeEQParameterHead(embedding_dim, num_bands)

        # DSP layer
        self.dsp_cascade = DifferentiableBiquadCascade(num_bands, sample_rate)

        # Streaming state
        self._streaming_buffer = None
        self._cumulative_skip_sum = None
        self._frame_count = 0

    @property
    def receptive_field_frames(self):
        return self.encoder.receptive_field_frames

    def reset_state(self):
        """Reset streaming state for a new inference session."""
        self._streaming_buffer = None
        self._cumulative_skip_sum = None
        self._frame_count = 0

    def forward(self, mel_frames):
        """
        Batch-mode forward pass for training.

        Args:
            mel_frames: (Batch, n_mels, T) — full mel-spectrogram sequence

        Returns:
            dict with:
                params: (gain_db, freq, q) — each (Batch, num_bands)
                type_logits: (Batch, num_bands, 5)
                type_probs: (Batch, num_bands, 5)
                filter_type: (Batch, num_bands)
                H_mag: (Batch, n_fft//2+1) — predicted frequency response
                embedding: (Batch, embedding_dim)
        """
        # 1. Encode
        embedding, skip_sum = self.encoder(mel_frames)

        # 2. Predict parameters
        gain_db, freq, q, type_logits, type_probs, filter_type = self.param_head(
            embedding, hard_types=not self.training
        )

        # 3. Compute frequency response
        if self.training:
            H_mag = self.dsp_cascade.forward_soft(gain_db, freq, q, type_probs, self.n_fft)
        else:
            H_mag = self.dsp_cascade(gain_db, freq, q, self.n_fft, filter_type)

        return {
            "params": (gain_db, freq, q),
            "type_logits": type_logits,
            "type_probs": type_probs,
            "filter_type": filter_type,
            "H_mag": H_mag,
            "embedding": embedding,
        }

    def init_streaming(self, batch_size=1):
        """Initialize streaming state."""
        rf = self.receptive_field_frames
        self._streaming_buffer = torch.zeros(batch_size, self.n_mels, rf)
        self._cumulative_skip_sum = None
        self._frame_count = 0

    def process_frame(self, mel_frame):
        """
        Streaming inference: process a single mel frame.

        Args:
            mel_frame: (Batch, n_mels) — single mel-spectrogram frame

        Returns:
            Same dict as forward(), with parameters estimated from
            cumulative context up to this frame.
        """
        if self._streaming_buffer is None:
            self.init_streaming(batch_size=mel_frame.shape[0])

        # Shift buffer left and insert new frame
        self._streaming_buffer = torch.cat([
            self._streaming_buffer[:, :, 1:],
            mel_frame.unsqueeze(-1)
        ], dim=-1)

        self._frame_count += 1

        # Run encoder on buffer
        x = self.encoder.input_proj(self._streaming_buffer)

        skip_total = 0
        for stack in self.encoder.stacks:
            x, skip = stack(x)
            skip_total = skip_total + skip

        # Cumulative skip sum (streaming version)
        if self._cumulative_skip_sum is None:
            self._cumulative_skip_sum = skip_total
        else:
            self._cumulative_skip_sum = self._cumulative_skip_sum + skip_total

        # Use running mean
        cummean = self._cumulative_skip_sum / self._frame_count
        last_frame = cummean[:, :, -1]
        embedding = self.encoder.output_proj(last_frame)

        # Predict parameters
        gain_db, freq, q, type_logits, type_probs, filter_type = self.param_head(
            embedding, hard_types=True
        )

        H_mag = self.dsp_cascade(gain_db, freq, q, self.n_fft, filter_type)

        return {
            "params": (gain_db, freq, q),
            "type_logits": type_logits,
            "type_probs": type_probs,
            "filter_type": filter_type,
            "H_mag": H_mag,
            "embedding": embedding,
        }
