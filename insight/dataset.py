"""
On-the-fly synthetic multi-type EQ dataset with optional precomputation.

Generates (dry_audio, wet_audio, params) tuples with random multi-type
parametric EQ applied using the differentiable biquad cascade.
No external audio files needed — uses synthetic signals.

Supports precompute mode to cache mel-spectrograms for faster training.
Supports lazy/streaming mode to generate samples on-the-fly without
caching, reducing RAM footprint for large datasets.
"""

import torch
import torch.utils.data as data
import numpy as np
import random
import math
import json
import gc
import sys
from pathlib import Path
from differentiable_eq import (
    DifferentiableBiquadCascade,
    FILTER_PEAKING,
    FILTER_LOWSHELF,
    FILTER_HIGHSHELF,
    FILTER_HIGHPASS,
    FILTER_LOWPASS,
    NUM_FILTER_TYPES,
)
from pipeline_utils import (
    PIPELINE_SCHEMA_VERSION,
    compute_metadata_signature,
    resolve_trusted_artifact_path,
    seeded_index_context,
    utc_now_iso,
    validate_precompute_memory,
    get_available_memory,
)

FILTER_NAMES = ["peaking", "lowshelf", "highshelf", "highpass", "lowpass"]
# AUDIT: V-07 — Default type weights must match config.yaml defaults (equal weighting)
DEFAULT_TYPE_WEIGHTS = [0.2, 0.2, 0.2, 0.2, 0.2]
# AUDIT: V-09 — Default signal type weights reflecting real-world audio content
DEFAULT_SIGNAL_TYPE_WEIGHTS = {
    "speech_like": 0.30,
    "harmonic": 0.25,
    "pink_noise": 0.15,
    "percussive": 0.15,
    "noise": 0.10,
    "sweep": 0.05,
}


class SyntheticEQDataset(data.Dataset):
    """
    Generates synthetic (dry_audio, wet_audio, params) tuples on-the-fly.
    Applies random multi-type parametric EQ using the DifferentiableBiquadCascade.

    Returns dict with wet_audio, dry_audio, gain, freq, q, filter_type tensors.
    """

    def __init__(
        self,
        num_bands=5,
        sample_rate=44100,
        duration=3.0,
        duration_range=None,  # e.g., (1.0, 5.0) for random duration
        n_fft=2048,
        size=50000,
        gain_range=(-24.0, 24.0),
        freq_range=(20.0, 20000.0),
        q_range=(0.1, 10.0),
        type_weights=None,
        hp_lp_gain_target="zero",
        signal_types=("noise", "pink_noise", "sweep", "harmonic", "speech_like", "percussive"),
        signal_type_weights=None,  # AUDIT: V-09 — Add configurable signal type weights
        gain_distribution="beta",  # AUDIT: V-06 — "uniform" or "beta" (default: beta for realistic distribution)
        augment=True,
        precompute_mels=False,
        n_mels=128,
        base_seed=42,
    ):
        self.num_bands = num_bands
        self.sample_rate = sample_rate
        self.duration = duration
        self.duration_range = duration_range  # (min, max) for random duration
        self.num_samples = int(duration * sample_rate)
        self.n_fft = n_fft
        self.hop_length = n_fft // 4
        self._size = size
        self.gain_range = gain_range
        self.freq_range = freq_range
        self.q_range = q_range
        self.type_weights = type_weights or DEFAULT_TYPE_WEIGHTS
        # AUDIT: V-07 — Validate type weights sum to 1.0 (within tolerance)
        weight_sum = sum(self.type_weights)
        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError(
                f"type_weights must sum to 1.0, got {weight_sum:.6f}. "
                f"Normalize weights or adjust values."
            )
        self.hp_lp_gain_target = hp_lp_gain_target
        if self.hp_lp_gain_target != "zero":
            raise ValueError(
                "SyntheticEQDataset only supports `hp_lp_gain_target='zero'`."
            )
        self.signal_types = signal_types
        self.gain_distribution = str(gain_distribution).lower()
        if self.gain_distribution not in {"beta", "uniform"}:
            raise ValueError(
                f"Unsupported gain_distribution={gain_distribution!r}. "
                "Expected 'beta' or 'uniform'."
            )
        # AUDIT: V-09 — Configure signal type weights with validation
        if signal_type_weights is None:
            self.signal_type_weights = DEFAULT_SIGNAL_TYPE_WEIGHTS.copy()
        else:
            # Validate all signal types have weights
            missing_types = set(signal_types) - set(signal_type_weights.keys())
            if missing_types:
                raise ValueError(
                    f"signal_type_weights missing keys for: {missing_types}. "
                    f"Provide weights for all signal types in {signal_types}."
                )
            # Normalize to sum to 1.0
            total = sum(signal_type_weights[st] for st in signal_types)
            self.signal_type_weights = {
                st: signal_type_weights[st] / total for st in signal_types
            }
        self.augment = augment
        self.n_mels = n_mels
        self.precompute_mels = precompute_mels
        self.base_seed = int(base_seed)
        self._fallback_count = 0
        # AUDIT: V-06 — Track gain samples for distribution monitoring
        self._gain_samples = []
        # AUDIT: V-15 — Type-specific frequency bounds with soft overlap to prevent discontinuities
        # These can be overridden via curriculum stage's type_freq_bounds
        # Overlapping ranges prevent frequency gaps when curriculum changes type weights
        self._type_freq_bounds = {
            "peaking": freq_range,      # Full frequency range
            "lowshelf": (max(20, freq_range[0]), min(5500, freq_range[1])),    # Extended to 5500 for overlap
            "highshelf": (max(800, freq_range[0]), min(20000, freq_range[1])),  # Starts at 800 for overlap
            "highpass": (20, 600),       # Extended to 600 for overlap
            "lowpass": (1500, 20000),    # Starts at 1500 for overlap
        }
        # Type-specific Q bounds for HP/LP filters (typically narrower Q)
        self._type_q_bounds = {
            "highpass": (q_range[0], min(2.0, q_range[1])),
            "lowpass": (q_range[0], min(2.0, q_range[1])),
        }
        self.dsp = DifferentiableBiquadCascade(
            num_bands=num_bands, sample_rate=sample_rate
        )

        # Mel filterbank for precompute mode
        if precompute_mels:
            self._build_mel_filterbank()

    def _build_mel_filterbank(self):
        """Build mel filterbank for precomputed mel-spectrograms."""
        n_freqs = self.n_fft // 2 + 1
        f_min = 20.0
        f_max = self.sample_rate / 2.0

        def hz_to_mel(hz):
            return 2595.0 * np.log10(1.0 + hz / 700.0)

        def mel_to_hz(mel):
            return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

        f_min_mel = hz_to_mel(f_min)
        f_max_mel = hz_to_mel(f_max)
        mel_points = np.linspace(f_min_mel, f_max_mel, self.n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        bin_points = ((self.n_fft + 1) * hz_points / self.sample_rate).astype(int)

        fb = np.zeros((self.n_mels, n_freqs))
        for i in range(self.n_mels):
            f_left = bin_points[i]
            f_center = bin_points[i + 1]
            f_right = bin_points[i + 2]
            for j in range(f_left, f_center):
                if j < n_freqs and f_center > f_left:
                    fb[i, j] = (j - f_left) / (f_center - f_left)
            for j in range(f_center, f_right):
                if j < n_freqs and f_right > f_center:
                    fb[i, j] = (f_right - j) / (f_right - f_center)

        self.mel_fb = torch.tensor(fb, dtype=torch.float32)
        self.mel_window = torch.hann_window(self.n_fft)

    def __len__(self):
        return self._size

    def _cache_metadata(self):
        # AUDIT: HIGH-03 — Include generation parameters hash for staleness detection
        # AUDIT: MEDIUM-05 — Include lineage info for reproducibility
        metadata = {
            "schema_version": PIPELINE_SCHEMA_VERSION,
            "dataset_class": self.__class__.__name__,
            "num_bands": self.num_bands,
            "sample_rate": self.sample_rate,
            "duration": float(self.duration),
            "duration_range": list(self.duration_range)
            if self.duration_range is not None
            else None,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "size": self._size,
            "gain_range": list(self.gain_range),
            "freq_range": list(self.freq_range),
            "q_range": list(self.q_range),
            "type_weights": list(self.type_weights),
            "hp_lp_gain_target": self.hp_lp_gain_target,
            "signal_types": list(self.signal_types),
            "augment": bool(self.augment),
            "precompute_mels": bool(self.precompute_mels),
            "n_mels": self.n_mels,
            "base_seed": self.base_seed,
            # Lineage tracking (AUDIT: MEDIUM-05)
            "generation_params_hash": self._generation_params_hash(),
        }
        metadata.update(self._extra_cache_metadata())
        metadata["signature"] = compute_metadata_signature(metadata)
        return metadata

    def _generation_params_hash(self):
        """
        Compute a hash of all parameters that affect data generation.
        Used for cache staleness detection (AUDIT: HIGH-03).
        """
        params = {
            "num_bands": self.num_bands,
            "sample_rate": self.sample_rate,
            "duration": float(self.duration),
            "duration_range": list(self.duration_range) if self.duration_range else None,
            "n_fft": self.n_fft,
            "gain_range": list(self.gain_range),
            "freq_range": list(self.freq_range),
            "q_range": list(self.q_range),
            "type_weights": list(self.type_weights),
            "signal_types": list(self.signal_types),
            # AUDIT: V-09 — Include signal type weights in hash for cache invalidation
            "signal_type_weights": {st: self.signal_type_weights.get(st, 0.0)
                                   for st in self.signal_types},
            "hp_lp_gain_target": self.hp_lp_gain_target,
            # AUDIT: V-15 — Include type-specific frequency bounds in hash
            "type_freq_bounds": {k: list(v) for k, v in self._type_freq_bounds.items()},
            "type_q_bounds": {k: list(v) for k, v in self._type_q_bounds.items()},
        }
        return compute_metadata_signature(params)

    def _extra_cache_metadata(self):
        """AUDIT: MEDIUM-04 — Include code hash for staleness detection."""
        return {
            "code_hash": self._compute_code_hash(),
        }

    def _compute_code_hash(self):
        """
        Compute a hash of the code that affects data generation.
        Used for cache staleness detection (AUDIT: MEDIUM-04).
        Hashes the content of key source files that affect generation.
        """
        import hashlib
        import inspect

        # Files that affect data generation
        code_files = [
            "dataset.py",
            "differentiable_eq.py",
            "pipeline_utils.py",
        ]

        # Get class source code hash
        source_code = inspect.getsource(self.__class__)
        source_code += inspect.getsource(self.dsp.__class__)

        # Read source files if available
        for fname in code_files:
            try:
                fpath = Path(__file__).parent / fname
                if fpath.exists():
                    source_code += fpath.read_text()
            except Exception:
                pass  # File may not exist or be unreadable

        return hashlib.sha256(source_code.encode("utf-8")).hexdigest()[:16]

    def get_type_prior(self):
        weights = torch.tensor(self.type_weights, dtype=torch.float32)
        return weights / weights.sum().clamp(min=1e-8)

    def apply_curriculum_stage(self, stage):
        """Apply curriculum stage overrides.

        AUDIT: V-07 — Log type weight changes at curriculum stage transitions.
        AUDIT: V-15 — Support type-specific frequency bounds to prevent discontinuities.
        """
        if "gain_bounds" in stage:
            self.gain_range = tuple(stage["gain_bounds"])
        if "q_bounds" in stage:
            self.q_range = tuple(stage["q_bounds"])
        if "freq_bounds" in stage:
            self.freq_range = tuple(stage["freq_bounds"])
        # AUDIT: V-15 — Apply type-specific frequency bounds if provided
        # This allows curriculum stages to constrain filter types to appropriate
        # frequency ranges (e.g., lowshelf only below 2kHz, highshelf only above 500Hz)
        if "type_freq_bounds" in stage:
            for ftype, bounds in stage["type_freq_bounds"].items():
                if ftype in self._type_freq_bounds:
                    self._type_freq_bounds[ftype] = tuple(bounds)
            # Log the type-specific frequency bound changes
            print(f"  [curriculum] Type-specific frequency bounds updated:")
            for ftype, bounds in self._type_freq_bounds.items():
                print(f"    {ftype}: ({bounds[0]:.1f}, {bounds[1]:.1f}) Hz")
        if "type_weights" in stage:
            old_weights = list(self.type_weights)
            self.type_weights = list(stage["type_weights"])
            # AUDIT: V-07 — Validate and log type weight changes
            weight_sum = sum(self.type_weights)
            if abs(weight_sum - 1.0) > 1e-6:
                raise ValueError(
                    f"Curriculum stage type_weights must sum to 1.0, got {weight_sum:.6f}. "
                    f"Stage: {stage.get('name', 'unknown')}"
                )
            # Log the type weight transition
            print(f"  [curriculum] Type weights changed:")
            for i, (old, new) in enumerate(zip(old_weights, self.type_weights)):
                print(f"    {FILTER_NAMES[i]}: {old:.3f} -> {new:.3f}")
            print(f"    Sum: {sum(self.type_weights):.6f}")

    def _generate_dry_signal(self, signal_type, num_samples=None):
        N = self.num_samples if num_samples is None else int(num_samples)
        sr = self.sample_rate
        t = np.linspace(0, N / sr, N, endpoint=False)

        if signal_type == "noise":
            audio = np.random.randn(N).astype(np.float32) * 0.3
        elif signal_type == "pink_noise":
            # 1/f spectrum — more realistic than white noise for music/audio
            white = np.random.randn(N).astype(np.float32)
            fft = np.fft.rfft(white)
            freqs = np.arange(1, len(fft) + 1, dtype=np.float32)
            fft = fft / np.sqrt(freqs)
            audio = np.fft.irfft(fft).astype(np.float32)
            # Normalize to same RMS as white noise
            audio = audio / (np.std(audio) + 1e-8) * 0.3
        elif signal_type == "sweep":
            f0 = random.uniform(100, 2000)
            f1 = random.uniform(100, 2000)
            freq = np.linspace(f0, f1, N)
            phase = 2 * np.pi * np.cumsum(freq / sr)
            audio = (0.3 * np.sin(phase)).astype(np.float32)
        elif signal_type == "harmonic":
            audio = np.zeros(N, dtype=np.float32)
            n_harmonics = random.randint(3, 8)
            fundamental = random.uniform(80, 500)
            for h in range(1, n_harmonics + 1):
                amp = 0.2 / h
                audio += amp * np.sin(2 * np.pi * fundamental * h * t)
        elif signal_type == "speech_like":
            f0 = random.uniform(80, 300)
            audio = np.zeros(N, dtype=np.float32)
            for _ in range(random.randint(3, 12)):
                freq = random.uniform(f0 * 0.5, f0 * 4)
                amp = random.uniform(0.01, 0.1)
                max_dur = max(1, min(int(0.15 * sr), N))
                min_dur = max(1, min(int(0.01 * sr), max_dur))
                dur = random.randint(min_dur, max_dur)
                onset_max = max(0, N - dur)
                onset = random.randint(0, onset_max) if onset_max > 0 else 0
                env = np.hanning(dur)
                audio[onset : onset + dur] += (
                    amp * env * np.sin(2 * np.pi * freq * np.arange(dur) / sr)
                )
        elif signal_type == "percussive":
            # Transient-rich signal: sharp attacks with exponential decay
            audio = np.zeros(N, dtype=np.float32)
            n_transients = random.randint(5, 20)
            for _ in range(n_transients):
                onset = random.randint(0, N - 1)
                decay = random.randint(int(0.005 * sr), int(0.1 * sr))
                freq = random.uniform(200, 8000)
                envelope = np.exp(-np.arange(min(decay, N - onset)) / (decay * 0.3))
                envelope = np.pad(envelope, (0, N - onset - len(envelope)), 'constant')
                carrier = np.sin(2 * np.pi * freq * np.arange(N - onset) / sr)
                audio[onset:] += 0.3 * envelope * carrier[:len(envelope)]
        else:
            audio = np.random.randn(N).astype(np.float32) * 0.3

        # Mild gain variation (simulates recording level differences)
        gain_scale = random.uniform(0.8, 1.2)
        audio = audio * gain_scale

        peak = np.max(np.abs(audio)) + 1e-8
        audio = audio / peak

        # Ensure exact length (some generators like irfft/pad can be ±1)
        if len(audio) > N:
            audio = audio[:N]
        elif len(audio) < N:
            audio = np.pad(audio, (0, N - len(audio)), mode='constant')

        return audio

    def _generate_dry_mix(self, num_samples=None):
        """Mix 2-3 primitive sources so the encoder sees more varied spectra.

        AUDIT: V-09 — Uses weighted signal type selection from config.
        """
        N = self.num_samples if num_samples is None else int(num_samples)
        if len(self.signal_types) <= 1:
            selected_types = [random.choice(self.signal_types)]
        else:
            num_sources = random.randint(2, min(3, len(self.signal_types)))
            # AUDIT: V-09 — Use weighted random selection for signal types
            # Build probability list aligned with self.signal_types
            type_probs = [self.signal_type_weights.get(st, 1.0 / len(self.signal_types))
                         for st in self.signal_types]
            selected_types = random.choices(list(self.signal_types), weights=type_probs, k=num_sources)

        weights = np.random.dirichlet(np.ones(len(selected_types))).astype(np.float32)
        mix = np.zeros(N, dtype=np.float32)
        for weight, signal_type in zip(weights, selected_types):
            mix += float(weight) * self._generate_dry_signal(
                signal_type, num_samples=N
            )

        peak = np.max(np.abs(mix)) + 1e-8
        mix = mix / peak
        return torch.from_numpy(mix)

    def _sample_multitype_params(self):
        """Sample multi-type EQ parameters with weighted filter type distribution.

        AUDIT: V-15 — Uses configurable type-specific frequency bounds to prevent
        discontinuities when curriculum stages change. Type-specific bounds are
        stored in _type_freq_bounds and can be updated via apply_curriculum_stage().
        """
        gains = []
        freqs = []
        qs = []
        types = []

        for _ in range(self.num_bands):
            ftype = random.choices(
                range(NUM_FILTER_TYPES), weights=self.type_weights, k=1
            )[0]

            if ftype == FILTER_PEAKING:
                g = self._sample_gain()
                f = self._log_uniform(*self._type_freq_bounds["peaking"])
                q = self._log_uniform(self.q_range[0], self.q_range[1])
            elif ftype == FILTER_LOWSHELF:
                g = self._sample_gain()
                f = self._log_uniform(*self._type_freq_bounds["lowshelf"])
                q = self._log_uniform(self.q_range[0], self.q_range[1])
            elif ftype == FILTER_HIGHSHELF:
                g = self._sample_gain()
                f = self._log_uniform(*self._type_freq_bounds["highshelf"])
                q = self._log_uniform(self.q_range[0], self.q_range[1])
            elif ftype == FILTER_HIGHPASS:
                g = self._sample_hp_lp_gain()
                f = self._log_uniform(*self._type_freq_bounds["highpass"])
                q = self._log_uniform(*self._type_q_bounds["highpass"])
            elif ftype == FILTER_LOWPASS:
                g = self._sample_hp_lp_gain()
                f = self._log_uniform(*self._type_freq_bounds["lowpass"])
                q = self._log_uniform(*self._type_q_bounds["lowpass"])
            else:
                g = random.uniform(*self.gain_range)
                f = self._log_uniform(self.freq_range[0], self.freq_range[1])
                q = self._log_uniform(self.q_range[0], self.q_range[1])

            gains.append(g)
            freqs.append(f)
            qs.append(q)
            types.append(ftype)

        # Sort by frequency for canonical ordering
        order = sorted(range(len(freqs)), key=lambda i: freqs[i])
        gains = [gains[i] for i in order]
        freqs = [freqs[i] for i in order]
        qs = [qs[i] for i in order]
        types = [types[i] for i in order]

        return (
            torch.tensor(gains, dtype=torch.float32),
            torch.tensor(freqs, dtype=torch.float32),
            torch.tensor(qs, dtype=torch.float32),
            torch.tensor(types, dtype=torch.long),
        )

    def _sample_gain(self):
        """Sample gain from configured distribution (uniform or beta).

        AUDIT: V-06 — Beta distribution (default) matches offline pipeline's
        realistic gain distribution where small EQ adjustments are more common.
        Beta(2, 5) concentrates mass near 0 with long tail for larger adjustments.
        """
        if self.gain_distribution == "beta":
            # Beta distribution: most gains are small, few are large
            max_gain = max(abs(self.gain_range[0]), abs(self.gain_range[1]))
            sign = random.choice([-1, 1])
            magnitude = np.random.beta(2, 5) * max_gain
            gain = sign * magnitude
            # Clamp to configured bounds
            gain = max(self.gain_range[0], min(self.gain_range[1], gain))
            # Track for monitoring (AUDIT: V-06)
            if len(self._gain_samples) < 10000:  # Limit memory usage
                self._gain_samples.append(gain)
            return gain
        else:  # uniform
            return random.uniform(self.gain_range[0], self.gain_range[1])

    def _sample_hp_lp_gain(self):
        # Standardized label contract: HP/LP bands always use 0 dB gain.
        return 0.0

    def _log_uniform(self, low, high):
        """Sample from log-uniform distribution."""
        return math.exp(random.uniform(math.log(low), math.log(high)))

    def _log_gain_distribution(self):
        """
        Log the current gain distribution histogram.

        AUDIT: V-06 — Periodic logging of gain distribution for monitoring.
        Call this periodically (e.g., every 5000 samples) to track the actual
        gain distribution being generated during training.
        """
        if len(self._gain_samples) < 100:
            print(f"  [dataset] Gain distribution: insufficient samples ({len(self._gain_samples)}) to log histogram")
            return

        import numpy as np
        gains = np.array(self._gain_samples)
        print(f"  [dataset] Gain distribution (n={len(gains)}):")
        print(f"    mean={gains.mean():.2f} dB, std={gains.std():.2f} dB")
        print(f"    min={gains.min():.2f} dB, max={gains.max():.2f} dB")
        # Log histogram buckets
        hist, bins = np.histogram(gains, bins=20, range=(self.gain_range[0], self.gain_range[1]))
        print(f"    histogram (20 bins): {hist}")
        # Reset to avoid unbounded memory growth
        self._gain_samples = []

    def _apply_eq_freq_domain(self, dry_audio, gain_db, freq, q, filter_type):
        """Apply multi-type EQ in frequency domain."""
        window = torch.hann_window(self.n_fft)
        dry_tensor = dry_audio.unsqueeze(0)
        stft = torch.stft(
            dry_tensor,
            self.n_fft,
            self.hop_length,
            self.n_fft,
            window=window,
            return_complex=True,
        )
        wet_mag = self.dsp.apply_to_spectrum(
            torch.abs(stft),
            gain_db.unsqueeze(0),
            freq.unsqueeze(0),
            q.unsqueeze(0),
            filter_type=filter_type.unsqueeze(0),
        )
        phase = torch.angle(stft)
        wet_stft = wet_mag * torch.exp(1j * phase)
        wet_audio = torch.istft(
            wet_stft,
            self.n_fft,
            self.hop_length,
            self.n_fft,
            window=window,
            length=dry_audio.shape[0],
        )
        wet_audio = wet_audio.squeeze(0)
        # Data integrity: caller must regenerate labels/audio together if output is non-finite.
        if not torch.isfinite(wet_audio).all():
            return None
        return wet_audio

    def _audio_to_mel(self, audio):
        """Convert audio tensor to log-mel spectrogram. Returns (n_mels, T)."""
        stft = torch.stft(
            audio.unsqueeze(0),
            self.n_fft,
            self.hop_length,
            self.n_fft,
            window=self.mel_window,
            return_complex=True,
        )
        mag = torch.abs(stft).squeeze(0)  # (F, T)
        mel = self.mel_fb @ mag  # (n_mels, T)
        mel = torch.clamp(mel, min=1e-8)
        return torch.log(mel)

    def _augment_dry_audio(self, dry_audio):
        """Apply label-preserving augmentations before EQ is rendered."""
        if not self.augment:
            return dry_audio

        aug_audio = dry_audio.clone()
        if random.random() < 0.5:
            aug_audio = -aug_audio

        max_shift = min(int(0.05 * self.sample_rate), max(1, len(aug_audio) // 10))
        shift = random.randint(-max_shift, max_shift)
        if shift != 0:
            aug_audio = torch.roll(aug_audio, shifts=shift)

        return aug_audio

    def _augment_audio_pair(self, wet_audio, dry_audio):
        """Apply label-preserving augmentations jointly after EQ rendering."""
        if not self.augment:
            return wet_audio, dry_audio

        scale = random.uniform(0.5, 1.0)
        wet_audio = wet_audio * scale
        dry_audio = dry_audio * scale
        return wet_audio, dry_audio

    def _estimate_sample_memory_bytes(self) -> int:
        """
        Estimate memory usage per cached sample in bytes.
        Used for precompute memory estimation (AUDIT: MEDIUM-14).
        """
        # Estimate based on cached components (raw audio + optional mel):
        # - wet_audio, dry_audio: (num_samples,) float32 each
        # - wet_mel: (n_mels, time_frames) float32 when precompute_mels=True
        # - gain, freq, q: (num_bands,) float32 each
        # - filter_type: (num_bands,) int64
        # - active_band_mask: (num_bands,) bool
        mel_elements = self.n_mels * 128  # approximate time frames for 3s audio
        mel_bytes = mel_elements * 4  # float32
        param_bytes = self.num_bands * 4 * 3  # gain, freq, q (float32)
        type_bytes = self.num_bands * 8  # int64
        mask_bytes = self.num_bands * 1  # bool
        return mel_bytes + param_bytes + type_bytes + mask_bytes

    def precompute(self, skip_memory_check: bool = False, warn_threshold: float = 0.5):
        """
        Pre-generate all samples and cache mel-spectrograms + params in memory.
        Caches full training samples (wet_audio/dry_audio + params + optional wet_mel).

        AUDIT: CRITICAL-05 — Skip caching samples that used render fallback to prevent
        contaminating the training set with degenerate samples. These will be
        regenerated on-the-fly during training instead.

        AUDIT: MEDIUM-14 — Memory estimation before precompute to prevent OOM.
        AUDIT: V-18 — Warn at 50% memory usage, hard stop at 90%.

        Args:
            skip_memory_check: If True, bypass memory safety check (use with caution)
            warn_threshold: Memory usage fraction (0-1) at which to warn (default: 0.5)
        """
        # AUDIT: MEDIUM-14 — Use pipeline_utils memory estimation for accurate prediction
        from pipeline_utils import estimate_precompute_memory

        # Calculate number of time frames for this configuration
        num_samples = int(self.duration * self.sample_rate)
        num_frames = (num_samples // self.hop_length) + 1

        mem_est = estimate_precompute_memory(
            dataset_size=self._size,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            sample_rate=self.sample_rate,
            duration=self.duration,
            num_bands=self.num_bands,
            include_mel=self.precompute_mels,
        )

        print(f"Precomputing {self._size} samples...")
        print(f"  [memory] Memory breakdown:")
        print(f"    Mel cache: {mem_est['mel_mb']:.1f} MB")
        print(f"    Parameters: {mem_est['param_mb']:.1f} MB")
        print(f"    Overhead: {mem_est['overhead_mb']:.1f} MB")
        print(f"    Total estimated: {mem_est['total_mb']:.1f} MB ({mem_est['total_gb']:.2f} GB)")

        # Check available memory using utility function
        sys_mem = get_available_memory()
        if sys_mem["psutil_available"]:
            print(f"  [memory] System memory: {sys_mem['total_gb']:.1f} GB total, "
                  f"{sys_mem['available_gb']:.1f} GB available ({sys_mem['percent']:.0f}% used)")

            # AUDIT: V-18 — Two-tier memory checking: warn at 50%, hard stop at 90%
            usage_fraction = mem_est["total_gb"] / max(sys_mem["available_gb"], 0.001)

            # Warning tier (default 50%)
            if usage_fraction > warn_threshold:
                print(f"  [memory] WARNING: Estimated cache uses {usage_fraction:.0%} of available RAM.")
                print(f"  [memory] Suggestions:")
                print(f"    - Reduce dataset_size (current: {self._size})")
                print(f"    - Reduce n_mels (current: {self.n_mels})")
                print(f"    - Reduce duration (current: {self.duration}s)")
                print(f"    - Use on-the-fly generation (precompute_mels=False)")
                print(f"    - Close other applications to free RAM")

            # Safety tier (90% hard limit)
            safety_limit = 0.9
            if not skip_memory_check and usage_fraction > safety_limit:
                deficit_gb = mem_est["total_gb"] - (sys_mem["available_gb"] * safety_limit)
                raise MemoryError(
                    f"Insufficient memory for precomputation. "
                    f"Required: {mem_est['total_gb']:.2f} GB, "
                    f"Available (with {int(safety_limit*100)}% limit): {sys_mem['available_gb'] * safety_limit:.2f} GB. "
                    f"Deficit: {deficit_gb:.2f} GB. "
                    f"Options: 1) Reduce dataset_size, 2) Reduce n_mels, 3) Reduce duration, "
                    f"4) Use on-the-fly generation (precompute_mels=False), "
                    f"5) Pass skip_memory_check=True to bypass this check."
                )
        elif not skip_memory_check:
            print(f"  [memory] WARNING: psutil not available; cannot verify available memory.")
            print(f"  [memory] Proceeding with precomputation (may cause OOM if memory is insufficient).")

        self._cache = []
        skipped_fallbacks = 0
        for i in range(self._size):
            with seeded_index_context(self.base_seed, i):
                sample = self._generate_sample()
                # AUDIT: CRITICAL-05 — Do not cache fallback samples
                if sample.get("render_fallback", False):
                    skipped_fallbacks += 1
                    continue
                self._cache.append(sample)
            if (i + 1) % 2000 == 0:
                print(f"  {i + 1}/{self._size}")
                # AUDIT: V-06 — Log gain distribution every 2000 samples during precompute
                self._log_gain_distribution()

        if skipped_fallbacks > 0:
            print(f"  [precompute] Skipped {skipped_fallbacks} fallback samples (will regenerate on-the-fly)")

        print(f"Precomputation complete: {len(self._cache)} samples cached.")
        # AUDIT: V-06 — Final gain distribution log after precompute
        self._log_gain_distribution()
        gc.collect()  # Release temporary memory

    def _generate_sample(self):
        """Generate a single training sample."""
        # Duration randomization for temporal robustness
        if self.duration_range is not None:
            dur = random.uniform(*self.duration_range)
            num_samples = int(dur * self.sample_rate)
        else:
            num_samples = self.num_samples

        dry_audio = self._generate_dry_mix(num_samples)
        dry_audio = self._augment_dry_audio(dry_audio)

        render_fallback = False
        wet_audio = None
        gain_db = freq = q = filter_type = None
        for _ in range(3):
            gain_db, freq, q, filter_type = self._sample_multitype_params()
            wet_audio = self._apply_eq_freq_domain(
                dry_audio,
                gain_db,
                freq,
                q,
                filter_type,
            )
            if wet_audio is not None:
                break

        # Keep labels aligned with the rendered signal under repeated DSP instability.
        if wet_audio is None:
            render_fallback = True
            wet_audio = dry_audio.clone()
            gain_db = torch.zeros(self.num_bands, dtype=torch.float32)
            freq = torch.logspace(
                math.log10(self.freq_range[0]),
                math.log10(self.freq_range[1]),
                self.num_bands,
            )
            q = torch.ones(self.num_bands, dtype=torch.float32)
            filter_type = torch.zeros(self.num_bands, dtype=torch.long)

        if self.augment:
            wet_audio, dry_audio = self._augment_audio_pair(wet_audio, dry_audio)

        result = {
            "wet_audio": wet_audio,
            "dry_audio": dry_audio,
            "gain": gain_db,
            "freq": freq,
            "q": q,
            "filter_type": filter_type,
            "active_band_mask": torch.ones(self.num_bands, dtype=torch.bool),
            "render_fallback": render_fallback,
        }

        if self.precompute_mels:
            result["wet_mel"] = self._audio_to_mel(wet_audio)

        return result

    def __getitem__(self, idx):
        """
        Return a single training sample.

        Data integrity guarantees (AUDIT: CRITICAL-01, MEDIUM-06):
          - wet_audio and all parameters are validated as finite before return
          - amplitudes are clamped to [-1.0, 1.0] to prevent clip poisoning
          - non-finite samples are regenerated from the next index (max 5 retries)
          - duration mismatches are reconciled via padding/truncation
        """
        if hasattr(self, "_cache") and idx < len(self._cache):
            return self._cache[idx]
        # Retry up to 5 times for non-finite samples (AUDIT: CRITICAL-01)
        for attempt in range(5):
            with seeded_index_context(self.base_seed, idx + attempt):
                try:
                    sample = self._generate_sample()
                except Exception:
                    continue
            # Validate output tensors are finite
            wet = sample.get("wet_audio")
            if wet is not None and not torch.isfinite(wet).all():
                continue  # Regenerate from next seed
            if not torch.isfinite(sample["gain"]).all():
                continue
            if not torch.isfinite(sample["freq"]).all():
                continue
            if not torch.isfinite(sample["q"]).all():
                continue
            # Clamp wet_audio to safe range to prevent clip poisoning
            sample["wet_audio"] = torch.clamp(sample["wet_audio"], -1.0, 1.0)
            # Validate dry_audio if present
            dry = sample.get("dry_audio")
            if dry is None or not torch.isfinite(dry).all():
                continue
            sample["dry_audio"] = torch.clamp(dry, -1.0, 1.0)
            return sample
        # All retries failed — return a safe fallback sample
        return self._fallback_sample(idx)

    def _fallback_sample(self, idx):
        """
        Return a deterministic fallback sample when generation repeatedly fails.
        Uses simple white noise with zero-gain EQ (identity transform).
        """
        self._fallback_count += 1
        print(f"  [dataset] WARNING: fallback sample generated (total fallbacks: {self._fallback_count})")
        with seeded_index_context(self.base_seed, idx + 999999):
            # AUDIT: MEDIUM-06 — Use duration_range midpoint for fallback length
            # to match the expected sample length when duration randomization is active.
            if self.duration_range is not None:
                mid_duration = (self.duration_range[0] + self.duration_range[1]) / 2.0
            else:
                mid_duration = self.duration
            num_samples = int(mid_duration * self.sample_rate)
            dry_audio = torch.randn(num_samples, dtype=torch.float32) * 0.3
            peak = dry_audio.abs().max() + 1e-8
            dry_audio = dry_audio / peak
            gain_db = torch.zeros(self.num_bands, dtype=torch.float32)
            freq = torch.logspace(
                math.log10(self.freq_range[0]),
                math.log10(self.freq_range[1]),
                self.num_bands,
            )
            q = torch.ones(self.num_bands, dtype=torch.float32) * 1.0
            filter_type = torch.zeros(self.num_bands, dtype=torch.long)
            return {
                "wet_audio": dry_audio.clone(),
                "dry_audio": dry_audio.clone(),
                "gain": gain_db,
                "freq": freq,
                "q": q,
                "filter_type": filter_type,
                "active_band_mask": torch.ones(self.num_bands, dtype=torch.bool),
                "render_fallback": True,
            }

    def save_precomputed(self, path):
        """Save precomputed dataset to disk for reuse across training runs."""
        # AUDIT: MEDIUM-14 — Validate output path is under trusted roots
        path = resolve_trusted_artifact_path(
            path,
            allowed_roots=[
                Path.cwd().resolve(),
                Path(__file__).resolve().parent,
                Path(__file__).resolve().parent.parent,
            ],
            must_exist=False,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        if not hasattr(self, "_cache"):
            raise RuntimeError("Call precompute() before saving.")
        metadata = self._cache_metadata()
        torch.save({"metadata": metadata, "cache": self._cache}, path)
        manifest_path = path.with_suffix(path.suffix + ".manifest.json")
        with open(manifest_path, "w") as f:
            json.dump({**metadata, "saved_at": utc_now_iso()}, f, indent=2)
        print(f"Saved precomputed dataset to {path}")

    def load_precomputed(self, path, force_recompute=False):
        """
        Load precomputed dataset from disk, avoiding regeneration.

        Staleness detection (AUDIT: HIGH-03, CRITICAL-02):
          - Compares metadata signature between cache and current config
          - Rejects cache if generation parameters changed (gain bounds, type weights, etc.)
          - Hard failure if no signature is present (prevents silent use of stale caches)
          - force_recompute=True skips load entirely, forcing regeneration

        Args:
            path: Path to the cached .pt file
            force_recompute: If True, refuse to load and force regeneration

        Returns:
            True if cache was loaded successfully, False otherwise
        Raises:
            RuntimeError if cache is stale or unsigned
        """
        if force_recompute:
            print(f"  [cache] --force-recompute flag set; ignoring cache at {path}")
            return False

        path = Path(path)
        path = resolve_trusted_artifact_path(
            path,
            allowed_roots=[
                Path.cwd().resolve(),
                Path(__file__).resolve().parent,
                Path(__file__).resolve().parent.parent,
            ],
            must_exist=False,
        )
        if not path.exists():
            return False
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(payload, dict) and "cache" in payload and "metadata" in payload:
            metadata = payload["metadata"]
            expected_signature = self._cache_metadata()["signature"]
            cached_signature = metadata.get("signature")
            if cached_signature and cached_signature != expected_signature:
                raise RuntimeError(
                    f"STALE CACHE DETECTED: cache signature {cached_signature[:12]}... "
                    f"does not match current config {expected_signature[:12]}.... "
                    f"Regenerate cache or delete {path}. To force regeneration, "
                    f"pass --force-recompute or remove the cache file."
                )
            if not cached_signature:
                raise RuntimeError(
                    f"Unsigned cache at {path}: cache has no metadata signature. "
                    f"This is likely a legacy cache generated before staleness detection. "
                    f"Delete {path} to force regeneration, or pass --force-recompute."
                )
            self._cache = payload["cache"]
        else:
            raise RuntimeError(
                f"Unexpected cache format at {path}: missing 'cache' or 'metadata' keys. "
                f"Delete {path} to force regeneration."
            )
        self._size = len(self._cache)
        print(f"  [cache] Loaded precomputed dataset from {path}: {len(self._cache)} samples (verified)")
        return True


def collate_fn(batch):
    """Custom collate to pad variable-length audio and optional mel-spectrograms."""
    # Data integrity: filter out samples with non-finite parameters
    valid_batch = []
    for item in batch:
        if not torch.isfinite(item["gain"]).all():
            continue
        if not torch.isfinite(item["freq"]).all():
            continue
        if not torch.isfinite(item["q"]).all():
            continue
        if "wet_audio" in item and not torch.isfinite(item["wet_audio"]).all():
            continue
        valid_batch.append(item)
    n_dropped = len(batch) - len(valid_batch)
    if n_dropped > 0:
        print(f"  [dataset] WARNING: dropped {n_dropped} non-finite samples from batch")
    if not valid_batch:
        raise ValueError("All samples in batch are invalid/non-finite; refusing to collate")
    batch = valid_batch

    has_audio = all("wet_audio" in item for item in batch)
    has_mel = all("wet_mel" in item for item in batch)

    def pad_audio(audio, target_len):
        if audio.shape[0] < target_len:
            audio = torch.nn.functional.pad(audio, (0, target_len - audio.shape[0]))
        return audio

    def pad_mel(mel, target_len):
        if mel.shape[-1] < target_len:
            mel = torch.nn.functional.pad(mel, (0, target_len - mel.shape[-1]))
        return mel

    gains = torch.stack([item["gain"] for item in batch])
    freqs = torch.stack([item["freq"] for item in batch])
    qs = torch.stack([item["q"] for item in batch])
    types = torch.stack([item["filter_type"] for item in batch])
    masks = torch.stack([
        item.get("active_band_mask", torch.ones(item["gain"].shape[0], dtype=torch.bool))
        for item in batch
    ])

    result = {
        "gain": gains,
        "freq": freqs,
        "q": qs,
        "filter_type": types,
        "active_band_mask": masks,  # (B, N) bool
    }

    if has_audio:
        wet_list = [item["wet_audio"] for item in batch]
        max_len = max(x.shape[0] for x in wet_list)
        wet_padded = torch.stack([pad_audio(w, max_len) for w in wet_list])
        lengths = torch.tensor([w.shape[0] for w in wet_list])
        result["wet_audio"] = wet_padded
        result["lengths"] = lengths
        # dry_audio is optional (not present in precomputed cache)
        if all("dry_audio" in item for item in batch):
            dry_list = [item["dry_audio"] for item in batch]
            dry_padded = torch.stack([pad_audio(d, max_len) for d in dry_list])
            result["dry_audio"] = dry_padded

    if has_mel:
        mel_list = [item["wet_mel"] for item in batch]
        max_mel_len = max(m.shape[-1] for m in mel_list)
        mel_padded = torch.stack([pad_mel(m, max_mel_len) for m in mel_list])
        result["wet_mel"] = mel_padded

    return result
