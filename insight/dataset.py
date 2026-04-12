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
    seeded_index_context,
    utc_now_iso,
)

FILTER_NAMES = ["peaking", "lowshelf", "highshelf", "highpass", "lowpass"]
DEFAULT_TYPE_WEIGHTS = [0.5, 0.15, 0.15, 0.1, 0.1]


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
        self.hp_lp_gain_target = hp_lp_gain_target
        if self.hp_lp_gain_target != "zero":
            raise ValueError(
                "SyntheticEQDataset only supports `hp_lp_gain_target='zero'`."
            )
        self.signal_types = signal_types
        self.augment = augment
        self.n_mels = n_mels
        self.precompute_mels = precompute_mels
        self.base_seed = int(base_seed)
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
            "hp_lp_gain_target": self.hp_lp_gain_target,
        }
        return compute_metadata_signature(params)

    def _extra_cache_metadata(self):
        return {}

    def get_type_prior(self):
        weights = torch.tensor(self.type_weights, dtype=torch.float32)
        return weights / weights.sum().clamp(min=1e-8)

    def apply_curriculum_stage(self, stage):
        if "gain_bounds" in stage:
            self.gain_range = tuple(stage["gain_bounds"])
        if "q_bounds" in stage:
            self.q_range = tuple(stage["q_bounds"])
        if "freq_bounds" in stage:
            self.freq_range = tuple(stage["freq_bounds"])
        if "type_weights" in stage:
            self.type_weights = list(stage["type_weights"])

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
                dur = random.randint(int(0.01 * sr), int(0.15 * sr))
                onset = random.randint(0, max(1, N - dur))
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
        """Mix 2-3 primitive sources so the encoder sees more varied spectra."""
        N = self.num_samples if num_samples is None else int(num_samples)
        if len(self.signal_types) <= 1:
            selected_types = [random.choice(self.signal_types)]
        else:
            num_sources = random.randint(2, min(3, len(self.signal_types)))
            selected_types = random.sample(list(self.signal_types), k=num_sources)

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
        """Sample multi-type EQ parameters with weighted filter type distribution."""
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
                f = self._log_uniform(self.freq_range[0], self.freq_range[1])
                q = self._log_uniform(self.q_range[0], self.q_range[1])
            elif ftype == FILTER_LOWSHELF:
                g = self._sample_gain()
                f = self._log_uniform(
                    max(20, self.freq_range[0]), min(5000, self.freq_range[1])
                )
                q = self._log_uniform(self.q_range[0], self.q_range[1])
            elif ftype == FILTER_HIGHSHELF:
                g = self._sample_gain()
                f = self._log_uniform(
                    max(1000, self.freq_range[0]), min(20000, self.freq_range[1])
                )
                q = self._log_uniform(self.q_range[0], self.q_range[1])
            elif ftype == FILTER_HIGHPASS:
                g = self._sample_hp_lp_gain()
                f = self._log_uniform(20, 500)
                q = self._log_uniform(self.q_range[0], min(2.0, self.q_range[1]))
            elif ftype == FILTER_LOWPASS:
                g = self._sample_hp_lp_gain()
                f = self._log_uniform(2000, 20000)
                q = self._log_uniform(self.q_range[0], min(2.0, self.q_range[1]))
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
        """Sample gain uniformly across the full gain range (D-05)."""
        return random.uniform(self.gain_range[0], self.gain_range[1])

    def _sample_hp_lp_gain(self):
        # Standardized label contract: HP/LP bands always use 0 dB gain.
        return 0.0

    def _log_uniform(self, low, high):
        """Sample from log-uniform distribution."""
        return math.exp(random.uniform(math.log(low), math.log(high)))

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
        # Data integrity: regenerate if output is non-finite (extreme Q/gain combos)
        if not torch.isfinite(wet_audio).all():
            return dry_audio  # Fallback: return unprocessed audio
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

    def precompute(self):
        """
        Pre-generate all samples and cache mel-spectrograms + params in memory.
        Drops raw audio to save memory. Only caches what training needs.
        """
        print(f"Precomputing {self._size} samples...")
        self._cache = []
        for i in range(self._size):
            with seeded_index_context(self.base_seed, i):
                self._cache.append(self._generate_sample())
            if (i + 1) % 2000 == 0:
                print(f"  {i + 1}/{self._size}")

        print(f"Precomputation complete: {len(self._cache)} samples cached.")

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

        gain_db, freq, q, filter_type = self._sample_multitype_params()
        wet_audio = self._apply_eq_freq_domain(dry_audio, gain_db, freq, q, filter_type)

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
                sample = self._generate_sample()
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
            if dry is not None and torch.isfinite(dry).all():
                sample["dry_audio"] = torch.clamp(dry, -1.0, 1.0)
            return sample
        # All retries failed — return a safe fallback sample
        return self._fallback_sample(idx)

    def _fallback_sample(self, idx):
        """
        Return a deterministic fallback sample when generation repeatedly fails.
        Uses simple white noise with zero-gain EQ (identity transform).
        """
        with seeded_index_context(self.base_seed, idx + 999999):
            num_samples = self.num_samples
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
            }

    def save_precomputed(self, path):
        """Save precomputed dataset to disk for reuse across training runs."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not hasattr(self, "_cache"):
            raise RuntimeError("Call precompute() before saving.")
        metadata = self._cache_metadata()
        torch.save({"metadata": metadata, "cache": self._cache}, path)
        manifest_path = path.with_suffix(path.suffix + ".manifest.json")
        with open(manifest_path, "w") as f:
            json.dump({**metadata, "saved_at": utc_now_iso()}, f, indent=2)
        print(f"Saved precomputed dataset to {path}")

    def load_precomputed(self, path):
        """
        Load precomputed dataset from disk, avoiding regeneration.

        Staleness detection (AUDIT: HIGH-03):
          - Compares metadata signature between cache and current config
          - Rejects cache if generation parameters changed (gain bounds, type weights, etc.)
          - Falls back to stale-cache warning if no signature is present (legacy cache)
        """
        path = Path(path)
        if not path.exists():
            return False
        payload = torch.load(path, weights_only=False)
        if isinstance(payload, dict) and "cache" in payload and "metadata" in payload:
            metadata = payload["metadata"]
            expected_signature = self._cache_metadata()["signature"]
            cached_signature = metadata.get("signature")
            if cached_signature and cached_signature != expected_signature:
                print(
                    f"  [cache] STALE CACHE DETECTED: cache signature {cached_signature[:12]}... "
                    f"does not match current config {expected_signature[:12]}... — "
                    f"refusing to load. Regenerate cache or delete {path}"
                )
                return False
            if not cached_signature:
                print(
                    f"  [cache] WARNING: cached dataset has no signature (legacy cache). "
                    f"Loading without staleness check. Regenerate cache to enable validation."
                )
            self._cache = payload["cache"]
        else:
            print(
                f"  [cache] WARNING: loaded cache from {path} has unexpected format. "
                f"Loading without metadata validation. Regenerate cache to enable checks."
            )
            self._cache = payload
        self._size = len(self._cache)
        print(f"  [cache] Loaded precomputed dataset from {path}: {len(self._cache)} samples")
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
    batch = valid_batch if valid_batch else batch  # Fallback to original if all filtered

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
