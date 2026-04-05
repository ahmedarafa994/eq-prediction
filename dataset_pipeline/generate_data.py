"""
Multi-type synthetic EQ data generation.

Generates (dry, wet, params) pairs with support for peaking, low-shelf,
high-shelf, high-pass, and low-pass filter types.
"""
import os
import json
import random
import argparse
import math
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
import torch
import torchaudio
import torchaudio.functional as F


FILTER_TYPES = ["peaking", "lowshelf", "highshelf", "highpass", "lowpass"]
DEFAULT_TYPE_WEIGHTS = [0.5, 0.15, 0.15, 0.1, 0.1]


def log_uniform(low, high):
    """Sample from log-uniform distribution (equal probability per decade)."""
    return math.exp(random.uniform(math.log(low), math.log(high)))


def beta_gain(max_gain=24.0):
    """
    Sample gain using beta distribution.
    Most EQ adjustments are small, so concentrate mass near 0.
    """
    sign = random.choice([-1, 1])
    magnitude = np.random.beta(2, 5) * max_gain
    return sign * magnitude


def sample_band_params(ftype, gain_bounds, freq_bounds, q_bounds):
    """Sample parameters for a single band based on filter type."""
    params = {"type": ftype}

    if ftype == "peaking":
        params["gain"] = beta_gain(min(abs(gain_bounds[0]), abs(gain_bounds[1])))
        params["freq"] = log_uniform(freq_bounds[0], freq_bounds[1])
        params["q"] = log_uniform(q_bounds[0], q_bounds[1])

    elif ftype == "lowshelf":
        params["gain"] = beta_gain(min(abs(gain_bounds[0]), abs(gain_bounds[1])))
        params["freq"] = log_uniform(max(20, freq_bounds[0]),
                                      min(5000, freq_bounds[1]))
        params["q"] = log_uniform(q_bounds[0], q_bounds[1])

    elif ftype == "highshelf":
        params["gain"] = beta_gain(min(abs(gain_bounds[0]), abs(gain_bounds[1])))
        params["freq"] = log_uniform(max(1000, freq_bounds[0]),
                                      min(20000, freq_bounds[1]))
        params["q"] = log_uniform(q_bounds[0], q_bounds[1])

    elif ftype == "highpass":
        params["gain"] = 0.0  # Not applicable
        params["freq"] = log_uniform(20, 500)
        params["q"] = log_uniform(q_bounds[0], min(2.0, q_bounds[1]))

    elif ftype == "lowpass":
        params["gain"] = 0.0  # Not applicable
        params["freq"] = log_uniform(2000, 20000)
        params["q"] = log_uniform(q_bounds[0], min(2.0, q_bounds[1]))

    return params


def generate_eq_params(num_bands, bounds, type_weights=None):
    """Generate random multi-type EQ parameters."""
    if type_weights is None:
        type_weights = DEFAULT_TYPE_WEIGHTS

    params = []
    for _ in range(num_bands):
        ftype = random.choices(FILTER_TYPES, weights=type_weights, k=1)[0]
        band = sample_band_params(
            ftype,
            bounds["gain"],
            bounds["freq"],
            bounds["q"]
        )
        params.append(band)

    # Sort by frequency for canonical ordering
    params.sort(key=lambda b: b["freq"])

    return params


def compute_biquad_coeffs_peaking(gain_db, freq, q, sr):
    """Compute peaking biquad coefficients (Audio EQ Cookbook)."""
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * freq / sr
    alpha = math.sin(w0) / (2.0 * q)

    b0 = 1.0 + alpha * A
    b1 = -2.0 * math.cos(w0)
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * math.cos(w0)
    a2 = 1.0 - alpha / A

    return [b0/a0, b1/a0, b2/a0, a1/a0, a2/a0]


def compute_biquad_coeffs_lowshelf(gain_db, freq, q, sr):
    """Compute low-shelf biquad coefficients."""
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * freq / sr
    alpha = math.sin(w0) / (2.0 * q)
    sqA = math.sqrt(A)

    b0 = A * ((A + 1) - (A - 1) * math.cos(w0) + 2 * sqA * alpha)
    b1 = 2 * A * ((A - 1) - (A + 1) * math.cos(w0))
    b2 = A * ((A + 1) - (A - 1) * math.cos(w0) - 2 * sqA * alpha)
    a0 = (A + 1) + (A - 1) * math.cos(w0) + 2 * sqA * alpha
    a1 = -2 * ((A - 1) + (A + 1) * math.cos(w0))
    a2 = (A + 1) + (A - 1) * math.cos(w0) - 2 * sqA * alpha

    return [b0/a0, b1/a0, b2/a0, a1/a0, a2/a0]


def compute_biquad_coeffs_highshelf(gain_db, freq, q, sr):
    """Compute high-shelf biquad coefficients."""
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * freq / sr
    alpha = math.sin(w0) / (2.0 * q)
    sqA = math.sqrt(A)

    b0 = A * ((A + 1) + (A - 1) * math.cos(w0) + 2 * sqA * alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * math.cos(w0))
    b2 = A * ((A + 1) + (A - 1) * math.cos(w0) - 2 * sqA * alpha)
    a0 = (A + 1) - (A - 1) * math.cos(w0) + 2 * sqA * alpha
    a1 = 2 * ((A - 1) - (A + 1) * math.cos(w0))
    a2 = (A + 1) - (A - 1) * math.cos(w0) - 2 * sqA * alpha

    return [b0/a0, b1/a0, b2/a0, a1/a0, a2/a0]


def compute_biquad_coeffs_highpass(freq, q, sr):
    """Compute high-pass biquad coefficients."""
    w0 = 2.0 * math.pi * freq / sr
    alpha = math.sin(w0) / (2.0 * q)

    b0 = (1 + math.cos(w0)) / 2
    b1 = -(1 + math.cos(w0))
    b2 = (1 + math.cos(w0)) / 2
    a0 = 1 + alpha
    a1 = -2 * math.cos(w0)
    a2 = 1 - alpha

    return [b0/a0, b1/a0, b2/a0, a1/a0, a2/a0]


def compute_biquad_coeffs_lowpass(freq, q, sr):
    """Compute low-pass biquad coefficients."""
    w0 = 2.0 * math.pi * freq / sr
    alpha = math.sin(w0) / (2.0 * q)

    b0 = (1 - math.cos(w0)) / 2
    b1 = 1 - math.cos(w0)
    b2 = (1 - math.cos(w0)) / 2
    a0 = 1 + alpha
    a1 = -2 * math.cos(w0)
    a2 = 1 - alpha

    return [b0/a0, b1/a0, b2/a0, a1/a0, a2/a0]


COEFF_FUNCS = {
    "peaking": lambda g, f, q, sr: compute_biquad_coeffs_peaking(g, f, q, sr),
    "lowshelf": lambda g, f, q, sr: compute_biquad_coeffs_lowshelf(g, f, q, sr),
    "highshelf": lambda g, f, q, sr: compute_biquad_coeffs_highshelf(g, f, q, sr),
    "highpass": lambda g, f, q, sr: compute_biquad_coeffs_highpass(f, q, sr),
    "lowpass": lambda g, f, q, sr: compute_biquad_coeffs_lowpass(f, q, sr),
}

TYPE_TO_INDEX = {t: i for i, t in enumerate(FILTER_TYPES)}


def apply_eq_band(waveform, biquad_coeffs):
    """Apply a single biquad filter to a waveform using torchaudio lfilter."""
    b = torch.tensor(biquad_coeffs[:3], dtype=torch.float32).unsqueeze(0)
    a = torch.tensor([1.0] + biquad_coeffs[3:], dtype=torch.float32).unsqueeze(0)
    return F.lfilter(waveform, a, b, clamp=False)


def apply_eq_cascade(waveform, sample_rate, params):
    """Apply multi-type EQ cascade to dry audio."""
    wet = waveform.clone()
    for band in params:
        ftype = band["type"]
        gain = band.get("gain", 0.0)
        freq = band["freq"]
        q = band["q"]

        coeffs = COEFF_FUNCS[ftype](gain, freq, q, sample_rate)
        wet = apply_eq_band(wet, coeffs)

    return wet


def process_file(args):
    """Load dry audio, generate multi-type EQ params, apply, save."""
    file_path, out_dir, length_samples, sample_rate, bounds, num_bands, type_weights = args
    try:
        waveform, sr = torchaudio.load(file_path)
        if sr != sample_rate:
            waveform = F.resample(waveform, sr, sample_rate)

        # Ensure single channel
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Random crop or pad
        if waveform.size(1) > length_samples:
            start = random.randint(0, waveform.size(1) - length_samples)
            waveform = waveform[:, start:start + length_samples]
        elif waveform.size(1) < length_samples:
            pad_amount = length_samples - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))

        params = generate_eq_params(num_bands, bounds, type_weights)
        wet_audio = apply_eq_cascade(waveform, sample_rate, params)

        # Save
        file_stem = Path(file_path).stem
        wet_path = os.path.join(out_dir, f"{file_stem}_wet.wav")
        dry_path = os.path.join(out_dir, f"{file_stem}_dry.wav")
        param_path = os.path.join(out_dir, f"{file_stem}_params.json")

        torchaudio.save(wet_path, wet_audio, sample_rate)
        torchaudio.save(dry_path, waveform, sample_rate)
        with open(param_path, 'w') as f:
            json.dump(params, f, indent=2)

        return True
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def build_dataset(input_dir, output_dir, config):
    """Iterate over dry audio files and generate multi-type EQ data."""
    input_paths = list(Path(input_dir).rglob("*.wav")) + list(Path(input_dir).rglob("*.flac"))
    os.makedirs(output_dir, exist_ok=True)

    length_samples = int(config['audio_duration'] * config['sample_rate'])
    bounds = {
        'gain': config.get('gain_bounds', [-24.0, 24.0]),
        'freq': config.get('freq_bounds', [20.0, 20000.0]),
        'q': config.get('q_bounds', [0.1, 10.0]),
    }
    type_weights = config.get('type_weights', DEFAULT_TYPE_WEIGHTS)
    num_bands = config.get('num_bands', 5)
    sample_rate = config.get('sample_rate', 44100)

    tasks = [
        (str(p), output_dir, length_samples, sample_rate, bounds, num_bands, type_weights)
        for p in input_paths
    ]

    print(f"Generating multi-type EQ data for {len(tasks)} files...")
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_file, tasks)

    success = sum(results)
    print(f"Successfully generated {success}/{len(tasks)} samples.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Path to dry audio files")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save wet audio and targets")
    args = parser.parse_args()

    config = {
        "num_bands": 5,
        "sample_rate": 44100,
        "audio_duration": 3.0,
        "gain_bounds": [-24.0, 24.0],
        "freq_bounds": [20.0, 20000.0],
        "q_bounds": [0.1, 10.0],
        "type_weights": [0.5, 0.15, 0.15, 0.1, 0.1],
    }
    build_dataset(args.input_dir, args.output_dir, config)
