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
from pipeline_utils import (
    PIPELINE_SCHEMA_VERSION,
    build_sample_id,
    stable_int_hash,
    utc_now_iso,
    validate_band_list,
)


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
    (
        file_path,
        input_dir,
        out_dir,
        length_samples,
        sample_rate,
        bounds,
        num_bands,
        type_weights,
        base_seed,
    ) = args
    sample_id = build_sample_id(file_path, input_dir)
    sample_seed = (int(base_seed) + stable_int_hash(sample_id)) % (2 ** 32)
    random.seed(sample_seed)
    np.random.seed(sample_seed)
    torch.manual_seed(sample_seed)

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
        params = validate_band_list(params, expected_num_bands=num_bands)
        wet_audio = apply_eq_cascade(waveform, sample_rate, params)
        if not torch.isfinite(wet_audio).all():
            raise ValueError("Generated wet audio contains non-finite values")

        # Save
        wet_path = os.path.join(out_dir, f"{sample_id}_wet.wav")
        dry_path = os.path.join(out_dir, f"{sample_id}_dry.wav")
        param_path = os.path.join(out_dir, f"{sample_id}_params.json")
        relative_source = str(Path(file_path).resolve().relative_to(Path(input_dir).resolve()))

        torchaudio.save(wet_path, wet_audio, sample_rate)
        torchaudio.save(dry_path, waveform, sample_rate)
        param_payload = {
            "schema_version": PIPELINE_SCHEMA_VERSION,
            "sample_id": sample_id,
            "source_file": str(Path(file_path).resolve()),
            "source_relpath": relative_source,
            "sample_rate": sample_rate,
            "length_samples": length_samples,
            "bands": params,
        }
        with open(param_path, 'w') as f:
            json.dump(param_payload, f, indent=2)

        return {
            "success": True,
            "sample_id": sample_id,
            "source_file": str(Path(file_path).resolve()),
            "source_relpath": relative_source,
            "wet_path": wet_path,
            "dry_path": dry_path,
            "param_path": param_path,
        }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return {
            "success": False,
            "sample_id": sample_id,
            "source_file": str(Path(file_path).resolve()),
            "error": str(e),
        }


def build_dataset(input_dir, output_dir, config):
    """Iterate over dry audio files and generate multi-type EQ data."""
    input_root = Path(input_dir).resolve()
    input_paths = sorted(input_root.rglob("*.wav")) + sorted(input_root.rglob("*.flac"))
    input_paths = sorted({path.resolve() for path in input_paths})
    if not input_paths:
        raise FileNotFoundError(f"No input audio files found under {input_root}")
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
    base_seed = int(config.get('seed', 42))
    # AUDIT: LOW-25 — Respect cgroup/cpu quota limits and --max_processes CLI arg
    max_processes = int(config.get('max_processes', 8))  # Default cap to protect shared systems
    num_processes = int(config.get('num_processes', min(max(1, cpu_count() - 1), max_processes)))

    tasks = [
        (
            str(p),
            str(input_root),
            output_dir,
            length_samples,
            sample_rate,
            bounds,
            num_bands,
            type_weights,
            base_seed,
        )
        for p in input_paths
    ]

    print(f"Generating multi-type EQ data for {len(tasks)} files...")
    if num_processes == 1:
        results = [process_file(task) for task in tasks]
    else:
        with Pool(processes=num_processes) as pool:
            results = pool.map(process_file, tasks)

    successes = [result for result in results if result["success"]]
    failures = [result for result in results if not result["success"]]
    manifest = {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "created_at": utc_now_iso(),
        "input_dir": str(input_root),
        "output_dir": str(Path(output_dir).resolve()),
        "config": config,
        "num_requested": len(tasks),
        "num_succeeded": len(successes),
        "num_failed": len(failures),
        "samples": successes,
        "failures": failures,
    }
    manifest_path = Path(output_dir) / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Successfully generated {len(successes)}/{len(tasks)} samples.")
    print(f"Manifest written to {manifest_path}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Path to dry audio files")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save wet audio and targets")
    # AUDIT: LOW-25 — Allow limiting processes to protect shared systems
    parser.add_argument("--max_processes", type=int, default=8,
                        help="Maximum number of parallel processes (default: 8)")
    args = parser.parse_args()

    config = {
        "num_bands": 5,
        "sample_rate": 44100,
        "audio_duration": 3.0,
        "gain_bounds": [-24.0, 24.0],
        "freq_bounds": [20.0, 20000.0],
        "q_bounds": [0.1, 10.0],
        "type_weights": [0.5, 0.15, 0.15, 0.1, 0.1],
        "seed": 42,
        "max_processes": args.max_processes,
    }
    build_dataset(args.input_dir, args.output_dir, config)
