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
import hashlib
import tempfile
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
import torch
import torchaudio
import torchaudio.functional as F
import yaml
from pipeline_utils import (
    PIPELINE_SCHEMA_VERSION,
    build_sample_id,
    stable_int_hash,
    utc_now_iso,
    validate_band_list,
    resolve_trusted_artifact_path,
)


FILTER_TYPES = ["peaking", "lowshelf", "highshelf", "highpass", "lowpass"]
DEFAULT_TYPE_WEIGHTS = [0.2, 0.2, 0.2, 0.2, 0.2]


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path_value: str) -> str:
    """Prefer workspace-relative paths in metadata to avoid absolute-path leakage."""
    resolved = Path(path_value).resolve()
    cwd = Path.cwd().resolve()
    try:
        return str(resolved.relative_to(cwd))
    except ValueError:
        return str(resolved)


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
        include_source_abs_path,
        checksum_files,
    ) = args
    sample_id = build_sample_id(file_path, input_dir)
    source_path = Path(file_path).resolve()

    # AUDIT: V-08, V-12 — Sanitize home directory paths to prevent path leakage
    # When include_source_abs_path is True, absolute paths are stored in the output.
    # Replace home directory with ~ to avoid exposing system structure.
    source_abs_path = str(source_path)
    if include_source_abs_path:
        # Try to replace home directory with ~ for privacy
        try:
            home_dir = Path.home()
            if source_abs_path.startswith(str(home_dir)):
                source_abs_path = source_abs_path.replace(str(home_dir), "~", 1)
                # Also warn user about path leakage (once per worker would be ideal, but this is simple)
                import warnings
                warnings.warn(
                    "include_source_abs_path is enabled: absolute paths will be stored in outputs. "
                    "Home directory is sanitized to ~, but full paths may still expose system structure. "
                    "Use this flag only for debugging, not for production datasets.",
                    stacklevel=1
                )
        except Exception:
            # If home detection fails, use the raw path (but we tried)
            pass
    try:
        relative_source = str(source_path.relative_to(Path(input_dir).resolve()))
    except ValueError:
        relative_source = source_path.name
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
            return {"file_path": str(file_path), "success": False, "error": "NaN/inf in wet audio"}

        # Save only if finite
        wet_path = os.path.join(out_dir, f"{sample_id}_wet.wav")
        dry_path = os.path.join(out_dir, f"{sample_id}_dry.wav")
        param_path = os.path.join(out_dir, f"{sample_id}_params.json")

        torchaudio.save(wet_path, wet_audio, sample_rate)
        torchaudio.save(dry_path, waveform, sample_rate)
        checksums = None
        if checksum_files:
            checksums = {
                "wet_sha256": sha256_file(wet_path),
                "dry_sha256": sha256_file(dry_path),
            }
        param_payload = {
            "schema_version": PIPELINE_SCHEMA_VERSION,
            "sample_id": sample_id,
            "source_relpath": relative_source,
            "sample_rate": sample_rate,
            "length_samples": length_samples,
            "bands": params,
        }
        if include_source_abs_path:
            param_payload["source_file"] = source_abs_path
        if checksums is not None:
            param_payload["checksums"] = checksums
        with open(param_path, 'w') as f:
            json.dump(param_payload, f, indent=2)
        if checksum_files:
            checksums["params_sha256"] = sha256_file(param_path)

        result = {
            "success": True,
            "sample_id": sample_id,
            "source_relpath": relative_source,
            "wet_path": wet_path,
            "dry_path": dry_path,
            "param_path": param_path,
            "band_types": [band["type"] for band in params],
        }
        if include_source_abs_path:
            result["source_file"] = source_abs_path
        if checksums is not None:
            result["checksums"] = checksums
        return result
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        failure = {
            "success": False,
            "sample_id": sample_id,
            "source_relpath": relative_source,
            "error": str(e),
        }
        if include_source_abs_path:
            failure["source_file"] = source_abs_path
        return failure


def build_dataset(input_dir, output_dir, config):
    """Iterate over dry audio files and generate multi-type EQ data."""
    trusted_roots = [
        Path.cwd(),
        Path(__file__).resolve().parent.parent,
        Path(tempfile.gettempdir()),
    ]

    # AUDIT: HIGH-11 — Validate input/output paths are under allowed roots
    input_root = resolve_trusted_artifact_path(
        input_dir,
        allowed_roots=trusted_roots,
        must_exist=True
    )
    output_root = resolve_trusted_artifact_path(
        output_dir,
        allowed_roots=trusted_roots,
        must_exist=False
    )
    os.makedirs(output_root, exist_ok=True)

    input_paths = sorted(
        [p for ext in ("*.wav", "*.flac") for p in input_root.rglob(ext)]
    )
    input_paths = sorted({path.resolve() for path in input_paths})
    if not input_paths:
        raise FileNotFoundError(f"No input audio files found under {input_root}")

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
    include_source_abs_path = bool(config.get("include_source_abs_path", False))
    checksum_files = bool(config.get("checksum_files", True))
    fail_on_any_failure = bool(config.get("fail_on_any_failure", False))
    max_failure_rate = float(config.get("max_failure_rate", 0.01))
    # AUDIT: LOW-25 — Respect cgroup/cpu quota limits and --max_processes CLI arg
    max_processes = int(config.get('max_processes', 8))  # Default cap to protect shared systems
    num_processes = int(config.get('num_processes', min(max(1, cpu_count() - 1), max_processes)))

    tasks = [
        (
            str(p),
            str(input_root),
            str(output_root),
            length_samples,
            sample_rate,
            bounds,
            num_bands,
            type_weights,
            base_seed,
            include_source_abs_path,
            checksum_files,
        )
        for p in input_paths
    ]

    print(f"Generating multi-type EQ data for {len(tasks)} files...")

    # AUDIT: V-21 — Check for empty tasks before processing
    if not tasks:
        raise ValueError("No input files found to process. Check input_dir path and file extensions.")

    if num_processes == 1:
        results = [process_file(task) for task in tasks]
    else:
        # AUDIT: HIGH-11 — Robust pool error handling with proper cleanup
        # Pool.map can hang on worker exceptions; use imap with timeout and explicit error collection
        results = []
        try:
            with Pool(processes=num_processes) as pool:
                # Use imap_unordered for better progress visibility and error isolation
                # chunksize ensures work is distributed evenly
                iterator = pool.imap_unordered(
                    process_file,
                    tasks,
                    chunksize=max(1, len(tasks) // (num_processes * 4))
                )
                for i, result in enumerate(iterator, 1):
                    results.append(result)
                    if i % 100 == 0:
                        print(f"  [generate] Processed {i}/{len(tasks)} samples...")
        except KeyboardInterrupt:
            print("  [generate] Interrupted by user")
            raise
        except Exception as e:
            # AUDIT: V-21 — Don't set results=[] on error; preserve partial results
            # Log partial results before re-raising - user may want to salvage what we got
            print(f"  [generate] ERROR in pool processing: {e}")
            print(f"  [generate] Collected {len(results)}/{len(tasks)} results before failure")

            # If we have at least some results, write them to disk before failing
            if results:
                print(f"  [generate] Writing partial manifest with {len(results)} results...")
                # Write partial results to a manifest file for potential recovery
                partial_manifest_path = Path(output_root) / "manifest_partial.json"
                successes = [r for r in results if r.get("success", False)]
                failures = [r for r in results if not r.get("success", False)]
                partial_manifest = {
                    "schema_version": PIPELINE_SCHEMA_VERSION,
                    "created_at": utc_now_iso(),
                    "input_dir": str(input_root),
                    "output_dir": str(output_root),
                    "partial": True,
                    "num_requested": len(tasks),
                    "num_succeeded": len(successes),
                    "num_failed": len(failures),
                    "error": str(e),
                    "samples": successes,
                    "failures": failures,
                }
                with open(partial_manifest_path, "w") as f:
                    json.dump(partial_manifest, f, indent=2)
                print(f"  [generate] Partial manifest saved to {partial_manifest_path}")

            # Re-raise to fail the dataset generation (fail_on_any_failure logic below)
            raise

    successes = [result for result in results if result["success"]]
    failures = [result for result in results if not result["success"]]
    failure_rate = len(failures) / max(1, len(tasks))
    type_histogram = {name: 0 for name in FILTER_TYPES}
    for sample in successes:
        for filter_type in sample.get("band_types", []):
            if filter_type in type_histogram:
                type_histogram[filter_type] += 1
    status = "ok"
    if fail_on_any_failure and failures:
        status = "failed"
    if failure_rate > max_failure_rate:
        status = "failed"
    manifest = {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "created_at": utc_now_iso(),
        "input_dir": display_path(str(input_root)),
        "output_dir": display_path(str(output_root)),
        "config": config,
        "num_requested": len(tasks),
        "num_succeeded": len(successes),
        "num_failed": len(failures),
        "failure_rate": failure_rate,
        "status": status,
        "quality": {
            "quality_score": max(0.0, 1.0 - failure_rate),
            "type_histogram": type_histogram,
            "max_failure_rate": max_failure_rate,
            "fail_on_any_failure": fail_on_any_failure,
            "checksum_files": checksum_files,
            "absolute_source_paths": include_source_abs_path,
        },
        "samples": successes,
        "failures": failures,
    }
    manifest_path = Path(output_dir) / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Successfully generated {len(successes)}/{len(tasks)} samples.")
    print(f"Manifest written to {manifest_path}")
    if status == "failed":
        raise RuntimeError(
            "Dataset quality gate failed: "
            f"{len(failures)}/{len(tasks)} samples failed "
            f"(failure_rate={failure_rate:.2%}, max_failure_rate={max_failure_rate:.2%})"
        )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Path to dry audio files")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save wet audio and targets")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config path")
    # AUDIT: LOW-25 — Allow limiting processes to protect shared systems
    parser.add_argument("--max_processes", type=int, default=8,
                        help="Maximum number of parallel processes (default: 8)")
    parser.add_argument("--max_failure_rate", type=float, default=None,
                        help="Maximum tolerated sample failure rate before failing the build")
    parser.add_argument("--fail_on_any_failure", action="store_true",
                        help="Fail dataset generation if any sample fails")
    parser.add_argument("--include_source_abs_path", action="store_true",
                        help="Include absolute source paths in output payloads")
    parser.add_argument("--disable_checksums", action="store_true",
                        help="Disable per-artifact sha256 checksums in payloads/manifests")
    args = parser.parse_args()

    config = {
        "num_bands": 5,
        "sample_rate": 44100,
        "audio_duration": 3.0,
        "gain_bounds": [-24.0, 24.0],
        "freq_bounds": [20.0, 20000.0],
        "q_bounds": [0.1, 10.0],
        "type_weights": [0.2, 0.2, 0.2, 0.2, 0.2],
        "seed": 42,
        "max_processes": args.max_processes,
        "max_failure_rate": 0.01,
        "fail_on_any_failure": False,
        "include_source_abs_path": False,
        "checksum_files": True,
    }
    if args.config:
        with open(args.config, "r") as f:
            loaded = yaml.safe_load(f) or {}
        config.update(loaded)
    config["max_processes"] = args.max_processes
    if args.max_failure_rate is not None:
        config["max_failure_rate"] = float(args.max_failure_rate)
    if args.fail_on_any_failure:
        config["fail_on_any_failure"] = True
    if args.include_source_abs_path:
        config["include_source_abs_path"] = True
    if args.disable_checksums:
        config["checksum_files"] = False
    build_dataset(args.input_dir, args.output_dir, config)
