import hashlib
import json
import math
import random
import re
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


PIPELINE_SCHEMA_VERSION = 2
SUPPORTED_FILTER_TYPES = (
    "peaking",
    "lowshelf",
    "highshelf",
    "highpass",
    "lowpass",
)

# AUDIT: V-22 — Compute trusted roots at module load time for security
# These absolute paths are resolved once and cached, preventing runtime
# path traversal attacks on trusted root calculations.
_MODULE_ROOT = Path(__file__).resolve().parent
_CWD_ROOT = Path.cwd().resolve()
_PARENT_ROOT = _MODULE_ROOT.parent

# Default trusted roots for artifact loading (AUDIT: V-22)
DEFAULT_TRUSTED_ROOTS = [
    _CWD_ROOT,
    _MODULE_ROOT,
    _PARENT_ROOT,
]

# Make these available for import
TRUSTED_ROOTS = DEFAULT_TRUSTED_ROOTS.copy()

# ---------------------------------------------------------------------------
# Dependency validation (AUDIT: HIGH-08)
# ---------------------------------------------------------------------------

OPTIONAL_DEPENDENCIES = {
    "bitsandbytes": "8-bit AdamW optimizer (memory optimization)",
    "deepspeed": "DeepSpeed ZeRO-2 distributed training",
    "transformers": "Pre-trained audio encoder backends (wav2vec2, MERT, CLAP)",
    "timm": "Vision transformer backends (AST/ViT)",
    "wandb": "Experiment tracking and metrics visualization",
    "onnxruntime": "ONNX model export and benchmarking",
    "pytorch_lightning": "Alternative Lightning training pipeline",
    "scipy": "Hungarian matching (linear_sum_assignment) and loss functions",
}

REQUIRED_DEPENDENCIES = {"torch", "torchaudio", "numpy", "yaml", "scipy"}


def _check_dependency(dep_name: str, import_name: str | None = None) -> tuple[bool, str | None]:
    """Check if a dependency is available. Returns (available, error_msg)."""
    name = import_name or dep_name.replace("-", "_")
    try:
        __import__(name)
        return True, None
    except ImportError as e:
        return False, str(e)


def validate_dependencies(config: dict) -> list[str]:
    """
    Validate that all dependencies required by the given config are available.
    Returns a list of error messages (empty if all checks pass).

    Usage (AUDIT: HIGH-08): call this in Trainer.__init__ before any heavy work.
    """
    errors: list[str] = []

    # Check required dependencies
    for dep in REQUIRED_DEPENDENCIES:
        try:
            __import__(dep.replace("-", "_"))
        except ImportError:
            errors.append(f"REQUIRED dependency missing: {dep}")

    # Check optional dependencies based on config
    if config.get("trainer", {}).get("use_8bit_optimizer", False):
        try:
            import bitsandbytes
        except ImportError:
            errors.append(
                "Config requires 8-bit optimizer but bitsandbytes is not installed. "
                "Set trainer.use_8bit_optimizer=false or pip install bitsandbytes."
            )

    if config.get("trainer", {}).get("use_deepspeed", False):
        try:
            import deepspeed
        except ImportError:
            errors.append(
                "Config requires DeepSpeed but deepspeed is not installed. "
                "Set trainer.use_deepspeed=false or pip install deepspeed."
            )

    encoder_backend = config.get("model", {}).get("encoder", {}).get("backend", "hybrid_tcn")
    if encoder_backend in ("wav2vec2_frozen", "wav2vec2"):
        try:
            import transformers
        except ImportError:
            errors.append(
                f"Encoder backend '{encoder_backend}' requires transformers but it is not installed."
            )
    if encoder_backend == "ast":
        try:
            import timm
        except ImportError:
            errors.append(
                f"Encoder backend 'ast' requires timm but it is not installed."
            )
    if encoder_backend == "clap":
        try:
            import transformers
        except ImportError:
            errors.append(
                f"Encoder backend 'clap' requires transformers but it is not installed."
            )
    if encoder_backend == "mert":
        try:
            import transformers
        except ImportError:
            errors.append(
                f"Encoder backend 'mert' requires transformers but it is not installed."
            )

    dataset_type = str(config.get("data", {}).get("dataset_type", "synthetic")).lower()
    if dataset_type == "musdb":
        try:
            from dataset_musdb import MUSDB18EQDataset
        except ImportError:
            errors.append(
                "dataset_type='musdb' but dataset_musdb.py is not available."
            )
    if dataset_type == "litdata":
        try:
            from dataset_litdata import LitdataEQDataset
        except ImportError:
            errors.append(
                "dataset_type='litdata' but dataset_litdata.py is not available."
            )

    return errors


# ---------------------------------------------------------------------------
# Config schema validation (AUDIT: MEDIUM-11)
# ---------------------------------------------------------------------------

CONFIG_SCHEMA = {
    "required_top_level": ["data", "model", "loss", "trainer"],
    "data": {
        "required": ["num_bands", "sample_rate", "batch_size"],
        "optional": {
            "audio_duration": float,
            "dataset_size": int,
            "val_dataset_size": int,
            "n_fft": int,
            "hop_length": int,
            "n_mels": int,
            "num_workers": int,
            "precompute_mels": bool,
        },
    },
    "model": {
        "required": ["encoder", "num_bands"],
    },
    "trainer": {
        "required": ["max_epochs"],
        "optional": {
            "max_epochs": int,
            "log_every_n_steps": int,
            "early_stopping_patience": int,
            "gradient_accumulation_steps": int,
            "use_torch_compile": bool,
            "use_gradient_checkpointing": bool,
            "use_8bit_optimizer": bool,
            "use_deepspeed": bool,
            "precision": str,
        },
    },
}


def validate_config_schema(config: dict) -> list[str]:
    """
    Validate the configuration against the expected schema.
    Returns a list of warnings/errors (empty if valid).

    Usage (AUDIT: MEDIUM-11): call after yaml.safe_load() in Trainer.__init__.
    """
    issues: list[str] = []

    # Check required top-level keys
    for key in CONFIG_SCHEMA["required_top_level"]:
        if key not in config:
            issues.append(f"Config missing required top-level key: '{key}'")

    # Check section-level required keys
    for section, rules in CONFIG_SCHEMA.items():
        if section == "required_top_level":
            continue
        if section not in config:
            continue
        section_cfg = config[section]
        for key in rules.get("required", []):
            if key not in section_cfg:
                issues.append(f"Config['{section}'] missing required key: '{key}'")

    # Type-check known optional fields
    for section, rules in CONFIG_SCHEMA.items():
        if section == "required_top_level":
            continue
        if section not in config:
            continue
        section_cfg = config[section]
        for field_name, expected_type in rules.get("optional", {}).items():
            if field_name in section_cfg:
                val = section_cfg[field_name]
                if not isinstance(val, expected_type):
                    issues.append(
                        f"Config['{section}']['{field_name}'] should be {expected_type.__name__}, "
                        f"got {type(val).__name__} ({val!r})"
                    )

    # Value range checks
    data_cfg = config.get("data", {})
    if "batch_size" in data_cfg:
        bs = data_cfg["batch_size"]
        if not isinstance(bs, int) or bs < 1:
            issues.append(f"data.batch_size must be a positive integer, got {bs!r}")
    if "num_workers" in data_cfg:
        nw = data_cfg["num_workers"]
        if not isinstance(nw, int) or nw < 0:
            issues.append(f"data.num_workers must be a non-negative integer, got {nw!r}")

    trainer_cfg = config.get("trainer", {})
    if "max_epochs" in trainer_cfg:
        me = trainer_cfg["max_epochs"]
        if not isinstance(me, int) or me < 1:
            issues.append(f"trainer.max_epochs must be a positive integer, got {me!r}")
    if "early_stopping_patience" in trainer_cfg:
        ep = trainer_cfg["early_stopping_patience"]
        if not isinstance(ep, int) or ep < 1:
            issues.append(f"trainer.early_stopping_patience must be a positive integer, got {ep!r}")

    # Validate loss lambda values (AUDIT: H-05)
    loss_section = config.get("loss", {})
    lambda_keys = [k for k in loss_section if k.startswith("lambda_")]
    for key in lambda_keys:
        val = loss_section[key]
        if not isinstance(val, (int, float)):
            issues.append(f"loss.{key} must be a number, got {type(val).__name__}")
        elif val < 0:
            issues.append(f"loss.{key} must be non-negative, got {val}")
        elif val > 100:
            issues.append(f"loss.{key} is unusually large ({val}), may cause instability")

    # Check at least one lambda is positive
    if lambda_keys and all(loss_section.get(k, 0) == 0 for k in lambda_keys):
        issues.append("All loss lambda values are zero — no learning signal")

    # Validate curriculum stage lambdas
    curriculum = config.get("curriculum", {})
    for stage in curriculum.get("stages", []):
        stage_lambdas = {k: v for k, v in stage.items() if k.startswith("lambda_")}
        for key, val in stage_lambdas.items():
            if not isinstance(val, (int, float)):
                issues.append(f"curriculum stage '{stage.get('name', '?')}.{key}' must be a number")
            elif val < 0:
                issues.append(f"curriculum stage '{stage.get('name', '?')}.{key}' must be non-negative, got {val}")

    return issues


# ---------------------------------------------------------------------------
# Path sanitization (AUDIT: MEDIUM-23)
# ---------------------------------------------------------------------------

def validate_path_under_root(file_path: str | Path, root_dir: str | Path) -> Path:
    """
    Validate that file_path is strictly under root_dir (prevents path traversal).
    Returns the resolved Path. Raises ValueError if not under root.

    Usage (AUDIT: MEDIUM-23): call before loading any user-supplied path.
    """
    resolved = Path(file_path).resolve()
    root_resolved = Path(root_dir).resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError:
        raise ValueError(
            f"Path traversal detected: {resolved} is not under {root_resolved}"
        )
    return resolved


def resolve_trusted_artifact_path(
    file_path: str | Path,
    *,
    allowed_roots: list[str | Path] | None = None,
    must_exist: bool = True,
) -> Path:
    """
    Resolve a path and ensure it is under at least one trusted root.

    This is a defense-in-depth helper for loading checkpoints/caches/config artifacts.

    AUDIT: V-22 — Uses module-level TRUSTED_ROOTS computed at load time
    for security. Prevents runtime manipulation of trusted root paths.
    """
    candidate = Path(file_path).expanduser().resolve()

    roots = []
    if allowed_roots:
        roots.extend(Path(root).expanduser().resolve() for root in allowed_roots)
    else:
        # AUDIT: V-22 — Use pre-computed absolute trusted roots
        roots.extend(TRUSTED_ROOTS)

    for root in roots:
        try:
            validate_path_under_root(candidate, root)
            if must_exist and not candidate.exists():
                raise FileNotFoundError(f"Artifact path does not exist: {candidate}")
            return candidate
        except ValueError:
            continue

    roots_text = ", ".join(str(root) for root in roots)
    raise ValueError(
        f"Artifact path is outside trusted roots: {candidate}. "
        f"Allowed roots: {roots_text}"
    )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def stable_int_hash(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def set_global_seed(seed: int, deterministic: bool = False, num_workers: int = 0) -> None:
    """
    Set global random seeds for reproducibility.

    AUDIT: V-20 — Full reproducibility requires num_workers=0 for DataLoader.
    Warns user when deterministic=True is used with multiprocessing.

    Args:
        seed: Random seed for Python, NumPy, and PyTorch
        deterministic: If True, enables deterministic algorithms (slower)
        num_workers: DataLoader num_workers for reproducibility warning
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # AUDIT: V-20 — Warn about multiprocessing with deterministic mode
        if num_workers > 0:
            import warnings
            warnings.warn(
                f"deterministic=True is set but num_workers={num_workers} > 0. "
                f"Full reproducibility requires num_workers=0 for DataLoader. "
                f"With multiprocessing, CUDA operations in workers may have non-determinism. "
                f"Set num_workers=0 or accept reduced reproducibility.",
                UserWarning,
                stacklevel=2
            )
    else:
        torch.backends.cudnn.deterministic = False


def seed_worker(worker_id: int, base_seed: int) -> None:
    worker_seed = (int(base_seed) + int(worker_id)) % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


@contextmanager
def seeded_index_context(base_seed: int, index: int):
    seed = (int(base_seed) + (int(index) * 1_000_003)) % (2 ** 32)
    random_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    # CUDA state save/restore only works in the main process or with spawn start method.
    # In forked DataLoader workers, CUDA is not initialized and accessing it raises RuntimeError.
    cuda_states = None
    _cuda_available = False
    try:
        _cuda_available = torch.cuda.is_available() and torch.cuda.is_initialized()
    except RuntimeError:
        _cuda_available = False
    if _cuda_available:
        cuda_states = torch.cuda.get_rng_state_all()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if _cuda_available:
        torch.cuda.manual_seed_all(seed)

    try:
        yield seed
    finally:
        random.setstate(random_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def sanitize_path_fragment(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")


def build_sample_id(source_path: str | Path, source_root: str | Path | None = None) -> str:
    path = Path(source_path).resolve()
    relative_text = path.name
    if source_root is not None:
        try:
            relative_text = str(path.relative_to(Path(source_root).resolve()))
        except ValueError:
            relative_text = path.name

    stem = sanitize_path_fragment(str(Path(relative_text).with_suffix("")))
    digest = hashlib.sha256(str(relative_text).encode("utf-8")).hexdigest()[:12]
    return f"{stem}-{digest}" if stem else digest


def validate_band_dict(band: dict) -> dict:
    if not isinstance(band, dict):
        raise TypeError(f"Expected band dict, got {type(band)!r}")

    missing = {"type", "gain", "freq", "q"} - set(band.keys())
    if missing:
        raise ValueError(f"Band payload missing keys: {sorted(missing)}")

    filter_type = str(band["type"])
    if filter_type not in SUPPORTED_FILTER_TYPES:
        raise ValueError(f"Unsupported filter type: {filter_type}")

    gain = float(band["gain"])
    freq = float(band["freq"])
    q = float(band["q"])
    for field_name, value in (("gain", gain), ("freq", freq), ("q", q)):
        if not math.isfinite(value):
            raise ValueError(f"Band field `{field_name}` must be finite, got {value}")

    return {
        "type": filter_type,
        "gain": gain,
        "freq": freq,
        "q": q,
    }


def validate_band_list(params: list[dict], expected_num_bands: int | None = None) -> list[dict]:
    if not isinstance(params, list) or not params:
        raise ValueError("Band payload must be a non-empty list")

    normalized = [validate_band_dict(band) for band in params]
    if expected_num_bands is not None and len(normalized) != int(expected_num_bands):
        raise ValueError(
            f"Expected {expected_num_bands} bands, got {len(normalized)}"
        )
    return normalized


def compute_metadata_signature(metadata: dict) -> str:
    normalized = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def compute_version_hash() -> str:
    """
    Compute a hash of key source files for version tracking in checkpoints.

    AUDIT: LOW-16 — This hash is stored in checkpoints to identify the exact
    code version that created them. Useful for diagnosing checkpoint compatibility
    issues and tracking model evolution across training runs.

    Returns:
        First 12 characters of SHA256 hash (hex string)
    """
    import inspect

    # Core source files that affect training/behavior
    source_files = [
        "train.py",
        "model_tcn.py",
        "loss_multitype.py",
        "dataset.py",
        "differentiable_eq.py",
        "dsp_frontend.py",
        "pipeline_utils.py",
    ]

    source_code = ""
    for fname in source_files:
        try:
            fpath = Path(__file__).parent / fname
            if fpath.exists():
                source_code += fpath.read_text()
        except Exception:
            pass  # File may not exist or be unreadable

    # Also hash key class sources for changes that don't touch disk
    try:
        from train import Trainer
        from model_tcn import StreamingTCNModel
        from loss_multitype import MultiTypeEQLoss
        source_code += inspect.getsource(Trainer)
        source_code += inspect.getsource(StreamingTCNModel)
        source_code += inspect.getsource(MultiTypeEQLoss)
    except Exception:
        pass  # Classes may not be importable in all contexts

    return hashlib.sha256(source_code.encode("utf-8")).hexdigest()[:12]


def safe_load_checkpoint(
    checkpoint_path: str | Path,
    *,
    allowed_roots: list[str | Path] | None = None,
    map_location: str | None = None,
    pickle_module: Any | None = None,
) -> dict:
    """
    Safely load a PyTorch checkpoint with path validation.

    This is a security wrapper around torch.load that validates the checkpoint
    path is under a trusted root before loading. This prevents arbitrary code
    execution via path traversal attacks when loading user-supplied checkpoints.

    Args:
        checkpoint_path: Path to the checkpoint file (.pt, .pth, .ckpt, etc.)
        allowed_roots: List of trusted root directories. If None, uses defaults.
        map_location: torch.map_location argument (e.g., 'cpu', 'cuda:0')
        pickle_module: Custom pickle module for validation (advanced)

    Returns:
        The loaded checkpoint dictionary

    Raises:
        FileNotFoundError: Checkpoint file doesn't exist
        ValueError: Checkpoint path is outside trusted roots

    Usage (AUDIT: HIGH-11):
        checkpoint = safe_load_checkpoint("checkpoints/best.pt")
        # Or with custom roots:
        checkpoint = safe_load_checkpoint(
            "/shared/model.pt",
            allowed_roots=["/shared/models", "/training/artifacts"]
        )
    """
    # Resolve and validate path
    resolved_path = resolve_trusted_artifact_path(
        checkpoint_path,
        allowed_roots=allowed_roots,
        must_exist=True
    )

    # Validate file extension (basic sanity check)
    allowed_extensions = {".pt", ".pth", ".ckpt", ".pkl", ".bin"}
    if resolved_path.suffix.lower() not in allowed_extensions:
        raise ValueError(
            f"Checkpoint has unexpected extension: {resolved_path.suffix}. "
            f"Allowed: {sorted(allowed_extensions)}"
        )

    # Load with torch.load (still has pickle risks, but path is validated)
    # For production, consider torch.save with weights_only=True in PyTorch 2.0+
    checkpoint = torch.load(
        resolved_path,
        map_location=map_location,
        pickle_module=pickle_module,
    )

    # Basic structure validation
    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Checkpoint must be a dict, got {type(checkpoint).__name__}"
        )

    return checkpoint


# ---------------------------------------------------------------------------
# Memory estimation (AUDIT: MEDIUM-14)
# ---------------------------------------------------------------------------

def estimate_precompute_memory(
    dataset_size: int,
    n_mels: int,
    n_fft: int,
    sample_rate: int,
    duration: float,
    num_bands: int,
    include_mel: bool = True,
    precision: str = "float32",
) -> dict:
    """
    Estimate memory requirements for precomputed dataset.

    Helps prevent OOM errors during dataset precomputation by estimating
    RAM usage before allocating memory. Returns breakdown by component.

    Args:
        dataset_size: Number of samples in the dataset
        n_mels: Number of mel frequency bins
        n_fft: FFT size (determines frequency resolution)
        sample_rate: Audio sample rate
        duration: Duration of each audio sample in seconds
        num_bands: Number of EQ bands (affects parameter tensor sizes)
        include_mel: Whether mel-spectrograms are cached (major memory factor)
        precision: Data type precision ("float32" or "float16")

    Returns:
        dict with estimated memory usage in bytes/MB/GB per component

    Example (AUDIT: MEDIUM-14):
        >>> est = estimate_precompute_memory(50000, 128, 2048, 22050, 3.0, 5)
        >>> if est["total_gb"] > 8:
        ...     print("WARNING: Precomputation may exceed available RAM")
    """
    # Bytes per element based on precision
    bytes_per_element = 4 if precision == "float32" else 2
    bytes_per_long = 8  # int64

    # Time frames per sample (after STFT)
    num_samples = int(duration * sample_rate)
    hop_length = n_fft // 4
    num_frames = (num_samples // hop_length) + 1

    # 1. Mel-spectrogram cache (if enabled)
    # Shape: (n_mels, num_frames) per sample
    mel_bytes = 0
    if include_mel:
        mel_elements_per_sample = n_mels * num_frames
        mel_bytes = dataset_size * mel_elements_per_sample * bytes_per_element

    # 2. Parameter tensors (gain, freq, q, filter_type, active_mask)
    # All are (num_bands,) per sample except filter_type which is int64
    param_bytes = (
        # gain: float32
        dataset_size * num_bands * bytes_per_element +
        # freq: float32
        dataset_size * num_bands * bytes_per_element +
        # q: float32
        dataset_size * num_bands * bytes_per_element +
        # filter_type: int64
        dataset_size * num_bands * bytes_per_long +
        # active_band_mask: bool (1 byte, but Python overhead makes it more)
        dataset_size * num_bands * 1
    )

    # 3. Python object overhead (dict keys, list pointers, etc.)
    # This is significant but hard to estimate precisely
    # Rough estimate: ~500 bytes per sample for dict structure
    overhead_bytes = dataset_size * 500

    # 4. Sample metadata (render_fallback flag, etc.)
    metadata_bytes = dataset_size * 100  # Rough estimate

    total_bytes = mel_bytes + param_bytes + overhead_bytes + metadata_bytes

    return {
        "mel_bytes": mel_bytes,
        "mel_mb": mel_bytes / (1024 * 1024),
        "mel_gb": mel_bytes / (1024 * 1024 * 1024),
        "param_bytes": param_bytes,
        "param_mb": param_bytes / (1024 * 1024),
        "param_gb": param_bytes / (1024 * 1024 * 1024),
        "overhead_bytes": overhead_bytes,
        "overhead_mb": overhead_bytes / (1024 * 1024),
        "metadata_bytes": metadata_bytes,
        "metadata_mb": metadata_bytes / (1024 * 1024),
        "total_bytes": total_bytes,
        "total_mb": total_bytes / (1024 * 1024),
        "total_gb": total_bytes / (1024 * 1024 * 1024),
        "num_frames_per_sample": num_frames,
        "mel_elements_per_sample": n_mels * num_frames if include_mel else 0,
    }


def get_available_memory() -> dict:
    """
    Get available system memory information.

    Returns:
        dict with total, available, and used memory in bytes/GB

    Usage (AUDIT: MEDIUM-14):
        >>> mem = get_available_memory()
        >>> if mem["available_gb"] < 4:
        ...     print("WARNING: Low memory available")
    """
    try:
        import psutil
    except ImportError:
        # Fallback: return zeros if psutil not available
        return {
            "total_bytes": 0,
            "total_gb": 0.0,
            "available_bytes": 0,
            "available_gb": 0.0,
            "used_bytes": 0,
            "used_gb": 0.0,
            "percent": 0.0,
            "psutil_available": False,
        }

    mem = psutil.virtual_memory()
    return {
        "total_bytes": mem.total,
        "total_gb": mem.total / (1024**3),
        "available_bytes": mem.available,
        "available_gb": mem.available / (1024**3),
        "used_bytes": mem.used,
        "used_gb": mem.used / (1024**3),
        "percent": mem.percent,
        "psutil_available": True,
    }


def validate_precompute_memory(
    dataset_size: int,
    n_mels: int,
    n_fft: int,
    sample_rate: int,
    duration: float,
    num_bands: int,
    include_mel: bool = True,
    safety_margin: float = 0.8,
) -> tuple[bool, dict]:
    """
    Validate whether precomputation is safe given available memory.

    Args:
        dataset_size: Number of samples to precompute
        n_mels: Number of mel frequency bins
        n_fft: FFT size
        sample_rate: Audio sample rate
        duration: Duration per sample in seconds
        num_bands: Number of EQ bands
        include_mel: Whether mel-spectrograms are cached
        safety_margin: Fraction of available memory to use (default 0.8 = 80%)

    Returns:
        (is_safe, info_dict) where is_safe is True if precomputation is safe

    Usage (AUDIT: MEDIUM-14):
        >>> safe, info = validate_precompute_memory(50000, 128, 2048, 22050, 3.0, 5)
        >>> if not safe:
        ...     print(f"Insufficient memory: {info}")
    """
    est = estimate_precompute_memory(
        dataset_size, n_mels, n_fft, sample_rate, duration, num_bands, include_mel
    )
    mem = get_available_memory()

    required_bytes = est["total_bytes"]
    available_bytes = mem["available_bytes"] * safety_margin

    is_safe = required_bytes < available_bytes if available_bytes > 0 else False

    info = {
        "required_gb": est["total_gb"],
        "available_gb": mem["available_gb"],
        "safety_margin": safety_margin,
        "usable_gb": mem["available_gb"] * safety_margin,
        "is_safe": is_safe,
        "deficit_gb": max(0, est["total_gb"] - mem["available_gb"] * safety_margin) / (1024**3),
    }

    return is_safe, info


# ---------------------------------------------------------------------------
# Additional security utilities (AUDIT: MEDIUM-14)
# ---------------------------------------------------------------------------

def safe_yaml_load(
    file_path: str | Path,
    *,
    allowed_roots: list[str | Path] | None = None,
    must_exist: bool = True,
) -> dict:
    """
    Safely load a YAML config file with path validation.

    Args:
        file_path: Path to the YAML file
        allowed_roots: List of trusted root directories
        must_exist: Whether the file must exist

    Returns:
        Parsed YAML content as a dictionary

    Raises:
        FileNotFoundError: File doesn't exist and must_exist=True
        ValueError: Path is outside trusted roots

    Usage (AUDIT: MEDIUM-14):
        >>> config = safe_yaml_load("conf/config.yaml")
    """
    import yaml

    resolved_path = resolve_trusted_artifact_path(
        file_path,
        allowed_roots=allowed_roots,
        must_exist=must_exist
    )

    with open(resolved_path, "r") as f:
        # Use safe_load to prevent arbitrary code execution from YAML
        content = yaml.safe_load(f)

    if content is None:
        return {}
    if not isinstance(content, dict):
        raise ValueError(
            f"YAML file must contain a dict at top level, got {type(content).__name__}"
        )

    return content


def safe_json_load(
    file_path: str | Path,
    *,
    allowed_roots: list[str | Path] | None = None,
    must_exist: bool = True,
) -> dict:
    """
    Safely load a JSON file with path validation.

    Args:
        file_path: Path to the JSON file
        allowed_roots: List of trusted root directories
        must_exist: Whether the file must exist

    Returns:
        Parsed JSON content

    Raises:
        FileNotFoundError: File doesn't exist and must_exist=True
        ValueError: Path is outside trusted roots or JSON is invalid
    """
    resolved_path = resolve_trusted_artifact_path(
        file_path,
        allowed_roots=allowed_roots,
        must_exist=must_exist
    )

    with open(resolved_path, "r") as f:
        content = json.load(f)

    if not isinstance(content, dict):
        raise ValueError(
            f"JSON file must contain a dict at top level, got {type(content).__name__}"
        )

    return content