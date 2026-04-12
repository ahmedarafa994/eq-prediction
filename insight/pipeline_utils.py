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


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def stable_int_hash(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
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
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
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
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:10]
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