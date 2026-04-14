"""
Training script for the multi-type IDSP EQ Estimator.

Uses StreamingTCNModel with MultiTypeEQLoss (Hungarian matching,
permutation-invariant parameter regression, type classification).
On-the-fly synthetic multi-type data generation.

Optimization features (QLoRA/Unsloth/DeepSpeed-inspired):
  - 8-bit AdamW optimizer (bitsandbytes) for reduced memory
  - torch.compile for graph-level optimization
  - Gradient checkpointing for TCN stacks
  - Fused Triton kernels for gated activation
  - DeepSpeed ZeRO-2 integration for multi-GPU
  - Pin memory + async GPU transfer
  - Precomputed dataset caching to disk
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_float32_matmul_precision("high")  # TF32 for 3x matmul throughput on Ampere+
from torch.utils.data import DataLoader, random_split
from functools import partial
from model_tcn import StreamingTCNModel
from dsp_frontend import STFTFrontend
from differentiable_eq import FILTER_HIGHSHELF, FILTER_LOWSHELF
from loss_multitype import MultiTypeEQLoss as SimplifiedEQLoss
from loss_multitype import HungarianBandMatcher
from metrics import compute_eq_metrics
from dataset import SyntheticEQDataset, collate_fn
try:
    from dataset_musdb import MUSDB18EQDataset
except ImportError:
    MUSDB18EQDataset = None
try:
    from dataset_litdata import LitdataEQDataset
except ImportError:
    LitdataEQDataset = None
import yaml
from pathlib import Path
import time
import json
import math
import os
import signal
from contextlib import nullcontext
from pipeline_utils import seed_worker, set_global_seed, utc_now_iso
from pipeline_utils import (
    resolve_trusted_artifact_path,
    validate_dependencies,
    validate_config_schema,
    compute_version_hash,
)
from structured_logger import StructuredLogger
from tqdm import tqdm

try:
    import bitsandbytes as bnb

    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False

try:
    import deepspeed

    HAS_DEEPSPEED = True
except ImportError:
    HAS_DEEPSPEED = False


PRIMARY_VAL_SCORE_WEIGHTS = {
    "gain_mae_db_matched": 1.0,
    "type_error": 4.0,
    "freq_mae_oct_matched": 0.25,
}


def compute_primary_val_score(metrics):
    """Composite validation score used for default checkpointing/early stop."""
    gain_mae = float(metrics.get("gain_mae_db_matched", 0.0))
    type_acc = float(metrics.get("type_accuracy_matched", 0.0))
    freq_mae = float(metrics.get("freq_mae_oct_matched", 0.0))
    return (
        PRIMARY_VAL_SCORE_WEIGHTS["gain_mae_db_matched"] * gain_mae
        + PRIMARY_VAL_SCORE_WEIGHTS["type_error"] * (1.0 - type_acc)
        + PRIMARY_VAL_SCORE_WEIGHTS["freq_mae_oct_matched"] * freq_mae
    )


def resolve_monitor_value(metrics, monitor_metric, fallback):
    return float(metrics.get(monitor_metric, fallback))


def metric_direction(metric_name):
    if "accuracy" in metric_name:
        return "max"
    return "min"


def metric_improved(metric_name, current_value, best_value):
    if metric_direction(metric_name) == "max":
        return current_value > best_value
    return current_value < best_value


def apply_stage_to_training_state(train_dataset, criterion, stage):
    """Apply curriculum stage overrides to the dataset and loss priors."""
    apply_fn = getattr(train_dataset, "apply_curriculum_stage", None)
    if callable(apply_fn):
        apply_fn(stage)

    # Apply dynamic component loss weights from curriculum stage
    for weight_name in ["lambda_gain", "lambda_freq", "lambda_q", "lambda_type", "lambda_spectral", "lambda_type_match",
                        "matcher_lambda_gain", "matcher_lambda_freq", "matcher_lambda_q", "matcher_lambda_type_match"]:
        if weight_name in stage:
            setattr(criterion, weight_name, float(stage[weight_name]))

    current_prior = None
    prior_getter = getattr(train_dataset, "get_type_prior", None)
    update_priors = getattr(criterion, "update_type_priors", None)
    if callable(prior_getter):
        current_prior = prior_getter()
        # AUDIT: V-07 — Log effective type weights being used
        if "type_weights" in stage:
            print(f"  [train] Effective type prior: {current_prior.tolist()}")
        if callable(update_priors):
            update_priors(current_prior)
    return current_prior


def get_dataset_type_prior(dataset):
    current = dataset
    while current is not None:
        prior_getter = getattr(current, "get_type_prior", None)
        if callable(prior_getter):
            return prior_getter()
        current = getattr(current, "dataset", None)
    return None


def load_config(path="conf/config.yaml"):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    data_cfg = cfg.get("data", {})
    loss_cfg = cfg.get("loss", {})
    model_cfg = cfg.get("model", {})
    encoder_cfg = model_cfg.get("encoder", {})
    dataset_type = str(data_cfg.get("dataset_type", "synthetic")).lower()
    lambda_gain = float(loss_cfg.get("lambda_gain", 1.0))
    lambda_freq = float(loss_cfg.get("lambda_freq", 1.0))
    lambda_q = float(loss_cfg.get("lambda_q", 1.0))
    lambda_spectral = float(loss_cfg.get("lambda_spectral", 5.0))
    lambda_typed_spectral = float(loss_cfg.get("lambda_typed_spectral", 0.0))
    lambda_hmag = float(loss_cfg.get("lambda_hmag", 0.0))
    lambda_multi_scale = float(loss_cfg.get("lambda_multi_scale", 1.0))
    type_loss_mode = str(loss_cfg.get("type_loss_mode", "focal")).lower()
    hp_lp_gain_target = data_cfg.get("hp_lp_gain_target", "zero")
    encoder_backend = encoder_cfg.get("backend", "hybrid_tcn")
    if dataset_type not in {"synthetic", "musdb", "litdata"}:
        raise ValueError(
            "`data.dataset_type` must be one of {'synthetic', 'musdb', 'litdata'}."
        )
    if dataset_type == "musdb" and MUSDB18EQDataset is None:
        raise ValueError(
            "`data.dataset_type=musdb` requires `dataset_musdb.py` to be available."
        )
    if dataset_type == "litdata" and LitdataEQDataset is None:
        raise ValueError(
            "`data.dataset_type=litdata` requires the optional litdata dataset backend."
        )
    if hp_lp_gain_target != "zero":
        raise ValueError(
            "Only `data.hp_lp_gain_target: zero` is supported for the "
            "multi-type EQ label contract."
        )
    if (
        lambda_gain <= 0.0
        and lambda_freq <= 0.0
        and lambda_q <= 0.0
        and lambda_spectral <= 0.0
        and lambda_typed_spectral <= 0.0
        and lambda_hmag <= 0.0
        and lambda_multi_scale <= 0.0
    ):
        raise ValueError(
            "SimplifiedEQLoss requires at least one active supervision term: "
            "`loss.lambda_gain`, `loss.lambda_freq`, `loss.lambda_q`, "
            "`loss.lambda_spectral`, `loss.lambda_typed_spectral`, or "
            "`loss.lambda_hmag`, or `loss.lambda_multi_scale`."
        )
    if type_loss_mode not in {"focal", "balanced_softmax"}:
        raise ValueError(
            "`loss.type_loss_mode` must be one of {'focal', 'balanced_softmax'}."
        )
    if encoder_backend == "wav2vec2_frozen":
        if data_cfg.get("precompute_mels", False):
            raise ValueError(
                "`wav2vec2_frozen` requires raw `wet_audio`; "
                "`data.precompute_mels` must be false."
            )
        if not encoder_cfg.get("wav2vec2_checkpoint"):
            raise ValueError(
                "`model.encoder.wav2vec2_checkpoint` must be set for "
                "`encoder.backend=wav2vec2_frozen`."
            )
    if encoder_backend == "mert":
        if data_cfg.get("precompute_mels", False):
            raise ValueError(
                "`mert` requires raw `wet_audio`; "
                "`data.precompute_mels` must be false."
            )


class Trainer:
    def __init__(self, config_path="conf/config.yaml", resume_path=None, force_recompute=False):
        self.cfg = load_config(config_path)

        # AUDIT: MEDIUM-11 — Config schema validation (fail fast on typos/misconfig)
        config_issues = validate_config_schema(self.cfg)
        if config_issues:
            for issue in config_issues:
                print(f"  [config] VALIDATION ISSUE: {issue}")
            if any("missing required" in i.lower() for i in config_issues):
                raise ValueError(
                    f"Config has {len(config_issues)} validation issue(s). "
                    f"Fix required keys before training."
                )

        # AUDIT: HIGH-08 — Dependency validation (fail fast, not mid-training)
        dep_errors = validate_dependencies(self.cfg)
        if dep_errors:
            for err in dep_errors:
                print(f"  [deps] MISSING: {err}")
            raise RuntimeError(
                f"{len(dep_errors)} dependency issue(s) found. "
                f"Install missing packages or adjust config."
            )

        data_cfg = self.cfg["data"]
        model_cfg = self.cfg["model"]
        loss_cfg = self.cfg["loss"]
        trainer_cfg = self.cfg["trainer"]

        trusted_roots = [
            Path.cwd().resolve(),
            Path(__file__).resolve().parent,
            Path(__file__).resolve().parent.parent,
        ]
        for root in trainer_cfg.get("trusted_artifact_roots", []):
            trusted_roots.append(Path(root).expanduser().resolve())
        # Deduplicate while preserving order.
        self.trusted_artifact_roots = list(dict.fromkeys(trusted_roots))
        self.resume_path = None
        if resume_path is not None:
            self.resume_path = str(
                resolve_trusted_artifact_path(
                    resume_path,
                    allowed_roots=self.trusted_artifact_roots,
                    must_exist=True,
                )
            )

        self.seed = int(self.cfg.get("seed", 42))
        self.deterministic = bool(trainer_cfg.get("deterministic", False))
        # AUDIT: V-20 — Get num_workers early for deterministic mode warning
        # Default to auto-detected value for warning purposes
        import os
        default_num_workers = min(max(1, (os.cpu_count() or 4) - 1), 8)
        self.num_workers = data_cfg.get("num_workers", default_num_workers)
        set_global_seed(self.seed, deterministic=self.deterministic, num_workers=self.num_workers)
        torch.backends.cudnn.benchmark = bool(
            trainer_cfg.get("cudnn_benchmark", not self.deterministic)
        ) and not self.deterministic

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_bands = data_cfg["num_bands"]
        self.sample_rate = data_cfg["sample_rate"]
        self.max_epochs = trainer_cfg["max_epochs"]
        self.log_every = trainer_cfg["log_every_n_steps"]
        self.n_fft = data_cfg.get("n_fft", 2048)
        self.hop_length = data_cfg.get("hop_length", 256)
        self.gain_bounds = tuple(data_cfg.get("gain_bounds", (-24.0, 24.0)))
        self.monitor_val_metric = trainer_cfg.get(
            "monitor_val_metric", "primary_val_score"
        )
        self.val_compute_soft_every_n = trainer_cfg.get("val_compute_soft_every_n", 5)
        self.gain_mix_value = 0.3
        self.curriculum_stages = self.cfg.get("curriculum", {}).get("stages", [])
        self._current_curriculum_stage_idx = None
        self._curriculum_warned_precomputed = False
        self.current_epoch = 0
        self._received_signal = None
        self._signal_handlers_registered = False
        self.force_recompute = force_recompute  # AUDIT: CRITICAL-02

        # Optimization flags from config
        self.use_8bit_optimizer = (
            trainer_cfg.get("use_8bit_optimizer", False) and HAS_BITSANDBYTES
        )
        self.use_torch_compile = trainer_cfg.get("use_torch_compile", False)
        self.use_gradient_checkpointing = trainer_cfg.get(
            "use_gradient_checkpointing", False
        )
        self.use_activation_quantization = trainer_cfg.get(
            "use_activation_quantization", False
        )
        self.use_deepspeed = trainer_cfg.get("use_deepspeed", False) and HAS_DEEPSPEED
        self.precompute_cache_path = trainer_cfg.get("precompute_cache_path", None)
        if self.precompute_cache_path:
            self.precompute_cache_path = str(
                resolve_trusted_artifact_path(
                    self.precompute_cache_path,
                    allowed_roots=self.trusted_artifact_roots,
                    must_exist=False,
                )
            )
        self.validate_render_audio = trainer_cfg.get("validate_render_audio", False)
        # AUDIT: LOW-34 — Profiling infrastructure
        self.profile_n_batches = 0  # Set via --profile CLI flag

        if self.use_8bit_optimizer:
            print("  [opt] Using 8-bit AdamW optimizer (bitsandbytes)")
        if self.use_torch_compile:
            print("  [opt] Using torch.compile for graph optimization")
        if self.use_gradient_checkpointing:
            print("  [opt] Using gradient checkpointing for TCN stacks")
        if self.use_activation_quantization:
            print("  [opt] Using INT8 activation quantization")
        if self.use_deepspeed:
            print("  [opt] Using DeepSpeed ZeRO-2 optimization")

        # Dataset selection: "musdb" for real audio, "synthetic" (or unset) for synthetic
        dataset_type = str(data_cfg.get("dataset_type", "synthetic")).lower()
        dataset_size = data_cfg.get("dataset_size", 30000)
        val_dataset_size = data_cfg.get("val_dataset_size", 2000)
        shared_ds_kwargs = dict(
            num_bands=self.num_bands,
            sample_rate=self.sample_rate,
            duration=data_cfg.get("audio_duration", 1.5),
            n_fft=self.n_fft,
            gain_range=tuple(data_cfg["gain_bounds"]),
            freq_range=tuple(data_cfg["freq_bounds"]),
            q_range=tuple(data_cfg["q_bounds"]),
            type_weights=data_cfg.get("type_weights", None),
            hp_lp_gain_target=data_cfg.get("hp_lp_gain_target", "zero"),
            # AUDIT: V-09 — Pass signal type weights from config
            signal_type_weights=data_cfg.get("signal_type_weights", None),
            # AUDIT: V-06 — Pass gain distribution from config
            gain_distribution=data_cfg.get("gain_distribution", "beta"),
            precompute_mels=data_cfg.get("precompute_mels", True),
            n_mels=data_cfg.get("n_mels", 128),
            base_seed=self.seed,
        )
        train_ds_kwargs = {
            **shared_ds_kwargs,
            "size": dataset_size,
            "base_seed": self.seed,
        }
        val_ds_kwargs = {
            **shared_ds_kwargs,
            "size": val_dataset_size,
            "base_seed": self.seed + 1_000_000,
        }
        synthetic_train_kwargs = {
            **train_ds_kwargs,
            "duration_range": tuple(data_cfg["duration_range"])
            if data_cfg.get("duration_range") is not None
            else None,
        }
        synthetic_val_kwargs = {
            **val_ds_kwargs,
            "duration_range": tuple(data_cfg["duration_range"])
            if data_cfg.get("duration_range") is not None
            else None,
        }

        if dataset_type == "litdata":
            litdata_dir = data_cfg.get("litdata_dir")
            if LitdataEQDataset is None:
                raise RuntimeError(
                    "`dataset_type=litdata` was selected but `dataset_litdata.py` is unavailable."
                )
            print(f"  [data] Using litdata streaming dataset from {litdata_dir}")
            self.train_dataset = LitdataEQDataset(
                data_dir=litdata_dir,
                num_bands=self.num_bands,
                duration=data_cfg.get("audio_duration", 5.0),
                sample_rate=self.sample_rate,
                n_mels=data_cfg.get("n_mels", 128),
            )
            # litdata streams from disk — no precompute step needed
        elif dataset_type == "musdb":
            musdb_root = data_cfg.get("musdb_root", "")
            musdb_subsets = data_cfg.get("musdb_subsets", ["train", "test"])
            print(f"  [data] Using MUSDB18 dataset from {musdb_root}")
            self.train_dataset = MUSDB18EQDataset(
                musdb_root=musdb_root,
                subsets=musdb_subsets,
                **train_ds_kwargs,
            )
        else:
            print(f"  [data] Using synthetic dataset")
            self.train_dataset = SyntheticEQDataset(**synthetic_train_kwargs)

        # Try loading precomputed cache from disk (skip for litdata)
        if dataset_type != "litdata":
            if self.precompute_cache_path and self.train_dataset.load_precomputed(
                self.precompute_cache_path, force_recompute=self.force_recompute
            ):
                print(f"  [data] Loaded cached dataset from {self.precompute_cache_path}")
            elif data_cfg.get("precompute_mels", True):
                # Precompute all samples with mel-spectrograms upfront
                self.train_dataset.precompute()
                # Save cache if path specified
                if self.precompute_cache_path:
                    self.train_dataset.save_precomputed(self.precompute_cache_path)
            else:
                print(f"  [data] On-the-fly generation (no precompute)")

        # Fixed validation dataset — generated once with full type distribution.
        if dataset_type == "litdata":
            litdata_val_dir = data_cfg.get("litdata_val_dir", None)
            if litdata_val_dir:
                self.val_dataset = LitdataEQDataset(
                    data_dir=litdata_val_dir,
                    num_bands=self.num_bands,
                    duration=data_cfg.get("audio_duration", 5.0),
                    sample_rate=self.sample_rate,
                    n_mels=data_cfg.get("n_mels", 128),
                )
            else:
                # Split training set: use last val_dataset_size samples for validation
                total = len(self.train_dataset)
                val_size = min(val_dataset_size, total // 10)
                train_size = total - val_size
                print(f"  [data] Splitting litdata: {train_size:,} train / {val_size:,} val")
                self.train_dataset, self.val_dataset = random_split(
                    self.train_dataset,
                    [train_size, val_size],
                    generator=torch.Generator().manual_seed(self.seed + 17),
                )
        elif dataset_type == "musdb":
            self.val_dataset = MUSDB18EQDataset(
                musdb_root=musdb_root,
                subsets=musdb_subsets,
                **val_ds_kwargs,
            )
        else:
            self.val_dataset = SyntheticEQDataset(
                **synthetic_val_kwargs,
            )
        if dataset_type != "litdata":
            if data_cfg.get("precompute_mels", True) or data_cfg.get(
                "freeze_val_set", True
            ):
                self.val_dataset.precompute()

            # Train split: fold val portion into training since we have a separate val set.
            n_train = int(
                len(self.train_dataset) * (data_cfg["train_split"] + data_cfg["val_split"])
            )
            n_test = len(self.train_dataset) - n_train
            self.train_set, self.test_set = random_split(
                self.train_dataset,
                [n_train, n_test],
                generator=torch.Generator().manual_seed(self.seed + 33),
            )
        else:
            # litdata: train_dataset/val_dataset are already split above
            self.train_set = self.train_dataset
            self.test_set = self.val_dataset

        # Pin memory for async GPU transfer
        pin_memory = self.device.type == "cuda"
        # AUDIT: HIGH-14 — num_workers already set in __init__ (for V-20 deterministic warning)
        num_workers = self.num_workers
        if num_workers == 0:
            print(
                "  [data] WARNING: num_workers=0 — data generation blocks GPU. "
                f"Set num_workers>=2 in config (auto-detected {os.cpu_count()} CPUs)"
            )
        train_loader_generator = torch.Generator().manual_seed(self.seed + 101)
        val_loader_generator = torch.Generator().manual_seed(self.seed + 202)
        worker_init = partial(seed_worker, base_seed=self.seed)

        self.train_loader = DataLoader(
            self.train_set,
            batch_size=data_cfg["batch_size"],
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=True,
            prefetch_factor=8 if num_workers > 0 else None,
            persistent_workers=True if num_workers > 0 else False,
            generator=train_loader_generator,
            worker_init_fn=worker_init,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=data_cfg["batch_size"],
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            prefetch_factor=8 if num_workers > 0 else None,
            persistent_workers=True if num_workers > 0 else False,
            generator=val_loader_generator,
            worker_init_fn=worker_init,
        )

        self.run_dir = Path("checkpoints")
        self.run_dir.mkdir(exist_ok=True)
        self.run_metadata_path = self.run_dir / "run_metadata.json"
        self.training_events_path = self.run_dir / "training_events.jsonl"
        self._write_run_metadata(
            config_path=config_path,
            resume_path=self.resume_path,
            dataset_type=dataset_type,
        )

        # Model
        enc_cfg = model_cfg.get("encoder", {})
        self.model = StreamingTCNModel(
            n_mels=data_cfg.get("n_mels", 128),
            embedding_dim=enc_cfg.get("embedding_dim", 128),
            num_bands=self.num_bands,
            channels=enc_cfg.get("channels", 128),
            num_blocks=enc_cfg.get("num_blocks", 4),
            num_stacks=enc_cfg.get("num_stacks", 2),
            kernel_size=enc_cfg.get("kernel_size", 3),
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            type_conditioned_frequency=model_cfg.get(
                "type_conditioned_frequency", True
            ),
            dropout=enc_cfg.get("dropout", 0.1),
            mel_noise_std=enc_cfg.get("mel_noise_std", 0.0),
            n_shelf_bands=model_cfg.get("n_shelf_bands", 16),
            encoder_backend=enc_cfg.get("backend", "hybrid_tcn"),
            wav2vec2_checkpoint=enc_cfg.get(
                "wav2vec2_checkpoint",
                "facebook/wav2vec2-base",
            ),
            ast_model_name=enc_cfg.get("ast_model_name", "vit_small_patch16_224"),
            ast_checkpoint_path=enc_cfg.get("ast_checkpoint_path", ""),
            clap_model_path=enc_cfg.get("clap_model_path"),
            mert_checkpoint=enc_cfg.get("mert_checkpoint", "m-a-p/MERT-v1-95M"),
            two_stage=model_cfg.get("two_stage", False),
            hierarchical_type_head=model_cfg.get("hierarchical_type_head", False),
        ).to(self.device)

        # Spectral pretrain initialization: load encoder weights from a pre-trained
        # spectral model so the encoder starts with robust spectral features.
        spectral_pretrain_path = model_cfg.get("spectral_pretrain_path", None)
        if spectral_pretrain_path and self.resume_path is None:
            sp_path = resolve_trusted_artifact_path(
                spectral_pretrain_path,
                allowed_roots=self.trusted_artifact_roots,
                must_exist=False,
            )
            if sp_path.exists():
                sp_state = torch.load(
                    sp_path,
                    map_location=self.device,
                    weights_only=True,
                )
                sp_sd = sp_state.get("model_state_dict", sp_state)
                # Strip torch.compile prefix if present
                if any(k.startswith("_orig_mod.") for k in sp_sd):
                    sp_sd = {k.replace("_orig_mod.", "", 1): v for k, v in sp_sd.items()}
                # Load only matching encoder keys (skip param_head, dsp_cascade)
                current_sd = self.model.state_dict()
                loaded, skipped = 0, 0
                for k, v in sp_sd.items():
                    if k in current_sd and current_sd[k].shape == v.shape and "encoder" in k:
                        current_sd[k] = v
                        loaded += 1
                    else:
                        skipped += 1
                self.model.load_state_dict(current_sd)
                print(f"  [pretrain] Loaded {loaded} encoder weights from {sp_path} (skipped {skipped})")
            else:
                print(f"  [pretrain] WARNING: spectral pretrain not found at {sp_path}")

        # STFT frontend
        self.frontend = STFTFrontend(
            n_fft=self.n_fft,
            hop_length=data_cfg.get("hop_length", 256),
            win_length=self.n_fft,
            mel_bins=data_cfg.get("n_mels", 128),
            sample_rate=self.sample_rate,
        ).to(self.device)

        # Loss — spectral-dominant with parameter supervision
        self.criterion = SimplifiedEQLoss(
            lambda_param=loss_cfg.get("lambda_param", 1.0),
            lambda_spectral=loss_cfg.get("lambda_spectral", 5.0),
            lambda_type=loss_cfg.get("lambda_type", 2.0),
            lambda_hmag=loss_cfg.get("lambda_hmag", 0.3),
            lambda_coarse_type=loss_cfg.get("lambda_coarse_type", 0.0),
            lambda_triplet=loss_cfg.get("lambda_triplet", 0.0),
            triplet_margin=loss_cfg.get("triplet_margin", 0.2),
            triplet_max_pairs=loss_cfg.get("triplet_max_pairs", 128),
            lambda_gain=loss_cfg.get("lambda_gain", 1.0),
            lambda_freq=loss_cfg.get("lambda_freq", 1.0),
            lambda_q=loss_cfg.get("lambda_q", 1.0),
            lambda_gain_zero=loss_cfg.get("lambda_gain_zero", 1.0),
            label_smoothing=loss_cfg.get("label_smoothing", 0.02),
            focal_gamma=loss_cfg.get("focal_gamma", 2.0),
            type_class_priors=data_cfg.get("type_weights", None),
            type_loss_mode=loss_cfg.get("type_loss_mode", "focal"),
            lambda_type_match=loss_cfg.get("lambda_type_match", 0.0),
            lambda_type_prior=loss_cfg.get("lambda_type_prior", 0.0),
            lambda_type_entropy=loss_cfg.get("lambda_type_entropy", 0.0),
            lambda_embed_var=loss_cfg.get("lambda_embed_var", 0.5),
            lambda_contrastive=loss_cfg.get("lambda_contrastive", 0.0),
            embed_var_threshold=loss_cfg.get("embed_var_threshold", 0.1),
            class_weight_multipliers=self.cfg.get("loss", {}).get("class_weight_multipliers", None),
            sign_penalty_weight=self.cfg.get("loss", {}).get("sign_penalty_weight", 0.0),
            lambda_hdb=self.cfg.get("loss", {}).get("lambda_hdb", 2.0),
            lambda_typed_spectral=self.cfg.get("loss", {}).get("lambda_typed_spectral", 2.0),
            lambda_film_diversity=self.cfg.get("loss", {}).get("lambda_film_diversity", 0.0),
            lambda_multi_scale=self.cfg.get("loss", {}).get("lambda_multi_scale", 1.0),
            lambda_perceptual=self.cfg.get("loss", {}).get("lambda_perceptual", 2.0),
            lambda_shape=self.cfg.get("loss", {}).get("lambda_shape", 1.0),
            lambda_activity=self.cfg.get("loss", {}).get("lambda_activity", 0.1),
            lambda_spread=self.cfg.get("loss", {}).get("lambda_spread", 0.05),
            perceptual_mel_bins=loss_cfg.get("perceptual_mel_bins", 64),
            perceptual_mfcc_bins=loss_cfg.get("perceptual_mfcc_bins", 20),
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            dsp_cascade=self.model.dsp_cascade,
        )
        dataset_type_prior = get_dataset_type_prior(self.train_set)
        if dataset_type_prior is not None:
            self.criterion.update_type_priors(dataset_type_prior)
        self.metric_lambda_type_match = loss_cfg.get("lambda_type_match", 1.0)

        # Hungarian matcher for fair validation metrics (matches target ordering to predictions)
        self.matcher = HungarianBandMatcher(
            lambda_gain=1.0,
            lambda_freq=1.0,
            lambda_q=1.0,
            lambda_type_match=loss_cfg.get("lambda_type_match", 1.0),
        )

        # Optimizer: the parameter head runs at a higher LR than the encoder
        # because the spectral trunk is deeper and adapts more slowly.
        base_lr = model_cfg["learning_rate"]
        wd = model_cfg["weight_decay"]
        head_lr_mult = model_cfg.get("head_lr_multiplier", 3.0)
        # Optional separate backbone LR (for unfrozen wav2vec2 fine-tuning)
        backbone_lr = model_cfg.get("encoder", {}).get("backbone_lr", None)

        # Classify parameters into 3 groups:
        #   1. backbone (wav2vec2 transformer) — lowest LR when unfrozen
        #   2. encoder other (layer_weights, temporal_pool, output_proj) — base LR
        #   3. param_head — highest LR
        backbone_params = []
        encoder_params = []
        head_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "param_head" in name:
                head_params.append(param)
            elif backbone_lr is not None and "encoder.backbone" in name:
                backbone_params.append(param)
            else:
                encoder_params.append(param)

        loss_params = [param for param in self.criterion.parameters() if param.requires_grad]

        if backbone_params:
            print(f"  [opt] LR groups: backbone={backbone_lr:.1e}, encoder={base_lr:.1e}, head={base_lr*head_lr_mult:.1e} ({head_lr_mult:.1f}x)")
            print(f"  [opt] Backbone trainable params: {sum(p.numel() for p in backbone_params):,}")
        else:
            print(f"  [opt] LR groups: encoder={base_lr:.1e}, head={base_lr*head_lr_mult:.1e} ({head_lr_mult:.1f}x)")

        param_groups = []
        if backbone_params:
            param_groups.append({
                "params": backbone_params,
                "lr": backbone_lr,
                "weight_decay": wd,
                "initial_lr": backbone_lr,
            })
        param_groups.extend([
            {
                "params": encoder_params,
                "lr": base_lr,
                "weight_decay": wd,
                "initial_lr": base_lr,
            },
            {
                "params": head_params,
                "lr": base_lr * head_lr_mult,
                "weight_decay": wd,
                "initial_lr": base_lr * head_lr_mult,
            },
        ])
        if loss_params:
            print(f"  [opt] Criterion trainables: {sum(p.numel() for p in loss_params):,}")
            param_groups.append({
                "params": loss_params,
                "lr": base_lr,
                "weight_decay": 0.0,
                "initial_lr": base_lr,
            })

        if self.use_8bit_optimizer:
            self.optimizer = bnb.optim.AdamW8bit(param_groups)
        else:
            self.optimizer = torch.optim.AdamW(
                param_groups,
                fused=self.device.type == "cuda",
            )
        self._optimizer_includes_backbone = bool(backbone_params)
        # CosineAnnealingWarmRestarts — cyclical LR to escape plateaus
        # T_0=30: first cycle length, T_mult=2: double each cycle (30, 60, 120...)
        # Steps once per epoch via scheduler.step() in fit()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=30, T_mult=2, eta_min=1e-6
        )

        # H-01: Linear LR warmup for the first warmup_epochs (default 5).
        # Wraps the cosine schedule: during warmup, LR ramps linearly from 0 to
        # the value the cosine schedule would produce, then hands off to cosine.
        self.warmup_epochs = trainer_cfg.get("warmup_epochs", 5)
        if self.warmup_epochs > 0:
            # Save the cosine scheduler as inner; create a LambdaLR that scales it
            self._inner_scheduler = self.scheduler
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=self._warmup_cosine_lr_lambda,
            )
            print(f"  [opt] Linear warmup for {self.warmup_epochs} epochs, then cosine restarts")

        self.gradient_accumulation_steps = trainer_cfg.get(
            "gradient_accumulation_steps", 1
        )
        self.global_step = 0
        self.best_monitor_value = (
            float("-inf")
            if metric_direction(self.monitor_val_metric) == "max"
            else float("inf")
        )
        self.best_named_metrics = {
            "primary_val_score": float("inf"),
            "gain_mae_db_matched": float("inf"),
            "type_accuracy_matched": float("-inf"),
            "val_spectral_loss": float("inf"),
        }
        self.patience_counter = 0
        self.early_stopping_patience = trainer_cfg.get("early_stopping_patience", 5)
        self.min_epochs_before_early_stop = trainer_cfg.get(
            "min_epochs_before_early_stop", 15
        )
        self.alert_thresholds = trainer_cfg.get(
            "alert_thresholds",
            {
                "gain_mae_db_matched": 4.0,
                "primary_val_score": 8.0,
                "type_accuracy_matched_min": 0.70,
            },
        )
        self.fail_on_alert = bool(trainer_cfg.get("fail_on_alert", False))
        self.alert_grace_epochs = int(trainer_cfg.get("alert_grace_epochs", 0))
        self.gain_mae_ema = None  # C-02: track for checkpoint save/restore
        self.history = []
        self.start_epoch = 1
        self._total_nan_grad_steps = 0  # C-04: cumulative NaN gradient step counter
        self._oom_batch_size_floor = None  # C-03: track reduced batch size after OOM
        # AUDIT: MEDIUM-12 — Data distribution tracking for monitoring training stability
        self._distribution_stats = {
            "gain_mean": [],
            "gain_std": [],
            "freq_mean": [],
            "type_counts": [],
        }

        if self.resume_path is not None:
            self._load_checkpoint(self.resume_path)
        self.criterion.to(self.device)

        # bf16-mixed precision: autocast context (no GradScaler needed for bf16)
        self.use_bf16 = trainer_cfg.get("precision", "").lower() == "bf16-mixed"
        if self.use_bf16:
            device_type = "cuda" if self.device.type == "cuda" else "cpu"
            self.autocast_ctx = torch.autocast(device_type, dtype=torch.bfloat16)
        else:
            self.autocast_ctx = nullcontext()

        # DeepSpeed initialization
        if self.use_deepspeed:
            ds_config_path = trainer_cfg.get(
                "deepspeed_config", "conf/deepspeed_config.json"
            )
            if os.path.exists(ds_config_path):
                with open(ds_config_path, "r") as f:
                    ds_config = json.load(f)
                self.model, self.optimizer, _, _ = deepspeed.initialize(
                    model=self.model,
                    optimizer=self.optimizer,
                    config=ds_config,
                )
                print(f"  [opt] DeepSpeed initialized with config: {ds_config_path}")
            else:
                print(
                    f"  [opt] WARNING: DeepSpeed config not found at {ds_config_path}, disabling"
                )
                self.use_deepspeed = False

        # torch.compile after DeepSpeed init to avoid conflicts
        if self.use_torch_compile:
            self.model = torch.compile(self.model)
            print("  [opt] torch.compile applied to model")

        # AUDIT: CRITICAL-30 — Structured logging with WandB/TensorBoard support
        log_cfg = trainer_cfg.get("logging", {})
        self.logger = StructuredLogger(
            log_dir=str(self.run_dir),
            enable_wandb=log_cfg.get("enable_wandb", False),
            wandb_project=log_cfg.get("wandb_project", "idsp-eq"),
            wandb_run_name=self.cfg.get("experiment_name", None),
            enable_tensorboard=log_cfg.get("enable_tensorboard", False),
        )

        self._register_signal_handlers()
        self._append_event(
            "trainer_initialized",
            dataset_type=str(data_cfg.get("dataset_type", "synthetic")).lower(),
            train_examples=len(self.train_set),
            val_examples=len(self.val_dataset),
            batch_size=data_cfg["batch_size"],
            device=str(self.device),
        )

    def _write_run_metadata(self, config_path, resume_path, dataset_type):
        payload = {
            "created_at": utc_now_iso(),
            "config_path": str(Path(config_path).resolve()),
            "resume_path": str(Path(resume_path).resolve()) if resume_path else None,
            "dataset_type": dataset_type,
            "seed": self.seed,
            "deterministic": self.deterministic,
            "device": str(self.device),
            "monitor_val_metric": self.monitor_val_metric,
            "precompute_cache_path": self.precompute_cache_path,
            "max_epochs": self.max_epochs,
            "train_examples": len(self.train_set),
            "val_examples": len(self.val_dataset),
            "test_examples": len(self.test_set),
        }
        with open(self.run_metadata_path, "w") as f:
            json.dump(payload, f, indent=2)

    def _append_event(self, event, **payload):
        record = {
            "timestamp": utc_now_iso(),
            "event": event,
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            **payload,
        }
        with open(self.training_events_path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def _register_signal_handlers(self):
        if self._signal_handlers_registered:
            return
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self._handle_signal)
        self._signal_handlers_registered = True

    def _warmup_cosine_lr_lambda(self, epoch):
        """H-01: LR multiplier that linearly ramps from 0→1 over warmup_epochs,
        then delegates to the inner cosine schedule.  The LambdaLR scheduler
        calls this once per .step(), which happens once per epoch in fit()."""
        if self.warmup_epochs <= 0 or epoch >= self.warmup_epochs:
            # After warmup: step the inner cosine scheduler and use its LR ratio
            # LambdaLR applies this multiplier to initial_lr, so we need to return
            # the ratio that the cosine schedule would produce.
            return 1.0  # cosine schedule handles the rest via its own state
        # During warmup: linear ramp from near-zero to 1.0
        return min(1.0, max(0.01, epoch / self.warmup_epochs))

    def _handle_signal(self, signum, _frame):
        if self._received_signal is not None:
            return
        try:
            signal_name = signal.Signals(signum).name
        except ValueError:
            signal_name = str(signum)
        self._received_signal = signal_name
        print(f"  [signal] Received {signal_name}; saving emergency checkpoint")
        self._append_event("signal_received", signal=signal_name)
        try:
            self.save_emergency_checkpoint(signal_name)
        except Exception as exc:
            print(f"  [signal] Emergency checkpoint failed: {exc}")

    def save_emergency_checkpoint(self, reason):
        checkpoint_path = self.run_dir / "interrupted.pt"
        if self.use_deepspeed:
            interrupted_dir = self.run_dir / "interrupted"
            self.model.save_checkpoint(str(interrupted_dir), tag="signal")
            self._append_event(
                "emergency_checkpoint",
                reason=reason,
                path=str(interrupted_dir),
            )
            return interrupted_dir

        model_ref = self.model.module if hasattr(self.model, "module") else self.model
        # C-07: Save full training state for reliable resume
        state = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),  # C-01
            "monitor_metric": self.monitor_val_metric,
            "monitor_value": self.best_monitor_value,
            "global_step": self.global_step,
            "curriculum_stage_idx": self._current_curriculum_stage_idx,  # C-02
            "gain_mae_ema": self.gain_mae_ema,  # C-02
            "best_named_metrics": dict(self.best_named_metrics),
            "gumbel_tau": (
                model_ref.param_head.gumbel_tau.item()
                if hasattr(model_ref, 'param_head') and hasattr(model_ref.param_head, 'gumbel_tau')
                else None
            ),  # C-07
            "loss_weights": {  # C-07
                "lambda_gain": self.criterion.lambda_gain,
                "lambda_freq": self.criterion.lambda_freq,
                "lambda_q": self.criterion.lambda_q,
                "lambda_spectral": self.criterion.lambda_spectral,
                "lambda_type": self.criterion.lambda_type,
                "lambda_hdb": self.criterion.lambda_hdb,
            },
            "signal_reason": reason,
            "saved_at": utc_now_iso(),
        }
        # AUDIT: LOW-16 — Add version hash for code tracking
        try:
            state["code_version_hash"] = compute_version_hash()
        except Exception:
            state["code_version_hash"] = "unknown"
        tmp_path = checkpoint_path.with_suffix('.pt.tmp')
        torch.save(state, tmp_path)
        tmp_path.rename(checkpoint_path)
        self._append_event(
            "emergency_checkpoint",
            reason=reason,
            path=str(checkpoint_path),
        )
        return checkpoint_path

    def _prepare_wet_audio(self, batch):
        if "wet_audio" not in batch:
            raise ValueError(
                "Audio-first training requires `wet_audio` in every batch. "
                "Disable mel-only precompute."
            )
        return batch["wet_audio"].to(self.device, non_blocking=True)

    def _prepare_dry_audio(self, batch):
        if "dry_audio" not in batch:
            raise ValueError(
                "Audio-first training requires `dry_audio` in every batch. "
                "Disable mel-only precompute or frozen-audio caching that drops dry audio."
            )
        return batch["dry_audio"].to(self.device, non_blocking=True)

    def _prepare_input(self, batch, wet_audio=None):
        """Get mel-spectrogram from precomputed cache or compute on-the-fly."""
        if "wet_mel" in batch:
            return batch["wet_mel"].to(self.device, non_blocking=True)
        if wet_audio is None:
            wet_audio = self._prepare_wet_audio(batch)
        mel_spec = self.frontend.mel_spectrogram(wet_audio)
        return mel_spec.squeeze(1)

    def _update_distribution_stats(self, batch, pred_gain, pred_freq, pred_ft):
        """
        Update tracking statistics for data distribution monitoring (AUDIT: MEDIUM-12).
        Helps detect training instability or data drift issues.
        """
        with torch.no_grad():
            # Target gain distribution
            target_gain = batch["gain"].to(self.device)
            self._distribution_stats["gain_mean"].append(target_gain.mean().item())
            self._distribution_stats["gain_std"].append(target_gain.std().item())

            # Target frequency distribution (log scale for better monitoring)
            target_freq = batch["freq"].to(self.device)
            self._distribution_stats["freq_mean"].append(torch.log(target_freq + 1e-8).mean().item())

            # Type distribution
            target_ft = batch["filter_type"].to(self.device)
            for ti in range(5):
                count = (target_ft == ti).sum().item()
                if len(self._distribution_stats["type_counts"]) <= ti:
                    self._distribution_stats["type_counts"].append([])
                self._distribution_stats["type_counts"][ti].append(count)

    def _log_distribution_stats(self, epoch):
        """
        Log accumulated distribution statistics (AUDIT: MEDIUM-12).
        Called once per epoch to summarize distribution monitoring.
        """
        if not self._distribution_stats["gain_mean"]:
            return

        print(f"  [dist] Epoch {epoch} target distribution:")
        print(f"    gain: mean={np.mean(self._distribution_stats['gain_mean']):.3f} dB, "
              f"std={np.mean(self._distribution_stats['gain_std']):.3f} dB")
        print(f"    freq (log Hz): mean={np.mean(self._distribution_stats['freq_mean']):.3f}")
        if self._distribution_stats["type_counts"]:
            type_names = ["peak", "lshf", "hshf", "hpas", "lpas"]
            total = sum(sum(self._distribution_stats["type_counts"][ti]) for ti in range(5))
            for ti, name in enumerate(type_names):
                count = sum(self._distribution_stats["type_counts"][ti]) if self._distribution_stats["type_counts"][ti] else 0
                pct = 100 * count / total if total > 0 else 0
                print(f"    {name}: {pct:.1f}%")

        # Log to structured logger
        self.logger.log_metric(
            "dist/gain_mean_db", np.mean(self._distribution_stats["gain_mean"]),
            epoch=epoch, step=self.global_step
        )
        self.logger.log_metric(
            "dist/gain_std_db", np.mean(self._distribution_stats["gain_std"]),
            epoch=epoch, step=self.global_step
        )

        # Reset stats for next epoch
        self._distribution_stats = {
            "gain_mean": [],
            "gain_std": [],
            "freq_mean": [],
            "type_counts": [],
        }

    def _curriculum_stage_index(self, epoch):
        if not self.curriculum_stages:
            return None

        cumulative_epochs = 0
        for idx, stage in enumerate(self.curriculum_stages):
            cumulative_epochs += int(stage.get("epochs", 0))
            if epoch <= cumulative_epochs:
                return idx
        return len(self.curriculum_stages) - 1

    def _apply_curriculum_stage(self, epoch):
        # AUDIT: MEDIUM-03 — Single source of truth for warmup.
        # The loss-level warmup (in loss_multitype.py) is the authoritative mechanism.
        # epoch counter is set here; gain MAE EMA is updated each batch via update_gain_mae().
        # The H_db ramp (Fix 7) was removed to consolidate warmup logic into loss_multitype.py.
        self.criterion.current_epoch = epoch

        # Gumbel temperature annealing for differentiable type selection
        model_ref = self.model.module if hasattr(self.model, "module") else self.model
        gumbel_cfg = self.cfg.get("gumbel", {})
        if hasattr(model_ref, 'param_head'):
            ph = model_ref.param_head
            if hasattr(ph, 'gumbel_tau'):
                start_tau = gumbel_cfg.get("start_tau", 2.0)
                min_tau = gumbel_cfg.get("min_tau", 0.1)
                warmup = gumbel_cfg.get("warmup_epochs", 10)
                if epoch <= warmup:
                    tau = start_tau
                else:
                    progress = (epoch - warmup) / max(1, self.max_epochs - warmup)
                    tau = min_tau + (start_tau - min_tau) * 0.5 * (1 + math.cos(math.pi * progress))
                ph.gumbel_tau.fill_(tau)

        # FiLM diversity loss: penalize near-zero gamma norms
        if hasattr(model_ref, 'param_head') and hasattr(model_ref.param_head, 'type_film_gamma'):
            film_gamma_norm = model_ref.param_head.type_film_gamma.abs().mean()
            self.criterion.lambda_film_diversity = self.cfg.get("loss", {}).get("lambda_film_diversity", 0.0)
            # Store norm for loss computation
            self.criterion._film_gamma_norm = film_gamma_norm

        # AST / Wav2Vec2 Encoder freeze schedule
        freeze_epochs = self.cfg.get("model", {}).get("encoder", {}).get("freeze_epochs", 0)
        if hasattr(model_ref, "encoder") and hasattr(model_ref.encoder, "freeze_backbone"):
            if epoch <= freeze_epochs:
                model_ref.encoder.freeze_backbone()
            else:
                model_ref.encoder.unfreeze_backbone()
                # When backbone is newly unfrozen, its params are not yet in the optimizer.
                # Rebuild optimizer param groups to include them.
                self._rebuild_optimizer_if_needed()

        # Type classifier pretrain phase
        # If true, first 5 epochs only use type loss and 0 for others to prevent param regression interference
        type_pretrain_epochs = self.cfg.get("model", {}).get("type_pretrain_epochs", 0)
        if epoch <= type_pretrain_epochs:
            self.criterion.lambda_gain = 0.0
            self.criterion.lambda_freq = 0.0
            self.criterion.lambda_q = 0.0
            self.criterion.lambda_spectral = 0.0
            self.criterion.lambda_typed_spectral = 0.0
            self.criterion.lambda_hdb = 0.0
        else:
            # Restore to config defaults
            loss_cfg = self.cfg.get("loss", {})
            self.criterion.lambda_gain = loss_cfg.get("lambda_gain", 1.0)
            self.criterion.lambda_freq = loss_cfg.get("lambda_freq", 1.0)
            self.criterion.lambda_q = loss_cfg.get("lambda_q", 1.0)
            self.criterion.lambda_spectral = loss_cfg.get("lambda_spectral", 5.0)
            self.criterion.lambda_typed_spectral = loss_cfg.get("lambda_typed_spectral", 2.0)
            # AUDIT: MEDIUM-03 — lambda_hdb uses config default (H_db ramp removed, consolidated into loss warmup)
            self.criterion.lambda_hdb = loss_cfg.get("lambda_hdb", 2.0)

        stage_idx = self._curriculum_stage_index(epoch)
        if stage_idx is None:
            return

        stage = self.curriculum_stages[stage_idx]

        # C-06: Always apply curriculum to the loss criterion (type priors, lambda
        # weights, gumbel tau) even when dataset has precomputed cache.  Only skip
        # dataset-level overrides (apply_curriculum_stage on the dataset itself).
        for weight_name in ["lambda_gain", "lambda_freq", "lambda_q", "lambda_type", "lambda_spectral", "lambda_type_match",
                            "matcher_lambda_gain", "matcher_lambda_freq", "matcher_lambda_q", "matcher_lambda_type_match"]:
            if weight_name in stage:
                setattr(self.criterion, weight_name, float(stage[weight_name]))

        # Update type priors on criterion even with cached data
        prior_getter = getattr(self.train_dataset, "get_type_prior", None)
        update_priors = getattr(self.criterion, "update_type_priors", None)
        if callable(prior_getter) and callable(update_priors):
            current_prior = prior_getter()
            if current_prior is not None:
                update_priors(current_prior)

        if hasattr(self.train_dataset, "_cache"):
            if not self._curriculum_warned_precomputed:
                print(
                    "  [curriculum] train dataset is precomputed; dataset-level curriculum "
                    "updates are skipped, but loss criterion is still updated"
                )
                self._curriculum_warned_precomputed = True
            # Skip only dataset-level overrides, continue to stage change logging
        else:
            apply_stage_to_training_state(self.train_dataset, self.criterion, stage)
        focus = "shelf_focus" if stage_idx in (0, 1) else "balanced"
        set_focus = getattr(self.train_dataset, "set_adversarial_focus", None)
        if callable(set_focus):
            set_focus(focus)

        if stage_idx != self._current_curriculum_stage_idx:
            self._current_curriculum_stage_idx = stage_idx
            stage_name = stage.get("name", f"stage_{stage_idx}")
            weights_str = f"L_type={self.criterion.lambda_type:.2f} | L_gain={self.criterion.lambda_gain:.2f} | L_freq={self.criterion.lambda_freq:.2f} | L_q={self.criterion.lambda_q:.2f} | L_spec={self.criterion.lambda_spectral:.2f}"
            print(
                f"  [curriculum] epoch={epoch} stage={stage_name} "
                f"filter_types={stage.get('filter_types', 'all')} "
                f"gain_bounds={stage.get('gain_bounds', self.gain_bounds)} "
                f"q_bounds={stage.get('q_bounds', 'default')} "
                f"type_weights={stage.get('type_weights', 'base')} "
                f"adversarial_fraction={stage.get('adversarial_fraction', getattr(self.train_dataset, 'adversarial_fraction', 'default'))} "
                f"weights: {weights_str}"
            )
            # AUDIT: MEDIUM-33 — Log curriculum stage change as structured event
            self._append_event(
                "curriculum_stage_changed",
                epoch=epoch,
                stage_name=stage_name,
                stage_index=stage_idx,
                filter_types=stage.get("filter_types", "all"),
                gain_bounds=stage.get("gain_bounds", self.gain_bounds),
                q_bounds=stage.get("q_bounds", "default"),
                type_weights=stage.get("type_weights", "base"),
                lambda_type=self.criterion.lambda_type,
                lambda_gain=self.criterion.lambda_gain,
                lambda_freq=self.criterion.lambda_freq,
                lambda_q=self.criterion.lambda_q,
                lambda_spectral=self.criterion.lambda_spectral,
            )

    def _gain_aux_alignment(self, batch, wet_mel, output):
        gain_aux_summary = output.get("gain_aux_summary")
        if "dry_audio" not in batch or gain_aux_summary is None:
            return None
        model_ref = self.model.module if hasattr(self.model, "module") else self.model
        dry_audio = batch["dry_audio"].to(self.device, non_blocking=True)
        with torch.no_grad():
            dry_mel = self.frontend.mel_spectrogram(dry_audio).squeeze(1).float()
            mel_delta = (wet_mel.detach().float() - dry_mel).mean(dim=-1)
            gain_delta_summary = model_ref.param_head.summarize_gain_aux_features(
                mel_delta
            )
            if gain_delta_summary is None:
                return None
            lhs = F.normalize(gain_aux_summary.detach().float(), dim=-1)
            rhs = F.normalize(gain_delta_summary.detach().float(), dim=-1)
            alignment = F.cosine_similarity(lhs, rhs, dim=-1).mean()
            if torch.isfinite(alignment):
                return alignment.item()
        return None

    def train_one_epoch(self, epoch):
        self.model.train()
        self.criterion.train()
        epoch_loss = 0.0
        n_batches = 0
        n_nan_batches = 0

        # Accumulate per-component losses for epoch-level logging
        component_accum = {}

        self.optimizer.zero_grad(set_to_none=True)
        model_ref = self.model.module if hasattr(self.model, "module") else self.model

        # AUDIT: LOW-34 — Profiling setup
        prof = None
        if self.profile_n_batches > 0 and epoch == self.start_epoch:
            prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    wait=0, warmup=1, active=self.profile_n_batches
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    str(self.run_dir / "profile_trace")
                ),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            )
            prof.start()
            print(f"  [profile] Profiling first {self.profile_n_batches} batches")

        # H-10: tqdm progress bar for epoch training loop
        pbar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc=f"  epoch {epoch}",
            leave=False,
            ncols=120,
        )
        for batch_idx, batch in pbar:
            # H-17: Check signal every batch (not just at loop start)
            if self._received_signal is not None:
                print(f"  [signal] Stopping train loop after {self._received_signal}")
                break
            wet_audio = batch.get("wet_audio", None)
            if wet_audio is not None:
                wet_audio = wet_audio.to(self.device, non_blocking=True)
            dry_audio = batch.get("dry_audio", None)
            if dry_audio is not None:
                dry_audio = dry_audio.to(self.device, non_blocking=True)
            mel_frames = self._prepare_input(batch, wet_audio=wet_audio)
            target_gain = batch["gain"].to(self.device, non_blocking=True)
            target_freq = batch["freq"].to(self.device, non_blocking=True)
            target_q = batch["q"].to(self.device, non_blocking=True)
            target_ft = batch["filter_type"].to(self.device, non_blocking=True)

            # Forward (wrapped in autocast for bf16-mixed precision)
            # C-03: OOM detection and recovery — wrap forward+backward in try/except
            try:
                with self.autocast_ctx:
                    output = self.model(mel_frames, wet_audio=wet_audio)
                    pred_gain, pred_freq, pred_q = output["params"]

                    # Ground truth frequency response
                    target_H_mag = model_ref.dsp_cascade(
                        target_gain,
                        target_freq,
                        target_q,
                        filter_type=target_ft,
                        n_fft=self.n_fft,
                    )

                    # Per-band H_db target (not cascade product)
                    # Each band has its own frequency response — the H_db head
                    # predicts per-band responses so gain extraction works correctly.
                    b0, b1, b2, a1, a2 = model_ref.dsp_cascade.compute_biquad_coeffs_multitype(
                        target_gain, target_freq, target_q, target_ft
                    )
                    H_mag_per_band = model_ref.dsp_cascade.freq_response(
                        b0, b1, b2, a1, a2, n_fft=self.n_fft
                    )  # (B, num_bands, n_fft_bins)
                    # C-05: Compute spectral loss components in fp32 to prevent bf16
                    # underflow in log-domain operations (log10, log mel filterbank)
                    with torch.amp.autocast('cuda', enabled=False):
                        h_db_target = 20.0 * torch.log10(
                            H_mag_per_band.float().clamp(min=1e-6)
                        )
                        h_db_pred = (
                            output.get("h_db_pred").float()
                            if output.get("h_db_pred") is not None
                            else None
                        )
                    # Breaks spectral shortcut — wrong params + correct types != target
                    H_mag_typed = model_ref.dsp_cascade(
                        pred_gain,
                        pred_freq,
                        pred_q,
                        filter_type=target_ft,
                        n_fft=self.n_fft,
                    )

                    total_loss, components = self.criterion(
                        pred_gain,
                        pred_freq,
                        pred_q,
                        output["type_logits"],
                        output["H_mag_soft"],
                        output.get("H_mag", output["H_mag_soft"]),
                        target_gain,
                        target_freq,
                        target_q,
                        target_ft,
                        target_H_mag,
                        embedding=output["embedding"],
                        h_db_pred=h_db_pred,
                        h_db_target=h_db_target,
                        H_mag_typed=H_mag_typed,
                        type_probs=output["type_probs"],
                        hier_aux=output.get("hier_aux"),
                    )
            except torch.cuda.OutOfMemoryError:
                # C-03: OOM recovery — halve batch size, clear cache, log, continue
                torch.cuda.empty_cache()
                old_bs = self.train_loader.batch_size
                new_bs = max(1, old_bs // 2)
                if self._oom_batch_size_floor is None:
                    self._oom_batch_size_floor = new_bs
                print(
                    f"  [oom] CUDA OOM at batch {batch_idx}! "
                    f"Reducing batch_size {old_bs} -> {new_bs}, clearing cache. "
                    f"Skipping batch."
                )
                self._append_event(
                    "oom_recovery",
                    epoch=epoch,
                    batch_idx=batch_idx,
                    old_batch_size=old_bs,
                    new_batch_size=new_bs,
                )
                self.optimizer.zero_grad(set_to_none=True)
                continue

            # Skip batch if loss is NaN (from early training instability)
            if not torch.isfinite(total_loss):
                # Identify which component caused NaN
                nan_components = [
                    k
                    for k, v in components.items()
                    if isinstance(v, torch.Tensor) and not torch.isfinite(v).all()
                ]
                print(
                    f"  [nan] Non-finite loss at train step {self.global_step}: "
                    f"components={nan_components}"
                )
                self.optimizer.zero_grad(set_to_none=True)
                n_nan_batches += 1
                if n_nan_batches >= 10:
                    print(
                        f"  [train] WARNING: {n_nan_batches} NaN batches in epoch {epoch}, stopping epoch early"
                    )
                    break
                continue

            # Scale loss for gradient accumulation
            scaled_loss = total_loss / self.gradient_accumulation_steps

            if self.use_deepspeed:
                self.model.backward(scaled_loss)
            else:
                scaled_loss.backward()

            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if not self.use_deepspeed:
                    # C-04: Detect NaN gradients before clipping — identify which params
                    # have NaN and skip the optimizer step to prevent weight corruption.
                    nan_grad_params = []
                    for name, param in self.model.named_parameters():
                        if param.grad is not None and not torch.isfinite(param.grad).all():
                            nan_grad_params.append(name)
                            param.grad.zero_()
                    if nan_grad_params:
                        self._total_nan_grad_steps += 1
                        print(
                            f"  [nan-grad] step={self.global_step}: NaN gradients in "
                            f"{len(nan_grad_params)} params — skipping optimizer step. "
                            f"First few: {nan_grad_params[:5]}"
                        )
                        self.optimizer.zero_grad(set_to_none=True)
                        n_nan_batches += 1
                        continue

                    # Zero out remaining NaN gradients before clipping
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            grad_norm = param.grad.norm()
                            if not torch.isfinite(grad_norm):
                                param.grad.zero_()

                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )
                    if not torch.isfinite(grad_norm):
                        self.optimizer.zero_grad(set_to_none=True)
                        n_nan_batches += 1
                        continue

                    # Gradient explosion monitoring: warn when clipping is severe
                    clip_ratio = grad_norm.item() / 1.0  # grad_norm / max_norm
                    if clip_ratio > 2.0 and self.global_step % self.log_every == 0:
                        print(f"  [WARN] Heavy gradient clipping at step {self.global_step}: "
                              f"grad_norm={grad_norm.item():.2f}, clip_ratio={clip_ratio:.1f}x")

                    # Per-parameter gradient norm monitoring (logged every N steps)
                    if self.global_step % self.log_every == 0:
                        grad_parts = []
                        for name, param in self.model.named_parameters():
                            if param.grad is not None and param.grad.norm() > 0:
                                if "gain_head" in name:
                                    grad_parts.append(("gain", param.grad.norm().item()))
                                elif "q_head" in name:
                                    grad_parts.append(("q", param.grad.norm().item()))
                                elif "type_head" in name or "type_mel_proj" in name:
                                    grad_parts.append(("type", param.grad.norm().item()))
                                elif "freq_head" in name or "freq_context_proj" in name:
                                    grad_parts.append(("freq", param.grad.norm().item()))
                                elif "param_head" in name:
                                    # Catch-all for trunk, cnn_merge, query_proj, etc.
                                    grad_parts.append(("param_head_other", param.grad.norm().item()))
                                elif "encoder.backbone" in name:
                                    grad_parts.append(("backbone", param.grad.norm().item()))
                                elif "encoder" in name:
                                    grad_parts.append(("encoder", param.grad.norm().item()))
                        if grad_parts:
                            grouped = {}
                            for k, v in grad_parts:
                                grouped.setdefault(k, []).append(v)
                            avg_grads = {k: sum(v) / len(v) for k, v in grouped.items()}
                            grad_str = " | ".join(
                                f"grad_{k}={v:.4f}"
                                for k, v in sorted(avg_grads.items())
                            )
                            print(f"  [grads] step={self.global_step} {grad_str}")
                            # AUDIT: CRITICAL-30 — Log gradient norms to structured logger
                            self.logger.log_metrics_batch(
                                {f"grad_norm/{k}": v for k, v in avg_grads.items()},
                                step=self.global_step,
                            )

                    self.optimizer.step()

                    # AUDIT: CRITICAL-07 — Sanitize optimizer state: reset any Adam momentum/variance
                    # buffers that contain NaN (prevents permanent training death).
                    # Also reset the 'step' count which can become NaN/inf and prevent convergence.
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor) and not torch.isfinite(v).all():
                                v.zero_()
                            # Reset scalar state values (step count) that may be NaN/inf
                            elif isinstance(v, float) and not math.isfinite(v):
                                state[k] = 0.0
                else:
                    self.model.step()

                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1

            # AUDIT: LOW-34 — Profiler step
            if prof is not None:
                prof.step()

            epoch_loss += total_loss.item()
            n_batches += 1

            # Accumulate per-component losses
            for k, v in components.items():
                val = v.item() if isinstance(v, torch.Tensor) else v
                component_accum[k] = component_accum.get(k, 0.0) + val

            if self.global_step % self.log_every == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                parts = [
                    f"step={self.global_step} loss={total_loss.item():.4f} lr={lr:.2e}"
                ]
                for k, v in components.items():
                    parts.append(f"{k}={v:.4f}")
                # Log embedding variance diagnostic (monitors anti-collapse health)
                if "embed_var_loss" in components:
                    embed_var = output["embedding"].var(dim=0).mean().item()
                    parts.append(f"embed_var={embed_var:.4f}")
                gain_aux_alignment = self._gain_aux_alignment(batch, mel_frames, output)
                if gain_aux_alignment is not None:
                    parts.append(f"gain_aux_delta_cos={gain_aux_alignment:.4f}")
                print(f"  [train] " + " | ".join(parts))

            # H-10: Update tqdm progress bar with current loss and LR
            if n_batches > 0:
                lr_now = self.optimizer.param_groups[0]["lr"]
                pbar.set_postfix(
                    loss=f"{total_loss.item():.4f}",
                    lr=f"{lr_now:.2e}",
                    step=self.global_step,
                    refresh=False,
                )

            # H-08: Per-type accuracy logging (training) — compute from type_probs
            if self.global_step % self.log_every == 0 and "type_loss" in components:
                with torch.no_grad():
                    pred_types = output["type_probs"].argmax(dim=-1)  # (B, num_bands)
                    type_names_short = ["peak", "lshf", "hshf", "hpas", "lpas"]
                    for ti, tn in enumerate(type_names_short):
                        mask = target_ft == ti
                        if mask.any():
                            acc = (pred_types[mask] == ti).float().mean().item()
                            self.logger.log_metric(
                                f"train/type_accuracy_{tn}", acc,
                                epoch=epoch, step=self.global_step,
                            )
                # AUDIT: MEDIUM-12 — Update distribution statistics for monitoring
                self._update_distribution_stats(batch, pred_gain, pred_freq, target_ft)

        if n_nan_batches > 0:
            print(f"  [train] Epoch {epoch}: {n_nan_batches} NaN batches skipped")

        # AUDIT: LOW-34 — Stop profiler and print summary
        if prof is not None:
            prof.stop()
            print(f"  [profile] Trace saved to {self.run_dir / 'profile_trace'}")
            # Print top time-consuming operations
            summary = prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)
            print(f"  [profile] Top 10 CUDA operations:\n{summary}")

        # Per-component epoch averages
        if n_batches > 0:
            comp_strs = []
            for k in sorted(component_accum.keys()):
                avg = component_accum[k] / n_batches
                comp_strs.append(f"{k}={avg:.4f}")
            print(f"  [train] Epoch {epoch} components: " + " | ".join(comp_strs))

        # AUDIT: MEDIUM-12 — Log distribution statistics once per epoch
        self._log_distribution_stats(epoch)

        return epoch_loss / max(n_batches, 1)

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        self.criterion.eval()
        val_loss_soft = 0.0
        val_loss_hard = 0.0
        val_spectral_loss_soft = 0.0
        val_spectral_loss_hard = 0.0
        n_batches = 0
        metric_tensors = {
            "gain_pred": [],
            "gain_gt": [],
            "freq_pred": [],
            "freq_gt": [],
            "q_pred": [],
            "q_gt": [],
            "type_pred": [],
            "type_gt": [],
            "type_logits": [],
        }
        val_component_accum = {}
        shelf_diag = {
            "pred_lowshelf_count": 0,
            "pred_highshelf_count": 0,
            "low_shelf_bias_sum": 0.0,
            "low_shelf_bias_count": 0,
            "high_shelf_bias_sum": 0.0,
            "high_shelf_bias_count": 0,
        }
        model_ref = self.model.module if hasattr(self.model, "module") else self.model
        compute_soft = (epoch % self.val_compute_soft_every_n == 0)

        for batch in self.val_loader:
            wet_audio = batch.get("wet_audio", None)
            if wet_audio is not None:
                wet_audio = wet_audio.to(self.device, non_blocking=True)
            dry_audio = batch.get("dry_audio", None)
            if dry_audio is not None:
                dry_audio = dry_audio.to(self.device, non_blocking=True)
            mel_frames = self._prepare_input(batch, wet_audio=wet_audio)
            target_gain = batch["gain"].to(self.device, non_blocking=True)
            target_freq = batch["freq"].to(self.device, non_blocking=True)
            target_q = batch["q"].to(self.device, non_blocking=True)
            target_ft = batch["filter_type"].to(self.device, non_blocking=True)

            with self.autocast_ctx:
                output_hard = self.model(
                    mel_frames=mel_frames,
                    wet_audio=wet_audio,
                    hard_types=True,
                )
                if compute_soft:
                    output_soft = self.model(
                        mel_frames=mel_frames,
                        wet_audio=wet_audio,
                        hard_types=False,
                        force_soft_response=True,
                    )
                else:
                    output_soft = output_hard
                pred_gain, pred_freq, pred_q = output_hard["params"]
                pred_audio_hard = None
                pred_audio_soft = None
                if self.validate_render_audio:
                    pred_audio_hard = model_ref.dsp_cascade.process_audio(
                        dry_audio,
                        output_hard["params"][0],
                        output_hard["params"][1],
                        output_hard["params"][2],
                        filter_type=output_hard["filter_type"],
                        n_fft=self.n_fft,
                        hop_length=self.hop_length,
                    )
                    pred_audio_soft = model_ref.dsp_cascade.process_audio(
                        dry_audio,
                        output_soft["params"][0],
                        output_soft["params"][1],
                        output_soft["params"][2],
                        type_probs=output_soft["type_probs"],
                        n_fft=self.n_fft,
                        hop_length=self.hop_length,
                    )

                # NaN diagnostic
                diagnostic_tensors = [
                    ("pred_gain", pred_gain),
                    ("pred_freq", pred_freq),
                    ("pred_q", pred_q),
                    ("embedding", output_hard["embedding"]),
                    ("H_mag", output_hard["H_mag"]),
                    ("H_mag_soft", output_soft["H_mag_soft"]),
                ]
                if pred_audio_hard is not None:
                    diagnostic_tensors.append(("pred_audio_hard", pred_audio_hard))
                if pred_audio_soft is not None:
                    diagnostic_tensors.append(("pred_audio_soft", pred_audio_soft))
                for name, tensor in diagnostic_tensors:
                    if not torch.isfinite(tensor).all():
                        print(f"  [nan] {name} has NaN/inf at val batch {n_batches}")

                target_H_mag = model_ref.dsp_cascade(
                    target_gain,
                    target_freq,
                    target_q,
                    filter_type=target_ft,
                    n_fft=self.n_fft,
                )
                b0, b1, b2, a1, a2 = model_ref.dsp_cascade.compute_biquad_coeffs_multitype(
                    target_gain, target_freq, target_q, target_ft
                )
                H_mag_per_band = model_ref.dsp_cascade.freq_response(
                    b0, b1, b2, a1, a2, n_fft=self.n_fft
                )
                h_db_target = 20.0 * torch.log10(H_mag_per_band.clamp(min=1e-6))
                H_mag_typed_soft = model_ref.dsp_cascade(
                    output_soft["params"][0],
                    output_soft["params"][1],
                    output_soft["params"][2],
                    filter_type=target_ft,
                    n_fft=self.n_fft,
                )
                H_mag_typed_hard = model_ref.dsp_cascade(
                    output_hard["params"][0],
                    output_hard["params"][1],
                    output_hard["params"][2],
                    filter_type=target_ft,
                    n_fft=self.n_fft,
                )

                total_loss_soft, components = self.criterion(
                    output_soft["params"][0],
                    output_soft["params"][1],
                    output_soft["params"][2],
                    output_soft["type_logits"],
                    output_soft["H_mag_soft"],
                    output_soft.get("H_mag", output_soft["H_mag_soft"]),
                    target_gain,
                    target_freq,
                    target_q,
                    target_ft,
                    target_H_mag,
                    embedding=output_soft["embedding"],
                    h_db_pred=output_soft.get("h_db_pred"),
                    h_db_target=h_db_target,
                    H_mag_typed=H_mag_typed_soft,
                    type_probs=output_soft["type_probs"],
                    hier_aux=output_soft.get("hier_aux"),
                )
                total_loss_hard, hard_components = self.criterion(
                    output_hard["params"][0],
                    output_hard["params"][1],
                    output_hard["params"][2],
                    output_hard["type_logits"],
                    output_hard["H_mag"],
                    output_hard["H_mag"],
                    target_gain,
                    target_freq,
                    target_q,
                    target_ft,
                    target_H_mag,
                    embedding=output_hard["embedding"],
                    h_db_pred=output_hard.get("h_db_pred"),
                    h_db_target=h_db_target,
                    H_mag_typed=H_mag_typed_hard,
                    type_probs=output_hard["type_probs"],
                    hier_aux=output_hard.get("hier_aux"),
                )

                # NaN diagnostic: identify which loss component causes NaN
                if not torch.isfinite(total_loss_soft):
                    nan_components = [
                        k
                        for k, v in components.items()
                        if isinstance(v, torch.Tensor) and not torch.isfinite(v).all()
                    ]
                    print(
                        f"  [nan] Non-finite loss at val batch {n_batches}: "
                        f"components={nan_components}"
                    )
                    total_loss_soft = torch.tensor(1e4, device=pred_gain.device)
                if not torch.isfinite(total_loss_hard):
                    total_loss_hard = torch.tensor(1e4, device=pred_gain.device)

            val_loss_soft += total_loss_soft.item()
            val_loss_hard += total_loss_hard.item()
            val_spectral_loss_soft += components.get(
                "spectral_loss", torch.tensor(0.0)
            ).item()
            val_spectral_loss_hard += hard_components.get(
                "spectral_loss", torch.tensor(0.0)
            ).item()
            n_batches += 1

            # D-02: Accumulate loss components for validation logging
            for k, v in components.items():
                val = v.item() if isinstance(v, torch.Tensor) else float(v)
                val_component_accum[k] = val_component_accum.get(k, 0.0) + val

            matched_filter_type = self.matcher(
                pred_gain,
                pred_freq,
                pred_q,
                target_gain,
                target_freq,
                target_q,
                target_filter_type=target_ft,
                pred_type_logits=output_hard["type_logits"],
            )[3]
            pred_filter_type = output_hard["filter_type"]
            shelf_diag["pred_lowshelf_count"] += int(
                (pred_filter_type == FILTER_LOWSHELF).sum().item()
            )
            shelf_diag["pred_highshelf_count"] += int(
                (pred_filter_type == FILTER_HIGHSHELF).sum().item()
            )
            shelf_bias = output_hard.get("shelf_bias")
            if shelf_bias is not None:
                low_mask = matched_filter_type == FILTER_LOWSHELF
                high_mask = matched_filter_type == FILTER_HIGHSHELF
                if low_mask.any():
                    shelf_diag["low_shelf_bias_sum"] += (
                        shelf_bias[..., 0][low_mask].float().sum().item()
                    )
                    shelf_diag["low_shelf_bias_count"] += int(low_mask.sum().item())
                if high_mask.any():
                    shelf_diag["high_shelf_bias_sum"] += (
                        shelf_bias[..., 1][high_mask].float().sum().item()
                    )
                    shelf_diag["high_shelf_bias_count"] += int(high_mask.sum().item())

            metric_tensors["gain_pred"].append(pred_gain.detach().float().cpu())
            metric_tensors["gain_gt"].append(target_gain.detach().float().cpu())
            metric_tensors["freq_pred"].append(pred_freq.detach().float().cpu())
            metric_tensors["freq_gt"].append(target_freq.detach().float().cpu())
            metric_tensors["q_pred"].append(pred_q.detach().float().cpu())
            metric_tensors["q_gt"].append(target_q.detach().float().cpu())
            metric_tensors["type_pred"].append(output_hard["filter_type"].detach().cpu())
            metric_tensors["type_gt"].append(target_ft.detach().cpu())
            metric_tensors["type_logits"].append(
                output_hard["type_logits"].detach().float().cpu()
            )

        avg_val_loss_soft = val_loss_soft / max(n_batches, 1)
        avg_val_loss_hard = val_loss_hard / max(n_batches, 1)
        avg_val_spectral_loss_soft = val_spectral_loss_soft / max(n_batches, 1)
        avg_val_spectral_loss_hard = val_spectral_loss_hard / max(n_batches, 1)
        if metric_tensors["gain_pred"]:
            metrics = compute_eq_metrics(
                torch.cat(metric_tensors["gain_pred"]),
                torch.cat(metric_tensors["freq_pred"]),
                torch.cat(metric_tensors["q_pred"]),
                torch.cat(metric_tensors["type_pred"]),
                torch.cat(metric_tensors["type_logits"]),
                torch.cat(metric_tensors["gain_gt"]),
                torch.cat(metric_tensors["freq_gt"]),
                torch.cat(metric_tensors["q_gt"]),
                torch.cat(metric_tensors["type_gt"]),
                lambda_type_match=self.metric_lambda_type_match,
            )
        else:
            metrics = {}
        metrics["val_spectral_loss_soft"] = avg_val_spectral_loss_soft
        metrics["val_spectral_loss_hard"] = avg_val_spectral_loss_hard
        metrics["val_spectral_loss"] = avg_val_spectral_loss_hard
        metrics["val_loss_soft"] = avg_val_loss_soft
        metrics["val_loss_hard"] = avg_val_loss_hard
        metrics["val_loss"] = avg_val_loss_soft
        metrics["primary_val_score"] = compute_primary_val_score(metrics)
        metrics["pred_lowshelf_count"] = shelf_diag["pred_lowshelf_count"]
        metrics["pred_highshelf_count"] = shelf_diag["pred_highshelf_count"]
        metrics["mean_low_shelf_bias_on_lowshelf"] = (
            shelf_diag["low_shelf_bias_sum"] / shelf_diag["low_shelf_bias_count"]
            if shelf_diag["low_shelf_bias_count"] > 0
            else 0.0
        )
        metrics["mean_high_shelf_bias_on_highshelf"] = (
            shelf_diag["high_shelf_bias_sum"] / shelf_diag["high_shelf_bias_count"]
            if shelf_diag["high_shelf_bias_count"] > 0
            else 0.0
        )

        # D-02: Log averaged validation loss components
        if n_batches > 0 and val_component_accum:
            comp_strs = []
            for k in sorted(val_component_accum.keys()):
                avg = val_component_accum[k] / n_batches
                comp_strs.append(f"{k}={avg:.4f}")
            print(f"  [val] epoch={epoch} components: " + " | ".join(comp_strs))

        print(
            f"  [val] epoch={epoch} val_spectral_loss_soft={avg_val_spectral_loss_soft:.4f} "
            f"val_spectral_loss_hard={avg_val_spectral_loss_hard:.4f} "
        )
        print(
            f"  [val] epoch={epoch} val_loss_soft={avg_val_loss_soft:.4f} "
            f"val_loss_hard={avg_val_loss_hard:.4f} "
            f"primary_val_score={metrics.get('primary_val_score', 0):.4f} "
            f"gain_mae_db_matched={metrics.get('gain_mae_db_matched', 0):.2f} "
            f"gain_mae_db_raw={metrics.get('gain_mae_db_raw', 0):.2f} "
            f"freq_mae_oct_matched={metrics.get('freq_mae_oct_matched', 0):.3f} "
            f"freq_mae_oct_raw={metrics.get('freq_mae_oct_raw', 0):.3f} "
            f"q_mae_dec_matched={metrics.get('q_mae_dec_matched', 0):.3f} "
            f"type_accuracy_matched={metrics.get('type_accuracy_matched', 0):.1%} "
            f"type_accuracy_raw={metrics.get('type_accuracy_raw', 0):.1%}"
        )
        # D-11: Per-type accuracy reporting
        per_type_parts = []
        for tn in ["peaking", "lowshelf", "highshelf", "highpass", "lowpass"]:
            v = metrics.get(f"type_accuracy_{tn}_matched", 0)
            per_type_parts.append(f"{tn}={v:.1%}")
        print(f"  [val] per-type: {' | '.join(per_type_parts)}")
        print(
            f"  [val] shelf: pred_lowshelf={metrics['pred_lowshelf_count']} "
            f"pred_highshelf={metrics['pred_highshelf_count']} "
            f"low_bias_on_lowshelf={metrics['mean_low_shelf_bias_on_lowshelf']:.3f} "
            f"high_bias_on_highshelf={metrics['mean_high_shelf_bias_on_highshelf']:.3f}"
        )
        # Confusion matrix: rows=true, cols=predicted
        cm = metrics.get("confusion_matrix_matched")
        if cm is not None:
            type_names = ["peak", "lshf", "hshf", "hpas", "lpas"]
            print(f"  [val] confusion (true\\pred): {' '.join(f'{n:>6s}' for n in type_names)}")
            for i, row in enumerate(cm):
                row_total = sum(row)
                row_str = ' '.join(f'{int(v):>6d}' for v in row)
                print(f"  [val]   {type_names[i]:>4s}: {row_str}  ({row_total:.0f})")
        # Gumbel tau diagnostic
        model_ref = self.model.module if hasattr(self.model, "module") else self.model
        if hasattr(model_ref, 'param_head') and hasattr(model_ref.param_head, 'gumbel_tau'):
            print(f"  [val] gumbel_tau={model_ref.param_head.gumbel_tau.item():.4f}")
        return avg_val_loss_soft, metrics

    def save_checkpoint(self, epoch, monitor_value, save_tags=None, metrics=None, is_best=False):
        ckpt_dir = Path("checkpoints")
        ckpt_dir.mkdir(exist_ok=True)
        save_tags = list(save_tags or [])

        if self.use_deepspeed:
            # DeepSpeed handles checkpointing
            ckpt_path = ckpt_dir / f"epoch_{epoch:03d}"
            self.model.save_checkpoint(str(ckpt_path), tag=f"epoch_{epoch}")
            for tag in save_tags:
                self.model.save_checkpoint(str(ckpt_dir / tag), tag=tag)
        else:
            state = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),  # C-01
                "val_loss": monitor_value,
                "monitor_metric": self.monitor_val_metric,
                "monitor_value": monitor_value,
                "global_step": self.global_step,
                "curriculum_stage_idx": self._current_curriculum_stage_idx,  # C-02
                "gain_mae_ema": self.gain_mae_ema,  # C-02
            }
            if metrics is not None:
                state["primary_val_score"] = metrics.get(
                    "primary_val_score", monitor_value
                )
                state["val_spectral_loss"] = metrics.get(
                    "val_spectral_loss", monitor_value
                )
                state["val_spectral_loss_soft"] = metrics.get(
                    "val_spectral_loss_soft", monitor_value
                )
                state["val_spectral_loss_hard"] = metrics.get(
                    "val_spectral_loss_hard", monitor_value
                )
                state["val_audio_loss"] = state["val_spectral_loss"]
                state["val_loss_soft"] = metrics.get("val_loss_soft", monitor_value)
                state["val_loss_hard"] = metrics.get("val_loss_hard", monitor_value)
                state["gain_mae_db_matched"] = metrics.get("gain_mae_db_matched")
                state["type_accuracy_matched"] = metrics.get("type_accuracy_matched")
                state["best_named_metrics"] = dict(self.best_named_metrics)
            # AUDIT: LOW-16 — Add version hash for code tracking and checkpoint compatibility
            try:
                state["code_version_hash"] = compute_version_hash()
            except Exception:
                state["code_version_hash"] = "unknown"
            path = ckpt_dir / f"epoch_{epoch:03d}.pt"
            tmp_path = path.with_suffix('.pt.tmp')
            torch.save(state, tmp_path)
            tmp_path.rename(path)
            print(f"  Saved checkpoint: {path}")
            for tag in save_tags:
                tag_path = ckpt_dir / f"{tag}.pt"
                tmp_tag = tag_path.with_suffix('.pt.tmp')
                torch.save(state, tmp_tag)
                tmp_tag.rename(tag_path)
                print(f"  Updated checkpoint: {tag_path}")
                # AUDIT: HIGH-09 — When 'best' tag is set, also create best.pt alias
                if tag == "best":
                    best_path = ckpt_dir / "best.pt"
                    tmp_best = best_path.with_suffix('.pt.tmp')
                    torch.save(state, tmp_best)
                    tmp_best.rename(best_path)
                    print(f"  Updated checkpoint: {best_path}")
        self._append_event(
            "checkpoint_saved",
            epoch=epoch,
            monitor_value=monitor_value,
            tags=save_tags,
        )

        # Checkpoint pruning: keep only last N epoch checkpoints + named best/last
        self._prune_old_checkpoints(ckpt_dir, keep_last_n=1)

    def _prune_old_checkpoints(self, ckpt_dir, keep_last_n=3):
        """Delete old epoch_NNN.pt checkpoints, keeping the last N plus named ones."""
        epoch_ckpts = sorted(ckpt_dir.glob("epoch_*.pt"))
        if len(epoch_ckpts) <= keep_last_n:
            return
        to_delete = epoch_ckpts[:-keep_last_n]
        for p in to_delete:
            try:
                p.unlink()
                print(f"  Pruned old checkpoint: {p}")
            except OSError:
                pass

    def _has_nan_weights(self):
        """Check if any trainable parameter or active BN buffer contains NaN.

        Excludes:
        - Frozen backbone parameters (e.g., AST during freeze_epochs)
        - Non-trainable buffers
        - BatchNorm running stats that haven't been initialized yet (all zeros)
        """
        for p in self.model.parameters():
            if not p.requires_grad:
                continue
            if not torch.isfinite(p).all():
                return True
        # Only check buffers that are actually BN running stats (non-zero, updated)
        for name, buf in self.model.named_buffers():
            # Skip uninitialized BN stats (all zeros = never seen data)
            if buf.numel() == 0:
                continue
            if "running" in name.lower() and (buf == 0.0).all():
                continue
            if not torch.isfinite(buf).all():
                return True
        return False

    def _load_checkpoint(self, path):
        """Load model (and optimizer) state from a checkpoint to resume training.

        AUDIT: MEDIUM-21 — Uses weights_only=False because checkpoints contain
        optimizer state, scheduler state, and training metadata (not just weights).
        Security is mitigated by resolve_trusted_artifact_path() which validates
        that the checkpoint path is under trusted root directories.
        Never load checkpoints from untrusted sources.
        """
        path = resolve_trusted_artifact_path(
            path,
            allowed_roots=self.trusted_artifact_roots,
            must_exist=True,
        )
        state = torch.load(path, map_location=self.device, weights_only=False)
        sd = state["model_state_dict"]
        # torch.compile wraps keys with "_orig_mod." — strip it for non-compiled load
        if any(k.startswith("_orig_mod.") for k in sd):
            sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
        self.model.to(self.device)
        result, skipped_keys = self.model.load_compatible_state_dict(sd)
        if skipped_keys:
            print(
                f"  [resume] Dropped {len(skipped_keys)} incompatible keys "
                f"(shape/name mismatch after architecture change)"
            )
        if result.unexpected_keys:
            print(f"  [resume] Dropped {len(result.unexpected_keys)} extra keys from checkpoint (architecture change): {[k.split('.')[-1] for k in result.unexpected_keys]}")
        if result.missing_keys:
            print(f"  [resume] {len(result.missing_keys)} keys initialized randomly (not in checkpoint): {[k.split('.')[-1] for k in result.missing_keys]}")
        try:
            opt_sd = state["optimizer_state_dict"]
            self.optimizer.load_state_dict(opt_sd)
            # Ensure optimizer state tensors are on the same device as model params
            for s in self.optimizer.state.values():
                for k, v in s.items():
                    if isinstance(v, torch.Tensor):
                        s[k] = v.to(self.device)
        except Exception as e:
            print(f"  [resume] Could not restore optimizer state ({e}); starting fresh optimizer")
        self.global_step = state.get("global_step", 0)
        self.best_monitor_value = state.get(
            "monitor_value", state.get("val_loss", float("inf"))
        )
        for metric_name in (
            "primary_val_score",
            "val_spectral_loss",
            "gain_mae_db_matched",
            "type_accuracy_matched",
        ):
            saved_value = state.get(metric_name)
            if saved_value is not None:
                self.best_named_metrics[metric_name] = saved_value
        self.start_epoch = state.get("epoch", 0) + 1

        # C-02: Restore curriculum stage and gain_mae_ema
        saved_stage_idx = state.get("curriculum_stage_idx")
        if saved_stage_idx is not None:
            self._current_curriculum_stage_idx = saved_stage_idx
        self.gain_mae_ema = state.get("gain_mae_ema", None)

        # H-02: Reset gumbel tau based on restored epoch/curriculum stage
        # (re-applied properly via _apply_curriculum_stage before first epoch)
        restored_epoch = self.start_epoch
        model_ref_for_tau = self.model
        if hasattr(model_ref_for_tau, '_orig_mod'):
            model_ref_for_tau = model_ref_for_tau._orig_mod
        gumbel_cfg = self.cfg.get("gumbel", {})
        if hasattr(model_ref_for_tau, 'param_head') and hasattr(model_ref_for_tau.param_head, 'gumbel_tau'):
            start_tau = gumbel_cfg.get("start_tau", 2.0)
            min_tau = gumbel_cfg.get("min_tau", 0.1)
            warmup = gumbel_cfg.get("warmup_epochs", 10)
            if restored_epoch <= warmup:
                tau = start_tau
            else:
                progress = (restored_epoch - warmup) / max(1, self.max_epochs - warmup)
                tau = min_tau + (start_tau - min_tau) * 0.5 * (1 + math.cos(math.pi * progress))
            model_ref_for_tau.param_head.gumbel_tau.fill_(tau)
            print(f"  [resume] Restored gumbel_tau={tau:.4f} for epoch {restored_epoch}")

        # Always start scheduler fresh for warm restart.
        # The loaded optimizer state has initial_lr from the exhausted schedule (1e-6).
        # Reset both lr and initial_lr to config values so WarmRestarts starts from peak.
        base_lr = self.cfg["model"]["learning_rate"]
        head_lr_mult = self.cfg["model"].get("head_lr_multiplier", 1.5)
        backbone_lr = self.cfg["model"].get("encoder", {}).get("backbone_lr", None)

        # Set LR per param group based on group structure
        # With backbone_lr: groups are [backbone, encoder, head, (loss)]
        # Without backbone_lr: groups are [encoder, head, (loss)]
        if backbone_lr is not None and len(self.optimizer.param_groups) >= 3:
            # 3+ groups: backbone, encoder, head, (loss)
            self.optimizer.param_groups[0]["initial_lr"] = backbone_lr
            self.optimizer.param_groups[0]["lr"] = backbone_lr
            self.optimizer.param_groups[1]["initial_lr"] = base_lr
            self.optimizer.param_groups[1]["lr"] = base_lr
            self.optimizer.param_groups[2]["initial_lr"] = base_lr * head_lr_mult
            self.optimizer.param_groups[2]["lr"] = base_lr * head_lr_mult
            # Loss group (if present)
            for pg in self.optimizer.param_groups[3:]:
                pg["initial_lr"] = base_lr
                pg["lr"] = base_lr
            print(f"  [resume] Starting fresh LR schedule (backbone_lr={backbone_lr:.1e}, encoder_lr={base_lr:.1e}, head_lr={base_lr*head_lr_mult:.1e})")
        else:
            for pg in self.optimizer.param_groups:
                pg["initial_lr"] = base_lr
                pg["lr"] = base_lr
            # Second group (param head) gets higher LR
            if len(self.optimizer.param_groups) > 1:
                self.optimizer.param_groups[1]["initial_lr"] = base_lr * head_lr_mult
                self.optimizer.param_groups[1]["lr"] = base_lr * head_lr_mult
            print(f"  [resume] Starting fresh LR schedule (warm restart, lr={base_lr:.1e}, head_lr={base_lr*head_lr_mult:.1e})")

        print(
            f"  [resume] Loaded checkpoint from {path} "
            f"(epoch {state.get('epoch')}, monitor_value={self.best_monitor_value:.4f})"
        )
        print(f"  [resume] Resuming from epoch {self.start_epoch}")
        self._append_event(
            "checkpoint_loaded",
            path=str(path),
            resumed_epoch=self.start_epoch,
            monitor_value=self.best_monitor_value,
        )

    def _evaluate_quality_alerts(self, epoch, metrics):
        breaches = []

        gain_max = self.alert_thresholds.get("gain_mae_db_matched")
        gain_value = metrics.get("gain_mae_db_matched")
        if gain_max is not None and gain_value is not None and gain_value > float(gain_max):
            breaches.append(
                {
                    "metric": "gain_mae_db_matched",
                    "value": float(gain_value),
                    "threshold": float(gain_max),
                    "operator": ">",
                }
            )

        score_max = self.alert_thresholds.get("primary_val_score")
        score_value = metrics.get("primary_val_score")
        if score_max is not None and score_value is not None and score_value > float(score_max):
            breaches.append(
                {
                    "metric": "primary_val_score",
                    "value": float(score_value),
                    "threshold": float(score_max),
                    "operator": ">",
                }
            )

        type_min = self.alert_thresholds.get("type_accuracy_matched_min")
        type_value = metrics.get("type_accuracy_matched")
        if type_min is not None and type_value is not None and type_value < float(type_min):
            breaches.append(
                {
                    "metric": "type_accuracy_matched",
                    "value": float(type_value),
                    "threshold": float(type_min),
                    "operator": "<",
                }
            )

        if not breaches:
            return

        summary = "; ".join(
            f"{b['metric']}={b['value']:.4f} {b['operator']} {b['threshold']:.4f}"
            for b in breaches
        )
        print(f"  [alert] quality threshold breach at epoch {epoch}: {summary}")
        self._append_event("quality_alert", breaches=breaches)
        self.logger.log_event("quality_alert", {"epoch": epoch, "breaches": breaches})

        if self.fail_on_alert and epoch >= self.alert_grace_epochs:
            raise RuntimeError(
                "Quality threshold breached and `trainer.fail_on_alert=true`: "
                f"{summary}"
            )

    _optimizer_includes_backbone = False  # Track whether backbone params are in optimizer

    def _rebuild_optimizer_if_needed(self):
        """Rebuild optimizer param groups when backbone is newly unfrozen.

        When freeze_epochs > 0, the backbone starts frozen (requires_grad=False)
        so its parameters are absent from the optimizer built at __init__.
        This method detects that case and rebuilds param groups to include
        the backbone at its own LR, preserving head/encoder momentum states.
        """
        if self._optimizer_includes_backbone:
            return  # Already rebuilt

        model_cfg = self.cfg["model"]
        backbone_lr = model_cfg.get("encoder", {}).get("backbone_lr", None)
        if backbone_lr is None:
            # No backbone_lr configured — backbone params get base_lr via the
            # normal encoder group (they'll be picked up on next optimizer reset)
            backbone_lr = model_cfg["learning_rate"]

        model_ref = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model

        # Collect newly-unfrozen backbone params
        existing_param_ids = {
            id(param)
            for group in self.optimizer.param_groups
            for param in group["params"]
        }
        backbone_params = [
            p for n, p in model_ref.named_parameters()
            if "encoder.backbone" in n and p.requires_grad and id(p) not in existing_param_ids
        ]
        if not backbone_params:
            self._optimizer_includes_backbone = True
            return

        # Add as a new param group (index 0 position matters for LR schedule,
        # but AdamW just needs them in a group — order doesn't affect backward)
        self.optimizer.add_param_group({
            "params": backbone_params,
            "lr": backbone_lr,
            "weight_decay": model_cfg.get("weight_decay", 0.01),
            "initial_lr": backbone_lr,
        })

        self._optimizer_includes_backbone = True
        n_params = sum(p.numel() for p in backbone_params)
        print(f"  [opt] Added {n_params:,} backbone params to optimizer (lr={backbone_lr:.1e})")

    def _recover_from_nan(self, epoch):
        """
        Reload last good checkpoint to recover from NaN state.

        AUDIT: CRITICAL-31 — Logs recovery attempts and outcomes as structured events.
        AUDIT: MEDIUM-24 — Uses weights_only=False (required for model state dicts
        containing optimizer states), but validates loaded state dict for NaN.
        """
        ckpt_dir = Path("checkpoints")
        # Try checkpoints in reverse order (most recent first), find one that's clean
        candidates = sorted(ckpt_dir.glob("epoch_*.pt"), reverse=True)
        best_path = ckpt_dir / "best.pt"
        if best_path.exists():
            candidates.insert(0, best_path)

        attempts = 0
        for ckpt_path in candidates:
            attempts += 1
            # AUDIT: MEDIUM-24 — Validate checkpoint file exists and is readable
            if not ckpt_path.exists() or ckpt_path.stat().st_size == 0:
                continue
            state = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            # Skip checkpoints with NaN val_loss
            if not math.isfinite(state.get("val_loss", float("inf"))):
                continue
            # Load and verify no NaN in state dict
            try:
                self.model.load_state_dict(state["model_state_dict"])
            except Exception:
                continue
            if self._has_nan_weights():
                continue  # This checkpoint is also corrupted
            # Found a clean checkpoint — reset optimizer (old state may have NaN)
            base_lr = self.cfg["model"]["learning_rate"]
            wd = self.cfg["model"]["weight_decay"]
            encoder_params = []
            head_params = []
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                if "param_head" in name:
                    head_params.append(param)
                else:
                    encoder_params.append(param)
            head_lr_mult = self.cfg["model"].get("head_lr_multiplier", 3.0)
            param_groups = [
                {
                    "params": encoder_params,
                    "lr": base_lr,
                    "weight_decay": wd,
                    "initial_lr": base_lr,
                },
                {
                    "params": head_params,
                    "lr": base_lr * head_lr_mult,
                    "weight_decay": wd,
                    "initial_lr": base_lr * head_lr_mult,
                },
            ]
            self.optimizer = torch.optim.AdamW(param_groups)
            self.global_step = state.get("global_step", 0)

            # H-18: Reduce LR by 0.5x after NaN recovery to stabilize training
            for pg in self.optimizer.param_groups:
                pg["lr"] = pg["lr"] * 0.5
                pg["initial_lr"] = pg["initial_lr"] * 0.5
            print(f"  [recovery] Reduced LR by 0.5x for stability (encoder_lr={self.optimizer.param_groups[0]['lr']:.2e})")

            # C-04/H-03: Restore scheduler state and best_named_metrics
            if "scheduler_state_dict" in state:
                try:
                    self.scheduler.load_state_dict(state["scheduler_state_dict"])
                except Exception:
                    pass  # Scheduler rebuild is acceptable fallback
            if "best_named_metrics" in state:
                for k, v in state["best_named_metrics"].items():
                    if k in self.best_named_metrics:
                        self.best_named_metrics[k] = v

            print(
                f"  [recovery] Reloaded clean checkpoint from {ckpt_path}, "
                f"restored scheduler + best_metrics, reset optimizer"
            )
            # AUDIT: MEDIUM-33 — Log NaN recovery as structured event
            self._append_event(
                "nan_recovery_successful",
                epoch=epoch,
                recovery_checkpoint=str(ckpt_path),
                attempts_tried=attempts,
                global_step=self.global_step,
            )
            self.logger.log_event(
                "nan_recovery",
                {"epoch": epoch, "checkpoint": str(ckpt_path), "attempts": attempts},
            )
            return True

        # AUDIT: MEDIUM-33 — Log failed recovery attempt
        self._append_event(
            "nan_recovery_failed",
            epoch=epoch,
            checkpoints_checked=attempts,
        )
        print(f"  [recovery] No clean checkpoint found. Cannot recover.")
        return False

    def fit(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        model_ref = self.model.module if hasattr(self.model, "module") else self.model
        print(f"IDSP Multi-Type EQ Estimator Training")
        print(f"  Device: {self.device}")
        print(f"  Parameters: {total_params:,}")
        print(
            f"  Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}"
        )
        print(f"  Max epochs: {self.max_epochs}")
        print(f"  Receptive field: {model_ref.receptive_field_frames} frames")
        print()
        self._append_event(
            "training_started",
            total_params=total_params,
            train_batches=len(self.train_loader),
            val_batches=len(self.val_loader),
        )

        consecutive_nan_epochs = 0

        for epoch in range(self.start_epoch, self.max_epochs + 1):
            if self._received_signal is not None:
                break
            t0 = time.time()
            self.current_epoch = epoch

            self._apply_curriculum_stage(epoch)

            train_loss = self.train_one_epoch(epoch)

            if self._received_signal is not None:
                print(f"  [signal] Training interrupted by {self._received_signal}")
                break

            # Check for NaN in weights AND buffers (BatchNorm running stats)
            if self._has_nan_weights():
                print(
                    f"  [nan] NaN in model state after epoch {epoch}. "
                    f"Attempting checkpoint recovery..."
                )
                if self._recover_from_nan(epoch):
                    consecutive_nan_epochs = 0
                    continue  # Retry this epoch with recovered model
                else:
                    print(f"  ERROR: Cannot recover. Stopping.")
                    break

            val_loss, metrics = self.validate(epoch)
            # H-14: Step scheduler accounting for gradient accumulation.
            # Scheduler steps once per epoch (epoch-level schedule), but if using
            # grad accumulation, the effective steps per epoch are fewer, so we
            # compensate by tracking the epoch-level step count.
            self.scheduler.step()
            # H-01: After warmup, also step the inner cosine schedule
            if self.warmup_epochs > 0 and hasattr(self, '_inner_scheduler') and epoch > self.warmup_epochs:
                self._inner_scheduler.step()

            # Curriculum schedule for H_db gain mix: ramp self.gain_mix_value → 0.85 over training
            # Start higher (e.g. 0.3) since H_db head is now stronger with expanded arch.
            # Cap at 0.85 to keep 15% raw path as regularizer.
            model_ref = self.model.module if hasattr(self.model, "module") else self.model
            mix_schedule = min(0.85, self.gain_mix_value + (0.85 - self.gain_mix_value) * (epoch - 1) / max(self.max_epochs - 1, 1))
            model_ref.param_head.gain_mix_value.fill_(mix_schedule)
            if epoch <= 3 or epoch % 5 == 0:
                print(f"  [schedule] gain_mix={mix_schedule:.3f} (H_db weight)")
            elapsed = time.time() - t0

            # Track consecutive NaN val epochs (death spiral detection)
            if math.isnan(val_loss) or math.isnan(train_loss):
                consecutive_nan_epochs += 1
                if consecutive_nan_epochs >= 2:
                    print(
                        f"  [nan] NaN loss for {consecutive_nan_epochs} consecutive epochs. "
                        f"Recovering from best checkpoint..."
                    )
                    if self._recover_from_nan(epoch):
                        consecutive_nan_epochs = 0
                        continue
                    else:
                        print(f"  ERROR: Cannot recover. Stopping.")
                        break
            else:
                consecutive_nan_epochs = 0

            monitor_value = resolve_monitor_value(
                metrics, self.monitor_val_metric, val_loss
            )
            self._evaluate_quality_alerts(epoch, metrics)
            # AUDIT: HIGH-09 — Single source of truth for is_best (used for early stopping + checkpoints)
            is_best = metric_improved(
                self.monitor_val_metric, monitor_value, self.best_monitor_value
            )
            if is_best:
                self.best_monitor_value = monitor_value
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            # Only save best.pt (primary metric) and last.pt (resume point).
            # Additional metric tracking is logged; separate files are unnecessary.
            save_tags = ["last"]
            if is_best:
                save_tags.append("best")
            # Track best individual metrics for logging only
            primary_score = metrics.get("primary_val_score")
            if primary_score is not None and metric_improved(
                "primary_val_score",
                primary_score,
                self.best_named_metrics["primary_val_score"],
            ):
                self.best_named_metrics["primary_val_score"] = primary_score
            gain_mae = metrics.get("gain_mae_db_matched")
            if gain_mae is not None and metric_improved(
                "gain_mae_db_matched",
                gain_mae,
                self.best_named_metrics["gain_mae_db_matched"],
            ):
                self.best_named_metrics["gain_mae_db_matched"] = gain_mae
            type_acc = metrics.get("type_accuracy_matched")
            if type_acc is not None and metric_improved(
                "type_accuracy_matched",
                type_acc,
                self.best_named_metrics["type_accuracy_matched"],
            ):
                self.best_named_metrics["type_accuracy_matched"] = type_acc
            spectral_loss = metrics.get("val_spectral_loss")
            if spectral_loss is not None and metric_improved(
                "val_spectral_loss",
                spectral_loss,
                self.best_named_metrics["val_spectral_loss"],
            ):
                self.best_named_metrics["val_spectral_loss"] = spectral_loss
            # Always save last.pt for resume capability
            save_tags.append("last")
            self.save_checkpoint(epoch, monitor_value, save_tags, metrics=metrics, is_best=is_best)

            self.history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": monitor_value,
                    "metrics": metrics,
                    "time_s": elapsed,
                }
            )
            self._append_event(
                "epoch_completed",
                epoch=epoch,
                train_loss=train_loss,
                monitor_value=monitor_value,
                primary_val_score=metrics.get("primary_val_score"),
                gain_mae_db_matched=metrics.get("gain_mae_db_matched"),
                type_accuracy_matched=metrics.get("type_accuracy_matched"),
            )

            # AUDIT: CRITICAL-30 — Structured logging of epoch metrics
            self.logger.log_metrics_batch(
                {
                    "epoch/train_loss": train_loss,
                    "epoch/val_loss": monitor_value,
                    "epoch/val_loss_soft": metrics.get("val_loss_soft", float("nan")),
                    "epoch/gain_mae_db_matched": metrics.get("gain_mae_db_matched", float("nan")),
                    "epoch/gain_mae_db_raw": metrics.get("gain_mae_db_raw", float("nan")),
                    "epoch/freq_mae_oct_matched": metrics.get("freq_mae_oct_matched", float("nan")),
                    "epoch/q_mae_dec_matched": metrics.get("q_mae_dec_matched", float("nan")),
                    "epoch/type_accuracy_matched": metrics.get("type_accuracy_matched", float("nan")),
                    "epoch/primary_val_score": metrics.get("primary_val_score", float("nan")),
                    "epoch/lr": self.optimizer.param_groups[0]["lr"],
                    "epoch/time_s": elapsed,
                    "epoch/patience_counter": self.patience_counter,
                },
                epoch=epoch,
                step=self.global_step,
            )
            # Per-type accuracy logging (AUDIT: MEDIUM-33 — enhanced events)
            for tn in ["peaking", "lowshelf", "highshelf", "highpass", "lowpass"]:
                v = metrics.get(f"type_accuracy_{tn}_matched")
                if v is not None:
                    self.logger.log_metric(
                        f"epoch/type_accuracy_{tn}", v, epoch=epoch, step=self.global_step
                    )

            print(
                f"Epoch {epoch}/{self.max_epochs} ({elapsed:.1f}s) "
                f"train={train_loss:.4f} "
                f"{self.monitor_val_metric}={monitor_value:.4f} "
                f"val_spectral_loss={metrics.get('val_spectral_loss', float('nan')):.4f} "
                f"val_loss_hard={metrics.get('val_loss_hard', float('nan')):.4f}"
            )

            # Early stopping: stop if no improvement for patience epochs
            # AUDIT: HIGH-32 — Minimum epochs before early stopping can trigger
            min_epochs_es = self.min_epochs_before_early_stop
            if epoch < min_epochs_es:
                # Reset patience counter if we're still in warmup period
                if self.patience_counter > 0 and epoch < min_epochs_es:
                    self.patience_counter = max(0, self.patience_counter - 1)
            if self.patience_counter >= self.early_stopping_patience:
                print(
                    f"  [early_stop] No improvement for {self.patience_counter} epochs "
                    f"(min_epochs_before_early_stop={min_epochs_es} satisfied). "
                    f"Stopping early at epoch {epoch}."
                )
                self._append_event(
                    "early_stopping",
                    epoch=epoch,
                    patience_counter=self.patience_counter,
                    min_epochs_before_early_stop=min_epochs_es,
                )
                break

        hist_path = Path("checkpoints") / "training_history.json"
        with open(hist_path, "w") as f:
            json.dump(self.history, f, indent=2, default=str)
        self._append_event(
            "training_finished",
            best_monitor_value=self.best_monitor_value,
            interrupted_by=self._received_signal,
            history_path=str(hist_path),
        )
        print(
            f"\nTraining complete. Best {self.monitor_val_metric}: "
            f"{self.best_monitor_value:.4f}"
        )
        print(f"History saved to {hist_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="conf/config.yaml",
                        help="Path to config YAML")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from (e.g. best.pt)")
    # AUDIT: LOW-34 — Optional profiling for bottleneck analysis
    parser.add_argument("--profile", type=int, default=0,
                        help="Profile first N batches and save trace to checkpoints/profile_trace.json")
    # AUDIT: CRITICAL-02 — Force cache regeneration to prevent stale data
    parser.add_argument("--force-recompute", action="store_true",
                        help="Force regeneration of precomputed dataset cache (ignores existing cache)")
    # AUDIT: MEDIUM-30 — Prometheus metrics export for production monitoring
    parser.add_argument("--prometheus-port", type=int, default=0,
                        help="Expose Prometheus metrics on this port (0 = disabled)")
    args = parser.parse_args()
    trainer = Trainer(config_path=args.config, resume_path=args.resume, force_recompute=args.force_recompute)
    if args.profile > 0:
        trainer.profile_n_batches = args.profile
    if args.prometheus_port > 0:
        trainer.logger._metrics.start_http_server(args.prometheus_port)
    trainer.fit()
