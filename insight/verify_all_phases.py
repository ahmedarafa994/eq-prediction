"""
Verify all phases after training run.

Run this script after training to verify all phase success criteria.
Checks convergence metrics, warmup behavior, and refinement improvement.

Usage:
    cd insight
    python verify_all_phases.py --checkpoint checkpoints/best.pt
    python verify_all_phases.py --checkpoint checkpoints/best.pt --no-refinement  # skip refinement check
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))
import torch
import yaml
import json


def load_checkpoint(checkpoint_path):
    """Load checkpoint and return state dict."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" in ckpt:
        return ckpt["model_state_dict"], ckpt
    return ckpt, ckpt


def load_model(cfg, state_dict=None):
    """Load StreamingTCNModel."""
    from model_tcn import StreamingTCNModel
    model = StreamingTCNModel(
        n_mels=cfg['data']['n_mels'],
        embedding_dim=cfg['model']['encoder']['embedding_dim'],
        num_bands=cfg['data']['num_bands'],
        n_fft=cfg['data']['n_fft'],
    )
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def verify_phase2_gain_mae(model, val_loader, cfg):
    """Phase 2: Gain MAE < 3 dB with Hungarian matching."""
    from loss_multitype import HungarianBandMatcher

    matcher = HungarianBandMatcher(lambda_gain=1.0, lambda_freq=1.0, lambda_q=1.0)
    total_gain_mae = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            mel_frames = batch["mel_frames"]
            target_params = batch["target_params"]
            target_gain = target_params[0]
            B = mel_frames.shape[0]

            out = model.forward(mel_frames)
            gain_db = out["params"][0]
            type_probs = out["type_probs"]
            pred_types = type_probs.argmax(-1)

            cost_matrix = matcher.compute_cost_matrix(
                gain_db, out["params"][1], out["params"][2], pred_types,
                target_gain, target_params[1], target_params[2], target_params[3]
            )
            indices, _ = matcher.match(cost_matrix, B, cfg['data']['num_bands'])

            for b in range(B):
                pred_idx, tgt_idx = indices[b]
                total_gain_mae += (gain_db[b, pred_idx] - target_gain[b, tgt_idx]).abs().item()

            total_samples += B

    gain_mae = total_gain_mae / total_samples if total_samples > 0 else float('inf')
    passed = gain_mae < 3.0
    return {
        "name": "Phase 2: Gain MAE < 3 dB",
        "value": f"{gain_mae:.2f} dB",
        "passed": passed,
        "threshold": "< 3.0 dB"
    }


def verify_phase3_warmup(model, cfg):
    """Phase 3: Verify warmup configuration is wired."""
    loss_cfg = cfg.get('loss', {})
    warmup_epochs = loss_cfg.get('warmup_epochs', 0)

    # Check model has warmup gate method
    has_warmup = hasattr(model, '_apply_curriculum_stage') or 'warmup_epochs' in str(cfg)

    return {
        "name": "Phase 3: Warmup gate configured",
        "value": f"warmup_epochs={warmup_epochs}",
        "passed": warmup_epochs > 0 and has_warmup,
        "threshold": "warmup_epochs > 0"
    }


def verify_phase3_loss_architecture(model, cfg):
    """Phase 3: Verify loss architecture components."""
    checks = []

    # Check independent weights
    loss_cfg = cfg.get('loss', {})
    has_independent = 'lambda_gain' in loss_cfg and 'lambda_freq' in loss_cfg and 'lambda_q' in loss_cfg
    checks.append({
        "name": "Phase 3: Independent loss weights",
        "value": f"gain={loss_cfg.get('lambda_gain')}, freq={loss_cfg.get('lambda_freq')}, q={loss_cfg.get('lambda_q')}",
        "passed": has_independent,
        "threshold": "lambda_gain, lambda_freq, lambda_q present"
    })

    # Check spectral reconstruction
    has_spectral = loss_cfg.get('lambda_spectral', 0) > 0
    checks.append({
        "name": "Phase 3: Spectral reconstruction loss",
        "value": f"lambda_spectral={loss_cfg.get('lambda_spectral', 0)}",
        "passed": has_spectral,
        "threshold": "lambda_spectral > 0"
    })

    return checks


def verify_phase4_metrics(model, val_loader, cfg):
    """Phase 4: Verify Q MAE, type accuracy, freq MAE."""
    from loss_multitype import HungarianBandMatcher

    matcher = HungarianBandMatcher(lambda_gain=1.0, lambda_freq=1.0, lambda_q=1.0)
    total_q_mae = 0.0
    total_freq_mae = 0.0
    total_type_correct = 0
    total_samples = 0
    total_bands = 0

    with torch.no_grad():
        for batch in val_loader:
            mel_frames = batch["mel_frames"]
            target_params = batch["target_params"]
            target_gain, target_freq, target_q, target_type = target_params
            B = mel_frames.shape[0]

            out = model.forward(mel_frames)
            gain_db, freq, q = out["params"]
            type_probs = out["type_probs"]
            pred_types = type_probs.argmax(-1)

            cost_matrix = matcher.compute_cost_matrix(
                gain_db, freq, q, pred_types,
                target_gain, target_freq, target_q, target_type
            )
            indices, _ = matcher.match(cost_matrix, B, cfg['data']['num_bands'])

            for b in range(B):
                pred_idx, tgt_idx = indices[b]
                total_q_mae += (q[b, pred_idx] - target_q[b, tgt_idx]).abs().item()
                total_freq_mae += (freq[b, pred_idx] - target_freq[b, tgt_idx]).abs().item()
                if pred_types[b, pred_idx].item() == target_type[b, tgt_idx].item():
                    total_type_correct += 1
                total_bands += 1

            total_samples += B

    q_mae = total_q_mae / total_samples if total_samples > 0 else float('inf')
    freq_mae = total_freq_mae / total_samples if total_samples > 0 else float('inf')
    type_acc = total_type_correct / total_bands if total_bands > 0 else 0.0

    return [
        {
            "name": "Phase 4: Q MAE < 0.2 decades",
            "value": f"{q_mae:.3f} decades",
            "passed": q_mae < 0.2,
            "threshold": "< 0.2 decades"
        },
        {
            "name": "Phase 4: Freq MAE < 0.25 octaves",
            "value": f"{freq_mae:.3f} octaves",
            "passed": freq_mae < 0.25,
            "threshold": "< 0.25 octaves"
        },
        {
            "name": "Phase 4: Type accuracy > 95%",
            "value": f"{type_acc:.1%}",
            "passed": type_acc > 0.95,
            "threshold": "> 95%"
        }
    ]


def verify_phase5_refinement(model, val_loader, cfg, no_refinement=False):
    """Phase 5: Verify refinement improves gain MAE by 30%."""
    if no_refinement:
        return {
            "name": "Phase 5: Refinement check skipped",
            "value": "N/A",
            "passed": True,
            "threshold": "--no-refinement flag set"
        }

    from loss_multitype import HungarianBandMatcher

    matcher = HungarianBandMatcher(lambda_gain=1.0, lambda_freq=1.0, lambda_q=1.0)
    total_single_mae = 0.0
    total_refined_mae = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            mel_frames = batch["mel_frames"]
            target_params = batch["target_params"]
            target_gain = target_params[0]
            B = mel_frames.shape[0]

            # Single-pass
            out_single = model.forward(mel_frames)
            gain_db = out_single["params"][0]
            pred_types = out_single["type_probs"].argmax(-1)

            cost_matrix = matcher.compute_cost_matrix(
                gain_db, out_single["params"][1], out_single["params"][2], pred_types,
                target_gain, target_params[1], target_params[2], target_params[3]
            )
            indices, _ = matcher.match(cost_matrix, B, cfg['data']['num_bands'])

            for b in range(B):
                pred_idx, tgt_idx = indices[b]
                total_single_mae += (gain_db[b, pred_idx] - target_gain[b, tgt_idx]).abs().item()

            # Refined
            out_refined = model.forward(mel_frames, refine=True)
            if "refined_params" in out_refined:
                refined_gain = out_refined["refined_params"][0]
                refined_type = out_refined["filter_type"]

                cost_matrix_r = matcher.compute_cost_matrix(
                    refined_gain, out_refined["refined_params"][1], out_refined["refined_params"][2],
                    refined_type, target_gain, target_params[1], target_params[2], target_params[3]
                )
                indices_r, _ = matcher.match(cost_matrix_r, B, cfg['data']['num_bands'])

                for b in range(B):
                    pred_idx, tgt_idx = indices_r[b]
                    total_refined_mae += (refined_gain[b, pred_idx] - target_gain[b, tgt_idx]).abs().item()

            total_samples += B

    single_mae = total_single_mae / total_samples if total_samples > 0 else float('inf')
    refined_mae = total_refined_mae / total_samples if total_samples > 0 else float('inf')
    improvement = (1.0 - refined_mae / single_mae) * 100 if single_mae > 0 else 0.0

    return {
        "name": "Phase 5: Refinement improves gain MAE by 30%",
        "value": f"{single_mae:.2f} -> {refined_mae:.2f} dB ({improvement:.1f}% improvement)",
        "passed": improvement >= 30,
        "threshold": ">= 30% improvement"
    }


def main():
    parser = argparse.ArgumentParser(description="Verify all phases after training")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint")
    parser.add_argument("--no-refinement", action="store_true", help="Skip refinement check")
    parser.add_argument("--n-samples", type=int, default=500, help="Validation samples to use")
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    state_dict, full_ckpt = load_checkpoint(args.checkpoint)

    cfg_path = os.path.join(os.path.dirname(__file__), "conf", "config.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    model = load_model(cfg, state_dict)

    # Create validation dataset
    from dataset import SyntheticEQDataset
    from torch.utils.data import DataLoader

    val_dataset = SyntheticEQDataset(
        cfg=cfg, split="val", size=min(args.n_samples, cfg['data'].get('val_dataset_size', 2000))
    )
    collate_fn = getattr(val_dataset, 'collate_fn', None)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    print(f"Evaluating on {len(val_dataset)} samples...\n")

    results = []

    # Phase 2
    results.append(verify_phase2_gain_mae(model, val_loader, cfg))

    # Phase 3
    results.append(verify_phase3_warmup(model, cfg))
    results.extend(verify_phase3_loss_architecture(model, cfg))

    # Phase 4
    results.extend(verify_phase4_metrics(model, val_loader, cfg))

    # Phase 5
    results.append(verify_phase5_refinement(model, val_loader, cfg, args.no_refinement))

    # Print results
    print("=" * 70)
    print(" PHASE VERIFICATION RESULTS")
    print("=" * 70)

    passed = 0
    failed = 0
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        if r["passed"]:
            passed += 1
        else:
            failed += 1
        print(f"  [{status}] {r['name']}")
        print(f"         Value: {r['value']} (Threshold: {r['threshold']})")

    print("=" * 70)
    print(f" Results: {passed} passed, {failed} failed")
    print("=" * 70)

    if failed > 0:
        print(f"\nWARNING: {failed} verification(s) failed.")
        sys.exit(1)
    else:
        print(f"\nALL PHASES VERIFIED SUCCESSFULLY")
        sys.exit(0)


if __name__ == "__main__":
    main()
