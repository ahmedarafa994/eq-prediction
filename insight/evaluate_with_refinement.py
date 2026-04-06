"""
Evaluate inference-time refinement improvement.

Compares single-pass vs gradient-refined gain MAE using Hungarian-matched
evaluation on a synthetic validation set. Reports INFR-01 gate criterion:
30% gain MAE improvement over single-pass.

Usage:
    cd insight
    python evaluate_with_refinement.py                            # uses checkpoints/best.pt
    python evaluate_with_refinement.py --checkpoint path/to.pt   # custom checkpoint
    python evaluate_with_refinement.py --no-checkpoint           # random model (structural test only)
"""
import sys, os, math, argparse
sys.path.insert(0, os.path.dirname(__file__))
import torch
import yaml


def load_model(checkpoint_path, cfg):
    """Load StreamingTCNModel from checkpoint or random initialization."""
    from model_tcn import StreamingTCNModel

    model = StreamingTCNModel(
        n_mels=cfg['data']['n_mels'],
        embedding_dim=cfg['model']['encoder']['embedding_dim'],
        num_bands=cfg['data']['num_bands'],
        n_fft=cfg['data']['n_fft'],
    )

    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print("Using randomly initialized model (structural test)")

    model.eval()
    return model


def evaluate_with_refinement(model, val_loader, cfg, n_refinement_steps=5):
    """
    Evaluate single-pass vs refined gain MAE with Hungarian matching.

    Returns dict with gain_mae_single, gain_mae_refined, improvement_pct,
    mean_type_entropy, mean_confidence.
    """
    from loss_multitype import HungarianBandMatcher

    matcher = HungarianBandMatcher(
        lambda_gain=1.0, lambda_freq=1.0, lambda_q=1.0, lambda_type_match=0.5
    )

    total_gain_mae_single = 0.0
    total_gain_mae_refined = 0.0
    total_samples = 0

    total_type_entropy = 0.0
    total_confidence = 0.0
    n_confidence_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            mel_frames = batch["mel_frames"]
            target_params = batch["target_params"]
            target_gain, target_freq, target_q, target_type = target_params
            B = mel_frames.shape[0]

            # Single-pass prediction
            out_single = model.forward(mel_frames)
            gain_db, freq, q = out_single["params"]
            type_probs = out_single["type_probs"]

            # Hungarian matching for single-pass
            pred_types = type_probs.argmax(-1)
            cost_matrix = matcher.compute_cost_matrix(
                gain_db, freq, q, pred_types,
                target_gain, target_freq, target_q, target_type
            )
            indices, _ = matcher.match(cost_matrix, B, cfg['data']['num_bands'])

            # Compute matched gain MAE for single-pass
            for b in range(B):
                pred_idx, tgt_idx = indices[b]
                matched_pred_gain = gain_db[b, pred_idx]
                matched_target_gain = target_gain[b, tgt_idx]
                total_gain_mae_single += (matched_pred_gain - matched_target_gain).abs().sum().item()

            # Refined prediction
            out_refined = model.forward(mel_frames, refine=True)
            if "refined_params" in out_refined:
                refined_gain, refined_freq, refined_q = out_refined["refined_params"]
                refined_type = out_refined["filter_type"]

                cost_matrix_refined = matcher.compute_cost_matrix(
                    refined_gain, refined_freq, refined_q, refined_type,
                    target_gain, target_freq, target_q, target_type
                )
                indices_refined, _ = matcher.match(cost_matrix_refined, B, cfg['data']['num_bands'])

                for b in range(B):
                    pred_idx, tgt_idx = indices_refined[b]
                    matched_pred_gain = refined_gain[b, pred_idx]
                    matched_target_gain = target_gain[b, tgt_idx]
                    total_gain_mae_refined += (matched_pred_gain - matched_target_gain).abs().item()

            # Confidence metrics
            if "confidence" in out_refined:
                conf = out_refined["confidence"]
                total_type_entropy += conf["type_entropy"].mean().item()
                total_confidence += conf["overall_confidence"].mean().item()
                n_confidence_samples += 1

            total_samples += B

    gain_mae_single = total_gain_mae_single / total_samples if total_samples > 0 else float('inf')
    gain_mae_refined = total_gain_mae_refined / total_samples if total_samples > 0 else float('inf')

    if gain_mae_single > 0:
        improvement_pct = (1.0 - gain_mae_refined / gain_mae_single) * 100
    else:
        improvement_pct = 0.0

    return {
        "gain_mae_single": gain_mae_single,
        "gain_mae_refined": gain_mae_refined,
        "improvement_pct": improvement_pct,
        "mean_type_entropy": total_type_entropy / n_confidence_samples if n_confidence_samples > 0 else 0.0,
        "mean_confidence": total_confidence / n_confidence_samples if n_confidence_samples > 0 else 0.0,
        "n_samples": total_samples,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate inference refinement improvement")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--no-checkpoint", action="store_true",
                        help="Use random model (structural test only)")
    parser.add_argument("--n-samples", type=int, default=200,
                        help="Number of validation samples")
    parser.add_argument("--refine-steps", type=int, default=None,
                        help="Number of refinement steps (overrides config)")
    args = parser.parse_args()

    # Load config
    cfg_path = os.path.join(os.path.dirname(__file__), "conf", "config.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # Determine refinement steps
    refine_steps = args.refine_steps or cfg.get("refinement", {}).get("grad_refine_steps", 5)

    # Load model
    checkpoint_path = None if args.no_checkpoint else args.checkpoint
    model = load_model(checkpoint_path, cfg)

    # Create validation dataset
    from dataset import SyntheticEQDataset
    from torch.utils.data import DataLoader

    val_dataset = SyntheticEQDataset(
        cfg=cfg,
        split="val",
        size=min(args.n_samples, cfg['data'].get('val_dataset_size', 2000))
    )
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=val_dataset.collate_fn if hasattr(val_dataset, 'collate_fn') else None)

    print(f"\nEvaluating on {len(val_dataset)} samples with {refine_steps} refinement steps...")

    # Run evaluation
    results = evaluate_with_refinement(model, val_loader, cfg, n_refinement_steps=refine_steps)

    # Print results
    in_pass = "PASS" if results["improvement_pct"] >= 30 else "FAIL"
    print(f"\n{'=' * 50}")
    print(f" Inference Refinement Evaluation")
    print(f"{'=' * 50}")
    print(f" Single-pass gain MAE:   {results['gain_mae_single']:.2f} dB")
    print(f" Refined gain MAE:       {results['gain_mae_refined']:.2f} dB")
    print(f" Improvement:            {results['improvement_pct']:.1f}% [{in_pass}]")
    print(f" Mean type entropy:      {results['mean_type_entropy']:.4f}")
    print(f" Mean confidence:        {results['mean_confidence']:.4f}")
    print(f" Samples evaluated:      {results['n_samples']}")
    print(f"{'=' * 50}")

    if results["improvement_pct"] < 30:
        print(f"\nWARNING: INFR-01 gate NOT met (need >= 30% improvement)")
        sys.exit(1)
    else:
        print(f"\nINFR-01 gate PASSED: {results['improvement_pct']:.1f}% >= 30%")
        sys.exit(0)


if __name__ == "__main__":
    main()
