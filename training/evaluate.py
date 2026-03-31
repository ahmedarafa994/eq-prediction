"""
Evaluation script for the multi-type EQ estimator.

Computes per-parameter accuracy metrics, spectral reconstruction,
filter type confusion, and generates visualization plots.
"""
import sys
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from model_tcn import StreamingTCNModel
from dsp_frontend import STFTFrontend
from dataset_pipeline.dataset import create_dataloaders


def evaluate_model(model, dataloader, frontend, device="cuda", n_fft=2048):
    """
    Evaluate model on a dataset.

    Returns dict of metrics and per-sample predictions.
    """
    model.eval()
    model.to(device)

    all_gain_pred = []
    all_gain_gt = []
    all_freq_pred = []
    all_freq_gt = []
    all_q_pred = []
    all_q_gt = []
    all_type_pred = []
    all_type_gt = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            wet_audio = batch["wet_audio"].to(device)

            target_gain = batch["gain"]
            target_freq = batch["freq"]
            target_q = batch["q"]
            target_ft = batch["filter_type"]

            # Convert to mel
            mel_spec = frontend.mel_spectrogram(wet_audio)
            mel_frames = mel_spec.squeeze(1)

            # Forward
            output = model(mel_frames)
            pred_gain, pred_freq, pred_q = output["params"]

            all_gain_pred.append(pred_gain.cpu())
            all_gain_gt.append(target_gain)
            all_freq_pred.append(pred_freq.cpu())
            all_freq_gt.append(target_freq)
            all_q_pred.append(pred_q.cpu())
            all_q_gt.append(target_q)
            all_type_pred.append(output["filter_type"].cpu())
            all_type_gt.append(target_ft)

    # Concatenate
    gain_pred = torch.cat(all_gain_pred)
    gain_gt = torch.cat(all_gain_gt)
    freq_pred = torch.cat(all_freq_pred)
    freq_gt = torch.cat(all_freq_gt)
    q_pred = torch.cat(all_q_pred)
    q_gt = torch.cat(all_q_gt)
    type_pred = torch.cat(all_type_pred)
    type_gt = torch.cat(all_type_gt)

    # Compute metrics
    metrics = {}

    # Gain MAE (dB)
    metrics["gain_mae_db"] = (gain_pred - gain_gt).abs().mean().item()
    metrics["gain_median_ae_db"] = (gain_pred - gain_gt).abs().median().item()

    # Frequency MAE (octaves)
    freq_oct = (torch.log2(freq_pred / (freq_gt + 1e-8))).abs()
    metrics["freq_mae_oct"] = freq_oct.mean().item()
    metrics["freq_median_ae_oct"] = freq_oct.median().item()

    # Q MAE (decades)
    q_dec = (torch.log10(q_pred / (q_gt + 1e-8))).abs()
    metrics["q_mae_dec"] = q_dec.mean().item()
    metrics["q_median_ae_dec"] = q_dec.median().item()

    # Filter type accuracy
    metrics["type_accuracy"] = (type_pred == type_gt).float().mean().item()

    # Per-type accuracy
    FILTER_NAMES = ["peaking", "lowshelf", "highshelf", "highpass", "lowpass"]
    for i, name in enumerate(FILTER_NAMES):
        mask = type_gt == i
        if mask.sum() > 0:
            metrics[f"type_acc_{name}"] = (type_pred[mask] == type_gt[mask]).float().mean().item()

    # Confusion matrix
    num_types = 5
    confusion = torch.zeros(num_types, num_types)
    for p, t in zip(type_pred.flatten(), type_gt.flatten()):
        confusion[t.item(), p.item()] += 1
    metrics["confusion_matrix"] = confusion.tolist()

    return metrics


def print_metrics(metrics):
    """Print evaluation metrics in a readable format."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(f"\nParameter Estimation:")
    print(f"  Gain MAE:        {metrics['gain_mae_db']:.2f} dB (target: < 2.0)")
    print(f"  Gain Median AE:  {metrics['gain_median_ae_db']:.2f} dB")
    print(f"  Freq MAE:        {metrics['freq_mae_oct']:.3f} oct (target: < 0.5)")
    print(f"  Freq Median AE:  {metrics['freq_median_ae_oct']:.3f} oct")
    print(f"  Q MAE:           {metrics['q_mae_dec']:.3f} decades (target: < 0.3)")
    print(f"  Q Median AE:     {metrics['q_median_ae_dec']:.3f} decades")

    print(f"\nFilter Type Classification:")
    print(f"  Overall Accuracy: {metrics['type_accuracy']:.1%} (target: > 85%)")
    for name in ["peaking", "lowshelf", "highshelf", "highpass", "lowpass"]:
        key = f"type_acc_{name}"
        if key in metrics:
            print(f"  {name:>12s}:     {metrics[key]:.1%}")

    # Confusion matrix
    if "confusion_matrix" in metrics:
        print(f"\nConfusion Matrix (rows=true, cols=predicted):")
        cm = np.array(metrics["confusion_matrix"])
        # Normalize by row
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.where(row_sums > 0, cm / row_sums, 0)
        header = "         " + "  ".join(f"{n[:5]:>5s}" for n in ["peak", "lshe", "hshe", "hpf", "lpf"])
        print(header)
        names = ["peaking", "lowshelf", "highshelf", "highpass", "lowpass"]
        for i, name in enumerate(names):
            row = "  ".join(f"{cm_norm[i,j]:5.2f}" for j in range(5))
            print(f"  {name:>8s}  {row}")

    print("\n" + "=" * 60)

    # Pass/fail summary
    gain_pass = metrics['gain_mae_db'] < 2.0
    freq_pass = metrics['freq_mae_oct'] < 0.5
    q_pass = metrics['q_mae_dec'] < 0.3
    type_pass = metrics['type_accuracy'] > 0.85

    print("TARGET SUMMARY:")
    print(f"  Gain MAE < 2dB:   {'PASS' if gain_pass else 'FAIL'}")
    print(f"  Freq MAE < 0.5:   {'PASS' if freq_pass else 'FAIL'}")
    print(f"  Q MAE < 0.3:      {'PASS' if q_pass else 'FAIL'}")
    print(f"  Type Acc > 85%:   {'PASS' if type_pass else 'FAIL'}")
    print()


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--config", type=str, default="../conf/config.yaml")
    parser.add_argument("--output", type=str, default="eval_results.json")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Load model
    model = StreamingTCNModel.load_from_checkpoint(args.checkpoint, config=config)

    # Frontend
    frontend = STFTFrontend(
        n_fft=config["data"].get("n_fft", 2048),
        hop_length=config["data"].get("hop_length", 256),
        win_length=config["data"].get("n_fft", 2048),
        mel_bins=config["data"].get("n_mels", 128),
        sample_rate=config["data"].get("sample_rate", 44100),
    )

    # Data
    _, _, test_loader = create_dataloaders(
        args.data_dir,
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
    )

    # Evaluate
    metrics = evaluate_model(model, test_loader, frontend, device=args.device)
    print_metrics(metrics)

    # Save
    with open(args.output, 'w') as f:
        # Remove non-serializable confusion matrix for JSON
        save_metrics = {k: v for k, v in metrics.items() if k != "confusion_matrix"}
        json.dump(save_metrics, f, indent=2)
    print(f"Results saved to {args.output}")
