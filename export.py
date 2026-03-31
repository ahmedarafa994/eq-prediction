"""
ONNX export and optimization for real-time DAW plugin deployment.

Exports the TCN encoder + parameter head to ONNX format.
The biquad coefficient computation stays in the plugin host code
since it is trivial computation that doesn't benefit from ONNX.
"""
import sys
import argparse
import time
import torch
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from model_tcn import StreamingTCNModel


def export_onnx(model, output_path, batch_size=1, n_frames=1, opset_version=17):
    """
    Export the model to ONNX format.

    The model is exported with a fixed input shape for streaming inference:
    (batch_size, n_mels, n_frames) where n_frames is typically 1 for streaming.
    """
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(batch_size, model.encoder.n_mels, n_frames)

    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=opset_version,
        input_names=["mel_frames"],
        output_names=["gain_db", "freq", "q", "type_logits", "type_probs",
                       "filter_type", "H_mag", "embedding"],
        dynamic_axes={
            "mel_frames": {0: "batch_size", 2: "time_frames"},
        },
        do_constant_folding=True,
    )
    print(f"ONNX model exported to {output_path}")


def benchmark_onnx(onnx_path, n_runs=1000, warmup=100):
    """Benchmark ONNX model inference latency."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed. Skipping ONNX benchmark.")
        return None

    session = ort.InferenceSession(onnx_path)

    # Warmup
    dummy = np.random.randn(1, 128, 1).astype(np.float32)
    for _ in range(warmup):
        session.run(None, {"mel_frames": dummy})

    # Benchmark
    times = []
    for _ in range(n_runs):
        dummy = np.random.randn(1, 128, 1).astype(np.float32)
        t0 = time.perf_counter()
        session.run(None, {"mel_frames": dummy})
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times.sort()
    median = times[n_runs // 2]
    p95 = times[int(n_runs * 0.95)]
    mean = sum(times) / n_runs

    print(f"ONNX Runtime latency ({n_runs} runs):")
    print(f"  Median: {median:.3f} ms")
    print(f"  P95:    {p95:.3f} ms")
    print(f"  Mean:   {mean:.3f} ms")

    return {"median_ms": median, "p95_ms": p95, "mean_ms": mean}


def benchmark_pytorch(model, n_runs=1000, warmup=100):
    """Benchmark PyTorch model inference latency."""
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            dummy = torch.randn(1, 128, 1)
            model(dummy)

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            dummy = torch.randn(1, 128, 1)
            t0 = time.perf_counter()
            model(dummy)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

    times.sort()
    median = times[n_runs // 2]
    p95 = times[int(n_runs * 0.95)]
    mean = sum(times) / n_runs

    print(f"PyTorch latency ({n_runs} runs):")
    print(f"  Median: {median:.3f} ms")
    print(f"  P95:    {p95:.3f} ms")
    print(f"  Mean:   {mean:.3f} ms")

    return {"median_ms": median, "p95_ms": p95, "mean_ms": mean}


def benchmark_streaming(model, n_runs=1000, warmup=100):
    """Benchmark streaming frame-by-frame inference."""
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            model.init_streaming(1)
            frame = torch.randn(1, 128)
            model.process_frame(frame)

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            frame = torch.randn(1, 128)
            t0 = time.perf_counter()
            model.process_frame(frame)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

    times.sort()
    median = times[n_runs // 2]
    p95 = times[int(n_runs * 0.95)]
    mean = sum(times) / n_runs

    print(f"Streaming latency ({n_runs} runs):")
    print(f"  Median: {median:.3f} ms")
    print(f"  P95:    {p95:.3f} ms")
    print(f"  Mean:   {mean:.3f} ms")

    return {"median_ms": median, "p95_ms": p95, "mean_ms": mean}


def verify_onnx_pytorch_match(onnx_path, model, n_samples=10, atol=1e-5):
    """Verify ONNX model produces same outputs as PyTorch."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed. Skipping verification.")
        return False

    session = ort.InferenceSession(onnx_path)
    model.eval()

    max_diffs = {"gain_db": 0, "freq": 0, "q": 0, "H_mag": 0}

    with torch.no_grad():
        for _ in range(n_samples):
            dummy = torch.randn(1, 128, 10)

            # PyTorch
            pt_output = model(dummy)
            pt_gain = pt_output["params"][0].numpy()
            pt_freq = pt_output["params"][1].numpy()
            pt_q = pt_output["params"][2].numpy()
            pt_H = pt_output["H_mag"].numpy()

            # ONNX
            ort_inputs = {"mel_frames": dummy.numpy()}
            ort_outputs = session.run(None, ort_inputs)
            # Outputs: gain_db, freq, q, type_logits, type_probs, filter_type, H_mag, embedding

            diffs = {
                "gain_db": np.abs(pt_gain - ort_outputs[0]).max(),
                "freq": np.abs(pt_freq - ort_outputs[1]).max(),
                "q": np.abs(pt_q - ort_outputs[2]).max(),
                "H_mag": np.abs(pt_H - ort_outputs[6]).max(),
            }
            for k, d in diffs.items():
                max_diffs[k] = max(max_diffs[k], d)

    print("ONNX vs PyTorch max absolute differences:")
    all_pass = True
    for k, d in max_diffs.items():
        status = "PASS" if d < atol else "FAIL"
        if d >= atol:
            all_pass = False
        print(f"  {k}: {d:.2e} ({status}, threshold={atol})")

    return all_pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="eq_estimator.onnx")
    parser.add_argument("--config", type=str, default="conf/config.yaml")
    parser.add_argument("--benchmark_only", action="store_true")
    args = parser.parse_args()

    if args.checkpoint:
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

        model = StreamingTCNModel(
            n_mels=config["data"].get("n_mels", 128),
            num_bands=config["data"].get("num_bands", 5),
            channels=config["model"]["encoder"].get("channels", 128),
            num_blocks=config["model"]["encoder"].get("num_blocks", 4),
            num_stacks=config["model"]["encoder"].get("num_stacks", 2),
            sample_rate=config["data"].get("sample_rate", 44100),
            n_fft=config["data"].get("n_fft", 2048),
        )

        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"]
                               if "model_state_dict" in checkpoint
                               else checkpoint)
    else:
        # Use a fresh model for export testing
        model = StreamingTCNModel(n_mels=128, num_bands=5)

    print("=== Model Info ===")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Model size (FP32): {total_params * 4 / 1024 / 1024:.2f} MB")
    print()

    # Export ONNX
    export_onnx(model, args.output)

    # Verify
    print("\n=== Verification ===")
    verify_onnx_pytorch_match(args.output, model)

    # Benchmarks
    print("\n=== Benchmarks ===")
    benchmark_pytorch(model)
    print()
    benchmark_streaming(model)
    print()
    benchmark_onnx(args.output)
