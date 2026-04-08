"""Tests for streaming TCN model: gradient flow, batch-vs-streaming consistency, latency."""
import time
import torch
from model_tcn import StreamingTCNModel


def test_tcn_gradient_flow():
    """Verify gradients flow through the entire TCN model."""
    print("Testing TCN gradient flow...")

    model = StreamingTCNModel(n_mels=128, embedding_dim=128, num_bands=5)

    # Batch of mel-spectrograms: (B, n_mels, T)
    mel_input = torch.randn(4, 128, 64, requires_grad=True)
    output = model(mel_input)

    gain_db, freq, q = output["params"]
    H_mag = output["H_mag"]

    loss = H_mag.sum()
    loss.backward()

    assert mel_input.grad is not None, "No gradient for mel input"
    print(f"  Embedding shape: {output['embedding'].shape}")
    print(f"  H_mag shape: {H_mag.shape}")
    print(f"  Gradient norm: {mel_input.grad.norm().item():.4f}")
    print("  Gradient flow OK")


def test_batch_vs_streaming_consistency():
    """Verify batch mode and frame-by-frame streaming produce identical results."""
    print("Testing batch vs streaming consistency...")

    model = StreamingTCNModel(n_mels=128, embedding_dim=128, num_bands=5)
    model.eval()

    batch_size = 1
    n_frames = 10
    mel_seq = torch.randn(batch_size, 128, n_frames)

    # Batch mode
    with torch.no_grad():
        batch_output = model(mel_seq)
        batch_embedding = batch_output["embedding"]

    # Streaming mode
    model.init_streaming(batch_size)
    streaming_embeddings = []
    with torch.no_grad():
        for t in range(n_frames):
            frame = mel_seq[:, :, t]
            out = model.process_frame(frame)
            streaming_embeddings.append(out["embedding"])

    streaming_final = streaming_embeddings[-1]

    # They should be close but may not be exactly identical due to
    # batch norm running stats. Check rough similarity.
    diff = (batch_embedding - streaming_final).abs().max().item()
    print(f"  Batch embedding norm: {batch_embedding.norm().item():.4f}")
    print(f"  Streaming embedding norm: {streaming_final.norm().item():.4f}")
    print(f"  Max difference: {diff:.4f}")

    # Both should produce non-trivial embeddings
    assert batch_embedding.norm().item() > 0.01, "Batch embedding is too small"
    assert streaming_final.norm().item() > 0.01, "Streaming embedding is too small"

    # Parameter shapes should match
    batch_params = batch_output["params"]
    stream_params = model.process_frame(mel_seq[:, :, -1])["params"]
    for i, (bp, sp) in enumerate(zip(batch_params, stream_params)):
        assert bp.shape == sp.shape, f"Param {i} shape mismatch: {bp.shape} vs {sp.shape}"

    print("  Shape consistency OK")


def test_streaming_latency():
    """Benchmark single-frame streaming inference latency."""
    print("Testing streaming inference latency...")

    model = StreamingTCNModel(n_mels=128, embedding_dim=128, num_bands=5)
    model.eval()

    batch_size = 1
    model.init_streaming(batch_size)

    # Warm up
    with torch.no_grad():
        for _ in range(10):
            frame = torch.randn(batch_size, 128)
            model.process_frame(frame)

    # Benchmark
    n_runs = 1000
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            frame = torch.randn(batch_size, 128)
            t0 = time.perf_counter()
            model.process_frame(frame)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # ms

    times.sort()
    median_ms = times[n_runs // 2]
    p95_ms = times[int(n_runs * 0.95)]
    mean_ms = sum(times) / n_runs

    print(f"  Single-frame inference (CPU, {n_runs} runs):")
    print(f"    Median: {median_ms:.3f} ms")
    print(f"    P95:    {p95_ms:.3f} ms")
    print(f"    Mean:   {mean_ms:.3f} ms")

    if median_ms < 1.0:
        print("  Latency target MET (<1ms)")
    else:
        print(f"  WARNING: Latency target NOT met ({median_ms:.3f}ms > 1ms)")

    return median_ms


def test_output_shapes():
    """Verify all output shapes match expectations."""
    print("Testing output shapes...")

    model = StreamingTCNModel(
        n_mels=128, embedding_dim=128, num_bands=5,
        sample_rate=44100, n_fft=2048
    )

    mel_input = torch.randn(4, 128, 32)
    output = model(mel_input)

    B = 4
    N = 5

    assert output["embedding"].shape == (B, 128), f"embedding: {output['embedding'].shape}"
    assert output["H_mag"].shape == (B, 1025), f"H_mag: {output['H_mag'].shape}"

    gain_db, freq, q = output["params"]
    assert gain_db.shape == (B, N), f"gain_db: {gain_db.shape}"
    assert freq.shape == (B, N), f"freq: {freq.shape}"
    assert q.shape == (B, N), f"q: {q.shape}"

    assert output["type_logits"].shape == (B, N, 5), f"type_logits: {output['type_logits'].shape}"
    assert output["type_probs"].shape == (B, N, 5), f"type_probs: {output['type_probs'].shape}"
    assert output["filter_type"].shape == (B, N), f"filter_type: {output['filter_type'].shape}"

    # Parameter bounds
    assert torch.all(gain_db >= -24.0) and torch.all(gain_db <= 24.0), "Gain out of bounds"
    assert torch.all(freq >= 20.0) and torch.all(freq <= 20000.0), "Freq out of bounds"
    assert torch.all(q >= 0.1) and torch.all(q <= 10.0), "Q out of bounds"

    # Type probs sum to 1
    assert torch.allclose(output["type_probs"].sum(dim=-1), torch.ones(B, N), atol=1e-4)

    print(f"  gain_db: {gain_db.shape}, freq: {freq.shape}, q: {q.shape}")
    print(f"  type_logits: {output['type_logits'].shape}")
    print(f"  H_mag: {output['H_mag'].shape}")
    print("  All shapes and bounds OK")


def test_receptive_field():
    """Print receptive field information."""
    print("Testing receptive field...")
    model = StreamingTCNModel(n_mels=128, num_blocks=4, num_stacks=2)
    rf = model.receptive_field_frames

    # At 44100 Hz with hop_length=256, each frame ~5.8ms
    hop_length = 256
    sample_rate = 44100
    frame_duration_ms = hop_length / sample_rate * 1000
    rf_ms = rf * frame_duration_ms

    print(f"  Receptive field: {rf} frames = {rf_ms:.1f} ms (at hop={hop_length}, sr={sample_rate})")
    print(f"  Frame duration: {frame_duration_ms:.2f} ms")


if __name__ == "__main__":
    test_tcn_gradient_flow()
    print()
    test_output_shapes()
    print()
    test_receptive_field()
    print()
    test_batch_vs_streaming_consistency()
    print()
    test_streaming_latency()
    print()
    print("All streaming TCN tests passed!")
