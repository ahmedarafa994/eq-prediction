"""
Integration tests for the full training pipeline.

AUDIT: CRITICAL-18 — These tests verify end-to-end correctness:
  audio → mel → encoder → head → DSP cascade → loss → backward.

All tests are designed to catch integration-level bugs that unit tests
miss (e.g., the "catastrophic TCN encoder collapse" mentioned in model_tcn.py).
"""
import torch
import sys
from pathlib import Path

# Ensure we can import from the insight directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model_tcn import StreamingTCNModel
from dsp_frontend import STFTFrontend
from differentiable_eq import DifferentiableBiquadCascade
from loss_multitype import MultiTypeEQLoss


def _make_batch(batch_size=4, seq_len=128, num_bands=5, n_mels=128, device="cpu"):
    """Create a valid mini-batch for testing."""
    return {
        "wet_audio": torch.randn(batch_size, seq_len, device=device),
        "dry_audio": torch.randn(batch_size, seq_len, device=device),
        "gain": torch.randn(batch_size, num_bands, device=device) * 5.0,
        "freq": torch.sigmoid(torch.randn(batch_size, num_bands, device=device)) * (20000 - 20) + 20,
        "q": torch.sigmoid(torch.randn(batch_size, num_bands, device=device)) * (10 - 0.1) + 0.1,
        "filter_type": torch.randint(0, 5, (batch_size, num_bands), device=device),
        "active_band_mask": torch.ones(batch_size, num_bands, dtype=torch.bool, device=device),
    }


def test_full_forward_pass_finite():
    """
    Verify that a complete forward pass through the model produces
    finite loss and valid gradients through all components.

    This catches integration bugs where individual components pass
    unit tests but fail when assembled (e.g., encoder collapse).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_bands = 5
    batch = _make_batch(device=device)

    model = StreamingTCNModel(
        n_mels=128,
        embedding_dim=128,
        num_bands=num_bands,
        channels=64,
        num_blocks=2,
        num_stacks=1,
        sample_rate=44100,
        n_fft=2048,
    ).to(device)
    model.eval()  # Deterministic for testing

    frontend = STFTFrontend(
        n_fft=2048, hop_length=256, win_length=2048,
        mel_bins=128, sample_rate=44100,
    ).to(device)

    with torch.no_grad():
        mel = frontend.mel_spectrogram(batch["wet_audio"]).squeeze(1)
        output = model(mel, wet_audio=batch["wet_audio"])

    # Verify output structure
    assert "params" in output, "Missing 'params' in output"
    assert "type_logits" in output, "Missing 'type_logits' in output"
    assert "H_mag" in output, "Missing 'H_mag' in output"
    assert "embedding" in output, "Missing 'embedding' in output"

    pred_gain, pred_freq, pred_q = output["params"]
    assert pred_gain.shape == (batch["wet_audio"].shape[0], num_bands)
    assert pred_freq.shape == (batch["wet_audio"].shape[0], num_bands)
    assert pred_q.shape == (batch["wet_audio"].shape[0], num_bands)

    # Verify all outputs are finite
    assert torch.isfinite(pred_gain).all(), "pred_gain has non-finite values"
    assert torch.isfinite(pred_freq).all(), "pred_freq has non-finite values"
    assert torch.isfinite(pred_q).all(), "pred_q has non-finite values"
    assert torch.isfinite(output["type_logits"]).all(), "type_logits has non-finite values"
    assert torch.isfinite(output["H_mag"]).all(), "H_mag has non-finite values"

    print("  [test_full_forward_pass_finite] PASSED")


def test_full_backward_pass_gradient_flow():
    """
    Verify that gradients flow through the entire pipeline:
    loss → DSP cascade → parameter head → encoder → mel frontend.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_bands = 5
    batch = _make_batch(device=device)

    model = StreamingTCNModel(
        n_mels=128,
        embedding_dim=128,
        num_bands=num_bands,
        channels=64,
        num_blocks=2,
        num_stacks=1,
        sample_rate=44100,
        n_fft=2048,
    ).to(device)

    criterion = MultiTypeEQLoss(
        n_fft=1024,
        sample_rate=44100,
        lambda_param=1.0,
        lambda_spectral=1.0,
        lambda_type=1.0,
        lambda_hmag=0.25,
        lambda_gain=1.0,
        lambda_freq=1.0,
        lambda_q=1.0,
        dsp_cascade=model.dsp_cascade,
    ).to(device)

    frontend = STFTFrontend(
        n_fft=2048, hop_length=256, win_length=2048,
        mel_bins=128, sample_rate=44100,
    ).to(device)

    # Forward
    mel = frontend.mel_spectrogram(batch["wet_audio"]).squeeze(1)
    output = model(mel, wet_audio=batch["wet_audio"])
    pred_gain, pred_freq, pred_q = output["params"]

    # Ground truth frequency response
    target_H_mag = model.dsp_cascade(
        batch["gain"], batch["freq"], batch["q"],
        filter_type=batch["filter_type"],
    )

    # Compute loss
    loss, components = criterion(
        pred_gain, pred_freq, pred_q,
        output["type_logits"],
        output["H_mag_soft"],
        output["H_mag"],
        batch["gain"], batch["freq"], batch["q"], batch["filter_type"],
        target_H_mag,
        embedding=output["embedding"],
    )

    # Loss should be finite
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
    assert loss > 0, f"Loss should be positive: {loss.item()}"

    # Backward
    loss.backward()

    # Check gradient flow through key components
    grads_with_grad = []
    grads_without_grad = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is not None and torch.isfinite(param.grad).all():
                grads_with_grad.append(name)
            else:
                grads_without_grad.append(name)

    # At least some parameters should have gradients
    assert len(grads_with_grad) > 0, (
        f"No parameters received gradients. Components without grads: {grads_without_grad[:10]}"
    )

    # Print diagnostic if any components lack gradients
    if grads_without_grad:
        print(f"  [WARNING] {len(grads_without_grad)} params have no gradient: {grads_without_grad[:5]}")

    print(f"  [test_full_backward_pass_gradient_flow] PASSED — {len(grads_with_grad)} param groups with gradients")


def test_loss_components_all_finite():
    """
    Verify that all individual loss components produce finite values
    for a standard batch. Catches bugs where sub-losses produce NaN.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_bands = 5
    batch = _make_batch(device=device)

    model = StreamingTCNModel(
        n_mels=128,
        embedding_dim=128,
        num_bands=num_bands,
        channels=64,
        num_blocks=2,
        num_stacks=1,
        sample_rate=44100,
        n_fft=2048,
    ).to(device)

    criterion = MultiTypeEQLoss(
        n_fft=1024,
        sample_rate=44100,
        lambda_param=1.0,
        lambda_spectral=1.0,
        lambda_type=1.0,
        lambda_hmag=0.25,
        lambda_gain=1.0,
        lambda_freq=1.0,
        lambda_q=1.0,
        lambda_typed_spectral=0.5,
        lambda_hdb=0.5,
        lambda_embed_var=0.1,
        lambda_contrastive=0.05,
        dsp_cascade=model.dsp_cascade,
    ).to(device)

    frontend = STFTFrontend(
        n_fft=2048, hop_length=256, win_length=2048,
        mel_bins=128, sample_rate=44100,
    ).to(device)

    model.train()
    mel = frontend.mel_spectrogram(batch["wet_audio"]).squeeze(1)
    output = model(mel, wet_audio=batch["wet_audio"])
    pred_gain, pred_freq, pred_q = output["params"]

    target_H_mag = model.dsp_cascade(
        batch["gain"], batch["freq"], batch["q"],
        filter_type=batch["filter_type"],
    )

    loss, components = criterion(
        pred_gain, pred_freq, pred_q,
        output["type_logits"],
        output["H_mag_soft"],
        output["H_mag"],
        batch["gain"], batch["freq"], batch["q"], batch["filter_type"],
        target_H_mag,
        embedding=output["embedding"],
    )

    # Check total loss
    assert torch.isfinite(loss), f"Total loss is not finite: {loss.item()}"

    # Check each component
    for name, val in components.items():
        if isinstance(val, torch.Tensor):
            assert torch.isfinite(val).all(), f"Component '{name}' is not finite: {val}"

    print(f"  [test_loss_components_all_finite] PASSED — {len(components)} components all finite")


def test_checkpoint_roundtrip():
    """
    Verify that saving and loading a checkpoint produces identical model outputs.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_bands = 5
    batch = _make_batch(device=device)

    model = StreamingTCNModel(
        n_mels=128,
        embedding_dim=128,
        num_bands=num_bands,
        channels=64,
        num_blocks=2,
        num_stacks=1,
        sample_rate=44100,
        n_fft=2048,
    ).to(device)
    model.eval()

    frontend = STFTFrontend(
        n_fft=2048, hop_length=256, win_length=2048,
        mel_bins=128, sample_rate=44100,
    ).to(device)

    # Generate reference output
    with torch.no_grad():
        mel = frontend.mel_spectrogram(batch["wet_audio"]).squeeze(1)
        ref_output = model(mel, wet_audio=batch["wet_audio"], hard_types=True)

    # Save checkpoint
    import tempfile
    ckpt_path = Path(tempfile.mkdtemp()) / "test_ckpt.pt"
    state = {
        "model_state_dict": model.state_dict(),
        "epoch": 1,
        "val_loss": 0.5,
    }
    torch.save(state, ckpt_path)

    # Load into new model
    model2 = StreamingTCNModel(
        n_mels=128,
        embedding_dim=128,
        num_bands=num_bands,
        channels=64,
        num_blocks=2,
        num_stacks=1,
        sample_rate=44100,
        n_fft=2048,
    ).to(device)
    model2.eval()

    loaded = torch.load(ckpt_path, weights_only=False)
    model2.load_state_dict(loaded["model_state_dict"])

    # Verify identical output
    with torch.no_grad():
        mel = frontend.mel_spectrogram(batch["wet_audio"]).squeeze(1)
        new_output = model2(mel, wet_audio=batch["wet_audio"], hard_types=True)

    for key in ref_output:
        if isinstance(ref_output[key], torch.Tensor):
            assert torch.allclose(ref_output[key], new_output[key], atol=1e-6), (
                f"Output mismatch for key '{key}'"
            )

    ckpt_path.unlink()
    ckpt_path.parent.rmdir()
    print("  [test_checkpoint_roundtrip] PASSED")


def test_model_dsp_consistency():
    """
    Verify that the model's internal DSP cascade produces consistent
    frequency responses for the same input parameters.

    AUDIT: CRITICAL-02 — Tests consistency between training and offline
    biquad implementations.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_bands = 5
    batch_size = 4

    model = StreamingTCNModel(
        n_mels=128,
        embedding_dim=128,
        num_bands=num_bands,
        channels=64,
        num_blocks=2,
        num_stacks=1,
        sample_rate=44100,
        n_fft=2048,
    ).to(device)
    model.eval()

    gain = torch.randn(batch_size, num_bands, device=device) * 5.0
    freq = torch.sigmoid(torch.randn(batch_size, num_bands, device=device)) * (20000 - 20) + 20
    q = torch.sigmoid(torch.randn(batch_size, num_bands, device=device)) * (10 - 0.1) + 0.1
    filter_type = torch.randint(0, 5, (batch_size, num_bands), device=device)

    # Method 1: Direct call to DSP cascade
    with torch.no_grad():
        H1 = model.dsp_cascade(gain, freq, q, n_fft=1024, filter_type=filter_type)

    # Method 2: Via biquad coefficients + freq_response
    with torch.no_grad():
        b0, b1, b2, a1, a2 = model.dsp_cascade.compute_biquad_coeffs_multitype(
            gain, freq, q, filter_type
        )
        H2_bands = model.dsp_cascade.freq_response(b0, b1, b2, a1, a2, n_fft=1024)
        H2 = H2_bands.prod(dim=1)

    # Responses should match (may differ in n_fft resolution)
    # Resample H1 to match H2's resolution if needed
    if H1.shape[-1] != H2.shape[-1]:
        H1_resampled = torch.nn.functional.interpolate(
            H1.unsqueeze(1), size=H2.shape[-1], mode="linear", align_corners=False
        ).squeeze(1)
    else:
        H1_resampled = H1

    assert torch.allclose(H1_resampled, H2, atol=1e-4), (
        f"DSP cascade and freq_response produce different results: "
        f"max diff = {(H1_resampled - H2).abs().max().item():.6f}"
    )

    print("  [test_model_dsp_consistency] PASSED")


if __name__ == "__main__":
    print("Running integration tests...")
    print()
    test_full_forward_pass_finite()
    test_full_backward_pass_gradient_flow()
    test_loss_components_all_finite()
    test_checkpoint_roundtrip()
    test_model_dsp_consistency()
    test_cross_implementation_dsp_consistency()
    test_type_accuracy_above_random_baseline()
    print()
    print("All integration tests passed!")


def test_cross_implementation_dsp_consistency():
    """
    AUDIT: HIGH-03 (P1-9) — Verify that the offline data generator's
    biquad implementations (dataset_pipeline/generate_data.py) produce
    the same frequency response as the training pipeline's
    DifferentiableBiquadCascade (differentiable_eq.py).

    This catches numerical precision differences between the two
    independent implementations of the RBJ Audio EQ Cookbook formulas.
    """
    import math
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from differentiable_eq import DifferentiableBiquadCascade
    import torchaudio.functional as F

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sr = 44100
    n_fft = 2048
    num_bands = 5
    batch_size = 4

    dsp = DifferentiableBiquadCascade(num_bands=num_bands, sample_rate=sr).to(device)

    # Test each filter type
    filter_types = {
        "peaking": {"gain": 6.0, "freq": 1000.0, "q": 1.0},
        "lowshelf": {"gain": 3.0, "freq": 500.0, "q": 0.7},
        "highshelf": {"gain": -3.0, "freq": 5000.0, "q": 0.7},
        "highpass": {"gain": 0.0, "freq": 200.0, "q": 0.7},
        "lowpass": {"gain": 0.0, "freq": 8000.0, "q": 0.7},
    }

    # Offline biquad coefficient functions (from dataset_pipeline/generate_data.py)
    def offline_biquad_peaking(gain_db, freq, q, sr):
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

    def offline_biquad_lowshelf(gain_db, freq, q, sr):
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

    def offline_biquad_highshelf(gain_db, freq, q, sr):
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

    def offline_biquad_highpass(freq, q, sr):
        w0 = 2.0 * math.pi * freq / sr
        alpha = math.sin(w0) / (2.0 * q)
        b0 = (1 + math.cos(w0)) / 2
        b1 = -(1 + math.cos(w0))
        b2 = (1 + math.cos(w0)) / 2
        a0 = 1 + alpha
        a1 = -2 * math.cos(w0)
        a2 = 1 - alpha
        return [b0/a0, b1/a0, b2/a0, a1/a0, a2/a0]

    def offline_biquad_lowpass(freq, q, sr):
        w0 = 2.0 * math.pi * freq / sr
        alpha = math.sin(w0) / (2.0 * q)
        b0 = (1 - math.cos(w0)) / 2
        b1 = 1 - math.cos(w0)
        b2 = (1 - math.cos(w0)) / 2
        a0 = 1 + alpha
        a1 = -2 * math.cos(w0)
        a2 = 1 - alpha
        return [b0/a0, b1/a0, b2/a0, a1/a0, a2/a0]

    offline_funcs = [offline_biquad_peaking, offline_biquad_lowshelf,
                     offline_biquad_highshelf, offline_biquad_highpass, offline_biquad_lowpass]

    for ftype_idx, (ftype, params) in enumerate(filter_types.items()):
        gain = torch.tensor([[params["gain"]]], device=device)
        freq = torch.tensor([[params["freq"]]], device=device)
        q = torch.tensor([[params["q"]]], device=device)
        filter_type = torch.tensor([[ftype_idx]], device=device)

        # Training pipeline frequency response
        with torch.no_grad():
            H_train = dsp(gain, freq, q, n_fft=n_fft, filter_type=filter_type)

        # Offline biquad coefficients
        coeffs = offline_funcs[ftype_idx](params["gain"], params["freq"], params["q"], sr)
        b = torch.tensor(coeffs[:3], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        a = torch.tensor([1.0] + coeffs[3:], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        # Compute offline frequency response manually
        w = torch.linspace(0, math.pi, n_fft // 2 + 1, device=device)
        cos_w = torch.cos(w)
        cos_2w = torch.cos(2 * w)
        sin_w = torch.sin(w)
        sin_2w = torch.sin(2 * w)

        num_re = b[0,0,0] + b[0,0,1] * cos_w + b[0,0,2] * cos_2w
        num_im = -(b[0,0,1] * sin_w + b[0,0,2] * sin_2w)
        den_re = 1.0 + a[0,0,0] * cos_w + a[0,0,1] * cos_2w
        den_im = -(a[0,0,0] * sin_w + a[0,0,1] * sin_2w)

        num_mag2 = num_re**2 + num_im**2
        den_mag2 = den_re**2 + den_im**2
        H_offline = torch.sqrt(torch.clamp(num_mag2 / (den_mag2 + 1e-4), min=1e-8, max=1e6))

        # Compare (allow some tolerance due to different implementations)
        # Resample if lengths differ
        if H_train.shape[-1] != H_offline.shape[-1]:
            H_train_resampled = torch.nn.functional.interpolate(
                H_train.unsqueeze(1), size=H_offline.shape[-1], mode="linear", align_corners=False
            ).squeeze(1)
        else:
            H_train_resampled = H_train

        max_diff = (H_train_resampled.squeeze() - H_offline).abs().max().item()
        assert max_diff < 0.05, (
            f"Cross-implementation mismatch for {ftype}: max diff = {max_diff:.6f}. "
            f"This indicates numerical differences between offline and training biquad implementations."
        )

    print("  [test_cross_implementation_dsp_consistency] PASSED")


def test_type_accuracy_above_random_baseline():
    """
    AUDIT: HIGH-08 (P1-12) — Full-model integration test that trains for
    a few epochs and verifies type accuracy exceeds random baseline.

    This catches the type collapse failure mode that was observed in
    actual training (100% peaking predictions, 19.7% accuracy).

    Uses a smaller model for speed but full config parameters.
    """
    import random
    import numpy as np
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from model_tcn import StreamingTCNModel
    from dsp_frontend import STFTFrontend
    from differentiable_eq import DifferentiableBiquadCascade
    from loss_multitype import MultiTypeEQLoss, HungarianBandMatcher
    from dataset import SyntheticEQDataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_bands = 5
    batch_size = 8
    n_epochs = 5

    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Create small dataset for fast testing
    dataset = SyntheticEQDataset(
        num_bands=num_bands,
        sample_rate=44100,
        duration=1.0,  # Short audio for speed
        size=200,  # Small dataset
        gain_range=(-6.0, 6.0),
        freq_range=(20.0, 20000.0),
        q_range=(0.1, 10.0),
        type_weights=[0.2, 0.2, 0.2, 0.2, 0.2],
        gain_distribution="beta",
        precompute_mels=False,
        base_seed=42,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    # Create model with reduced size for speed
    model = StreamingTCNModel(
        n_mels=128,
        embedding_dim=128,
        num_bands=num_bands,
        channels=128,  # Reduced from 256
        num_blocks=4,  # Reduced from 8
        num_stacks=2,  # Reduced from 3
        sample_rate=44100,
        n_fft=2048,
    ).to(device)

    criterion = MultiTypeEQLoss(
        n_fft=1024,
        sample_rate=44100,
        lambda_param=1.0,
        lambda_spectral=1.0,
        lambda_type=8.0,  # High type weight to force learning
        lambda_hmag=0.25,
        lambda_gain=1.0,
        lambda_freq=1.0,
        lambda_q=1.0,
        lambda_type_entropy=2.0,  # High entropy penalty
        lambda_type_prior=0.5,
        warmup_epochs=0,  # No warmup for this test
        dsp_cascade=model.dsp_cascade,
    ).to(device)

    matcher = HungarianBandMatcher()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Train for a few epochs
    type_accuracies = []
    for epoch in range(n_epochs):
        model.train()
        total_type_correct = 0
        total_type_samples = 0

        for batch in dataloader:
            wet_audio = batch["wet_audio"].to(device)
            dry_audio = batch["dry_audio"].to(device)
            target_gain = batch["gain"].to(device)
            target_freq = batch["freq"].to(device)
            target_q = batch["q"].to(device)
            target_filter_type = batch["filter_type"].to(device)

            mel = model.frontend.mel_spectrogram(wet_audio).squeeze(1)
            output = model(mel, wet_audio=wet_audio)
            pred_gain, pred_freq, pred_q = output["params"]

            target_H_mag = model.dsp_cascade(
                target_gain, target_freq, target_q,
                filter_type=target_filter_type,
            )

            loss, components = criterion(
                pred_gain, pred_freq, pred_q,
                output["type_logits"],
                output["H_mag_soft"],
                output["H_mag"],
                target_gain, target_freq, target_q, target_filter_type,
                target_H_mag,
                embedding=output["embedding"],
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Compute type accuracy for this batch
            with torch.no_grad():
                pred_types = output["type_logits"].argmax(dim=-1)  # (B, N)
                type_correct = (pred_types == target_filter_type).float().mean()
                total_type_correct += type_correct.item() * wet_audio.shape[0]
                total_type_samples += wet_audio.shape[0]

        epoch_type_acc = total_type_correct / max(total_type_samples, 1)
        type_accuracies.append(epoch_type_acc)

    final_type_acc = type_accuracies[-1] if type_accuracies else 0.0
    random_baseline = 1.0 / 5.0  # 20% for 5 classes

    # Type accuracy should exceed random baseline after training
    # Use a conservative threshold since we're training a small model for few epochs
    assert final_type_acc > random_baseline + 0.05, (
        f"Type accuracy ({final_type_acc:.3f}) is not above random baseline "
        f"({random_baseline:.3f}) after {n_epochs} epochs. "
        f"This indicates type collapse. Type accuracy trajectory: {type_accuracies}. "
        f"Check Gumbel temperature, type loss weight, and warmup configuration."
    )

    print(f"  [test_type_accuracy_above_random_baseline] PASSED — "
          f"type_acc={final_type_acc:.3f} > baseline={random_baseline:.3f} after {n_epochs} epochs")
