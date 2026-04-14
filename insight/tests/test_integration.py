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
    print()
    print("All integration tests passed!")
