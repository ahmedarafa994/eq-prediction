"""
Comprehensive loss function correctness tests.

AUDIT: HIGH-20 — Verifies that each loss sub-component produces finite
values and gradients, that Hungarian matching works correctly, and that
edge cases (all-zero predictions, all-same-type) don't produce NaN.
"""
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loss_multitype import (
    MultiTypeEQLoss,
    HungarianBandMatcher,
    log_cosh_loss,
)
from model_tcn import StreamingTCNModel
from differentiable_eq import DifferentiableBiquadCascade


def _make_loss_input(batch_size=4, num_bands=5, device="cpu"):
    """Create well-behaved loss inputs."""
    return {
        "pred_gain": torch.randn(batch_size, num_bands, device=device) * 5.0,
        "pred_freq": torch.sigmoid(torch.randn(batch_size, num_bands, device=device)) * (20000 - 20) + 20,
        "pred_q": torch.sigmoid(torch.randn(batch_size, num_bands, device=device)) * (10 - 0.1) + 0.1,
        "pred_type_logits": torch.randn(batch_size, num_bands, 5, device=device),
        "target_gain": torch.randn(batch_size, num_bands, device=device) * 5.0,
        "target_freq": torch.sigmoid(torch.randn(batch_size, num_bands, device=device)) * (20000 - 20) + 20,
        "target_q": torch.sigmoid(torch.randn(batch_size, num_bands, device=device)) * (10 - 0.1) + 0.1,
        "target_ft": torch.randint(0, 5, (batch_size, num_bands), device=device),
    }


def test_loss_all_components_finite():
    """All loss sub-components should produce finite values for standard input."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_bands = 5
    in_data = _make_loss_input(device=device)

    dsp = DifferentiableBiquadCascade(num_bands=num_bands, sample_rate=44100).to(device)
    target_H_mag = dsp(
        in_data["target_gain"], in_data["target_freq"], in_data["target_q"],
        filter_type=in_data["target_ft"],
    )
    pred_H_mag = dsp(
        in_data["pred_gain"], in_data["pred_freq"], in_data["pred_q"],
        filter_type=in_data["target_ft"],
    )

    criterion = MultiTypeEQLoss(
        n_fft=1024, sample_rate=44100,
        lambda_param=1.0, lambda_spectral=1.0, lambda_type=1.0,
        lambda_hmag=0.25, lambda_gain=1.0, lambda_freq=1.0, lambda_q=1.0,
        lambda_typed_spectral=0.5, lambda_hdb=0.5,
        lambda_embed_var=0.1, lambda_contrastive=0.05,
        lambda_film_diversity=0.0,
        dsp_cascade=dsp,
    ).to(device)

    # Create a mock embedding
    embedding = torch.randn(4, 128, device=device, requires_grad=True)

    loss, components = criterion(
        in_data["pred_gain"], in_data["pred_freq"], in_data["pred_q"],
        in_data["pred_type_logits"],
        pred_H_mag, pred_H_mag,
        in_data["target_gain"], in_data["target_freq"], in_data["target_q"],
        in_data["target_ft"],
        target_H_mag,
        embedding=embedding,
    )

    assert torch.isfinite(loss), f"Total loss is not finite: {loss.item()}"

    for name, val in components.items():
        if isinstance(val, torch.Tensor):
            assert torch.isfinite(val).all(), (
                f"Component '{name}' is not finite: {val.item() if val.numel() == 1 else val}"
            )

    print(f"  [test_loss_all_components_finite] PASSED — {len(components)} components")


def test_loss_gradients_finite():
    """All trainable inputs should receive finite gradients after backward."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_bands = 5
    in_data = _make_loss_input(device=device)
    # Make prediction tensors leaf tensors with requires_grad
    in_data["pred_gain"] = in_data["pred_gain"].detach().requires_grad_(True)
    in_data["pred_freq"] = in_data["pred_freq"].detach().requires_grad_(True)
    in_data["pred_q"] = in_data["pred_q"].detach().requires_grad_(True)
    in_data["pred_type_logits"] = in_data["pred_type_logits"].detach().requires_grad_(True)

    dsp = DifferentiableBiquadCascade(num_bands=num_bands, sample_rate=44100).to(device)
    target_H_mag = dsp(
        in_data["target_gain"], in_data["target_freq"], in_data["target_q"],
        filter_type=in_data["target_ft"],
    )
    pred_H_mag = dsp(
        in_data["pred_gain"], in_data["pred_freq"], in_data["pred_q"],
        filter_type=in_data["target_ft"],
    )

    criterion = MultiTypeEQLoss(
        n_fft=1024, sample_rate=44100,
        lambda_param=1.0, lambda_spectral=1.0, lambda_type=1.0,
        lambda_hmag=0.25, lambda_gain=1.0, lambda_freq=1.0, lambda_q=1.0,
        dsp_cascade=dsp,
    ).to(device)

    embedding = torch.randn(4, 128, device=device, requires_grad=True)

    loss, _ = criterion(
        in_data["pred_gain"], in_data["pred_freq"], in_data["pred_q"],
        in_data["pred_type_logits"],
        pred_H_mag, pred_H_mag,
        in_data["target_gain"], in_data["target_freq"], in_data["target_q"],
        in_data["target_ft"],
        target_H_mag,
        embedding=embedding,
    )

    loss.backward()

    for name, tensor in [
        ("pred_gain", in_data["pred_gain"]),
        ("pred_freq", in_data["pred_freq"]),
        ("pred_q", in_data["pred_q"]),
        ("pred_type_logits", in_data["pred_type_logits"]),
        ("embedding", embedding),
    ]:
        assert tensor.grad is not None, f"No gradient for {name}"
        assert torch.isfinite(tensor.grad).all(), f"Gradient for {name} is not finite"

    print("  [test_loss_gradients_finite] PASSED")


def test_hungarian_matching_correctness():
    """Hungarian matcher should assign predictions to targets optimally."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_bands = 3

    # Perfect match scenario: predictions exactly match targets in order
    pred_gain = torch.tensor([[1.0, 2.0, 3.0]], device=device)
    pred_freq = torch.tensor([[100.0, 200.0, 400.0]], device=device)
    pred_q = torch.tensor([[1.0, 1.0, 1.0]], device=device)
    target_gain = torch.tensor([[1.0, 2.0, 3.0]], device=device)
    target_freq = torch.tensor([[100.0, 200.0, 400.0]], device=device)
    target_q = torch.tensor([[1.0, 1.0, 1.0]], device=device)
    target_ft = torch.tensor([[0, 1, 2]], device=device)
    type_logits = torch.zeros(1, num_bands, 5, device=device)
    type_logits[:, :, 0] = 10.0  # Strong signal for correct types

    matcher = HungarianBandMatcher(
        lambda_gain=1.0, lambda_freq=1.0, lambda_q=1.0,
        lambda_type_match=0.5,
    )

    matched_gain, matched_freq, matched_q, matched_type = matcher(
        pred_gain, pred_freq, pred_q,
        target_gain, target_freq, target_q,
        target_filter_type=target_ft,
        pred_type_logits=type_logits,
    )

    # In perfect match, assigned targets should match predictions
    assert torch.allclose(matched_gain, pred_gain, atol=1e-4)
    assert torch.allclose(matched_freq, pred_freq, atol=1e-4)

    print("  [test_hungarian_matching_correctness] PASSED")


def test_hungarian_matching_permutation():
    """Hungarian matcher should handle permuted predictions correctly."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_bands = 3

    # Predictions in reverse order of targets
    pred_gain = torch.tensor([[3.0, 2.0, 1.0]], device=device)
    pred_freq = torch.tensor([[400.0, 200.0, 100.0]], device=device)
    pred_q = torch.tensor([[1.0, 1.0, 1.0]], device=device)
    target_gain = torch.tensor([[1.0, 2.0, 3.0]], device=device)
    target_freq = torch.tensor([[100.0, 200.0, 400.0]], device=device)
    target_q = torch.tensor([[1.0, 1.0, 1.0]], device=device)
    target_ft = torch.tensor([[0, 0, 0]], device=device)  # Same type for simplicity
    type_logits = torch.zeros(1, num_bands, 5, device=device)

    matcher = HungarianBandMatcher(
        lambda_gain=1.0, lambda_freq=1.0, lambda_q=1.0,
        lambda_type_match=0.0,
    )

    matched_gain, matched_freq, matched_q, matched_type = matcher(
        pred_gain, pred_freq, pred_q,
        target_gain, target_freq, target_q,
        target_filter_type=target_ft,
        pred_type_logits=type_logits,
    )

    # After matching, pred and target should align
    gain_err = (pred_gain - matched_gain).abs().sum()
    assert gain_err < 0.1, f"Matching failed: gain error = {gain_err.item():.4f}"

    print("  [test_hungarian_matching_permutation] PASSED")


def test_edge_case_all_zero_predictions():
    """All-zero predictions should produce finite loss (not NaN)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_bands = 5

    pred_gain = torch.zeros(4, num_bands, device=device)
    pred_freq = torch.full((4, num_bands), 1000.0, device=device)
    pred_q = torch.ones(4, num_bands, device=device)
    pred_type_logits = torch.zeros(4, num_bands, 5, device=device)

    dsp = DifferentiableBiquadCascade(num_bands=num_bands, sample_rate=44100).to(device)
    target_gain = torch.randn(4, num_bands, device=device) * 5.0
    target_freq = torch.sigmoid(torch.randn(4, num_bands, device=device)) * (20000 - 20) + 20
    target_q = torch.sigmoid(torch.randn(4, num_bands, device=device)) * (10 - 0.1) + 0.1
    target_ft = torch.randint(0, 5, (4, num_bands), device=device)

    target_H_mag = dsp(target_gain, target_freq, target_q, filter_type=target_ft)
    pred_H_mag = dsp(pred_gain, pred_freq, pred_q, filter_type=target_ft)

    criterion = MultiTypeEQLoss(
        n_fft=1024, sample_rate=44100,
        lambda_param=1.0, lambda_spectral=1.0, lambda_type=1.0,
        lambda_hmag=0.25, lambda_gain=1.0, lambda_freq=1.0, lambda_q=1.0,
        dsp_cascade=dsp,
    ).to(device)

    embedding = torch.randn(4, 128, device=device)

    loss, components = criterion(
        pred_gain, pred_freq, pred_q, pred_type_logits,
        pred_H_mag, pred_H_mag,
        target_gain, target_freq, target_q, target_ft,
        target_H_mag, embedding=embedding,
    )

    assert torch.isfinite(loss), f"Loss with all-zero predictions is not finite: {loss.item()}"

    print("  [test_edge_case_all_zero_predictions] PASSED")


def test_edge_case_all_same_type():
    """All bands having the same filter type should not cause NaN."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_bands = 5

    pred_gain = torch.randn(4, num_bands, device=device) * 3.0
    pred_freq = torch.sigmoid(torch.randn(4, num_bands, device=device)) * (20000 - 20) + 20
    pred_q = torch.sigmoid(torch.randn(4, num_bands, device=device)) * (10 - 0.1) + 0.1
    # Strong signal for peaking (type 0)
    pred_type_logits = torch.zeros(4, num_bands, 5, device=device)
    pred_type_logits[:, :, 0] = 100.0

    dsp = DifferentiableBiquadCascade(num_bands=num_bands, sample_rate=44100).to(device)
    target_ft = torch.zeros(4, num_bands, dtype=torch.long, device=device)
    target_gain = torch.randn(4, num_bands, device=device) * 3.0
    target_freq = torch.sigmoid(torch.randn(4, num_bands, device=device)) * (20000 - 20) + 20
    target_q = torch.sigmoid(torch.randn(4, num_bands, device=device)) * (10 - 0.1) + 0.1
    target_H_mag = dsp(target_gain, target_freq, target_q, filter_type=target_ft)
    pred_H_mag = dsp(pred_gain, pred_freq, pred_q, filter_type=target_ft)

    criterion = MultiTypeEQLoss(
        n_fft=1024, sample_rate=44100,
        lambda_param=1.0, lambda_spectral=1.0, lambda_type=1.0,
        lambda_hmag=0.25, lambda_gain=1.0, lambda_freq=1.0, lambda_q=1.0,
        dsp_cascade=dsp,
    ).to(device)

    embedding = torch.randn(4, 128, device=device)

    loss, _ = criterion(
        pred_gain, pred_freq, pred_q, pred_type_logits,
        pred_H_mag, pred_H_mag,
        target_gain, target_freq, target_q, target_ft,
        target_H_mag, embedding=embedding,
    )

    assert torch.isfinite(loss), f"Loss with all-same-type is not finite: {loss.item()}"

    print("  [test_edge_case_all_same_type] PASSED")


def test_log_cosh_loss_properties():
    """Verify log-cosh loss has expected mathematical properties."""
    # Near zero: should approximate 0.5 * x^2
    x_small = torch.tensor([0.001, -0.001, 0.01])
    loss_small = log_cosh_loss(x_small, torch.zeros_like(x_small))
    expected_small = 0.5 * x_small ** 2
    assert torch.allclose(loss_small, expected_small, atol=1e-6)

    # Far from zero: should approximate |x| - log(2)
    x_large = torch.tensor([10.0, -10.0, 100.0])
    loss_large = log_cosh_loss(x_large, torch.zeros_like(x_large))
    import math
    expected_large = x_large.abs() - math.log(2)
    assert torch.allclose(loss_large, expected_large, atol=1e-3)

    # Should be C2 continuous everywhere (second derivative exists)
    x = torch.linspace(-10, 10, 100, requires_grad=True)
    y = log_cosh_loss(x, torch.zeros_like(x))
    grad1 = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
    grad2 = torch.autograd.grad(grad1.sum(), x, create_graph=True)[0]
    assert torch.isfinite(grad2).all(), "Second derivative not finite (not C2 continuous)"

    print("  [test_log_cosh_loss_properties] PASSED")


if __name__ == "__main__":
    print("Running loss function correctness tests...")
    print()
    test_loss_all_components_finite()
    test_loss_gradients_finite()
    test_hungarian_matching_correctness()
    test_hungarian_matching_permutation()
    test_edge_case_all_zero_predictions()
    test_edge_case_all_same_type()
    test_log_cosh_loss_properties()
    print()
    print("All loss function tests passed!")
