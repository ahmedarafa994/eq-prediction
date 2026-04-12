"""Tests for Hungarian matching correctness, metric computation, and loss component verification."""

import math
import torch
import numpy as np
from loss_multitype import HungarianBandMatcher, MultiTypeEQLoss
from model_tcn import StreamingTCNModel


def test_hungarian_matching_identity():
    """When predictions equal targets, Hungarian matching returns identity permutation."""
    print("Testing Hungarian matching (identity case)...")

    matcher = HungarianBandMatcher()
    B, N = 4, 5

    gain = torch.randn(B, N) * 10.0
    freq = torch.sigmoid(torch.randn(B, N)) * 20000 + 20
    q = torch.sigmoid(torch.randn(B, N)) * 10 + 0.1
    ft = torch.randint(0, 5, (B, N))

    matched_gain, matched_freq, matched_q, matched_ft = matcher(
        gain, freq, q,
        gain, freq, q,
        target_filter_type=ft,
        pred_type_logits=torch.randn(B, N, 5),
    )

    assert torch.allclose(matched_gain, gain, atol=1e-6), (
        f"Identity matching failed for gain: max diff = {(matched_gain - gain).abs().max():.8f}"
    )
    assert torch.allclose(matched_freq, freq, atol=1e-6), (
        f"Identity matching failed for freq: max diff = {(matched_freq - freq).abs().max():.8f}"
    )
    assert torch.allclose(matched_q, q, atol=1e-6), (
        f"Identity matching failed for q: max diff = {(matched_q - q).abs().max():.8f}"
    )

    print("  PASSED")


def test_hungarian_matching_permuted():
    """When target bands are permuted, matching recovers the correct permutation."""
    print("Testing Hungarian matching (permuted case)...")

    matcher = HungarianBandMatcher()
    B, N = 2, 5

    # Ordered targets
    target_gain = torch.tensor([
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [5.0, 4.0, 3.0, 2.0, 1.0],
    ])
    target_freq = torch.tensor([
        [100.0, 500.0, 1000.0, 5000.0, 10000.0],
        [10000.0, 5000.0, 1000.0, 500.0, 100.0],
    ])
    target_q = torch.tensor([
        [0.5, 1.0, 2.0, 4.0, 8.0],
        [8.0, 4.0, 2.0, 1.0, 0.5],
    ])
    target_ft = torch.tensor([
        [0, 1, 2, 3, 4],
        [4, 3, 2, 1, 0],
    ])

    # Predictions match the original ordering
    pred_gain = target_gain.clone()
    pred_freq = target_freq.clone()
    pred_q = target_q.clone()

    # Permute targets by swapping bands 0 and 4
    perm = [4, 1, 2, 3, 0]
    perm_gain = target_gain[:, perm]
    perm_freq = target_freq[:, perm]
    perm_q = target_q[:, perm]
    perm_ft = target_ft[:, perm]

    # Matcher should recover the permutation so matched == original targets
    matched_gain, matched_freq, matched_q = matcher(
        pred_gain, pred_freq, pred_q,
        perm_gain, perm_freq, perm_q,
    )

    assert torch.allclose(matched_gain, target_gain, atol=1e-4), (
        f"Permuted matching failed for gain: max diff = {(matched_gain - target_gain).abs().max():.6f}"
    )
    assert torch.allclose(matched_freq, target_freq, atol=1e-4), (
        f"Permuted matching failed for freq: max diff = {(matched_freq - target_freq).abs().max():.6f}"
    )
    assert torch.allclose(matched_q, target_q, atol=1e-4), (
        f"Permuted matching failed for q: max diff = {(matched_q - target_q).abs().max():.6f}"
    )

    print("  PASSED")


def test_per_param_mae_accuracy():
    """MAE computation matches hand-calculated values."""
    print("Testing per-parameter MAE accuracy...")

    # Simple case: constant offset
    pred = torch.tensor([[1.0, 2.0, 3.0]])
    target = torch.tensor([[1.5, 2.5, 3.5]])
    expected_mae = 0.5
    actual_mae = (pred - target).abs().mean().item()
    assert abs(actual_mae - expected_mae) < 1e-6, (
        f"MAE mismatch: expected {expected_mae}, got {actual_mae}"
    )

    # Frequency MAE: |log2(pred/target)|
    pred_freq = torch.tensor([[1000.0, 2000.0, 4000.0]])
    target_freq = torch.tensor([[500.0, 2000.0, 8000.0]])
    freq_mae = (torch.log2(pred_freq / target_freq)).abs().mean().item()
    expected_freq_mae = (1.0 + 0.0 + 1.0) / 3.0  # 1 octave, 0, 1 octave
    assert abs(freq_mae - expected_freq_mae) < 1e-6, (
        f"Freq MAE mismatch: expected {expected_freq_mae}, got {freq_mae}"
    )

    # Q MAE: |log10(pred/target)|
    pred_q = torch.tensor([[1.0, 5.0, 2.0]])
    target_q = torch.tensor([[10.0, 5.0, 0.2]])
    q_mae = (torch.log10(pred_q / target_q)).abs().mean().item()
    expected_q_mae = (1.0 + 0.0 + 1.0) / 3.0  # 1 decade, 0, 1 decade
    assert abs(q_mae - expected_q_mae) < 1e-6, (
        f"Q MAE mismatch: expected {expected_q_mae}, got {q_mae}"
    )

    print("  PASSED")


def test_loss_component_keys():
    """MultiTypeEQLoss.forward() returns all expected component keys with finite scalar values."""
    print("Testing loss component keys and finiteness...")

    criterion = MultiTypeEQLoss()
    B, N = 2, 3

    pred_gain = torch.randn(B, N)
    pred_freq = torch.abs(torch.randn(B, N)) * 5000 + 100
    pred_q = torch.abs(torch.randn(B, N)) * 2 + 0.1
    pred_type_logits = torch.randn(B, N, 5)
    pred_H_mag_soft = torch.abs(torch.randn(B, 513))
    pred_H_mag_hard = torch.abs(torch.randn(B, 513))
    target_gain = torch.randn(B, N)
    target_freq = torch.abs(torch.randn(B, N)) * 5000 + 100
    target_q = torch.abs(torch.randn(B, N)) * 2 + 0.1
    target_ft = torch.randint(0, 5, (B, N))
    target_H_mag = torch.abs(torch.randn(B, 513))
    embedding = torch.randn(B, 128)

    total_loss, components = criterion(
        pred_gain, pred_freq, pred_q, pred_type_logits,
        pred_H_mag_soft, pred_H_mag_hard,
        target_gain, target_freq, target_q, target_ft, target_H_mag,
        embedding=embedding,
    )

    expected_keys = {
        "loss_gain", "loss_freq", "loss_q", "type_loss",
        "hmag_loss", "spectral_loss", "activity_loss", "spread_loss",
        "embed_var_loss", "contrastive_loss",
    }

    actual_keys = set(components.keys())
    missing = expected_keys - actual_keys
    extra = actual_keys - expected_keys

    assert not missing, f"Missing component keys: {missing}"
    # Extra keys (e.g. param_loss) are acceptable — we only check required ones
    if extra:
        print(f"  Note: extra keys returned (acceptable): {extra}")

    for k in expected_keys:
        v = components[k]
        val = v.item() if isinstance(v, torch.Tensor) else float(v)
        assert math.isfinite(val), f"Component {k} is not finite: {val}"

    # Total loss should also be finite
    assert torch.isfinite(total_loss), f"Total loss is not finite: {total_loss.item()}"

    print(f"  Found {len(actual_keys)} component keys, all finite")
    print("  PASSED")


def test_gradient_monitoring_groups():
    """Verify that model parameter names match the expected gradient monitoring groups."""
    print("Testing gradient monitoring parameter name groups...")

    model = StreamingTCNModel(n_mels=128, n_fft=2048, num_bands=5)
    names = [n for n, _ in model.named_parameters()]

    # Check gain head group
    assert any("gain_head" in n for n in names), (
        f"No parameter with 'gain_head' in name. Names: {[n for n in names if 'gain' in n]}"
    )

    # Check type classifier group
    assert any("type_head" in n for n in names), (
        f"No parameter with 'type_head' in name. Names: {[n for n in names if 'type' in n]}"
    )

    # Check q_head group
    assert any("q_head" in n for n in names), (
        f"No parameter with 'q_head' in name. Names: {[n for n in names if 'q' in n]}"
    )

    # Check encoder group
    assert any(n.startswith("encoder") for n in names), (
        f"No parameter starting with 'encoder'. Prefixes: {set(n.split('.')[0] for n in names)}"
    )

    print(f"  Model has {len(names)} parameters")
    print(f"  gain_head params: {sum(1 for n in names if 'gain_head' in n)}")
    print(f"  type_head params: {sum(1 for n in names if 'type_head' in n)}")
    print(f"  q_head params: {sum(1 for n in names if 'q_head' in n)}")
    print(f"  encoder params: {sum(1 for n in names if n.startswith('encoder'))}")
    print("  PASSED")


if __name__ == "__main__":
    test_hungarian_matching_identity()
    print()
    test_hungarian_matching_permuted()
    print()
    test_per_param_mae_accuracy()
    print()
    test_loss_component_keys()
    print()
    test_gradient_monitoring_groups()
    print()
    print("=== ALL METRICS TESTS PASSED ===")
