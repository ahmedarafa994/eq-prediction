"""Tests for multi-type differentiable biquad cascade and parameter head."""
import torch
import math
from differentiable_eq import (
    DifferentiableBiquadCascade,
    MultiTypeEQParameterHead,
    FILTER_PEAKING, FILTER_LOWSHELF, FILTER_HIGHSHELF,
    FILTER_HIGHPASS, FILTER_LOWPASS,
)


def test_multitype_gradient_flow():
    """Verify gradients flow through all 5 filter types."""
    print("Testing gradient flow through all filter types...")

    batch_size = 4
    num_bands = 5
    cascade = DifferentiableBiquadCascade(num_bands, sample_rate=44100)

    # Use raw tensors that will be leaf tensors
    gain_raw = torch.randn(batch_size, num_bands, requires_grad=True)
    freq_raw = torch.randn(batch_size, num_bands, requires_grad=True)
    q_raw = torch.randn(batch_size, num_bands, requires_grad=True)

    for ft_val, ft_name in enumerate(["peaking", "lowshelf", "highshelf", "highpass", "lowpass"]):
        gain_db = gain_raw * 10.0
        freq = torch.sigmoid(freq_raw) * (20000 - 20) + 20
        q = torch.sigmoid(q_raw) * (10 - 0.1) + 0.1

        filter_type = torch.full((batch_size, num_bands), ft_val, dtype=torch.long)
        H_mag = cascade(gain_db, freq, q, n_fft=1024, filter_type=filter_type)
        loss = H_mag.sum()
        loss.backward(retain_graph=True)

        assert gain_raw.grad is not None, f"No gradient for gain_db with {ft_name}"
        assert freq_raw.grad is not None, f"No gradient for freq with {ft_name}"
        assert q_raw.grad is not None, f"No gradient for q with {ft_name}"

        print(f"  {ft_name}: H_mag shape={H_mag.shape}, loss={loss.item():.4f} - gradients OK")

        # Zero grads for next iteration
        gain_raw.grad.zero_()
        freq_raw.grad.zero_()
        q_raw.grad.zero_()

    print("Gradient flow test passed!")


def test_peaking_backward_compat():
    """Verify multitype peaking matches the original peaking-only implementation."""
    print("Testing peaking backward compatibility...")

    batch_size = 4
    num_bands = 5
    cascade = DifferentiableBiquadCascade(num_bands, sample_rate=44100)

    gain_db = torch.randn(batch_size, num_bands) * 5.0
    freq = torch.sigmoid(torch.randn(batch_size, num_bands)) * (20000 - 20) + 20
    q = torch.sigmoid(torch.randn(batch_size, num_bands)) * (10 - 0.1) + 0.1

    # Original peaking-only path
    b0_old, b1_old, b2_old, a1_old, a2_old = cascade.compute_biquad_coeffs(gain_db, freq, q)

    # Multitype path with all peaking
    filter_type = torch.full((batch_size, num_bands), FILTER_PEAKING, dtype=torch.long)
    b0_new, b1_new, b2_new, a1_new, a2_new = cascade.compute_biquad_coeffs_multitype(
        gain_db, freq, q, filter_type
    )

    torch.testing.assert_close(b0_old, b0_new, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(b1_old, b1_new, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(b2_old, b2_new, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(a1_old, a1_new, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(a2_old, a2_new, atol=1e-6, rtol=1e-6)

    print("  Coefficients match between old and new peaking implementation.")


def test_hp_frequency_response():
    """Verify HP filter rolls off below cutoff frequency."""
    print("Testing high-pass frequency response shape...")

    batch_size = 1
    num_bands = 1
    n_fft = 4096
    cascade = DifferentiableBiquadCascade(num_bands, sample_rate=44100)

    cutoff = 1000.0
    gain_db = torch.zeros(batch_size, num_bands)  # irrelevant for HP
    freq = torch.full((batch_size, num_bands), cutoff)
    q = torch.full((batch_size, num_bands), 0.707)  # Butterworth
    filter_type = torch.full((batch_size, num_bands), FILTER_HIGHPASS, dtype=torch.long)

    H_mag = cascade(gain_db, freq, q, n_fft=n_fft, filter_type=filter_type)
    H_mag = H_mag.squeeze()  # (n_fft//2+1,)

    freqs = torch.linspace(0, 22050, n_fft // 2 + 1)

    # Below cutoff (e.g., 500 Hz): response should be < 0.707 (-3dB point area)
    below_idx = freqs < cutoff * 0.5
    assert H_mag[below_idx].max() < 0.9, (
        f"HP should attenuate below cutoff, got max {H_mag[below_idx].max():.3f}"
    )

    # Well above cutoff (e.g., 4000 Hz): response should be ~1.0
    above_idx = freqs > cutoff * 4
    assert H_mag[above_idx].mean() > 0.8, (
        f"HP should pass above cutoff, got mean {H_mag[above_idx].mean():.3f}"
    )

    print(f"  HP at {cutoff}Hz: below={H_mag[below_idx].max():.3f}, above={H_mag[above_idx].mean():.3f} - OK")


def test_lp_frequency_response():
    """Verify LP filter rolls off above cutoff frequency."""
    print("Testing low-pass frequency response shape...")

    batch_size = 1
    num_bands = 1
    n_fft = 4096
    cascade = DifferentiableBiquadCascade(num_bands, sample_rate=44100)

    cutoff = 5000.0
    gain_db = torch.zeros(batch_size, num_bands)
    freq = torch.full((batch_size, num_bands), cutoff)
    q = torch.full((batch_size, num_bands), 0.707)
    filter_type = torch.full((batch_size, num_bands), FILTER_LOWPASS, dtype=torch.long)

    H_mag = cascade(gain_db, freq, q, n_fft=n_fft, filter_type=filter_type)
    H_mag = H_mag.squeeze()

    freqs = torch.linspace(0, 22050, n_fft // 2 + 1)

    # Below cutoff: should pass ~1.0
    below_idx = freqs < cutoff * 0.25
    assert H_mag[below_idx].mean() > 0.8, (
        f"LP should pass below cutoff, got mean {H_mag[below_idx].mean():.3f}"
    )

    # Above cutoff: should attenuate
    above_idx = freqs > cutoff * 2
    assert H_mag[above_idx].max() < 0.9, (
        f"LP should attenuate above cutoff, got max {H_mag[above_idx].max():.3f}"
    )

    print(f"  LP at {cutoff}Hz: below={H_mag[below_idx].mean():.3f}, above={H_mag[above_idx].max():.3f} - OK")


def test_shelf_response():
    """Verify shelf filters produce gain at the correct frequency range."""
    print("Testing shelf filter response...")

    batch_size = 1
    num_bands = 1
    n_fft = 4096
    cascade = DifferentiableBiquadCascade(num_bands, sample_rate=44100)

    target_gain_db = 6.0
    freq = torch.full((batch_size, num_bands), 1000.0)
    q = torch.full((batch_size, num_bands), 0.707)
    gain_db = torch.full((batch_size, num_bands), target_gain_db)

    freqs = torch.linspace(0, 22050, n_fft // 2 + 1)

    # Low-shelf: boost below center
    filter_type = torch.full((batch_size, num_bands), FILTER_LOWSHELF, dtype=torch.long)
    H_ls = cascade(gain_db, freq, q, n_fft=n_fft, filter_type=filter_type).squeeze()
    low_idx = freqs < 200
    assert H_ls[low_idx].mean() > 1.0, (
        f"Low-shelf should boost below center, got {H_ls[low_idx].mean():.3f}"
    )

    # High-shelf: boost above center
    filter_type = torch.full((batch_size, num_bands), FILTER_HIGHSHELF, dtype=torch.long)
    H_hs = cascade(gain_db, freq, q, n_fft=n_fft, filter_type=filter_type).squeeze()
    high_idx = freqs > 5000
    assert H_hs[high_idx].mean() > 1.0, (
        f"High-shelf should boost above center, got {H_hs[high_idx].mean():.3f}"
    )

    print(f"  Low-shelf gain at low freq: {H_ls[low_idx].mean():.3f}")
    print(f"  High-shelf gain at high freq: {H_hs[high_idx].mean():.3f}")


def test_multitype_parameter_head():
    """Test MultiTypeEQParameterHead output shapes and bounds."""
    print("Testing MultiTypeEQParameterHead...")

    batch_size = 8
    embedding_dim = 128
    num_bands = 5

    head = MultiTypeEQParameterHead(embedding_dim, num_bands)
    embedding = torch.randn(batch_size, embedding_dim)

    # Training mode (soft types)
    head.train()
    gain_db, freq, q, type_logits, type_probs, filter_type = head(embedding, hard_types=False)

    assert gain_db.shape == (batch_size, num_bands), f"gain_db shape: {gain_db.shape}"
    assert freq.shape == (batch_size, num_bands), f"freq shape: {freq.shape}"
    assert q.shape == (batch_size, num_bands), f"q shape: {q.shape}"
    assert type_logits.shape == (batch_size, num_bands, 5), f"type_logits shape: {type_logits.shape}"
    assert type_probs.shape == (batch_size, num_bands, 5), f"type_probs shape: {type_probs.shape}"
    assert filter_type.shape == (batch_size, num_bands), f"filter_type shape: {filter_type.shape}"

    # Bounds
    assert torch.all(gain_db >= -24.0) and torch.all(gain_db <= 24.0), "Gain out of bounds"
    assert torch.all(freq >= 20.0) and torch.all(freq <= 20000.0), f"Freq out of bounds: [{freq.min()}, {freq.max()}]"
    assert torch.all(q >= 0.1) and torch.all(q <= 10.0), "Q out of bounds"

    # Type probs should sum to ~1
    assert torch.allclose(type_probs.sum(dim=-1), torch.ones(batch_size, num_bands), atol=1e-5)

    # Eval mode (hard types)
    head.eval()
    gain_db2, freq2, q2, type_logits2, type_probs2, filter_type2 = head(embedding, hard_types=True)
    assert torch.allclose(type_probs2.sum(dim=-1), torch.ones(batch_size, num_bands), atol=1e-5)

    print(f"  Shapes: gain={gain_db.shape}, freq={freq.shape}, q={q.shape}")
    print(f"  Type logits: {type_logits.shape}, Type probs sum check: OK")
    print(f"  Bounds: gain [{gain_db.min():.1f}, {gain_db.max():.1f}], "
          f"freq [{freq.min():.0f}, {freq.max():.0f}], q [{q.min():.2f}, {q.max():.2f}]")


def test_soft_multitype_forward():
    """Test the soft (Gumbel-Softmax) forward pass through the DSP cascade."""
    print("Testing soft multitype forward pass...")

    batch_size = 4
    num_bands = 5
    cascade = DifferentiableBiquadCascade(num_bands, sample_rate=44100)
    head = MultiTypeEQParameterHead(128, num_bands)

    embedding = torch.randn(batch_size, 128)
    gain_db, freq, q, type_logits, type_probs, filter_type = head(embedding)

    # Soft forward
    H_mag_soft = cascade.forward_soft(gain_db, freq, q, type_probs, n_fft=1024)

    # Hard forward (for comparison)
    H_mag_hard = cascade(gain_db, freq, q, n_fft=1024, filter_type=filter_type)

    assert H_mag_soft.shape == (batch_size, 1024 // 2 + 1), f"Soft H_mag shape: {H_mag_soft.shape}"
    assert H_mag_hard.shape == (batch_size, 1024 // 2 + 1), f"Hard H_mag shape: {H_mag_hard.shape}"

    # Soft H_mag should be > 0 (magnitude response)
    assert torch.all(H_mag_soft > 0), "Soft magnitude response should be positive"

    # Gradients should flow
    loss = H_mag_soft.sum()
    loss.backward()
    assert embedding.grad is None  # embedding doesn't require_grad here

    # Test with requires_grad embedding
    embedding2 = torch.randn(batch_size, 128, requires_grad=True)
    gain_db2, freq2, q2, type_logits2, type_probs2, filter_type2 = head(embedding2)
    H_mag2 = cascade.forward_soft(gain_db2, freq2, q2, type_probs2, n_fft=1024)
    H_mag2.sum().backward()
    assert embedding2.grad is not None, "Gradient did not flow through soft forward"

    print(f"  Soft H_mag shape: {H_mag_soft.shape}")
    print(f"  Hard H_mag shape: {H_mag_hard.shape}")
    print("  Gradient flow through soft forward: OK")


if __name__ == "__main__":
    test_multitype_gradient_flow()
    print()
    test_peaking_backward_compat()
    print()
    test_hp_frequency_response()
    print()
    test_lp_frequency_response()
    print()
    test_shelf_response()
    print()
    test_multitype_parameter_head()
    print()
    test_soft_multitype_forward()
    print()
    print("All multi-type EQ tests passed!")
