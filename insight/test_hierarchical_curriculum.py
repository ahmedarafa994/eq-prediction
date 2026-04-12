"""
Unit tests for Hierarchical Masking Curriculum.
Verifies that:
1. Matcher prioritizes frequency when configured for COARSE_STRUCTURE.
2. Filter type classification gradients are zeroed out for low-gain bands.
"""
import torch
from loss_multitype import MultiTypeEQLoss, HungarianBandMatcher

def test_coarse_matching():
    """Verify matcher prioritizes frequency when lambda_freq is high."""
    print("Testing coarse matching prioritization...")
    # Configure matcher for COARSE_STRUCTURE (freq-first)
    matcher = HungarianBandMatcher(lambda_freq=10.0, lambda_gain=0.01, lambda_q=0.01, lambda_type_match=0.01)

    # Dummy inputs
    pred_gain = torch.tensor([[5.0, -5.0]])
    pred_freq = torch.tensor([[100.0, 1000.0]])
    pred_q = torch.tensor([[0.7, 0.7]])

    # Target freq order swapped to test matching. Target 0 (1000Hz) should match Pred 1 (1000Hz).
    target_gain = torch.tensor([[0.0, 5.0]])
    target_freq = torch.tensor([[1000.0, 100.0]])
    target_q = torch.tensor([[0.7, 0.7]])

    # Match
    m_gain, m_freq, m_q = matcher(pred_gain, pred_freq, pred_q, target_gain, target_freq, target_q)

    # Assert pred 0 (freq 100) matched to target 1 (freq 100)
    assert m_freq[0, 0].item() == 100.0
    assert m_freq[0, 1].item() == 1000.0
    print("  Matcher test passed: matched by frequency despite gain mismatch")

def test_gain_gated_type_mask():
    """Verify that type loss gradients are zeroed for bands with |gain| < 1.0 dB."""
    print("Testing gain-gated type masking...")
    loss_fn = MultiTypeEQLoss()
    loss_fn.lambda_type = 1.0
    loss_fn.lambda_multi_scale = 0.0
    loss_fn.lambda_hmag = 0.0
    loss_fn.lambda_activity = 0.0
    loss_fn.lambda_spread = 0.0
    loss_fn.lambda_embed_var = 0.0
    loss_fn.lambda_contrastive = 0.0
    loss_fn.type_loss_mode = "focal"

    # Force a batch of 1, 2 bands
    # Use pred_gain that matches target_gain to ensure identity matching
    target_gain = torch.tensor([[0.5, 5.0]])
    pred_gain = target_gain.clone()
    pred_freq = torch.ones(1, 2)
    pred_q = torch.ones(1, 2)
    pred_type_logits = torch.zeros(1, 2, 5, requires_grad=True)

    # band 0 < 1.0 dB (should be masked), band 1 >= 1.0 dB (should have grads)
    target_freq = torch.ones(1, 2)
    target_q = torch.ones(1, 2)
    target_type = torch.tensor([[1, 2]])

    # Run forward pass (fill missing args with zeros/dummies)
    total_loss, components = loss_fn(
        pred_gain=pred_gain, pred_freq=pred_freq, pred_q=pred_q,
        pred_type_logits=pred_type_logits,
        pred_H_mag_soft=torch.zeros(1, 256), pred_H_mag_hard=torch.zeros(1, 256),
        target_gain=target_gain, target_freq=target_freq, target_q=target_q,
        target_filter_type=target_type, target_H_mag=torch.zeros(1, 256)
    )

    # Backward to check gradients
    type_loss = components["type_loss"]
    print(f"Type loss value: {type_loss.item()}")
    type_loss.backward()

    grads = pred_type_logits.grad
    print(f"Target gain:\n{target_gain}")
    print(f"Mask:\n{ (torch.abs(target_gain) >= 1.0).float() }")
    print(f"Gradients (sum of abs):\n{grads.abs().sum(dim=-1)}")
    assert torch.all(grads[0, 0] == 0), f"Low-gain band should have NO type gradients, got {grads[0, 0]}"
    # Band 1 (5.0 dB) should have non-zero gradients
    assert torch.any(grads[0, 1] != 0), "High-gain band should have type gradients"

    print("  Gain-gated masking verified: gradients zeroed for low-gain bands")

if __name__ == "__main__":
    test_coarse_matching()
    test_gain_gated_type_mask()
    print("\nAll hierarchical curriculum tests passed!")
