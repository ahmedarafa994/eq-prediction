"""
Standalone test for Phase 3 loss architecture changes.
Tests: LOSS-01 (independent weights), LOSS-02 (warmup gating),
       LOSS-03 (log-cosh for gain), LOSS-04 (dual H_mag path),
       DATA-02 (Gumbel detach during warmup).
Run from: cd insight && python test_loss_architecture.py
"""
import torch
import torch.nn as nn
import sys
import os
import math
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from loss_multitype import MultiTypeEQLoss, log_cosh_loss


def make_dummy_inputs(B=2, N=5, F=1025, device="cpu"):
    """Create minimal valid inputs for MultiTypeEQLoss.forward()."""
    pred_gain = torch.randn(B, N, device=device)
    pred_freq = torch.rand(B, N, device=device) * (20000 - 20) + 20
    pred_q = torch.rand(B, N, device=device) * 9.9 + 0.1
    pred_type_logits = torch.randn(B, N, 5, device=device)
    pred_H_mag_soft = torch.ones(B, F, device=device)
    pred_H_mag_hard = torch.ones(B, F, device=device)
    target_gain = torch.randn(B, N, device=device)
    target_freq = torch.rand(B, N, device=device) * (20000 - 20) + 20
    target_q = torch.rand(B, N, device=device) * 9.9 + 0.1
    target_filter_type = torch.randint(0, 5, (B, N), device=device)
    target_H_mag = torch.ones(B, F, device=device)
    return (pred_gain, pred_freq, pred_q, pred_type_logits,
            pred_H_mag_soft, pred_H_mag_hard,
            target_gain, target_freq, target_q, target_filter_type, target_H_mag)


n_pass = 0
n_fail = 0


def run_test(name, fn):
    global n_pass, n_fail
    try:
        fn()
        print(f"  PASS: {name}")
        n_pass += 1
    except Exception as e:
        print(f"  FAIL: {name} -- {e}")
        n_fail += 1


def _call_forward(loss_fn, pred_gain, pred_freq, pred_q, pred_type_logits,
                  pred_H_mag_soft, pred_H_mag_hard,
                  target_gain, target_freq, target_q,
                  target_filter_type, target_H_mag, embedding=None):
    """Call forward() with either dual or single H_mag signature, depending on
    which the loss module currently supports."""
    import inspect
    params = list(inspect.signature(loss_fn.forward).parameters.keys())
    if "pred_H_mag_hard" in params:
        return loss_fn(
            pred_gain, pred_freq, pred_q, pred_type_logits,
            pred_H_mag_soft, pred_H_mag_hard,
            target_gain, target_freq, target_q,
            target_filter_type, target_H_mag,
            embedding=embedding,
        )
    else:
        return loss_fn(
            pred_gain, pred_freq, pred_q, pred_type_logits,
            pred_H_mag_soft,
            target_gain, target_freq, target_q,
            target_filter_type, target_H_mag,
            embedding=embedding,
        )


def test_log_cosh_wired():
    """LOSS-03: Verify log_cosh_loss is wired for gain regression."""
    loss_fn = MultiTypeEQLoss(warmup_epochs=0)
    pred_gain = torch.tensor([[2.0, -3.0]])
    target_gain = torch.tensor([[0.0, 0.0]])
    B, N = pred_gain.shape
    F = 1025
    pred_freq = torch.ones(B, N) * 1000.0
    pred_q = torch.ones(B, N) * 1.0
    pred_type_logits = torch.randn(B, N, 5)
    pred_H_mag_soft = torch.ones(B, F)
    pred_H_mag_hard = torch.ones(B, F)
    target_freq = torch.ones(B, N) * 1000.0
    target_q = torch.ones(B, N) * 1.0
    target_filter_type = torch.zeros(B, N, dtype=torch.long)
    target_H_mag = torch.ones(B, F)

    total_loss, components = _call_forward(
        loss_fn,
        pred_gain, pred_freq, pred_q, pred_type_logits,
        pred_H_mag_soft, pred_H_mag_hard,
        target_gain, target_freq, target_q,
        target_filter_type, target_H_mag,
    )

    assert components["loss_gain"] > 0, f"loss_gain should be positive, got {components['loss_gain']}"

    # Verify it is log_cosh, not Huber.
    # log_cosh(2.0, 0.0) ~ 1.325; Huber(2.0, 0.0, delta=5.0) = 2.0
    # Mean over 2 bands: (log_cosh(2.0)+log_cosh(3.0))/2
    expected_log_cosh_2 = log_cosh_loss(torch.tensor(2.0), torch.tensor(0.0))
    expected_log_cosh_3 = log_cosh_loss(torch.tensor(-3.0), torch.tensor(0.0))
    expected_mean = (expected_log_cosh_2 + expected_log_cosh_3) / 2.0
    actual = components["loss_gain"]
    assert abs(actual.item() - expected_mean.item()) < 0.01, (
        f"Expected log_cosh mean ~{expected_mean.item():.4f}, got {actual.item():.4f}"
    )


def test_independent_weights():
    """LOSS-01: Verify independent per-parameter weights are used (lambda_param=0)."""
    loss_fn = MultiTypeEQLoss(
        lambda_param=0.0, lambda_gain=2.5, lambda_freq=1.5, lambda_q=1.5,
        warmup_epochs=0,
    )
    assert loss_fn.lambda_param == 0.0, "lambda_param should be 0.0"
    loss_fn.current_epoch = 10  # past warmup

    (pred_gain, pred_freq, pred_q, pred_type_logits,
     pred_H_mag_soft, pred_H_mag_hard,
     target_gain, target_freq, target_q, target_filter_type, target_H_mag) = make_dummy_inputs()

    total_loss, components = _call_forward(
        loss_fn,
        pred_gain, pred_freq, pred_q, pred_type_logits,
        pred_H_mag_soft, pred_H_mag_hard,
        target_gain, target_freq, target_q,
        target_filter_type, target_H_mag,
    )

    assert "param_loss" in components, "param_loss key should exist in components"
    assert components["loss_gain"] > 0, "loss_gain should be positive"
    assert components["loss_freq"] > 0, "loss_freq should be positive"
    assert components["loss_q"] > 0, "loss_q should be positive"

    # Verify the weighted sum matches expected contribution
    expected_param = (
        2.5 * components["loss_gain"]
        + 1.5 * components["loss_freq"]
        + 1.5 * components["loss_q"]
    )
    actual_param = components["param_loss"]
    assert abs(expected_param.item() - actual_param.item()) < 1e-4, (
        f"Weighted param mismatch: expected {expected_param.item():.6f}, "
        f"got {actual_param.item():.6f}"
    )


def test_warmup_gating():
    """LOSS-02: Verify warmup gating zeros freq/Q/type during warmup."""
    loss_fn = MultiTypeEQLoss(warmup_epochs=5)
    loss_fn.current_epoch = 2  # warmup active
    # Set gain_mae_ema high so hybrid gate keeps warmup on
    if hasattr(loss_fn, 'gain_mae_ema'):
        loss_fn.gain_mae_ema = 999.0

    (pred_gain, pred_freq, pred_q, pred_type_logits,
     pred_H_mag_soft, pred_H_mag_hard,
     target_gain, target_freq, target_q, target_filter_type, target_H_mag) = make_dummy_inputs()

    total_loss, components = _call_forward(
        loss_fn,
        pred_gain, pred_freq, pred_q, pred_type_logits,
        pred_H_mag_soft, pred_H_mag_hard,
        target_gain, target_freq, target_q,
        target_filter_type, target_H_mag,
    )

    assert components["loss_freq"] == 0.0, (
        f"loss_freq should be 0 during warmup, got {components['loss_freq']}"
    )
    assert components["loss_q"] == 0.0, (
        f"loss_q should be 0 during warmup, got {components['loss_q']}"
    )
    assert components["type_loss"] == 0.0, (
        f"type_loss should be 0 during warmup, got {components['type_loss']}"
    )

    # After warmup: epoch >= 5 and gain_mae_ema should allow activation
    loss_fn.current_epoch = 5
    if hasattr(loss_fn, 'gain_mae_ema'):
        loss_fn.gain_mae_ema = 1.0  # below 2.5 threshold

    (pred_gain, pred_freq, pred_q, pred_type_logits,
     pred_H_mag_soft, pred_H_mag_hard,
     target_gain, target_freq, target_q, target_filter_type, target_H_mag) = make_dummy_inputs()

    total_loss, components = _call_forward(
        loss_fn,
        pred_gain, pred_freq, pred_q, pred_type_logits,
        pred_H_mag_soft, pred_H_mag_hard,
        target_gain, target_freq, target_q,
        target_filter_type, target_H_mag,
    )

    assert components["loss_freq"] > 0.0, (
        f"loss_freq should be positive after warmup, got {components['loss_freq']}"
    )


def test_dual_hmag_signature():
    """LOSS-04: Verify forward() accepts separate pred_H_mag_soft and pred_H_mag_hard."""
    import inspect
    sig = inspect.signature(MultiTypeEQLoss.forward)
    params = list(sig.parameters.keys())
    assert "pred_H_mag_soft" in params, (
        f"pred_H_mag_soft not in forward signature: {params}"
    )
    assert "pred_H_mag_hard" in params, (
        f"pred_H_mag_hard not in forward signature: {params}"
    )

    loss_fn = MultiTypeEQLoss(warmup_epochs=0)
    loss_fn.current_epoch = 10

    B, N, F = 2, 5, 1025
    pred_gain = torch.randn(B, N)
    pred_freq = torch.ones(B, N) * 1000.0
    pred_q = torch.ones(B, N) * 1.0
    pred_type_logits = torch.randn(B, N, 5)
    pred_H_mag_soft = torch.ones(B, F) * 2.0
    pred_H_mag_hard = torch.ones(B, F) * 3.0
    target_gain = torch.randn(B, N)
    target_freq = torch.ones(B, N) * 1000.0
    target_q = torch.ones(B, N) * 1.0
    target_filter_type = torch.zeros(B, N, dtype=torch.long)
    target_H_mag = torch.ones(B, F) * 1.0

    total_loss, components = loss_fn(
        pred_gain, pred_freq, pred_q, pred_type_logits,
        pred_H_mag_soft, pred_H_mag_hard,
        target_gain, target_freq, target_q,
        target_filter_type, target_H_mag,
    )

    # Verify hmag_loss uses pred_H_mag_hard (detached), not pred_H_mag_soft.
    # With pred_H_mag_hard=3.0 and target_H_mag=1.0:
    #   hmag_loss = |log(3.0) - log(1.0)| = log(3.0) ~ 1.0986
    # With pred_H_mag_soft=2.0:
    #   hmag_loss = |log(2.0) - log(1.0)| = log(2.0) ~ 0.6931
    expected_hmag = abs(math.log(3.0) - math.log(1.0))
    actual_hmag = components["hmag_loss"].item()
    assert abs(actual_hmag - expected_hmag) < 0.01, (
        f"hmag_loss should use pred_H_mag_hard (expected ~{expected_hmag:.4f} from hard path), "
        f"got {actual_hmag:.4f} (if ~0.69, using soft path instead)"
    )


def test_gumbel_detach_warmup():
    """DATA-02: Verify type_logits are detached from gain gradient during warmup."""
    loss_fn = MultiTypeEQLoss(warmup_epochs=5)
    loss_fn.current_epoch = 2  # warmup active
    if hasattr(loss_fn, 'gain_mae_ema'):
        loss_fn.gain_mae_ema = 999.0  # keep warmup active

    B, N, F = 2, 5, 1025
    pred_gain = torch.randn(B, N)
    pred_freq = torch.ones(B, N) * 1000.0
    pred_q = torch.ones(B, N) * 1.0
    pred_type_logits = torch.randn(B, N, 5, requires_grad=True)
    pred_H_mag_soft = torch.ones(B, F)
    pred_H_mag_hard = torch.ones(B, F)
    target_gain = torch.randn(B, N)
    target_freq = torch.ones(B, N) * 1000.0
    target_q = torch.ones(B, N) * 1.0
    target_filter_type = torch.zeros(B, N, dtype=torch.long)
    target_H_mag = torch.ones(B, F)

    total_loss, components = _call_forward(
        loss_fn,
        pred_gain, pred_freq, pred_q, pred_type_logits,
        pred_H_mag_soft, pred_H_mag_hard,
        target_gain, target_freq, target_q,
        target_filter_type, target_H_mag,
    )

    loss_gain_val = components["loss_gain"]
    loss_gain_val.backward()

    assert pred_type_logits.grad is None, (
        f"During warmup, type_logits should NOT receive gradients from gain loss. "
        f"Got grad with shape {pred_type_logits.grad.shape if pred_type_logits.grad is not None else None}"
    )


def test_hybrid_warmup_gate():
    """D-03/D-04: Verify hybrid warmup gate (epoch AND gain_mae_ema, hard cap 15)."""
    loss_fn = MultiTypeEQLoss(warmup_epochs=5)

    # Case 1: epoch=6 (past threshold) but gain_mae_ema=3.0 (>2.5 dB) => warmup still active
    loss_fn.current_epoch = 6
    loss_fn.gain_mae_ema = 3.0

    (pred_gain, pred_freq, pred_q, pred_type_logits,
     pred_H_mag_soft, pred_H_mag_hard,
     target_gain, target_freq, target_q, target_filter_type, target_H_mag) = make_dummy_inputs()

    total_loss, components = _call_forward(
        loss_fn,
        pred_gain, pred_freq, pred_q, pred_type_logits,
        pred_H_mag_soft, pred_H_mag_hard,
        target_gain, target_freq, target_q,
        target_filter_type, target_H_mag,
    )

    assert components["loss_freq"] == 0.0, (
        f"Warmup should still be active (epoch=6, gain_mae_ema=3.0 > 2.5). "
        f"Got loss_freq={components['loss_freq']}"
    )

    # Case 2: epoch=6, gain_mae_ema=2.0 (< 2.5 dB) => warmup ends
    loss_fn.gain_mae_ema = 2.0

    (pred_gain, pred_freq, pred_q, pred_type_logits,
     pred_H_mag_soft, pred_H_mag_hard,
     target_gain, target_freq, target_q, target_filter_type, target_H_mag) = make_dummy_inputs()

    total_loss, components = _call_forward(
        loss_fn,
        pred_gain, pred_freq, pred_q, pred_type_logits,
        pred_H_mag_soft, pred_H_mag_hard,
        target_gain, target_freq, target_q,
        target_filter_type, target_H_mag,
    )

    assert components["loss_freq"] > 0.0, (
        f"Warmup should end (epoch=6, gain_mae_ema=2.0 < 2.5). "
        f"Got loss_freq={components['loss_freq']}"
    )

    # Case 3: epoch=14, gain_mae_ema=3.5 (above threshold, epoch < hard cap 15) => warmup still active
    loss_fn.current_epoch = 14
    loss_fn.gain_mae_ema = 3.5

    (pred_gain, pred_freq, pred_q, pred_type_logits,
     pred_H_mag_soft, pred_H_mag_hard,
     target_gain, target_freq, target_q, target_filter_type, target_H_mag) = make_dummy_inputs()

    total_loss, components = _call_forward(
        loss_fn,
        pred_gain, pred_freq, pred_q, pred_type_logits,
        pred_H_mag_soft, pred_H_mag_hard,
        target_gain, target_freq, target_q,
        target_filter_type, target_H_mag,
    )

    assert components["loss_freq"] == 0.0, (
        f"Warmup should still be active at epoch=14 (hard cap=15). "
        f"Got loss_freq={components['loss_freq']}"
    )

    # Case 4: epoch=15 (hard cap reached) => warmup ends regardless of gain_mae_ema
    loss_fn.current_epoch = 15
    loss_fn.gain_mae_ema = 3.5

    (pred_gain, pred_freq, pred_q, pred_type_logits,
     pred_H_mag_soft, pred_H_mag_hard,
     target_gain, target_freq, target_q, target_filter_type, target_H_mag) = make_dummy_inputs()

    total_loss, components = _call_forward(
        loss_fn,
        pred_gain, pred_freq, pred_q, pred_type_logits,
        pred_H_mag_soft, pred_H_mag_hard,
        target_gain, target_freq, target_q,
        target_filter_type, target_H_mag,
    )

    assert components["loss_freq"] > 0.0, (
        f"Hard cap at epoch=15 should force warmup end regardless of gain_mae_ema. "
        f"Got loss_freq={components['loss_freq']}"
    )


def test_warmup_config():
    """Verify warmup_epochs is present in conf/config.yaml under loss section."""
    config_path = os.path.join(os.path.dirname(__file__), "conf", "config.yaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    assert "loss" in cfg, "config.yaml should have a 'loss' section"
    assert "warmup_epochs" in cfg["loss"], (
        "warmup_epochs should be present under loss section in config.yaml"
    )
    warmup_val = cfg["loss"]["warmup_epochs"]
    assert isinstance(warmup_val, int) and warmup_val == 5, (
        f"warmup_epochs should be integer 5, got {warmup_val}"
    )


if __name__ == "__main__":
    print("Phase 3 Loss Architecture Tests")
    print("=" * 50)

    run_test("test_log_cosh_wired", test_log_cosh_wired)
    run_test("test_independent_weights", test_independent_weights)
    run_test("test_warmup_gating", test_warmup_gating)
    run_test("test_dual_hmag_signature", test_dual_hmag_signature)
    run_test("test_gumbel_detach_warmup", test_gumbel_detach_warmup)
    run_test("test_hybrid_warmup_gate", test_hybrid_warmup_gate)
    run_test("test_warmup_config", test_warmup_config)

    print(f"\n{n_pass + n_fail} tests: {n_pass} passed, {n_fail} failed")
    sys.exit(0 if n_fail == 0 else 1)
