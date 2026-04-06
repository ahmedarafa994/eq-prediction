"""
Standalone test for Phase 3 loss architecture changes.
Tests: LOSS-01 (independent weights), LOSS-02 (warmup gating),
       LOSS-03 (log-cosh for gain), LOSS-04 (dual H_mag path),
       LOSS-05 (spectral reconstruction), LOSS-06 (activity mask),
       DATA-02 (Gumbel detach / hybrid warmup gate).

Run from: cd insight && python test_loss_architecture.py
"""
import torch
import torch.nn as nn
import sys
import os
import math

sys.path.insert(0, os.path.dirname(__file__))
from loss_multitype import MultiTypeEQLoss, log_cosh_loss


def make_dummy_inputs(B=2, N=5, F=1025, device="cpu"):
    """Create minimal valid inputs for MultiTypeEQLoss.forward() with dual H_mag."""
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


# ====================================================================
# LOSS-03: Log-cosh for gain
# ====================================================================
def test_log_cosh_wired():
    """Verify loss_gain uses log-cosh (not Huber)."""
    loss_fn = MultiTypeEQLoss(warmup_epochs=0)
    loss_fn.current_epoch = 10  # all losses active
    (pg, pf, pq, pt, psoft, phard, tg, tf, tq, tft, tH) = make_dummy_inputs()
    # Set specific values for gain to verify log-cosh computation
    pred_gain = torch.tensor([[2.0, -3.0]])
    target_gain = torch.tensor([[0.0, 0.0]])
    _, components = loss_fn(
        pred_gain, torch.rand(1, 2) * 10000 + 20, torch.rand(1, 2) * 9.9 + 0.1,
        torch.randn(1, 2, 5), torch.ones(1, 1025), torch.ones(1, 1025),
        target_gain, torch.rand(1, 2) * 10000 + 20, torch.rand(1, 2) * 9.9 + 0.1,
        torch.randint(0, 5, (1, 2)), torch.ones(1, 1025),
    )
    loss_gain = components["loss_gain"].item()
    # log_cosh(2.0) ≈ 1.325, log_cosh(3.0) ≈ 2.308, mean ≈ 1.817
    expected = (1.325 + 2.308) / 2
    assert abs(loss_gain - expected) < 0.1, f"loss_gain={loss_gain:.4f}, expected ~{expected:.4f}"


# ====================================================================
# LOSS-01: Independent loss weights
# ====================================================================
def test_independent_weights():
    """Verify lambda_param=0 path uses independent weights."""
    loss_fn = MultiTypeEQLoss(warmup_epochs=0, lambda_param=0.0, lambda_gain=2.5, lambda_freq=1.5, lambda_q=0.5)
    loss_fn.current_epoch = 10
    (pg, pf, pq, pt, psoft, phard, tg, tf, tq, tft, tH) = make_dummy_inputs()
    _, components = loss_fn(pg, pf, pq, pt, psoft, phard, tg, tf, tq, tft, tH)
    assert "loss_gain" in components
    assert "loss_freq" in components
    assert "loss_q" in components
    assert components["loss_gain"].item() > 0
    assert components["loss_freq"].item() > 0
    assert components["loss_q"].item() > 0


# ====================================================================
# LOSS-02: Warmup gating
# ====================================================================
def test_warmup_gating():
    """During warmup, only gain loss is active."""
    loss_fn = MultiTypeEQLoss(warmup_epochs=5)
    loss_fn.current_epoch = 2  # warmup active
    loss_fn.gain_mae_ema = 999  # keep warmup on via hybrid gate
    (pg, pf, pq, pt, psoft, phard, tg, tf, tq, tft, tH) = make_dummy_inputs()
    _, components = loss_fn(pg, pf, pq, pt, psoft, phard, tg, tf, tq, tft, tH)
    assert components["loss_gain"].item() > 0, "gain should be active"
    assert components["loss_freq"].item() < 0.001, f"freq should be ~zero during warmup: {components['loss_freq'].item()}"
    assert components["loss_q"].item() < 0.001, f"q should be ~zero during warmup: {components['loss_q'].item()}"
    assert components["type_loss"].item() == 0.0, "type should be zero"

    # After warmup
    loss_fn.current_epoch = 5
    loss_fn.gain_mae_ema = 1.0
    _, components = loss_fn(pg, pf, pq, pt, psoft, phard, tg, tf, tq, tft, tH)
    assert components["loss_freq"].item() > 0, "freq should be active after warmup"


# ====================================================================
# LOSS-04: Dual H_mag signature
# ====================================================================
def test_dual_hmag_signature():
    """Verify forward accepts pred_H_mag_soft and pred_H_mag_hard separately."""
    loss_fn = MultiTypeEQLoss(warmup_epochs=0)
    loss_fn.current_epoch = 10
    (pg, pf, pq, pt, psoft, phard, tg, tf, tq, tft, tH) = make_dummy_inputs()
    # Verify hmag_loss is computed from hard path (check detach doesn't break)
    _, components = loss_fn(pg, pf, pq, pt, psoft, phard, tg, tf, tq, tft, tH)
    assert components["hmag_loss"].item() >= 0


# ====================================================================
# DATA-02 / D-04: Hybrid warmup gate
# ====================================================================
def test_hybrid_warmup_gate():
    """Warmup ends when BOTH epoch AND gain_mae_ema thresholds are met."""
    loss_fn = MultiTypeEQLoss(warmup_epochs=5)

    # Past epoch but gain_mae_ema too high → still warmup
    loss_fn.current_epoch = 6
    loss_fn.gain_mae_ema = 5.0  # above 4.0 threshold → still warmup
    (pg, pf, pq, pt, psoft, phard, tg, tf, tq, tft, tH) = make_dummy_inputs()
    _, components = loss_fn(pg, pf, pq, pt, psoft, phard, tg, tf, tq, tft, tH)
    assert components["loss_freq"].item() < 0.01, f"Warmup should still be on (ramp ≈0): {components['loss_freq'].item()}"

    # Past epoch AND gain_mae_ema low → warmup ends
    loss_fn.gain_mae_ema = 2.0
    _, components = loss_fn(pg, pf, pq, pt, psoft, phard, tg, tf, tq, tft, tH)
    assert components["loss_freq"].item() > 0, "Warmup should end when gain converged"

    # Hard cap at epoch 15
    loss_fn.current_epoch = 15
    loss_fn.gain_mae_ema = 3.5  # above threshold but hard cap should force warmup end
    _, components = loss_fn(pg, pf, pq, pt, psoft, phard, tg, tf, tq, tft, tH)
    assert components["loss_freq"].item() > 0, "Hard cap at epoch 15 should force warmup end"


# ====================================================================
# LOSS-05: Spectral reconstruction
# ====================================================================
def test_spectral_reconstruction_fires():
    """Spectral loss fires when is_spectral_active (epoch >= warmup_epochs + 2)."""
    loss_fn = MultiTypeEQLoss(warmup_epochs=0, lambda_spectral=1.0)
    loss_fn.current_epoch = 2  # is_spectral_active: epoch >= 0 + 2 = 2

    B, N, F = 2, 5, 1025
    pred_H_soft = torch.ones(B, F) * 2.0  # double the target
    pred_H_hard = torch.ones(B, F) * 2.0
    pred_gain = torch.zeros(B, N)
    pred_freq = torch.full((B, N), 1000.0)
    pred_q = torch.full((B, N), 1.0)
    pred_type_logits = torch.zeros(B, N, 5)
    target_gain = torch.zeros(B, N)
    target_freq = torch.full((B, N), 1000.0)
    target_q = torch.full((B, N), 1.0)
    target_ft = torch.zeros(B, N, dtype=torch.long)
    target_H = torch.ones(B, F)

    _, comp = loss_fn(
        pred_gain, pred_freq, pred_q, pred_type_logits,
        pred_H_soft, pred_H_hard,
        target_gain, target_freq, target_q, target_ft, target_H,
    )
    # log(2.0) - log(1.0) = log(2.0) ≈ 0.693, scaled by spectral_weight ramp
    from loss_multitype import sigmoid_ramp
    sw = sigmoid_ramp(2, 0 + 5, width=3.0)  # warmup=0, spectral starts at warmup+5=5
    expected = math.log(2.0) * sw
    assert abs(comp["spectral_loss"].item() - expected) < 0.1, \
        f"spectral_loss={comp['spectral_loss'].item():.4f}, expected ~{expected:.4f} (sw={sw:.4f})"


# ====================================================================
# LOSS-06: Activity mask
# ====================================================================
def test_activity_mask_fires():
    """Activity loss fires when inactive bands have nonzero predicted gain."""
    loss_fn = MultiTypeEQLoss(warmup_epochs=0)
    loss_fn.current_epoch = 10

    B, N, F = 2, 5, 1025
    active_band_mask = torch.tensor([
        [True, True, False, True, False],
        [True, True, True, True, True],
    ])
    # Set nonzero gain in inactive positions for first batch element
    pred_gain = torch.zeros(B, N)
    pred_gain[0, 2] = 5.0  # inactive band, nonzero gain → should fire
    pred_gain[0, 4] = -3.0  # inactive band, nonzero gain → should fire

    pred_freq = torch.full((B, N), 1000.0)
    pred_q = torch.full((B, N), 1.0)
    pred_type_logits = torch.zeros(B, N, 5)
    target_gain = torch.zeros(B, N)
    target_freq = torch.full((B, N), 1000.0)
    target_q = torch.full((B, N), 1.0)
    target_ft = torch.zeros(B, N, dtype=torch.long)
    target_H = torch.ones(B, F)

    _, comp = loss_fn(
        pred_gain, pred_freq, pred_q, pred_type_logits,
        torch.ones(B, F), torch.ones(B, F),
        target_gain, target_freq, target_q, target_ft, target_H,
        active_band_mask=active_band_mask,
    )
    # Expected: mean of |5.0| + |-3.0| over 2 inactive bands in batch 0 = (5 + 3) / 2 / B*N
    # Actually: (pred_gain * inactive_mask).abs().mean() over entire (B, N) tensor
    # inactive: [0,2]=5, [0,4]=-3 → |5| + |3| = 8, mean over B*N=10 → 0.8
    expected = 8.0 / (B * N)
    assert abs(comp["activity_loss"].item() - expected) < 0.01, \
        f"activity_loss={comp['activity_loss'].item():.4f}, expected ~{expected:.4f}"


# ====================================================================
# Warmup config in YAML
# ====================================================================
def test_warmup_config():
    """Verify warmup_epochs key exists in config.yaml."""
    import yaml
    config_path = os.path.join(os.path.dirname(__file__), "conf", "config.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    assert "warmup_epochs" in cfg["loss"], "warmup_epochs missing from loss section"
    assert cfg["loss"]["warmup_epochs"] == 5, f"warmup_epochs={cfg['loss']['warmup_epochs']}, expected 5"


# ====================================================================
# update_gain_mae method
# ====================================================================
def test_update_gain_mae():
    """Verify update_gain_mae updates the EMA correctly."""
    loss_fn = MultiTypeEQLoss(warmup_epochs=5)
    initial_ema = loss_fn.gain_mae_ema
    loss_fn.update_gain_mae(5.0, alpha=0.1)
    expected = 0.1 * 5.0 + 0.9 * initial_ema
    assert abs(loss_fn.gain_mae_ema - expected) < 1e-6, \
        f"gain_mae_ema={loss_fn.gain_mae_ema}, expected {expected}"


# ====================================================================
# Main
# ====================================================================
if __name__ == "__main__":
    print("=== Phase 3 Loss Architecture Tests ===")
    run_test("log_cosh_wired (LOSS-03)", test_log_cosh_wired)
    run_test("independent_weights (LOSS-01)", test_independent_weights)
    run_test("warmup_gating (LOSS-02)", test_warmup_gating)
    run_test("dual_hmag_signature (LOSS-04)", test_dual_hmag_signature)
    run_test("hybrid_warmup_gate (DATA-02)", test_hybrid_warmup_gate)
    run_test("spectral_reconstruction (LOSS-05)", test_spectral_reconstruction_fires)
    run_test("activity_mask (LOSS-06)", test_activity_mask_fires)
    run_test("warmup_config (config.yaml)", test_warmup_config)
    run_test("update_gain_mae", test_update_gain_mae)

    print(f"\n{n_pass + n_fail} tests: {n_pass} passed, {n_fail} failed")
    sys.exit(0 if n_fail == 0 else 1)
