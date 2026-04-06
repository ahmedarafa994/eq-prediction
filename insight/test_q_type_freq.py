"""Phase 4 test suite: Q MLP, focal loss, metric-gated curriculum, per-type accuracy."""
import sys, os, math
sys.path.insert(0, os.path.dirname(__file__))
import torch
import yaml

def test_q_mlp_structure():
    from differentiable_eq import MultiTypeEQParameterHead
    head = MultiTypeEQParameterHead(embedding_dim=64, num_bands=5)
    assert hasattr(head, 'q_mlp'), 'q_mlp not found'
    assert not hasattr(head, 'q_head'), 'Old q_head still exists'
    assert len(head.q_mlp) == 5
    assert head.q_mlp[4].out_features == 1
    return True

def test_q_output_bounds():
    from differentiable_eq import MultiTypeEQParameterHead
    head = MultiTypeEQParameterHead(embedding_dim=64, num_bands=5)
    embedding = torch.randn(4, 64)
    mel = torch.randn(4, 128)
    out = head(embedding, mel)
    q = out[2]  # (gain_db, freq, q, type_logits, type_probs, filter_type)
    assert (q >= 0.1).all() and (q <= 10.0).all(), f'Q OOB: {q.min():.3f}-{q.max():.3f}'
    assert not torch.isnan(q).any()
    return True

def test_q_gradient_flow():
    from differentiable_eq import MultiTypeEQParameterHead
    head = MultiTypeEQParameterHead(embedding_dim=64, num_bands=5)
    embedding = torch.randn(4, 64)
    mel = torch.randn(4, 128)
    out = head(embedding, mel)
    out[2].sum().backward()
    assert head.q_mlp[0].weight.grad is not None and head.q_mlp[0].weight.grad.abs().sum() > 0
    assert head.q_mlp[4].weight.grad is not None and head.q_mlp[4].weight.grad.abs().sum() > 0
    return True

def test_equalized_cost_matrix():
    from loss_multitype import HungarianBandMatcher
    m = HungarianBandMatcher()
    assert m.lambda_gain == 1.0 and m.lambda_freq == 1.0 and m.lambda_q == 1.0
    B, N = 2, 5
    cost = m.compute_cost_matrix(
        torch.randn(B,N), torch.rand(B,N)*19000+20, torch.rand(B,N)*9.9+0.1,
        torch.randn(B,N), torch.rand(B,N)*19000+20, torch.rand(B,N)*9.9+0.1,
    )
    assert cost.shape == (B, N, N) and not torch.isnan(cost).any()
    return True

def test_focal_loss_wired():
    from loss_multitype import MultiTypeEQLoss
    fn = MultiTypeEQLoss()
    assert fn.focal_gamma == 2.0
    assert hasattr(fn, 'type_class_weights') and fn.type_class_weights.shape == (5,)
    assert fn.type_class_weights[3] > fn.type_class_weights[0]
    assert not hasattr(fn, 'type_loss')
    return True

def test_metric_gated_curriculum():
    with open(os.path.join(os.path.dirname(__file__), 'conf/config.yaml')) as f:
        cfg = yaml.safe_load(f)
    stages = cfg['curriculum']['stages']
    for i, s in enumerate(stages):
        assert 'metric_thresholds' in s, f"Stage {i} ({s['name']}) missing metric_thresholds"
        assert 'epoch_cap' in s, f"Stage {i} ({s['name']}) missing epoch_cap"
    assert 'gain_mae' in stages[2]['metric_thresholds']
    assert 'type_acc' in stages[2]['metric_thresholds']
    return True

def test_per_type_accuracy():
    from differentiable_eq import FILTER_NAMES
    assert len(FILTER_NAMES) == 5
    matched = torch.tensor([0, 0, 1, 2, 3, 4, 0])
    pred = torch.tensor([0, 1, 1, 2, 3, 3, 0])
    accs = {}
    for i, name in enumerate(FILTER_NAMES):
        mask = (matched == i)
        accs[name] = (pred[mask] == matched[mask]).float().mean().item() if mask.sum() > 0 else 0.0
    assert abs(accs['peaking'] - 2/3) < 0.01
    assert abs(accs['highpass'] - 1.0) < 0.01
    return True

if __name__ == "__main__":
    tests = [
        ("q_mlp_structure", test_q_mlp_structure),
        ("q_output_bounds", test_q_output_bounds),
        ("q_gradient_flow", test_q_gradient_flow),
        ("equalized_cost_matrix", test_equalized_cost_matrix),
        ("focal_loss_wired", test_focal_loss_wired),
        ("metric_gated_curriculum", test_metric_gated_curriculum),
        ("per_type_accuracy", test_per_type_accuracy),
    ]
    n_pass = n_fail = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
            n_pass += 1
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
            n_fail += 1
    print(f"\n{n_pass}/{n_pass+n_fail} passed")
    if n_fail:
        sys.exit(1)
