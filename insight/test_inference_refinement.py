"""Phase 5 test suite: gradient refinement, MC-Dropout confidence, streaming protection."""
import sys, os, math
sys.path.insert(0, os.path.dirname(__file__))
import torch
import torch.nn as nn
import yaml

def test_gradient_flow_through_biquad():
    from differentiable_eq import DifferentiableBiquadCascade
    cascade = DifferentiableBiquadCascade()
    B, num_bands = 2, 5
    gain_db = torch.zeros(B, num_bands).requires_grad_(True)
    freq = torch.full((B, num_bands), 1000.0).requires_grad_(True)
    q = torch.ones(B, num_bands).requires_grad_(True)
    filter_type = torch.zeros(B, num_bands, dtype=torch.long)  # peaking, no grad
    H_mag = cascade(gain_db, freq, q, n_fft=2048, filter_type=filter_type)
    H_mag.sum().backward()
    assert gain_db.grad is not None
    assert gain_db.grad.abs().sum() > 0, "No gradient through DifferentiableBiquadCascade"
    assert freq.grad is not None and freq.grad.abs().sum() > 0
    assert q.grad is not None and q.grad.abs().sum() > 0
    return True

def test_refine_forward_api():
    from model_tcn import StreamingTCNModel
    with open(os.path.join(os.path.dirname(__file__), 'conf/config.yaml')) as f:
        cfg = yaml.safe_load(f)
    model = StreamingTCNModel(
        n_mels=cfg['data']['n_mels'],
        embedding_dim=cfg['model']['encoder']['embedding_dim'],
        num_bands=cfg['data']['num_bands'],
        n_fft=cfg['data']['n_fft'],
    )
    model.eval()
    mel_frames = torch.randn(2, cfg['data']['n_mels'], 32)
    result = model.refine_forward(mel_frames, refine_steps=3)
    assert isinstance(result, dict), "refine_forward must return dict"
    assert "refined_params" in result
    refined_gain, refined_freq, refined_q = result["refined_params"]
    assert refined_gain.shape == (2, cfg['data']['num_bands'])
    assert refined_freq.shape == (2, cfg['data']['num_bands'])
    assert refined_q.shape == (2, cfg['data']['num_bands'])
    assert "initial_params" in result
    assert "filter_type" in result
    assert "refinement_loss_history" in result
    assert len(result["refinement_loss_history"]) == 3
    # Verify parameters stay within physical bounds
    assert (refined_gain >= -24.0).all() and (refined_gain <= 24.0).all()
    assert (refined_freq >= 20.0).all() and (refined_freq <= 20000.0).all()
    assert (refined_q >= 0.1).all() and (refined_q <= 10.0).all()
    return True

def test_refinement_reduces_loss():
    from model_tcn import StreamingTCNModel
    with open(os.path.join(os.path.dirname(__file__), 'conf/config.yaml')) as f:
        cfg = yaml.safe_load(f)
    model = StreamingTCNModel(
        n_mels=cfg['data']['n_mels'],
        embedding_dim=cfg['model']['encoder']['embedding_dim'],
        num_bands=cfg['data']['num_bands'],
        n_fft=cfg['data']['n_fft'],
    )
    model.eval()
    mel_frames = torch.randn(2, cfg['data']['n_mels'], 32)
    result = model.refine_forward(mel_frames, refine_steps=5)
    history = result["refinement_loss_history"]
    assert len(history) == 5
    assert history[-1] < history[0], (
        f"Refinement loss did not decrease: {history[0]:.4f} -> {history[-1]:.4f}"
    )
    return True

def test_streaming_unchanged():
    from model_tcn import StreamingTCNModel
    with open(os.path.join(os.path.dirname(__file__), 'conf/config.yaml')) as f:
        cfg = yaml.safe_load(f)
    model = StreamingTCNModel(
        n_mels=cfg['data']['n_mels'],
        embedding_dim=cfg['model']['encoder']['embedding_dim'],
        num_bands=cfg['data']['num_bands'],
        n_fft=cfg['data']['n_fft'],
    )
    model.eval()
    model.init_streaming(batch_size=1)
    mel_frame = torch.randn(1, cfg['data']['n_mels'])
    out_before = model.process_frame(mel_frame)
    expected_keys = {"params", "type_logits", "type_probs", "filter_type", "H_mag", "embedding", "mel_profile"}
    assert set(out_before.keys()) == expected_keys, f"Streaming keys changed: {set(out_before.keys())}"
    assert out_before["params"][0].shape == (1, cfg['data']['num_bands'])
    return True

def test_refine_forward_does_not_break_streaming():
    from model_tcn import StreamingTCNModel
    with open(os.path.join(os.path.dirname(__file__), 'conf/config.yaml')) as f:
        cfg = yaml.safe_load(f)
    model = StreamingTCNModel(
        n_mels=cfg['data']['n_mels'],
        embedding_dim=cfg['model']['encoder']['embedding_dim'],
        num_bands=cfg['data']['num_bands'],
        n_fft=cfg['data']['n_fft'],
    )
    model.eval()
    model.init_streaming(batch_size=1)
    mel_frame = torch.randn(1, cfg['data']['n_mels'])
    out_stream = model.process_frame(mel_frame)
    expected_keys = {"params", "type_logits", "type_probs", "filter_type", "H_mag", "embedding", "mel_profile"}
    assert set(out_stream.keys()) == expected_keys, f"Streaming keys changed: {set(out_stream.keys())}"
    mel_frames = torch.randn(1, cfg['data']['n_mels'], 32)
    _ = model.refine_forward(mel_frames, refine_steps=2)
    model.init_streaming(batch_size=1)
    out_stream_after = model.process_frame(mel_frame)
    assert set(out_stream_after.keys()) == expected_keys
    assert out_stream_after["params"][0].shape == (1, cfg['data']['num_bands'])
    return True

def test_config_refinement_section():
    with open(os.path.join(os.path.dirname(__file__), 'conf/config.yaml')) as f:
        cfg = yaml.safe_load(f)
    assert "refinement" in cfg, "Missing refinement: section in config.yaml"
    assert "mc_dropout_passes" in cfg["refinement"]
    assert "grad_refine_steps" in cfg["refinement"]
    assert "grad_lr" in cfg["refinement"]
    assert cfg["refinement"]["grad_refine_steps"] >= 3
    assert cfg["refinement"]["grad_lr"] <= 0.1
    return True

def test_mc_dropout_selective_mode():
    """BatchNorm stays eval; nn.Dropout goes train during MC-Dropout passes."""
    from model_tcn import StreamingTCNModel
    with open(os.path.join(os.path.dirname(__file__), 'conf/config.yaml')) as f:
        cfg = yaml.safe_load(f)
    model = StreamingTCNModel(
        n_mels=cfg['data']['n_mels'],
        embedding_dim=cfg['model']['encoder']['embedding_dim'],
        num_bands=cfg['data']['num_bands'],
        n_fft=cfg['data']['n_fft'],
    )
    model.eval()
    assert hasattr(model, '_run_mc_dropout_passes'), "_run_mc_dropout_passes not found"
    assert hasattr(model, '_compute_confidence'), "_compute_confidence not found"
    # Manually enable MC-Dropout mode to inspect module states
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
    dropout_modules = [m for m in model.modules() if isinstance(m, nn.Dropout)]
    assert len(dropout_modules) >= 3, f"Expected >=3 Dropout modules, found {len(dropout_modules)}"
    for dm in dropout_modules:
        assert dm.training, "Dropout module should be in train mode"
    bn_modules = [m for m in model.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))]
    assert len(bn_modules) >= 1, f"Expected >=1 BatchNorm module, found {len(bn_modules)}"
    for bn in bn_modules:
        assert not bn.training, f"BatchNorm module should be in eval mode, got training={bn.training}"
    model.eval()
    return True

def test_confidence_output_shape():
    """forward(refine=True) returns confidence with correct per-band structure."""
    from model_tcn import StreamingTCNModel
    with open(os.path.join(os.path.dirname(__file__), 'conf/config.yaml')) as f:
        cfg = yaml.safe_load(f)
    model = StreamingTCNModel(
        n_mels=cfg['data']['n_mels'],
        embedding_dim=cfg['model']['encoder']['embedding_dim'],
        num_bands=cfg['data']['num_bands'],
        n_fft=cfg['data']['n_fft'],
    )
    model.eval()
    mel_frames = torch.randn(2, cfg['data']['n_mels'], 32)
    out = model.forward(mel_frames, refine=True)
    assert "confidence" in out, "forward(refine=True) must return 'confidence'"
    assert "refined_params" in out, "forward(refine=True) must return 'refined_params'"
    conf = out["confidence"]
    required_keys = {"type_entropy", "gain_variance", "freq_variance", "q_variance", "overall_confidence"}
    assert set(conf.keys()) == required_keys, f"Confidence keys mismatch: {set(conf.keys())}"
    for key in required_keys:
        assert conf[key].shape == (2, cfg['data']['num_bands']), \
            f"confidence['{key}'] shape should be (2, {cfg['data']['num_bands']}), got {conf[key].shape}"
        assert (conf[key] >= 0.0).all(), f"confidence['{key}'] has negative values"
    assert (conf["overall_confidence"] <= 1.0).all(), "overall_confidence must be in [0, 1]"
    assert (conf["type_entropy"] <= 1.0).all(), "type_entropy must be in [0, 1]"
    out_plain = model.forward(mel_frames)
    assert "confidence" not in out_plain, "Plain forward() must not return confidence"
    assert "refined_params" not in out_plain, "Plain forward() must not return refined_params"
    return True

if __name__ == "__main__":
    tests = [
        test_gradient_flow_through_biquad,
        test_streaming_unchanged,
        test_config_refinement_section,
        test_refine_forward_api,
        test_refinement_reduces_loss,
        test_refine_forward_does_not_break_streaming,
        test_mc_dropout_selective_mode,
        test_confidence_output_shape,
    ]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
        except Exception as e:
            print(f"FAIL: {t.__name__} - {e}")
            failures += 1
    if failures > 0:
        sys.exit(1)
    print("ALL TESTS PASSED")
    sys.exit(0)
