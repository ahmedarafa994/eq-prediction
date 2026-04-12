"""
Reproducibility regression test.

AUDIT: MEDIUM-21 — Verifies that two runs with the same seed produce
identical results (at least for the first 10 training steps).
"""
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model_tcn import StreamingTCNModel
from dsp_frontend import STFTFrontend
from loss_multitype import MultiTypeEQLoss
from differentiable_eq import DifferentiableBiquadCascade


def _run_steps(n_steps=10, seed=42, device="cpu"):
    """Run n_steps of training and return loss trajectory."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    num_bands = 5
    model = StreamingTCNModel(
        n_mels=128, embedding_dim=128, num_bands=num_bands,
        channels=64, num_blocks=2, num_stacks=1,
        sample_rate=44100, n_fft=2048,
    ).to(device)
    model.train()

    criterion = MultiTypeEQLoss(
        n_fft=1024, sample_rate=44100,
        lambda_param=1.0, lambda_spectral=1.0, lambda_type=1.0,
        lambda_hmag=0.25, lambda_gain=1.0, lambda_freq=1.0, lambda_q=1.0,
        dsp_cascade=model.dsp_cascade,
    ).to(device)

    frontend = STFTFrontend(
        n_fft=2048, hop_length=256, win_length=2048,
        mel_bins=128, sample_rate=44100,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    losses = []
    for step in range(n_steps):
        # Generate deterministic batch
        batch_size = 4
        seq_len = 256
        wet_audio = torch.randn(batch_size, seq_len, device=device)
        dry_audio = torch.randn(batch_size, seq_len, device=device)
        target_gain = torch.randn(batch_size, num_bands, device=device) * 5.0
        target_freq = torch.sigmoid(torch.randn(batch_size, num_bands, device=device)) * (20000 - 20) + 20
        target_q = torch.sigmoid(torch.randn(batch_size, num_bands, device=device)) * (10 - 0.1) + 0.1
        target_ft = torch.randint(0, 5, (batch_size, num_bands), device=device)

        mel = frontend.mel_spectrogram(wet_audio).squeeze(1)
        output = model(mel, wet_audio=wet_audio)
        pred_gain, pred_freq, pred_q = output["params"]

        target_H_mag = model.dsp_cascade(
            target_gain, target_freq, target_q, filter_type=target_ft,
        )

        loss, _ = criterion(
            pred_gain, pred_freq, pred_q, output["type_logits"],
            output["H_mag_soft"], output["H_mag"],
            target_gain, target_freq, target_q, target_ft,
            target_H_mag, embedding=output["embedding"],
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses.append(loss.item())

    return losses


def test_reproducibility_deterministic():
    """Two runs with the same seed should produce identical loss trajectories."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    losses_a = _run_steps(n_steps=10, seed=42, device=device)
    losses_b = _run_steps(n_steps=10, seed=42, device=device)

    for i, (a, b) in enumerate(zip(losses_a, losses_b)):
        assert abs(a - b) < 1e-5, (
            f"Step {i}: loss diverged ({a:.6f} vs {b:.6f}) — non-deterministic training"
        )

    print(f"  [test_reproducibility_deterministic] PASSED — {len(losses_a)} steps identical")


def test_different_seeds_diverge():
    """Different seeds should produce different results (sanity check)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    losses_a = _run_steps(n_steps=10, seed=42, device=device)
    losses_b = _run_steps(n_steps=10, seed=123, device=device)

    # At least some steps should differ
    max_diff = max(abs(a - b) for a, b in zip(losses_a, losses_b))
    assert max_diff > 1e-3, (
        f"Different seeds produced identical results (max_diff={max_diff:.6f})"
    )

    print(f"  [test_different_seeds_diverge] PASSED — max_diff={max_diff:.6f}")


if __name__ == "__main__":
    print("Running reproducibility tests...")
    print()
    test_reproducibility_deterministic()
    test_different_seeds_diverge()
    print()
    print("All reproducibility tests passed!")
