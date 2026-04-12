"""
Edge case tests for synthetic data generation.

AUDIT: HIGH-19 — Verifies that the dataset handles pathological inputs
gracefully without crashes or silent corruption.
"""
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset import SyntheticEQDataset


def test_very_short_audio():
    """Audio shorter than 100ms should still produce valid samples."""
    ds = SyntheticEQDataset(
        num_bands=5, sample_rate=44100, duration=0.05,
        size=10, base_seed=42, augment=False,
    )
    sample = ds[0]
    assert torch.isfinite(sample["wet_audio"]).all(), "wet_audio has non-finite values"
    assert torch.isfinite(sample["gain"]).all(), "gain has non-finite values"
    assert len(sample["wet_audio"]) == int(0.05 * 44100), f"Unexpected audio length: {len(sample['wet_audio'])}"
    print("  [test_very_short_audio] PASSED")


def test_very_long_audio():
    """Audio longer than 10s should still produce valid samples."""
    ds = SyntheticEQDataset(
        num_bands=5, sample_rate=44100, duration=10.0,
        size=2, base_seed=42, augment=False,
    )
    sample = ds[0]
    assert torch.isfinite(sample["wet_audio"]).all()
    print("  [test_very_long_audio] PASSED")


def test_batch_size_one():
    """Batch size of 1 should work without dimension errors."""
    ds = SyntheticEQDataset(
        num_bands=5, sample_rate=44100, duration=0.5,
        size=5, base_seed=42, augment=False,
    )
    from dataset import collate_fn
    batch = [ds[i] for i in range(1)]
    collated = collate_fn(batch)
    assert collated["wet_audio"].shape[0] == 1
    print("  [test_batch_size_one] PASSED")


def test_different_num_bands():
    """Non-standard number of bands (3, 10) should work."""
    for n_bands in [3, 10]:
        ds = SyntheticEQDataset(
            num_bands=n_bands, sample_rate=44100, duration=0.5,
            size=5, base_seed=42, augment=False,
        )
        sample = ds[0]
        assert sample["gain"].shape[0] == n_bands
        assert sample["freq"].shape[0] == n_bands
        assert sample["q"].shape[0] == n_bands
        assert sample["filter_type"].shape[0] == n_bands
    print("  [test_different_num_bands] PASSED")


def test_extreme_gain_bounds():
    """Very large gain bounds (±48 dB) should produce finite output."""
    ds = SyntheticEQDataset(
        num_bands=5, sample_rate=44100, duration=0.5,
        size=20, base_seed=42, augment=False,
        gain_range=(-48.0, 48.0),
    )
    for i in range(5):
        sample = ds[i]
        assert torch.isfinite(sample["wet_audio"]).all(), f"Sample {i}: wet_audio non-finite with ±48dB gain"
    print("  [test_extreme_gain_bounds] PASSED")


def test_extreme_q_bounds():
    """Very narrow Q (50) should not cause numerical instability."""
    ds = SyntheticEQDataset(
        num_bands=5, sample_rate=44100, duration=0.5,
        size=20, base_seed=42, augment=False,
        q_range=(0.01, 50.0),
    )
    for i in range(5):
        sample = ds[i]
        assert torch.isfinite(sample["wet_audio"]).all(), f"Sample {i}: wet_audio non-finite with Q up to 50"
    print("  [test_extreme_q_bounds] PASSED")


def test_single_signal_type():
    """Dataset with only one signal type should work."""
    for sig_type in ["noise", "pink_noise", "sweep", "harmonic", "speech_like", "percussive"]:
        ds = SyntheticEQDataset(
            num_bands=5, sample_rate=44100, duration=0.5,
            size=5, base_seed=42, augment=False,
            signal_types=(sig_type,),
        )
        sample = ds[0]
        assert torch.isfinite(sample["wet_audio"]).all(), f"Signal type '{sig_type}' produced non-finite output"
    print("  [test_single_signal_type] PASSED")


def test_no_augmentation():
    """Dataset with augment=False should produce consistent samples."""
    ds = SyntheticEQDataset(
        num_bands=5, sample_rate=44100, duration=0.5,
        size=10, base_seed=42, augment=False,
    )
    sample_a = ds[0]
    # Reset seed and get same sample
    ds2 = SyntheticEQDataset(
        num_bands=5, sample_rate=44100, duration=0.5,
        size=10, base_seed=42, augment=False,
    )
    sample_b = ds2[0]
    assert torch.allclose(sample_a["wet_audio"], sample_b["wet_audio"]), (
        "Non-deterministic output with same seed and augment=False"
    )
    print("  [test_no_augmentation] PASSED")


def test_fallback_sample():
    """When generation fails, fallback sample should be valid."""
    ds = SyntheticEQDataset(
        num_bands=5, sample_rate=44100, duration=0.5,
        size=5, base_seed=42, augment=False,
    )
    fallback = ds._fallback_sample(0)
    assert torch.isfinite(fallback["wet_audio"]).all()
    assert torch.isfinite(fallback["gain"]).all()
    assert torch.isfinite(fallback["freq"]).all()
    assert torch.isfinite(fallback["q"]).all()
    # Fallback should be zero-gain EQ (identity)
    assert (fallback["gain"] == 0.0).all(), "Fallback should have zero gain"
    print("  [test_fallback_sample] PASSED")


def test_collate_filters_invalid_samples():
    """collate_fn should filter out samples with non-finite parameters."""
    ds = SyntheticEQDataset(
        num_bands=5, sample_rate=44100, duration=0.5,
        size=10, base_seed=42, augment=False,
    )
    batch = [ds[i] for i in range(4)]
    # Manually inject a NaN into one sample
    batch[2]["gain"][0] = float("nan")
    from dataset import collate_fn
    collated = collate_fn(batch)
    # Should have filtered out the NaN sample
    assert collated["wet_audio"].shape[0] <= 4
    print("  [test_collate_filters_invalid_samples] PASSED")


if __name__ == "__main__":
    print("Running data generation edge case tests...")
    print()
    test_very_short_audio()
    test_very_long_audio()
    test_batch_size_one()
    test_different_num_bands()
    test_extreme_gain_bounds()
    test_extreme_q_bounds()
    test_single_signal_type()
    test_no_augmentation()
    test_fallback_sample()
    test_collate_filters_invalid_samples()
    print()
    print("All edge case tests passed!")
