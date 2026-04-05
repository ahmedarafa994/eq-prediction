# Testing Patterns

**Analysis Date:** 2026-04-05

## Test Framework

**Runner:** No framework (no pytest, unittest, or pytest.ini found)

**Execution:** Tests are standalone Python scripts run directly with `python test_<module>.py`

**Assertion Library:** `torch.testing.assert_close` for PyTorch tensors; standard `assert` statements

**Run Commands:**
```bash
python test_eq.py              # Differentiable biquad gradient flow
python test_model.py           # CNN model forward/inverse/cycle/gradient
python test_streaming.py       # TCN streaming mode, batch-vs-streaming consistency
python test_multitype_eq.py    # Multi-type filters (HP/LP/shelf) + parameter head
python test_checkpoint.py      # Best checkpoint evaluation
python test_checkpoint_multi.py # Multi-checkpoint evaluation
python test_collapse_fix.py    # TCN encoder collapse fix
python test_ste_clamp.py       # Straight-through estimator clamp
python test_lightning_dummy.py # PyTorch Lightning fast_dev_run smoke test
python fixes/test_fixes.py     # Experimental fix unit tests
```

## Test File Organization

**Location:** Tests are co-located with source code in `insight/` directory

**Naming:**
- Test files: `test_<module>.py` pattern
- Fix tests: `fixes/test_fixes.py` in experimental fix directory

**Structure:**
- Single file contains multiple test functions
- Tests organized by functionality (gradient flow, shape checking, frequency response)
- Main execution in `if __name__ == "__main__":` block

**Examples:**
```
insight/
├── test_eq.py                  # EQ layer tests
├── test_model.py               # CNN model tests
├── test_streaming.py           # TCN streaming tests
├── test_multitype_eq.py        # Multi-type EQ tests
├── test_checkpoint.py          # Checkpoint evaluation
└── fixes/
    └── test_fixes.py           # Experimental fix tests
```

## Test Structure

**Suite Organization:**
```python
def test_gradient_flow():
    """Verify gradients flow through the entire TCN model."""
    print("Testing TCN gradient flow...")

    # 1. Setup
    model = StreamingTCNModel(n_mels=128, embedding_dim=128, num_bands=5)
    mel_input = torch.randn(4, 128, 64, requires_grad=True)

    # 2. Forward Pass
    output = model(mel_input)
    loss = output["H_mag"].sum()
    loss.backward()

    # 3. Assertions
    assert mel_input.grad is not None, "No gradient for mel input"
    print(f"  Gradient norm: {mel_input.grad.norm().item():.4f}")
    print("  Gradient flow OK")


def test_frequency_response():
    """Verify HP filter rolls off below cutoff frequency."""
    print("Testing high-pass frequency response shape...")

    cascade = DifferentiableBiquadCascade(num_bands=1, sample_rate=44100)

    # Setup parameters
    cutoff = 1000.0
    freq = torch.full((1, 1), cutoff)
    q = torch.full((1, 1), 0.707)
    filter_type = torch.full((1, 1), FILTER_HIGHPASS, dtype=torch.long)

    # Forward pass
    H_mag = cascade(gain_db, freq, q, n_fft=4096, filter_type=filter_type)

    # Verify shape
    assert H_mag.shape == (1, 2049)

    # Verify frequency response shape
    below_idx = freqs < cutoff * 0.5
    assert H_mag[below_idx].max() < 0.9
    print(f"  HP at {cutoff}Hz: below={H_mag[below_idx].max():.3f} - OK")


if __name__ == "__main__":
    test_tcn_gradient_flow()
    test_frequency_response()
    print("All streaming TCN tests passed!")
```

**Patterns:**
- Setup → Forward Pass → Assertions → Print Summary
- Helper functions for common checks
- Test functions are self-contained
- Verbose output with progress indicators

## Mocking

**Framework:** No mocking framework found (no unittest.mock, pytest.mock, etc.)

**Approach:**
- Direct tensor creation with torch.randn/torch.tensor
- Model instantiation with dummy parameters
- No mocking of external dependencies

**What to Mock:**
- External audio files: Not applicable (all synthetic)
- Model checkpoints: No mocking (use saved checkpoints)
- External APIs: Not applicable (internal model)

**What NOT to Mock:**
- PyTorch tensors and operations (used directly)
- Model components (test with actual model instances)
- DSP operations (test with actual differentiable DSP)

**Example:**
```python
# Direct tensor creation
model = StreamingTCNModel(n_mels=128, embedding_dim=128, num_bands=5)
batch_size = 4
mel_input = torch.randn(batch_size, 128, 64, requires_grad=True)

# Model instantiation (no mocks)
head = EQParameterHead(embedding_dim, num_bands)
cascade = DifferentiableBiquadCascade(num_bands)

# Direct assertions
assert torch.all(gain_db >= -24.0) and torch.all(gain_db <= 24.0)
assert torch.isfinite(H_mag).all()
```

## Fixtures and Factories

**Test Data:**
```python
# Helper for tensor checks
def check_tensor(name, tensor, expected_shape=None, finite=True):
    """Assert tensor properties."""
    assert tensor is not None, f"{name}: tensor is None"
    if expected_shape is not None:
        assert tensor.shape == expected_shape, \
            f"{name}: expected shape {expected_shape}, got {tensor.shape}"
    if finite:
        assert torch.isfinite(tensor).all(), \
            f"{name}: contains NaN or inf"
    print(f"  [PASS] {name}: shape={tuple(tensor.shape)}, "
          f"range=[{tensor.min().item():.4f}, {tensor.max().item():.4f}]")


# Helper for gradient checks
def check_gradient(name, tensor):
    """Assert gradient flows through tensor."""
    assert tensor.grad is not None, f"{name}: no gradient"
    assert torch.isfinite(tensor.grad).all(), \
        f"{name}: gradient contains NaN or inf"
    grad_norm = tensor.grad.norm().item()
    assert grad_norm > 0, f"{name}: zero gradient"
    print(f"  [PASS] {name}: grad_norm={grad_norm:.6f}")
```

**Test Data Factories:**
```python
# Small fixed batches for deterministic testing
batch_size = 4
num_bands = 5
embedding_dim = 128

# Reusable parameter generation
gain_db = torch.randn(batch_size, num_bands) * 5.0
freq = torch.sigmoid(torch.randn(batch_size, num_bands)) * (20000 - 20) + 20
q = torch.sigmoid(torch.randn(batch_size, num_bands)) * (10 - 0.1) + 0.1
```

**Location:**
- Test helpers defined within test files
- No separate fixtures directory
- Helper functions imported in test file if needed

## Coverage

**Requirements:** None enforced (no coverage tool detected)

**Coverage Tools:**
- No coverage configuration files found (no .coveragerc, pytest-cov, etc.)
- No coverage output in tests

**View Coverage:**
- No standard coverage command defined
- Tests run directly with `python test_*.py`

## Test Types

**Unit Tests:**
- **Scope:** Individual components (parameter head, biquad cascade, TCN blocks)
- **Approach:** Forward pass, gradient flow, shape checking, parameter bounds
- **Coverage:**
  - Parameter head: output shapes, bounds, gradient flow
  - Biquad cascade: coefficient computation, frequency response shapes
  - TCN model: gradient flow, batch vs streaming consistency, receptive field

**Integration Tests:**
- **Scope:** End-to-end workflows (forward pass through entire pipeline)
- **Approach:** Full model forward pass, loss computation, gradient backpropagation
- **Coverage:**
  - CNN model: forward/inverse filter, cycle consistency, gradient flow
  - Full pipeline: dataset loading, training step, loss computation

**E2E Tests:**
- **Framework:** Not used (no separate E2E test framework)
- **Approach:** Standalone test scripts simulating real usage
- **Coverage:**
  - Checkpoint evaluation against real audio
  - Multi-checkpoint comparison

**Performance Tests:**
- **Scope:** Model inference latency
- **Approach:** Benchmark metrics (median, P95, mean latency)
- **Coverage:**
  - Single-frame streaming latency
  - Warmup runs to stabilize measurements

**Examples:**
```python
# Unit test
def test_eq_gradients():
    """Test gradient flow through EQ layer."""
    head = EQParameterHead(embedding_dim, num_bands)
    cascade = DifferentiableBiquadCascade(num_bands)

    dummy_embedding = torch.randn(batch_size, embedding_dim, requires_grad=True)
    gain_db, freq, q = head(dummy_embedding)

    # Check bounds
    assert torch.all(gain_db >= -24.0) and torch.all(gain_db <= 24.0)

    # Forward pass
    H_mag = cascade(gain_db, freq, q, n_fft=n_fft)

    # Gradient check
    loss = torch.nn.functional.mse_loss(H_mag, target_H)
    loss.backward()
    assert dummy_embedding.grad is not None

# Integration test
def test_combined_loss():
    """Test full pipeline with dataset and loss."""
    model = EQEstimatorCNN(num_bands=num_bands, sample_rate=sample_rate)
    dataset = SyntheticEQDataset(num_bands=num_bands, ...)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    batch = next(iter(loader))
    output = model(batch["wet_audio"])
    loss = criterion(output["H_mag"], H_target)

    loss.backward()
    assert all(p.grad is not None for p in model.parameters())

# Performance test
def test_streaming_latency():
    """Benchmark single-frame streaming inference latency."""
    model.init_streaming(batch_size)
    times = []

    for _ in range(n_runs):
        frame = torch.randn(batch_size, 128)
        t0 = time.perf_counter()
        model.process_frame(frame)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times.sort()
    median_ms = times[n_runs // 2]
    print(f"  Median: {median_ms:.3f} ms")
```

## Common Patterns

**Async Testing:**
```python
# No async tests needed (PyTorch CPU/GPU operations)
# No asyncio tests found

# All tests are synchronous
```

**Error Testing:**
```python
# Gradient flow validation
assert dummy_embedding.grad is not None, "Gradients did not flow back!"

# Parameter bounds
assert torch.all(gain_db >= -24.0) and torch.all(gain_db <= 24.0), "Gain out of bounds"
assert torch.all(freq >= 20.0) and torch.all(freq <= 20000.0), "Freq out of bounds"
assert torch.all(q >= 0.1) and torch.all(q <= 10.0), "Q out of bounds"

# NaN/Inf checks
assert torch.isfinite(H_mag).all(), "Output contains NaN or inf"
assert torch.isfinite(loss).item(), "Loss contains NaN or inf"
```

**Shape Validation:**
```python
# Expected shapes
assert output["embedding"].shape == (B, 128)
assert H_mag.shape == (B, 1025)
assert gain_db.shape == (B, N), f"gain_db shape: {gain_db.shape}"

# Type verification
assert isinstance(output["params"], tuple)
assert len(output["params"]) == 3
```

**Statistical Validation:**
```python
# Type probability sums
assert torch.allclose(output["type_probs"].sum(dim=-1), torch.ones(B, N), atol=1e-4)

# Magnitude response positivity
assert torch.all(H_mag > 0), "Magnitude response should be positive"

# Gradient norm checks
assert grad_norm > 0, "Zero gradient detected"
print(f"  Gradient norm: {grad_norm:.4f}")
```

**Consistency Checks:**
```python
# Batch vs streaming consistency
batch_embedding = model(batch_mel)["embedding"]
streaming_embedding = model.process_frame(mel_frame)["embedding"]
diff = (batch_embedding - streaming_embedding).abs().max().item()
assert diff < 0.01, f"Streaming not consistent with batch: {diff}"

# Coefficient comparison (backward compatibility)
torch.testing.assert_close(b0_old, b0_new, atol=1e-6, rtol=1e-6)
```

---

*Testing analysis: 2026-04-05*
