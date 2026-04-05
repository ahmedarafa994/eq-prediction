# Coding Conventions

**Analysis Date:** 2026-04-05

## Naming Patterns

**Files:**
- Snake_case for Python files: `differentiable_eq.py`, `train.py`, `test_eq.py`
- Test files: `test_<module>.py` pattern, e.g., `test_model.py`, `test_streaming.py`
- Fix directory: `fixes/` contains experimental implementations and their tests
- Config files: `conf/*.yaml` for YAML-based configuration

**Functions:**
- Snake_case for functions: `compute_biquad_coeffs`, `forward_pass`, `apply_eq_cascade`
- Test functions: `test_<function_name>()` pattern for unit tests
- Diagnostic functions: `diagnose_<name>()`, `debug_<name>()` pattern

**Variables:**
- Snake_case for variables: `n_fft`, `batch_size`, `gain_db`, `mel_frames`
- Short descriptive names for loop variables: `i`, `t`, `b`, `n`

**Types/Constants:**
- PascalCase for classes: `StreamingTCNModel`, `DifferentiableBiquadCascade`, `HungarianBandMatcher`
- UPPER_SNAKE_CASE for constants: `FILTER_PEAKING`, `FILTER_LOWSHELF`, `NUM_FILTER_TYPES`

## Code Style

**Formatting:**
- 2-space indentation
- Blank lines between logical sections (docstrings, class definitions, function definitions)
- Maximum line length: ~100-120 characters (examples wrap for readability)

**Import Organization:**
1. Standard library imports (math, sys, collections, typing)
2. Third-party imports (torch, torchaudio, scipy, numpy)
3. Local module imports (relative imports from same directory)

**Example:**
```python
import math
import torch
import torch.nn as nn
from pathlib import Path
from differentiable_eq import DifferentiableBiquadCascade
from loss_multitype import MultiTypeEQLoss
```

## Error Handling

**Patterns:**
- Assert for debug checks with descriptive messages
- Type hints for function signatures
- Try-except for optional dependencies (e.g., bitsandbytes, deepspeed)

**Example:**
```python
try:
    import bitsandbytes as bnb
    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False

try:
    import deepspeed
    HAS_DEEPSPEED = True
except ImportError:
    HAS_DEEPSPEED = False
```

**Shape assertions:**
```python
assert output["embedding"].shape == (B, 128), f"embedding: {output['embedding'].shape}"
assert torch.all(gain_db >= -24.0) and torch.all(gain_db <= 24.0), "Gain out of bounds"
```

## Logging

**Framework:** `print()` statements for tests and diagnostics

**Patterns:**
- Test output includes success/failure messages with details
- Diagnostic tools print summary statistics and visualizations
- Verbose output includes intermediate values for debugging

**Example:**
```python
print("Testing TCN gradient flow...")
print(f"  Embedding shape: {output['embedding'].shape}")
print(f"  Gradient norm: {mel_input.grad.norm().item():.4f}")
print("  Gradient flow OK")
```

## Comments

**When to Comment:**
- Module-level docstrings describing purpose and usage
- Complex algorithm explanations (DSP coefficient formulas)
- Parameter ranges and constraints
- Workarounds for specific issues

**Docstring Pattern:**
```python
def test_tcn_gradient_flow():
    """
    Verify gradients flow through the entire TCN model.

    This test ensures:
      - Forward pass produces expected shapes
      - Backward pass propagates gradients to input
      - Parameter bounds are maintained

    Returns:
        bool: True if all checks pass
    """
```

**Inline Comments:**
- Multi-line docstrings use `"""` triple quotes
- Short comments use `#` single quotes
- Algorithm rationale documented in module-level comments

## Function Design

**Size:**
- Functions are generally < 50 lines
- Complex logic extracted to helper functions
- Test functions each cover one specific aspect

**Parameters:**
- Type hints for all parameters and return values
- Descriptive parameter names
- Optional parameters with default values

**Return Values:**
- Functions returning multiple values use tuple: `return (gain_db, freq, q)`
- Dict returns for complex output: `return {"params": (..., ...), "H_mag": ...}`
- Test functions typically return nothing (print success/failure)

**Examples:**
```python
# Single return
def compute_biquad_coeffs(self, gain_db, freq, q):
    """Computes biquad coefficients for Peaking/Bell EQ filters."""
    return b0, b1, b2, a1, a2

# Dict return
def forward(self, wet_audio):
    """
    Args:
        wet_audio: (Batch, Time) raw audio waveform

    Returns dict with:
        params: tuple of (gain_db, freq, q)
        H_mag: forward magnitude response
    """
    return {
        "params": (gain_db, freq, q),
        "H_mag": H_mag,
        "dry_mag_est": dry_mag_est,
    }
```

## Module Design

**Exports:**
- Classes exported from modules: `DifferentiableBiquadCascade`, `StreamingTCNModel`
- Functions exported when useful: `ste_clamp`, `load_config`
- Constants at module level: `FILTER_NAMES`, `FILTER_PEAKING`

**Module Structure:**
1. Docstring at top explaining purpose
2. Imports (standard library, third-party, local)
3. Constants
4. Helper functions
5. Main classes
6. CLI entry points in `if __name__ == "__main__":` blocks

**Example:**
```python
# Module docstring
"""
Differentiable biquad filter cascade with multi-type support.

Computes biquad coefficients from (gain, freq, Q, filter_type) predictions
and evaluates frequency response entirely in PyTorch for gradient flow.
"""

# Imports
import torch
import torch.nn as nn

# Constants
FILTER_PEAKING = 0
FILTER_LOWSHELF = 1
# ...

# Helper functions
def ste_clamp(x, min_val, max_val):
    return StraightThroughClamp.apply(x, min_val, max_val)

# Main classes
class DifferentiableBiquadCascade(nn.Module):
    # ...

if __name__ == "__main__":
    test_eq_gradients()
```

## Configuration

**YAML Configuration:**
- All configs in `conf/` directory
- Top-level sections: `data`, `model`, `loss`, `trainer`, `curriculum`
- Hierarchical structure with nested dictionaries
- Comments explain each parameter's purpose

**Config Files:**
- `conf/config.yaml` — Main training configuration
- `conf/config_simple.yaml` — Simplified peaking-only config
- `conf/config_spectral.yaml` — Spectral model config
- `conf/config_musdb_200k.yaml` — MUSDB18 200k dataset config
- `conf/deepspeed_config.json` — DeepSpeed optimization settings
- `fixes/recommended_config.yaml` — Recommended config for gain fixes

**Config Loading:**
```python
def load_config(path="conf/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

self.cfg = load_config(config_path)
```

## Type Hints

**Use:**
- Type hints for all function signatures
- Optional types for optional parameters
- Tuple unpacking for multiple returns

**Examples:**
```python
def test_eq_gradients():
    """Test gradient flow through EQ layer."""
    batch_size = 4
    embedding_dim = 128
    n_fft = 1024
    head = EQParameterHead(embedding_dim, num_bands)
    # ...

def check_tensor(name, tensor, expected_shape=None, finite=True) -> None:
    """Assert tensor properties."""
    assert tensor is not None, f"{name}: tensor is None"
    if expected_shape is not None:
        assert tensor.shape == expected_shape
    # ...

def load_config(path="conf/config.yaml") -> dict:
    """Load YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)
```

## Documentation

**Docstrings:**
- Module docstrings describe purpose, key components, usage
- Class docstrings describe purpose, main methods, parameters
- Function docstrings describe purpose, parameters, returns, side effects
- Use reST-style format with Args:, Returns:, Raises: sections

**Examples:**
```python
"""
Differentiable biquad filter cascade with multi-type support.

Computes biquad coefficients from (gain, freq, Q, filter_type) predictions
and evaluates frequency response entirely in PyTorch for gradient flow.

Supported filter types:
  - peaking (biquad bell)
  - lowshelf
  - highshelf
  - highpass (Butterworth)
  - lowpass (Butterworth)

Example:
    >>> cascade = DifferentiableBiquadCascade(num_bands=5, sample_rate=44100)
    >>> gain = torch.randn(4, 5)
    >>> freq = torch.sigmoid(torch.randn(4, 5)) * 20000 + 20
    >>> q = torch.sigmoid(torch.randn(4, 5)) * 9.9 + 0.1
    >>> H_mag = cascade(gain, freq, q, n_fft=2048)
"""

class DifferentiableBiquadCascade(nn.Module):
    """
    A differentiable biquad filter bank supporting multiple filter types.

    Translates [Gain, Freq, Q, FilterType] predictions into biquad coefficients
    and computes the frequency response, enabling exact gradient flow in PyTorch.

    Coefficient formulas from the Robert Bristow-Johnson Audio EQ Cookbook.
    """

    def compute_biquad_coeffs(self, gain_db, freq, q):
        """
        Computes biquad coefficients for Peaking/Bell EQ filters.

        Inputs:
            gain_db: (Batch, Num_Bands) — gain in dB
            freq: (Batch, Num_Bands) — center frequency in Hz
            q: (Batch, Num_Bands) — quality factor

        Returns:
            b0, b1, b2, a1, a2: (Batch, Num_Bands) — biquad coefficients
        """
```

---

*Convention analysis: 2026-04-05*
