import torch
from differentiable_eq import DifferentiableBiquadCascade, EQParameterHead


def test_eq_gradients():
    print("Testing Differentiable EQ Layer...")

    # 1. Setup
    batch_size = 4
    embedding_dim = 128
    num_bands = 5
    n_fft = 1024

    head = EQParameterHead(embedding_dim, num_bands)
    cascade = DifferentiableBiquadCascade(num_bands)

    # 2. Dummy Input (representing CNN features)
    dummy_embedding = torch.randn(batch_size, embedding_dim, requires_grad=True)

    # 3. Forward Pass
    gain_db, freq, q = head(dummy_embedding)

    # Ensure bounds are working
    assert torch.all(gain_db >= -24.0) and torch.all(gain_db <= 24.0), (
        "Gain out of bounds"
    )
    assert torch.all(freq >= 20.0) and torch.all(freq <= 20000.0), "Freq out of bounds"
    assert torch.all(q >= 0.1) and torch.all(q <= 10.0), "Q out of bounds"

    print(
        f"Forward Pass Config - Gain: {gain_db.shape}, Freq: {freq.shape}, Q: {q.shape}"
    )

    # 4. Filter generation
    H_mag = cascade(gain_db, freq, q, n_fft=n_fft)

    print(f"Frequency Response Shape: {H_mag.shape}")
    assert H_mag.shape == (batch_size, n_fft // 2 + 1)

    # 5. Dummy Loss and Backward Pass
    # Let's say we want the response to be perfectly flat (all 1s)
    target_H = torch.ones_like(H_mag)
    loss = torch.nn.functional.mse_loss(H_mag, target_H)

    print(f"Initial Loss: {loss.item():.4f}")

    loss.backward()

    # 6. Check Gradients
    assert dummy_embedding.grad is not None, "Gradients did not flow back to embedding!"
    print("Gradients successfully propagated through the biquad cascade!")
    print("Test Passed")


if __name__ == "__main__":
    test_eq_gradients()
