"""
Dimensionality reduction: PCA, incremental PCA, UMAP, autoencoders, denoising autoencoders.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import StandardScaler


from sklearn.random_projection import SparseRandomProjection


def apply_pca(X, variance_threshold=0.95):
    """Apply PCA with automatic component selection based on variance explained.

    Returns dict with X_reduced, pca, scaler, n_components, explained_variance.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    n_components = int(np.argmax(cumsum >= variance_threshold)) + 1
    pca_optimal = PCA(n_components=n_components)
    X_reduced = pca_optimal.fit_transform(X_scaled)
    return {
        "X_reduced": X_reduced,
        "pca": pca_optimal,
        "scaler": scaler,
        "n_components": n_components,
        "explained_variance": pca_optimal.explained_variance_ratio_.sum(),
    }


def batch_pca(X, batch_size=1000, n_components=50):
    """Apply IncrementalPCA for datasets that don't fit in memory."""
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    X_reduced = ipca.fit_transform(X)
    return X_reduced, ipca


class Autoencoder(nn.Module):
    """Basic autoencoder for dimensionality reduction."""

    def __init__(self, input_dim, encoding_dims):
        super().__init__()
        encoder_layers = []
        prev_dim = input_dim
        for dim in encoding_dims:
            encoder_layers.extend([nn.Linear(prev_dim, dim), nn.ReLU(), nn.BatchNorm1d(dim)])
            prev_dim = dim
        decoder_dims = list(reversed(encoding_dims[:-1])) + [input_dim]
        decoder_layers = []
        prev_dim = encoding_dims[-1]
        for dim in decoder_dims:
            decoder_layers.extend([nn.Linear(prev_dim, dim), nn.ReLU() if dim != input_dim else nn.Sigmoid()])
            prev_dim = dim
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def encode(self, x):
        return self.encoder(x)


class DenoisingAutoencoder(nn.Module):
    """Denoising autoencoder for robust representations."""

    def __init__(self, input_dim, hidden_dims, noise_factor=0.2):
        super().__init__()
        self.noise_factor = noise_factor
        encoder = []
        prev_dim = input_dim
        for dim in hidden_dims:
            encoder.extend([nn.Linear(prev_dim, dim), nn.ReLU(), nn.Dropout(0.2)])
            prev_dim = dim
        decoder = []
        prev_dim = hidden_dims[-1]
        for dim in reversed(hidden_dims[:-1]):
            decoder.extend([nn.Linear(prev_dim, dim), nn.ReLU()])
            prev_dim = dim
        decoder.extend([nn.Linear(prev_dim, input_dim), nn.Sigmoid()])
        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        noisy = x + self.noise_factor * torch.randn_like(x) if self.training else x
        noisy = torch.clamp(noisy, 0, 1)
        encoded = self.encoder(noisy)
        decoded = self.decoder(encoded)
        return encoded, decoded


def train_autoencoder(X_train, X_val, input_dim, encoding_dims, epochs=100, batch_size=256, lr=0.001):
    """Train an autoencoder and return the model with loss histories."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder(input_dim, encoding_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    X_train_t = torch.FloatTensor(np.array(X_train)).to(device)
    X_val_t = torch.FloatTensor(np.array(X_val)).to(device)
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        _, decoded = model(X_train_t)
        loss = criterion(decoded, X_train_t)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            _, val_decoded = model(X_val_t)
            val_loss = criterion(val_decoded, X_val_t)
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
    return model, train_losses, val_losses
