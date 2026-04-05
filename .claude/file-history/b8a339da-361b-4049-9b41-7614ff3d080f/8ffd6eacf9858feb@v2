"""
Encoding strategies: hashing, entity embeddings, frequency encoding.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction import FeatureHasher
from sklearn.base import BaseEstimator, TransformerMixin


class HashingEncoder(BaseEstimator, TransformerMixin):
    """Encode categoricals using the hashing trick for fixed memory footprint."""

    def __init__(self, n_features=2 ** 18, input_type="string"):
        self.n_features = n_features
        self.input_type = input_type

    def fit(self, X, y=None):
        return self

    def transform(self, X, cols=None):
        if isinstance(X, pd.DataFrame):
            if cols is None:
                cols = X.select_dtypes(include=["object", "category"]).columns
            X_list = X[cols].to_dict("records")
        else:
            X_list = X
        hasher = FeatureHasher(
            n_features=self.n_features, input_type=self.input_type, alternate_sign=True
        )
        return hasher.transform(X_list).toarray()


class EntityEmbedding(nn.Module):
    """Learn continuous embeddings for categorical features."""

    def __init__(self, cardinalities, embedding_dims=None, interaction_layers=None):
        super().__init__()
        if embedding_dims is None:
            embedding_dims = [min(50, (c // 2) + 1) for c in cardinalities]
        if interaction_layers is None:
            interaction_layers = [64, 32]
        self.embeddings = nn.ModuleList(
            [nn.Embedding(card, dim) for card, dim in zip(cardinalities, embedding_dims)]
        )
        total_embed_dim = sum(embedding_dims)
        layers = []
        input_dim = total_embed_dim
        for hidden_dim in interaction_layers:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.1),
            ])
            input_dim = hidden_dim
        self.interaction = nn.Sequential(*layers)
        self.output_dim = input_dim

    def forward(self, categorical_inputs):
        embedded = [emb(cat) for emb, cat in zip(self.embeddings, categorical_inputs)]
        combined = torch.cat(embedded, dim=1)
        return self.interaction(combined)


def frequency_encoding(train_df, test_df, cols):
    """Encode categorical variables by their frequency. Returns (train_encoded, test_encoded)."""
    result_train = train_df.copy()
    result_test = test_df.copy()
    for col in cols:
        freq_map = train_df[col].value_counts(normalize=True).to_dict()
        result_train[f"{col}_freq"] = train_df[col].map(freq_map).fillna(0)
        result_test[f"{col}_freq"] = test_df[col].map(freq_map).fillna(0)
        result_train[f"{col}_missing"] = train_df[col].isna().astype(int)
        result_test[f"{col}_missing"] = test_df[col].isna().astype(int)
    return result_train, result_test
