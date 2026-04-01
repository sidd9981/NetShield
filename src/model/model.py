"""
NetShield -- Transformer Autoencoder for Anomaly Detection

A small Transformer encoder-decoder that learns to reconstruct normal
network traffic. High reconstruction error signals an anomaly.

Each of the input features is treated as a "token" -- projected to
an embedding dimension, then processed through self-attention layers.
This lets the model learn feature interactions (e.g. high SYN count +
short duration + high packet rate = suspicious combination) rather than
treating each feature independently like a dense autoencoder would.

Anomaly scoring uses a blended approach:
  - 70% mean per-feature error (catches attacks spread across many features)
  - 30% max per-feature error (catches attacks concentrated in one feature)
"""

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Learnable positional encoding for tabular features."""
    def __init__(self, d_model: int, n_features: int):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, n_features, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_embedding


class TransformerAutoencoder(nn.Module):
    def __init__(
        self,
        n_features: int = 72,
        d_model: int = 64,
        n_heads: int = 4,
        n_encoder_layers: int = 2,
        n_decoder_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1,
        bottleneck_dim: int = 32,
    ):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model

        # --- Encoder ---
        self.feature_embedding = nn.Linear(1, d_model)
        self.pos_encoding = PositionalEncoding(d_model, n_features)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)

        # --- Bottleneck ---
        self.to_bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_features * d_model, bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.from_bottleneck = nn.Sequential(
            nn.Linear(bottleneck_dim, n_features * d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # --- Decoder ---
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=n_decoder_layers)

        self.output_projection = nn.Linear(d_model, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        x = self.feature_embedding(x)
        x = self.pos_encoding(x)
        x = self.encoder(x)
        x = self.to_bottleneck(x)
        return x

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.from_bottleneck(z)
        x = x.view(-1, self.n_features, self.d_model)
        x = self.decoder(x)
        x = self.output_projection(x).squeeze(-1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Blended anomaly score: 70% mean + 30% max per-feature error.

        Mean catches attacks that are slightly anomalous across many features.
        Max catches attacks that are extremely anomalous in a single feature
        (e.g. DDOS-HOIC with Init Fwd Win Byts at 225x normal error).
        """
        x_hat = self.forward(x)
        per_feature_err = (x - x_hat) ** 2
        mean_err = per_feature_err.mean(dim=1)
        max_err = per_feature_err.max(dim=1).values
        return 0.7 * mean_err + 0.3 * max_err