"""
lstm_ae.py — Shared LSTM-Autoencoder architecture for one-class anomaly detection.

Architecture (per Li et al., adapted):
    Encoder: 2 stacked LSTM layers → FC layer → low-dim embedding
    Decoder: FC layer → 2 stacked LSTM layers → FC layer → reconstructed sequence

Both the mouse and keystroke modalities instantiate this same class with
different input_dim / embedding_dim / hidden_dim values read from config.
"""

import logging
from typing import Tuple

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


class LSTMAutoencoder(nn.Module):
    """Sequence-to-sequence LSTM autoencoder for reconstruction-based anomaly
    detection.

    Args:
        input_dim:   Number of features per timestep.
        hidden_dim:  Number of LSTM hidden units.
        embedding_dim: Bottleneck dimensionality.
        seq_length:  Expected sequence length (used by decoder reshape).
        num_layers:  Number of stacked LSTM layers in encoder & decoder.
        dropout:     Dropout probability between LSTM layers.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        seq_length: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.seq_length = seq_length
        self.num_layers = num_layers

        # ---- Encoder ----
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.encoder_fc = nn.Linear(hidden_dim, embedding_dim)

        # ---- Decoder ----
        self.decoder_fc_in = nn.Linear(embedding_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.decoder_fc_out = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input sequence to a low-dim embedding.

        Args:
            x: (batch, seq_length, input_dim)

        Returns:
            z: (batch, embedding_dim)
        """
        # LSTM output: (batch, seq, hidden)
        _, (h_n, _) = self.encoder_lstm(x)
        # h_n shape: (num_layers, batch, hidden) — take last layer
        last_hidden = h_n[-1]  # (batch, hidden)
        z = self.encoder_fc(last_hidden)  # (batch, embedding_dim)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode embedding back to a reconstructed sequence.

        Args:
            z: (batch, embedding_dim)

        Returns:
            x_hat: (batch, seq_length, input_dim)
        """
        # Project back to hidden dim and repeat across timesteps
        h = self.decoder_fc_in(z)  # (batch, hidden)
        h = h.unsqueeze(1).repeat(1, self.seq_length, 1)  # (batch, seq, hidden)

        lstm_out, _ = self.decoder_lstm(h)  # (batch, seq, hidden)
        x_hat = self.decoder_fc_out(lstm_out)  # (batch, seq, input_dim)
        return x_hat

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full forward pass: encode → decode.

        Returns:
            x_hat: Reconstructed sequence.
            z: Latent embedding (useful for inspection / fusion).
        """
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


# ---------------------------------------------------------------------------
# Convenience: reconstruction error
# ---------------------------------------------------------------------------

def reconstruction_mse(
    model: LSTMAutoencoder,
    x: torch.Tensor,
    reduction: str = "per_sample",
) -> torch.Tensor:
    """Compute MSE reconstruction error.

    Args:
        model: Trained LSTMAutoencoder.
        x: Input tensor (batch, seq_length, input_dim).
        reduction:
            'per_sample' → (batch,) mean over seq & features
            'mean'       → scalar mean over entire batch

    Returns:
        Reconstruction error tensor.
    """
    model.eval()
    with torch.no_grad():
        x_hat, _ = model(x)
        mse = (x - x_hat) ** 2

    if reduction == "per_sample":
        return mse.mean(dim=(1, 2))  # (batch,)
    elif reduction == "mean":
        return mse.mean()
    else:
        return mse  # raw (batch, seq, features)


# ---------------------------------------------------------------------------
# Save / Load checkpoints
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: LSTMAutoencoder, path: str, extra: dict | None = None
) -> None:
    """Persist model weights and optional metadata."""
    payload = {
        "state_dict": model.state_dict(),
        "config": {
            "input_dim": model.input_dim,
            "hidden_dim": model.hidden_dim,
            "embedding_dim": model.embedding_dim,
            "seq_length": model.seq_length,
            "num_layers": model.num_layers,
        },
    }
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)
    log.info("Checkpoint saved → %s", path)


def load_checkpoint(path: str, device: str = "cpu") -> Tuple[LSTMAutoencoder, dict]:
    """Load model weights from a checkpoint file.

    Returns:
        model: LSTMAutoencoder with restored weights.
        extra: Any extra metadata dict stored alongside the weights.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    mcfg = ckpt["config"]
    model = LSTMAutoencoder(
        input_dim=mcfg["input_dim"],
        hidden_dim=mcfg["hidden_dim"],
        embedding_dim=mcfg["embedding_dim"],
        seq_length=mcfg["seq_length"],
        num_layers=mcfg["num_layers"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    log.info("Checkpoint loaded ← %s", path)
    return model, ckpt.get("extra", {})
