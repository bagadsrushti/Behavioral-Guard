"""
train_mouse.py — Train the LSTM-Autoencoder on genuine mouse data (one-class).

Reads windowed .npy features produced by mouse_features.py, trains with MSE
reconstruction loss, and saves a checkpoint.
"""

import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import pickle

from src.models.lstm_ae import (
    LSTMAutoencoder,
    reconstruction_mse,
    save_checkpoint,
)
from src.utils.config_loader import load_config

log = logging.getLogger(__name__)


def _build_dataloader(
    arrays: list[np.ndarray], batch_size: int, shuffle: bool = True
) -> DataLoader:
    """Stack arrays into a single tensor and wrap in a DataLoader."""
    combined = np.concatenate(arrays, axis=0)
    tensor = torch.tensor(combined, dtype=torch.float32)
    return DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=shuffle)


def train_mouse_model(cfg: dict | None = None) -> dict:
    """Full training loop for the mouse LSTM-AE.

    Returns:
        A dict with: model, threshold, train_losses, val_losses
    """
    if cfg is None:
        cfg = load_config()

    mcfg = cfg["model_mouse"]
    eval_cfg = cfg["evaluation"]
    processed_dir = cfg["paths"]["processed_data_dir"]
    ckpt_dir = cfg["paths"]["model_checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Training mouse LSTM-AE on device=%s", device)

    # --- Load manifest & collect genuine sessions --------------------------
    import pandas as pd

    manifest_path = os.path.join(processed_dir, "mouse_manifest.csv")
    manifest = pd.read_csv(manifest_path)
    genuine = manifest[manifest["label"] == "genuine"].reset_index(drop=True)

    if genuine.empty:
        raise RuntimeError("No genuine mouse sessions found in manifest.")

    # Time-ordered split (sessions are assumed chronologically ordered)
    n = len(genuine)
    n_train = max(1, int(n * eval_cfg["train_ratio"]))
    n_val = max(1, int(n * eval_cfg["val_ratio"]))

    train_sessions = genuine.iloc[:n_train]
    val_sessions = genuine.iloc[n_train : n_train + n_val]

    train_arrays = [np.load(p) for p in train_sessions["npy_path"]]
    val_arrays = [np.load(p) for p in val_sessions["npy_path"]]

    if not train_arrays or all(a.shape[0] == 0 for a in train_arrays):
        raise RuntimeError("Training set is empty after loading .npy files.")

    train_combined = np.concatenate(train_arrays, axis=0) if train_arrays else np.empty((0, mcfg["seq_length"], mcfg["input_dim"]))
    
    # --- Fit and Apply Scaler ----------------------------------------------
    scaler = StandardScaler()
    shape = train_combined.shape
    if shape[0] > 0:
        # Reshape to 2D for scaling: (N * seq_length, input_dim)
        train_flat = train_combined.reshape(-1, shape[-1])
        train_flat = scaler.fit_transform(train_flat)
        train_combined = train_flat.reshape(shape)
        
    scaler_path = os.path.join(ckpt_dir, "mouse_scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    log.info("Fitted and saved mouse scaler to %s", scaler_path)
    
    train_loader = _build_dataloader([train_combined], mcfg["batch_size"], shuffle=True)
    
    val_combined = np.concatenate(val_arrays, axis=0) if val_arrays else np.empty((0,))
    if val_combined.shape[0] > 0:
        val_shape = val_combined.shape
        val_flat = val_combined.reshape(-1, val_shape[-1])
        val_flat = scaler.transform(val_flat)
        val_combined = val_flat.reshape(val_shape)
        
    val_tensor = torch.tensor(val_combined, dtype=torch.float32).to(device)

    # --- Build model -------------------------------------------------------
    model = LSTMAutoencoder(
        input_dim=mcfg["input_dim"],
        hidden_dim=mcfg["hidden_dim"],
        embedding_dim=mcfg["embedding_dim"],
        seq_length=mcfg["seq_length"],
        num_layers=mcfg["num_lstm_layers"],
        dropout=mcfg["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=mcfg["learning_rate"])
    criterion = torch.nn.MSELoss()

    # --- Training loop -----------------------------------------------------
    train_losses, val_losses = [], []

    for epoch in range(1, mcfg["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for (batch,) in train_loader:
            batch = batch.to(device)
            x_hat, _ = model(batch)
            loss = criterion(x_hat, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_train = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_train)

        # Validation loss
        model.eval()
        if val_tensor.numel() > 0:
            with torch.no_grad():
                v_hat, _ = model(val_tensor)
                v_loss = criterion(v_hat, val_tensor).item()
        else:
            v_loss = float("nan")
        val_losses.append(v_loss)

        if epoch % 10 == 0 or epoch == 1:
            log.info(
                "Mouse AE  epoch %3d/%d — train_loss=%.6f  val_loss=%.6f",
                epoch,
                mcfg["epochs"],
                avg_train,
                v_loss,
            )

    # --- Threshold from validation genuine errors --------------------------
    if val_tensor.numel() > 0:
        errors = reconstruction_mse(model, val_tensor, reduction="per_sample")
        threshold = float(
            np.percentile(errors.cpu().numpy(), mcfg["threshold_percentile"])
        )
    else:
        threshold = float("inf")
        log.warning("No validation data — threshold set to inf.")

    log.info("Mouse threshold (p%d) = %.6f", mcfg["threshold_percentile"], threshold)

    # --- Save checkpoint ---------------------------------------------------
    ckpt_path = os.path.join(ckpt_dir, mcfg["checkpoint_file"])
    save_checkpoint(
        model,
        ckpt_path,
        extra={"threshold": threshold, "train_losses": train_losses},
    )

    return {
        "model": model,
        "threshold": threshold,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "device": device,
    }
