"""
NetShield -- Training Script with MLflow

Trains the Transformer autoencoder and logs everything to MLflow:
  - Hyperparameters
  - Per-epoch metrics
  - Final test metrics + per-attack breakdown
  - Model artifacts (weights, scaler, threshold, feature meta)

Usage:
    # Start MLflow UI (separate terminal)
    mlflow ui --port 5000

    # Train
    python -m src.model.train
"""

import logging
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, confusion_matrix,
)
import mlflow
import mlflow.pytorch

from src.model.model import TransformerAutoencoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SPLITS_DIR = Path("data/splits")
ARTIFACTS_DIR = Path("artifacts")
FEATURE_META = ARTIFACTS_DIR / "feature_meta.json"


class TrainConfig:
    n_features: int = 72
    d_model: int = 64
    n_heads: int = 4
    n_encoder_layers: int = 2
    n_decoder_layers: int = 2
    d_ff: int = 128
    dropout: float = 0.1
    bottleneck_dim: int = 32

    epochs: int = 50
    batch_size: int = 512
    lr: float = 3e-4
    weight_decay: float = 1e-5
    patience: int = 15
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 6
    max_train_samples: int = 500_000
    threshold_percentile: float = 85.0

    device: str = "mps"


def load_splits():
    train = np.load(SPLITS_DIR / "train.npz", allow_pickle=True)
    val = np.load(SPLITS_DIR / "val.npz", allow_pickle=True)
    test = np.load(SPLITS_DIR / "test.npz", allow_pickle=True)
    return train, val, test


def make_benign_loader(split, batch_size, shuffle=True, max_samples=None):
    X, y = split["X"], split["y"]
    benign_mask = y == 0
    X_benign = X[benign_mask]

    if max_samples and len(X_benign) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_benign), size=max_samples, replace=False)
        X_benign = X_benign[idx]

    X_t = torch.tensor(X_benign, dtype=torch.float32)
    dataset = TensorDataset(X_t, X_t)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)


def make_full_loader(split, batch_size):
    X = torch.tensor(split["X"], dtype=torch.float32)
    y = torch.tensor(split["y"], dtype=torch.float32)
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n_batches = len(loader)

    for i, (X_batch, _) in enumerate(loader):
        X_batch = X_batch.to(device)
        optimizer.zero_grad()
        X_hat = model(X_batch)
        loss = criterion(X_hat, X_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

        if (i + 1) % 50 == 0 or (i + 1) == n_batches:
            log.info("  batch %d/%d  loss=%.6f", i + 1, n_batches, loss.item())

    return total_loss / n_batches


def evaluate(model, loader, device):
    model.eval()
    all_scores, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            scores = model.anomaly_score(X_batch)
            all_scores.append(scores.cpu())
            all_labels.append(y_batch)

    scores = torch.cat(all_scores).numpy()
    labels = torch.cat(all_labels).numpy()

    benign_scores = scores[labels == 0]
    attack_scores = scores[labels == 1]

    metrics = {
        "val_loss": float(scores.mean()),
        "benign_score_mean": float(benign_scores.mean()),
        "benign_score_std": float(benign_scores.std()),
        "attack_score_mean": float(attack_scores.mean()),
        "attack_score_std": float(attack_scores.std()),
        "score_separation": float(attack_scores.mean() - benign_scores.mean()),
    }

    try:
        metrics["auc_roc"] = float(roc_auc_score(labels, scores))
    except ValueError:
        metrics["auc_roc"] = 0.0

    return metrics


def find_threshold(model, val_loader, device, percentile=85):
    model.eval()
    benign_scores = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            scores = model.anomaly_score(X_batch)
            mask = y_batch == 0
            if mask.any():
                benign_scores.append(scores.cpu()[mask])

    benign_scores = torch.cat(benign_scores).numpy()
    threshold = float(np.percentile(benign_scores, percentile))
    return threshold


def run_training():
    cfg = TrainConfig()

    with open(FEATURE_META) as f:
        meta = json.load(f)
    cfg.n_features = meta["n_features"]
    log.info("Features: %d, Device: %s", cfg.n_features, cfg.device)

    #  MLflow setup 
    mlflow.set_experiment("netshield-intrusion-detection")

    with mlflow.start_run(run_name=f"transformer-ae-{time.strftime('%Y%m%d-%H%M%S')}"):
        # Log hyperparameters
        mlflow.log_params({
            "model": "TransformerAutoencoder",
            "n_features": cfg.n_features,
            "d_model": cfg.d_model,
            "n_heads": cfg.n_heads,
            "n_encoder_layers": cfg.n_encoder_layers,
            "n_decoder_layers": cfg.n_decoder_layers,
            "d_ff": cfg.d_ff,
            "dropout": cfg.dropout,
            "bottleneck_dim": cfg.bottleneck_dim,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "patience": cfg.patience,
            "max_train_samples": cfg.max_train_samples,
            "threshold_percentile": cfg.threshold_percentile,
            "device": cfg.device,
            "scoring": "blended_70mean_30max",
        })

        #  Load data 
        log.info("Loading data...")
        train_split, val_split, test_split = load_splits()

        train_loader = make_benign_loader(
            train_split, cfg.batch_size, max_samples=cfg.max_train_samples,
        )
        val_loader = make_full_loader(val_split, cfg.batch_size)
        test_loader = make_full_loader(test_split, cfg.batch_size)

        n_benign = min(cfg.max_train_samples, int((train_split["y"] == 0).sum()))
        log.info("Train batches: %d (benign only, %d samples)", len(train_loader), n_benign)
        mlflow.log_param("n_train_benign", n_benign)

        #  Model 
        model = TransformerAutoencoder(
            n_features=cfg.n_features,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_encoder_layers=cfg.n_encoder_layers,
            n_decoder_layers=cfg.n_decoder_layers,
            d_ff=cfg.d_ff,
            dropout=cfg.dropout,
            bottleneck_dim=cfg.bottleneck_dim,
        ).to(cfg.device)

        param_count = sum(p.numel() for p in model.parameters())
        log.info("Parameters: %d (%.2f KB)", param_count, param_count * 4 / 1024)
        mlflow.log_param("param_count", param_count)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=cfg.lr_scheduler_factor,
            patience=cfg.lr_scheduler_patience,
        )

        best_val_auc = 0.0
        patience_counter = 0

        #  Training loop 
        for epoch in range(1, cfg.epochs + 1):
            t0 = time.time()

            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, cfg.device,
            )
            val_metrics = evaluate(model, val_loader, cfg.device)
            val_loss = val_metrics["val_loss"]

            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0

            # Log epoch metrics to MLflow
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "benign_score_mean": val_metrics["benign_score_mean"],
                "attack_score_mean": val_metrics["attack_score_mean"],
                "score_separation": val_metrics["score_separation"],
                "val_auc_roc": val_metrics["auc_roc"],
                "learning_rate": current_lr,
            }, step=epoch)

            log.info(
                "Epoch %02d/%02d  train=%.6f  benign=%.6f  attack=%.4f  "
                "auc=%.4f  sep=%.4f  lr=%.1e  (%.1fs)",
                epoch, cfg.epochs, train_loss,
                val_metrics["benign_score_mean"], val_metrics["attack_score_mean"],
                val_metrics["auc_roc"], val_metrics["score_separation"],
                current_lr, elapsed,
            )

            val_auc = val_metrics["auc_roc"]
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                torch.save(model.state_dict(), ARTIFACTS_DIR / "best_model.pt")
                log.info("  -> New best (auc=%.4f)", val_auc)
            else:
                patience_counter += 1
                if patience_counter >= cfg.patience:
                    log.info("Early stopping at epoch %d (best auc=%.4f)",
                             epoch, best_val_auc)
                    mlflow.log_metric("stopped_at_epoch", epoch)
                    break

        #  Load best and evaluate 
        model.load_state_dict(
            torch.load(ARTIFACTS_DIR / "best_model.pt", weights_only=True)
        )

        threshold = find_threshold(
            model, val_loader, cfg.device, cfg.threshold_percentile,
        )
        log.info("Anomaly threshold (%.0fth pctl benign): %.6f",
                 cfg.threshold_percentile, threshold)

        test_metrics = evaluate(model, test_loader, cfg.device)
        log.info("Test AUC-ROC: %.4f", test_metrics["auc_roc"])

        # Score test set in batches
        test_scores, test_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(cfg.device)
                scores = model.anomaly_score(X_batch)
                test_scores.append(scores.cpu())
                test_labels.append(y_batch)

        test_scores = torch.cat(test_scores).numpy()
        test_labels = torch.cat(test_labels).numpy()
        predictions = (test_scores > threshold).astype(int)

        precision = precision_score(test_labels, predictions)
        recall = recall_score(test_labels, predictions)
        f1 = f1_score(test_labels, predictions)
        cm = confusion_matrix(test_labels, predictions)
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn)

        log.info("At threshold %.6f:", threshold)
        log.info("  Precision: %.4f  Recall: %.4f  F1: %.4f", precision, recall, f1)
        log.info("  TN=%d FP=%d FN=%d TP=%d  FPR=%.4f", tn, fp, fn, tp, fpr)

        # Log final metrics to MLflow
        mlflow.log_metrics({
            "test_auc_roc": test_metrics["auc_roc"],
            "test_precision": float(precision),
            "test_recall": float(recall),
            "test_f1": float(f1),
            "test_fpr": float(fpr),
            "threshold": threshold,
            "best_val_auc": best_val_auc,
        })

        # Per-attack breakdown
        test_split_labels = test_split["labels"]
        attack_recalls = {}
        for label in sorted(np.unique(test_split_labels)):
            mask = test_split_labels == label
            lbl_preds = predictions[mask]
            lbl_recall = lbl_preds.mean()
            safe_label = label.replace(" ", "_").replace("-", "_")
            attack_recalls[f"recall_{safe_label}"] = float(lbl_recall)
            log.info("  %-30s recall=%.4f", label, lbl_recall)

        mlflow.log_metrics(attack_recalls)

        #  Log artifacts to MLflow 
        with open(ARTIFACTS_DIR / "threshold.json", "w") as f:
            json.dump({
                "threshold": threshold,
                "method": f"{cfg.threshold_percentile}th_percentile_benign",
            }, f, indent=2)

        final_metrics = {
            "test_auc_roc": test_metrics["auc_roc"],
            "test_precision": float(precision),
            "test_recall": float(recall),
            "test_f1": float(f1),
            "test_fpr": float(fpr),
            "threshold": threshold,
            "best_val_auc": best_val_auc,
            "config": {
                "d_model": cfg.d_model, "n_heads": cfg.n_heads,
                "n_layers": cfg.n_encoder_layers, "bottleneck": cfg.bottleneck_dim,
                "max_train_samples": cfg.max_train_samples,
                "scoring": "blended_70mean_30max",
            },
        }
        with open(ARTIFACTS_DIR / "training_metrics.json", "w") as f:
            json.dump(final_metrics, f, indent=2)

        # Log model and artifacts
        mlflow.pytorch.log_model(model, "model")
        mlflow.log_artifact(str(ARTIFACTS_DIR / "best_model.pt"))
        mlflow.log_artifact(str(ARTIFACTS_DIR / "threshold.json"))
        mlflow.log_artifact(str(ARTIFACTS_DIR / "feature_meta.json"))
        mlflow.log_artifact(str(ARTIFACTS_DIR / "scaler.joblib"))
        mlflow.log_artifact(str(ARTIFACTS_DIR / "training_metrics.json"))

        log.info("")
        log.info("MLflow run logged. View at: mlflow ui --port 5000")
        log.info("Done. Artifacts saved to %s/", ARTIFACTS_DIR)


if __name__ == "__main__":
    run_training()