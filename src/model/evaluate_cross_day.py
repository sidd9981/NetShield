"""
NetShield -- Cross-Day Evaluation

Tests the trained model on a different day's data to check
generalization to unseen attack types.

Usage:
    python -m src.model.evaluate_cross_day
"""

import logging
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)

from src.model.model import TransformerAutoencoder
from src.data.preprocessor import (
    load_raw, drop_corrupt_rows, handle_inf_and_nan,
    drop_useless_features, get_feature_columns, clip_outliers,
    log_transform_skewed, PreprocessingStats,
)
import joblib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# --- Paths ---
RAW_DATA_DIR = Path("data/raw")
ARTIFACTS_DIR = Path("artifacts")
EVAL_FILE = RAW_DATA_DIR / "Friday-16-02-2018_TrafficForML_CICFlowMeter.csv"


def preprocess_new_day(csv_path: Path) -> tuple:
    """Apply the same preprocessing to a new day's CSV.

    Uses the SAME scaler and feature set from training -- critical for
    valid evaluation. We cannot fit a new scaler on test data.
    """
    # Load feature metadata
    with open(ARTIFACTS_DIR / "feature_meta.json") as f:
        meta = json.load(f)
    feature_names = meta["feature_names"]
    log_transformed = meta["log_transformed"]

    # Load the scaler fitted on training benign data
    scaler = joblib.load(ARTIFACTS_DIR / "scaler.joblib")

    # Load and clean
    df = load_raw(csv_path)
    stats = PreprocessingStats()

    if "Timestamp" in df.columns:
        df = df.drop(columns=["Timestamp"])

    # Force numeric conversion -- some days have string-typed numeric columns
    label_col = None
    for col in df.columns:
        if col.lower() == "label":
            label_col = col
            break

    for col in df.columns:
        if col != label_col and df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = drop_corrupt_rows(df, stats)
    df = handle_inf_and_nan(df, stats)
    df = drop_useless_features(df, stats)

    # Check that all expected features exist
    available = set(df.columns)
    missing = [f for f in feature_names if f not in available]
    if missing:
        log.warning("Missing features in new data: %s", missing)
        # Add missing columns as zeros
        for col in missing:
            df[col] = 0.0

    # Clip outliers using same approach
    numeric_features = [c for c in feature_names if c in df.columns]
    df = clip_outliers(df, numeric_features)

    # Log transform the same features
    for col in log_transformed:
        if col in df.columns and (df[col] >= 0).all():
            df[col] = np.log1p(df[col])

    # Extract labels before selecting features
    label_col = "Label"
    for col in df.columns:
        if col.lower() == "label":
            label_col = col
            break

    labels_original = df[label_col].values
    y = (df[label_col] != "Benign").astype(int).values

    # Select and order features to match training
    X = df[feature_names].values

    # Apply the TRAINING scaler
    X = scaler.transform(X)

    return X, y, labels_original


def evaluate_model(X, y, labels_original, threshold):
    """Run the model and compute metrics per attack type."""
    # Load model
    with open(ARTIFACTS_DIR / "feature_meta.json") as f:
        meta = json.load(f)

    model = TransformerAutoencoder(n_features=meta["n_features"])
    model.load_state_dict(torch.load(ARTIFACTS_DIR / "best_model.pt", weights_only=True))
    model.eval()

    # Score all samples in batches
    X_tensor = torch.tensor(X, dtype=torch.float32)
    all_scores = []

    batch_size = 512
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i + batch_size]
            scores = model.anomaly_score(batch)
            all_scores.append(scores)

    scores = torch.cat(all_scores).numpy()
    predictions = (scores > threshold).astype(int)

    # Overall metrics
    log.info("\n")
    log.info("OVERALL METRICS (threshold=%.6f)", threshold)
    log.info("\n")

    auc = roc_auc_score(y, scores)
    log.info("AUC-ROC:   %.4f", auc)
    log.info("Precision: %.4f", precision_score(y, predictions, zero_division=0))
    log.info("Recall:    %.4f", recall_score(y, predictions, zero_division=0))
    log.info("F1:        %.4f", f1_score(y, predictions, zero_division=0))

    cm = confusion_matrix(y, predictions)
    tn, fp, fn, tp = cm.ravel()
    log.info("Confusion: TN=%d FP=%d FN=%d TP=%d", tn, fp, fn, tp)
    log.info("FPR:       %.4f", fp / (fp + tn) if (fp + tn) > 0 else 0)

    # Per-attack-type breakdown
    log.info("")
    log.info("\n")
    log.info("PER ATTACK TYPE BREAKDOWN")
    log.info("\n")

    unique_labels = np.unique(labels_original)
    log.info("")
    log.info("%-25s %8s %8s %10s %10s %10s",
             "Label", "Count", "Detect", "Recall", "Mean Score", "Median Score")
    log.info("-" * 80)

    for label in sorted(unique_labels):
        mask = labels_original == label
        label_scores = scores[mask]
        label_preds = predictions[mask]
        count = mask.sum()
        detected = label_preds.sum()
        recall = detected / count if count > 0 else 0

        log.info("%-25s %8d %8d %10.4f %10.4f %10.4f",
                 label, count, detected, recall,
                 label_scores.mean(), np.median(label_scores))

    # Score distribution comparison
    log.info("")
    log.info("SCORE DISTRIBUTIONS")
    log.info("\n")

    benign_scores = scores[y == 0]
    attack_scores = scores[y == 1]

    log.info("Benign:  mean=%.6f  std=%.6f  median=%.6f  p95=%.6f",
             benign_scores.mean(), benign_scores.std(),
             np.median(benign_scores), np.percentile(benign_scores, 95))
    log.info("Attack:  mean=%.6f  std=%.6f  median=%.6f  p5=%.6f",
             attack_scores.mean(), attack_scores.std(),
             np.median(attack_scores), np.percentile(attack_scores, 5))

    return {
        "auc_roc": auc,
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "fpr": float(fp / (fp + tn) if (fp + tn) > 0 else 0),
    }


def main():
    if not EVAL_FILE.exists():
        log.error("File not found: %s", EVAL_FILE)
        log.error("Download it with:")
        log.error("  aws s3 cp --no-sign-request \\")
        log.error("    's3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/"
                   "Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv' \\")
        log.error("    data/raw/")
        return

    # Load threshold from training
    with open(ARTIFACTS_DIR / "threshold.json") as f:
        threshold = json.load(f)["threshold"]
    log.info("Using threshold: %.6f", threshold)

    # Preprocess
    log.info("Preprocessing %s ...", EVAL_FILE.name)
    X, y, labels_original = preprocess_new_day(EVAL_FILE)
    log.info("Preprocessed: %d samples, %d features", X.shape[0], X.shape[1])
    log.info("Labels: %s", dict(zip(*np.unique(labels_original, return_counts=True))))

    # Evaluate
    metrics = evaluate_model(X, y, labels_original, threshold)

    # Save
    with open(ARTIFACTS_DIR / "cross_day_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Metrics saved to %s/cross_day_metrics.json", ARTIFACTS_DIR)


if __name__ == "__main__":
    main()