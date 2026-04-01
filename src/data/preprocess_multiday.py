"""
NetShield -- Multi-Day Preprocessing 

  1. KEEP volatile absolute features (Flow Byts/s, Pkts/s, etc.)
     — these carry the strongest attack signal.
  2. Use QuantileTransformer instead of StandardScaler. This maps
     every feature to a uniform/gaussian distribution, which:
     - Handles extreme values and heavy tails naturally
     - Makes cross-day scale differences irrelevant (rank-based)
     - Preserves the ordering that separates attacks from benign
  3. Remove benign variance filter (no longer needed with quantile transform)
  4. Everything else unchanged.

Usage:
    python -m src.data.preprocess_multiday
"""

import logging
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
import joblib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RAW_DATA_DIR  = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
SPLITS_DIR    = Path("data/splits")
ARTIFACTS_DIR = Path("artifacts")

# Exact duplicates only — keep everything else
REDUNDANT_FEATURES = [
    "Subflow Fwd Pkts",    # == Tot Fwd Pkts
    "Subflow Bwd Pkts",    # == Tot Bwd Pkts
    "Subflow Fwd Byts",    # == TotLen Fwd Pkts
    "Subflow Bwd Byts",    # == TotLen Bwd Pkts
    "Fwd Seg Size Avg",    # == Fwd Pkt Len Mean
    "Bwd Seg Size Avg",    # == Bwd Pkt Len Mean
    "ECE Flag Cnt",        # == RST Flag Cnt
]

# Truly zero variance across all days
ZERO_VAR_FEATURES = [
    "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "CWE Flag Count",
    "Fwd Byts/b Avg", "Fwd Pkts/b Avg", "Fwd Blk Rate Avg",
    "Bwd Byts/b Avg", "Bwd Pkts/b Avg", "Bwd Blk Rate Avg",
]

# NOT dropping volatile features anymore — they carry attack signal

NON_NEGATIVE_FEATURES = [
    "Flow Duration",
    "Fwd Pkt Len Max", "Fwd Pkt Len Min", "Fwd Pkt Len Mean",
    "Bwd Pkt Len Max", "Bwd Pkt Len Min", "Bwd Pkt Len Mean",
]

HOLDOUT_DAY = "Friday-16-02-2018"

TRAINING_DAYS = [
    "Wednesday-14-02-2018",
    "Thursday-15-02-2018",
    "Thuesday-20-02-2018",
    "Wednesday-21-02-2018",
    "Thursday-22-02-2018",
    "Wednesday-28-02-2018",
    "Thursday-01-03-2018",
    "Friday-02-03-2018",
    "Friday-23-02-2018",
]

CLAMP_RANGE = 5.0  # tighter clamp for quantile-transformed data


#  Loading & cleaning 

def load_and_clean(csv_path: Path) -> pd.DataFrame:
    log.info("Loading %s ...", csv_path.name)
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = df.columns.str.strip()
    log.info("  Raw: %d rows x %d cols", len(df), len(df.columns))

    if "Timestamp" in df.columns:
        df = df.drop(columns=["Timestamp"])

    label_col = next((c for c in df.columns if c.lower() == "label"), None)
    for col in df.columns:
        if col != label_col and df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop physically impossible rows
    mask = pd.Series(True, index=df.index)
    for col in NON_NEGATIVE_FEATURES:
        if col in df.columns:
            mask &= df[col] >= 0
    n_corrupt = (~mask).sum()
    df = df[mask].copy()
    if n_corrupt > 0:
        log.info("  Dropped %d corrupt rows", n_corrupt)

    # Inf -> NaN, rate columns -> 0
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    rate_cols = [c for c in df.columns if any(kw in c for kw in
                 ["Byts/s", "Pkts/s", "Bytes/s", "Packets/s"])]
    for col in rate_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    n_before = len(df)
    df = df.dropna()
    dropped = n_before - len(df)
    if dropped > 0:
        log.info("  Dropped %d NaN rows", dropped)

    # Only drop true duplicates and zero-variance — keep volatile features
    drop_cols = [c for c in ZERO_VAR_FEATURES + REDUNDANT_FEATURES if c in df.columns]
    df = df.drop(columns=drop_cols)

    df["_source_file"] = csv_path.name
    log.info("  Cleaned: %d rows x %d cols", len(df), len(df.columns))
    return df


#  Ratio features 

def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-8

    tot_fwd_pkts = df.get("Tot Fwd Pkts",    pd.Series(0, index=df.index))
    tot_bwd_pkts = df.get("Tot Bwd Pkts",    pd.Series(0, index=df.index))
    totlen_fwd   = df.get("TotLen Fwd Pkts", pd.Series(0, index=df.index))
    totlen_bwd   = df.get("TotLen Bwd Pkts", pd.Series(0, index=df.index))
    total_pkts   = tot_fwd_pkts + tot_bwd_pkts + eps
    total_bytes  = totlen_fwd + totlen_bwd

    df["bytes_per_pkt"]      = total_bytes / total_pkts
    df["fwd_bwd_pkt_ratio"]  = (tot_fwd_pkts + eps) / (tot_bwd_pkts + eps)
    df["fwd_bwd_byte_ratio"] = (totlen_fwd   + eps) / (totlen_bwd   + eps)

    fwd_max  = df.get("Fwd Pkt Len Max",  pd.Series(0, index=df.index))
    fwd_min  = df.get("Fwd Pkt Len Min",  pd.Series(0, index=df.index))
    fwd_mean = df.get("Fwd Pkt Len Mean", pd.Series(0, index=df.index))
    df["fwd_pkt_size_range"] = fwd_max - fwd_min
    df["fwd_pkt_cv"] = (
        df.get("Fwd Pkt Len Std", pd.Series(0, index=df.index)) / (fwd_mean + eps)
    )

    iat_mean = df.get("Flow IAT Mean", pd.Series(0, index=df.index)) + eps
    df["iat_cv"] = df.get("Flow IAT Std", pd.Series(0, index=df.index)) / iat_mean

    df["syn_per_pkt"] = df.get("SYN Flag Cnt", pd.Series(0, index=df.index)) / total_pkts
    df["fin_per_pkt"] = df.get("FIN Flag Cnt", pd.Series(0, index=df.index)) / total_pkts
    df["rst_per_pkt"] = df.get("RST Flag Cnt", pd.Series(0, index=df.index)) / total_pkts
    df["psh_per_pkt"] = df.get("PSH Flag Cnt", pd.Series(0, index=df.index)) / total_pkts
    df["ack_per_pkt"] = df.get("ACK Flag Cnt", pd.Series(0, index=df.index)) / total_pkts

    active_mean = df.get("Active Mean", pd.Series(0, index=df.index))
    idle_mean   = df.get("Idle Mean",   pd.Series(0, index=df.index))
    df["active_idle_ratio"] = (active_mean + eps) / (idle_mean + eps)

    duration = df.get("Flow Duration", pd.Series(0, index=df.index)) + eps
    df["pkts_per_duration"] = total_pkts / duration

    log.info("  Added 14 ratio features")
    return df


#  Main pipeline 

def run_multiday_pipeline():
    log.info("=" * 60)
    log.info("NetShield -- Multi-Day Preprocessing v4")
    log.info("=" * 60)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(RAW_DATA_DIR.glob("*_TrafficForML_CICFlowMeter.csv"))
    if not csv_files:
        log.error("No CSV files found in %s/", RAW_DATA_DIR)
        return

    log.info("Found %d CSV files", len(csv_files))

    train_files   = [f for f in csv_files if any(d in f.name for d in TRAINING_DAYS)]
    holdout_files = [f for f in csv_files if HOLDOUT_DAY in f.name]

    if not train_files:
        log.error("No matching training files found!")
        return

    log.info("Training files: %d", len(train_files))
    log.info("Holdout files:  %d (%s)", len(holdout_files), HOLDOUT_DAY)

    #  Load all training days 
    log.info("")
    log.info("--- Loading training data ---")
    dfs = []
    for path in train_files:
        df = load_and_clean(path)
        df = add_ratio_features(df)
        dfs.append(df)

    # Feature columns: numeric intersection across all days
    label_col = "Label"
    exclude = {label_col, "_source_file", "Flow ID", "Src IP", "Dst IP",
               "Src Port", "Dst Port", "Protocol", "Timestamp"}

    common_cols = None
    for d in dfs:
        if len(d) == 0:
            continue
        numeric = set(d.select_dtypes(include=[np.number]).columns) - exclude
        common_cols = numeric if common_cols is None else common_cols & numeric

    feature_cols = sorted(common_cols)
    log.info("Features (common across all days): %d", len(feature_cols))

    dfs = [d for d in dfs if len(d) > 0]
    df  = pd.concat(dfs, ignore_index=True)
    log.info("Combined training data: %d rows", len(df))

    #  Label distribution 
    log.info("")
    log.info("Label distribution:")
    for label, count in df[label_col].value_counts().items():
        log.info("  %-30s %10d  (%.1f%%)", label, count, count / len(df) * 100)

    #  Encode labels 
    df["label_original"] = df[label_col]
    df["label"] = (df[label_col] != "Benign").astype(int)
    df = df.drop(columns=[label_col])

    #  Split BEFORE any transform (prevents data leakage) 
    log.info("Splitting data (70/15/15)...")
    X = df[feature_cols].values
    y = df["label"].values
    labels_original = df["label_original"].values

    X_trainval, X_test, y_trainval, y_test, lo_trainval, lo_test = train_test_split(
        X, y, labels_original, test_size=0.15, stratify=y, random_state=42,
    )
    X_train, X_val, y_train, y_val, lo_train, lo_val = train_test_split(
        X_trainval, y_trainval, lo_trainval,
        test_size=0.15 / 0.85, stratify=y_trainval, random_state=42,
    )

    log.info("  Train: %d (benign=%d, attack=%d)",
             len(X_train), (y_train == 0).sum(), (y_train == 1).sum())
    log.info("  Val:   %d", len(X_val))
    log.info("  Test:  %d", len(X_test))

    #  Fit QuantileTransformer on benign training data only 
    log.info("Fitting QuantileTransformer on benign training data...")
    benign_mask = y_train == 0
    X_benign = X_train[benign_mask]

    # Subsample for fitting if too large (QuantileTransformer is memory-heavy)
    max_fit_samples = 500_000
    if len(X_benign) > max_fit_samples:
        rng = np.random.default_rng(42)
        fit_idx = rng.choice(len(X_benign), size=max_fit_samples, replace=False)
        X_fit = X_benign[fit_idx]
    else:
        X_fit = X_benign

    scaler = QuantileTransformer(
        n_quantiles=min(10000, len(X_fit)),
        output_distribution="normal",  # map to standard normal
        subsample=len(X_fit),
        random_state=42,
    )
    scaler.fit(X_fit)
    log.info("  Fit on %d benign samples (%d quantiles)",
             len(X_fit), scaler.n_quantiles)

    X_train = scaler.transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # Clamp post-transform (quantile transform can produce extreme values for OOD data)
    X_train = np.clip(X_train, -CLAMP_RANGE, CLAMP_RANGE)
    X_val   = np.clip(X_val,   -CLAMP_RANGE, CLAMP_RANGE)
    X_test  = np.clip(X_test,  -CLAMP_RANGE, CLAMP_RANGE)
    log.info("  Clamped to [%.1f, %.1f]", -CLAMP_RANGE, CLAMP_RANGE)

    #  Quick signal check 
    benign_means = X_train[y_train == 0].mean(axis=0)
    attack_means = X_train[y_train == 1].mean(axis=0)
    diffs = np.abs(attack_means - benign_means)
    top_idx = np.argsort(diffs)[-10:][::-1]
    log.info("")
    log.info("Top 10 features by |benign_mean - attack_mean| after transform:")
    for i in top_idx:
        log.info("  %-30s  benign=%.4f  attack=%.4f  diff=%.4f",
                 feature_cols[i], benign_means[i], attack_means[i], diffs[i])

    #  Save splits 
    log.info("")
    log.info("Saving splits...")
    np.savez_compressed(SPLITS_DIR / "train.npz", X=X_train, y=y_train, labels=lo_train)
    np.savez_compressed(SPLITS_DIR / "val.npz",   X=X_val,   y=y_val,   labels=lo_val)
    np.savez_compressed(SPLITS_DIR / "test.npz",  X=X_test,  y=y_test,  labels=lo_test)

    joblib.dump(scaler, ARTIFACTS_DIR / "scaler.joblib")

    feature_meta = {
        "feature_names":   feature_cols,
        "n_features":      len(feature_cols),
        "log_transformed": [],  # no manual log transform — quantile handles it
        "training_days":   [f.name for f in train_files],
        "holdout_day":     HOLDOUT_DAY,
        "scaler":          "QuantileTransformer(output_distribution='normal')",
        "ratio_features": [
            "bytes_per_pkt", "fwd_bwd_pkt_ratio", "fwd_bwd_byte_ratio",
            "fwd_pkt_size_range", "fwd_pkt_cv", "iat_cv",
            "syn_per_pkt", "fin_per_pkt", "rst_per_pkt", "psh_per_pkt",
            "ack_per_pkt", "active_idle_ratio", "pkts_per_duration",
        ],
    }
    with open(ARTIFACTS_DIR / "feature_meta.json", "w") as f:
        json.dump(feature_meta, f, indent=2)

    #  Process holdout day 
    if holdout_files:
        log.info("")
        log.info("--- Processing holdout day ---")
        hdf = load_and_clean(holdout_files[0])
        hdf = add_ratio_features(hdf)

        h_label_col = next((c for c in hdf.columns if c.lower() == "label"), "Label")

        for col in feature_cols:
            if col not in hdf.columns:
                log.warning("  Missing feature in holdout: %s (filling 0)", col)
                hdf[col] = 0.0

        X_holdout = hdf[feature_cols].values
        y_holdout = (hdf[h_label_col] != "Benign").astype(int).values
        labels_holdout = hdf[h_label_col].values

        X_holdout = scaler.transform(X_holdout)
        X_holdout = np.clip(X_holdout, -CLAMP_RANGE, CLAMP_RANGE)

        np.savez_compressed(
            SPLITS_DIR / "holdout.npz",
            X=X_holdout, y=y_holdout, labels=labels_holdout,
        )
        log.info("  Holdout: %d samples", len(X_holdout))
        log.info("  Labels: %s", dict(zip(*np.unique(labels_holdout, return_counts=True))))

    #  Summary 
    log.info("")
    log.info("=" * 60)
    log.info("DONE")
    log.info("=" * 60)
    log.info("  Features:    %d", len(feature_cols))
    log.info("  Scaler:      QuantileTransformer (normal, benign-only)")
    log.info("  Train days:  %d", len(train_files))
    log.info("  Holdout:     %s", HOLDOUT_DAY)
    log.info("")
    log.info("Next: python -m src.model.train")


if __name__ == "__main__":
    run_multiday_pipeline()