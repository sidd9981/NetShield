"""
NetShield -- Data Drift Detection

Monitors whether incoming network traffic patterns are shifting
away from the training data distribution.

How it works:
  - Reference: benign training data (what the model learned)
  - Current: recent production flows
  - Per-feature KS test: is the distribution different?
  - If >30% of features show significant drift (p < 0.05),
    flag for retraining

Usage:
    python -m src.monitoring.drift_detector test
    python -m src.monitoring.drift_detector holdout
    python -m src.monitoring.drift_detector monitor
"""

import argparse
import json
import logging
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ARTIFACTS_DIR = Path("artifacts")
SPLITS_DIR = Path("data/splits")
REPORTS_DIR = Path("artifacts/drift_reports")


class DriftDetector:
    """Monitors data drift using per-feature KS tests."""

    def __init__(self, p_threshold: float = 0.05, drift_share_threshold: float = 0.3):
        """
        Args:
            p_threshold: p-value below which a feature is considered drifted
            drift_share_threshold: fraction of features that must drift to trigger retraining
        """
        self.p_threshold = p_threshold
        self.drift_share_threshold = drift_share_threshold

        with open(ARTIFACTS_DIR / "feature_meta.json") as f:
            self.meta = json.load(f)
        self.feature_names = self.meta["feature_names"]

        # Load reference data: benign training samples
        train = np.load(SPLITS_DIR / "train.npz", allow_pickle=True)
        benign_mask = train["y"] == 0
        benign_X = train["X"][benign_mask]

        # Subsample for speed
        rng = np.random.default_rng(42)
        if len(benign_X) > 10000:
            idx = rng.choice(len(benign_X), size=10000, replace=False)
            benign_X = benign_X[idx]

        self.reference = benign_X
        log.info("Reference data: %d benign samples, %d features",
                 len(self.reference), len(self.feature_names))

    def check_drift(self, current_data: np.ndarray, report_name: str = None) -> dict:
        """Run KS test on each feature: current vs reference.

        Args:
            current_data: (n_samples, n_features) scaled feature array
            report_name: optional name for saving JSON report

        Returns:
            dict with drift results
        """
        n_features = len(self.feature_names)
        drifted_features = []

        for i, fname in enumerate(self.feature_names):
            ref_col = self.reference[:, i]
            cur_col = current_data[:, i]

            ks_stat, p_value = stats.ks_2samp(ref_col, cur_col)

            if p_value < self.p_threshold:
                drifted_features.append({
                    "feature": fname,
                    "ks_stat": float(ks_stat),
                    "p_value": float(p_value),
                    "ref_mean": float(ref_col.mean()),
                    "cur_mean": float(cur_col.mean()),
                    "mean_shift": float(cur_col.mean() - ref_col.mean()),
                })

        n_drifted = len(drifted_features)
        drift_share = n_drifted / n_features
        needs_retraining = drift_share > self.drift_share_threshold

        result = {
            "is_drifted": n_drifted > 0,
            "drift_share": float(drift_share),
            "n_drifted": n_drifted,
            "n_features": n_features,
            "needs_retraining": needs_retraining,
            "timestamp": datetime.now().isoformat(),
            "drifted_features": drifted_features,
        }

        log.info("")
        log.info("=== Drift Detection Results ===")
        log.info("  Drifted features: %d / %d (%.1f%%)",
                 n_drifted, n_features, drift_share * 100)
        log.info("  Drift threshold:  %.0f%%", self.drift_share_threshold * 100)
        log.info("  Needs retraining: %s", "YES" if needs_retraining else "NO")

        if drifted_features:
            log.info("")
            log.info("  Top drifted features:")
            log.info("  %-30s %10s %12s %10s",
                     "Feature", "KS Stat", "p-value", "Mean Shift")
            log.info("  " + "-" * 65)
            for feat in sorted(drifted_features,
                               key=lambda x: x["ks_stat"],
                               reverse=True)[:10]:
                log.info("  %-30s %10.4f %12.2e %10.4f",
                         feat["feature"], feat["ks_stat"],
                         feat["p_value"], feat["mean_shift"])

        # Save report
        if report_name:
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            report_path = REPORTS_DIR / f"{report_name}.json"
            with open(report_path, "w") as f:
                json.dump(result, f, indent=2)
            log.info("  Report saved: %s", report_path)

        return result


def run_test_drift():
    """Check drift between training benign and test data.
    Expect some drift since test includes attacks.
    """
    log.info("Checking drift: training vs test data...")
    detector = DriftDetector()

    test = np.load(SPLITS_DIR / "test.npz", allow_pickle=True)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(test["X"]), size=min(10000, len(test["X"])), replace=False)

    result = detector.check_drift(test["X"][idx], report_name="test_drift")
    return result


def run_holdout_drift():
    """Check drift against holdout day — expect significant drift."""
    holdout_path = SPLITS_DIR / "holdout.npz"
    if not holdout_path.exists():
        log.error("Holdout data not found at %s", holdout_path)
        return

    log.info("Checking drift: training vs holdout day...")
    detector = DriftDetector()

    holdout = np.load(holdout_path, allow_pickle=True)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(holdout["X"]), size=min(10000, len(holdout["X"])), replace=False)

    result = detector.check_drift(holdout["X"][idx], report_name="holdout_drift")
    return result


def run_monitor(interval: int = 30, n_batches: int = 10):
    """Simulate periodic drift monitoring."""
    log.info("Starting drift monitor (interval=%ds, batches=%d)...",
             interval, n_batches)
    detector = DriftDetector()

    test = np.load(SPLITS_DIR / "test.npz", allow_pickle=True)
    holdout_path = SPLITS_DIR / "holdout.npz"
    holdout = np.load(holdout_path, allow_pickle=True) if holdout_path.exists() else None

    rng = np.random.default_rng()
    history = []

    for i in range(n_batches):
        log.info("")
        log.info("--- Monitoring check %d/%d ---", i + 1, n_batches)

        # Alternate: test (mild drift) vs holdout (heavy drift)
        if holdout is not None and i % 3 == 2:
            source = holdout
            source_name = "holdout"
        else:
            source = test
            source_name = "test"

        idx = rng.choice(len(source["X"]), size=5000, replace=False)
        batch = source["X"][idx]

        log.info("  Source: %s (%d samples)", source_name, len(batch))
        result = detector.check_drift(batch)
        result["source"] = source_name
        result["check_number"] = i + 1
        history.append(result)

        if result["needs_retraining"]:
            log.warning("  *** RETRAINING RECOMMENDED ***")

        if i < n_batches - 1:
            log.info("  Next check in %ds...", interval)
            time.sleep(interval)

    # Summary
    log.info("")
    log.info("\n")
    log.info("MONITORING SUMMARY")
    log.info("\n")
    n_alerts = sum(1 for h in history if h["needs_retraining"])
    log.info("  Checks:     %d", len(history))
    log.info("  Alerts:     %d", n_alerts)
    log.info("  Alert rate: %.1f%%", n_alerts / len(history) * 100)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "monitor_history.json", "w") as f:
        json.dump(history, f, indent=2, default=str)
    log.info("  History saved: %s", REPORTS_DIR / "monitor_history.json")


def main():
    parser = argparse.ArgumentParser(description="NetShield Drift Detection")
    parser.add_argument("command", choices=["test", "holdout", "monitor"],
                        help="test: check test data | holdout: check holdout day | monitor: simulate periodic checks")
    parser.add_argument("--interval", type=int, default=30,
                        help="Monitoring interval in seconds")
    parser.add_argument("--n-batches", type=int, default=10,
                        help="Number of monitoring checks")
    args = parser.parse_args()

    if args.command == "test":
        run_test_drift()
    elif args.command == "holdout":
        run_holdout_drift()
    elif args.command == "monitor":
        run_monitor(interval=args.interval, n_batches=args.n_batches)


if __name__ == "__main__":
    main()