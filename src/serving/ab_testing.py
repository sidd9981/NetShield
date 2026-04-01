"""
NetShield -- A/B Testing & Canary Deployment
===============================================
Manages safe model rollouts using canary deployments with Bayesian
statistical testing to decide promotion vs rollback.

How it works:
  1. New model (challenger) gets 10% of traffic, current model (champion) gets 90%
  2. Both models score the same flows, results are logged
  3. Bayesian test compares anomaly detection performance
  4. If challenger is significantly better -> promote to 100%
  5. If challenger is worse -> rollback, keep champion
  6. If inconclusive -> continue collecting data

Usage:
    # Run A/B test simulation using test data
    python -m src.serving.ab_testing simulate

    # Analyze results from a running test
    python -m src.serving.ab_testing analyze
"""

import argparse
import json
import logging
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict

import numpy as np
import torch
from scipy import stats as sp_stats

from src.model.model import TransformerAutoencoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ARTIFACTS_DIR = Path("artifacts")
SPLITS_DIR = Path("data/splits")


@dataclass
class ABConfig:
    canary_fraction: float = 0.1       # 10% traffic to challenger
    min_samples: int = 1000            # minimum before making decision
    significance_threshold: float = 0.95  # Bayesian probability threshold
    max_samples: int = 50000           # stop test after this many


@dataclass
class ModelVariant:
    name: str
    model: object
    threshold: float
    scores: list = field(default_factory=list)
    predictions: list = field(default_factory=list)
    true_labels: list = field(default_factory=list)


class ABTestManager:
    """Manages A/B testing between champion and challenger models."""

    def __init__(self, cfg: ABConfig = None):
        self.cfg = cfg or ABConfig()
        self.champion = None
        self.challenger = None
        self.results_log = []

    def load_champion(self, model_path: Path, threshold_path: Path):
        """Load the current production model."""
        with open(ARTIFACTS_DIR / "feature_meta.json") as f:
            meta = json.load(f)

        model = TransformerAutoencoder(n_features=meta["n_features"])
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()

        with open(threshold_path) as f:
            threshold = json.load(f)["threshold"]

        self.champion = ModelVariant(
            name="champion",
            model=model,
            threshold=threshold,
        )
        log.info("Champion loaded (threshold=%.6f)", threshold)

    def load_challenger(self, model_path: Path, threshold_path: Path):
        """Load the candidate model to test."""
        with open(ARTIFACTS_DIR / "feature_meta.json") as f:
            meta = json.load(f)

        model = TransformerAutoencoder(n_features=meta["n_features"])
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()

        with open(threshold_path) as f:
            threshold = json.load(f)["threshold"]

        self.challenger = ModelVariant(
            name="challenger",
            model=model,
            threshold=threshold,
        )
        log.info("Challenger loaded (threshold=%.6f)", threshold)

    def route_traffic(self) -> str:
        """Decide which model handles this request.

        Returns 'champion' or 'challenger' based on canary fraction.
        """
        if np.random.random() < self.cfg.canary_fraction:
            return "challenger"
        return "champion"

    def score_flow(self, X: torch.Tensor, y_true: int, variant_name: str):
        """Score a flow with the specified model variant and log results."""
        variant = self.champion if variant_name == "champion" else self.challenger

        with torch.no_grad():
            score = variant.model.anomaly_score(X.unsqueeze(0)).item()

        prediction = 1 if score > variant.threshold else 0

        variant.scores.append(score)
        variant.predictions.append(prediction)
        variant.true_labels.append(y_true)

    def score_both(self, X: torch.Tensor, y_true: int):
        """Score a flow with BOTH models (for fair comparison)."""
        for variant in [self.champion, self.challenger]:
            with torch.no_grad():
                score = variant.model.anomaly_score(X.unsqueeze(0)).item()

            prediction = 1 if score > variant.threshold else 0
            variant.scores.append(score)
            variant.predictions.append(prediction)
            variant.true_labels.append(y_true)

    def compute_metrics(self, variant: ModelVariant) -> dict:
        """Compute detection metrics for a model variant."""
        if len(variant.predictions) == 0:
            return {}

        y_true = np.array(variant.true_labels)
        y_pred = np.array(variant.predictions)
        scores = np.array(variant.scores)

        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(y_true, scores)
        except ValueError:
            auc = 0.0

        return {
            "n_samples": len(y_true),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "fpr": float(fpr),
            "auc_roc": float(auc),
            "tp": int(tp), "fp": int(fp),
            "tn": int(tn), "fn": int(fn),
        }

    def bayesian_test(self) -> dict:
        """Bayesian comparison of champion vs challenger F1 scores.

        Uses a Beta-Binomial model:
        - Each correct detection is a "success"
        - Model the success rate with Beta(successes+1, failures+1)
        - Compare: P(challenger_rate > champion_rate)

        Returns dict with decision and probability.
        """
        champ_preds = np.array(self.champion.predictions)
        champ_labels = np.array(self.champion.true_labels)
        chall_preds = np.array(self.challenger.predictions)
        chall_labels = np.array(self.challenger.true_labels)

        # Correct predictions = true positives + true negatives
        champ_correct = ((champ_preds == champ_labels)).sum()
        champ_wrong = len(champ_preds) - champ_correct

        chall_correct = ((chall_preds == chall_labels)).sum()
        chall_wrong = len(chall_preds) - chall_correct

        # Beta posteriors
        alpha_champ = champ_correct + 1
        beta_champ = champ_wrong + 1
        alpha_chall = chall_correct + 1
        beta_chall = chall_wrong + 1

        # Monte Carlo estimate: P(challenger > champion)
        n_mc = 100000
        rng = np.random.default_rng(42)
        champ_samples = rng.beta(alpha_champ, beta_champ, size=n_mc)
        chall_samples = rng.beta(alpha_chall, beta_chall, size=n_mc)

        prob_challenger_better = (chall_samples > champ_samples).mean()

        # Decision
        if prob_challenger_better > self.cfg.significance_threshold:
            decision = "PROMOTE"
        elif prob_challenger_better < (1 - self.cfg.significance_threshold):
            decision = "ROLLBACK"
        else:
            decision = "CONTINUE"

        return {
            "prob_challenger_better": float(prob_challenger_better),
            "decision": decision,
            "champion_accuracy": float(champ_correct / len(champ_preds)),
            "challenger_accuracy": float(chall_correct / len(chall_preds)),
            "champion_samples": len(champ_preds),
            "challenger_samples": len(chall_preds),
        }


def run_simulation():
    """Simulate an A/B test using test data.

    Uses the same model as both champion and challenger (with a slightly
    different threshold for the challenger) to demonstrate the framework.
    """
    log.info("=" * 60)
    log.info("A/B Test Simulation")
    log.info("=" * 60)

    cfg = ABConfig()
    manager = ABTestManager(cfg)

    # Load champion
    manager.load_champion(
        ARTIFACTS_DIR / "best_model.pt",
        ARTIFACTS_DIR / "threshold.json",
    )

    # For simulation: use same model with adjusted threshold as challenger
    # In production, this would be a newly trained model
    manager.load_challenger(
        ARTIFACTS_DIR / "best_model.pt",
        ARTIFACTS_DIR / "threshold.json",
    )
    # Simulate a "new" model by slightly adjusting the threshold
    manager.challenger.threshold *= 0.9
    log.info("Challenger threshold adjusted to %.6f (simulating new model)",
             manager.challenger.threshold)

    # Load test data
    test = np.load(SPLITS_DIR / "test.npz", allow_pickle=True)
    X_test = torch.tensor(test["X"], dtype=torch.float32)
    y_test = test["y"]

    # Subsample for speed
    rng = np.random.default_rng(42)
    n_samples = min(cfg.max_samples, len(X_test))
    idx = rng.choice(len(X_test), size=n_samples, replace=False)

    log.info("")
    log.info("Running A/B test with %d samples...", n_samples)
    log.info("  Canary fraction: %.0f%%", cfg.canary_fraction * 100)

    # Run the test
    check_interval = 5000
    for i, sample_idx in enumerate(idx):
        X = X_test[sample_idx]
        y = int(y_test[sample_idx])

        # Score with BOTH models for fair comparison
        manager.score_both(X, y)

        # Periodic check
        if (i + 1) % check_interval == 0:
            champ_metrics = manager.compute_metrics(manager.champion)
            chall_metrics = manager.compute_metrics(manager.challenger)
            bayes = manager.bayesian_test()

            log.info("")
            log.info("--- Check at %d samples ---", i + 1)
            log.info("  Champion:   F1=%.4f  AUC=%.4f  Recall=%.4f  FPR=%.4f",
                     champ_metrics["f1"], champ_metrics["auc_roc"],
                     champ_metrics["recall"], champ_metrics["fpr"])
            log.info("  Challenger: F1=%.4f  AUC=%.4f  Recall=%.4f  FPR=%.4f",
                     chall_metrics["f1"], chall_metrics["auc_roc"],
                     chall_metrics["recall"], chall_metrics["fpr"])
            log.info("  P(challenger > champion) = %.4f",
                     bayes["prob_challenger_better"])
            log.info("  Decision: %s", bayes["decision"])

            if bayes["decision"] in ("PROMOTE", "ROLLBACK"):
                log.info("  -> Early decision reached at %d samples", i + 1)
                break

    # Final results
    log.info("")
    log.info("=" * 60)
    log.info("FINAL RESULTS")
    log.info("=" * 60)

    champ_metrics = manager.compute_metrics(manager.champion)
    chall_metrics = manager.compute_metrics(manager.challenger)
    bayes = manager.bayesian_test()

    log.info("")
    log.info("Champion (%d samples):", champ_metrics["n_samples"])
    log.info("  Precision: %.4f  Recall: %.4f  F1: %.4f",
             champ_metrics["precision"], champ_metrics["recall"], champ_metrics["f1"])
    log.info("  AUC: %.4f  FPR: %.4f", champ_metrics["auc_roc"], champ_metrics["fpr"])

    log.info("")
    log.info("Challenger (%d samples):", chall_metrics["n_samples"])
    log.info("  Precision: %.4f  Recall: %.4f  F1: %.4f",
             chall_metrics["precision"], chall_metrics["recall"], chall_metrics["f1"])
    log.info("  AUC: %.4f  FPR: %.4f", chall_metrics["auc_roc"], chall_metrics["fpr"])

    log.info("")
    log.info("Bayesian Test:")
    log.info("  P(challenger better): %.4f", bayes["prob_challenger_better"])
    log.info("  Decision:             %s", bayes["decision"])

    # Save results
    results = {
        "champion": champ_metrics,
        "challenger": chall_metrics,
        "bayesian_test": bayes,
        "config": asdict(cfg),
    }
    results_path = ARTIFACTS_DIR / "ab_test_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("  Results saved: %s", results_path)


def run_analyze():
    """Analyze saved A/B test results."""
    results_path = ARTIFACTS_DIR / "ab_test_results.json"
    if not results_path.exists():
        log.error("No results found. Run 'simulate' first.")
        return

    with open(results_path) as f:
        results = json.load(f)

    log.info("=" * 60)
    log.info("A/B Test Analysis")
    log.info("=" * 60)

    champ = results["champion"]
    chall = results["challenger"]
    bayes = results["bayesian_test"]

    log.info("")
    log.info("%-20s %12s %12s %10s", "Metric", "Champion", "Challenger", "Delta")
    log.info("-" * 56)
    for metric in ["precision", "recall", "f1", "auc_roc", "fpr"]:
        c = champ[metric]
        ch = chall[metric]
        delta = ch - c
        direction = "+" if delta > 0 else ""
        log.info("%-20s %12.4f %12.4f %10s",
                 metric, c, ch, f"{direction}{delta:.4f}")

    log.info("")
    log.info("Decision: %s (P=%.4f)",
             bayes["decision"], bayes["prob_challenger_better"])


def main():
    parser = argparse.ArgumentParser(description="NetShield A/B Testing")
    parser.add_argument("command", choices=["simulate", "analyze"],
                        help="simulate: run A/B test | analyze: view results")
    args = parser.parse_args()

    if args.command == "simulate":
        run_simulation()
    elif args.command == "analyze":
        run_analyze()


if __name__ == "__main__":
    main()