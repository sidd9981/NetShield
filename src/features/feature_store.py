"""
NetShield -- Redis Feature Store

Stores precomputed, scaled feature vectors in Redis for sub-10ms
retrieval during streaming inference. Replaces per-request
preprocessing with a single Redis GET.

Architecture:
    [Preprocessing Pipeline] -> Redis (hash per flow_id)
                                  |
                            GET flow_id
                                  |
                         Inference Engine (<10ms)

Each flow is stored as a Redis hash:
    flow:{flow_id} -> { feature_name: scaled_value, ... }

Usage:
    # Load features into Redis
    python -m src.features.feature_store load

    # Benchmark retrieval
    python -m src.features.feature_store bench

    # Test full pipeline: Redis -> Model -> Score
    python -m src.features.feature_store test
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import redis
import torch
import joblib

from src.model.model import TransformerAutoencoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ARTIFACTS_DIR = Path("artifacts")
SPLITS_DIR = Path("data/splits")
CLAMP_RANGE = 5.0


class FeatureStore:
    """Redis-backed feature store for precomputed flow features.

    Features are stored pre-scaled (QuantileTransformer already applied),
    so retrieval + model inference is all that's needed at serving time.
    No preprocessing per request — just GET and score.
    """

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        self.r = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        self.r.ping()
        log.info("Connected to Redis at %s:%d", host, port)

        with open(ARTIFACTS_DIR / "feature_meta.json") as f:
            self.meta = json.load(f)
        self.feature_names = self.meta["feature_names"]
        self.n_features = self.meta["n_features"]

    def store_flow(self, flow_id: str, features: np.ndarray):
        """Store a single flow's scaled feature vector."""
        key = f"flow:{flow_id}"
        data = {
            fname: str(features[i])
            for i, fname in enumerate(self.feature_names)
        }
        self.r.hset(key, mapping=data)

    def store_batch(self, flow_ids: list, features: np.ndarray, ttl: int = 86400):
        """Store a batch of flows using Redis pipeline for speed.

        Args:
            flow_ids: list of flow ID strings
            features: (n_flows, n_features) scaled feature array
            ttl: time-to-live in seconds (default 24h)
        """
        pipe = self.r.pipeline()
        for i, flow_id in enumerate(flow_ids):
            key = f"flow:{flow_id}"
            data = {
                fname: str(features[i, j])
                for j, fname in enumerate(self.feature_names)
            }
            pipe.hset(key, mapping=data)
            if ttl > 0:
                pipe.expire(key, ttl)

        pipe.execute()

    def get_flow(self, flow_id: str) -> np.ndarray | None:
        """Retrieve a single flow's feature vector from Redis.

        Returns:
            (n_features,) numpy array of scaled features, or None if not found
        """
        key = f"flow:{flow_id}"
        data = self.r.hgetall(key)
        if not data:
            return None

        features = np.array([
            float(data.get(fname, 0.0))
            for fname in self.feature_names
        ], dtype=np.float32)

        return features

    def get_batch(self, flow_ids: list) -> np.ndarray:
        """Retrieve multiple flows using Redis pipeline.

        Returns:
            (n_flows, n_features) array. Missing flows get zeros.
        """
        pipe = self.r.pipeline()
        for flow_id in flow_ids:
            pipe.hgetall(f"flow:{flow_id}")

        results = pipe.execute()
        features = np.zeros((len(flow_ids), self.n_features), dtype=np.float32)

        for i, data in enumerate(results):
            if data:
                for j, fname in enumerate(self.feature_names):
                    features[i, j] = float(data.get(fname, 0.0))

        return features

    def flow_count(self) -> int:
        """Count stored flows."""
        return len(list(self.r.scan_iter(match="flow:*", count=1000)))

    def flush_flows(self):
        """Delete all stored flows."""
        pipe = self.r.pipeline()
        for key in self.r.scan_iter(match="flow:*", count=1000):
            pipe.delete(key)
        pipe.execute()


def load_features(max_flows: int = 50000):
    """Load scaled test features into Redis."""
    log.info("Loading features into Redis...")
    store = FeatureStore()

    # Clear existing
    log.info("  Flushing existing flows...")
    store.flush_flows()

    # Load test split (already scaled)
    test = np.load(SPLITS_DIR / "test.npz", allow_pickle=True)
    X = test["X"]
    y = test["y"]
    labels = test["labels"]

    n = min(max_flows, len(X))
    log.info("  Loading %d flows...", n)

    # Store in batches of 1000
    batch_size = 1000
    t0 = time.time()
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        flow_ids = [f"flow-{i}" for i in range(start, end)]
        store.store_batch(flow_ids, X[start:end])

        if (start + batch_size) % 10000 == 0:
            log.info("    Stored %d / %d", start + batch_size, n)

    elapsed = time.time() - t0
    log.info("  Loaded %d flows in %.1fs (%.0f flows/sec)",
             n, elapsed, n / elapsed)

    # Store metadata
    store.r.set("meta:n_flows", n)
    store.r.set("meta:n_features", store.n_features)
    store.r.set("meta:loaded_at", time.strftime("%Y-%m-%d %H:%M:%S"))

    log.info("  Done. Redis keys: %d", store.flow_count())


def run_benchmark():
    """Benchmark Redis feature retrieval latency."""
    log.info("Benchmarking feature retrieval...")
    store = FeatureStore()

    # Single flow retrieval
    log.info("")
    log.info("--- Single flow retrieval ---")
    times = []
    for i in range(100):
        t0 = time.time()
        features = store.get_flow(f"flow-{i}")
        times.append((time.time() - t0) * 1000)

    log.info("  Mean: %.2fms  Median: %.2fms  p95: %.2fms  p99: %.2fms",
             np.mean(times), np.median(times),
             np.percentile(times, 95), np.percentile(times, 99))

    # Batch retrieval
    for batch_size in [10, 50, 100, 500]:
        log.info("")
        log.info("--- Batch retrieval (%d flows) ---", batch_size)
        flow_ids = [f"flow-{i}" for i in range(batch_size)]
        times = []
        for _ in range(20):
            t0 = time.time()
            features = store.get_batch(flow_ids)
            times.append((time.time() - t0) * 1000)

        log.info("  Mean: %.2fms  Median: %.2fms  p95: %.2fms",
                 np.mean(times), np.median(times), np.percentile(times, 95))
        log.info("  Per-flow: %.3fms", np.mean(times) / batch_size)


def run_test():
    """Test full pipeline: Redis -> Model -> Score."""
    log.info("Testing full pipeline: Redis -> Model -> Score")
    store = FeatureStore()

    # Load model
    model = TransformerAutoencoder(n_features=store.n_features)
    model.load_state_dict(torch.load(ARTIFACTS_DIR / "best_model.pt", weights_only=True))
    model.eval()

    with open(ARTIFACTS_DIR / "threshold.json") as f:
        threshold = json.load(f)["threshold"]

    # Load ground truth for comparison
    test = np.load(SPLITS_DIR / "test.npz", allow_pickle=True)
    y = test["y"]
    labels = test["labels"]

    # Retrieve and score 200 flows
    n_test = 200
    flow_ids = [f"flow-{i}" for i in range(n_test)]

    t0 = time.time()

    # Step 1: Redis retrieval
    t_redis = time.time()
    X = store.get_batch(flow_ids)
    redis_ms = (time.time() - t_redis) * 1000

    # Step 2: Model inference
    t_model = time.time()
    X_t = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        scores = model.anomaly_score(X_t).numpy()
    model_ms = (time.time() - t_model) * 1000

    total_ms = (time.time() - t0) * 1000
    predictions = (scores > threshold).astype(int)

    # Compare with ground truth
    y_sample = y[:n_test]
    accuracy = (predictions == y_sample).mean()
    n_anomalies = predictions.sum()

    log.info("")
    log.info("Results:")
    log.info("")
    log.info("  Flows:         %d", n_test)
    log.info("  Redis GET:     %.1fms (%.2fms/flow)", redis_ms, redis_ms / n_test)
    log.info("  Model score:   %.1fms (%.2fms/flow)", model_ms, model_ms / n_test)
    log.info("  Total:         %.1fms (%.2fms/flow)", total_ms, total_ms / n_test)
    log.info("  Anomalies:     %d", n_anomalies)
    log.info("  Accuracy:      %.4f", accuracy)
    log.info("")

    # Show samples
    log.info("%-12s %-20s %-10s %-10s",
             "Score", "Label", "Predicted", "Correct")
    log.info("-" * 55)
    for i in range(min(10, n_test)):
        pred = "ANOMALY" if predictions[i] else "benign"
        correct = "Y" if predictions[i] == y[i] else "N"
        log.info("%-12.6f %-20s %-10s %-10s",
                 scores[i], str(labels[i]), pred, correct)


def main():
    parser = argparse.ArgumentParser(description="NetShield Feature Store")
    parser.add_argument("command", choices=["load", "bench", "test"],
                        help="load: push features to Redis | bench: benchmark latency | test: full pipeline test")
    parser.add_argument("--max-flows", type=int, default=50000,
                        help="Max flows to load into Redis")
    args = parser.parse_args()

    if args.command == "load":
        load_features(max_flows=args.max_flows)
    elif args.command == "bench":
        run_benchmark()
    elif args.command == "test":
        run_test()


if __name__ == "__main__":
    main()