"""
NetShield — Kafka Streaming Inference with Prometheus Metrics

Consumes network flow records from Kafka, scores them through
the Transformer autoencoder, produces results, and exposes
Prometheus metrics on :${METRICS_PORT}/metrics.

Metrics exposed:
  - netshield_flows_processed_total     (counter, per worker)
  - netshield_anomalies_detected_total  (counter, per worker)
  - netshield_inference_latency_seconds (histogram)
  - netshield_anomaly_score             (histogram)
  - netshield_batch_size                (histogram)
  - netshield_anomaly_rate              (gauge — rolling 1000 flows)
  - netshield_model_threshold           (gauge)

Usage:
    # Terminal 1: infrastructure
    docker compose up -d

    # Terminal 2: inference consumer
    python -m src.serving.kafka_inference consume

    # Terminal 3: send test flows
    python -m src.serving.kafka_inference produce --n-flows 2000

    # Terminal 4: local test (no Kafka needed)
    python -m src.serving.kafka_inference test --n-flows 500

    # Multi-worker (each in its own terminal)
    WORKER_ID=worker-1 METRICS_PORT=8001 python -m src.serving.kafka_inference consume
    WORKER_ID=worker-2 METRICS_PORT=8002 python -m src.serving.kafka_inference consume
    WORKER_ID=worker-3 METRICS_PORT=8003 python -m src.serving.kafka_inference consume

    # Metrics at http://localhost:${METRICS_PORT}/metrics
"""

import argparse
import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
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
SPLITS_DIR    = Path("data/splits")
CLAMP_RANGE   = 5.0

WORKER_ID    = os.getenv("WORKER_ID", "worker-1")
METRICS_PORT = int(os.getenv("METRICS_PORT", "8000"))


#  Prometheus metrics 

try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server

    FLOWS_PROCESSED = Counter(
        "netshield_flows_processed_total",
        "Total flows processed",
        ["worker"],
    )
    ANOMALIES_DETECTED = Counter(
        "netshield_anomalies_detected_total",
        "Total anomalies detected",
        ["worker"],
    )
    INFERENCE_LATENCY = Histogram(
        "netshield_inference_latency_seconds",
        "Batch scoring latency",
        buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
    )
    ANOMALY_SCORE = Histogram(
        "netshield_anomaly_score",
        "Distribution of anomaly scores",
        buckets=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
    )
    BATCH_SIZE_HIST = Histogram(
        "netshield_batch_size",
        "Flows per inference batch",
        buckets=[1, 2, 4, 8, 16, 32, 64, 128],
    )
    ANOMALY_RATE = Gauge(
        "netshield_anomaly_rate",
        "Rolling anomaly rate (last 1000 flows)",
    )
    MODEL_THRESHOLD = Gauge(
        "netshield_model_threshold",
        "Current anomaly detection threshold",
    )

    PROMETHEUS_AVAILABLE = True

except ImportError:
    PROMETHEUS_AVAILABLE = False
    log.warning("prometheus_client not installed — metrics disabled. pip install prometheus-client")


#  Configuration 

@dataclass
class KafkaConfig:
    bootstrap_servers: str = "localhost:9092"
    input_topic:       str = "network-flows"
    output_topic:      str = "anomaly-scores"
    consumer_group:    str = "netshield-inference"
    batch_size:        int = 64
    batch_timeout_ms:  int = 500
    metrics_port:      int = METRICS_PORT


#  Inference engine 

class InferenceEngine:
    """Wraps model + scaler + threshold. Loaded once at startup."""

    def __init__(self):
        log.info("Loading inference engine (worker=%s)...", WORKER_ID)

        with open(ARTIFACTS_DIR / "feature_meta.json") as f:
            self.meta = json.load(f)

        self.feature_names   = self.meta["feature_names"]
        self.n_features      = self.meta["n_features"]
        self.log_transformed = set(self.meta.get("log_transformed", []))

        self.scaler = joblib.load(ARTIFACTS_DIR / "scaler.joblib")

        with open(ARTIFACTS_DIR / "threshold.json") as f:
            self.threshold = json.load(f)["threshold"]

        self.model = TransformerAutoencoder(n_features=self.n_features)
        self.model.load_state_dict(
            torch.load(ARTIFACTS_DIR / "best_model.pt", weights_only=True)
        )
        self.model.eval()

        if PROMETHEUS_AVAILABLE:
            MODEL_THRESHOLD.set(self.threshold)

        log.info("  Loaded: %d features, threshold=%.6f", self.n_features, self.threshold)

    def preprocess_flow(self, flow: dict) -> np.ndarray:
        """Preprocess a single raw flow dict into a scaled feature vector.

        Applies identical transforms to the training pipeline:
          1. Extract features in training order (missing → 0)
          2. Clip negatives for physical features
          3. Log-transform skewed features (same set as training)
          4. QuantileTransformer scale
          5. Clamp to [-CLAMP_RANGE, CLAMP_RANGE]
        """
        values = []
        for fname in self.feature_names:
            val = flow.get(fname, 0.0)
            try:
                val = float(val)
            except (ValueError, TypeError):
                val = 0.0
            if not np.isfinite(val):
                val = 0.0
            if val < 0 and any(
                kw in fname.lower()
                for kw in ["packet", "byte", "length", "duration", "size"]
            ):
                val = 0.0
            values.append(val)

        x = np.array(values, dtype=np.float64)

        # Log-transform same features as training
        for i, fname in enumerate(self.feature_names):
            if fname in self.log_transformed and x[i] >= 0:
                x[i] = np.log1p(x[i])

        x = self.scaler.transform(x.reshape(1, -1))
        x = np.clip(x, -CLAMP_RANGE, CLAMP_RANGE)
        return x.flatten()

    def predict_batch(self, flows: list[dict]) -> list[dict]:
        """Score a batch of raw flow dicts. Returns one result dict per flow."""
        if not flows:
            return []

        t_start = time.time()

        vectors = np.array([self.preprocess_flow(f) for f in flows])
        X = torch.tensor(vectors, dtype=torch.float32)

        with torch.no_grad():
            scores = self.model.anomaly_score(X).numpy()

        elapsed_ms = (time.time() - t_start) * 1000
        ms_per_flow = elapsed_ms / len(flows)

        results = []
        for i, (flow, score) in enumerate(zip(flows, scores)):
            is_anomaly = bool(score > self.threshold)
            result = {
                "flow_id":       flow.get("flow_id", f"flow-{i}"),
                "anomaly_score": float(score),
                "is_anomaly":    is_anomaly,
                "threshold":     self.threshold,
                "timestamp":     time.time(),
                "worker_id":     WORKER_ID,
                "processing_ms": round(ms_per_flow, 3),
                # Ground truth passthrough — set by mock producer, null in production
                "_label":        flow.get("_label"),
                "_attack_type":  flow.get("_label"),  # attack type == label
            }
            for key in ["Src IP", "Dst IP", "Src Port", "Dst Port", "Protocol"]:
                if key in flow:
                    result[key] = flow[key]
            results.append(result)

        return results

    def predict_single(self, flow: dict) -> dict:
        return self.predict_batch([flow])[0]


#  Kafka consumer 

def run_consumer(cfg: KafkaConfig) -> None:
    """Consume flows from Kafka, score them, produce results, expose metrics."""
    try:
        from kafka import KafkaConsumer, KafkaProducer
    except ImportError:
        log.error("kafka-python not installed: pip install kafka-python")
        return

    if PROMETHEUS_AVAILABLE:
        start_http_server(cfg.metrics_port)
        log.info("Prometheus metrics at http://localhost:%d/metrics", cfg.metrics_port)

    engine = InferenceEngine()

    consumer = KafkaConsumer(
        cfg.input_topic,
        bootstrap_servers=cfg.bootstrap_servers,
        group_id=cfg.consumer_group,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="latest",
        enable_auto_commit=True,
        max_poll_records=cfg.batch_size,
        fetch_max_wait_ms=cfg.batch_timeout_ms,
    )
    producer = KafkaProducer(
        bootstrap_servers=cfg.bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    log.info("Worker %s consuming '%s' → '%s'",
             WORKER_ID, cfg.input_topic, cfg.output_topic)
    log.info("Waiting for messages...")

    total_processed = 0
    total_anomalies = 0
    recent = deque(maxlen=1000)   # rolling window for anomaly rate

    try:
        while True:
            records = consumer.poll(
                timeout_ms=cfg.batch_timeout_ms,
                max_records=cfg.batch_size,
            )
            if not records:
                continue

            flows = [
                msg.value
                for tp, messages in records.items()
                for msg in messages
            ]
            if not flows:
                continue

            t0 = time.time()
            results = engine.predict_batch(flows)
            elapsed = time.time() - t0

            n_anomalies = 0
            for result in results:
                producer.send(cfg.output_topic, value=result)
                is_anom = result["is_anomaly"]
                if is_anom:
                    n_anomalies += 1
                recent.append(1 if is_anom else 0)
                if PROMETHEUS_AVAILABLE:
                    ANOMALY_SCORE.observe(result["anomaly_score"])

            producer.flush()
            total_processed += len(flows)
            total_anomalies += n_anomalies

            if PROMETHEUS_AVAILABLE:
                FLOWS_PROCESSED.labels(worker=WORKER_ID).inc(len(flows))
                ANOMALIES_DETECTED.labels(worker=WORKER_ID).inc(n_anomalies)
                INFERENCE_LATENCY.observe(elapsed)
                BATCH_SIZE_HIST.observe(len(flows))
                if recent:
                    ANOMALY_RATE.set(sum(recent) / len(recent))

            anomaly_rate = (sum(recent) / len(recent) * 100) if recent else 0
            log.info(
                "Batch: %d flows | %d anomalies | %.1fms | "
                "total: %d processed / %d anomalies | rate: %.1f%%",
                len(flows), n_anomalies, elapsed * 1000,
                total_processed, total_anomalies, anomaly_rate,
            )

    except KeyboardInterrupt:
        log.info("Shutting down %s ...", WORKER_ID)
    finally:
        consumer.close()
        producer.close()
        log.info("Done. Processed %d flows, %d anomalies.", total_processed, total_anomalies)


#  Mock producer 

def run_producer(cfg: KafkaConfig, n_flows: int = 1000, delay: float = 0.01) -> None:
    """Produce test flows from the test split to simulate live traffic.

    Inverse-transforms scaled test data back to raw feature values,
    undoing both the QuantileTransformer scaling and the log1p transforms,
    so the consumer's preprocessing pipeline runs on realistic raw values.
    """
    try:
        from kafka import KafkaProducer
    except ImportError:
        log.error("kafka-python not installed: pip install kafka-python")
        return

    with open(ARTIFACTS_DIR / "feature_meta.json") as f:
        meta = json.load(f)

    feature_names   = meta["feature_names"]
    log_transformed = set(meta.get("log_transformed", []))

    test   = np.load(SPLITS_DIR / "test.npz", allow_pickle=True)
    X_test = test["X"]
    labels = test["labels"]

    scaler = joblib.load(ARTIFACTS_DIR / "scaler.joblib")

    # Undo clamping then undo scaling
    X_unclamped = np.clip(X_test, -CLAMP_RANGE, CLAMP_RANGE)
    X_raw = scaler.inverse_transform(X_unclamped)

    # Undo log1p for log-transformed features
    for i, fname in enumerate(feature_names):
        if fname in log_transformed:
            X_raw[:, i] = np.expm1(X_raw[:, i])

    producer = KafkaProducer(
        bootstrap_servers=cfg.bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    log.info("Producing %d test flows to '%s'...", n_flows, cfg.input_topic)

    indices = np.random.default_rng(42).choice(
        len(X_raw), size=min(n_flows, len(X_raw)), replace=False,
    )

    sent = 0
    for idx in indices:
        flow = {"flow_id": f"test-{idx}"}
        for i, fname in enumerate(feature_names):
            flow[fname] = float(X_raw[idx, i])
        flow["_label"] = str(labels[idx])

        producer.send(cfg.input_topic, value=flow)
        sent += 1

        if sent % 100 == 0:
            log.info("  Sent %d/%d flows", sent, n_flows)
            producer.flush()

        time.sleep(delay)

    producer.flush()
    producer.close()
    log.info("Done. Sent %d flows.", sent)


#  Local test (no Kafka) 

def run_local_test(n_flows: int = 200) -> None:
    """Validate the full inference pipeline locally without Kafka.

    Two modes:
      Direct   — already-scaled test data → model (verifies model + threshold)
      Roundtrip — inverse-scale → preprocess → model (verifies full pipeline)
    """
    with open(ARTIFACTS_DIR / "feature_meta.json") as f:
        meta = json.load(f)

    model = TransformerAutoencoder(n_features=meta["n_features"])
    model.load_state_dict(
        torch.load(ARTIFACTS_DIR / "best_model.pt", weights_only=True)
    )
    model.eval()

    with open(ARTIFACTS_DIR / "threshold.json") as f:
        threshold = json.load(f)["threshold"]

    test    = np.load(SPLITS_DIR / "test.npz", allow_pickle=True)
    X_scaled = test["X"]
    y        = test["y"]
    labels   = test["labels"]

    rng     = np.random.default_rng(42)
    indices = rng.choice(len(X_scaled), size=min(n_flows, len(X_scaled)), replace=False)

    #  Direct mode 
    log.info("=== DIRECT MODE (scaled data → model) ===")
    X_sample = torch.tensor(X_scaled[indices], dtype=torch.float32)

    t0 = time.time()
    with torch.no_grad():
        scores = model.anomaly_score(X_sample).numpy()
    elapsed = time.time() - t0

    predictions = (scores > threshold).astype(int)
    y_sample    = y[indices]
    accuracy    = (predictions == y_sample).sum() / len(predictions)

    log.info("  Flows scored:  %d", len(predictions))
    log.info("  Anomalies:     %d", predictions.sum())
    log.info("  Accuracy:      %.4f", accuracy)
    log.info("  Time:          %.3fs", elapsed)
    log.info("  Throughput:    %.0f flows/sec", len(predictions) / elapsed)

    #  Roundtrip mode 
    log.info("")
    log.info("=== ROUNDTRIP MODE (raw → preprocess → model) ===")

    engine          = InferenceEngine()
    scaler          = joblib.load(ARTIFACTS_DIR / "scaler.joblib")
    log_transformed = set(meta.get("log_transformed", []))
    feature_names   = meta["feature_names"]

    X_raw = scaler.inverse_transform(X_scaled)
    for i, fname in enumerate(feature_names):
        if fname in log_transformed:
            X_raw[:, i] = np.expm1(X_raw[:, i])

    flows = [
        {fname: float(X_raw[idx, i]) for i, fname in enumerate(feature_names)}
        for idx in indices
    ]

    t0      = time.time()
    results = engine.predict_batch(flows)
    elapsed = time.time() - t0

    n_correct = sum(
        1 for result, idx in zip(results, indices)
        if (1 if result["is_anomaly"] else 0) == int(y[idx])
    )

    log.info("  Flows scored:  %d", len(results))
    log.info("  Anomalies:     %d", sum(1 for r in results if r["is_anomaly"]))
    log.info("  Accuracy:      %.4f", n_correct / len(results))
    log.info("  Time:          %.3fs", elapsed)
    log.info("  Throughput:    %.0f flows/sec", len(results) / elapsed)

    #  Sample predictions 
    log.info("")
    log.info("Sample predictions (direct mode):")
    log.info("%-12s %-20s %-10s %-6s", "Score", "Label", "Predicted", "OK?")
    log.info("-" * 52)
    for i in range(min(10, len(scores))):
        idx  = indices[i]
        pred = "ANOMALY" if predictions[i] else "benign"
        ok   = "Y" if predictions[i] == y[idx] else "N"
        log.info("%-12.6f %-20s %-10s %-6s", scores[i], str(labels[idx]), pred, ok)


#  Entry point 

def main() -> None:
    parser = argparse.ArgumentParser(description="NetShield Kafka Inference")
    parser.add_argument(
        "mode", choices=["consume", "produce", "test"],
        help="consume: run Kafka consumer | produce: send test flows | test: local test",
    )
    parser.add_argument("--n-flows",   type=int, default=1000, help="Flows to produce/test")
    parser.add_argument("--bootstrap", default="localhost:9092", help="Kafka bootstrap servers")
    args = parser.parse_args()

    cfg = KafkaConfig(bootstrap_servers=args.bootstrap)

    if args.mode == "consume":
        run_consumer(cfg)
    elif args.mode == "produce":
        run_producer(cfg, n_flows=args.n_flows)
    elif args.mode == "test":
        run_local_test(n_flows=args.n_flows)


if __name__ == "__main__":
    main()