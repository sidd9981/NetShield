"""
NetShield -- BigQuery Sink

Reads scored anomaly results from the Kafka `anomaly-scores` topic
and streams them into BigQuery for dashboarding in Looker Studio.

Architecture:
    [anomaly-scores topic] -> Consumer -> batch -> BigQuery

Usage:
    # Install deps
    pip install kafka-python google-cloud-bigquery

    # Authenticate (one-time)
    gcloud auth application-default login

    # Run
    python -m src.serving.bigquery_sink

    # Or set project via env
    GCP_PROJECT=your-project-id python -m src.serving.bigquery_sink
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

#  Config 

GCP_PROJECT   = os.getenv("GCP_PROJECT", "your-project-id")   # override via env
BQ_DATASET    = "netshield"
BQ_TABLE      = "anomaly_events"
BQ_TABLE_ID   = f"{GCP_PROJECT}.{BQ_DATASET}.{BQ_TABLE}"

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
KAFKA_TOPIC     = "anomaly-scores"
KAFKA_GROUP     = "netshield-bq-sink"

BATCH_SIZE     = 500          # rows per BigQuery insert
BATCH_TIMEOUT  = 5.0          # flush after this many seconds even if batch isn't full


#  BigQuery schema 

TABLE_SCHEMA = [
    # Flow identity
    {"name": "event_time",     "type": "TIMESTAMP", "mode": "REQUIRED"},
    {"name": "flow_id",        "type": "STRING",    "mode": "NULLABLE"},
    {"name": "src_ip",         "type": "STRING",    "mode": "NULLABLE"},
    {"name": "dst_ip",         "type": "STRING",    "mode": "NULLABLE"},
    {"name": "src_port",       "type": "INTEGER",   "mode": "NULLABLE"},
    {"name": "dst_port",       "type": "INTEGER",   "mode": "NULLABLE"},
    {"name": "protocol",       "type": "STRING",    "mode": "NULLABLE"},
    # Scoring
    {"name": "anomaly_score",  "type": "FLOAT",     "mode": "REQUIRED"},
    {"name": "is_anomaly",     "type": "BOOLEAN",   "mode": "REQUIRED"},
    {"name": "threshold",      "type": "FLOAT",     "mode": "REQUIRED"},
    # Ground truth (present in test mode, null in production)
    {"name": "true_label",     "type": "STRING",    "mode": "NULLABLE"},
    {"name": "attack_type",    "type": "STRING",    "mode": "NULLABLE"},
    # Operational metadata
    {"name": "worker_id",      "type": "STRING",    "mode": "NULLABLE"},
    {"name": "processing_ms",  "type": "FLOAT",     "mode": "NULLABLE"},
    {"name": "ingested_at",    "type": "TIMESTAMP", "mode": "REQUIRED"},
]


def ensure_table(client) -> None:
    """Create the BigQuery dataset and table if they don't exist."""
    from google.cloud import bigquery
    from google.api_core.exceptions import Conflict

    # Dataset
    ds = bigquery.Dataset(f"{GCP_PROJECT}.{BQ_DATASET}")
    ds.location = "US"
    try:
        client.create_dataset(ds, timeout=30)
        log.info("Created dataset %s.%s", GCP_PROJECT, BQ_DATASET)
    except Conflict:
        log.info("Dataset %s already exists", BQ_DATASET)

    # Table
    schema = [
        bigquery.SchemaField(f["name"], f["type"], mode=f["mode"])
        for f in TABLE_SCHEMA
    ]
    table = bigquery.Table(BQ_TABLE_ID, schema=schema)

    # Partition by event_time for cheaper queries in Looker Studio
    table.time_partitioning = bigquery.TimePartitioning(
        type_=bigquery.TimePartitioningType.DAY,
        field="event_time",
    )
    table.clustering_fields = ["is_anomaly"]

    try:
        client.create_table(table, timeout=30)
        log.info("Created table %s", BQ_TABLE_ID)
    except Conflict:
        log.info("Table %s already exists", BQ_TABLE)


def kafka_msg_to_row(msg: dict) -> dict:
    """Convert a Kafka anomaly-scores message to a BigQuery row."""
    ts = msg.get("timestamp", time.time())
    event_dt = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    now_iso  = datetime.now(tz=timezone.utc).isoformat()

    return {
        "event_time":    event_dt,
        "flow_id":       msg.get("flow_id"),
        "src_ip":        msg.get("Src IP"),
        "dst_ip":        msg.get("Dst IP"),
        "src_port":      int(msg["Src Port"]) if msg.get("Src Port") is not None else None,
        "dst_port":      int(msg["Dst Port"]) if msg.get("Dst Port") is not None else None,
        "protocol":      str(msg["Protocol"]) if msg.get("Protocol") is not None else None,
        "anomaly_score": float(msg["anomaly_score"]),
        "is_anomaly":    bool(msg["is_anomaly"]),
        "threshold":     float(msg["threshold"]),
        "true_label":    msg.get("_label"),
        "attack_type":   msg.get("_attack_type"),     # set by mock producer if known
        "worker_id":     msg.get("worker_id"),
        "processing_ms": float(msg["processing_ms"]) if msg.get("processing_ms") is not None else None,
        "ingested_at":   now_iso,
    }


def flush_batch(client, rows: list) -> int:
    """Insert a batch into BigQuery. Returns number of rows inserted."""
    if not rows:
        return 0

    errors = client.insert_rows_json(BQ_TABLE_ID, rows)
    if errors:
        log.error("BigQuery insert errors: %s", errors[:3])
        return 0

    return len(rows)


def run_sink() -> None:
    """Main loop: consume from Kafka, batch, write to BigQuery."""
    try:
        from kafka import KafkaConsumer
    except ImportError:
        log.error("kafka-python not installed: pip install kafka-python")
        return

    try:
        from google.cloud import bigquery
    except ImportError:
        log.error("google-cloud-bigquery not installed: pip install google-cloud-bigquery")
        return

    client = bigquery.Client(project=GCP_PROJECT)
    ensure_table(client)

    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id=KAFKA_GROUP,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        max_poll_records=BATCH_SIZE,
        fetch_max_wait_ms=int(BATCH_TIMEOUT * 1000),
    )

    log.info("BigQuery sink started → %s", BQ_TABLE_ID)
    log.info("Consuming '%s' on %s ...", KAFKA_TOPIC, KAFKA_BOOTSTRAP)

    batch: list[dict] = []
    last_flush = time.time()
    total_rows = 0
    total_anomalies = 0

    try:
        while True:
            records = consumer.poll(
                timeout_ms=int(BATCH_TIMEOUT * 1000),
                max_records=BATCH_SIZE,
            )

            for tp, messages in records.items():
                for msg in messages:
                    try:
                        row = kafka_msg_to_row(msg.value)
                        batch.append(row)
                        if row["is_anomaly"]:
                            total_anomalies += 1
                    except Exception as e:
                        log.warning("Failed to parse message: %s", e)

            # Flush if batch is full or timeout elapsed
            elapsed = time.time() - last_flush
            if len(batch) >= BATCH_SIZE or (batch and elapsed >= BATCH_TIMEOUT):
                n = flush_batch(client, batch)
                total_rows += n
                log.info(
                    "Flushed %d rows to BigQuery (total: %d rows, %d anomalies)",
                    n, total_rows, total_anomalies,
                )
                batch.clear()
                last_flush = time.time()

    except KeyboardInterrupt:
        log.info("Shutting down ...")
        if batch:
            n = flush_batch(client, batch)
            total_rows += n
            log.info("Final flush: %d rows", n)
    finally:
        consumer.close()
        log.info("Done. Total: %d rows, %d anomalies.", total_rows, total_anomalies)


#  One-time setup helper 

def setup_only() -> None:
    """Just create the BQ table without consuming. Useful for first-run setup."""
    try:
        from google.cloud import bigquery
    except ImportError:
        log.error("pip install google-cloud-bigquery")
        return
    client = bigquery.Client(project=GCP_PROJECT)
    ensure_table(client)
    log.info("Setup complete. Table: %s", BQ_TABLE_ID)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        setup_only()
    else:
        run_sink()