"""
NetShield  Airflow Continuous Training DAG

Orchestrates the full CT (Continuous Training) pipeline:

    detect_drift - retrain_model - evaluate_model - ab_test - promote_or_rollback

Schedule: Runs daily (configurable). In production, drift detection
could also be triggered by a Kafka consumer detecting anomalous
score distributions.

Each task is a BashOperator calling the existing Python modules,
keeping the DAG lightweight and the logic in the application code.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.utils.trigger_rule import TriggerRule

import json
import os

# Path to the netshield project root (mounted into the container)
PROJECT_ROOT = os.environ.get("NETSHIELD_ROOT", "/opt/airflow/netshield")

default_args = {
    "owner": "netshield",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="netshield_continuous_training",
    default_args=default_args,
    description="NetShield CT pipeline: drift detection - retraining - evaluation - deployment",
    schedule="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["netshield", "mlops", "ct"],
) as dag:

    #  Task 1: Detect Drift 
    detect_drift = BashOperator(
        task_id="detect_drift",
        bash_command=f"""
            cd {PROJECT_ROOT} && \
            python -m src.monitoring.drift_detector test 2>&1 | tee /tmp/drift_result.log
        """,
    )

    #  Task 2: Check if Retraining Needed 
    def check_drift_result(**context):
        """Branch: if drift detected - retrain, otherwise - skip."""
        report_path = f"{PROJECT_ROOT}/artifacts/drift_reports/test_drift.json"
        try:
            with open(report_path) as f:
                result = json.load(f)
            if result.get("needs_retraining", False):
                return "retrain_model"
            else:
                return "no_drift_skip"
        except Exception as e:
            print(f"Error reading drift report: {e}")
            return "no_drift_skip"

    check_drift = BranchPythonOperator(
        task_id="check_drift",
        python_callable=check_drift_result,
    )

    #  Task 3a: Skip (No Drift) 
    no_drift_skip = BashOperator(
        task_id="no_drift_skip",
        bash_command='echo "No significant drift detected. Skipping retraining."',
    )

    #  Task 3b: Retrain Model 
    retrain_model = BashOperator(
        task_id="retrain_model",
        bash_command=f"""
            cd {PROJECT_ROOT} && \
            echo "Starting model retraining..." && \
            python -m src.model.train 2>&1 | tee /tmp/train_result.log && \
            echo "Training complete."
        """,
        execution_timeout=timedelta(hours=2),
    )

    #  Task 4: Evaluate New Model 
    evaluate_model = BashOperator(
        task_id="evaluate_model",
        bash_command=f"""
            cd {PROJECT_ROOT} && \
            echo "Evaluating new model on test set..." && \
            python -c "
import json, numpy as np, torch
from src.model.model import TransformerAutoencoder
from sklearn.metrics import roc_auc_score

with open('artifacts/feature_meta.json') as f:
    meta = json.load(f)
model = TransformerAutoencoder(n_features=meta['n_features'])
model.load_state_dict(torch.load('artifacts/best_model.pt', weights_only=True))
model.eval()

with open('artifacts/threshold.json') as f:
    threshold = json.load(f)['threshold']

test = np.load('data/splits/test.npz', allow_pickle=True)
X, y = test['X'], test['y']

# Score in batches
scores = []
with torch.no_grad():
    for i in range(0, len(X), 512):
        batch = torch.tensor(X[i:i+512], dtype=torch.float32)
        scores.append(model.anomaly_score(batch).numpy())
scores = np.concatenate(scores)

auc = roc_auc_score(y, scores)
preds = (scores > threshold).astype(int)
recall = preds[y == 1].mean()
fpr = preds[y == 0].mean()

result = {{'auc': float(auc), 'recall': float(recall), 'fpr': float(fpr)}}
with open('artifacts/eval_result.json', 'w') as f:
    json.dump(result, f, indent=2)
print(f'AUC: {{auc:.4f}}, Recall: {{recall:.4f}}, FPR: {{fpr:.4f}}')

# Gate: fail if AUC dropped below 0.80
assert auc > 0.80, f'Model AUC {{auc:.4f}} below minimum threshold 0.80'
print('Model passed quality gate.')
" 2>&1 | tee /tmp/eval_result.log
        """,
    )

    #  Task 5: A/B Test 
    ab_test = BashOperator(
        task_id="ab_test",
        bash_command=f"""
            cd {PROJECT_ROOT} && \
            echo "Running A/B test simulation..." && \
            python -m src.serving.ab_testing simulate 2>&1 | tee /tmp/ab_result.log && \
            echo "A/B test complete."
        """,
        execution_timeout=timedelta(minutes=30),
    )

    #  Task 6: Promote or Rollback 
    def promote_or_rollback(**context):
        """Check A/B test results and decide deployment action."""
        results_path = f"{PROJECT_ROOT}/artifacts/ab_test_results.json"
        try:
            with open(results_path) as f:
                results = json.load(f)

            decision = results["bayesian_test"]["decision"]
            prob = results["bayesian_test"]["prob_challenger_better"]

            if decision == "PROMOTE":
                print(f"PROMOTING new model (P={prob:.4f})")
                # In production: update model symlink, restart inference workers
                return "deploy_new_model"
            elif decision == "ROLLBACK":
                print(f"ROLLING BACK (P={prob:.4f})")
                return "rollback_model"
            else:
                print(f"INCONCLUSIVE (P={prob:.4f}), keeping current model")
                return "rollback_model"
        except Exception as e:
            print(f"Error reading A/B results: {e}")
            return "rollback_model"

    deployment_decision = BranchPythonOperator(
        task_id="deployment_decision",
        python_callable=promote_or_rollback,
    )

    #  Task 7a: Deploy New Model 
    deploy_new_model = BashOperator(
        task_id="deploy_new_model",
        bash_command=f"""
            cd {PROJECT_ROOT} && \
            echo "Deploying new model to production..." && \
            cp artifacts/best_model.pt artifacts/production_model.pt && \
            cp artifacts/threshold.json artifacts/production_threshold.json && \
            echo "Model deployed. Inference workers will pick up new model on next restart." && \
            echo "Deployment timestamp: $(date u +%Y%m%dT%H:%M:%SZ)" > artifacts/last_deployment.txt
        """,
    )

    #  Task 7b: Rollback 
    rollback_model = BashOperator(
        task_id="rollback_model",
        bash_command='echo "Keeping current production model. No changes deployed."',
    )

    #  Task 8: Notify 
    notify_complete = BashOperator(
        task_id="notify_complete",
        bash_command=f"""
            cd {PROJECT_ROOT} && \
            python -c "
import json
from datetime import datetime

summary = {{'timestamp': datetime.utcnow().isoformat()}}

# Drift results
try:
    with open('artifacts/drift_reports/test_drift.json') as f:
        drift = json.load(f)
    summary['drift_share'] = drift['drift_share']
    summary['needs_retraining'] = drift['needs_retraining']
except: pass

# Eval results
try:
    with open('artifacts/eval_result.json') as f:
        eval_r = json.load(f)
    summary['model_auc'] = eval_r['auc']
except: pass

# A/B results
try:
    with open('artifacts/ab_test_results.json') as f:
        ab = json.load(f)
    summary['ab_decision'] = ab['bayesian_test']['decision']
except: pass

with open('artifacts/ct_pipeline_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print('Pipeline summary:', json.dumps(summary, indent=2))
"
        """,
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    #  DAG Dependencies 
    # Main flow: drift - check - retrain - evaluate - A/B - deploy/rollback - notify
    detect_drift >> check_drift

    check_drift >> no_drift_skip >> notify_complete
    check_drift >> retrain_model >> evaluate_model >> ab_test >> deployment_decision

    deployment_decision >> deploy_new_model >> notify_complete
    deployment_decision >> rollback_model >> notify_complete