import os
import sys
from datetime import datetime

from airflow.models import DAG
from airflow.operators.bash import BashOperator

# Add project root to Pythonpath
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# === DAG Configuration ===
with DAG(
    dag_id="eval_content_vector_dvc_mlflow",
    start_date=datetime(2024, 1, 1),
    schedule=None,  # Or Cron expression
    catchup=False,
    tags=["recommender", "dvc", "mlflow"],
) as dag:

    eval_content_vector = BashOperator(
        task_id="eval_content_vector_dvc",
        bash_command="cd /home/saatarko/PycharmProjects/movies_rec_system && dvc repro eval_content_train_test_vectors_stage",
    )

    eval_content_vector
