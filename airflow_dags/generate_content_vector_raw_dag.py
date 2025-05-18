import os
import sys
from datetime import datetime

from airflow.models import DAG
from airflow.operators.bash import BashOperator

# Add project root to Pythonpath
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# === DAG Configuration ===
with DAG(
    dag_id="dvc_generate_vectors_raw",
    start_date=datetime(2024, 1, 1),
    schedule=None,  # Or Cron expression
    catchup=False,
    tags=["recommender", "dvc", "mlflow"],
) as dag:

    dvc_generate_vectors_raw = BashOperator(
        task_id="dvc_generate_vectors_raw",
        bash_command="cd /home/saatarko/PycharmProjects/movies_rec_system && dvc repro generate_vectors_raw_stage",
    )

    dvc_generate_vectors_raw
