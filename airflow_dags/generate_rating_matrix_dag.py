import os
import sys
from datetime import datetime

from airflow.models import DAG
from airflow.operators.bash import BashOperator

# Add project root to Pythonpath
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# === DAG Configuration ===
with DAG(
    dag_id="generate_rating_matrix",
    start_date=datetime(2024, 1, 1),
    schedule=None,  # Or Cron expression
    catchup=False,
    tags=["recommender", "dvc", "mlflow", "user"],
) as dag:

    generate_rating_matrix = BashOperator(
        task_id="generate_rating_matrix",
        bash_command="cd /home/saatarko/PycharmProjects/movies_rec_system && dvc repro generate_rating_matrix_stage",
    )

    generate_rating_matrix
