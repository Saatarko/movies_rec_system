from datetime import datetime
from airflow.models import DAG
from airflow.operators.bash import BashOperator

import sys
import os

# Добавляем корень проекта в PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# === DAG конфигурация ===
with DAG(
    dag_id="clusters_movies_vectors_dvc_raw",
    start_date=datetime(2024, 1, 1),
    schedule=None,  # или cron выражение
    catchup=False,
    tags=["recommender", "dvc", "mlflow"]
) as dag:

    clusters_movies_vectors_dvc_raw = BashOperator(
        task_id="clusters_movies_vectors_dvc",
        bash_command="cd /home/saatarko/PycharmProjects/movies_rec_system && dvc repro cluster_movies_stage_raw"
    )

    clusters_movies_vectors_dvc_raw