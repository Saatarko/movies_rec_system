from datetime import datetime
from airflow.models import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

import sys
import os

# Добавляем корень проекта в PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# === DAG конфигурация ===
with DAG(
    dag_id="generate_vectors_dvc_mlflow",
    start_date=datetime(2024, 1, 1),
    schedule=None,  # или cron выражение
    catchup=False,
    tags=["recommender", "dvc", "mlflow"]
) as dag:

    dvc_generate_vectors = BashOperator(
        task_id="generate_vectors_dvc",
        bash_command="cd /home/saatarko/PycharmProjects/movies_rec_system && dvc repro generate_vectors_stage"
    )

    dvc_generate_vectors