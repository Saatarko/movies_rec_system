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
    dag_id="make_bridge_als",
    start_date=datetime(2024, 1, 1),
    schedule=None,  # или cron выражение
    catchup=False,
    tags=["recommender", "dvc", "mlflow", 'user']
) as dag:

    make_bridge_als = BashOperator(
        task_id="make_bridge_als",
        bash_command="cd /home/saatarko/PycharmProjects/movies_rec_system && dvc repro make_bridge_als_stage"
    )

    make_bridge_als