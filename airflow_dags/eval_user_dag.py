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
    dag_id="eval_user",
    start_date=datetime(2024, 1, 1),
    schedule=None,  # или cron выражение
    catchup=False,
    tags=["recommender", "dvc", "mlflow", 'user']
) as dag:

    eval_user = BashOperator(
        task_id="eval_user",
        bash_command="cd /home/saatarko/PycharmProjects/movies_rec_system && dvc repro eval_user_stage",
        doc_md = "**Фомирование вектора пользователей**"
    )


    eval_user