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
    dag_id="train_nn_embedding_model",
    start_date=datetime(2024, 1, 1),
    schedule=None,  # или cron выражение
    catchup=False,
    tags=["recommender", "dvc", "mlflow", 'user']
) as dag:

    train_nn_embedding_model = BashOperator(
        task_id="train_nn_embedding_model",
        bash_command="cd /home/saatarko/PycharmProjects/movies_rec_system && dvc repro train_embedding_nn_stage",
        doc_md = "**Обучение нейросети с получением эмбеддингов**"
    )


    train_nn_embedding_model