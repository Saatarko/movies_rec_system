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
    dag_id="user_autoencoder_and_rec_nn",
    start_date=datetime(2024, 1, 1),
    schedule=None,  # или cron выражение
    catchup=False,
    tags=["recommender", "dvc", "mlflow", 'user']
) as dag:

    user_autoencoder = BashOperator(
        task_id="user_autoencoder",
        bash_command="cd /home/saatarko/PycharmProjects/movies_rec_system && dvc repro user_autoencoder_stage",
        doc_md = "**Тренировка автоэнкодера пользователей**"
    )

    # rec_nn = BashOperator(
    #     task_id="rec_nn",
    #     bash_command="cd /home/saatarko/PycharmProjects/movies_rec_system && dvc repro train_nn_model_stage",
    #     doc_md="**Тренировка нейросети на взаимодействиях**"
    # )


    user_autoencoder