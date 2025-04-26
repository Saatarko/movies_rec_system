from datetime import datetime

import pandas as pd
import yaml
from airflow.models import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

import sys
import os

# Добавляем корень проекта в PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# === код автоэнкодера (можно импортировать, если модуль отдельно) ===
def train_autoencoder():
    from scripts.data_processing import generate_content_vector_for_offtest
    from scripts.train_autoencoder import content_vector_autoencoder  # импорт своей функции
    from scripts.train_autoencoder import eval_content_train_test_vectors

    # Путь к корню проекта
    PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

    # Загрузка данных
    genome_tags = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "raw", "genome-tags.csv"))
    genome_scores = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "raw", "genome-scores.csv"))

    # Передаем загруженные данные в функцию
    train_vectors, test_vectors = generate_content_vector_for_offtest(genome_scores, genome_tags)
    content_vector_autoencoder(train_vectors)

    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)["autoencoder"]

    eval_content_train_test_vectors( config["model_output_path"], train_vectors, test_vectors)

# === DAG конфигурация ===
with DAG(
    dag_id="train_movie_autoencoder_dvc_mlflow",
    start_date=datetime(2024, 1, 1),
    schedule=None,  # или cron выражение
    catchup=False,
    tags=["recommender", "dvc", "mlflow"]
) as dag:

    dvc_generate_vectors = BashOperator(
        task_id="generate_vectors_dvc",
        bash_command="cd /home/saatarko/PycharmProjects/movies_rec_system && dvc repro generate_vectors_stage"
    )

    run_autoencoder = PythonOperator(
        task_id="run_autoencoder",
        python_callable=train_autoencoder
    )

    dvc_generate_vectors >> run_autoencoder