import os
import sys
from datetime import datetime

from airflow.models import DAG
from airflow.operators.bash import BashOperator

# Add project root to Pythonpath
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# === DAG Configuration ===
with DAG(
    dag_id="segment_movies_stage",
    start_date=datetime(2024, 1, 1),
    schedule=None,  # Or Cron expression
    catchup=False,
    tags=["recommender", "dvc", "mlflow", "user"],
) as dag:

    segment_movies_stage = BashOperator(
        task_id="segment_movies_stage",
        bash_command="cd /home/saatarko/PycharmProjects/movies_rec_system && dvc repro segment_movies_stage",
        doc_md="**Сегментирование фильмов по жанрам и рейтинга с кластеризацией**",
    )

    segment_movies_stage
