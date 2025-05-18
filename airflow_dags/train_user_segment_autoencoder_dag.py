import os
import sys
from datetime import datetime

from airflow.models import DAG
from airflow.operators.bash import BashOperator

# Add project root to Pythonpath
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# === DAG Configuration ===
with DAG(
    dag_id="train_user_segment_autoencoder",
    start_date=datetime(2024, 1, 1),
    schedule=None,  # Or Cron expression
    catchup=False,
    tags=["recommender", "dvc", "mlflow", "user"],
) as dag:

    train_user_segment_autoencoder = BashOperator(
        task_id="train_user_segment_autoencoder",
        bash_command="cd /home/saatarko/PycharmProjects/movies_rec_system && dvc repro train_user_segment_autoencoder_stage",
        doc_md="**Тренировка автоэнкодера пользователей**",
    )

    train_user_segment_autoencoder
