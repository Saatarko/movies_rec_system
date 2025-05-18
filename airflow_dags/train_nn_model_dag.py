import os
import sys
from datetime import datetime

from airflow.models import DAG
from airflow.operators.bash import BashOperator

# Add project root to Pythonpath
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# === DAG Configuration ===
with DAG(
    dag_id="train_nn_model",
    start_date=datetime(2024, 1, 1),
    schedule=None,  # Or Cron expression
    catchup=False,
    tags=["recommender", "dvc", "mlflow", "user"],
) as dag:

    train_nn_model = BashOperator(
        task_id="train_nn_model",
        bash_command="cd /home/saatarko/PycharmProjects/movies_rec_system && dvc repro train_nn_model_stage",
        doc_md="**Фомирование вектора пользователей**",
    )

    train_nn_model
