import os
import sys
from datetime import datetime

from airflow.models import DAG
from airflow.operators.bash import BashOperator

# Add project root to Pythonpath
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# === DAG Configuration ===
with DAG(
    dag_id="combine_content_als_vector",
    start_date=datetime(2024, 1, 1),
    schedule=None,  # Or Cron expression
    catchup=False,
    tags=["recommender", "dvc", "mlflow", "user"],
) as dag:

    combine_content_als_vector = BashOperator(
        task_id="combine_content_als_vector",
        bash_command="cd /home/saatarko/PycharmProjects/movies_rec_system && dvc repro combine_content_als_vector_stage",
    )

    combine_content_als_vector
