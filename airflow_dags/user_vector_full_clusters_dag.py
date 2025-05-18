import os
import sys
from datetime import datetime

from airflow.models import DAG
from airflow.operators.bash import BashOperator

# Add project root to Pythonpath
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# === DAG Configuration ===
with DAG(
    dag_id="user_vector_full_clusters",
    start_date=datetime(2024, 1, 1),
    schedule=None,  # Or Cron expression
    catchup=False,
    tags=["recommender", "dvc", "mlflow", "user"],
) as dag:

    user_vector_full_clusters = BashOperator(
        task_id="user_vector_full_clusters",
        bash_command="cd /home/saatarko/PycharmProjects/movies_rec_system && dvc repro user_vector_full_clusters_stage",
        doc_md="**Кластеризация полного вектора пользователей**",
    )

    user_vector_full_clusters
