import argparse

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import json
import mlflow
import mlflow.pytorch
import sys, os
import numpy as np
from sklearn.model_selection import train_test_split

from utils import get_project_paths, convert_numpy_types

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
mlflow.set_tracking_uri("http://localhost:5000")
from task_registry import task, main


@task("data:generate_movie_vector_oftest_clusters")
def make_movie_vector_oftest_clusters():
    # Загрузка данных
    with open("params.yaml", "r") as f:
        paths = get_project_paths()
        cluster = yaml.safe_load(f)["clustering"]
    X = np.load(paths["models_dir"] / "movie_content_vectors_train.npz")['vectors']

    # Кластеризация
    n_clusters = cluster['n_clusters']
    kmeans_movies = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    kmeans_movies.fit(X)

    # Сохраняем модель
    joblib.dump(kmeans_movies, 'models/movie_clusters.pkl')

    # Метрики

    sil_score = silhouette_score(X, kmeans_movies.labels_)

    # Логирование в MLflow
    with mlflow.start_run(run_name="movie_kmeans_clustering"):
        mlflow.log_param("n_clusters", n_clusters)

        # Метрики
        metrics = {
            "silhouette_score": float(silhouette_score(X, kmeans_movies.labels_)),
            "davies_bouldin_score": float(davies_bouldin_score(X, kmeans_movies.labels_)),
            "inertia": float(kmeans_movies.inertia_),  # внутренняя метрика KMeans
        }

        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(kmeans_movies, artifact_path="kmeans_model")

        # Можно дополнительно сохранить другие параметры
        with open("models/cluster_metrics.json", "w") as f:
            json.dump({"silhouette_score": sil_score}, f, default=convert_numpy_types)

        mlflow.log_artifact("models/cluster_metrics.json")

@task("data:generate_movie_vector_oftest_clusters_raw")
def make_movie_vector_oftest_clusters_raw():

    with open("params.yaml", "r") as f:
        paths = get_project_paths()
        cluster = yaml.safe_load(f)["clustering"]

    # Кластеризация
    model_movies_full_vectors_raw = np.load(paths["models_dir"] / "model_movies_full_vectors_raw.npz")['vectors']

    movie_indices = np.arange(model_movies_full_vectors_raw.shape[0])
    train_idx, test_idx = train_test_split(movie_indices, test_size=0.2, random_state=42)

    raw_vectors_train = model_movies_full_vectors_raw[train_idx]
    raw_vectors_test = model_movies_full_vectors_raw[test_idx]

    np.savez_compressed("models/movies_vectors_train_raw.npz", vectors=raw_vectors_train)
    np.savez_compressed("models/movies_vectors_test_raw.npz", vectors=raw_vectors_test)

    # Кластеризация
    n_clusters = cluster['n_clusters']
    kmeans_movies = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    kmeans_movies.fit(raw_vectors_train)

    # Сохраняем модель
    joblib.dump(kmeans_movies, 'models/movie_clusters_raw.pkl')

    # Логирование в MLflow
    with mlflow.start_run(run_name="movie_kmeans_clustering_raw"):
        mlflow.log_param("n_clusters", n_clusters)

        # Метрики
        metrics = {
            "silhouette_score": float(silhouette_score(raw_vectors_train, kmeans_movies.labels_)),
            "davies_bouldin_score": float(davies_bouldin_score(raw_vectors_train, kmeans_movies.labels_)),
            "inertia": float(kmeans_movies.inertia_),  # внутренняя метрика KMeans
        }

        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(kmeans_movies, artifact_path="kmeans_model")

        # Можно дополнительно сохранить другие параметры
        with open("models/cluster_metrics_raw.json", "w") as f:
            json.dump({"metrics": metrics}, f, default=convert_numpy_types)

        mlflow.log_artifact("models/cluster_metrics_raw.json")

    return kmeans_movies, raw_vectors_train, raw_vectors_test, movie_indices

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", help="Список задач для выполнения")
    args = parser.parse_args()

    if args.tasks:
        main(args.tasks)  # Здесь передаем задачи, которые указаны в командной строке