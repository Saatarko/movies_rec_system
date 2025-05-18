import argparse
import json
import os
import sys

import joblib
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from scipy.sparse import load_npz
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils import convert_numpy_types, get_project_paths

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
mlflow.set_tracking_uri("http://localhost:5000")
from task_registry import main, task


@task("data:generate_movie_vector_oftest_clusters")
def make_movie_vector_oftest_clusters():
    """
    Clustering function taking into account vector splitting for offline testing (on train/test).
    With the ability to run Airflow-> DVC and mlflow logging

    """
    # Data load
    with open("params.yaml", "r") as f:
        paths = get_project_paths()
        cluster = yaml.safe_load(f)["clustering"]
    X = np.load(paths["models_dir"] / "movie_content_vectors_train.npz")["vectors"]

    # Clastorization
    n_clusters = cluster["n_clusters"]
    kmeans_movies = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    kmeans_movies.fit(X)

    # We save the model
    joblib.dump(kmeans_movies, "models/movie_clusters.pkl")

    # Metrics

    sil_score = silhouette_score(X, kmeans_movies.labels_)

    # Logging in MLFLOW
    with mlflow.start_run(run_name="movie_kmeans_clustering"):
        mlflow.log_param("n_clusters", n_clusters)

        # Metrics
        metrics = {
            "silhouette_score": float(silhouette_score(X, kmeans_movies.labels_)),
            "davies_bouldin_score": float(
                davies_bouldin_score(X, kmeans_movies.labels_)
            ),
            "inertia": float(kmeans_movies.inertia_),  # Inner Metric Kmeans
        }

        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(kmeans_movies, artifact_path="kmeans_model")

        # You can additionally save other parameters
        with open("models/cluster_metrics.json", "w") as f:
            json.dump({"silhouette_score": sil_score}, f, default=convert_numpy_types)

        mlflow.log_artifact("models/cluster_metrics.json")


@task("data:generate_movie_vector_oftest_clusters_raw")
def make_movie_vector_oftest_clusters_raw():
    """
    Common content vector clustering function.
    With the ability to run Airflow-> DVC and mlflow logging
    """
    with open("params.yaml", "r") as f:
        paths = get_project_paths()
        cluster = yaml.safe_load(f)["clustering"]

    # Clastorization
    model_movies_full_vectors_raw = np.load(
        paths["models_dir"] / "model_movies_full_vectors_raw.npz"
    )["vectors"]

    movie_indices = np.arange(model_movies_full_vectors_raw.shape[0])
    train_idx, test_idx = train_test_split(
        movie_indices, test_size=0.2, random_state=42
    )

    raw_vectors_train = model_movies_full_vectors_raw[train_idx]
    raw_vectors_test = model_movies_full_vectors_raw[test_idx]

    np.savez_compressed(
        "models/movies_vectors_train_raw.npz", vectors=raw_vectors_train
    )
    np.savez_compressed("models/movies_vectors_test_raw.npz", vectors=raw_vectors_test)

    # Clastorization
    n_clusters = cluster["n_clusters"]
    kmeans_movies = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    kmeans_movies.fit(raw_vectors_train)

    # We save the model
    joblib.dump(kmeans_movies, "models/movie_clusters_raw.pkl")

    # Logging in MLFLOW
    with mlflow.start_run(run_name="movie_kmeans_clustering_raw"):
        mlflow.log_param("n_clusters", n_clusters)

        # Metrics
        metrics = {
            "silhouette_score": float(
                silhouette_score(raw_vectors_train, kmeans_movies.labels_)
            ),
            "davies_bouldin_score": float(
                davies_bouldin_score(raw_vectors_train, kmeans_movies.labels_)
            ),
            "inertia": float(kmeans_movies.inertia_),  # Inner Metric Kmeans
        }

        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(kmeans_movies, artifact_path="kmeans_model")

        # You can additionally save other parameters
        with open("models/cluster_metrics_raw.json", "w") as f:
            json.dump({"metrics": metrics}, f, default=convert_numpy_types)

        mlflow.log_artifact("models/cluster_metrics_raw.json")

    return kmeans_movies, raw_vectors_train, raw_vectors_test, movie_indices


@task("data:make_movie_vector_als_cluster")
def make_movie_vector_als_cluster():
    """
    Hybrid vector clustering function (content + als)
    With the ability to run Airflow->DVC and mlflow logging

    """
    with open("params.yaml", "r") as f:
        paths = get_project_paths()
        cluster = yaml.safe_load(f)["clustering"]

    # Clastorization
    hybrid_movie_vector_full = np.load(
        paths["models_dir"] / "hybrid_movie_vector_full.npz"
    )["vectors"]
    hybrid_movie_vector_full = hybrid_movie_vector_full.astype(np.float64)
    # Clastorization
    n_clusters = cluster["n_clusters"]
    kmeans_movies = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    kmeans_movies.fit(hybrid_movie_vector_full)

    # We save the model
    joblib.dump(kmeans_movies, "models/movie_and_als_clusters.pkl")

    # Logging in MLFLOW
    with mlflow.start_run(run_name="movie_and_als_clusters"):
        mlflow.log_param("n_clusters", n_clusters)

        # Metrics
        metrics = {
            "silhouette_score": float(
                silhouette_score(hybrid_movie_vector_full, kmeans_movies.labels_)
            ),
            "davies_bouldin_score": float(
                davies_bouldin_score(hybrid_movie_vector_full, kmeans_movies.labels_)
            ),
            "inertia": float(kmeans_movies.inertia_),  # Inner Metric Kmeans
        }

        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(kmeans_movies, artifact_path="kmeans_model")

        # You can additionally save other parameters
        with open("models/cluster_metrics_als.json", "w") as f:
            json.dump({"metrics": metrics}, f, default=convert_numpy_types)

        mlflow.log_artifact("models/cluster_metrics_als.json")

    return kmeans_movies


@task("data:user_vector_oftest_clusters")
def user_vector_oftest_clusters():
    """
    User vector clustering function taking into account vector splitting for offline testing (train/test).
    With the ability to run Airflow-> DVC and mlflow logging

    """
    with open("params.yaml", "r") as f:
        paths = get_project_paths()
        cluster = yaml.safe_load(f)["clustering"]

    # Clastorization
    user_content_vector = np.load(paths["models_dir"] / "user_content_vector.npz")[
        "vectors"
    ]

    user_indices = np.arange(user_content_vector.shape[0])
    train_idx, test_idx = train_test_split(user_indices, test_size=0.2, random_state=42)

    user_vectors_train = user_content_vector[train_idx]
    user_vectors_test = user_content_vector[test_idx]

    user_vectors_train = user_vectors_train.astype(np.float64)
    user_vectors_test = user_vectors_test.astype(np.float64)

    np.savez_compressed(
        paths["models_dir"] / "user_vectors_train.npz", vectors=user_vectors_train
    )
    np.savez_compressed(
        paths["models_dir"] / "user_vectors_test.npz", vectors=user_vectors_test
    )

    # Clastorization
    n_clusters = cluster["n_clusters"]
    kmeans_users = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    kmeans_users.fit(user_vectors_train)

    # We save the model
    joblib.dump(kmeans_users, paths["models_dir"] / "kmeans_users.pkl")

    # Logging in MLFLOW
    with mlflow.start_run(run_name="kmeans_users"):
        mlflow.log_param("n_clusters", n_clusters)

        # Metrics
        metrics = {
            "silhouette_score": float(
                silhouette_score(user_vectors_train, kmeans_users.labels_)
            ),
            "davies_bouldin_score": float(
                davies_bouldin_score(user_vectors_train, kmeans_users.labels_)
            ),
            "inertia": float(kmeans_users.inertia_),  # Inner Metric Kmeans
        }

        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(kmeans_users, artifact_path="kmeans_users")

        # You can additionally save other parameters
        with open(paths["models_dir"] / "cluster_metrics_user.json", "w") as f:
            json.dump({"metrics": metrics}, f, default=convert_numpy_types)

        mlflow.log_artifact(paths["models_dir"] / "cluster_metrics_user.json")

    return kmeans_users


@task("data:user_vector_full_clusters")
def user_vector_full_clusters():
    """
    Full user vector clustering function
    With the ability to run Airflow-> DVC and mlflow logging

    """
    with open("params.yaml", "r") as f:
        paths = get_project_paths()
        cluster = yaml.safe_load(f)["clustering"]

    # Clastorization
    user_content_vector = np.load(paths["models_dir"] / "user_content_vector.npz")[
        "vectors"
    ]

    # Clastorization
    n_clusters = cluster["n_clusters"]
    kmeans_users = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    kmeans_users.fit(user_content_vector)

    # We save the model
    joblib.dump(kmeans_users, paths["models_dir"] / "kmeans_full_users.pkl")

    # Logging in MLFLOW
    with mlflow.start_run(run_name="kmeans_full_users"):
        mlflow.log_param("n_clusters", n_clusters)

        # Metrics
        metrics = {
            "silhouette_score": float(
                silhouette_score(user_content_vector, kmeans_users.labels_)
            ),
            "davies_bouldin_score": float(
                davies_bouldin_score(user_content_vector, kmeans_users.labels_)
            ),
            "inertia": float(kmeans_users.inertia_),  # Inner Metric Kmeans
        }

        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(kmeans_users, artifact_path="kmeans_full_users")

        # You can additionally save other parameters
        with open(paths["models_dir"] / "cluster_metrics_user_full.json", "w") as f:
            json.dump({"metrics": metrics}, f, default=convert_numpy_types)

        mlflow.log_artifact(paths["models_dir"] / "cluster_metrics_user_full.json")

    return kmeans_users


@task("data:cluster_user_segment_vectors")
def cluster_user_segment_vectors():

    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)["cluster_user_vectors"]

    paths = get_project_paths()
    max_k = config["max_k"]
    random_state = config.get("random_state", 42)
    force_k = 18

    print("[tgdm] Загрузка пользовательских векторов...")
    encoded_sparse = load_npz(paths["processed_dir"] / "encoded_user_vectors.npz")
    encoded_matrix = encoded_sparse.toarray()

    scaler = StandardScaler()
    vectors_scaled = scaler.fit_transform(encoded_matrix)

    print("[tgdm] Запуск кластеризации...")
    metrics_table = []
    k_values = range(2, max_k + 1)

    with mlflow.start_run():
        for k in k_values:
            model = MiniBatchKMeans(
                n_clusters=k, random_state=random_state, batch_size=1024
            )
            labels = model.fit_predict(vectors_scaled)

            silhouette = silhouette_score(
                vectors_scaled, labels, sample_size=10000, random_state=random_state
            )
            ch_score = calinski_harabasz_score(vectors_scaled, labels)
            db_score = davies_bouldin_score(vectors_scaled, labels)
            inertia = model.inertia_

            metrics_table.append(
                {
                    "k": k,
                    "silhouette": silhouette,
                    "calinski_harabasz": ch_score,
                    "davies_bouldin": db_score,
                    "inertia": inertia,
                }
            )

        df_metrics = pd.DataFrame(metrics_table)

        # Definition of the best k by most criteria
        ranks = pd.DataFrame()
        ranks["k"] = df_metrics["k"]
        ranks["silhouette_rank"] = df_metrics["silhouette"].rank(ascending=False)
        ranks["ch_rank"] = df_metrics["calinski_harabasz"].rank(ascending=False)
        ranks["db_rank"] = df_metrics["davies_bouldin"].rank(ascending=True)
        ranks["inertia_rank"] = df_metrics["inertia"].rank(ascending=True)
        ranks["total_rank"] = ranks[
            ["silhouette_rank", "ch_rank", "db_rank", "inertia_rank"]
        ].sum(axis=1)

        # Final Best_k
        if force_k is not None:
            best_k = force_k
            print(f"[tgdm] Используется заданное force_k = {best_k}")
        else:
            best_k = int(ranks.sort_values("total_rank")["k"].iloc[0])
            print(f"[tgdm] Лучшее K по множеству метрик: {best_k}")

        final_model = MiniBatchKMeans(
            n_clusters=best_k, random_state=random_state, batch_size=1024
        )
        final_labels = final_model.fit_predict(vectors_scaled)

        # Preservation of model and artifacts
        model_path = paths["models_dir"] / "user_segment_cluster_model.pkl"
        scaler_path = paths["models_dir"] / "user_segment_vector_scaler.pkl"
        joblib.dump(final_model, model_path)
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(scaler_path))

        user_clusters = pd.DataFrame(
            {"user_id": np.arange(len(final_labels)), "cluster": final_labels}
        )
        cluster_csv_path = paths["processed_dir"] / "user_clusters.csv"
        user_clusters.to_csv(cluster_csv_path, index=False)
        mlflow.log_artifact(str(cluster_csv_path))

        # Table metrics
        metrics_csv_path = paths["models_dir"] / "user_cluster_metrics.csv"
        df_metrics.to_csv(metrics_csv_path, index=False)
        mlflow.log_artifact(str(metrics_csv_path))

        # Visualization
        fig, ax = plt.subplots(2, 2, figsize=(12, 10))
        ax[0, 0].plot(df_metrics["k"], df_metrics["inertia"], marker="o")
        ax[0, 0].set_title("Inertia")
        ax[0, 1].plot(df_metrics["k"], df_metrics["silhouette"], marker="o")
        ax[0, 1].set_title("Silhouette Score")
        ax[1, 0].plot(df_metrics["k"], df_metrics["calinski_harabasz"], marker="o")
        ax[1, 0].set_title("Calinski-Harabasz Score")
        ax[1, 1].plot(df_metrics["k"], df_metrics["davies_bouldin"], marker="o")
        ax[1, 1].set_title("Davies-Bouldin Score")

        for a in ax.flat:
            a.set_xlabel("k")

        fig.tight_layout()
        fig_path = paths["models_dir"] / "user_cluster_selection.png"
        fig.savefig(fig_path)
        mlflow.log_artifact(str(fig_path))

        mlflow.log_param("best_k", best_k)
        mlflow.log_metric(
            "best_silhouette",
            df_metrics.loc[df_metrics.k == best_k, "silhouette"].values[0],
        )
        print(f"[tgdm] Сегментация завершена. Лучшее k: {best_k}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", help="Список задач для выполнения")
    args = parser.parse_args()

    if args.tasks:
        main(args.tasks)  # Here we transmit the tasks indicated on the command line
