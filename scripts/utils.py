import json
import os
from pathlib import Path
from typing import List

import gdown
import joblib
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import yaml
from matplotlib import pyplot as plt
from matplotlib_venn import venn2
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.python.ops.clustering_ops import KMeans


def preprocess_popularity(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with movieId and average rating for each movie.
    """
    popularity_df = (
        ratings_df.groupby("movieId")
        .agg(ave_rating=("rating", "mean"), rating_count=("rating", "count"))
        .reset_index()
    )
    return popularity_df


def get_all_genres(movies_df: pd.DataFrame) -> list:
    """
    Brings back unique genres from all movies.
    """
    genre_set = set()
    for genre_string in movies_df["genres"]:
        for g in genre_string.split("|"):
            genre_set.add(g.strip())
    return sorted(genre_set)


def get_project_paths():
    """
    Function to get paths for different data
    """
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    with open(PROJECT_ROOT / "params.yaml") as f:
        config = yaml.safe_load(f)

    paths = config["paths"]
    return {
        "project_root": PROJECT_ROOT,
        "raw_dir": PROJECT_ROOT / paths["raw_data_dir"],
        "processed_dir": PROJECT_ROOT / paths["processed_data_dir"],
        "models_dir": PROJECT_ROOT / paths["models_dir"],
        "scripts_dir": PROJECT_ROOT / paths["scripts"],
    }


def convert_numpy_types(obj: any) -> float | int | List:
    """
    Function to get paths for different data.
    :returns obj in mew format
    """
    if isinstance(obj, np.generic):
        return obj.item()  # np.float32 -> float, np.int32 -> int, etc.
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # NP.array -> List
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def precision_at_k(list1: List, list2: List, k: int = 5) -> float:
    """
    Precision calculation function
    :param: list1/list2 - list of recommendation
    :returns precision
    """
    set1 = set(list1[:k])
    set2 = set(list2[:k])
    return len(set1 & set2) / k


def jaccard_similarity(list1: List, list2: List) -> float:
    """
    jaccard_similarity calculation function
    :param: list1/list2 - list of recommendation
    :returns jaccard_similarity
    """
    set1 = set(list1)
    set2 = set(list2)
    return len(set1 & set2) / len(set1 | set2)


def intersection_count(list1, list2):
    return len(set(list1) & set(list2))


# Rbo Score
def rbo_score(list1: List, list2: List, p: int = 0.9):
    """
    Calculates RBO (Rank Biased Overlap) for two lists.
    p is the bias parameter (usually between 0 and 1).
    """
    min_len = min(len(list1), len(list2))
    score = 0.0
    for i in range(min_len):
        score += (p**i) * (list1[i] == list2[i])
    return score


# TOP-N intersection
def top_n_intersections(list1: List, list2: List, top_n: int = 10) -> List[int]:
    """
    intersections calculation function
    :param: list1/list2 - list of recommendation
    :returns intersections
    """
    return [len(set(list1[:i]) & set(list2[:i])) for i in range(1, top_n + 1)]


# Coincidence matrix for Heatmap
def build_match_matrix(list1: List, list2: List) -> np.ndarray:
    """
     match_matrix calculation function
    :param: list1/list2 - list of recommendation
    :returns matrix
    """
    matrix = np.zeros((len(list1), len(list2)))
    for i, l1 in enumerate(list1):
        for j, l2 in enumerate(list2):
            if l1 == l2:
                matrix[i][j] = 1
    return matrix


def visualize_recommendations_df(df1: np.ndarray, df2: np.ndarray):
    """
    function visualise two recommendation
    :param: list1/list2 - list of recommendation
    :returns matrix
    """
    list1 = df1["movie_id"].tolist()
    list2 = df2["movie_id"].tolist()

    set1 = set(list1)
    set2 = set(list2)

    # --- 1. Barplot Top-N Intersections ---
    def top_n_intersections(l1, l2, top_n=20):
        intersection_counts = []
        for n in range(1, top_n + 1):
            top1 = set(l1[:n])
            top2 = set(l2[:n])
            intersection_counts.append(len(top1 & top2))
        return intersection_counts

    # --- 2. Jaccard Similarity ---
    def jaccard_similarity(list1, list2):
        set1 = set(list1)
        set2 = set(list2)
        return len(set1 & set2) / len(set1 | set2)

    # --- 3. Rank Biased Overlap ---
    def rbo_score(list1, list2, p=0.9):
        score = 0.0
        depth = min(len(list1), len(list2))
        for d in range(1, depth + 1):
            set1 = set(list1[:d])
            set2 = set(list2[:d])
            overlap = len(set1 & set2)
            score += overlap / d * p ** (d - 1)
        return (1 - p) * score

    # --- 4. Heatmap of Position Matches ---
    def build_match_matrix(list1, list2):
        matrix = np.zeros((len(list1), len(list2)))
        for i, id1 in enumerate(list1):
            for j, id2 in enumerate(list2):
                if id1 == id2:
                    matrix[i, j] = 1
        return matrix

    # Bring metrics to Streamlit
    st.write("Jaccard Similarity:", jaccard_similarity(list1, list2))
    st.write("Precision@5:", precision_at_k(list1, list2, k=5))
    st.write("Intersection Count:", intersection_count(list1, list2))
    st.write("RBO Score:", rbo_score(list1, list2))

    # Construction of graphs
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Сравнение Рекомендательных Списков", fontsize=16)

    # --- 1. Venna diagram ---
    venn2([set1, set2], set_labels=("Список 1", "Список2"), ax=axs[0, 0])
    axs[0, 0].set_title("Диаграмма Венна")

    # --- 2. Barplot intersections ---
    intersection_counts = top_n_intersections(list1, list2, top_n=10)
    axs[0, 1].bar(range(1, 11), intersection_counts)
    axs[0, 1].set_title("Пересечения в Top-N")
    axs[0, 1].set_xlabel("Top-N")
    axs[0, 1].set_ylabel("Кол-во общих фильмов")

    # --- 3. Graph of Rbo ---
    rbo_scores = [
        rbo_score(list1[:k], list2[:k])
        for k in range(1, min(len(list1), len(list2)) + 1)
    ]
    axs[1, 0].plot(range(1, len(rbo_scores) + 1), rbo_scores, marker="o")
    axs[1, 0].set_title("RBO по глубине")
    axs[1, 0].set_xlabel("Глубина списка")
    axs[1, 0].set_ylabel("RBO")

    # --- 4. Heatmap ---
    match_matrix = build_match_matrix(list1, list2)
    sns.heatmap(
        match_matrix,
        cmap="Blues",
        cbar=False,
        ax=axs[1, 1],
        xticklabels=list2,
        yticklabels=list1,
        linewidths=0.5,
        linecolor="gray",
    )
    axs[1, 1].set_title("Heatmap Совпадений по Позициям")
    axs[1, 1].set_xlabel("Список 1")
    axs[1, 1].set_ylabel("Список 2")

    # We drive to Layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # We display graphs in Streamlit
    st.pyplot(fig)


# We predict clusters for test data
def predict_test_movie_clusters(movie_content_vectors_test) -> List[int]:
    """
    function predict clusters
    :param: list1/list2 - list of recommendation
    :returns List of new clusters
    """
    # Loading a trained model
    kmeans_movies = joblib.load("models/movie_clusters.pkl")

    # We predict clusters for test data
    test_movie_clusters = kmeans_movies.predict(movie_content_vectors_test)

    return test_movie_clusters


def predict_test_movie_clusters_raw(
    kmeans_model: KMeans, movie_content_vectors_test
) -> List[int]:
    """
    function predict clusters
    :returns List of new clusters
    """
    # We predict clusters for test data
    test_movie_clusters = kmeans_model.predict(movie_content_vectors_test)
    return test_movie_clusters


def find_closest_movie_in_test(
    train_movie_idx, train_vectors, test_vectors, kmeans_model, test_movie_clusters
):
    """
    Finds the most similar movie from the test set to a movie from the train set within a single cluster.

    :param train_movie_idx: index of the movie from train
    :param train_vectors: vectors of movies from train
    :param test_vectors: vectors of movies from test
    :param kmeans_model: trained clustering model (kmeans)
    :param test_movie_clusters: array of clusters for all test movies
    :return: index of the most similar movie in test and similarity value
    """
    train_movie_vector = train_vectors[train_movie_idx].reshape(1, -1)

    # We predict a cluster for the film film
    train_movie_cluster = kmeans_model.predict(train_movie_vector)[0]

    # Choose only test films from the same cluster
    same_cluster_indices = np.where(test_movie_clusters == train_movie_cluster)[0]

    if len(same_cluster_indices) == 0:
        raise ValueError(
            f"Нет фильмов в кластере {train_movie_cluster} среди test-фильмов!"
        )

    test_vectors_same_cluster = test_vectors[same_cluster_indices]

    # We consider the similarity only with them
    similarities = cosine_similarity(
        train_movie_vector, test_vectors_same_cluster
    ).flatten()
    top_match_idx_in_cluster = np.argmax(similarities)

    # We convert the index inside the cluster back to the global Test index
    top_match_idx = same_cluster_indices[top_match_idx_in_cluster]

    return top_match_idx, similarities[top_match_idx_in_cluster]


def get_random_train_sample_and_find_closest(
    movie_content_vectors_train, movie_content_vectors_test, n_samples=3
):
    from sklearn.metrics.pairwise import cosine_similarity

    n_train = len(movie_content_vectors_train)
    n_test = len(movie_content_vectors_test)

    print(f"Размеры массивов: train = {n_train}, test = {n_test}")

    # We take random indexes from Train
    random_train_indices = np.random.choice(n_train, n_samples, replace=False)
    print(f"Случайные индексы фильмов из train: {random_train_indices}")

    test_sample_indices = []

    for idx in random_train_indices:
        train_vector = movie_content_vectors_train[idx].reshape(1, -1)

        # We consider similarities with each film in a test set
        similarities = cosine_similarity(train_vector, movie_content_vectors_test)[0]

        # Find the index of the most similar film in the test
        most_similar_idx = similarities.argmax()

        # We check that the index does not go beyond
        if most_similar_idx < 0 or most_similar_idx >= n_test:
            print(
                f"Ошибка обработки фильма {idx}: индекс {most_similar_idx} выходит за пределы теста (size {n_test})"
            )
            continue

        print(
            f"Фильм из train (индекс {idx}) — наиболее похожий фильм в test (индекс {most_similar_idx}), сходство: {similarities[most_similar_idx]:.4f}"
        )

        test_sample_indices.append(most_similar_idx)

    return random_train_indices.tolist(), test_sample_indices


def create_train_movie_ids():
    project_root = Path(__file__).resolve().parent.parent  # or adapt the path for yourself
    params_path = project_root / "params.yaml"
    with open(params_path, "r") as f:
        paths = get_project_paths()

    # We download the source films
    movies_df = pd.read_csv(paths["raw_dir"] / "movies.csv")

    # We load movie_content_Vectors_train
    movie_content_vectors_train = np.load(
        paths["models_dir"] / "movie_content_vectors_train.npz"
    )["vectors"]

    # We leave only those films that were in the traine
    train_movie_ids = movies_df["movieId"].values[: len(movie_content_vectors_train)]

    # We save
    np.save(paths["models_dir"] / "train_movie_ids.npy", train_movie_ids)

    print(f"train_movie_ids.npy сохранён в {paths['models_dir']}")


def predict_item_factors_batch(content_vectors, models_bridge):
    """
    Predicts item_factors for multiple movies based on their content_vectors.

    :param content_vectors: np.array of shape (N, 64) — the movie content vectors
    :param models_bridge: list of LGBM models (or None for invalid columns)
    :return: np.array of shape (N, 64) — the predicted item_factors
    """
    content_vectors = np.asarray(content_vectors)  # (N, 64)
    n_samples = content_vectors.shape[0]
    n_factors = len(models_bridge)

    predictions = np.zeros((n_samples, n_factors))

    for i, model in enumerate(models_bridge):
        if model is not None:
            predictions[:, i] = model.predict(content_vectors)
        else:
            # Fill the column with zero if the model has not learned
            predictions[:, i] = 0.0

    return predictions


def load_vectors_npz(path: Path, key: str = "vectors") -> np.ndarray:
    return np.load(path)[key]


def log_training_metrics(train_losses, val_rmses):
    for epoch, (train_loss, val_rmse) in enumerate(zip(train_losses, val_rmses)):
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_rmse", val_rmse, step=epoch)


def save_model_metrics(metrics_path: Path, train_losses, val_rmses, best_rmse):
    metrics = {
        "final_train_loss": train_losses[-1],
        "final_val_rmse": val_rmses[-1],
        "best_val_rmse": best_rmse,
        "num_epochs": len(train_losses),
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)


def download_if_missing(file_path, file_id):
    file_path = str(file_path)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if not os.path.exists(file_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"⏬ Downloading {file_path} from Google Drive...")
        gdown.download(url, file_path, quiet=False)
    else:
        print(f"✅ {file_path} already exists.")


def build_user_ratings_dict(movie_ids, ratings):
    """
    Converts two lists into a dictionary {movie_id: rating}
    """
    if len(movie_ids) != len(ratings):
        raise ValueError("Длина списков movie_ids и ratings должна совпадать")

    return {int(mid): float(r) for mid, r in zip(movie_ids, ratings)}


# The function of adding data to the promotion/lock table
def update_importance_scores(
    importance_df: pd.DataFrame,
    movie_ids_to_promote: List[int] = None,
    movie_ids_to_block: List[int] = None,
) -> pd.DataFrame:
    """
    Updates importance_df based on the list of movies to promote and block.
    Returns importance_df (a dataframe with blocked movies)
    """

    promote_df = pd.DataFrame(
        {"movieId": movie_ids_to_promote or [], "importance_score": 1}
    )

    block_df = pd.DataFrame(
        {"movieId": movie_ids_to_block or [], "importance_score": -1}
    )

    updated = pd.concat([importance_df, promote_df, block_df], ignore_index=True)
    importance_df = updated.drop_duplicates(
        "movieId", keep="last"
    )  # We save the last installation

    return importance_df
