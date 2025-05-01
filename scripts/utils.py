import json
import os
from pathlib import Path

import gdown
import joblib
import mlflow
import numpy as np
import yaml
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib_venn import venn2
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity


def preprocess_popularity(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Возвращает DataFrame с movieId и средним рейтингом для каждого фильма.
    """
    popularity_df = ratings_df.groupby("movieId").agg(ave_rating=("rating", "mean"),
                                                      rating_count=("rating", "count")).reset_index()
    return popularity_df



def get_all_genres(movies_df: pd.DataFrame) -> list:
    """
    Возвращает уникальные жанры из всех фильмов.
    """
    genre_set = set()
    for genre_string in movies_df["genres"]:
        for g in genre_string.split("|"):
            genre_set.add(g.strip())
    return sorted(genre_set)

def get_project_paths():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    with open(PROJECT_ROOT / "params.yaml") as f:
        config = yaml.safe_load(f)

    paths = config["paths"]
    return {
        "project_root": PROJECT_ROOT,
        "raw_dir": PROJECT_ROOT / paths["raw_data_dir"],
        "processed_dir": PROJECT_ROOT / paths["processed_data_dir"],
        "models_dir": PROJECT_ROOT / paths["models_dir"],
        "scripts_dir": PROJECT_ROOT / paths["scripts"]
    }

def convert_numpy_types(obj):
    if isinstance(obj, np.generic):
        return obj.item()  # np.float32 -> float, np.int32 -> int и т.д.
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # np.array -> list
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def precision_at_k(list1, list2, k=5):
    set1 = set(list1[:k])
    set2 = set(list2[:k])
    return len(set1 & set2) / k


def jaccard_similarity(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    return len(set1 & set2) / len(set1 | set2)


def intersection_count(list1, list2):
    return len(set(list1) & set(list2))


# RBO Score
def rbo_score(list1, list2, p=0.9):
    """
    Вычисляет RBO (Rank Biased Overlap) для двух списков.
    p - параметр смещения (обычно в пределах от 0 до 1).
    """
    min_len = min(len(list1), len(list2))
    score = 0.0
    for i in range(min_len):
        score += (p ** i) * (list1[i] == list2[i])
    return score


# Top-N Пересечения
def top_n_intersections(list1, list2, top_n=10):
    return [len(set(list1[:i]) & set(list2[:i])) for i in range(1, top_n + 1)]


# Матрица совпадений для heatmap
def build_match_matrix(list1, list2):
    matrix = np.zeros((len(list1), len(list2)))
    for i, l1 in enumerate(list1):
        for j, l2 in enumerate(list2):
            if l1 == l2:
                matrix[i][j] = 1
    return matrix


def visualize_recommendations_df(df1, df2):
    list1 = df1['movie_id'].tolist()
    list2 = df2['movie_id'].tolist()

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

    # Выводим метрики в Streamlit
    st.write("Jaccard Similarity:", jaccard_similarity(list1, list2))
    st.write("Precision@5:", precision_at_k(list1, list2, k=5))
    st.write("Intersection Count:", intersection_count(list1, list2))
    st.write("RBO Score:", rbo_score(list1, list2))

    # Построение графиков
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Сравнение Рекомендательных Списков", fontsize=16)

    # --- 1. Диаграмма Венна ---
    venn2([set1, set2], set_labels=('Список 1', 'Список2'), ax=axs[0, 0])
    axs[0, 0].set_title("Диаграмма Венна")

    # --- 2. Barplot пересечений ---
    intersection_counts = top_n_intersections(list1, list2, top_n=10)
    axs[0, 1].bar(range(1, 11), intersection_counts)
    axs[0, 1].set_title("Пересечения в Top-N")
    axs[0, 1].set_xlabel("Top-N")
    axs[0, 1].set_ylabel("Кол-во общих фильмов")

    # --- 3. График RBO ---
    rbo_scores = [rbo_score(list1[:k], list2[:k]) for k in range(1, min(len(list1), len(list2)) + 1)]
    axs[1, 0].plot(range(1, len(rbo_scores) + 1), rbo_scores, marker='o')
    axs[1, 0].set_title("RBO по глубине")
    axs[1, 0].set_xlabel("Глубина списка")
    axs[1, 0].set_ylabel("RBO")

    # --- 4. Heatmap ---
    match_matrix = build_match_matrix(list1, list2)
    sns.heatmap(match_matrix, cmap='Blues', cbar=False, ax=axs[1, 1],
                xticklabels=list2, yticklabels=list1, linewidths=0.5, linecolor='gray')
    axs[1, 1].set_title("Heatmap Совпадений по Позициям")
    axs[1, 1].set_xlabel("Список 1")
    axs[1, 1].set_ylabel("Список 2")

    # Подгоняем под layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Отображаем графики в Streamlit
    st.pyplot(fig)


# Прогнозируем кластеры для тестовых данных
def predict_test_movie_clusters(movie_content_vectors_test):
    # Загрузка обученной модели
    kmeans_movies = joblib.load('models/movie_clusters.pkl')

    # Прогнозируем кластеры для тестовых данных
    test_movie_clusters = kmeans_movies.predict(movie_content_vectors_test)

    return test_movie_clusters



def predict_test_movie_clusters_raw(kmeans_model, movie_content_vectors_test):
    # Прогнозируем кластеры для тестовых данных
    test_movie_clusters = kmeans_model.predict(movie_content_vectors_test)
    return test_movie_clusters


def find_closest_movie_in_test(train_movie_idx, train_vectors, test_vectors, kmeans_model, test_movie_clusters):
    """
    Находит наиболее похожий фильм из test набора для фильма из train набора в рамках одного кластера.

    :param train_movie_idx: индекс фильма из train
    :param train_vectors: векторы фильмов из train
    :param test_vectors: векторы фильмов из test
    :param kmeans_model: обученная модель кластеризации (kmeans)
    :param test_movie_clusters: массив кластеров для всех test фильмов
    :return: индекс наиболее похожего фильма в test и значение сходства
    """
    train_movie_vector = train_vectors[train_movie_idx].reshape(1, -1)

    # Предсказываем кластер для train фильма
    train_movie_cluster = kmeans_model.predict(train_movie_vector)[0]

    # Выбираем только test фильмы из того же кластера
    same_cluster_indices = np.where(test_movie_clusters == train_movie_cluster)[0]

    if len(same_cluster_indices) == 0:
        raise ValueError(f"Нет фильмов в кластере {train_movie_cluster} среди test-фильмов!")

    test_vectors_same_cluster = test_vectors[same_cluster_indices]

    # Считаем сходство только с ними
    similarities = cosine_similarity(train_movie_vector, test_vectors_same_cluster).flatten()
    top_match_idx_in_cluster = np.argmax(similarities)

    # Преобразуем индекс внутри кластера обратно в глобальный индекс test
    top_match_idx = same_cluster_indices[top_match_idx_in_cluster]

    return top_match_idx, similarities[top_match_idx_in_cluster]


def get_random_train_sample_and_find_closest(
    movie_content_vectors_train,
    movie_content_vectors_test,
    n_samples=3
):
    from sklearn.metrics.pairwise import cosine_similarity

    n_train = len(movie_content_vectors_train)
    n_test = len(movie_content_vectors_test)

    print(f"Размеры массивов: train = {n_train}, test = {n_test}")

    # Берем случайные индексы из train
    random_train_indices = np.random.choice(n_train, n_samples, replace=False)
    print(f"Случайные индексы фильмов из train: {random_train_indices}")

    test_sample_indices = []

    for idx in random_train_indices:
        train_vector = movie_content_vectors_train[idx].reshape(1, -1)

        # Считаем сходства с каждым фильмом в тестовом наборе
        similarities = cosine_similarity(train_vector, movie_content_vectors_test)[0]

        # Находим индекс наиболее похожего фильма в тесте
        most_similar_idx = similarities.argmax()

        # Проверяем, что индекс не выходит за пределы
        if most_similar_idx < 0 or most_similar_idx >= n_test:
            print(f"Ошибка обработки фильма {idx}: индекс {most_similar_idx} выходит за пределы теста (size {n_test})")
            continue

        print(f"Фильм из train (индекс {idx}) — наиболее похожий фильм в test (индекс {most_similar_idx}), сходство: {similarities[most_similar_idx]:.4f}")

        test_sample_indices.append(most_similar_idx)

    return random_train_indices.tolist(), test_sample_indices



def create_train_movie_ids():
    project_root = Path(__file__).resolve().parent.parent  # или адаптируй путь под себя
    params_path = project_root / "params.yaml"
    with open(params_path, "r") as f:
        paths = get_project_paths()

    # Загружаем исходные фильмы
    movies_df = pd.read_csv(paths["raw_dir"] / "movies.csv")

    # Загружаем movie_content_vectors_train
    movie_content_vectors_train = np.load(paths["models_dir"] / "movie_content_vectors_train.npz")['vectors']

    # Оставляем только те фильмы, которые были в трейне
    train_movie_ids = movies_df['movieId'].values[:len(movie_content_vectors_train)]

    # Сохраняем
    np.save(paths["models_dir"] / "train_movie_ids.npy", train_movie_ids)

    print(f"train_movie_ids.npy сохранён в {paths['models_dir']}")


def predict_item_factors_batch(content_vectors, models_bridge):
    """
    Предсказывает item_factors для нескольких фильмов на основе их content_vectors.

    :param content_vectors: np.array формы (N, 64) — контентные векторы фильмов
    :param models_bridge: список из LGBM-моделей (или None для невалидных колонок)
    :return: np.array формы (N, 64) — предсказанные item_factors
    """
    content_vectors = np.asarray(content_vectors)  # (N, 64)
    n_samples = content_vectors.shape[0]
    n_factors = len(models_bridge)

    predictions = np.zeros((n_samples, n_factors))

    for i, model in enumerate(models_bridge):
        if model is not None:
            predictions[:, i] = model.predict(content_vectors)
        else:
            # Заполняем нулями колонку, если модель не обучалась
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
        "num_epochs": len(train_losses)
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
    Преобразует два списка в словарь {movie_id: rating}
    """
    if len(movie_ids) != len(ratings):
        raise ValueError("Длина списков movie_ids и ratings должна совпадать")

    return {int(mid): float(r) for mid, r in zip(movie_ids, ratings)}