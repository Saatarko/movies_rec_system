from pathlib import Path

import joblib
import numpy as np
import yaml
import pandas as pd
import argparse
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib_venn import venn2
import streamlit as st

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