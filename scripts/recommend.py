from collections import defaultdict
from itertools import permutations
from pathlib import Path
from typing import Optional, List, Any, Dict

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import yaml
from lightgbm import LGBMModel
from pandas import DataFrame
from scipy.sparse import load_npz, csr_matrix
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from scripts.train_autoencoder import MovieAutoencoder, RatingPredictor, Autoencoder
from scripts.utils import preprocess_popularity, get_project_paths, predict_test_movie_clusters, \
    predict_test_movie_clusters_raw, predict_item_factors_batch, load_vectors_npz, download_if_missing, \
    build_user_ratings_dict, update_importance_scores

project_root = Path(__file__).resolve().parent.parent  # Выходим из scripts/
params_path = project_root / "params.yaml"
with open(params_path, "r") as f:
    paths = get_project_paths()



def get_top_movies(selected_genres: List):
    """
    Функция получения популярности фильмов
    Аргументы:
    selected_genres - жанры для поиска популярности фильмов

    """
    project_root = Path(__file__).resolve().parent.parent  # Выходим из scripts/
    params_path = project_root / "params.yaml"
    with open(params_path, "r") as f:
        paths = get_project_paths()

    ratings = pd.read_csv(paths["raw_dir"] / "ratings.csv")
    movies = pd.read_csv(paths["raw_dir"] / "movies.csv")

    popularity_df = preprocess_popularity(ratings)

    if selected_genres:
        recommended_movies = recommend_top_movies_by_genres(*selected_genres, movies_df=movies,
                                                            popularity_df=popularity_df, top_n=10, min_votes=50)
        st.write("Топ 10 фильмов по жанрам:")
        st.dataframe(recommended_movies)  # Выводим DataFrame на экран
    else:
        st.warning("Пожалуйста, выберите хотя бы один жанр.")


def recommend_top_movies_by_genres(
        *genres: List,
        movies_df: pd.DataFrame,
        popularity_df: pd.DataFrame,
        top_n: int =10,
        min_votes: int =10)-> DataFrame:
    """
    Рекомендует топ-N фильмов по заданным жанрам, используя взвешенный рейтинг.

    Аргументы:
        *genres: Жанры (один или несколько), как строки.
        movies_df: датафрейм с фильмами и колонками movieId, title, genres.
        popularity_df: датафрейм с ave_rating и rating_count по movieId.
        top_n: сколько фильмов возвращать.
        min_votes: параметр "надежности" рейтинга (m в формуле).
    Возвращает рекомендации
    """
    # Убедимся, что все элементы в genres являются строками и приводим их к нижнему регистру
    genres = [str(g).lower() for g in genres if isinstance(g, str)]

    # Если жанров не выбрано, выбрасываем ошибку
    if not genres:
        raise ValueError("Список жанров не может быть пустым")

    # Объединение таблиц
    merged = pd.merge(movies_df, popularity_df, on="movieId")

    # Средний рейтинг по всей базе
    C = merged["ave_rating"].mean()

    # Фильтрация по жанрам (в порядке убывания количества совпадений)
    merged["genre_match"] = merged["genres"].str.lower().apply(
        lambda g: sum(genre in g for genre in genres)
    )

    merged = merged[merged["genre_match"] > 0]

    if merged.empty:
        print("⚠️ Не найдено фильмов с точным совпадением жанров. Показываем ближайшие совпадения.")
        merged = pd.merge(movies_df, popularity_df, on="movieId")
        merged["genre_match"] = merged["genres"].str.lower().apply(
            lambda g: sum(genre in g for genre in genres)
        )
        merged = merged.sort_values(by="genre_match", ascending=False).head(top_n)

    # Вычисляем взвешенный рейтинг
    merged["weighted_rating"] = (
        (merged["rating_count"] / (merged["rating_count"] + min_votes)) * merged["ave_rating"] +
        (min_votes / (merged["rating_count"] + min_votes)) * C
    )

    # Финальный топ
    top_movies = merged.sort_values(by="weighted_rating", ascending=False).head(top_n)

    return top_movies[["title", "genres", "ave_rating", "rating_count", "weighted_rating"]]


def recommend_movies_by_cluster_filtered(
        movie_ids:List,
        movie_content_vectors: np.ndarray,
        test:bool=False,
        top_n:int=50
)-> DataFrame | tuple[Any, list[dict[str, set[Any] | Any]]]:
    """
    Рекомендует топ-N фильмов по списку id просмотренных фильмов.

    Аргументы:
       movie_ids Список id фильмов.
       movie_content_vectors: Контент вектор
       test: флаг. Если False - передается train вектор, если True - test.
       top_n: сколько фильмов возвращать.
    Возвращает рекомендации
    """

    project_root = Path(__file__).resolve().parent.parent  # Выходим из scripts/
    params_path = project_root / "params.yaml"
    with open(params_path, "r") as f:
        paths = get_project_paths()

    kmeans_model = joblib.load(paths["models_dir"] / 'movie_clusters.pkl')
    movies_df = pd.read_csv(paths["raw_dir"] / "movies.csv")

    if test is False:
        cluster_labels = kmeans_model.labels_
    else:
        test_movie_clusters = predict_test_movie_clusters(movie_content_vectors)
        cluster_labels = np.concatenate([kmeans_model.labels_, test_movie_clusters])

    all_recommendations = []
    importance_df = pd.DataFrame(columns=['movieId', 'importance_score'])
    recommendation_info = []  # Создаём пустой список для хранения данных
    for movie_id in movie_ids:

        original_movie_row = movies_df[movies_df['movieId'] == movie_id]
        original_movie_title = original_movie_row['title'].values[0]
        original_movie_genres = original_movie_row['genres'].values[0]
        original_genres_set = set(original_movie_genres.split('|'))

        # Добавляем информацию в список вместо print
        recommendation_info.append({
            'movie_id': movie_id,
            'title': original_movie_title,
            'genres': original_genres_set
        })



        try:
            movie_idx = movies_df[movies_df['movieId'] == movie_id].index[0]
            movie_title = movies_df.iloc[movie_idx]['title']
            movie_genres = set(movies_df.iloc[movie_idx]['genres'].split('|'))

            cluster_label = cluster_labels[movie_idx]
            same_cluster_indices = np.where(cluster_labels == cluster_label)[0]
            same_cluster_indices = same_cluster_indices[same_cluster_indices != movie_idx]

            if len(same_cluster_indices) == 0:
                continue

            # Используем косинусное сходство для расчета схожести
            similarities = cosine_similarity(
                movie_content_vectors[movie_idx].reshape(1, -1),
                movie_content_vectors[same_cluster_indices]
            )[0]

            for idx, score in zip(same_cluster_indices, similarities):
                title = movies_df.iloc[idx]['title']
                genres = movies_df.iloc[idx]['genres']
                genres_set = set(genres.split('|'))

                if pd.isna(title) or pd.isna(genres):
                    continue

                if not movie_genres.intersection(genres_set):
                    score = 0  # Жестко, но ок для фильтрации

                all_recommendations.append({
                    'movie_id': movies_df.iloc[idx]['movieId'],
                    'title': title,
                    'genres': genres,
                    'similarity_score': score
                })
        except IndexError:
            continue

    all_recommendations_df = pd.DataFrame(all_recommendations)
    if all_recommendations_df.empty:
        return all_recommendations_df

    all_recommendations_df = all_recommendations_df.groupby(['movie_id', 'title', 'genres']).agg(
        {'similarity_score': 'sum'}).reset_index()

    drop_film = importance_df[importance_df['importance_score'] == -1]['movieId']
    drop_film = drop_film.astype(int)

    new_recs = all_recommendations_df[~all_recommendations_df['movie_id'].isin(drop_film)]
    all_recommendations_df = new_recs.sort_values('similarity_score', ascending=False).head(top_n)

    weights = np.linspace(20, 1, len(all_recommendations_df)).round()
    all_recommendations_df['weight'] = weights

    temp_list = importance_df[importance_df['importance_score'] == 1]
    temp_list = temp_list.rename(columns={'movie_id': 'movieId'})

    temp_list = temp_list.merge(
        movies_df[['movieId', 'title', 'genres']],
        on='movieId',
        how='left'
    ).dropna(subset=['title', 'genres'])

    if len(temp_list) != 0:
        print('Горячие новинки')
        print('-' * 50)
        print(temp_list[['movieId', 'title', 'genres']])

    return all_recommendations_df, recommendation_info


def recommend_movies_with_block(
        movie_ids:List,
        movie_content_vectors: np.ndarray,
        importance_df: pd.DataFrame,
        test:bool=False,
        top_n:int=50
)-> DataFrame | tuple[Any, list[dict[str, set[Any] | Any]]]:
    """
    Рекомендует топ-N фильмов по списку id просмотренных фильмов.

    Аргументы:
       movie_ids Список id фильмов.
       movie_content_vectors: Контент вектор
       test: флаг. Если False - передается train вектор, если True - test.
       top_n: сколько фильмов возвращать.
    Возвращает рекомендации
    """

    project_root = Path(__file__).resolve().parent.parent  # Выходим из scripts/
    params_path = project_root / "params.yaml"
    with open(params_path, "r") as f:
        paths = get_project_paths()

    kmeans_model = joblib.load(paths["models_dir"] / 'movie_clusters.pkl')
    movies_df = pd.read_csv(paths["raw_dir"] / "movies.csv")

    if test is False:
        cluster_labels = kmeans_model.labels_
    else:
        test_movie_clusters = predict_test_movie_clusters(movie_content_vectors)
        cluster_labels = np.concatenate([kmeans_model.labels_, test_movie_clusters])

    all_recommendations = []

    recommendation_info = []  # Создаём пустой список для хранения данных
    for movie_id in movie_ids:

        original_movie_row = movies_df[movies_df['movieId'] == movie_id]
        original_movie_title = original_movie_row['title'].values[0]
        original_movie_genres = original_movie_row['genres'].values[0]
        original_genres_set = set(original_movie_genres.split('|'))

        # Добавляем информацию в список вместо print
        recommendation_info.append({
            'movie_id': movie_id,
            'title': original_movie_title,
            'genres': original_genres_set
        })



        try:
            movie_idx = movies_df[movies_df['movieId'] == movie_id].index[0]
            movie_title = movies_df.iloc[movie_idx]['title']
            movie_genres = set(movies_df.iloc[movie_idx]['genres'].split('|'))

            cluster_label = cluster_labels[movie_idx]
            same_cluster_indices = np.where(cluster_labels == cluster_label)[0]
            same_cluster_indices = same_cluster_indices[same_cluster_indices != movie_idx]

            if len(same_cluster_indices) == 0:
                continue

            # Используем косинусное сходство для расчета схожести
            similarities = cosine_similarity(
                movie_content_vectors[movie_idx].reshape(1, -1),
                movie_content_vectors[same_cluster_indices]
            )[0]

            for idx, score in zip(same_cluster_indices, similarities):
                title = movies_df.iloc[idx]['title']
                genres = movies_df.iloc[idx]['genres']
                genres_set = set(genres.split('|'))

                if pd.isna(title) or pd.isna(genres):
                    continue

                if not movie_genres.intersection(genres_set):
                    score = 0  # Жестко, но ок для фильтрации

                all_recommendations.append({
                    'movie_id': movies_df.iloc[idx]['movieId'],
                    'title': title,
                    'genres': genres,
                    'similarity_score': score
                })
        except IndexError:
            continue

    all_recommendations_df = pd.DataFrame(all_recommendations)
    if all_recommendations_df.empty:
        return all_recommendations_df

    all_recommendations_df = all_recommendations_df.groupby(['movie_id', 'title', 'genres']).agg(
        {'similarity_score': 'sum'}).reset_index()

    drop_film = importance_df[importance_df['importance_score'] == -1]['movieId']
    drop_film = drop_film.astype(int)

    new_recs = all_recommendations_df[~all_recommendations_df['movie_id'].isin(drop_film)]
    all_recommendations_df = new_recs.sort_values('similarity_score', ascending=False).head(top_n)

    weights = np.linspace(20, 1, len(all_recommendations_df)).round()
    all_recommendations_df['weight'] = weights

    temp_list = importance_df[importance_df['importance_score'] == 1]
    temp_list = temp_list.rename(columns={'movie_id': 'movieId'})

    temp_list = temp_list.merge(
        movies_df[['movieId', 'title', 'genres']],
        on='movieId',
        how='left'
    ).dropna(subset=['title', 'genres'])

    if len(temp_list) != 0:
        print('Горячие новинки')
        print('-' * 50)
        print(temp_list[['movieId', 'title', 'genres']])

    return all_recommendations_df, recommendation_info, temp_list

def recommend_movies_by_cluster_filtered_raw(
        movie_ids:List,
        movie_content_vectors: np.ndarray,
        train_movie_ids:int=None,
        test:bool=False,
        top_n:int=50
)-> DataFrame | tuple[Any, list[dict[str, set[Any] | Any]]]:

    """
    Рекомендует топ-N фильмов по списку id просмотренных фильмов (для оффтеста по строкам).

    Аргументы:
       movie_ids Список id фильмов.
       movie_content_vectors: Контент вектор
       test: флаг. Если False - передается train вектор, если True - test.
       top_n: сколько фильмов возвращать.
    Возвращает рекомендации
    """

    project_root = Path(__file__).resolve().parent.parent
    params_path = project_root / "params.yaml"
    with open(params_path, "r") as f:
        paths = get_project_paths()

    kmeans_model = joblib.load(paths["models_dir"] / 'movie_clusters_raw.pkl')
    movies_df = pd.read_csv(paths["raw_dir"] / "movies.csv")

    n_train_movies = len(kmeans_model.labels_)

    if not test:
        cluster_labels = kmeans_model.labels_
    else:
        test_movie_clusters = predict_test_movie_clusters_raw(kmeans_model, movie_content_vectors)
        cluster_labels = np.concatenate([kmeans_model.labels_, test_movie_clusters])

    all_recommendations = []
    importance_df = pd.DataFrame(columns=['movieId', 'importance_score'])
    recommendation_info = []

    for movie_id in movie_ids:
        try:
            original_movie_row = movies_df[movies_df['movieId'] == movie_id]
            if original_movie_row.empty:
                print(f"Фильм {movie_id} не найден в movies.csv")
                continue

            original_movie_title = original_movie_row['title'].values[0]
            original_movie_genres = original_movie_row['genres'].values[0]
            original_genres_set = set(original_movie_genres.split('|'))

            recommendation_info.append({
                'movie_id': movie_id,
                'title': original_movie_title,
                'genres': original_genres_set
            })

            if not test:
                # Ищем индекс по train_movie_ids
                try:
                    movie_idx_in_train = np.where(train_movie_ids == movie_id)[0][0]
                except IndexError:
                    print(f"Фильм {movie_id} отсутствует в трейне.")
                    continue

                vector = movie_content_vectors[movie_idx_in_train]
                movie_idx = movie_idx_in_train
            else:
                # Для теста индекс просто ищем в movies_df
                movie_idx_in_test = np.where(movies_df['movieId'] == movie_id)[0][0]
                movie_idx = n_train_movies + movie_idx_in_test
                vector = movie_content_vectors[movie_idx_in_test]

            cluster_label = cluster_labels[movie_idx]
            same_cluster_indices = np.where(cluster_labels == cluster_label)[0]
            same_cluster_indices = same_cluster_indices[same_cluster_indices != movie_idx]

            if len(same_cluster_indices) == 0:
                continue

            similarities = []
            for idx in same_cluster_indices:
                try:
                    if idx < n_train_movies:
                        neighbor_vector = movie_content_vectors[idx]
                    else:
                        neighbor_vector = movie_content_vectors[idx - n_train_movies]

                    similarity = cosine_similarity(
                        vector.reshape(1, -1),
                        neighbor_vector.reshape(1, -1)
                    )[0][0]
                    similarities.append(similarity)

                except Exception as e:
                    print(f"Ошибка в расчете similarity для idx {idx}: {e}")
                    similarities.append(0)

            for idx, score in zip(same_cluster_indices, similarities):
                try:
                    neighbor_movie_id = movies_df.iloc[idx]['movieId']
                    neighbor_title = movies_df.iloc[idx]['title']
                    neighbor_genres = movies_df.iloc[idx]['genres']
                    genres_set = set(neighbor_genres.split('|'))

                    if pd.isna(neighbor_title) or pd.isna(neighbor_genres):
                        continue

                    if not original_genres_set.intersection(genres_set):
                        score = 0

                    all_recommendations.append({
                        'movie_id': neighbor_movie_id,
                        'title': neighbor_title,
                        'genres': neighbor_genres,
                        'similarity_score': score
                    })
                except Exception as e:
                    print(f"Ошибка обработки соседа idx {idx}: {e}")
                    continue

        except Exception as e:
            print(f"Ошибка обработки фильма {movie_id}: {e}")
            continue

    all_recommendations_df = pd.DataFrame(all_recommendations)
    if all_recommendations_df.empty:
        return all_recommendations_df, recommendation_info

    all_recommendations_df = all_recommendations_df.groupby(['movie_id', 'title', 'genres']).agg(
        {'similarity_score': 'sum'}).reset_index()

    drop_film = importance_df[importance_df['importance_score'] == -1]['movieId']
    drop_film = drop_film.astype(int)

    new_recs = all_recommendations_df[~all_recommendations_df['movie_id'].isin(drop_film)]
    all_recommendations_df = new_recs.sort_values('similarity_score', ascending=False).head(top_n)

    weights = np.linspace(20, 1, len(all_recommendations_df)).round()
    all_recommendations_df['weight'] = weights

    temp_list = importance_df[importance_df['importance_score'] == 1]
    temp_list = temp_list.rename(columns={'movie_id': 'movieId'})

    temp_list = temp_list.merge(
        movies_df[['movieId', 'title', 'genres']],
        on='movieId',
        how='left'
    ).dropna(subset=['title', 'genres'])

    if len(temp_list) != 0:
        print('Горячие новинки')
        print('-' * 50)
        print(temp_list[['movieId', 'title', 'genres']])

    return all_recommendations_df, recommendation_info


def recommend_movies_by_cluster_als(
        movie_ids: List,
        movie_content_vectors: np.ndarray,
        train_movie_ids: int=None,
        test: bool=False,
        top_n: int=50
)-> DataFrame | tuple[Any, list[dict[str, set[Any] | Any]]]:
    """
        Рекомендует топ-N фильмов по списку id просмотренных фильмов по гибридному вектору.

        Аргументы:
           movie_ids Список id фильмов.
           movie_content_vectors: Контент вектор
           test: флаг. Если False - передается train вектор, если True - test.
           top_n: сколько фильмов возвращать.
        Возвращает рекомендации
    """
    project_root = Path(__file__).resolve().parent.parent
    params_path = project_root / "params.yaml"
    with open(params_path, "r") as f:
        paths = get_project_paths()

    kmeans_model = joblib.load(paths["models_dir"] / 'movie_and_als_clusters.pkl')
    movies_df = pd.read_csv(paths["raw_dir"] / "movies.csv")

    n_train_movies = len(kmeans_model.labels_)

    if not test:
        cluster_labels = kmeans_model.labels_
    else:
        test_movie_clusters = predict_test_movie_clusters_raw(kmeans_model, movie_content_vectors)
        cluster_labels = np.concatenate([kmeans_model.labels_, test_movie_clusters])

    all_recommendations = []
    importance_df = pd.DataFrame(columns=['movieId', 'importance_score'])
    recommendation_info = []

    for movie_id in movie_ids:
        try:
            original_movie_row = movies_df[movies_df['movieId'] == movie_id]
            if original_movie_row.empty:
                print(f"Фильм {movie_id} не найден в movies.csv")
                continue

            original_movie_title = original_movie_row['title'].values[0]
            original_movie_genres = original_movie_row['genres'].values[0]
            original_genres_set = set(original_movie_genres.split('|'))

            recommendation_info.append({
                'movie_id': movie_id,
                'title': original_movie_title,
                'genres': original_genres_set
            })

            if not test:
                # Ищем индекс по train_movie_ids
                try:
                    movie_idx_in_train = np.where(train_movie_ids == movie_id)[0][0]
                except IndexError:
                    print(f"Фильм {movie_id} отсутствует в трейне.")
                    continue

                vector = movie_content_vectors[movie_idx_in_train]
                movie_idx = movie_idx_in_train
            else:
                # Для теста индекс просто ищем в movies_df
                movie_idx_in_test = np.where(movies_df['movieId'] == movie_id)[0][0]
                movie_idx = n_train_movies + movie_idx_in_test
                vector = movie_content_vectors[movie_idx_in_test]

            cluster_label = cluster_labels[movie_idx]
            same_cluster_indices = np.where(cluster_labels == cluster_label)[0]
            same_cluster_indices = same_cluster_indices[same_cluster_indices != movie_idx]

            if len(same_cluster_indices) == 0:
                continue

            similarities = []
            for idx in same_cluster_indices:
                try:
                    if idx < n_train_movies:
                        neighbor_vector = movie_content_vectors[idx]
                    else:
                        neighbor_vector = movie_content_vectors[idx - n_train_movies]

                    similarity = cosine_similarity(
                        vector.reshape(1, -1),
                        neighbor_vector.reshape(1, -1)
                    )[0][0]
                    similarities.append(similarity)

                except Exception as e:
                    print(f"Ошибка в расчете similarity для idx {idx}: {e}")
                    similarities.append(0)

            for idx, score in zip(same_cluster_indices, similarities):
                try:
                    neighbor_movie_id = movies_df.iloc[idx]['movieId']
                    neighbor_title = movies_df.iloc[idx]['title']
                    neighbor_genres = movies_df.iloc[idx]['genres']
                    genres_set = set(neighbor_genres.split('|'))

                    if pd.isna(neighbor_title) or pd.isna(neighbor_genres):
                        continue

                    if not original_genres_set.intersection(genres_set):
                        score = 0

                    all_recommendations.append({
                        'movie_id': neighbor_movie_id,
                        'title': neighbor_title,
                        'genres': neighbor_genres,
                        'similarity_score': score
                    })
                except Exception as e:
                    print(f"Ошибка обработки соседа idx {idx}: {e}")
                    continue

        except Exception as e:
            print(f"Ошибка обработки фильма {movie_id}: {e}")
            continue

    all_recommendations_df = pd.DataFrame(all_recommendations)
    if all_recommendations_df.empty:
        return all_recommendations_df, recommendation_info

    all_recommendations_df = all_recommendations_df.groupby(['movie_id', 'title', 'genres']).agg(
        {'similarity_score': 'sum'}).reset_index()

    # Удаляем фильмы, отмеченные как неважные (-1), и фильмы, по которым строились рекомендации
    drop_film = set(importance_df[importance_df['importance_score'] == -1]['movieId'].astype(int)) | set(movie_ids)

    new_recs = all_recommendations_df[~all_recommendations_df['movie_id'].isin(drop_film)]
    all_recommendations_df = new_recs.sort_values('similarity_score', ascending=False).head(top_n)

    weights = np.linspace(20, 1, len(all_recommendations_df)).round()
    all_recommendations_df['weight'] = weights

    temp_list = importance_df[importance_df['importance_score'] == 1]
    temp_list = temp_list.rename(columns={'movie_id': 'movieId'})

    temp_list = temp_list.merge(
        movies_df[['movieId', 'title', 'genres']],
        on='movieId',
        how='left'
    ).dropna(subset=['title', 'genres'])

    if len(temp_list) != 0:
        print('Горячие новинки')
        print('-' * 50)
        print(temp_list[['movieId', 'title', 'genres']])

    return all_recommendations_df, recommendation_info



def recommend_movies_by_new_films(
        movie_ids:List,
        movie_content_vectors: np.ndarray,
        updated_movies_df: pd.DataFrame,
        new_cluster: int,
        train_movie_ids:int=None,
        test:bool=False,
        top_n:int=50
)-> DataFrame | tuple[Any, list[dict[str, set[Any] | Any]]]:

    """
    Рекомендует топ-N по новому фильму.

    Аргументы:
       movie_ids Список id новых фильмов.
       movie_content_vectors: Контент вектор
       updated_movies_df: Обновленный датафрейм с фильами
       new_cluster: кластер нового фильма
       test: флаг. Если False - передается train вектор, если True - test.
       top_n: сколько фильмов возвращать.
    Возвращает рекомендации
    """

    project_root = Path(__file__).resolve().parent.parent
    params_path = project_root / "params.yaml"
    with open(params_path, "r") as f:
        paths = get_project_paths()

    kmeans_model = joblib.load(paths["models_dir"] / 'movie_and_als_clusters.pkl')
    movies_df = updated_movies_df

    if not test:
        cluster_labels = kmeans_model.labels_
    else:
        test_movie_clusters = predict_test_movie_clusters_raw(kmeans_model, movie_content_vectors)
        cluster_labels = np.concatenate([kmeans_model.labels_, test_movie_clusters])

    importance_df = pd.DataFrame(columns=['movieId', 'importance_score'])
    recommendation_info = []

    all_recommendations = []  # Здесь собираем все рекомендации

    for movie_id in movie_ids:
        original_movie_row = movies_df[movies_df['movieId'] == movie_id]
        if original_movie_row.empty:
            print(f"Фильм {movie_id} не найден в movies.csv")
            continue

        original_movie_title = original_movie_row['title'].values[0]
        original_movie_genres = original_movie_row['genres'].values[0]
        original_genres_set = set(original_movie_genres.split('|'))

        recommendation_info.append({
            'movie_id': movie_id,
            'title': original_movie_title,
            'genres': original_genres_set
        })

        same_cluster_indices = np.where(cluster_labels == new_cluster)[0]

        similarities = []
        for idx in same_cluster_indices:
            neighbor_vector = movie_content_vectors[idx]

            new_movie_idx = movie_content_vectors.shape[0] - 1
            new_movie_vector = movie_content_vectors[new_movie_idx]

            similarity = cosine_similarity(
                new_movie_vector.reshape(1, -1),
                neighbor_vector.reshape(1, -1)
            )[0][0]

            neighbor_movie = updated_movies_df.iloc[idx]
            neighbor_id = neighbor_movie['movieId']
            neighbor_title = neighbor_movie['title']
            neighbor_genres = neighbor_movie['genres']

            similarities.append({
                'movie_id': neighbor_id,
                'title': neighbor_title,
                'genres': neighbor_genres,
                'similarity_score': similarity
            })

        # После этого создаем DataFrame из списка similarities
        if similarities:
            all_recommendations.extend(similarities)

    # Создаем итоговый DataFrame
    all_recommendations_df = pd.DataFrame(all_recommendations)

    if all_recommendations_df.empty:
        return all_recommendations_df, recommendation_info

    all_recommendations_df = all_recommendations_df.groupby(['movie_id', 'title', 'genres']).agg(
        {'similarity_score': 'sum'}).reset_index()

    # Удаляем фильмы, отмеченные как неважные (-1), и фильмы, по которым строились рекомендации
    drop_film = set(importance_df[importance_df['importance_score'] == -1]['movieId'].astype(int)) | set(movie_ids)

    new_recs = all_recommendations_df[~all_recommendations_df['movie_id'].isin(drop_film)]
    all_recommendations_df = new_recs.sort_values('similarity_score', ascending=False).head(top_n)

    weights = np.linspace(20, 1, len(all_recommendations_df)).round()
    all_recommendations_df['weight'] = weights

    temp_list = importance_df[importance_df['importance_score'] == 1]
    temp_list = temp_list.rename(columns={'movie_id': 'movieId'})

    temp_list = temp_list.merge(
        movies_df[['movieId', 'title', 'genres']],
        on='movieId',
        how='left'
    ).dropna(subset=['title', 'genres'])

    if len(temp_list) != 0:
        print('Горячие новинки')
        print('-' * 50)
        print(temp_list[['movieId', 'title', 'genres']])

    return all_recommendations_df, recommendation_info


def get_rec_on_train_content_vector(movie_ids:List)-> DataFrame | tuple[Any, list[dict[str, set[Any] | Any]]]:
    """
    Сборная функция для передачи данных в streamlit для train вектора
    Вызывает recommend_movies_by_cluster_filtered
    Аргументы:
       movie_ids Список id новых фильмов.
    Возвращает рекомендации
    """
    project_root = Path(__file__).resolve().parent.parent  # Выходим из scripts/
    params_path = project_root / "params.yaml"
    with open(params_path, "r") as f:
        paths = get_project_paths()

    movie_content_vectors_train = np.load(paths["models_dir"] / "movie_content_vectors_train.npz")['vectors']
    all_recommendations_df, recommendation_info = recommend_movies_by_cluster_filtered(
        movie_ids, movie_content_vectors_train, test = False, top_n=50)

    return all_recommendations_df, recommendation_info

def get_rec_on_content_vector_block(
        movie_ids:List,
        promo_ids:List[int],
        block_idx:List[int])-> DataFrame | tuple[Any, list[dict[str, set[Any] | Any]]]:
    """
    Сборная функция для передачи данных в streamlit для train вектора с возможностью пролдвижения/блокировик фильмов
    Вызывает recommend_movies_by_cluster_filtered
    Аргументы:
       movie_ids Список id новых фильмов.
    Возвращает рекомендации
    """
    project_root = Path(__file__).resolve().parent.parent  # Выходим из scripts/
    params_path = project_root / "params.yaml"
    with open(params_path, "r") as f:
        paths = get_project_paths()

    importance_df = pd.DataFrame(columns=['movieId', 'importance_score'])

    importance_df= update_importance_scores(
        importance_df =importance_df,
        movie_ids_to_promote=promo_ids,
        movie_ids_to_block= block_idx
    )

    movie_content_vectors_train = np.load(paths["models_dir"] / "movie_content_vectors_train.npz")['vectors']
    all_recommendations_df, recommendation_info, temp_list = recommend_movies_with_block(
        movie_ids, movie_content_vectors_train,importance_df, test = False, top_n=15)

    return all_recommendations_df, recommendation_info, temp_list



def get_rec_on_test_content_vector(movie_ids:List)-> DataFrame | tuple[Any, list[dict[str, set[Any] | Any]]]:
    """
        Сборная функция для передачи данных в streamlit для test вектора
        Вызывает recommend_movies_by_cluster_filtered
        Аргументы:
           movie_ids Список id новых фильмов.
        Возвращает рекомендации
    """

    project_root = Path(__file__).resolve().parent.parent  # Выходим из scripts/
    params_path = project_root / "params.yaml"
    with open(params_path, "r") as f:
        paths = get_project_paths()

    movie_content_vectors_test = np.load(paths["models_dir"] / "movie_content_vectors_test.npz")['vectors']
    all_recommendations_df, recommendation_info = recommend_movies_by_cluster_filtered(movie_ids, movie_content_vectors_test, test = True, top_n=50)


    return all_recommendations_df, recommendation_info

def get_combine_content_vector(movie_ids:List)-> DataFrame | tuple[Any, list[dict[str, set[Any] | Any]]]:
    """
        Сборная функция для передачи данных в streamlit для обединенного вектора
        Вызывает recommend_movies_by_cluster_filtered
        Аргументы:
           movie_ids Список id новых фильмов.
        Возвращает рекомендации
    """

    project_root = Path(__file__).resolve().parent.parent  # Выходим из scripts/
    params_path = project_root / "params.yaml"
    with open(params_path, "r") as f:
        paths = get_project_paths()

    movie_content_vectors_train = np.load(paths["models_dir"] / "movie_content_vectors_train.npz")['vectors']
    movie_content_vectors_test = np.load(paths["models_dir"] / "movie_content_vectors_test.npz")['vectors']

    combined_movie_content_vectors = np.concatenate([movie_content_vectors_train, movie_content_vectors_test], axis=0)
    all_recommendations_df, recommendation_info = recommend_movies_by_cluster_filtered(movie_ids, combined_movie_content_vectors, test = True, top_n=50)


    return all_recommendations_df, recommendation_info


def get_rec_on_train_content_vector_raw(movie_ids:List)-> DataFrame | tuple[Any, list[dict[str, set[Any] | Any]]]:
    """
        Сборная функция для передачи данных в streamlit для train вектора (при построчном разделении)
        Вызывает recommend_movies_by_cluster_filtered
        Аргументы:
           movie_ids Список id новых фильмов.
        Возвращает рекомендации
    """

    project_root = Path(__file__).resolve().parent.parent
    params_path = project_root / "params.yaml"
    with open(params_path, "r") as f:
        paths = get_project_paths()

    movie_content_vectors_train = np.load(paths["models_dir"] / "movie_content_vectors_train.npz")['vectors']
    train_movie_ids = np.load(paths["models_dir"] / "train_movie_ids.npy")
    movie_content_vectors_test = np.load(paths["models_dir"] / "movies_vectors_test_raw.npz")['vectors']

    all_recommendations_df_test, recommendation_info = recommend_movies_by_cluster_filtered_raw(
        movie_ids,
        movie_content_vectors_train,
        train_movie_ids=train_movie_ids,
        test=False,
        top_n=50
    )

    all_recommendations_df_train, recommendation_info = recommend_movies_by_cluster_filtered_raw(
        movie_ids,
        movie_content_vectors_test,
        train_movie_ids=train_movie_ids,
        test=False,
        top_n=50
    )

    return all_recommendations_df_train, all_recommendations_df_test, recommendation_info

def get_als_and_content_vector(movie_ids:List)-> DataFrame | tuple[Any, list[dict[str, set[Any] | Any]]]:
    """
        Сборная функция для передачи данных в streamlit для test вектора (при построчном разделении)
        Вызывает recommend_movies_by_cluster_filtered
        Аргументы:
           movie_ids Список id новых фильмов.
        Возвращает рекомендации
    """

    project_root = Path(__file__).resolve().parent.parent  # Выходим из scripts/
    params_path = project_root / "params.yaml"
    with open(params_path, "r") as f:
        paths = get_project_paths()

    hybrid_movie_vector_full = np.load(paths["models_dir"] / "hybrid_movie_vector_full.npz")['vectors']

    all_recommendations_df, recommendation_info = recommend_movies_by_cluster_als(movie_ids, hybrid_movie_vector_full, test = True, top_n=50)


    return all_recommendations_df, recommendation_info


def add_new_movie_and_predict_cluster(
        new_movie_id:int,
        new_movie_title:str,
        new_movie_genres: List,
        original_movie_vector: np.ndarray,
        models_bridge:LGBMModel,
        hybrid_movie_vectors: np.ndarray,
        movies: pd.DataFrame,
        kmeans_model:KMeans)-> pd.DataFrame|np.ndarray|int:
    """
    Функция добавления нового фильма

    Аргументы:
        new_movie_id - id нового фильма.
        new_movie_title - название
        new_movie_genres: Жанры,
        original_movie_vector: вектор нового фильма,
        models_bridge: Модель моста для als,
        hybrid_movie_vectors: гибридный вектор,
        movies: датафрейм фильмов,
        kmeans_model: моедль кластеризации
    Возвращает рекомендации
    """

    # Предсказание item факторов для нового фильма (предполагаем, что вектор уже получен)
    predicted_item_factors = predict_item_factors_batch(original_movie_vector, models_bridge)

    # Создаем гибридный вектор для нового фильма
    hybrid_vector = np.hstack([original_movie_vector, predicted_item_factors])

    # Убедимся, что данные передаются в формате float64
    hybrid_vector = hybrid_vector.astype(np.float64)

    # Получаем кластер для нового фильма
    new_movie_cluster = kmeans_model.predict(hybrid_vector.reshape(1, -1))[0]


    # Обновляем гибридные векторы
    updated_hybrid_movie_vectors = np.vstack([hybrid_movie_vectors, hybrid_vector])

    # Создаем новый фильм
    new_movie = {'movieId': new_movie_id, 'title': new_movie_title, 'genres': new_movie_genres}

    # Обновляем DataFrame, добавляя новый фильм
    updated_movies_df = pd.concat([movies, pd.DataFrame([new_movie])], ignore_index=True)

    return updated_movies_df, updated_hybrid_movie_vectors, new_movie_cluster

def find_similar_movie_by_genres(
        movies_df: pd.DataFrame,
        movie_tag_matrix: pd.DataFrame,
        target_genres: list[str]) -> Optional[np.ndarray]:
    """
    Функция поиска похожего фильма по жанрам

    Аргументы:
        movies_df: датафрем фильмов,
        movie_tag_matrix: резвернутая матрица по фильмам с тегами и жанрами,
        target_genres: list[str]) -> жанры
    Возвращает рекомендации
    """

    def genre_match(genres_str: str, genres_subset: set[str]) -> bool:
        """
        Сабфункция получения жанров

        Аргументы:
        genres_str: жанры строкой,
        """
        return genres_subset.issubset(set(genres_str.split('|')))

    genres_set = set(target_genres)

    for i in range(len(genres_set), 0, -1):
        for subset in permutations(genres_set, i):
            subset_set = set(subset)
            candidates = movies_df[movies_df['genres'].apply(lambda g: genre_match(g, subset_set))]
            if not candidates.empty:
                for _, row in candidates.iterrows():
                    movie_id = row['movieId']
                    if movie_id in movie_tag_matrix.index:
                        return movie_tag_matrix.loc[movie_id].values
    return None

def add_new_films(new_movie_title:str, selected_genres: List[str])-> DataFrame | tuple[Any, list[dict[str, set[Any] | Any]]]:
    """
    Функция добавления нового фильма и выдачи рекомендаций

    Аргументы:
        new_movie_title: название фильма,
        selected_genres: жанры
    Возвращает рекомендации
    """
    project_root = Path(__file__).resolve().parent.parent  # Выходим из scripts/
    params_path = project_root / "params.yaml"
    with open(params_path, "r") as f:
        config = yaml.safe_load(f)["autoencoder"]
        paths = get_project_paths()


    movies = pd.read_csv(paths["raw_dir"] / "movies.csv")

    data = np.load(paths["raw_dir"] /"movie_tag_matrix_small.npz")

    matrix = data["matrix"]
    movie_ids = data["movie_ids"]
    tag_ids = data["tag_ids"]

    movie_tag_matrix = pd.DataFrame(matrix, index=movie_ids, columns=tag_ids)


    # Находим похожий фильм по жанрам
    original_movie_vector = find_similar_movie_by_genres(movies, movie_tag_matrix, selected_genres)

    # Масштабируем вектор
    scaler_full_vector = MinMaxScaler()
    scaler_full_vector.fit_transform(movie_tag_matrix)
    scaled = scaler_full_vector.transform([original_movie_vector])

    # Загружаем модель и генерируем вектор
    model_path = paths["models_dir"] / 'movies_content_autoencoder_raw.pt'
    movie_vectors_scaled_full = np.load(paths["processed_dir"] / "movie_vectors_scaled_full.npy")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = movie_vectors_scaled_full.shape[1]
    hybrid_movie_vector_full = np.load(paths["models_dir"] / "hybrid_movie_vector_full.npz")['vectors']

    model = MovieAutoencoder(input_dim=input_dim, encoding_dim=64).to(device)
    model.load_state_dict(torch.load(model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Перемещаем модель на выбранное устройство
    scaled_tensor = torch.tensor(scaled, dtype=torch.float32).to(device)

    new_movie_vector = model.encoder(scaled_tensor).cpu().detach().numpy()  # Переносим результат обратно на CPU для дальнейшей обработки

    kmeans_hibrid_full = joblib.load(paths["models_dir"] / 'movie_and_als_clusters.pkl')
    models_bridge = joblib.load(paths["models_dir"] / 'models_bridge.pkl')

    # Получаем результат добавления фильма и кластер
    updated_movies_df, updated_hybrid_vectors, new_cluster = add_new_movie_and_predict_cluster(
        movies['movieId'].max() + 1,  # Новый ID фильма
        new_movie_title,
        "|".join(selected_genres),
        new_movie_vector,
        models_bridge,
        hybrid_movie_vector_full,
        movies,
        kmeans_hibrid_full
    )


    st.success(f"✅ Фильм добавлен! Кластер: {new_cluster}")

    # Генерация рекомендаций
    all_recommendations_hybrid, recommendation_info_hybrid = recommend_movies_by_new_films(
        movie_ids=[movies['movieId'].max() + 1],
        movie_content_vectors=updated_hybrid_vectors,
        updated_movies_df=updated_movies_df,
        new_cluster=new_cluster,
        train_movie_ids=None,
        test=False,
        top_n=50
    )

    return all_recommendations_hybrid, recommendation_info_hybrid


def get_user_recommendations_with_ensemble(
        user_id:int,
        user_content_vector: np.ndarray,
        user_encoder: LabelEncoder,
        ratings_csr: csr_matrix,
        kmeans_users: KMeans,
        importance_df: pd.DataFrame,
        movies_df:pd.DataFrame,
        item_encoder: LabelEncoder,
        top_n:int =50,
        num_user:int =5,
        is_test: bool=False
)-> pd.DataFrame | pd.DataFrame:

    """
    Функция выдачи рекомендаций на основе профиля пользователя (через асамблирование ближайших пользователей)
    Аргументы:
        user_id: - id пользователя,
        user_content_vector: общий вектор пользователей,
        user_encoder: кодировщик для пользователей,
        ratings_csr: матрица взаимодействий,
        kmeans_users: модель кластеризации,
        importance_df: датафрейм продвигаемых/блокируемых фильмов,
        movies_df: датафрейм фильмов,
        item_encoder: кодировщик для фильмов,
        top_n:int = кол-во рекомендаций к выдаче,
        num_user кол-во пользователь которые участвуют в составлении рекомендаций,
        is_test: Флаг, для сообщения обучающий это вектор или тестовый
    Возвращает рекомендации
    """

    project_root = Path(__file__).resolve().parent.parent  # Выходим из scripts/
    params_path = project_root / "params.yaml"
    with open(params_path, "r") as f:
        paths = get_project_paths()

    # Построим маппинг user_id → индекс относительно user_content_vector
    available_user_ids = user_encoder.inverse_transform(range(len(user_content_vector)))
    user_id_to_idx_map = {uid: idx for idx, uid in enumerate(available_user_ids)}

    if user_id not in user_id_to_idx_map:
        raise ValueError(f"❌ User ID {user_id} не найден в переданном векторе пользователей.")

    user_idx = user_id_to_idx_map[user_id]
    user_vector = user_content_vector[user_idx]

    user_cluster = kmeans_users.predict([user_vector])[0]
    cluster_users_idx = np.where(kmeans_users.labels_ == user_cluster)[0]

    user_content_vector_full = np.load(paths["models_dir"] / "user_content_vector.npz")['vectors']

    cluster_user_vectors = user_content_vector_full[cluster_users_idx]
    distances = cosine_similarity(user_vector.reshape(1, -1), cluster_user_vectors).flatten()

    filtered_users = []
    for lower in np.arange(0.01, 0.99, 0.05):
        upper = lower + 0.1
        for idx, dist in zip(cluster_users_idx, distances):
            if (not is_test and idx != user_idx) or is_test:
                if lower < dist <= upper:
                    filtered_users.append((idx, dist))
                    if len(filtered_users) >= num_user:
                        break
        if len(filtered_users) >= num_user:
            break

    if not filtered_users:
        print(f"⚠️ Нет подходящих пользователей для user_id={user_id} в диапазоне dist ∈ (0.1, 0.99)")
    else:
        print(f"✅ Найдено {len(filtered_users)} похожих пользователей")

    filtered_users = sorted(filtered_users, key=lambda x: -x[1])[:num_user]

    recommendation_scores = defaultdict(lambda: {"score": 0, "users": set()})
    similar_users_recommendations = {}

    for rank, (sim_user_idx, sim_score) in enumerate(filtered_users):
        sim_user_id = user_encoder.inverse_transform([sim_user_idx])[0]
        recommended_items = get_recommendations(sim_user_idx, ratings_csr, top_n=100)

        user_rec_list = []
        for movie_id in recommended_items:
            if movie_id in item_encoder.classes_:
                movie_row = movies_df[movies_df['movieId'] == movie_id]
                if movie_row.empty:
                    continue
                title = movie_row['title'].values[0]
                genres = movie_row['genres'].values[0]

                recommendation_scores[movie_id]["score"] += 1
                recommendation_scores[movie_id]["users"].add(sim_user_id)

                user_rec_list.append({
                    'user_id': sim_user_id,
                    'movie_id': movie_id,
                    'title': title,
                    'genres': genres,
                    'similarity_rank': rank + 1
                })

        similar_users_recommendations[sim_user_id] = user_rec_list

    final_recs = []
    for movie_id, data in recommendation_scores.items():
        movie_row = movies_df[movies_df['movieId'] == movie_id]
        if movie_row.empty:
            continue
        title = movie_row['title'].values[0]
        genres = movie_row['genres'].values[0]
        final_recs.append({
            'movie_id': movie_id,
            'title': title,
            'genres': genres,
            'ensemble_score': data['score'],
            'supported_by_users': list(data['users'])
        })

    final_df = pd.DataFrame(final_recs).sort_values(by='ensemble_score', ascending=False).head(top_n)

    similar_users_recommendations_df = pd.DataFrame([
        rec for recs in similar_users_recommendations.values() for rec in recs
    ])

    return final_df, similar_users_recommendations_df


# Функция для получения рекомендаций для конкретного пользователя
def get_recommendations(user_idx:int, ratings_csr: csr_matrix, top_n:int=5)-> pd.DataFrame:
    """
    Получает топ-N рекомендаций для пользователя на основе разреженной матрицы взаимодействий.
    Атрибуты:
        user_idx: Индекс пользователя.
        ratings_csr: Разреженная матрица взаимодействий.
        top_n: Количество рекомендованных фильмов.
    Возвращает- Список рекомендованных фильмов.
    """
    # Получаем оценки текущего пользователя
    user_ratings = ratings_csr[user_idx].toarray().flatten()  # Рейтинг пользователя
    unrated_movies = np.where(user_ratings == 0)[0]  # Фильмы, которые не оценены

    # Оцениваем все фильмы, которые пользователь не оценивал, на основе сходства
    # Работаем с разреженными матрицами
    scores = ratings_csr[unrated_movies].dot(ratings_csr[user_idx].T).toarray().flatten()

    # Получаем индексы рекомендованных фильмов
    recommended_movies_idx = unrated_movies[np.argsort(scores)[-top_n:]]

    return recommended_movies_idx


def find_top_similar_users(user_id, user_vectors_train, user_vectors_test, user_encoder, top_k=10):
    """
    Находит топ-N похожих пользователей из теста по user_id из трейна.
    Атрибуты:
        user_id: ID пользователя из трейна
        user_vectors_train: Векторы пользователей из трейна
        user_vectors_test: Векторы пользователей из теста
        user_encoder: Encoder, общий для train и test
        top_k: Сколько похожих пользователей из теста вернуть
    Возвращает: DataFrame с топ-N наиболее похожими пользователями
    """
    user_idx_train = user_encoder.transform([user_id])[0]
    user_vector_train = user_vectors_train[user_idx_train].reshape(1, -1)

    similarities = cosine_similarity(user_vector_train, user_vectors_test).flatten()
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    top_user_ids = user_encoder.inverse_transform(top_indices)
    top_scores = similarities[top_indices]

    def categorize_similarity(score):
        if score == 1.0:
            return "🔁 идентичный"
        elif score > 0.99:
            return "✅ почти идентичный"
        elif score > 0.95:
            return "👍 близкий"
        elif score > 0.90:
            return "👌 средний"
        else:
            return "❔ слабый"

    data = {
        "train_user_id": [user_id] * top_k,
        "test_user_id": top_user_ids,
        "similarity": top_scores,
        "similarity_level": [categorize_similarity(s) for s in top_scores]
    }

    return pd.DataFrame(data)


def get_similar_user_in_test_set(
        user_id:int,
        user_vectors_train: np.ndarray,
        user_vectors_test: np.ndarray,
        user_encoder: LabelEncoder,
        top_k:int=10)-> tuple[Any, Any] | tuple[list[Any], list[Any]]:

    """
    Находит в тестовой векторе фильм похожий на искомый фильм в трейн векторе
    Атрибуты:
        user_vectors_train: Обучающий вектора,
        user_vectors_test: Тестовый вектор,
        user_encoder: Кодировщик для пользователей,
    Возвращает: DataFrame с топ-N наиболее похожими пользователями
    """


    try:
        # Преобразуем ID пользователя в индекс
        user_idx_train = user_encoder.transform([user_id])[0]

        # Получаем вектор пользователя из тренировочного набора
        user_vector_train = user_vectors_train[user_idx_train]

        # Ищем похожих пользователей в тестовом наборе
        knn = NearestNeighbors(n_neighbors=top_k, metric='cosine')
        knn.fit(user_vectors_test)

        distances, indices = knn.kneighbors([user_vector_train])

        # Выводим идентификаторы пользователей, которые похожи
        similar_users = user_encoder.inverse_transform(indices[0])
        return similar_users, distances[0]

    except Exception as e:
        print(f"Error in finding similar user: {e}")
        return [], []

def get_recommendation_on_user_vector(userid:int)-> pd.DataFrame|pd.DataFrame:

    """
    Функция для получения рекомендаций по id пользователя
    Атрибуты:
        userid: id пользователя
    Возвращает: DataFrame с топ-N наиболее похожими пользователями
    """

    project_root = Path(__file__).resolve().parent.parent  # Выходим из scripts/
    params_path = project_root / "params.yaml"
    with open(params_path, "r") as f:
        paths = get_project_paths()

    user_vectors_train = np.load(paths["models_dir"] / "user_vectors_train.npz")['vectors']
    user_vectors_test = np.load(paths["models_dir"] / "user_vectors_test.npz")['vectors']
    user_encoder = joblib.load(paths["processed_dir"] / "user_encoder.pkl")
    ratings_csr = load_npz(paths["processed_dir"] / "ratings_csr.npz")
    kmeans_users = joblib.load(paths["models_dir"] / 'kmeans_users.pkl')
    importance_df = pd.DataFrame(columns=['movieId', 'importance_score'])
    movies = pd.read_csv(paths["raw_dir"] / "movies.csv")
    item_encoder = joblib.load(paths["processed_dir"] / "item_encoder.pkl")

    user_idx_train = user_encoder.transform([userid])[0]
    print(f"Train user index: {user_idx_train}")  # Печатаем индекс пользователя для тренировки

    # Для теста:
    user_idx_test = user_encoder.transform([userid])[0]
    print(f"Test user index: {user_idx_test}")  # Печатаем индекс пользователя для теста

    # Убедитесь, что индексы не выходят за пределы массива
    if user_idx_test >= len(user_vectors_test):
        print(f"Error: Test index {user_idx_test} out of bounds for test set.")
    else:
        cluster_user_vectors = user_vectors_test[user_idx_test]
        print(f"Cluster user vectors for test: {cluster_user_vectors}")


    final_df, similar_users_recommendations_df = get_user_recommendations_with_ensemble(
        user_id=userid,
        user_content_vector=user_vectors_train,
        user_encoder=user_encoder,
        ratings_csr=ratings_csr,
        kmeans_users= kmeans_users,
        importance_df=importance_df,
        movies_df=movies,
        item_encoder=item_encoder,
        top_n=50,
        num_user=5,
        is_test=False
    )


    similar_users, distances = get_similar_user_in_test_set(
        user_id=userid,
        user_vectors_train=user_vectors_train,
        user_vectors_test=user_vectors_test,
        user_encoder=user_encoder,
        top_k=10
    )

    first_test_user_id = similar_users[0]

    user_idx_train = user_encoder.transform([first_test_user_id])[0]
    print(f"Train user index: {user_idx_train}")  # Печатаем индекс пользователя для тренировки

    # Для теста:
    user_idx_test = user_encoder.transform([first_test_user_id])[0]
    print(f"Test user index: {user_idx_test}")  # Печатаем индекс пользователя для теста

    # Убедитесь, что индексы не выходят за пределы массива
    if user_idx_test >= len(user_vectors_test):
        print(f"Error: Test index {user_idx_test} out of bounds for test set.")
    else:
        cluster_user_vectors = user_vectors_test[user_idx_test]
        print(f"Cluster user vectors for test: {cluster_user_vectors}")

    final_df_test, recs_test = get_user_recommendations_with_ensemble(
        user_id=first_test_user_id,
        user_content_vector=user_vectors_test,
        user_encoder=user_encoder,
        ratings_csr=ratings_csr,
        kmeans_users=kmeans_users,
        importance_df=importance_df,
        movies_df=movies,
        item_encoder=item_encoder,
        top_n=50,
        num_user=5,
        is_test=True
    )

    return final_df, final_df_test


def predict_recommendations(user_id:int, top_k:int=50)-> pd.DataFrame:
    """
    Функция для получения рекомендаций по предикту нейросети
    Атрибуты:
        userid: id пользователя
        top_k - кол-во рекомендаций
    Возвращает: DataFrame с топ-N рекомендаций
    """

    paths = get_project_paths()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Загрузка всего
    user_vectors = load_vectors_npz(paths["models_dir"] / "embedding_user_vectors.npz")
    item_vectors = load_vectors_npz(paths["models_dir"] / "embedding_item_vectors.npz")

    user_encoder = joblib.load(paths["processed_dir"] / "user_encoder.pkl")
    item_encoder = joblib.load(paths["processed_dir"] / "item_encoder.pkl")
    movies = pd.read_csv(paths["raw_dir"] / "movies.csv")

    model = RatingPredictor(vector_dim=user_vectors.shape[1])
    model.load_state_dict(torch.load(paths["models_dir"] / "neural_model_best.pt", map_location=device))
    model.to(device)
    model.eval()

    # Получение вектора пользователя
    user_idx = user_encoder.transform([user_id])[0]
    user_vec = torch.tensor(user_vectors[user_idx], dtype=torch.float32).to(device)
    user_vec = user_vec.unsqueeze(0).repeat(item_vectors.shape[0], 1)

    # Векторы фильмов
    item_vecs = torch.tensor(item_vectors, dtype=torch.float32).to(device)

    with torch.no_grad():
        preds = model(user_vec, item_vecs).squeeze().cpu().numpy()

    top_k_indices = preds.argsort()[-top_k:][::-1]
    top_k_scores = preds[top_k_indices]

    # Преобразуем индексы обратно в movieId
    movie_ids = item_encoder.inverse_transform(top_k_indices)
    result = []

    for movie_id, score in zip(movie_ids, top_k_scores):
        row = movies[movies["movieId"] == movie_id]
        if not row.empty:
            result.append({
                "movie_id": movie_id,
                "title": row.iloc[0]["title"],
                "genres": row.iloc[0]["genres"],
                "score": score
            })
    temp = pd.DataFrame(result)

    return temp

def get_user_recommendations_new_user(
        user_idx:int,  # теперь это индекс пользователя в user_content_vector
        user_content_vector: np.ndarray,
        ratings_csr: csr_matrix,
        kmeans_users: KMeans,
        importance_df: pd.DataFrame,
        movies_df: pd.DataFrame,
        item_encoder: LabelEncoder,
        top_n:int=50,
        num_user:int=5,
        is_test:bool=False
)-> pd.DataFrame| List[pd.DataFrame]:
    """
    Получает рекомендации для нового пользователя асамблирование соседних пользоватлей
    Атрибуты:
        Прогоняет через автоэнкодер, добавляет в user_content_vector.
        user_idx:int,  id пользователя
        user_content_vector: общий вектор пользователей,
        ratings_csr: матрица взаимодействий,
        kmeans_users: модель кластеризатора,
        importance_df: датафрейм блокируемых и продвигаемых фильмов,
        movies_df:датафрем фильмов,
        item_encoder: Кодировщик фильмов,
        top_n: кол-во рекомендаций,
        num_user:кол-во пользователей которые составляют ансамбль
        is_test: Флаг для указания функции какой вектор подается тестовый или обучающий/общий
    Возвращает рекомендации
    """

    project_root = Path(__file__).resolve().parent.parent
    params_path = project_root / "params.yaml"
    with open(params_path, "r") as f:
        paths = get_project_paths()

    # Вектор текущего пользователя
    user_vector = user_content_vector[user_idx]

    # Определяем кластер пользователя
    user_cluster = kmeans_users.predict([user_vector])[0]
    cluster_users_idx = np.where(kmeans_users.labels_ == user_cluster)[0]

    cluster_user_vectors = user_content_vector[cluster_users_idx]

    # Считаем косинусное сходство
    distances = cosine_similarity(user_vector.reshape(1, -1), cluster_user_vectors).flatten()

    # Отбираем пользователей с похожим профилем
    filtered_users = []
    for lower in np.arange(0.01, 0.99, 0.05):
        upper = lower + 0.1
        for idx, dist in zip(cluster_users_idx, distances):
            if (not is_test and idx != user_idx) or is_test:
                if lower < dist <= upper:
                    filtered_users.append((idx, dist))
                    if len(filtered_users) >= num_user:
                        break
        if len(filtered_users) >= num_user:
            break

    if not filtered_users:
        print(f"⚠️ Нет подходящих пользователей для user_idx={user_idx} в диапазоне dist ∈ (0.1, 0.99)")
    else:
        print(f"✅ Найдено {len(filtered_users)} похожих пользователей")

    filtered_users = sorted(filtered_users, key=lambda x: -x[1])[:num_user]

    # Сбор рекомендаций
    recommendation_scores = defaultdict(lambda: {"score": 0, "users": set()})
    similar_users_recommendations = {}

    for rank, (sim_user_idx, sim_score) in enumerate(filtered_users):
        recommended_items = get_recommendations(sim_user_idx, ratings_csr, top_n=100)

        user_rec_list = []
        for movie_id in recommended_items:
            if movie_id in item_encoder.classes_:
                movie_row = movies_df[movies_df['movieId'] == movie_id]
                if movie_row.empty:
                    continue
                title = movie_row['title'].values[0]
                genres = movie_row['genres'].values[0]

                recommendation_scores[movie_id]["score"] += 1
                recommendation_scores[movie_id]["users"].add(sim_user_idx)

                user_rec_list.append({
                    'user_id': sim_user_idx,
                    'movie_id': movie_id,
                    'title': title,
                    'genres': genres,
                    'similarity_rank': rank + 1
                })

        similar_users_recommendations[sim_user_idx] = user_rec_list

    # Финальный список
    final_recs = []
    for movie_id, data in recommendation_scores.items():
        movie_row = movies_df[movies_df['movieId'] == movie_id]
        if movie_row.empty:
            continue
        title = movie_row['title'].values[0]
        genres = movie_row['genres'].values[0]
        final_recs.append({
            'movie_id': movie_id,
            'title': title,
            'genres': genres,
            'ensemble_score': data['score'],
            'supported_by_users': list(data['users'])
        })

    final_df = pd.DataFrame(final_recs).sort_values(by='ensemble_score', ascending=False).head(top_n)

    similar_users_recommendations_df = pd.DataFrame([
        rec for recs in similar_users_recommendations.values() for rec in recs
    ])

    return final_df, similar_users_recommendations_df





# функция добавления пользователя
def add_new_user_to_system(user_ratings_dict: Dict)->np.ndarray | int:
    """
    Добавляет нового пользователя по его рейтингам (movie_id -> rating).
    Прогоняет через автоэнкодер, добавляет в user_content_vector.
    Атрибуты:
        user_ratings_dict - словарь с id фильмов и их оценками
    Возвращает обновлённый user_content_vector и индекс нового пользователя.
    """

    project_root = Path(__file__).resolve().parent.parent  # Выходим из scripts/
    params_path = project_root / "params.yaml"
    with open(params_path, "r") as f:
        config = yaml.safe_load(f)["autoencoder"]
        paths = get_project_paths()

    item_encoder = joblib.load(paths["processed_dir"] / "item_encoder.pkl")


    user_content_vector = np.load(paths["models_dir"] / "user_content_vector.npz")['vectors']
    encoding_dim = config["encoding_dim"]
    ratings_csr = load_npz(paths["processed_dir"] / "ratings_csr.npz")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = ratings_csr.shape[1]

    model = Autoencoder(input_dim=input_dim, encoding_dim=encoding_dim).to(device)
    model_path = paths["models_dir"] / 'user_autoencoder_model.pt'
    model.load_state_dict(torch.load(model_path, map_location=device))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kmeans_full_users = joblib.load(paths["models_dir"] / 'kmeans_full_users.pkl')

    # Создание разреженного вектора
    num_items = len(item_encoder.classes_)
    new_user_vector = np.zeros(num_items)

    for movie_id, rating in user_ratings_dict.items():
        if movie_id in item_encoder.classes_:
            item_idx = item_encoder.transform([movie_id])[0]
            new_user_vector[item_idx] = rating

    # Преобразуем в torch-тензор и прогоняем через encoder
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(new_user_vector, dtype=torch.float32).to(device)
        encoded_vector = model.encoder(input_tensor).cpu().numpy()

    new_user_cluster = kmeans_full_users.predict(encoded_vector.reshape(1, -1))[0]

    # Добавим к user_content_vector
    updated_user_content_vector = np.vstack([user_content_vector, encoded_vector])
    new_user_idx = updated_user_content_vector.shape[0] - 1

    return updated_user_content_vector, new_user_idx


def get_new_user(movie_ids:List, ratings: List) -> pd.DataFrame | int:
    """
    Выдает рекомендации для нового пользователя по его спискапм фильмов и их оценкам
    Атрибуты:
        movie_ids - список фильмов
        ratings - список оценок
    Возвращает рекомендации для нового пользователя и его id
    """

    project_root = Path(__file__).resolve().parent.parent
    params_path = project_root / "params.yaml"
    with open(params_path, "r") as f:
        paths = get_project_paths()

    # Загружаем нужные артефакты
    ratings_csr = load_npz(paths["processed_dir"] / "ratings_csr.npz")
    kmeans_users = joblib.load(paths["models_dir"] / 'kmeans_users.pkl')
    importance_df = pd.DataFrame(columns=['movieId', 'importance_score'])
    movies = pd.read_csv(paths["raw_dir"] / "movies.csv")
    item_encoder = joblib.load(paths["processed_dir"] / "item_encoder.pkl")

    # Собираем рейтинг нового пользователя
    user_ratings_dict = build_user_ratings_dict(movie_ids, ratings)

    # Получаем обновлённый вектор пользователей и индекс нового пользователя
    updated_user_content_vector, new_user_idx = add_new_user_to_system(user_ratings_dict)

    # Генерируем рекомендации
    final_df, similar_users_recommendations_df = get_user_recommendations_new_user(
        user_idx=new_user_idx,
        user_content_vector=updated_user_content_vector,
        ratings_csr=ratings_csr,
        kmeans_users=kmeans_users,
        importance_df=importance_df,
        movies_df=movies,
        item_encoder=item_encoder,
        top_n=50,
        num_user=5,
        is_test=False
    )

    return final_df, new_user_idx



def recommend_by_watched_ids(watched_movie_ids: list, top_k: int = 10)-> pd.DataFrame | List[str]:
    """
    Выдает рекомендации по просмотренным фильмам по сегментированной кластеризации (по контенту)
    Атрибуты:
        watched_movie_ids - список id фильмов
        top_k - неоьходимое кол-во рекомендаций
    Возвращает рекомендации
    """

    project_root = Path(__file__).resolve().parent.parent  # Выходим из scripts/
    params_path = project_root / "params.yaml"
    with open(params_path, "r") as f:
        paths = get_project_paths()

    # 1. Загрузка моделей

    model = joblib.load(paths["models_dir"] / 'final_model.pkl')
    scaler = joblib.load(paths["models_dir"] / 'scaler.pkl')
    mlb = joblib.load(paths["models_dir"] / 'mlb.pkl')
    movies = pd.read_csv(paths["raw_dir"] / "movies.csv")
    ratings = pd.read_csv(paths["raw_dir"] / "ratings.csv")

    # 2. Обработка жанров
    movies = movies.copy()
    movies['genres'] = movies['genres'].apply(lambda x: x.split('|') if isinstance(x, str) else [])
    genre_matrix = pd.DataFrame(mlb.transform(movies['genres']), columns=mlb.classes_, index=movies.index)

    # 3. Средний рейтинг и количество оценок
    agg = ratings.groupby('movieId')['rating'].agg(['mean', 'count']).reset_index()
    agg.columns = ['movieId', 'mean_rating', 'rating_count']
    df = movies.merge(agg, on='movieId', how='left').fillna({'mean_rating': 0, 'rating_count': 0})
    df_full = pd.concat([df[['movieId', 'title', 'mean_rating', 'rating_count']], genre_matrix], axis=1)

    # 4. Признаки и кластеризация
    features = scaler.transform(df_full.drop(columns=['movieId', 'title']))
    df_full['cluster'] = model.predict(features)

    # 5. Определение кластеров просмотренных фильмов
    watched_clusters = df_full[df_full['movieId'].isin(watched_movie_ids)]['cluster'].unique()

    # 6. Фильтрация по кластерам, исключение просмотренных
    candidate_movies = df_full[
        (df_full['cluster'].isin(watched_clusters)) &
        (~df_full['movieId'].isin(watched_movie_ids))
    ]

    # 7. Взвешенный рейтинг
    C = df_full['mean_rating'].mean()
    m = df_full['rating_count'].quantile(0.60)  # можно изменить
    candidate_movies = candidate_movies[candidate_movies['rating_count'] >= 1]  # фильтруем мусор

    v = candidate_movies['rating_count']
    R = candidate_movies['mean_rating']
    candidate_movies['score'] = (v / (v + m)) * R + (m / (v + m)) * C

    recommended = candidate_movies.sort_values(by='score', ascending=False)
    recommendation_info =[]
    for movie_id in watched_movie_ids:
        try:
            original_movie_row = movies[movies['movieId'] == movie_id]
            original_movie_title = original_movie_row['title'].values[0]
            original_movie_genres = original_movie_row['genres'].values[0]

            original_genres_set = (
                set(original_movie_genres.split('|'))
                if isinstance(original_movie_genres, str)
                else set(original_movie_genres)
            )

            recommendation_info.append({
                'movie_id': movie_id,
                'title': original_movie_title,
                'genres': original_genres_set
            })

        except Exception as e:
            print(f"Ошибка обработки фильма {movie_id}: {e}")

    result = recommended[['movieId', 'title', 'mean_rating', 'rating_count', 'score']].head(top_k)

    return result, recommendation_info



def get_recommendations_for_user_streamlit(user_id:int, top_n:int=20)-> pd.DataFrame:
    """
    Функция для получения рекомендаций для пользователя на основе его кластера.
    Загружает все необходимые данные внутри функции.

    :param user_id: ID пользователя, для которого нужно получить рекомендации.
    :param top_n: количество рекомендаций.
    :return: список рекомендованных фильмов.
    """

    project_root = Path(__file__).resolve().parent.parent
    params_path = project_root / "params.yaml"
    with open(params_path, "r") as f:
        paths = get_project_paths()

    user_clusters = pd.read_csv(paths["processed_dir"] / "user_clusters.csv")
    user_ratings = pd.read_csv(paths["raw_dir"] / "ratings.csv")
    movies = pd.read_csv(paths["raw_dir"] / "movies.csv")

    if user_id not in user_clusters["user_id"].values:
        raise ValueError(f"Пользователь с ID {user_id} не найден в кластере.")

    user_cluster = user_clusters[user_clusters["user_id"] == user_id]["cluster"].values[0]
    cluster_users = user_clusters[user_clusters["cluster"] == user_cluster]["user_id"].values
    user_ratings = user_ratings[user_ratings["userId"].isin(cluster_users)]

    # Расчёт средневзвешенного рейтинга
    movie_stats = (
        user_ratings
        .groupby("movieId")
        .agg(avg_rating=("rating", "mean"), rating_count=("rating", "count"))
        .reset_index()
    )
    m = movie_stats["rating_count"].median()
    movie_stats["weighted_rating"] = (
            (movie_stats["rating_count"] / (movie_stats["rating_count"] + m)) * movie_stats["avg_rating"]
    )

    top_movies = movie_stats.sort_values("weighted_rating", ascending=False).head(top_n)
    recommended = top_movies.merge(movies, on="movieId")[
        ["movieId", "title", "avg_rating", "rating_count", "weighted_rating"]]

    return recommended

get_recommendations_for_user_streamlit(1)