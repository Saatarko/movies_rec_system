import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity

from scripts.utils import preprocess_popularity, get_project_paths, predict_test_movie_clusters


def get_top_movies(selected_genres):
    # Получение популярности для холодного старта
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


def recommend_top_movies_by_genres(*genres, movies_df, popularity_df, top_n=10, min_votes=10):
    """
    Рекомендует топ-N фильмов по заданным жанрам, используя взвешенный рейтинг.

    Аргументы:
        *genres: Жанры (один или несколько), как строки.
        movies_df: датафрейм с фильмами и колонками movieId, title, genres.
        popularity_df: датафрейм с ave_rating и rating_count по movieId.
        top_n: сколько фильмов возвращать.
        min_votes: параметр "надежности" рейтинга (m в формуле).
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


# def recommend_movies_by_cluster_filtered(
#         movie_ids,
#         movie_content_vectors,
#         test = False,
#         top_n=50
# ):
#     project_root = Path(__file__).resolve().parent.parent  # Выходим из scripts/
#     params_path = project_root / "params.yaml"
#     with open(params_path, "r") as f:
#         paths = get_project_paths()
#
#     kmeans_model = joblib.load(paths["models_dir"] / 'movie_clusters.pkl')
#     movies_df = pd.read_csv(paths["raw_dir"] / "movies.csv")
#
#     if test is False:
#         cluster_labels = kmeans_model.labels_
#     else:
#         test_movie_clusters = predict_test_movie_clusters(movie_content_vectors)
#         cluster_labels = np.concatenate([kmeans_model.labels_, test_movie_clusters])
#
#     all_recommendations = []
#
#     importance_df = pd.DataFrame(columns=['movieId', 'importance_score'])
#
#     for movie_id in movie_ids:
#
#         original_movie_row = movies_df[movies_df['movieId'] == movie_id]
#         original_movie_title = original_movie_row['title'].values[0]
#         original_movie_genres = original_movie_row['genres'].values[0]
#         original_genres_set = set(original_movie_genres.split('|'))
#
#         print(
#             f"Получаем рекомендации для фильма с ID: {movie_id} (Название: {original_movie_title}) (Жанры: {original_genres_set})")
#
#         try:
#             movie_idx = movies_df[movies_df['movieId'] == movie_id].index[0]
#             movie_title = movies_df.iloc[movie_idx]['title']
#             movie_genres = set(movies_df.iloc[movie_idx]['genres'].split('|'))
#
#             cluster_label = cluster_labels[movie_idx]
#             same_cluster_indices = np.where(cluster_labels == cluster_label)[0]
#             same_cluster_indices = same_cluster_indices[same_cluster_indices != movie_idx]
#
#             if len(same_cluster_indices) == 0:
#                 continue
#
#             distances = euclidean_distances(
#                 movie_content_vectors[movie_idx].reshape(1, -1),
#                 movie_content_vectors[same_cluster_indices]
#             )[0]
#             similarities = 1 / (1 + distances)
#
#             for idx, score in zip(same_cluster_indices, similarities):
#                 title = movies_df.iloc[idx]['title']
#                 genres = movies_df.iloc[idx]['genres']
#                 genres_set = set(genres.split('|'))
#
#                 if pd.isna(title) or pd.isna(genres):
#                     continue
#
#                 if not movie_genres.intersection(genres_set):
#                     score = 0  # Жестко, но ок для фильтрации
#
#                 all_recommendations.append({
#                     'movie_id': movies_df.iloc[idx]['movieId'],
#                     'title': title,
#                     'genres': genres,
#                     'similarity_score': score
#                 })
#         except IndexError:
#             continue
#
#     all_recommendations_df = pd.DataFrame(all_recommendations)
#     if all_recommendations_df.empty:
#         return all_recommendations_df
#
#     all_recommendations_df = all_recommendations_df.groupby(['movie_id', 'title', 'genres']).agg(
#         {'similarity_score': 'sum'}).reset_index()
#
#     # Получаем список фильмов, которые нужно заблокировать
#     drop_film = importance_df[importance_df['importance_score'] == -1]['movieId']
#
#     # Преобразуем drop_film в числовой тип,
#     drop_film = drop_film.astype(int)  # Если movie_id в final_recs - int
#
#     # Удаляем фильмы, которые имеют importance_score == -1
#     new_recs = all_recommendations_df[~all_recommendations_df['movie_id'].isin(drop_film)]
#
#     all_recommendations_df = new_recs.sort_values('similarity_score', ascending=False).head(top_n)
#
#     weights = np.linspace(20, 1, len(all_recommendations_df)).round()
#     all_recommendations_df['weight'] = weights
#
#     temp_list = importance_df[importance_df['importance_score'] == 1]
#     temp_list = temp_list.rename(columns={'movie_id': 'movieId'})
#
#     # Выполняем слияние с DataFrame movies
#     temp_list = temp_list.merge(
#         movies_df[['movieId', 'title', 'genres']],
#         on='movieId',
#         how='left'
#     ).dropna(subset=['title', 'genres'])  # Удаляем строки без названия или жанра
#
#     # Выводим горячие новинки
#     if len(temp_list) != 0:
#         print('Горячие новинки')
#         print('-' * 50)
#         print(temp_list[['movieId', 'title', 'genres']])
#
#     return all_recommendations_df

def recommend_movies_by_cluster_filtered(
        movie_ids,
        movie_content_vectors,
        test=False,
        top_n=50
):
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

def get_rec_on_train_content_vector(movie_ids):
    project_root = Path(__file__).resolve().parent.parent  # Выходим из scripts/
    params_path = project_root / "params.yaml"
    with open(params_path, "r") as f:
        paths = get_project_paths()

    movie_content_vectors_train = np.load(paths["models_dir"] / "movie_content_vectors_train.npz")['vectors']
    all_recommendations_df, recommendation_info = recommend_movies_by_cluster_filtered(movie_ids, movie_content_vectors_train, test = False, top_n=50)

    return all_recommendations_df, recommendation_info


def get_rec_on_test_content_vector(movie_ids):
    project_root = Path(__file__).resolve().parent.parent  # Выходим из scripts/
    params_path = project_root / "params.yaml"
    with open(params_path, "r") as f:
        paths = get_project_paths()

    movie_content_vectors_test = np.load(paths["models_dir"] / "movie_content_vectors_test.npz")['vectors']
    all_recommendations_df, recommendation_info = recommend_movies_by_cluster_filtered(movie_ids, movie_content_vectors_test, test = True, top_n=50)


    return all_recommendations_df, recommendation_info

def get_combine_content_vector(movie_ids):
    project_root = Path(__file__).resolve().parent.parent  # Выходим из scripts/
    params_path = project_root / "params.yaml"
    with open(params_path, "r") as f:
        paths = get_project_paths()

    movie_content_vectors_train = np.load(paths["models_dir"] / "movie_content_vectors_train.npz")['vectors']
    movie_content_vectors_test = np.load(paths["models_dir"] / "movie_content_vectors_test.npz")['vectors']

    combined_movie_content_vectors = np.concatenate([movie_content_vectors_train, movie_content_vectors_test], axis=0)
    all_recommendations_df, recommendation_info = recommend_movies_by_cluster_filtered(movie_ids, combined_movie_content_vectors, test = True, top_n=50)


    return all_recommendations_df, recommendation_info
