import pandas as pd
import streamlit as st

from scripts.data_processing import ratings, movies
from scripts.utils import preprocess_popularity


def get_top_movies(selected_genres):
    # Получение популярности для холодного старта
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
