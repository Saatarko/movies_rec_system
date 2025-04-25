import sys
import os
from scripts.data_processing import ratings, movies
from scripts.recommend import recommend_top_movies_by_genres, get_top_movies
from scripts.utils import preprocess_popularity, get_all_genres

import streamlit as st
import pandas as pd



st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")

st.title("🎥 Movie Recommendation System")


movies = movies

# --- Выбор режима ---
mode = st.sidebar.selectbox("Выбери режим:", [
    "Холодный старт",
    "Контентные рекомендации",
    "Гибридные рекомендации",
    "По пользовательскому вектору"
])

st.sidebar.markdown("---")

# --- Поиск по ID ---
movie_id = st.sidebar.number_input("Введите Movie ID", min_value=1, value=1)
if movie_id in movies["movieId"].values:
    st.sidebar.success(f"Название: {movies[movies['movieId'] == movie_id]['title'].values[0]}")
else:
    st.sidebar.warning("Фильм с таким ID не найден.")

# --- Основная логика ---
if mode == "Холодный старт":
    st.subheader("🧊 Режим холодного старта")
    st.info("Вы можете получить рекомендации по жанру ранжированные по популярности.")

    selected_genres = st.multiselect("Выберите жанры:", options=get_all_genres(movies),
                                     default=["Action", "Comedy"])
    # Кнопка для выполнения действий
    if st.button("Сформировать рекомендации"):

        get_top_movies(selected_genres)

elif mode == "Контентные рекомендации":
    st.subheader("📚 Контентные рекомендации")
    st.info("Здесь будет поиск фильмов на основе похожести.")

elif mode == "Гибридные рекомендации":
    st.subheader("🔀 Гибридные рекомендации")
    st.info("Здесь подключим контент + ALS item factors.")

elif mode == "По пользовательскому вектору":
    st.subheader("👤 Рекомендации на основе предпочтений пользователя")
    st.info("Появятся после оценки нескольких фильмов.")

st.markdown("---")
st.write("📊 В этом блоке позже будет график оффлайн тестирования.")

