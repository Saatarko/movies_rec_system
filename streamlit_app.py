from scripts.recommend import get_rec_on_train_content_vector, get_top_movies, get_rec_on_test_content_vector, \
    get_combine_content_vector, get_rec_on_train_content_vector_raw
from scripts.utils import get_all_genres, get_project_paths, visualize_recommendations_df

import streamlit as st
import pandas as pd



st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")

st.title("🎥 Movie Recommendation System")

with open("params.yaml", "r") as f:
    paths = get_project_paths()

movies = pd.read_csv(paths["raw_dir"] / "movies.csv")
movies = movies

# --- Выбор режима ---
mode = st.sidebar.selectbox("Выбери режим:", [
    "Холодный старт",
    "Контентные рекомендации (тест на тегах)",
    "Контентные рекомендации (тест на фильмах)",
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

elif mode == "Контентные рекомендации (тест на тегах)":
    st.subheader("📚 Контентные рекомендации(тест на тегах)")
    st.info("Предсказание на основе контента с порогом релевантности. Данные разбиты по наборам тегов на обучающую "
            "и тестовую группы. Результаты показывают что совпадения по рекомендациям есть, но слабые из-за слабых тегов")
    stage_rec = False

    # Одна горизонтальная линия: поле + кнопка
    with st.container():
        col1, col2 = st.columns([5, 1])  # Сделал пропорцию 5 к 1 для красоты

        with col1:
            movie_ids_input = st.text_input(
                "Введите ID фильмов через запятую:",
                placeholder="Например: 1, 23, 45",
                label_visibility="collapsed"  # Убираем дублирующий лейбл над полем
            )

        with col2:
            show_recs = st.button("Показать", use_container_width=True)  # Кнопка растягивается на всю ширину колонки

    # Логика обработки после нажатия кнопки
    if show_recs:
        if movie_ids_input:
            try:
                # Преобразуем введённый текст в список чисел
                movie_ids = [int(id_.strip()) for id_ in movie_ids_input.split(",") if id_.strip().isdigit()]

                if movie_ids:
                    stage_rec = True
                    all_recommendations_train, recommendation_info = get_rec_on_train_content_vector(movie_ids)
                    all_recommendations_test, _ = get_rec_on_test_content_vector(movie_ids)
                    all_recommendations_full, _ = get_combine_content_vector(movie_ids)
                else:
                    st.error("Введите хотя бы один корректный ID фильма.")
            except Exception as e:
                st.error(f"Ошибка обработки ввода: {e}")
        else:
            st.error("Пожалуйста, введите хотя бы один ID фильма.")

    # Вывод рекомендаций на всю ширину
    if stage_rec:

        for movie in recommendation_info:
            movie_id = movie['movie_id']
            title = movie['title']
            genres = movie['genres']

            # Формируем строку с нужным форматом
            st.write(f"Получаем рекомендации для фильма с ID: {movie_id} (Название: {title}) (Жанры: {genres})")

        st.markdown("---")  # Разделительная линия для красоты
        st.subheader("🎬 Рекомендации на основе обучающего вектора:")
        st.dataframe(all_recommendations_train, use_container_width=True)

        st.subheader("🎬 Рекомендации на основе тестового вектора:")
        st.dataframe(all_recommendations_test, use_container_width=True)

        st.subheader("🎬 Рекомендации на основе общего вектора:")
        st.dataframe(all_recommendations_full, use_container_width=True)

        st.subheader("Оценка тестирования Train/Test:")
        visualize_recommendations_df(all_recommendations_train, all_recommendations_test)

        st.subheader("Оценка тестирования Train/общий вектор:")
        visualize_recommendations_df(all_recommendations_train, all_recommendations_full)

elif mode == "Контентные рекомендации (тест на фильмах)":
    st.subheader("📚 Контентные рекомендации (тест на фильмах)")
    st.info("Предсказание на основе контента. Данные разбиты по наборам фильмов на обучающую и тестовую группы. ")

    stage_rec = False

    # Одна горизонтальная линия: поле + кнопка
    with st.container():
        col1, col2 = st.columns([5, 1])  # Сделал пропорцию 5 к 1 для красоты

        with col1:
            movie_ids_input = st.text_input(
                "Введите ID фильмов через запятую:",
                placeholder="Например: 1, 23, 45",
                label_visibility="collapsed"  # Убираем дублирующий лейбл над полем
            )

        with col2:
            show_recs = st.button("Показать", use_container_width=True)  # Кнопка растягивается на всю ширину колонки

    # Логика обработки после нажатия кнопки
    if show_recs:
        if movie_ids_input:
            try:
                # Преобразуем введённый текст в список чисел
                movie_ids = [int(id_.strip()) for id_ in movie_ids_input.split(",") if id_.strip().isdigit()]

                if movie_ids:
                    stage_rec = True
                    all_recommendations_df_train, all_recommendations_df_test, recommendation_info= get_rec_on_train_content_vector_raw(movie_ids)

                else:
                    st.error("Введите хотя бы один корректный ID фильма.")
            except Exception as e:
                st.error(f"Ошибка обработки ввода: {e}")
        else:
            st.error("Пожалуйста, введите хотя бы один ID фильма.")

    # Вывод рекомендаций на всю ширину
    if stage_rec:

        for movie in recommendation_info:
            movie_id = movie['movie_id']
            title = movie['title']
            genres = movie['genres']
            # Формируем строку с нужным форматом
            st.write(f"Получаем рекомендации для фильма с ID: {movie_id} (Название: {title}) (Жанры: {genres})")

        st.markdown("---")  # Разделительная линия для красоты
        st.subheader("🎬 Рекомендации на основе обучающего вектора:")
        st.dataframe(all_recommendations_df_train, use_container_width=True)

        st.subheader("🎬 Рекомендации на основе тестового вектора:")
        st.dataframe(all_recommendations_df_test, use_container_width=True)

        st.subheader("Оценка тестирования Train/Test:")
        visualize_recommendations_df(all_recommendations_df_train, all_recommendations_df_test)

elif mode == "Гибридные рекомендации":
    st.subheader("🔀 Гибридные рекомендации")
    st.info("Здесь подключим контент + ALS item factors.")


elif mode == "По пользовательскому вектору":
    st.subheader("👤 Рекомендации на основе предпочтений пользователя")
    st.info("Появятся после оценки нескольких фильмов.")

st.markdown("---")
st.write("📊 В этом блоке позже будет график оффлайн тестирования.")

