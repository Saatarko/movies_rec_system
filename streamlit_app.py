from scripts.recommend import get_rec_on_train_content_vector, get_top_movies, get_rec_on_test_content_vector, \
    get_combine_content_vector, get_rec_on_train_content_vector_raw, get_als_and_content_vector, add_new_films, \
    get_recommendation_on_user_vector
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
    "Добавление нового фильма",
    "Гибридные рекомендации",
    "По пользовательскому вектору",
    "Гибридные рекомендации на нейросети",
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
    st.info("Рекомендации по просмотренным фильмам на основе контента с порогом релевантности (простое сходство). Данные разбиты по наборам тегов на обучающую "
            "и тестовую группы.")
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

        st.subheader("Оффлайн тестирование Train/Test:")
        visualize_recommendations_df(all_recommendations_train, all_recommendations_test)

        st.subheader("📊 Сравнение рекомендаций Train/общий вектор:")
        visualize_recommendations_df(all_recommendations_train, all_recommendations_full)

elif mode == "Контентные рекомендации (тест на фильмах)":
    st.subheader("📚 Контентные рекомендации (тест на фильмах)")
    st.info("Рекомендации по просмотренным фильмам на основе контентного вектора (простое сходство). Данные разбиты по наборам фильмов на обучающую и тестовую группы. ")

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

        st.subheader("📊 Оффлайн тестирование Train/Test:")
        visualize_recommendations_df(all_recommendations_df_train, all_recommendations_df_test)

elif mode == "Гибридные рекомендации":
    st.subheader("🔀 Гибридные рекомендации")
    st.info("Рекомендации по просмотренным фильмам на основе контентного вектора + ALS item factors. (простое сходство)")

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
                    all_recommendations_hybrid,  recommendation_info_hybrid = get_als_and_content_vector(
                        movie_ids)
                    all_recommendations_train, recommendation_info = get_rec_on_train_content_vector(movie_ids)

                else:
                    st.error("Введите хотя бы один корректный ID фильма.")
            except Exception as e:
                st.error(f"Ошибка обработки ввода: {e}")
        else:
            st.error("Пожалуйста, введите хотя бы один ID фильма.")

    # Вывод рекомендаций на всю ширину
    if stage_rec:

        for movie in recommendation_info_hybrid:
            movie_id = movie['movie_id']
            title = movie['title']
            genres = movie['genres']
            # Формируем строку с нужным форматом
            st.write(f"Получаем рекомендации для фильма с ID: {movie_id} (Название: {title}) (Жанры: {genres})")

        st.markdown("---")  # Разделительная линия для красоты
        st.subheader("🎬 Рекомендации на основе гибридного вектора:")
        st.dataframe(all_recommendations_hybrid, use_container_width=True)

        st.subheader("🎬 Рекомендации на основе контентного вектора:")
        st.dataframe(all_recommendations_train, use_container_width=True)

        st.subheader("📊 Сравнение рекомендаций на гибридном векторе / чистом контентном векторе:")
        visualize_recommendations_df(all_recommendations_hybrid, all_recommendations_train)

elif mode == "По пользовательскому вектору":
    st.subheader("👤 Рекомендации на основе предпочтений пользователя")
    st.info("Рекомендации по ID пользователя по пользовательскому вектору с разбивкой для оффлайн тестирования  (простое сходство)")

    stage_rec = False

    # Одна горизонтальная линия: поле + кнопка
    with st.container():
        col1, col2 = st.columns([5, 1])  # Сделал пропорцию 5 к 1 для красоты

        with col1:
            user_id = st.text_input(
                "Введите id пользователя для поиска рекомендации:",
                placeholder="Например: 1",
                label_visibility="collapsed"  # Убираем дублирующий лейбл над полем
            )

        with col2:
            show_recs = st.button("Показать", use_container_width=True)  # Кнопка растягивается на всю ширину колонки

    # Логика обработки после нажатия кнопки
    if show_recs:
        if user_id:
            try:
                # Преобразуем введённый текст в список чисел
                user_id = int(user_id)

                if user_id:
                    stage_rec = True
                    final_df, final_df_test = get_recommendation_on_user_vector(user_id)
                else:
                    st.error("Введите хотя бы один корректный ID пользователя.")
            except Exception as e:
                st.error(f"Ошибка обработки ввода: {e}")
        else:
            st.error("Пожалуйста, введите ID пользователя.")

    # Вывод рекомендаций на всю ширину
    if stage_rec:


        st.markdown("---")  # Разделительная линия для красоты
        st.subheader(f"🎬 Рекомендации для пользователя {user_id} на основе train выборки:")
        st.dataframe(final_df, use_container_width=True)

        st.subheader(f"🎬 Рекомендации для пользователя {user_id} на основе test выборки")
        st.dataframe(final_df_test, use_container_width=True)

        st.subheader("📊 Результаты оффлайн тестирования на train / test выборке:")
        visualize_recommendations_df(final_df, final_df_test)

elif mode == "Добавление нового фильма":
    st.subheader("👤 Добавление нового фильма")
    st.info(
        "Дайте название новому фильму и выберите жанры. Теги в силу их большого кол-ва (более 1000) и сомнительного "
        "качества в качестве демонстрации будут присвоены автоматически. После добавления нового фильма будут выданы рекомендации по нему.")

    # Ввод данных для нового фильма
    st.subheader("➕ Добавление нового фильма")
    new_movie_title = st.text_input("Введите название нового фильма:", "Test")
    selected_genres = st.multiselect("Выберите жанры:", options=sorted(get_all_genres(movies)),
                                     default=["Action", "Comedy"])

    # Проверка наличия жанров
    if not selected_genres:
        st.info("ℹ️ Выберите хотя бы один жанр.")

    if st.button("Добавить фильм"):
        # Добавление фильма и получение рекомендаций
        all_recommendations_hybrid, recommendation_info_hybrid = add_new_films(new_movie_title, selected_genres)

        if all_recommendations_hybrid is not None:
            for movie in recommendation_info_hybrid:
                st.write(
                    f"Получаем рекомендации для фильма с ID: {movie['movie_id']} (Название: {movie['title']}) (Жанры: {movie['genres']})")
            # Вывод рекомендаций
            st.subheader("🎬 Рекомендации по новому фильму на основе гибридного вектора:")
            st.dataframe(all_recommendations_hybrid, use_container_width=True)

elif mode == "Гибридные рекомендации на нейросети":
    st.subheader("👤 Гибридные рекомендации на нейросети")
    st.info(
        "Рекомендации по ID пользователя по готовым контентному и пользовательскому векторам (через нейросеть)")

    stage_rec = False

    # Одна горизонтальная линия: поле + кнопка
    with st.container():
        col1, col2 = st.columns([5, 1])  # Сделал пропорцию 5 к 1 для красоты

        with col1:
            user_id = st.text_input(
                "Введите id пользователя для поиска рекомендации:",
                placeholder="Например: 1",
                label_visibility="collapsed"  # Убираем дублирующий лейбл над полем
            )

        with col2:
            show_recs = st.button("Показать",
                                  use_container_width=True)  # Кнопка растягивается на всю ширину колонки

    # Логика обработки после нажатия кнопки
    if show_recs:
        if user_id:
            try:
                # Преобразуем введённый текст в список чисел
                user_id = int(user_id)

                if user_id:
                    stage_rec = True
                    final_nn = recommend_for_user_df(user_id, top_k=50)
                    final_user, final_df_test = get_recommendation_on_user_vector(user_id)
                else:
                    st.error("Введите хотя бы один корректный ID пользователя.")
            except Exception as e:
                st.error(f"Ошибка обработки ввода: {e}")
        else:
            st.error("Пожалуйста, введите ID пользователя.")

    # Вывод рекомендаций на всю ширину
    if stage_rec:
        st.markdown("---")  # Разделительная линия для красоты
        st.subheader(f"🎬 Рекомендации для пользователя {user_id} на основе предсказаний нейросети:")
        st.dataframe(final_nn, use_container_width=True)

        st.subheader(f"📊 Сравнение рекомендаций для пользователя {user_id} на основе предсказаний нейросети  с простым сходством на пользовательком векторе:")
        visualize_recommendations_df(final_nn, final_user)

        st.markdown("---")  # Разделительная линия

st.markdown("---")


