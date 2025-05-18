# 🌐 Проект: Рекомендательная система
Оригиналы датасетов: https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset

---

# Запуск рекомендательной системы

## Вариант 1. Запуск готового интерактива (урезанный функционал)
Предупреждение - сервер может отключаться из-за неактивности, в этом случае первичный запуск может занять 5-10 минут. 
Так же из-за ограничений хранилища датасет пользователей в демоцелях урезан, что может вызывать ошибки при запросах связанных с пользователями.
Для минимизации ошибок при подборе по id пользователя использовать маленькие id

Готовый интерактив: https://huggingface.co/spaces/Saatarkin/rec   


## Вариант 2. Локальный запуск

### 2.1 Установка с git
- Скачать проект
- Скачать готовые вектора, модели и сами датасеты (нужны если проект качается с GIT https://huggingface.co/datasets/Saatarkin/movies/blob/main/movies_rec_system.tar.gz)
- Установить нужные библиотеки (pip install --no-cache-dir -r requirements.txt)
- Запустить командой streamlit run streamlit_app.py

### 2.2 Docker image
- Скачать Docker images: saatarko/rec:latest (образ очень большой, проборос моста будет не очень целесообразным)
- Загрузить себе в контейнер Docker images и запустить контейнер

---

## 🧰 Используемые технологии
- DVC, Airflow, MLflow
- Scikit-learn (KMeans), Implicit ALS
- Autoencoder, PyTorch (RatingPredictor)

---

## Введение
**Цель**: Построить рекомендательную систему, способную обрабатывать сценарии холодного старта и динамического добавления пользователей/фильмов.
---
## Основные задачи
1. **Холодный старт**: юзер выбирает жанр — выдаём top-5 по popularity.
2. **Мало оценок**: несколько просмотренных фильмов — ищем похожие (жанры + теги + relevance).
3. **Мало рейтингов**: <5 оценок, или >5, но <3 положительных.
4. **Полноценные рекомендации**: >5 оценок — кластеризуем user-вектор.
5. **Продвижение/блокировка**: система фильтрации на уровне выдачи.

---

## 🔹 Часть 1: Контент-базированная модель

1. Вычисление средней популярности фильмов.
2. Создание `movies_vector`:
   - Нормализация признаков (сохранение `scaler`)
   - Понижение размерности (`encoder`)
   - Кластеризация (KMeans)
3. Реализация функции рекомендации на основе кластеров.
4. Оффлайн-тестирование.
5. Продвижение и фильтрация фильмов по специальному флагу.

---

## 🔸 Часть 2: Гибрид (ALS + Контент)

6. Формирование матрицы взаимодействий.
7. Обучение Implicit ALS → `item_factors`.
8. Фильтрация по фильмам с тегами.
9. Объединение `item_factors` + `movies_vector`.
10. Понижение размерности + кластеризация (KMeans).  Реализация функции рекомендации на основе кластеров
11. Оффлайн-тестирование.
12. Динамическое добавление фильма:
    - Добавление в датасеты.
    - Построение вектора.
    - Обработка scaler → encoder → KMeans.

---

## 🔹 Часть 3: Пользовательская модель

13. Понижение размерности матрицы взаимодействий.
14. Кластеризация пользователей (KMeans).  Реализация функции рекомендации на основе кластеров
15. Динамическое добавление пользователя:
    - Добавление рейтингов.
    - Обработка encoder → KMeans.

---

## 🔸 Часть 4: RatingPredictor (Нейросеть)

16. Подготовка:
    - Загрузка `ratings.csv`, `movies.csv`
    - Кодировка ID: `user_encoder.pkl`, `item_encoder.pkl`
    - Формирование CSR матрицы: `ratings_csr.npz`

17. Построение эмбеддингов:
    - `user_content_vector.npz` (autoencoder)
    - `model_movies_full_vectors_raw.npz` (теги)

18. Обучение нейросети `RatingPredictor`:
    - Вход: user-вектор + item-вектор
    - Выход: предсказание рейтинга
    - Логирование в MLflow, сохранение `neural_model_best.pt`

19. Функция рекомендации через модель:
    - Использование эмбеддингов + нейросети для персонализированной выдачи.

---
## 🔸 Часть 5: Управляемая кластеризация (сегментация)

20. Подготовка:

✅ Сегментация по фильмам:
- Жанры (One-Hot или Multi-Hot Encoding) и  Средний рейтинг

✅ Сегментация по пользователям:
 - Берём только оценки > 3.5 (положительное отношение)
- Для каждого пользователя составляем профиль: предпочтения по жанрам, средние оценки по кластерам фильмов,  плотность оценок по типам фильмов

21. Понижение размерности:
    - `user_segment_autoencoder` (autoencoder)
    - `encoded_user_vectors.npz` 

22. Кластеризация:
    - Расчет нужного кол-ва кластеров разными методами
    - Логирование в MLflow

19. Функция рекомендации:
    - Сегментация по фильмам
    - Сегментация по пользователям


---
# 🌐 Project: Recommender system
Original datasets: https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset

---

# Launching the recommender system

## Option 1. Launching a ready-made interactive (reduced functionality)
Warning - the server may shut down due to inactivity, in which case the initial launch may take 5-10 minutes.

Also, due to storage limitations, the user dataset in demo targets is truncated, which may cause errors in user-related queries.
To minimize errors when selecting by user ID, use small IDs

Ready-made interactive: https://huggingface.co/spaces/Saatarkin/rec

## Option 2. Local launch

### 2.1 Installation from git
- Download the project
- Download ready-made vectors, models and the datasets themselves (needed if the project is downloaded from GIT https://huggingface.co/datasets/Saatarkin/movies/blob/main/movies_rec_system.tar.gz)
- Install the necessary libraries (pip install --no-cache-dir -r requirements.txt)
- Run with the command streamlit run streamlit_app.py

### 2.2 Docker image
- Download Docker images: saatarko/rec:latest (the image is very large, breaking the bridge will not be very practical)
- Upload Docker images to your container and run the container

---

## 🧰 Technologies used
- DVC, Airflow, MLflow
- Scikit-learn (KMeans), Implicit ALS
- Autoencoder, PyTorch (RatingPredictor)

---

## Introduction
**Goal**: Build a recommender system that can handle cold start and dynamic user/movie addition scenarios.
---
## Main tasks
1. **Cold start**: user selects a genre — output top-5 by popularity.
2. **Few ratings**: several watched movies — search for similar ones (genres + tags + relevance).
3. **Few ratings**: <5 ratings, or >5, but <3 positive.
4. **Full recommendations**: >5 ratings — cluster the user vector.
5. **Promotion/blocking**: filtering system at the search results level.

---

## 🔹 Part 1: Content-based model

1. Calculating average popularity of movies.
2. Creating `movies_vector`:
- Feature normalization (saving `scaler`)
- Dimensionality reduction (`encoder`)
- Clustering (KMeans)
3. Implementing a cluster-based recommendation function.
4. Offline testing.
5. Promoting and filtering movies by a special flag.

---

## 🔸 Part 2: Hybrid (ALS + Content)

6. Forming an interaction matrix.
7. Training Implicit ALS → `item_factors`.
8. Filtering by movies with tags.
9. Merging `item_factors` + `movies_vector`.
10. Dimensionality reduction + clustering (KMeans). Implementation of cluster-based recommendation function
11. Offline testing.
12. Dynamically adding movie:
- Adding to datasets.
- Vector construction.
- Scaler → encoder → KMeans processing.

---

## 🔹 Part 3: User model

13. Interaction matrix dimensionality reduction.
14. User clustering (KMeans). Implementation of cluster-based recommendation function
15. Dynamically adding user:
- Adding ratings.
- Encoder → KMeans processing.

---

## 🔸 Part 4: RatingPredictor (Neural Network)

16. Preparation:
- Loading `ratings.csv`, `movies.csv`
- ID encoding: `user_encoder.pkl`, `item_encoder.pkl`
- Forming CSR matrix: `ratings_csr.npz`

17. Building embeddings:
- `user_content_vector.npz` (autoencoder)
- `model_movies_full_vectors_raw.npz` (tags)

18. Training the `RatingPredictor` neural network:
- Input: user vector + item vector
- Output: rating prediction
- Logging in MLflow, saving `neural_model_best.pt`

19. Model recommendation function:
- Using embeddings + neural networks for personalized results.

---
## 🔸 Part 5: Supervised clustering (segmentation)

20. Preparation:

✅ Segmentation by movies:
- Genres (One-Hot or Multi-Hot Encoding) and Average rating

✅ Segmentation by users:
- We take only ratings > 3.5 (appropriate - this is a positive attitude)
- For each user, we create a profile: preferences by genres, average ratings by movie clusters, rating density by movie types

21. Dimensionality reduction:
- `user_segment_autoencoder` (autoencoder)
- `encoded_user_vectors.npz`

22. Clustering:
- Calculation of the required number of clusters using different methods
- Logging in MLflow

19. Recommendation function:
- Segmentation by movies
- Segmentation by users