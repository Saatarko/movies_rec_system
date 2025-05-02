# 🌐 Проект: Рекомендательная система

## Введение
**Цель**: Построить рекомендательную систему, способную обрабатывать сценарии холодного старта и динамического добавления пользователей/фильмов.

## Основные задачи
1. **Холодный старт**: юзер выбирает жанр — выдаём top-5 по popularity.
2. **Мало оценок**: несколько просмотренных фильмов — ищем похожие (жанры + теги + relevance).
3. **Мало рейтингов**: <5 оценок, или >5, но <3 положительных.
4. **Полноценные рекомендации**: >5 оценок — кластеризуем user-вектор.
5. **Продвижение/блокировка**: система фильтрации на уровне выдачи.

## 🧰 Используемые технологии
- DVC, Airflow, MLflow
- Scikit-learn (KMeans), Implicit ALS
- Autoencoder, PyTorch (RatingPredictor)
- Интерактивный макет выложен на:
  (*Предупреждение, серввер уходит в спящий режим после 48 часовой неактивности. Любой запрос выведет его из 
этого состояния, но потребуется около 20-30 минут (сервер будет заружать в память датасеты размером около 1.5 Гб))
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
 - Берём только оценки > 3.5 (уместно — это положительное отношение)
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
# 🌐 Project: Recommender System

## Introduction
**Goal**: Build a recommender system capable of handling cold start and dynamic user/movie addition scenarios.

## Main tasks
1. **Cold start**: user selects a genre — we return top-5 by popularity.
2. **Few ratings**: several viewed films — we search for similar ones (genres + tags + relevance).
3. **Few ratings**: <5 ratings, or >5, but <3 positive.
4. **Full recommendations**: >5 ratings — we cluster the user vector.
5. **Promotion/blocking**: a filtering system at the search results level.

## 🧰 Technologies used
- DVC, Airflow, MLflow
- Scikit-learn (KMeans), Implicit ALS
- Autoencoder, PyTorch (RatingPredictor)
- Interactive mockup is available at:
(*Warning, the server goes into sleep mode after 48 hours of inactivity. Any request will wake it up from this state, but it will take about 20-30 minutes (the server will load datasets of about 1.5 GB into memory))
---

## 🔹 Part 1: Content-based model

1. Calculating the average popularity of movies.
2. Creating `movies_vector`:
- Normalization of features (saving `scaler`)
- Dimensionality reduction (`encoder`)
- Clustering (KMeans)
3. Implementing a cluster-based recommendation function.
4. Offline testing.
5. Promoting and filtering movies by a special flag.

---

## 🔸 Part 2: Hybrid (ALS + Content)

6. Forming the interaction matrix.
7. Training Implicit ALS → `item_factors`.
8. Filtering by movies with tags.
9. Combining `item_factors` + `movies_vector`.
10. Dimensionality reduction + clustering (KMeans). Implementing a cluster-based recommendation function
11. Offline testing.
12. Dynamically adding a movie:
- Adding to datasets.
- Vector construction.
- Processing scaler → encoder → KMeans.

---

## 🔹 Part 3: User Model

13. Interaction Matrix Dimensionality Reduction.
14. User Clustering (KMeans). Implementing a Cluster-Based Recommendation Function
15. Dynamic User Addition:
- Adding Ratings.
- Processing encoder → KMeans.

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

## 🔸 Part 5: Supervised clustering (segmentation)

20. Preparation:

✅ Segmentation by movies:
- Genres (One-Hot or Multi-Hot Encoding) and Average rating

✅ Segmentation by users:
- Take only ratings > 3.5 (appropriate - this is a positive attitude)
- Create a profile for each user: preferences by genre, average ratings by movie clusters, rating density by movie types

21. Dimensionality reduction:
- `user_segment_autoencoder` (autoencoder)
- `encoded_user_vectors.npz`

22. Clustering:
- Calculate the required number of clusters using different methods
- Logging in MLflow

19. Recommendation function:
- Segmentation by movies
- Segmentation by users