import os
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Float
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import pandas as pd

from core_base import Base, Movie, Genre, GenomeTag, Rating, MovieGenres, GenomeScore, Importance, User

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

# Теперь можно строить пути правильно
movies = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "raw", "movies.csv"))
ratings = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "raw", "ratings.csv"))
tags = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "raw", "tags.csv"))
genome_tags = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "raw", "genome-tags.csv"))
genome_scores = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "raw", "genome-scores.csv"))
importance_df = pd.DataFrame(columns=['movieId', 'importance_score'])

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
db_path = os.path.join(PROJECT_ROOT, 'movie_recommender.db')

# Создаем движок для SQLite
engine = create_engine(f'sqlite:///{db_path}')
Session = sessionmaker(bind=engine)


def get_all_genres(movies_df: pd.DataFrame) -> None:
    """
    Записывает все уникальные жанры из DataFrame в таблицу genres.
    """
    # Получаем уникальные жанры из DataFrame
    genre_set = set()
    for genre_string in movies_df["genres"]:
        for g in genre_string.split("|"):
            genre_set.add(g.strip())

    # Открываем сессию для записи в базу данных
    with Session() as session:
        # Добавляем жанры в базу данных
        for genre in genre_set:
            genre_obj = Genre(title=genre)
            session.add(genre_obj)

        # Сохраняем изменения в базе данных
        session.commit()

    print(f"Added {len(genre_set)} genres to the database.")


def populate_movies_table(movies_df: pd.DataFrame):
    """
    Функция для заполнения таблицы фильмов в базе данных
    с учетом жанров, представленных в формате строки с разделителем |.
    """
    with Session() as session:
        for index, row in movies_df.iterrows():
            movieId = row['movieId']
            title = row['title']
            genres = row['genres'].split('|')

            # Добавление фильма в таблицу movies
            movie = Movie(movieId=movieId, title=title)
            session.add(movie)

            # Для каждого жанра ищем его в базе
            genre_ids = []
            for genre_name in genres:
                genre_name = genre_name.strip()  # Убираем пробелы
                genre = session.query(Genre).filter(Genre.title == genre_name).first()
                genre_ids.append(genre.id)

            # Связываем фильм с жанрами через промежуточную таблицу
            for genre_id in genre_ids:
                movie_genre = MovieGenres(movieId=movieId, genreId=genre_id)
                session.add(movie_genre)

        # Фиксируем все изменения
        session.commit()

def get_all_users(ratings_df: pd.DataFrame) -> None:
    """
    Записывает все уникальные userId из DataFrame в таблицу users.
    """
    # Получаем уникальные userId из DataFrame
    users_set = set(ratings_df["userId"])

    # Открываем сессию для записи в базу данных
    with Session() as session:
        # Добавляем пользователей в базу данных
        for user in users_set:
            user_obj = User(user_id=user)
            session.add(user_obj)

        # Сохраняем изменения в базе данных
        session.commit()

    print(f"Added {len(users_set)} users to the database.")


def populate_ratings_table(ratings_df: pd.DataFrame, batch_size: int = 5000):
    """
    Функция для заполнения таблицы рейтингов в базе данных с использованием пакетной записи
    """
    ratings_list = []
    for index, row in ratings_df.iterrows():
        ratings_list.append({
            'user_id': row['userId'],
            'movieId': row['movieId'],
            'rating': row['rating'],
            'timestamp': row['timestamp']
        })

        # Если размер пакета достигнут, вставляем данные в базу
        if len(ratings_list) >= batch_size:
            with Session() as session:
                session.bulk_insert_mappings(Rating, ratings_list)
                session.commit()
            ratings_list = []  # Очищаем список для следующей партии

    # Вставляем оставшиеся данные
    if ratings_list:
        with Session() as session:
            session.bulk_insert_mappings(Rating, ratings_list)
            session.commit()

    print("Data successfully inserted.")


def populate_genome_tags_table(genome_tags_df: pd.DataFrame, batch_size: int = 5000):
    """
    Функция для заполнения таблицы тегов в базе данных с использованием пакетной записи
    """
    taglist = []
    for index, row in genome_tags_df.iterrows():
        taglist.append({
            'tagId': row['tagId'],
            'tag': row['tag'],
        })

        # Если размер пакета достигнут, вставляем данные в базу
        if len(taglist) >= batch_size:
            with Session() as session:
                session.bulk_insert_mappings(GenomeTag, taglist)
                session.commit()
            taglist = []  # Очищаем список для следующей партии

    # Вставляем оставшиеся данные
    if taglist:
        with Session() as session:
            session.bulk_insert_mappings(GenomeTag, taglist)
            session.commit()

    print("Data successfully inserted.")


def populate_genome_score_table(genome_scores_df: pd.DataFrame, batch_size: int = 5000):
    """
    Функция для заполнения таблицы тегов в базе данных с использованием пакетной записи
    """
    tag_score_list = []
    for index, row in genome_scores_df.iterrows():
        tag_score_list.append({
            'movieId': row['movieId'],
            'tagId': row['tagId'],
            'relevance': row['relevance'],
        })

        # Если размер пакета достигнут, вставляем данные в базу
        if len(tag_score_list) >= batch_size:
            with Session() as session:
                session.bulk_insert_mappings(GenomeScore, tag_score_list)
                session.commit()
            tag_score_list = []  # Очищаем список для следующей партии

    # Вставляем оставшиеся данные
    if tag_score_list:
        with Session() as session:
            session.bulk_insert_mappings(GenomeScore, tag_score_list)
            session.commit()

    print("Data successfully inserted.")


populate_genome_score_table(genome_scores)