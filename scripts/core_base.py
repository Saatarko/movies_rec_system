import os
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Float
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import pandas as pd

# Базовый класс для всех моделей
Base = declarative_base()


# Модель для таблицы movies
class Movie(Base):
    __tablename__ = 'movies'

    movieId = Column(Integer, primary_key=True)
    title = Column(String(250), nullable=False)

    # Связь с жанрами через промежуточную таблицу
    genres = relationship("Genre", secondary="movie_genres", back_populates="movies")

    def __repr__(self):
        return f"<Movie(title={self.title})>"


# Модель для таблицы genres
class Genre(Base):
    __tablename__ = 'genres'

    id = Column(Integer, primary_key=True)
    title = Column(String(50), nullable=False)

    # Связь с фильмами через промежуточную таблицу
    movies = relationship("Movie", secondary="movie_genres", back_populates="genres")

    def __repr__(self):
        return f"<Genre(title={self.title})>"


# Промежуточная таблица для связи movies и genres (многие ко многим)
class MovieGenres(Base):
    __tablename__ = 'movie_genres'

    movieId = Column(Integer, ForeignKey('movies.movieId'), primary_key=True)
    genreId = Column(Integer, ForeignKey('genres.id'), primary_key=True)


# Модель для таблицы genome_tags
class GenomeTag(Base):
    __tablename__ = 'genome_tags'

    tagId = Column(Integer, primary_key=True)
    tag = Column(String(50), nullable=False)

    def __repr__(self):
        return f"<GenomeTag(tag={self.tag})>"


# Модель для таблицы genome_scores
class GenomeScore(Base):
    __tablename__ = 'genome_scores'

    id = Column(Integer, primary_key=True)
    movieId = Column(Integer, ForeignKey('movies.movieId'), nullable=False)
    tagId = Column(Integer, ForeignKey('genome_tags.tagId'), nullable=False)
    relevance = Column(Float, nullable=False)

    movie = relationship("Movie")
    tag = relationship("GenomeTag")


# Модель для таблицы ratings
class Rating(Base):
    __tablename__ = 'ratings'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False)
    movieId = Column(Integer, ForeignKey('movies.movieId'), nullable=False)
    rating = Column(Integer, nullable=False)
    timestamp = Column(Integer, nullable=False)

    movie = relationship("Movie")

    def __repr__(self):
        return f"<Rating(user_id={self.user_id}, movieId={self.movieId}, rating={self.rating})>"


# Модель для таблицы users
class User(Base):
    __tablename__ = 'users'

    user_id = Column(Integer, primary_key=True)

    def __repr__(self):
        return f"<User(user_id={self.user_id})>"

# Модель для таблицы importance
class Importance(Base):
    __tablename__ = 'importance'

    id = Column(Integer, primary_key=True)
    movieId = Column(Integer, ForeignKey('movies.movieId'), nullable=False)
    value = Column(Integer, nullable=False)

    movie = relationship("Movie")
    def __repr__(self):
        return f"<Importance(movieId={self.movieId}, value={self.value})>"


import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
db_path = os.path.join(PROJECT_ROOT, 'movie_recommender.db')

# Создаем движок для SQLite
engine = create_engine(f'sqlite:///{db_path}')
Base.metadata.create_all(engine)

