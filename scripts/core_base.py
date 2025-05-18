import os

from sqlalchemy import Column, Float, ForeignKey, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

# Basic class for all models
Base = declarative_base()


# Model for the Movies table
class Movie(Base):
    __tablename__ = "movies"

    movieId = Column(Integer, primary_key=True)
    title = Column(String(250), nullable=False)

    # Communication with genres through an intermediate table
    genres = relationship("Genre", secondary="movie_genres", back_populates="movies")

    def __repr__(self):
        return f"<Movie(title={self.title})>"


# Model for Genres table
class Genre(Base):
    __tablename__ = "genres"

    id = Column(Integer, primary_key=True)
    title = Column(String(50), nullable=False)

    # Communication with films through an intermediate table
    movies = relationship("Movie", secondary="movie_genres", back_populates="genres")

    def __repr__(self):
        return f"<Genre(title={self.title})>"


# Intermediate table for communication Movies and Genres (many to many)
class MovieGenres(Base):
    __tablename__ = "movie_genres"

    movieId = Column(Integer, ForeignKey("movies.movieId"), primary_key=True)
    genreId = Column(Integer, ForeignKey("genres.id"), primary_key=True)


# Model for Genome_Tags table
class GenomeTag(Base):
    __tablename__ = "genome_tags"

    tagId = Column(Integer, primary_key=True)
    tag = Column(String(50), nullable=False)

    def __repr__(self):
        return f"<GenomeTag(tag={self.tag})>"


# Model for the Genome_Scores table
class GenomeScore(Base):
    __tablename__ = "genome_scores"

    id = Column(Integer, primary_key=True)
    movieId = Column(Integer, ForeignKey("movies.movieId"), nullable=False)
    tagId = Column(Integer, ForeignKey("genome_tags.tagId"), nullable=False)
    relevance = Column(Float, nullable=False)

    movie = relationship("Movie")
    tag = relationship("GenomeTag")


# Model for the ratings table
class Rating(Base):
    __tablename__ = "ratings"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False)
    movieId = Column(Integer, ForeignKey("movies.movieId"), nullable=False)
    rating = Column(Integer, nullable=False)
    timestamp = Column(Integer, nullable=False)

    movie = relationship("Movie")

    def __repr__(self):
        return f"<Rating(user_id={self.user_id}, movieId={self.movieId}, rating={self.rating})>"


# Model for users table
class User(Base):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True)

    def __repr__(self):
        return f"<User(user_id={self.user_id})>"


# Model for the IMPortance table
class Importance(Base):
    __tablename__ = "importance"

    id = Column(Integer, primary_key=True)
    movieId = Column(Integer, ForeignKey("movies.movieId"), nullable=False)
    value = Column(Integer, nullable=False)

    movie = relationship("Movie")

    def __repr__(self):
        return f"<Importance(movieId={self.movieId}, value={self.value})>"


import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
db_path = os.path.join(PROJECT_ROOT, "movie_recommender.db")

# Create a engine for sqlite
engine = create_engine(f"sqlite:///{db_path}")
Base.metadata.create_all(engine)
