import os

import pandas as pd
from core_base import GenomeScore, GenomeTag, Genre, Movie, MovieGenres, Rating, User
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

# Now you can build ways correctly
movies = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "raw", "movies.csv"))
ratings = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "raw", "ratings.csv"))
tags = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "raw", "tags.csv"))
genome_tags = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "raw", "genome-tags.csv"))
genome_scores = pd.read_csv(
    os.path.join(PROJECT_ROOT, "data", "raw", "genome-scores.csv")
)
importance_df = pd.DataFrame(columns=["movieId", "importance_score"])

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
db_path = os.path.join(PROJECT_ROOT, "movie_recommender.db")

# Create a engine for sqlite
engine = create_engine(f"sqlite:///{db_path}")
Session = sessionmaker(bind=engine)


def get_all_genres(movies_df: pd.DataFrame) -> None:
    """
    Writes all unique genres from a DataFrame to the genres table.
    """
    # We get unique genres from Dataframe
    genre_set = set()
    for genre_string in movies_df["genres"]:
        for g in genre_string.split("|"):
            genre_set.add(g.strip())

    # Open the session for recording in the database
    with Session() as session:
        # Add genres to the database
        for genre in genre_set:
            genre_obj = Genre(title=genre)
            session.add(genre_obj)

        # We keep changes in the database
        session.commit()

    print(f"Added {len(genre_set)} genres to the database.")


def populate_movies_table(movies_df: pd.DataFrame):
    """
    Function to fill the movie table in the database
    taking into account genres presented in the string format with the | separator.
    """
    with Session() as session:
        for index, row in movies_df.iterrows():
            movieId = row["movieId"]
            title = row["title"]
            genres = row["genres"].split("|")

            # Adding a film to the Movies table
            movie = Movie(movieId=movieId, title=title)
            session.add(movie)

            # For each genre, we are looking for it in the database
            genre_ids = []
            for genre_name in genres:
                genre_name = genre_name.strip()  # We remove the gaps
                genre = session.query(Genre).filter(Genre.title == genre_name).first()
                genre_ids.append(genre.id)

            # We associate a film with genres through an intermediate table
            for genre_id in genre_ids:
                movie_genre = MovieGenres(movieId=movieId, genreId=genre_id)
                session.add(movie_genre)

        # We fix all the changes
        session.commit()


def get_all_users(ratings_df: pd.DataFrame) -> None:
    """
    Writes all unique userIds from the DataFrame to the users table.
    """
    # We get unique Userid from Dataframe
    users_set = set(ratings_df["userId"])

    # Open the session for recording in the database
    with Session() as session:
        # Add users to the database
        for user in users_set:
            user_obj = User(user_id=user)
            session.add(user_obj)

        # We keep changes in the database
        session.commit()

    print(f"Added {len(users_set)} users to the database.")


def populate_ratings_table(ratings_df: pd.DataFrame, batch_size: int = 5000):
    """
    Function to populate rating table in database using batch write
    """
    ratings_list = []
    for index, row in ratings_df.iterrows():
        ratings_list.append(
            {
                "user_id": row["userId"],
                "movieId": row["movieId"],
                "rating": row["rating"],
                "timestamp": row["timestamp"],
            }
        )

        # If the size of the package is reached, we insert data into the database
        if len(ratings_list) >= batch_size:
            with Session() as session:
                session.bulk_insert_mappings(Rating, ratings_list)
                session.commit()
            ratings_list = []  # We clean the list for the next batch

    # We insert the remaining data
    if ratings_list:
        with Session() as session:
            session.bulk_insert_mappings(Rating, ratings_list)
            session.commit()

    print("Data successfully inserted.")


def populate_genome_tags_table(genome_tags_df: pd.DataFrame, batch_size: int = 5000):
    """
    Function to populate tag table in database using batch write
    """
    taglist = []
    for index, row in genome_tags_df.iterrows():
        taglist.append(
            {
                "tagId": row["tagId"],
                "tag": row["tag"],
            }
        )

        # If the size of the package is reached, we insert data into the database
        if len(taglist) >= batch_size:
            with Session() as session:
                session.bulk_insert_mappings(GenomeTag, taglist)
                session.commit()
            taglist = []  # We clean the list for the next batch

    # We insert the remaining data
    if taglist:
        with Session() as session:
            session.bulk_insert_mappings(GenomeTag, taglist)
            session.commit()

    print("Data successfully inserted.")


def populate_genome_score_table(genome_scores_df: pd.DataFrame, batch_size: int = 5000):
    """
    Function to populate tag table in database using batch write
    """
    tag_score_list = []
    for index, row in genome_scores_df.iterrows():
        tag_score_list.append(
            {
                "movieId": row["movieId"],
                "tagId": row["tagId"],
                "relevance": row["relevance"],
            }
        )

        # If the size of the package is reached, we insert data into the database
        if len(tag_score_list) >= batch_size:
            with Session() as session:
                session.bulk_insert_mappings(GenomeScore, tag_score_list)
                session.commit()
            tag_score_list = []  # We clean the list for the next batch

    # We insert the remaining data
    if tag_score_list:
        with Session() as session:
            session.bulk_insert_mappings(GenomeScore, tag_score_list)
            session.commit()

    print("Data successfully inserted.")


populate_genome_score_table(genome_scores)
