import pandas as pd

def preprocess_popularity(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Возвращает DataFrame с movieId и средним рейтингом для каждого фильма.
    """
    popularity_df = ratings_df.groupby("movieId").agg(ave_rating=("rating", "mean"),
                                                      rating_count=("rating", "count")).reset_index()
    return popularity_df



def get_all_genres(movies_df: pd.DataFrame) -> list:
    """
    Возвращает уникальные жанры из всех фильмов.
    """
    genre_set = set()
    for genre_string in movies_df["genres"]:
        for g in genre_string.split("|"):
            genre_set.add(g.strip())
    return sorted(genre_set)