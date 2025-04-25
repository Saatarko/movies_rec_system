import pandas as pd


movies = pd.read_csv("data/raw/movies.csv")
ratings = pd.read_csv("data/raw/ratings.csv")
tags = pd.read_csv("data/raw/tags.csv")
genome_tags = pd.read_csv("data/raw/genome-tags.csv")
genome_scores = pd.read_csv("data/raw/genome-scores.csv")
importance_df = pd.DataFrame(columns=['movieId', 'importance_score'])
