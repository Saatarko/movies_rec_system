from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scripts.utils import get_project_paths



with open("params.yaml", "r") as f:
    config = yaml.safe_load(f)["autoencoder"]
    paths = get_project_paths()

# Загружаем данные
genome_scores = pd.read_csv(paths["raw_dir"] / "genome-scores.csv")
ratings = pd.read_csv(paths["raw_dir"] / "ratings.csv")


small_df = ratings.head(10000)
small_df.to_csv("ratings.csv", index=False)

# Строим матрицу тегов
movie_tag_matrix = genome_scores.pivot(index='movieId', columns='tagId', values='relevance').fillna(0)

matrix = movie_tag_matrix.values
movie_ids = movie_tag_matrix.index.values
tag_ids = movie_tag_matrix.columns.values

# Сохраняем массив, индексы и названия колонок
np.savez_compressed(
    "movie_tag_matrix_small.npz",
    matrix=matrix,
    movie_ids=movie_ids,
    tag_ids=tag_ids
)