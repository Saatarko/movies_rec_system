import argparse
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from scipy.sparse import csr_matrix, save_npz, load_npz
import implicit
from implicit.als import AlternatingLeastSquares
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.decomposition import NMF, TruncatedSVD
from scipy.sparse import coo_matrix
from tqdm import tqdm
import json
import os
import math
import glob
import dill
import matplotlib.pyplot as plt
import hdbscan
from sklearn.model_selection import train_test_split
import psutil
from time import time
import tracemalloc
import multiprocessing as mp
from sklearn.cluster import DBSCAN
import traceback
import sys
import gc
from sklearn.cluster import SpectralClustering
from multiprocessing import Process, Queue
from sklearn.mixture import GaussianMixture
import joblib
from sklearn.cluster import KMeans
import seaborn as sns
from implicit.nearest_neighbours import bm25_weight
from datetime import datetime
import lightgbm as lgb
import rbo
from matplotlib_venn import venn2
from sklearn.metrics import precision_score, mean_squared_error
from collections import defaultdict, Counter, OrderedDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.sparse import hstack
import umap

from utils import get_project_paths
from task_registry import task, main
import yaml

# Путь к корню проекта (там где data/, scripts/ и так далее)
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
mlflow.set_tracking_uri("http://localhost:5000")

@task("data:generate_content_vector")
def generate_content_vector_for_offtest():
    # Путь к корню проекта (там где data/, scripts/ и так далее)
    paths = get_project_paths()
    genome_tags = pd.read_csv(paths["raw_dir"] / "genome-tags.csv")
    genome_scores = pd.read_csv(paths["raw_dir"] / "genome-scores.csv")

    relevance_threshold = genome_scores['relevance'].quantile(0.75)
    high_relevance_scores = genome_scores[genome_scores['relevance'] >= relevance_threshold]

    # Формируем матрицу movie_tag_matrix только с высокорелевантными тегами
    movie_tag_matrix_filtered = high_relevance_scores.pivot(index='movieId', columns='tagId',
                                                            values='relevance').fillna(0)

    # Сопоставление tagId с именами тегов
    tag_id_to_name = genome_tags.set_index('tagId')['tag']
    movie_tag_matrix_filtered.columns = movie_tag_matrix_filtered.columns.map(tag_id_to_name)

    # Разделяем данные на обучающие и тестовые
    tag_columns = movie_tag_matrix_filtered.columns
    tag_train, tag_test = train_test_split(tag_columns, test_size=0.2, random_state=42)

    movie_tag_matrix_train = movie_tag_matrix_filtered[tag_train]
    movie_tag_matrix_test = movie_tag_matrix_filtered[tag_test]
    movie_tag_matrix_test_aligned = movie_tag_matrix_test.reindex(columns=movie_tag_matrix_train.columns, fill_value=0)

    # Нормализация
    scaler = MinMaxScaler()
    movie_vectors_scaled_train = scaler.fit_transform(movie_tag_matrix_train)
    movie_vectors_scaled_test = scaler.transform(movie_tag_matrix_test_aligned)

    # os.makedirs((paths["processed_dir"]), exist_ok=True)
    np.save((paths["processed_dir"] / "movie_vectors_scaled_train.npy"), movie_vectors_scaled_train)
    np.save((paths["processed_dir"] / "movie_vectors_scaled_test.npy"), movie_vectors_scaled_test)

    print("Files saved successfully!")

    return movie_vectors_scaled_train, movie_vectors_scaled_test

@task("data:generate_content_vector_raw")
def generate_content_vector_raw():
    paths = get_project_paths()
    genome_tags = pd.read_csv(paths["raw_dir"] / "genome-tags.csv")
    genome_scores = pd.read_csv(paths["raw_dir"] / "genome-scores.csv")

    movie_tag_matrix = genome_scores.pivot(index='movieId', columns='tagId', values='relevance').fillna(0)

    # Добавим к movie_tag_matrix названия тегов
    tag_id_to_name = genome_tags.set_index('tagId')['tag']
    movie_tag_matrix.columns = movie_tag_matrix.columns.map(tag_id_to_name)

    scaler_full_vector = MinMaxScaler()
    movie_vectors_scaled = scaler_full_vector.fit_transform(movie_tag_matrix)

    np.save((paths["processed_dir"] / "movie_vectors_scaled_full.npy"), movie_vectors_scaled)

    return movie_vectors_scaled

@task("data:generate_ratings_matrix")
def generate_and_save_ratings_matrix():
    # Пути
    paths = get_project_paths()
    ratings = pd.read_csv(paths["raw_dir"] / "ratings.csv")

    # Энкодеры
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    ratings['user_idx'] = user_encoder.fit_transform(ratings['userId'])
    ratings['item_idx'] = item_encoder.fit_transform(ratings['movieId'])

    # Сохранение энкодеров
    joblib.dump(user_encoder, paths["processed_dir"] / "user_encoder.pkl")
    joblib.dump(item_encoder, paths["processed_dir"] / "item_encoder.pkl")

    # COO и CSR матрицы
    ratings_coo = coo_matrix(
        (ratings['rating'].astype(float), (ratings['user_idx'], ratings['item_idx']))
    )
    ratings_csr = ratings_coo.tocsr()

    # Сохранение CSR
    save_npz(paths["processed_dir"] / "ratings_csr.npz", ratings_csr)

    print(f"Матрица размера {ratings_csr.shape} сохранена в processed_dir/ratings_csr.npz")


@task("data:generate_als_vectors")
def generate_als_vectors():
    with mlflow.start_run(run_name="ALS full training"):
        paths = get_project_paths()
        ratings_csr = load_npz(paths["processed_dir"] / "ratings_csr.npz")

        # Параметры модели
        factors = 64
        reg = 0.1
        iters = 15

        mlflow.log_param("factors", factors)
        mlflow.log_param("regularization", reg)
        mlflow.log_param("iterations", iters)

        mlflow.set_tag("model_type", "ALS")
        mlflow.set_tag("library", "implicit")
        mlflow.set_tag("format", "joblib + npy")

        start_time = time()

        als_model_full = AlternatingLeastSquares(factors=factors, regularization=0.1, iterations=15)
        als_model_full.fit(ratings_csr.T)

        training_time = time() - start_time
        mlflow.log_metric("training_time_seconds", training_time)

        # GPU → NumPy
        item_factors_np = als_model_full.item_factors.to_numpy()
        user_factors_np = als_model_full.user_factors.to_numpy()

        np.save(paths["models_dir"] / "item_factors.npy", item_factors_np)
        joblib.dump(als_model_full, paths["models_dir"] / "als_model.pkl")

        mlflow.log_artifact(str(paths["models_dir"] / "item_factors.npy"))
        mlflow.log_artifact(str(paths["models_dir"] / "als_model.pkl"))

@task("data:combine_content_als_vector")
def combine_content_als_vector():

    paths = get_project_paths()
    genome_scores = pd.read_csv(paths["raw_dir"] / "genome-scores.csv")

    movie_tag_matrix = genome_scores.pivot(index='movieId', columns='tagId', values='relevance').fillna(0)
    movie_ids_with_tags = movie_tag_matrix.index.to_numpy()

    item_factors = np.load(paths["models_dir"] / "item_factors.npy")
    item_encoder = joblib.load(paths["processed_dir"] / 'item_encoder.pkl')
    # Получаем индексы этих movieId'ов в item_matrix с помощью item_encoder

    item_indices = item_encoder.transform(movie_ids_with_tags)

    # Отбираем из item_matrix только нужные строки
    filtered_item_matrix_full = item_factors[item_indices]
    model_movies_full_vectors_raw = np.load(paths["models_dir"] / "model_movies_full_vectors_raw.npz")['vectors']

    # Объединяем по признакам (горизонтально)
    hybrid_movie_vector_full = np.hstack([
        model_movies_full_vectors_raw,
        filtered_item_matrix_full
    ])
    np.savez_compressed(paths["models_dir"] /"hybrid_movie_vector_full.npz", vectors=hybrid_movie_vector_full)

    return hybrid_movie_vector_full


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", help="Список задач для выполнения")
    args = parser.parse_args()

    if args.tasks:
        main(args.tasks)  # Здесь передаем задачи, которые указаны в командной строке

