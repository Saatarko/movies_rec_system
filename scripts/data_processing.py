import argparse
from pathlib import Path

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
from scipy.sparse import csr_matrix
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
import time
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
from sklearn.metrics import precision_score
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", help="Список задач для выполнения")
    args = parser.parse_args()

    if args.tasks:
        main(args.tasks)  # Здесь передаем задачи, которые указаны в командной строке

