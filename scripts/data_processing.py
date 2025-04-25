import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
import tensorflow as tf
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


movies = pd.read_csv("data/raw/movies.csv")
ratings = pd.read_csv("data/raw/ratings.csv")
tags = pd.read_csv("data/raw/tags.csv")
genome_tags = pd.read_csv("data/raw/genome-tags.csv")
genome_scores = pd.read_csv("data/raw/genome-scores.csv")
importance_df = pd.DataFrame(columns=['movieId', 'importance_score'])

def generate_content_vector_for_offtest():

    global movies, ratings, tags, genome_tags, genome_scores, importance_df


    movie_tag_matrix = genome_scores.pivot(index='movieId', columns='tagId', values='relevance').fillna(0)

    # Добавим к movie_tag_matrix названия тегов
    tag_id_to_name = genome_tags.set_index('tagId')['tag']
    movie_tag_matrix.columns = movie_tag_matrix.columns.map(tag_id_to_name)

    # Вместо сплита по фильмам — сплит по колонкам (тэгам)
    tag_columns = movie_tag_matrix.columns
    tag_train, tag_test = train_test_split(tag_columns, test_size=0.2, random_state=42)

    # Используем один и тот же набор фильмов
    movie_tag_matrix_train = movie_tag_matrix[tag_train]
    movie_tag_matrix_test = movie_tag_matrix[tag_test]

    # Приводим test к той же структуре, что и train
    movie_tag_matrix_test_aligned = movie_tag_matrix_test.reindex(columns=movie_tag_matrix_train.columns, fill_value=0)

    # Теперь всё сработает
    scaler = MinMaxScaler()
    movie_vectors_scaled_train = scaler.fit_transform(movie_tag_matrix_train)
    movie_vectors_scaled_test = scaler.transform(movie_tag_matrix_test_aligned)

    return movie_vectors_scaled_train, movie_vectors_scaled_test


