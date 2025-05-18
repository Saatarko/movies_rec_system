import argparse
import os
from time import time

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from scipy.sparse import coo_matrix, csr_matrix, load_npz, save_npz
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    MultiLabelBinarizer,
    StandardScaler,
)
from task_registry import main, task
from utils import get_project_paths

# The path to the root of the project (where Data/, scripts/ and so on)
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
mlflow.set_tracking_uri("http://localhost:5000")


@task("data:generate_content_vector")
def generate_content_vector_for_offtest():
    """
    Collects and normalizes the content vector (for columnar splitting)
    """

    # The path to the root of the project (where Data/, scripts/ and so on)
    paths = get_project_paths()
    genome_tags = pd.read_csv(paths["raw_dir"] / "genome-tags.csv")
    genome_scores = pd.read_csv(paths["raw_dir"] / "genome-scores.csv")

    relevance_threshold = genome_scores["relevance"].quantile(0.75)
    high_relevance_scores = genome_scores[
        genome_scores["relevance"] >= relevance_threshold
    ]

    # We form the matrix movie_tag_matrix only with highly sting tags
    movie_tag_matrix_filtered = high_relevance_scores.pivot(
        index="movieId", columns="tagId", values="relevance"
    ).fillna(0)

    # Tagid comparison with tag names
    tag_id_to_name = genome_tags.set_index("tagId")["tag"]
    movie_tag_matrix_filtered.columns = movie_tag_matrix_filtered.columns.map(
        tag_id_to_name
    )

    # We share data into training and test
    tag_columns = movie_tag_matrix_filtered.columns
    tag_train, tag_test = train_test_split(tag_columns, test_size=0.2, random_state=42)

    movie_tag_matrix_train = movie_tag_matrix_filtered[tag_train]
    movie_tag_matrix_test = movie_tag_matrix_filtered[tag_test]
    movie_tag_matrix_test_aligned = movie_tag_matrix_test.reindex(
        columns=movie_tag_matrix_train.columns, fill_value=0
    )

    # Normalization
    scaler = MinMaxScaler()
    movie_vectors_scaled_train = scaler.fit_transform(movie_tag_matrix_train)
    movie_vectors_scaled_test = scaler.transform(movie_tag_matrix_test_aligned)

    # OS.MAKEDIRS ((Paths ["Processed_dir"]), Exist_ok = True)
    np.save(
        (paths["processed_dir"] / "movie_vectors_scaled_train.npy"),
        movie_vectors_scaled_train,
    )
    np.save(
        (paths["processed_dir"] / "movie_vectors_scaled_test.npy"),
        movie_vectors_scaled_test,
    )

    print("Files saved successfully!")

    return movie_vectors_scaled_train, movie_vectors_scaled_test


@task("data:generate_content_vector_raw")
def generate_content_vector_raw():
    """
    Collects and normalizes the content vector (with row-by-row splitting)
    """

    paths = get_project_paths()
    genome_tags = pd.read_csv(paths["raw_dir"] / "genome-tags.csv")
    genome_scores = pd.read_csv(paths["raw_dir"] / "genome-scores.csv")

    movie_tag_matrix = genome_scores.pivot(
        index="movieId", columns="tagId", values="relevance"
    ).fillna(0)

    # Add to movie_tag_matrix names of tags
    tag_id_to_name = genome_tags.set_index("tagId")["tag"]
    movie_tag_matrix.columns = movie_tag_matrix.columns.map(tag_id_to_name)

    scaler_full_vector = MinMaxScaler()
    movie_vectors_scaled = scaler_full_vector.fit_transform(movie_tag_matrix)

    np.save(
        (paths["processed_dir"] / "movie_vectors_scaled_full.npy"), movie_vectors_scaled
    )

    return movie_vectors_scaled


@task("data:generate_ratings_matrix")
def generate_and_save_ratings_matrix():
    """
    Collects, encodes and normalizes the interaction matrix
    """

    # Ways
    paths = get_project_paths()
    ratings = pd.read_csv(paths["raw_dir"] / "ratings.csv")

    # Encoders
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    ratings["user_idx"] = user_encoder.fit_transform(ratings["userId"])
    ratings["item_idx"] = item_encoder.fit_transform(ratings["movieId"])

    # Preservation of encoders
    joblib.dump(user_encoder, paths["processed_dir"] / "user_encoder.pkl")
    joblib.dump(item_encoder, paths["processed_dir"] / "item_encoder.pkl")

    # COO and CSR matrices
    ratings_coo = coo_matrix(
        (ratings["rating"].astype(float), (ratings["user_idx"], ratings["item_idx"]))
    )
    ratings_csr = ratings_coo.tocsr()

    # Conservation CSR
    save_npz(paths["processed_dir"] / "ratings_csr.npz", ratings_csr)

    print(
        f"Матрица размера {ratings_csr.shape} сохранена в processed_dir/ratings_csr.npz"
    )


@task("data:generate_als_vectors")
def generate_als_vectors():
    """
    Assembles ALS vector
    """

    with mlflow.start_run(run_name="ALS full training"):
        paths = get_project_paths()
        ratings_csr = load_npz(paths["processed_dir"] / "ratings_csr.npz")

        # Model parameters
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

        als_model_full = AlternatingLeastSquares(
            factors=factors, regularization=0.1, iterations=15
        )
        als_model_full.fit(ratings_csr.T)

        training_time = time() - start_time
        mlflow.log_metric("training_time_seconds", training_time)

        # GPU → Numpy
        item_factors_np = als_model_full.item_factors.to_numpy()
        als_model_full.user_factors.to_numpy()

        np.save(paths["models_dir"] / "item_factors.npy", item_factors_np)
        joblib.dump(als_model_full, paths["models_dir"] / "als_model.pkl")

        mlflow.log_artifact(str(paths["models_dir"] / "item_factors.npy"))
        mlflow.log_artifact(str(paths["models_dir"] / "als_model.pkl"))


@task("data:combine_content_als_vector")
def combine_content_als_vector():
    """
    Assembles hybrid vector (content + ALS vector)
    """

    paths = get_project_paths()
    genome_scores = pd.read_csv(paths["raw_dir"] / "genome-scores.csv")

    movie_tag_matrix = genome_scores.pivot(
        index="movieId", columns="tagId", values="relevance"
    ).fillna(0)
    movie_ids_with_tags = movie_tag_matrix.index.to_numpy()

    item_factors = np.load(paths["models_dir"] / "item_factors.npy")
    item_encoder = joblib.load(paths["processed_dir"] / "item_encoder.pkl")
    # We get the indices of these movieids in item_matrix using item_encoder

    item_indices = item_encoder.transform(movie_ids_with_tags)

    # We select only the necessary lines from item_matrix
    filtered_item_matrix_full = item_factors[item_indices]
    model_movies_full_vectors_raw = np.load(
        paths["models_dir"] / "model_movies_full_vectors_raw.npz"
    )["vectors"]

    # We combine by signs (horizontally)
    hybrid_movie_vector_full = np.hstack(
        [model_movies_full_vectors_raw, filtered_item_matrix_full]
    )
    np.savez_compressed(
        paths["models_dir"] / "hybrid_movie_vector_full.npz",
        vectors=hybrid_movie_vector_full,
    )

    return hybrid_movie_vector_full


@task("data:segment_movies")
def segment_movies(mlflow_experiment: str = "MovieSegmentation"):
    """
    Collects a segmented (by genre and rating) content matrix
    """

    paths = get_project_paths()
    movies = pd.read_csv(paths["raw_dir"] / "movies.csv")
    ratings = pd.read_csv(paths["raw_dir"] / "ratings.csv")

    mlflow.set_experiment(mlflow_experiment)

    with mlflow.start_run(run_name="Movie Clustering"):
        # 1. Preparation of genres
        movies["genres"] = movies["genres"].apply(
            lambda x: x.split("|") if isinstance(x, str) else []
        )
        mlb = MultiLabelBinarizer()
        genre_matrix = pd.DataFrame(
            mlb.fit_transform(movies["genres"]),
            columns=mlb.classes_,
            index=movies.index,
        )

        # 2. Average rating and number of assessments
        agg = ratings.groupby("movieId")["rating"].agg(["mean", "count"]).reset_index()
        agg.columns = ["movieId", "mean_rating", "rating_count"]

        # 3. Association with Movies
        df = movies.merge(agg, on="movieId", how="left").fillna(
            {"mean_rating": 0, "rating_count": 0}
        )
        df = pd.concat(
            [df[["movieId", "mean_rating", "rating_count"]], genre_matrix], axis=1
        )

        # 4. Scaling
        scaler = StandardScaler()
        features = scaler.fit_transform(df.drop(columns=["movieId"]))

        # 5. Search for optimal k (elbow + silhouette)
        distortions = []
        silhouettes = []
        K = range(2, 21)

        for k in K:
            kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=1024)
            labels = kmeans.fit_predict(features)
            distortions.append(kmeans.inertia_)
            silhouettes.append(silhouette_score(features, labels))

        # 6. Logging of graphs
        fig, ax = plt.subplots()
        ax.plot(K, distortions, marker="o")
        ax.set_title("Метод локтя")
        ax.set_xlabel("Количество кластеров")
        ax.set_ylabel("Инерция")
        mlflow.log_figure(fig, "elbow_plot.png")
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.plot(K, silhouettes, marker="x", color="green")
        ax.set_title("Коэффициент силуэта")
        ax.set_xlabel("Количество кластеров")
        ax.set_ylabel("Silhouette Score")
        mlflow.log_figure(fig, "silhouette_plot.png")
        plt.close(fig)

        # 7. Final clustering
        best_k = K[np.argmax(silhouettes)]
        final_model = MiniBatchKMeans(n_clusters=best_k, random_state=42)
        df["cluster"] = final_model.fit_predict(features)

        # 8. Preservation of the model
        model_path = paths["models_dir"] / "final_model.pkl"
        joblib.dump(final_model, model_path)
        mlflow.log_artifact(str(model_path))
        scaler_path = paths["models_dir"] / "scaler.pkl"
        mlb_path = paths["models_dir"] / "mlb.pkl"

        joblib.dump(final_model, model_path)
        joblib.dump(scaler, scaler_path)
        joblib.dump(mlb, mlb_path)

        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(scaler_path))
        mlflow.log_artifact(str(mlb_path))

        # 9. Logging results
        mlflow.log_param("best_k", best_k)
        mlflow.log_metric("best_silhouette", max(silhouettes))

        result_path = paths["processed_dir"] / "movie_clusters.csv"
        df[["movieId", "cluster"]].to_csv(result_path, index=False)
        mlflow.log_artifact(str(result_path))

        return str(result_path)


@task("data:build_user_segment_matrix")
def build_user_segment_matrix() -> pd.DataFrame:
    """
    Assembles a segmented (by ratings and genres) hybrid matrix
    """

    paths = get_project_paths()
    movies = pd.read_csv(paths["raw_dir"] / "movies.csv")
    ratings = pd.read_csv(paths["raw_dir"] / "ratings.csv")
    # We filter positive grades
    positive_ratings = ratings[ratings["rating"] > 3.5]

    # We connect genres
    movies = movies[["movieId", "genres"]].copy()
    movies["genres"] = movies["genres"].apply(lambda g: g.split("|"))

    # We expand to genres
    exploded_movies = movies.explode("genres")

    # We connect to estimates
    merged = positive_ratings.merge(exploded_movies, on="movieId")

    # For each user: % genres
    genre_counts = merged.groupby(["userId", "genres"]).size().unstack(fill_value=0)
    genre_distrib = genre_counts.div(genre_counts.sum(axis=1), axis=0)  # shares

    # For each user: Average score on films clusters
    # We load the clustering model
    joblib.load(paths["models_dir"] / "final_model.pkl")

    # It is assumed that the preserved movie_features (those by which the cluster studied)
    # Loading and predictions clusters
    movie_features = pd.read_csv(paths["processed_dir"] / "movie_clusters.csv")

    # Add clusters to films
    movie_cluster_map = movie_features[["movieId", "cluster"]]
    merged_clusters = positive_ratings.merge(movie_cluster_map, on="movieId")

    cluster_stats = (
        merged_clusters.groupby(["userId", "cluster"])["rating"]
        .mean()
        .unstack(fill_value=0)
        .add_prefix("cluster_rating_")
    )

    # The density of assessments (general number of positive films)
    rating_density = positive_ratings.groupby("userId").size().rename("positive_count")

    # We combine everything
    final_df = genre_distrib.join(cluster_stats, how="outer").join(
        rating_density, how="outer"
    )
    final_df = final_df.fillna(0)

    save_npz(
        paths["processed_dir"] / "user_segment_matrix.npz", csr_matrix(final_df.values)
    )

    final_df.to_csv(paths["processed_dir"] / "user_segment_matrix.csv")

    return final_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", help="Список задач для выполнения")
    args = parser.parse_args()

    if args.tasks:
        main(args.tasks)  # Here we transmit the tasks indicated on the command line
