import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.sparse import csr_matrix, load_npz, save_npz
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
mlflow.set_tracking_uri("http://localhost:5000")

from scripts.task_registry import main, task
from scripts.utils import load_vectors_npz, log_training_metrics, save_model_metrics


def get_project_paths():
    """
    Function to get paths for different data
    """
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    with open(PROJECT_ROOT / "params.yaml") as f:
        config = yaml.safe_load(f)

    paths = config["paths"]
    return {
        "project_root": PROJECT_ROOT,
        "raw_dir": PROJECT_ROOT / paths["raw_data_dir"],
        "processed_dir": PROJECT_ROOT / paths["processed_data_dir"],
        "models_dir": PROJECT_ROOT / paths["models_dir"],
        "scripts_dir": PROJECT_ROOT / paths["scripts"],
    }


def plot_losses(train_losses, val_losses) -> str:
    """
    The function prepares graphs for saving/transferring to mlflow
    :param train_losses:
    :param val_losses:
    :return: Path to the image
    """

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_path = "models/training_loss_curve.png"
    plt.savefig(plot_path)
    plt.close()

    return plot_path  # Return the path to the saved file


class MovieAutoencoder(nn.Module):
    """
    Autoencoder model for content vector
    """

    def __init__(self, input_dim, encoding_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, encoding_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class SparseRowDataset(Dataset):
    """
    Autoencoder dataset for content vector
    """

    def __init__(self, sparse_matrix):
        self.matrix = sparse_matrix

    def __len__(self):
        return self.matrix.shape[0]

    def __getitem__(self, idx):
        row = self.matrix.getrow(idx).toarray().squeeze()
        return torch.tensor(row, dtype=torch.float32)


# Model
class Autoencoder(nn.Module):
    """
    Autoencoder model for user vector
    """

    def __init__(self, input_dim, encoding_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Linear(512, encoding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out


@task("data:movie_autoencoder")
def content_vector_autoencoder():
    """
    Autoencoder training function
    returns the saved model
    """
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)["autoencoder"]
        paths = get_project_paths()
    encoding_dim = config["encoding_dim"]
    batch_size = config["batch_size"]
    num_epochs = config["num_epochs"]
    lr = config["learning_rate"]
    model_path = config["model_output_path"]

    movie_vectors_scaled_train = np.load(
        paths["processed_dir"] / "movie_vectors_scaled_train.npy"
    )

    input_dim = movie_vectors_scaled_train.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MovieAutoencoder(input_dim, encoding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X = torch.tensor(movie_vectors_scaled_train, dtype=torch.float32)
    train_size = int(0.9 * len(X))
    val_size = len(X) - train_size
    train_dataset, val_dataset = random_split(X, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    best_val_loss = float("inf")  # Initialization of the best validation loss
    best_model_state = None

    epoch_train_losses = []
    epoch_val_losses = []

    with mlflow.start_run():
        mlflow.log_params(config)

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            for batch in train_loader:
                batch = batch.to(device)
                output = model(batch)
                loss = criterion(output, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    output = model(batch)
                    loss = criterion(output, batch)
                    val_loss += loss.item()

            avg_train = train_loss / len(train_loader)
            avg_val = val_loss / len(val_loader)

            epoch_train_losses.append(avg_train)
            epoch_val_losses.append(avg_val)

            # Logging metrics for each era
            mlflow.log_metric("train_loss", avg_train, step=epoch)
            mlflow.log_metric("val_loss", avg_val, step=epoch)

            # We save the model if it is the best in validation loss
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_model_state = model.state_dict()

            print(
                f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}"
            )

        # We keep the best model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(best_model_state, model_path)
        mlflow.pytorch.log_model(model, "model")

        # Logging the loss schedule
        plot_path = plot_losses(epoch_train_losses, epoch_val_losses)
        mlflow.log_artifact(str(plot_path))  # We convert into a line for mlflow

        train_metrics = {
            "best_val_loss": best_val_loss,
            "final_train_loss": epoch_train_losses[-1],
            "final_val_loss": epoch_val_losses[-1],
            "num_epochs": num_epochs,
        }

        # Save metrics to the DVC file
        metrics_path = "models/train_metrics.json"
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(train_metrics, f, indent=4)

    return best_model_state


@task("data:movie_autoencoder_raw")
def content_vector_autoencoder_raw():
    """
    Autoencoder training function for full vector
    Returns the saved model
    """

    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)["autoencoder"]
        paths = get_project_paths()
    encoding_dim = config["encoding_dim"]
    batch_size = config["batch_size"]
    num_epochs = config["num_epochs"]
    lr = config["learning_rate"]
    model_path = config["model_output_path_raw"]

    movie_vectors_scaled_full = np.load(
        paths["processed_dir"] / "movie_vectors_scaled_full.npy"
    )

    input_dim = movie_vectors_scaled_full.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MovieAutoencoder(input_dim, encoding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X = torch.tensor(movie_vectors_scaled_full, dtype=torch.float32)
    train_size = int(0.9 * len(X))
    val_size = len(X) - train_size
    train_dataset, val_dataset = random_split(X, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    best_val_loss = float("inf")  # Initialization of the best validation loss
    best_model_state = None

    epoch_train_losses = []
    epoch_val_losses = []

    with mlflow.start_run():
        mlflow.log_params(config)

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            for batch in train_loader:
                batch = batch.to(device)
                output = model(batch)
                loss = criterion(output, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    output = model(batch)
                    loss = criterion(output, batch)
                    val_loss += loss.item()

            avg_train = train_loss / len(train_loader)
            avg_val = val_loss / len(val_loader)

            epoch_train_losses.append(avg_train)
            epoch_val_losses.append(avg_val)

            # Logging metrics for each era
            mlflow.log_metric("train_loss", avg_train, step=epoch)
            mlflow.log_metric("val_loss", avg_val, step=epoch)

            # We save the model if it is the best in validation loss
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_model_state = model.state_dict()

            print(
                f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}"
            )

        # We keep the best model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(best_model_state, model_path)
        mlflow.pytorch.log_model(model, "model_raw")

        # Logging the loss schedule
        plot_path = plot_losses(epoch_train_losses, epoch_val_losses)
        mlflow.log_artifact(str(plot_path))  # We convert into a line for mlflow

        train_metrics = {
            "best_val_loss_raw": best_val_loss,
            "final_train_loss_raw": epoch_train_losses[-1],
            "final_val_loss_raw": epoch_val_losses[-1],
            "num_epochs_raw": num_epochs,
        }

        # Save metrics to the DVC file
        metrics_path = "models/metrics_raw.json"
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(train_metrics, f, indent=4)

    return best_model_state


@task("data:user_autoencoder")
def user_vector_autoencoder():
    """
    Autoencoder training function for a user-defined vector
    Returns the saved model
    """

    # Loading parameters
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)["user_autoencoder"]
    paths = get_project_paths()

    encoding_dim = config["encoding_dim"]
    batch_size = config["batch_size"]
    num_epochs = config["num_epochs"]
    lr = config["learning_rate"]

    # Download CSR matrix
    ratings_csr = load_npz(paths["processed_dir"] / "ratings_csr.npz")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = ratings_csr.shape[1]

    # DATALOADER
    dataset = SparseRowDataset(ratings_csr)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Model
    model = Autoencoder(input_dim=input_dim, encoding_dim=encoding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    nn.MSELoss()

    epoch_train_losses = []

    with mlflow.start_run():
        mlflow.log_params(config)

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            start_epoch = time.time()

            for i, batch in enumerate(
                tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
            ):
                batch = batch.to(device)

                output = model(batch)
                # Modified MSE for accounting only non -equal values
                mask = batch != 0
                loss = ((output - batch) ** 2 * mask).sum() / mask.sum()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                if i % 50 == 0:
                    print(f"[{i}/{len(dataloader)}] loss={loss.item():.4f}")

            avg_loss = total_loss / len(dataloader)
            epoch_train_losses.append(avg_loss)
            print(
                f"Epoch {epoch + 1} завершён за {time.time() - start_epoch:.1f} сек. Средний loss: {avg_loss:.4f}"
            )

        # We save the model correctly
        torch.save(
            model.state_dict(), paths["models_dir"] / "user_autoencoder_model.pt"
        )
        mlflow.pytorch.log_model(model, "model")

        # Logging of the graph
        plot_path = plot_losses(epoch_train_losses, [])
        mlflow.log_artifact(str(plot_path))

        # Metrics for DVC
        train_metrics = {
            "final_train_loss": epoch_train_losses[-1],
            "num_epochs": num_epochs,
        }
        metrics_path = paths["models_dir"] / "user_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(train_metrics, f, indent=4)


class RatingPredictor(nn.Module):
    """
    Neural network model for recommendation prediction
    """

    def __init__(self, vector_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vector_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, user_vec, item_vec):
        x = torch.cat([user_vec, item_vec], dim=1)
        return self.net(x)


class UserItemRatingDataset(Dataset):
    def __init__(self, ratings_df, user_vectors, item_vectors):
        self.ratings = ratings_df.reset_index(drop=True)
        self.user_vectors = torch.tensor(user_vectors, dtype=torch.float32)
        self.item_vectors = torch.tensor(item_vectors, dtype=torch.float32)
        self.ratings_tensor = torch.tensor(
            self.ratings["rating"].values, dtype=torch.float32
        )

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        row = self.ratings.iloc[idx]
        user_idx = int(row["user_idx"])
        item_idx = int(row["item_idx"])

        user_vec = self.user_vectors[user_idx]
        item_vec = self.item_vectors[item_idx]
        rating = self.ratings_tensor[idx]

        return user_vec, item_vec, rating


def root_mean_squared_error(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


@task("data:train_recommender")
def train_recommender_nn():
    """
    Neural network training function
    Returns the saved model
    """
    paths = get_project_paths()
    config_path = Path("params.yaml")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)["recommender_nn"]

    batch_size = config["batch_size"]
    num_epochs = config["num_epochs"]
    lr = config["learning_rate"]
    patience = config["patience"]
    hidden_dim = config.get("hidden_dim", 128)

    best_model_path = paths["models_dir"] / "neural_model_best.pt"
    metrics_path = paths["models_dir"] / "neural_model_metrics.json"

    # Data load
    ratings = pd.read_csv(paths["raw_dir"] / "ratings.csv")
    user_vectors = load_vectors_npz(paths["models_dir"] / "user_content_vector.npz")
    item_vectors = load_vectors_npz(
        paths["models_dir"] / "model_movies_full_vectors_raw.npz"
    )

    user_encoder = joblib.load(paths["processed_dir"] / "user_encoder.pkl")
    item_encoder = joblib.load(paths["processed_dir"] / "item_encoder.pkl")

    valid_movie_idxs = np.arange(item_vectors.shape[0])
    valid_movie_ids = item_encoder.inverse_transform(valid_movie_idxs)

    filtered_ratings = ratings[ratings["movieId"].isin(valid_movie_ids)].copy()
    filtered_ratings["user_idx"] = user_encoder.transform(
        filtered_ratings["userId"]
    ).astype(int)
    filtered_ratings["item_idx"] = item_encoder.transform(
        filtered_ratings["movieId"]
    ).astype(int)

    train_df, val_df = train_test_split(
        filtered_ratings, test_size=0.1, random_state=42
    )
    max_user_idx = len(user_vectors) - 1
    max_item_idx = len(item_vectors) - 1
    val_df = val_df[
        (val_df["user_idx"] <= max_user_idx) & (val_df["item_idx"] <= max_item_idx)
    ]

    if val_df.empty:
        raise ValueError("Validation set is empty after filtering.")

    train_dataset = UserItemRatingDataset(train_df, user_vectors, item_vectors)
    val_dataset = UserItemRatingDataset(val_df, user_vectors, item_vectors)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RatingPredictor(vector_dim=user_vectors.shape[1], hidden_dim=hidden_dim).to(
        device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    best_rmse = float("inf")
    epochs_without_improvement = 0
    train_losses, val_rmses = [], []

    with mlflow.start_run():
        mlflow.log_params(
            {
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "learning_rate": lr,
                "patience": patience,
                "hidden_dim": hidden_dim,
            }
        )

        for epoch in range(num_epochs):
            tqdm.write(f"\n🔁 [Epoch {epoch + 1}/{num_epochs}] — обучение начато")
            model.train()
            total_loss = 0
            progress_bar = tqdm(
                train_loader, desc=f"🔄 Тренировка (эпоха {epoch + 1})", leave=False
            )

            for user_vec, item_vec, rating in progress_bar:
                user_vec, item_vec, rating = (
                    user_vec.to(device),
                    item_vec.to(device),
                    rating.to(device).unsqueeze(1),
                )
                optimizer.zero_grad()
                output = model(user_vec, item_vec)
                loss = criterion(output, rating)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            train_losses.append(avg_loss)

            # Validation
            model.eval()
            val_preds, val_targets = [], []
            with torch.no_grad():
                for user_vec, item_vec, rating in val_loader:
                    user_vec, item_vec, rating = (
                        user_vec.to(device),
                        item_vec.to(device),
                        rating.to(device).unsqueeze(1),
                    )
                    output = model(user_vec, item_vec)
                    val_preds.append(output.cpu().numpy())
                    val_targets.append(rating.cpu().numpy())

            val_preds = np.concatenate(val_preds).flatten()
            val_targets = np.concatenate(val_targets).flatten()
            rmse = root_mean_squared_error(val_targets, val_preds)
            val_rmses.append(rmse)

            tqdm.write(
                f"📊 Epoch {epoch + 1}: train_loss = {avg_loss:.4f}, val_rmse = {rmse:.4f}"
            )
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            mlflow.log_metric("val_rmse", rmse, step=epoch)

            if rmse < best_rmse:
                best_rmse = rmse
                epochs_without_improvement = 0
                torch.save(model.state_dict(), best_model_path)
                tqdm.write(f"✅ Модель сохранена (лучший RMSE: {best_rmse:.4f})")
            else:
                epochs_without_improvement += 1
                tqdm.write(
                    f"⚠️ Без улучшения RMSE ({epochs_without_improvement}/{patience})"
                )
                if epochs_without_improvement >= patience:
                    tqdm.write("⏹️ Ранняя остановка обучения.")
                    break

        mlflow.pytorch.log_model(model, artifact_path="model")
        log_training_metrics(train_losses, val_rmses)
        save_model_metrics(metrics_path, train_losses, val_rmses, best_rmse)


@task("data:movie_eval_vector")
def eval_content_train_test_vectors():
    """
    Content vector encoding function
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)["autoencoder"]
        paths = get_project_paths()

    model_path = config["model_output_path"]

    movie_vectors_scaled_train = np.load(
        paths["processed_dir"] / "movie_vectors_scaled_train.npy"
    )
    movie_vectors_scaled_test = np.load(
        paths["processed_dir"] / "movie_vectors_scaled_test.npy"
    )

    input_dim = movie_vectors_scaled_train.shape[1]

    # We load the trained model
    model = MovieAutoencoder(input_dim=input_dim, encoding_dim=64).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        # Restored vectors (Full Pipeline: Encoder -> Decoder)
        train_recon = (
            model(
                torch.tensor(movie_vectors_scaled_train, dtype=torch.float32).to(device)
            )
            .cpu()
            .numpy()
        )
        test_recon = (
            model(
                torch.tensor(movie_vectors_scaled_test, dtype=torch.float32).to(device)
            )
            .cpu()
            .numpy()
        )

        # Vector for Content-BASED filtration (Encoder output)
        movie_content_vectors_train = (
            model.encoder(
                torch.tensor(movie_vectors_scaled_train, dtype=torch.float32).to(device)
            )
            .cpu()
            .numpy()
        )
        movie_content_vectors_test = (
            model.encoder(
                torch.tensor(movie_vectors_scaled_test, dtype=torch.float32).to(device)
            )
            .cpu()
            .numpy()
        )

    # We count the metrics of recovery
    train_mse = mean_squared_error(movie_vectors_scaled_train, train_recon)
    test_mse = mean_squared_error(movie_vectors_scaled_test, test_recon)

    eval_metrics = {"train_mse": train_mse, "test_mse": test_mse}

    # We save metrics for DVC
    os.makedirs("models", exist_ok=True)
    with open("models/eval_metrics.json", "w") as f:
        json.dump(eval_metrics, f, indent=4)

    # We save the Content vector for filtering
    np.savez_compressed(
        "models/movie_content_vectors_train.npz", vectors=movie_content_vectors_train
    )
    np.savez_compressed(
        "models/movie_content_vectors_test.npz", vectors=movie_content_vectors_test
    )

    print("Модель оценена!")
    print(f"Train MSE: {train_mse:.6f}")
    print(f"Test MSE: {test_mse:.6f}")
    print(
        "Вектора сохранены в 'models/movie_content_vectors_train.npz' и 'models/movie_content_vectors_test.npz'"
    )


@task("data:eval_user_vectors")
def eval_user_vectors():
    """
    User defined vector encoding function
    """

    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)["user_autoencoder"]
        paths = get_project_paths()

    encoding_dim = config["encoding_dim"]
    batch_size = config["batch_size"]
    ratings_csr = load_npz(paths["processed_dir"] / "ratings_csr.npz")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = ratings_csr.shape[1]

    dataset = SparseRowDataset(ratings_csr)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    model = Autoencoder(input_dim=input_dim, encoding_dim=encoding_dim).to(device)
    model_path = paths["models_dir"] / "user_autoencoder_model.pt"
    model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()

    all_embeddings = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Encoding rows"):
            batch = batch.to(device)
            encoded = model.encoder(batch)
            all_embeddings.append(encoded.cpu())

    user_content_vector = torch.cat(all_embeddings, dim=0).numpy()
    np.savez_compressed(
        paths["models_dir"] / "user_content_vector.npz", vectors=user_content_vector
    )


@task("data:movie_eval_vector_raw")
def eval_content_train_test_vectors_raw():
    """
    Content vector encoding function (for line-by-line splitting)
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)["autoencoder"]
        paths = get_project_paths()

    model_path = config["model_output_path_raw"]

    movie_vectors_scaled_full = np.load(
        paths["processed_dir"] / "movie_vectors_scaled_full.npy"
    )

    input_dim = movie_vectors_scaled_full.shape[1]

    # We load the trained model
    model = MovieAutoencoder(input_dim=input_dim, encoding_dim=64).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        # Restored vectors (through the entire auto -fister: Encoder -> Decoder)
        recon = (
            model(
                torch.tensor(movie_vectors_scaled_full, dtype=torch.float32).to(device)
            )
            .cpu()
            .numpy()
        )

        # We drive out only through the encoder
        movie_vectors_tensor = torch.tensor(
            movie_vectors_scaled_full, dtype=torch.float32
        ).to(device)
        model_movies_full_vectors = model.encoder(movie_vectors_tensor).cpu().numpy()

    # We count the metrics of recovery
    mse = mean_squared_error(movie_vectors_scaled_full, recon)

    eval_metrics = {
        "mse": mse,
    }

    # We save metrics for DVC
    os.makedirs("models", exist_ok=True)
    with open("models/eval_metrics_raw.json", "w") as f:
        json.dump(eval_metrics, f, indent=4)

    # We save the Content vector for filtering
    np.savez_compressed(
        "models/model_movies_full_vectors_raw.npz", vectors=model_movies_full_vectors
    )

    print("Модель оценена!")
    print(f"MSE: {mse:.6f}")


@task("data:make_bridge_als")
def make_bridge_als() -> list[Optional[lgb.LGBMRegressor]]:
    """
    Trains bridge models to transform movie content vectors into ALS space.

    """
    with mlflow.start_run(run_name="Bridge_LGBM"):
        start_time = time.time()

        paths = get_project_paths()
        genome_scores = pd.read_csv(paths["raw_dir"] / "genome-scores.csv")

        movie_tag_matrix = genome_scores.pivot(
            index="movieId", columns="tagId", values="relevance"
        ).fillna(0)
        movie_ids_with_tags = movie_tag_matrix.index.to_numpy()

        item_factors = np.load(paths["models_dir"] / "item_factors.npy")
        item_encoder = joblib.load(paths["processed_dir"] / "item_encoder.pkl")
        item_indices = item_encoder.transform(movie_ids_with_tags)
        filtered_item_matrix_full = item_factors[item_indices]

        model_movies_full_vectors_raw = np.load(
            paths["models_dir"] / "model_movies_full_vectors_raw.npz"
        )["vectors"]

        models_bridge = []
        scores = []

        X_content = model_movies_full_vectors_raw
        Y_factors = filtered_item_matrix_full

        mlflow.set_tag("model_type", "LGBMRegressor")
        mlflow.set_experiment("BridgeModels")

        # Model parameters
        params = dict(
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=31,
            min_data_in_leaf=5,
            min_gain_to_split=0.0,
            random_state=42,
        )
        for key, val in params.items():
            mlflow.log_param(key, val)

        mlflow.log_param("X_content_shape", X_content.shape)
        mlflow.log_param("Y_factors_shape", Y_factors.shape)

        weak_models = []

        for i in tqdm(range(Y_factors.shape[1]), desc="Обучение bridge-моделей"):
            y = Y_factors[:, i]
            std_y = float(np.std(y))
            if std_y < 1e-4:
                models_bridge.append(None)
                scores.append({"r2": 0.0, "mae": None, "std_y": std_y})
                continue

            model = lgb.LGBMRegressor(**params)
            model.fit(X_content, y)
            y_pred = model.predict(X_content)

            r2 = float(r2_score(y, y_pred))
            mae = float(mean_absolute_error(y, y_pred))

            models_bridge.append(model)
            scores.append({"r2": r2, "mae": mae, "std_y": std_y})

            mlflow.log_metric(f"r2_{i}", r2)
            mlflow.log_metric(f"mae_{i}", mae)
            mlflow.log_metric(f"std_y_{i}", std_y)

            if r2 < 0.1:
                weak_models.append({"column": i, "r2": r2, "mae": mae, "std_y": std_y})

        # We save the models
        joblib.dump(models_bridge, paths["models_dir"] / "models_bridge.pkl")

        # We keep the metrics
        scores_path = paths["models_dir"] / "bridge_scores.json"
        with open(scores_path, "w") as f:
            json.dump(scores, f, indent=2)
        mlflow.log_artifact(str(scores_path))

        # We save weak models separately
        weak_models_path = paths["models_dir"] / "weak_models.json"
        with open(weak_models_path, "w") as f:
            json.dump(weak_models, f, indent=2)
        mlflow.log_artifact(str(weak_models_path))

        # Averaged metrics
        valid_scores = [s for s in scores if s["mae"] is not None]
        if valid_scores:
            mean_r2 = float(np.mean([s["r2"] for s in valid_scores]))
            mean_mae = float(np.mean([s["mae"] for s in valid_scores]))
            mean_std_y = float(np.mean([s["std_y"] for s in valid_scores]))

            mlflow.log_metric("mean_r2", mean_r2)
            mlflow.log_metric("mean_mae", mean_mae)
            mlflow.log_metric("mean_std_y", mean_std_y)

        mlflow.log_metric("training_time_sec", float(time.time() - start_time))

        return models_bridge


class EmbeddingRatingPredictor(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, hidden_dim=128):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        self.net = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, user_ids, item_ids):
        user_vecs = self.user_embedding(user_ids)
        item_vecs = self.item_embedding(item_ids)
        x = torch.cat([user_vecs, item_vecs], dim=1)
        return self.net(x)


class UserItemIDRatingDataset(Dataset):
    def __init__(self, ratings_df):
        self.user_ids = torch.tensor(ratings_df["user_idx"].values, dtype=torch.long)
        self.item_ids = torch.tensor(ratings_df["item_idx"].values, dtype=torch.long)
        self.ratings = torch.tensor(ratings_df["rating"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.item_ids[idx], self.ratings[idx]


@task("data:train_embedding_nn")
def train_embedding_nn():
    """
    Content vector encoding function on a trained neural network
    """

    paths = get_project_paths()
    config_path = Path("params.yaml")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)["embedding_recommender"]

    batch_size = config["batch_size"]
    num_epochs = config["num_epochs"]
    lr = config["learning_rate"]
    patience = config["patience"]

    hidden_dim = config["hidden_dim"]

    ratings = pd.read_csv(paths["raw_dir"] / "ratings.csv")
    user_encoder = joblib.load(paths["processed_dir"] / "user_encoder.pkl")
    item_encoder = joblib.load(paths["processed_dir"] / "item_encoder.pkl")

    user_vectors = load_vectors_npz(paths["models_dir"] / "user_content_vector.npz")
    embedding_dim = user_vectors.shape[1]

    ratings = ratings[ratings["movieId"].isin(item_encoder.classes_)].copy()
    ratings["user_idx"] = user_encoder.transform(ratings["userId"])
    ratings["item_idx"] = item_encoder.transform(ratings["movieId"])

    num_users = len(user_encoder.classes_)
    num_items = len(item_encoder.classes_)

    train_df, val_df = train_test_split(ratings, test_size=0.1, random_state=42)

    train_dataset = UserItemIDRatingDataset(train_df)
    val_dataset = UserItemIDRatingDataset(val_df)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmbeddingRatingPredictor(
        num_users, num_items, embedding_dim, hidden_dim
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_rmse = float("inf")
    epochs_without_improvement = 0
    model_path = paths["models_dir"] / "embedding_model_best.pt"

    with mlflow.start_run():
        mlflow.log_params(
            {
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "learning_rate": lr,
                "patience": patience,
                "embedding_dim": embedding_dim,
                "hidden_dim": hidden_dim,
            }
        )

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

            for user_ids, item_ids, ratings in progress_bar:
                user_ids, item_ids, ratings = (
                    user_ids.to(device),
                    item_ids.to(device),
                    ratings.to(device).unsqueeze(1),
                )
                optimizer.zero_grad()
                preds = model(user_ids, item_ids)
                loss = criterion(preds, ratings)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            # Validation
            model.eval()
            val_preds, val_targets = [], []
            with torch.no_grad():
                for user_ids, item_ids, ratings in val_loader:
                    user_ids, item_ids, ratings = (
                        user_ids.to(device),
                        item_ids.to(device),
                        ratings.to(device).unsqueeze(1),
                    )
                    preds = model(user_ids, item_ids)
                    val_preds.extend(preds.squeeze().cpu().numpy())
                    val_targets.extend(ratings.squeeze().cpu().numpy())

            rmse = root_mean_squared_error(val_targets, val_preds)
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            mlflow.log_metric("val_rmse", rmse, step=epoch)

            tqdm.write(
                f"📊 Epoch {epoch + 1}: train_loss = {avg_loss:.4f}, val_rmse = {rmse:.4f}"
            )

            if rmse < best_rmse:
                best_rmse = rmse
                epochs_without_improvement = 0
                torch.save(model.state_dict(), model_path)
                tqdm.write("✅ Model saved")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    tqdm.write("⏹️ Early stopping")
                    break

        mlflow.pytorch.log_model(model, artifact_path="embedding_model")

    # We save separately embedding
    model.load_state_dict(torch.load(model_path))
    model.eval()
    user_embs = model.user_embedding.weight.data.cpu().numpy()
    item_embs = model.item_embedding.weight.data.cpu().numpy()

    np.savez_compressed(
        paths["models_dir"] / "embedding_user_vectors.npz", vectors=user_embs
    )
    np.savez_compressed(
        paths["models_dir"] / "embedding_item_vectors.npz", vectors=item_embs
    )
    tqdm.write("💾 Эмбеддинги сохранены")


@task("data:train_user_segment_autoencoder")
def train_user_segment_autoencoder():
    """
    Autoencoder training function for content segmented vector
    """

    # Configuration loading
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)["user_segment_autoencoder"]
    paths = get_project_paths()

    encoding_dim = config["encoding_dim"]
    batch_size = config["batch_size"]
    num_epochs = config["num_epochs"]
    lr = config["learning_rate"]

    # Loading a segmented matrix
    segment_matrix = load_npz(paths["processed_dir"] / "user_segment_matrix.npz")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = segment_matrix.shape[1]

    # DATALOADER
    dataset = SparseRowDataset(segment_matrix)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Model
    model = Autoencoder(input_dim=input_dim, encoding_dim=encoding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    nn.MSELoss()

    epoch_train_losses = []

    with mlflow.start_run():
        mlflow.log_params(config)

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            start_epoch = time.time()

            progress_bar = tqdm(
                dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False
            )
            for i, batch in enumerate(progress_bar):
                batch = batch.to(device)

                output = model(batch)
                mask = batch != 0
                loss = ((output - batch) ** 2 * mask).sum() / mask.sum()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                if i % 50 == 0:
                    tqdm.write(
                        f"[Epoch {epoch + 1} | Batch {i}/{len(dataloader)}] Loss: {loss.item():.4f}"
                    )

            avg_loss = total_loss / len(dataloader)
            epoch_train_losses.append(avg_loss)
            tqdm.write(
                f"Epoch {epoch + 1} завершён за {time.time() - start_epoch:.1f} сек. Средний loss: {avg_loss:.4f}"
            )

        # We save the model
        model_path = paths["models_dir"] / "user_segment_autoencoder.pt"
        torch.save(model.state_dict(), model_path)
        mlflow.pytorch.log_model(model, "model")

        # Log of loss graphics
        plot_path = plot_losses(epoch_train_losses, [])
        mlflow.log_artifact(str(plot_path))

        # Metrics
        metrics_path = paths["models_dir"] / "user_segment_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump({"final_train_loss": epoch_train_losses[-1]}, f, indent=4)
        mlflow.log_artifact(str(metrics_path))


@task("data:encode_user_segment")
def encode_user_segment():
    """
    Function of encoding content segmented vector on auto encoder
    """

    # Loading parameters
    with open("params.yaml") as f:
        config = yaml.safe_load(f)["user_autoencoder"]
    paths = get_project_paths()

    encoding_dim = config["encoding_dim"]
    batch_size = config["batch_size"]

    # Data and models loading
    user_matrix = load_npz(paths["processed_dir"] / "user_segment_matrix.npz")
    input_dim = user_matrix.shape[1]
    dataset = SparseRowDataset(user_matrix)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Autoencoder(input_dim=input_dim, encoding_dim=encoding_dim).to(device)
    model.load_state_dict(
        torch.load(
            paths["models_dir"] / "user_segment_autoencoder.pt", map_location=device
        )
    )
    model.eval()

    # Run through the encoder
    encoded_rows = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Encoding users"):
            batch = batch.to(device)
            encoded = model.encoder(
                batch
            )  # or model.encoder (Batch) if you use another architecture
            encoded_rows.append(encoded.cpu())

    # Association and conservation
    encoded_matrix = torch.cat(encoded_rows).numpy()
    save_npz(
        paths["processed_dir"] / "encoded_user_vectors.npz", csr_matrix(encoded_matrix)
    )

    print("Готово: encoded_user_vectors.npz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", help="Список задач для выполнения")
    args = parser.parse_args()

    if args.tasks:
        main(args.tasks)  # Here we transmit the tasks indicated on the command line
