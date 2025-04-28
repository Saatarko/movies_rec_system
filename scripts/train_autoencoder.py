import argparse
import json
import torch
import mlflow
import mlflow.pytorch
import yaml
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
import sys, os
import numpy as np

from utils import get_project_paths

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
mlflow.set_tracking_uri("http://localhost:5000")
from task_registry import task, main


def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


    plot_path =  "models/training_loss_curve.png"
    plt.savefig(plot_path)
    plt.close()

    return plot_path  # Возвращаем путь к сохранённому файлу


class MovieAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

@task("data:movie_autoencoder")
def content_vector_autoencoder():
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)["autoencoder"]
        paths = get_project_paths()
    encoding_dim = config["encoding_dim"]
    batch_size = config["batch_size"]
    num_epochs = config["num_epochs"]
    lr = config["learning_rate"]
    model_path = config["model_output_path"]

    movie_vectors_scaled_train = np.load(paths["processed_dir"] / "movie_vectors_scaled_train.npy")


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

    best_val_loss = float("inf")  # Инициализация лучшей валидационной потери
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

            # Логируем метрики для каждой эпохи
            mlflow.log_metric("train_loss", avg_train, step=epoch)
            mlflow.log_metric("val_loss", avg_val, step=epoch)

            # Сохраняем модель, если она лучшая по валидационной потере
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_model_state = model.state_dict()

            print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

        # Сохраняем лучшую модель
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(best_model_state, model_path)
        mlflow.pytorch.log_model(model, "model")

        # Логируем график потерь
        plot_path = plot_losses(epoch_train_losses, epoch_val_losses)
        mlflow.log_artifact(str(plot_path))  # Преобразуем в строку для mlflow

        train_metrics = {
            "best_val_loss": best_val_loss,
            "final_train_loss": epoch_train_losses[-1],
            "final_val_loss": epoch_val_losses[-1],
            "num_epochs": num_epochs
        }

        # Сохраняем метрики в файл для DVC
        metrics_path = "models/train_metrics.json"
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(train_metrics, f, indent=4)

    return best_model_state


@task("data:movie_autoencoder_raw")
def content_vector_autoencoder_raw():
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)["autoencoder"]
        paths = get_project_paths()
    encoding_dim = config["encoding_dim"]
    batch_size = config["batch_size"]
    num_epochs = config["num_epochs"]
    lr = config["learning_rate"]
    model_path = config["model_output_path_raw"]

    movie_vectors_scaled_full = np.load(paths["processed_dir"] / "movie_vectors_scaled_full.npy")


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

    best_val_loss = float("inf")  # Инициализация лучшей валидационной потери
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

            # Логируем метрики для каждой эпохи
            mlflow.log_metric("train_loss", avg_train, step=epoch)
            mlflow.log_metric("val_loss", avg_val, step=epoch)

            # Сохраняем модель, если она лучшая по валидационной потере
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_model_state = model.state_dict()

            print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

        # Сохраняем лучшую модель
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(best_model_state, model_path)
        mlflow.pytorch.log_model(model, "model_raw")

        # Логируем график потерь
        plot_path = plot_losses(epoch_train_losses, epoch_val_losses)
        mlflow.log_artifact(str(plot_path))  # Преобразуем в строку для mlflow

        train_metrics = {
            "best_val_loss_raw": best_val_loss,
            "final_train_loss_raw": epoch_train_losses[-1],
            "final_val_loss_raw": epoch_val_losses[-1],
            "num_epochs_raw": num_epochs
        }

        # Сохраняем метрики в файл для DVC
        metrics_path = "models/metrics_raw.json"
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(train_metrics, f, indent=4)

    return best_model_state



@task("data:movie_eval_vector")
def eval_content_train_test_vectors():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)["autoencoder"]
        paths = get_project_paths()

    model_path = config["model_output_path"]

    movie_vectors_scaled_train = np.load(paths["processed_dir"] / "movie_vectors_scaled_train.npy")
    movie_vectors_scaled_test = np.load(paths["processed_dir"] / "movie_vectors_scaled_test.npy")

    input_dim = movie_vectors_scaled_train.shape[1]

    # Загружаем обученную модель
    model = MovieAutoencoder(input_dim=input_dim, encoding_dim=64).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        # Восстановленные вектора (full pipeline: encoder -> decoder)
        train_recon = model(torch.tensor(movie_vectors_scaled_train, dtype=torch.float32).to(device)).cpu().numpy()
        test_recon = model(torch.tensor(movie_vectors_scaled_test, dtype=torch.float32).to(device)).cpu().numpy()

        # Вектора для content-based фильтрации (encoder output)
        movie_content_vectors_train = model.encoder(
            torch.tensor(movie_vectors_scaled_train, dtype=torch.float32).to(device)).cpu().numpy()
        movie_content_vectors_test = model.encoder(
            torch.tensor(movie_vectors_scaled_test, dtype=torch.float32).to(device)).cpu().numpy()

    # Считаем метрики восстановления
    train_mse = mean_squared_error(movie_vectors_scaled_train, train_recon)
    test_mse = mean_squared_error(movie_vectors_scaled_test, test_recon)

    eval_metrics = {
        "train_mse": train_mse,
        "test_mse": test_mse
    }

    # Сохраняем метрики для DVC
    os.makedirs("models", exist_ok=True)
    with open("models/eval_metrics.json", "w") as f:
        json.dump(eval_metrics, f, indent=4)

    # Сохраняем content вектора для фильтрации
    np.savez_compressed("models/movie_content_vectors_train.npz", vectors=movie_content_vectors_train)
    np.savez_compressed("models/movie_content_vectors_test.npz", vectors=movie_content_vectors_test)

    print("Модель оценена!")
    print(f"Train MSE: {train_mse:.6f}")
    print(f"Test MSE: {test_mse:.6f}")
    print("Вектора сохранены в 'models/movie_content_vectors_train.npz' и 'models/movie_content_vectors_test.npz'")


@task("data:movie_eval_vector_raw")
def eval_content_train_test_vectors_raw():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)["autoencoder"]
        paths = get_project_paths()

    model_path = config["model_output_path_raw"]

    movie_vectors_scaled_full = np.load(paths["processed_dir"] / "movie_vectors_scaled_full.npy")

    input_dim = movie_vectors_scaled_full.shape[1]

    # Загружаем обученную модель
    model = MovieAutoencoder(input_dim=input_dim, encoding_dim=64).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        # Восстановленные вектора (через весь автоэнкодер: encoder -> decoder)
        recon = model(torch.tensor(movie_vectors_scaled_full, dtype=torch.float32).to(device)).cpu().numpy()

        # Прогоняем только через энкодер
        movie_vectors_tensor = torch.tensor(movie_vectors_scaled_full, dtype=torch.float32).to(device)
        model_movies_full_vectors = model.encoder(movie_vectors_tensor).cpu().numpy()

    # Считаем метрики восстановления
    mse = mean_squared_error(movie_vectors_scaled_full, recon)


    eval_metrics = {
        "mse": mse,
    }

    # Сохраняем метрики для DVC
    os.makedirs("models", exist_ok=True)
    with open("models/eval_metrics_raw.json", "w") as f:
        json.dump(eval_metrics, f, indent=4)

    # Сохраняем content вектора для фильтрации
    np.savez_compressed("models/model_movies_full_vectors_raw.npz", vectors=model_movies_full_vectors)

    print("Модель оценена!")
    print(f"MSE: {mse:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", help="Список задач для выполнения")
    args = parser.parse_args()

    if args.tasks:
        main(args.tasks)  # Здесь передаем задачи, которые указаны в командной строке
