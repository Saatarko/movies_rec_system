import os
import torch
import mlflow
import mlflow.pytorch
import yaml

from torch import nn, optim
from torch.utils.data import DataLoader, random_split

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

def content_vector_autoencoder(train_data):
    # Загружаем параметры из DVC-конфига
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)["autoencoder"]

    encoding_dim = config["encoding_dim"]
    batch_size = config["batch_size"]
    num_epochs = config["num_epochs"]
    lr = config["learning_rate"]
    model_path = config["model_output_path"]

    input_dim = train_data.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MovieAutoencoder(input_dim, encoding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X = torch.tensor(train_data, dtype=torch.float32)
    train_size = int(0.9 * len(X))
    val_size = len(X) - train_size
    train_dataset, val_dataset = random_split(X, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

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

            print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")
            mlflow.log_metric("train_loss", avg_train, step=epoch)
            mlflow.log_metric("val_loss", avg_val, step=epoch)

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)
        mlflow.pytorch.log_model(model, "model")


def eval_content_train_test_vectors(model_path, movie_vectors_scaled_train, movie_vectors_scaled_test):
    # Подгрузка модели
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_movies = MovieAutoencoder(input_dim=movie_vectors_scaled_train.shape[1], encoding_dim=64).to(device)
    model_movies.load_state_dict(torch.load(model_path))
    model_movies.eval()  # Переводим модель в режим инференса

    with torch.no_grad():
        # Прогонка через энкодер
        movie_content_vectors_train = model_movies.encoder(
            torch.tensor(movie_vectors_scaled_train, dtype=torch.float32).to(device)).cpu().numpy()
        movie_content_vectors_test = model_movies.encoder(
            torch.tensor(movie_vectors_scaled_test, dtype=torch.float32).to(device)).cpu().numpy()

    # Сохранение векторов
    os.makedirs('models', exist_ok=True)
    np.savez_compressed("models/movie_content_vectors_train.npz", vectors=movie_content_vectors_train)
    np.savez_compressed("models/movie_content_vectors_test.npz", vectors=movie_content_vectors_test)

    print("Вектора сохранены в 'models/movie_content_vectors_train.npz' и 'models/movie_content_vectors_test.npz'")

if __name__ == "__main__":
    import numpy as np

    # Пример загрузки данных — замени на реальные данные
    from data_processing import generate_content_vector_for_offtest
    train_vectors, test_vector = generate_content_vector_for_offtest()

    content_vector_autoencoder(train_vectors)

    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)["autoencoder"]

    eval_content_train_test_vectors( config["model_output_path"], train_vectors, test_vector)