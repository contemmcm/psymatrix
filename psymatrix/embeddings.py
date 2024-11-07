import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from psymatrix.utils import load_metafeatures

LEARNING_RATE = 1e-3
BATCH_SIZE = 32


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2_mean = nn.Linear(128, latent_dim)
        self.fc2_log_var = nn.Linear(128, latent_dim)

        # Decoder
        self.fc3 = nn.Linear(latent_dim, 128)
        self.fc4 = nn.Linear(128, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc2_mean(h1), self.fc2_log_var(h1)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, log_var = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var


def loss_function(recon_x, x, mu, log_var, n_features):

    BCE = F.binary_cross_entropy(recon_x, x.view(-1, n_features), reduction="sum")
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD


# Training loop
def train(epoch, model, train_loader, optimizer, n_features):
    model.train()
    train_loss = 0
    for batch_idx, (data,) in enumerate(train_loader):
        optimizer.zero_grad()
        recon_batch, mu, log_var = model(data)
        loss = loss_function(recon_batch, data, mu, log_var, n_features)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    train_loss /= len(train_loader.dataset)

    return train_loss


def floatOrZero(value):
    try:
        return float(value)
    except ValueError:
        return 0.0


def load_selected_features():
    """
    Load the selected features from the file metatafeatures.all.config
    """
    with open("metafeatures.config", encoding="utf8") as f:
        features = [line.strip() for line in f.readlines()]

    return np.array(features)


def load_dataset(datasets_fname, normalize=True):

    selectedf_features = load_selected_features()

    df = load_metafeatures(datasets_fname)
    df = df[selectedf_features]

    # Replace NaN values with 0
    df = df.fillna(0.0)

    if normalize:
        scaler = MinMaxScaler(feature_range=(0, 1))
        df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        x_train = torch.tensor(df_normalized.to_numpy(), dtype=torch.float32)
    else:
        x_train = torch.tensor(df.to_numpy(), dtype=torch.float32)

    return x_train


def train_vae(latent_dim, num_epochs=300, save=True):

    x_train = load_dataset("emnlp24_datasets.txt", normalize=True)

    input_dim = x_train.shape[1]

    # Model
    model = VAE(input_dim, latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Dataset
    train_dataset = TensorDataset(x_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(0, num_epochs):
        loss = train(epoch, model, train_loader, optimizer, n_features=input_dim)
        print(f"Epoch: {epoch}, Loss: {loss}")

    # Example of encoding and decoding
    with torch.no_grad():
        samples = model(x_train)

    mu = samples[1].numpy()
    logvar = samples[2].numpy()

    df_mu = pd.DataFrame(mu, columns=[f"mu_{i}" for i in range(latent_dim)])
    df_logvar = pd.DataFrame(logvar, columns=[f"logvar_{i}" for i in range(latent_dim)])

    # combine the results
    df = pd.concat([df_mu, df_logvar], axis=1)

    if save:
        df.to_csv(f"vae_embeddings/{latent_dim:03d}_embeddings.csv", index=False)
        np.savetxt(f"vae_embeddings/{latent_dim:03d}_embeddings_mu.txt", mu)
        np.savetxt(f"vae_embeddings/{latent_dim:03d}_embeddings_logvar.txt", logvar)

    return df


def run():
    latent_dim = 8
    train_vae(latent_dim=latent_dim, num_epochs=400, save=True)


if __name__ == "__main__":
    run()
