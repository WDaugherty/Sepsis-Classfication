import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim  # Store latent_dim as an instance variable
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim * 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Tanh()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = h[:, :self.latent_dim], h[:, self.latent_dim:]  # Use self.latent_dim
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar


def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.MSELoss()(recon_x, x)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss


def train_vae(data, latent_dim, epochs=1000, batch_size=32, lr=0.001):
    model = VAE(data.shape[1], latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    data_tensor = torch.tensor(data.values, dtype=torch.float32)

    for epoch in range(epochs):
        idx = torch.randint(0, data_tensor.size(0), (batch_size,))
        batch = data_tensor[idx]  # Use randomly selected batch
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(batch)
        loss = vae_loss(recon_batch, batch, mu, logvar)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    return model


def generate_vae_data(model, n_samples):
    latent_dim = model.latent_dim
    noise = torch.randn(n_samples, latent_dim)
    synthetic_data = model.decoder(noise).detach().numpy()
    num_columns = synthetic_data.shape[1]
    synthetic_data = pd.DataFrame(synthetic_data, columns=[f"Feature_{i}" for i in range(num_columns)])  # Specify column names
    return synthetic_data



def vae_plot(original_data, synthetic_data, target_column):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    original_counts = original_data[target_column].value_counts(normalize=True)
    synthetic_counts = synthetic_data[target_column].value_counts(normalize=True)

    axes[0].bar(original_counts.index, original_counts.values, alpha=0.5, label="Original Data")
    axes[1].bar(synthetic_counts.index, synthetic_counts.values, alpha=0.5, label="Synthetic Data")

    axes[0].set_title("Original Data")
    axes[1].set_title("Synthetic Data")
    axes[0].set_ylabel("Proportion")
    axes[1].set_ylabel("Proportion")

    for ax in axes:
        ax.set_xlabel(target_column)
        ax.legend()

    plt.show()


def vae(data, target_column, latent_dim=10, epochs=1000, batch_size=32, lr=0.001):
    # Autoencoder function to generate and plot synthetic data
    model = train_vae(data, latent_dim, epochs, batch_size, lr)
    synthetic_data = generate_vae_data(model, n_samples=len(data))
    vae_plot(data, synthetic_data, target_column)