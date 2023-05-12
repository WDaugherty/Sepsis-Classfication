import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def train_autoencoder(data, latent_dim, epochs=1000, batch_size=32, lr=0.001):
    model = Autoencoder(data.shape[1], latent_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    data_tensor = torch.tensor(data.values, dtype=torch.float32)
    
    for epoch in range(epochs):
        idx = torch.randint(0, data_tensor.size(0), (batch_size,))
        batch = data_tensor[idx]
        
        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, batch)
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
    
    return model

def generate_autoencoder_data(model, n_samples):
    latent_dim = model.encoder[-1].out_features
    noise = torch.randn(n_samples, latent_dim)
    synthetic_data = model.decoder(noise).detach().numpy()
    return synthetic_data


def autoencoder_plot(original_data, synthetic_data, target_column):
    if not isinstance(original_data, pd.DataFrame):
        raise ValueError("original_data should be a pandas DataFrame")
    if not isinstance(synthetic_data, pd.DataFrame):
        synthetic_data = pd.DataFrame(synthetic_data, columns=original_data.columns)  # Convert synthetic_data to DataFrame
    
    if target_column not in original_data.columns:
        raise ValueError(f"Invalid target_column: {target_column} not found in original_data")
    if target_column not in synthetic_data.columns:
        raise ValueError(f"Invalid target_column: {target_column} not found in synthetic_data")
    
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


def autoencoder(data, target_column, latent_dim=10, epochs=1000, batch_size=32, lr=0.001):
    # Autoencoder function to generate and plot synthetic data
    model = train_autoencoder(data, latent_dim, epochs, batch_size, lr)
    synthetic_data = generate_autoencoder_data(model, n_samples=len(data))
    autoencoder_plot(data, synthetic_data, target_column)