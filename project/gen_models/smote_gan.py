import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# Provided GAN code goes here
def generate_synthetic_data(data, n_samples):
    # Define the generator model
    # Define the generator model
    class Generator(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(Generator, self).__init__()
            self.fc1 = nn.Linear(input_dim, 128)
            self.fc2 = nn.Linear(128, 256)
            self.fc3 = nn.Linear(256, 512)
            self.fc4 = nn.Linear(512, output_dim)
            self.activation = nn.LeakyReLU(0.2)
            self.dropout = nn.Dropout(0.3)
            self.noise = nn.Parameter(torch.zeros(input_dim))

        def forward(self, x):
            x = x + self.noise * torch.randn(x.size())
            x = self.activation(self.fc1(x))
            x = self.dropout(x)
            x = self.activation(self.fc2(x))
            x = self.dropout(x)
            x = self.activation(self.fc3(x))
            x = torch.tanh(self.fc4(x))
            return x

    # Define the discriminator model
    class Discriminator(nn.Module):
        def __init__(self, input_dim):
            super(Discriminator, self).__init__()
            self.fc1 = nn.Linear(input_dim, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 128)
            self.fc4 = nn.Linear(128, 1)
            self.activation = nn.LeakyReLU(0.2)
            self.dropout = nn.Dropout(0.3)

        def forward(self, x):
            x = self.activation(self.fc1(x))
            x = self.dropout(x)
            x = self.activation(self.fc2(x))
            x = self.dropout(x)
            x = self.activation(self.fc3(x))
            x = torch.sigmoid(self.fc4(x))
            return x

    # Initialize models
    input_dim = data.shape[1]
    output_dim = data.shape[1]
    generator = Generator(input_dim, output_dim)
    discriminator = Discriminator(output_dim)

    # Define loss and optimizers
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Train the GAN model
    batch_size = 8
    epochs = 100
    for epoch in range(epochs):
        real_data = torch.tensor(data.sample(n=batch_size).values, dtype=torch.float32)
        real_labels = torch.ones((batch_size, 1))
        noise = torch.randn((batch_size, input_dim))
        synthetic_data = generator(noise)
        synthetic_labels = torch.zeros((batch_size, 1))

        # Train discriminator
        optimizer_D.zero_grad()

        real_loss = criterion(discriminator(real_data), real_labels)
        synthetic_loss = criterion(discriminator(synthetic_data.detach()), synthetic_labels)
        d_loss = (real_loss + synthetic_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        # Train generator
        optimizer_G.zero_grad()

        g_loss = criterion(discriminator(synthetic_data), real_labels)

        g_loss.backward()
        optimizer_G.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Discriminator Loss: {d_loss.item()}, Generator Loss: {g_loss.item()}")

    # Generate synthetic samples
    noise = torch.randn((n_samples, input_dim))
    synthetic_samples = generator(noise).detach().numpy()

    # Round synthetic samples to integers
    synthetic_samples = np.round(synthetic_samples)

    # Convert synthetic samples to pandas DataFrame
    synthetic_data = pd.DataFrame(synthetic_samples, columns=data.columns)

    # Make sure signal_length is a positive integer value
    synthetic_data['signal_length'] = synthetic_data['signal_length'].abs().astype(int)

    return synthetic_data  # Add this line

def plot_original_vs_synthetic_gan(original_data, synthetic_data, target_column):
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

def smote_gan(data, target_column, n_samples):
    # Split data into features and target
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # Apply SMOTE
    smote = SMOTE()

    #Test data
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("X head:", X.head())
    print("y head:", y.head())
    print("Target column value counts:", y.value_counts())


    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Create new DataFrame with SMOTE-resampled data
    resampled_data = pd.concat([X_resampled, y_resampled], axis=1)

    # Apply GAN to generate synthetic data
    synthetic_data = generate_synthetic_data(resampled_data, n_samples)

    return synthetic_data
