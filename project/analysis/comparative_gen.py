import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.stats as stats

from gen_models.smote_gan import smote_gan
from gen_models.autoencoder import generate_autoencoder_data

# MCMC function: generate_synthetic_data_mcmc
def generate_synthetic_data_mcmc(target_dist, proposal_std, n_samples, initial_state=0):
    samples, acceptance_rate = metropolis_hastings(target_dist, proposal_std, n_samples, initial_state)
    return samples

# Define target distribution for MCMC
def target_distribution(x):
    return 0.3 * stats.norm(-3, 1).pdf(x) + 0.7 * stats.norm(3, 1).pdf(x)

# Real data: generate real data
real_data = pd.DataFrame({
    'value': np.concatenate([
        np.random.normal(-3, 1, size=500),
        np.random.normal(3, 1, size=500)
    ])
})

# Generate synthetic data
n_samples = 1000
gan_synthetic_data = generate_synthetic_data_gan(real_data, n_samples)
rule_based_synthetic_data = generate_synthetic_data_rule_based(n_samples)
mcmc_synthetic_data = pd.DataFrame({'value': generate_synthetic_data_mcmc(target_distribution, 2, n_samples)})

# Compute and compare descriptive statistics
def compute_statistics(df):
    return df.describe().T

real_stats = compute_statistics(real_data)
gan_stats = compute_statistics(gan_synthetic_data)
rule_based_stats = compute_statistics(rule_based_synthetic_data)
mcmc_stats = compute_statistics(mcmc_synthetic_data)

print("Real Data:")
print(real_stats)
print("\nGAN Synthetic Data:")
print(gan_stats)
print("\nRule-based Synthetic Data:")
print(rule_based_stats)
print("\nMCMC Synthetic Data:")
print(mcmc_stats)

# Plot the real and synthetic data
def plot_data(real_data, gan_data, rule_based_data, mcmc_data):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)

    sns.histplot(real_data, kde=True, ax=axes[0, 0])
    axes[0, 0].set_title("Real Data")

    sns.histplot(gan_data, kde=True, ax=axes[0, 1])
    axes[0, 1].set_title("GAN Synthetic Data")

    sns.histplot(rule_based_data, kde=True, ax=axes[1, 0])
    axes[1, 0].set_title("Rule-based Synthetic Data")

    sns.histplot(mcmc_data, kde=True, ax=axes[1, 1])
    axes[1, 1].set_title("MCMC Synthetic Data")

    plt.tight_layout()
    plt.show()

plot_data(real_data, gan_synthetic_data, rule_based_synthetic_data, mcmc_synthetic_data)
