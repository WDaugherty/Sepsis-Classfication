import pandas as pd
import numpy as np
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro.infer.autoguide import AutoDelta, AutoDiagonalNormal, AutoContinuous
import matplotlib.pyplot as plt

def flow_based_model(data_tensor, n_flows=16):
    input_dim = data_tensor.shape[1]
    base_dist = dist.Normal(torch.zeros(input_dim).to(data_tensor.device), torch.ones(input_dim).to(data_tensor.device))
    flows = [dist.transforms.AffineTransform(torch.zeros(input_dim), torch.ones(input_dim)) for _ in range(n_flows)]
    flow_dist = dist.TransformedDistribution(base_dist, flows)
    return flow_dist

def train_flow_based_model(flow_dist, data):
    input_dim = data.shape[1]
    data_tensor = torch.from_numpy(data.values).float()

    # Move the data to the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_tensor = data_tensor.to(device)

    # Set up the base distribution with the correct device
    base_dist = dist.Normal(torch.zeros(input_dim).to(device), torch.ones(input_dim).to(device))

    # Set up the flow distribution with the updated base distribution
    flow_dist.base_dist = base_dist

    # Define the number of optimization steps
    n_steps = 1000

    # Set up the optimizer and the inference algorithm
    optimizer = Adam({"lr": 0.01})
    guide = AutoDiagonalNormal(flow_dist.base_dist)
    svi = SVI(flow_dist.model, guide, optimizer, loss=Trace_ELBO())

    # Perform the optimization
    for step in range(n_steps):
        loss = svi.step(data_tensor) / len(data)

        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss:.4f}")

    return flow_dist

def generate_flow_based_data(flow_dist, n_samples):
    latent_dim = flow_dist.base_dist.event_shape[0]
    device = next(flow_dist.parameters()).device
    noise = torch.randn(n_samples, latent_dim, device=device)
    synthetic_data = flow_dist.sample(noise.size()).detach().numpy()
    return synthetic_data

def plot_original_vs_synthetic(original_data, synthetic_data, target_column):
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

def flow(data, target_column):
    data = data.drop(target_column, axis=1)
    data_tensor = torch.tensor(data.values, dtype=torch.float32)

    flow_dist = flow_based_model(data_tensor)
    trained_flow_dist = train_flow_based_model(flow_dist, data)
    synthetic_data = generate_flow_based_data(trained_flow_dist, len(data))
    synthetic_df = pd.DataFrame(synthetic_data, columns=data.columns)

    plot_original_vs_synthetic(data, synthetic_df, target_column)

    return synthetic_df

