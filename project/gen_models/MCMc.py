import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd

def target_distribution(x, distribution='continuous', p=None):
    if distribution == 'continuous':
        return 0.3 * stats.norm(-3, 1).pdf(x) + 0.7 * stats.norm(3, 1).pdf(x)
    elif distribution == 'binary':
        return (1 - x) * (1 - p) + x * p
    else:
        raise ValueError("Invalid distribution type. Choose 'continuous' or 'binary'.")

def metropolis_hastings(target_dist, proposal_std, n_samples, distribution, p=None, initial_state=0):
    samples = np.zeros(n_samples)
    current_state = initial_state
    acceptance_count = 0

    for i in range(n_samples):
        if distribution == 'continuous':
            proposed_state = current_state + np.random.normal(0, proposal_std)
        elif distribution == 'binary':
            proposed_state = np.random.choice([0, 1])
        else:
            raise ValueError("Invalid distribution type. Choose 'continuous' or 'binary'.")

        acceptance_ratio = target_dist(proposed_state, distribution, p) / target_dist(current_state, distribution, p)

        if np.random.rand() < acceptance_ratio:
            current_state = proposed_state
            acceptance_count += 1

        samples[i] = current_state

    acceptance_rate = acceptance_count / n_samples
    return samples, acceptance_rate

def generate_synthetic_data_mcmc(df, target_column):
    has_sepsis_data = df[target_column].values

    # Estimate the Bernoulli parameter p
    p = np.mean(has_sepsis_data)

    # Generate synthetic data using Metropolis-Hastings algorithm
    n_samples = 10
    proposal_std = 2
    initial_state = 0
    distribution = 'binary'
    samples, acceptance_rate = metropolis_hastings(target_distribution, proposal_std, n_samples, distribution, p, initial_state)

    synthetic_data = pd.DataFrame(samples, columns=[target_column])

    return synthetic_data

def plot_original_vs_synthetic_mcmc(original_data, synthetic_data, target_column):
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

def mcmc(data, target_column):
    synthetic_data = generate_synthetic_data_mcmc(data, target_column)
    plot_original_vs_synthetic_mcmc(data, synthetic_data, target_column)
