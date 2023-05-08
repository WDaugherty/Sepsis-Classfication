import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Define the target distribution (Gaussian mixture model)
def target_distribution(x):
    return 0.3 * stats.norm(-3, 1).pdf(x) + 0.7 * stats.norm(3, 1).pdf(x)

# Metropolis-Hastings algorithm
def metropolis_hastings(target_dist, proposal_std, n_samples, initial_state=0):
    samples = np.zeros(n_samples)
    current_state = initial_state
    acceptance_count = 0

    for i in range(n_samples):
        proposed_state = current_state + np.random.normal(0, proposal_std)
        acceptance_ratio = target_dist(proposed_state) / target_dist(current_state)

        if np.random.rand() < acceptance_ratio:
            current_state = proposed_state
            acceptance_count += 1

        samples[i] = current_state

    acceptance_rate = acceptance_count / n_samples
    return samples, acceptance_rate


# Generate synthetic data using Metropolis-Hastings algorithm
n_samples = 10000
proposal_std = 2
initial_state = 0
samples, acceptance_rate = metropolis_hastings(target_distribution, proposal_std, n_samples, initial_state)

# Plot the synthetic data and the target distribution
x = np.linspace(-8, 8, 1000)
plt.hist(samples, bins=50, density=True, alpha=0.5, label="Samples")
plt.plot(x, target_distribution(x), label="Target Distribution")
plt.xlabel("x")
plt.ylabel("Density")
plt.legend()
plt.show()

print(f"Acceptance rate: {acceptance_rate}")
