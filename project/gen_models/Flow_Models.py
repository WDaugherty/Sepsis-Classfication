import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

# Define a flow-based model
def flow_based_model(data, n_flows=16):
    input_dim = data.shape[1]
    base_dist = dist.Normal(torch.zeros(input_dim), torch.ones(input_dim))
    flows = [dist.transforms.AffineTransform(torch.zeros(input_dim), torch.ones(input_dim)) for _ in range(n_flows)]
    flow_dist = dist.TransformedDistribution(base_dist, flows)
    return flow_dist

# Train the flow-based model
def train_flow_based_model(flow_dist, data, epochs=1000, lr=0.001):
    pyro.clear_param_store()
    svi = SVI(flow_dist, flow_dist, Adam({"lr": lr}), Trace_ELBO())

    data_tensor = torch.tensor(data.values, dtype=torch.float32)
    n_samples = data_tensor.size(0)

    for epoch in range(epochs):
        loss = svi.step(data_tensor) / n_samples
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")

    return flow_dist

# Generate synthetic data using the flow-based model
def generate_flow_based_data(flow_dist, n_samples):
    synthetic_data = flow_dist.sample(torch.Size([n_samples])).numpy()
    return synthetic_data