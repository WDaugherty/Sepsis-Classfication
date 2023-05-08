import torch
import torch.nn as nn
import torch.optim as optim

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
