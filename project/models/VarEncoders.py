import torch
import torch.nn as nn
import torch.optim as optim

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
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
        mu, logvar = h[:, :latent_dim], h[:, latent_dim:]
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
        batch = data_tensor
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(batch)
        loss = vae_loss(recon_batch, batch, mu, logvar)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    return model
def generate_vae_data(model, n_samples):
    latent_dim = model.encoder[-1].out_features // 2
    noise = torch.randn(n_samples, latent_dim)
    synthetic_data = model.decoder(noise).detach().numpy()
    return synthetic_data