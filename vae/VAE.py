import torch
import torch.nn as nn


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class VAE(nn.Module):
    def __init__(self, encoder, decoder, latent_dims:int=12):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dims = latent_dims

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var) # compute std.dev from log variance
        epsilon = torch.randn_like(std).to(device) 
        z = mu + (epsilon * std) # sampling
        return z
    
    def forward(self, x):
        x = x.to(device)
        mu, sigma, z = self.encoder(x)
        #z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, sigma
       
    def save(self, path):
        torch.save(self.state_dict(),path)

    def load(self, path):
        self.load_state_dict(torch.load(path))