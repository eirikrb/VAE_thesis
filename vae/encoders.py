import torch
import torch.nn as nn
import os
import numpy as np


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# TODO: Detatch convolutional block from encoder, so that the block can be modified on its own

class Encoder_conv_shallow(nn.Module):
    def __init__(self, 
                 latent_dims:int=12,
                 n_sensors:int=180, 
                 kernel_overlap = 0.25):  
        super(Encoder_conv_shallow, self).__init__()

        self.name = 'conv_shallow'
        self.latent_dims = latent_dims
        self.in_channels = 1 
        self.kernel_size = round(n_sensors * kernel_overlap)  # 45
        self.kernel_size = self.kernel_size + 1 if self.kernel_size % 2 == 0 else self.kernel_size  # Make it odd sized
        # self.padding = (self.kernel_size - 1) // 2  # 22
        self.padding = self.kernel_size // 3  # 15
        self.stride = self.padding

        self.encoder_layer1 = nn.Sequential(
            nn.Conv1d(
                in_channels  = self.in_channels,
                out_channels = 1,
                kernel_size  = self.kernel_size,
                stride       = self.stride,
                padding      = self.padding,
                padding_mode = 'circular'
            )#,
            #nn.Flatten()
        )
        
        len_flat = 12 # bc. of combination of input dim, kernel, stride and padding. TODO: Automate.
        #self.linear = nn.Sequential(nn.Linear(len_flat, self.latent_dims), nn.ReLU()) # trenger kanskje ikke denne eller????, bare smell på en reul på mu og sigma

        # TEST MED FC LAYER

        """self.fc = nn.Sequential(
            nn.Linear(len_flat, 32),
            nn.ReLU(),
            nn.Linear(32,len_flat),
            nn.ReLU()
        ) """# dobbeltsjekk om denne delen her skal være som dette!!


        # REMOVE RELU BELOW???
        self.fc_mu = nn.Sequential(nn.Linear(len_flat, self.latent_dims), nn.ReLU()) # dobbeltsjekk om denne delen her skal være som dette!!
        self.fc_logvar = nn.Sequential(nn.Linear(len_flat, self.latent_dims), nn.ReLU()) 
        
        #self.fc_mu = nn.Linear(len_flat, self.latent_dims)
        #self.fc_logvar = nn.Linear(len_flat, self.latent_dims)

        """
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(device)
        self.N.scale = self.N.scale.to(device)
        self.kl = 0
        """

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var) # compute std.dev from log variance
        epsilon = torch.randn_like(std).to(device) 
        z = mu + (epsilon * std) # sampling
        return z

    def forward(self, x):
        # We do (not do) reparameterization in forward to sample from reparameterized z later on when we use encoder for feature extraction
        x = x.to(device)
        x = self.encoder_layer1(x)
        #x = self.fc(x) # TEEEEST
        #x = self.linear(x)
        mu =  self.fc_mu(x)
        log_var = self.fc_logvar(x)
        z = self.reparameterize(mu, log_var)
        return mu, log_var, z

    
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
    




class Encoder_conv_deep(nn.Module):
    def __init__(self, 
                 n_sensors:int=180, 
                 output_channels:list=[3,2,1], 
                 kernel_size:int=45,
                 latent_dims:int=12):  
        super(Encoder_conv_deep, self).__init__()
        self.name = 'conv_deep'
        self.n_sensors = n_sensors
        self.kernel_size = kernel_size
        self.output_channels = output_channels
        self.latent_dims = latent_dims

        self.conv_block = nn.Sequential(
            nn.Conv1d(
                in_channels  = 1,
                out_channels = self.output_channels[0],
                kernel_size  = 45,
                stride       = 15,
                padding      = 15,
                padding_mode = 'circular'
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels  = self.output_channels[0],
                out_channels = self.output_channels[1],
                kernel_size  = 3,
                stride       = 1,
                padding      = 1,
                padding_mode = 'circular'
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels  = self.output_channels[1],
                out_channels = self.output_channels[2],
                kernel_size  = 3,
                stride       = 1,
                padding      = 1,
                padding_mode = 'circular'
            )#,
            #nn.ReLU()
            #nn.Flatten()
        )
        
        len_flat = 12 # bc. of combination of input dim, kernel, stride and padding. TODO: Automate.

        # REMOVE RELU BELOW???
        self.fc_mu = nn.Sequential(nn.Linear(len_flat, self.latent_dims), nn.ReLU()) 
        self.fc_logvar = nn.Sequential(nn.Linear(len_flat, self.latent_dims), nn.ReLU())

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var) # compute std.dev from log variance
        epsilon = torch.randn_like(std).to(device) 
        z = mu + (epsilon * std) # sampling
        return z

    def forward(self, x):
        x = x.to(device)
        x = self.conv_block(x)
        mu =  self.fc_mu(x)
        log_var = self.fc_logvar(x)
        z = self.reparameterize(mu, log_var)
        return mu, log_var, z

    
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))