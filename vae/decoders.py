import torch
import torch.nn as nn
import os
import numpy as np

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class Decoder_conv_shallow(nn.Module):
    def __init__(self, 
                 latent_dims:int=12,
                 n_sensors:int=180, 
                 kernel_overlap = 0.25):  
        super(Decoder_conv_shallow, self).__init__()
        self.latent_dims = latent_dims
        self.in_channels = 1 
        self.out_channels = 1
        self.kernel_size = round(n_sensors * kernel_overlap)  # 45
        self.kernel_size = self.kernel_size + 1 if self.kernel_size % 2 == 0 else self.kernel_size  # Make it odd sized
        # self.padding = (self.kernel_size - 1) // 2  # 22
        self.padding = self.kernel_size // 3  # 15
        self.stride = self.padding

        len_flat = 12 # bc. of combination of input dim, kernel, stride and padding. TODO: Automate.

        self.linear = nn.Sequential(
            nn.Linear(self.latent_dims, len_flat), 
            nn.ReLU()#,
            #nn.Linear(len_flat, 32),
            #nn.ReLU(),
            #nn.Linear(32,len_flat),
            #nn.ReLU()
        )

        self.decoder_layer1 = nn.Sequential(
            #nn.Unflatten(dim=0, unflattened_size=(1,len_flat)),
            nn.ConvTranspose1d(
                in_channels=self.in_channels, 
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                padding_mode='zeros'
            ),
            nn.Sigmoid() # alle har bare sigmoid her, idk why
        )#'circular' is not supported for convtranspose1d :(

    def forward(self, z):
        z = self.linear(z)
        x_hat = self.decoder_layer1(z)
        return x_hat
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))



class Decoder_circular_conv_shallow(nn.Module):
    def __init__(self, 
                 latent_dims:int=12,
                 n_sensors:int=180, 
                 kernel_overlap = 0.25):  
        super(Decoder_circular_conv_shallow, self).__init__()
        self.latent_dims = latent_dims
        self.in_channels = 1 
        self.out_channels = 1
        self.kernel_size = round(n_sensors * kernel_overlap)  # 45
        self.kernel_size = self.kernel_size + 1 if self.kernel_size % 2 == 0 else self.kernel_size  # Make it odd sized
        # self.padding = (self.kernel_size - 1) // 2  # 22
        self.padding = self.kernel_size // 3  # 15
        self.stride = self.padding

        len_flat = 12 # bc. of combination of input dim, kernel, stride and padding. TODO: Automate.

        self.linear = nn.Sequential(
            nn.Linear(self.latent_dims, len_flat), 
            nn.ReLU()
        )

        # No nn.sequentioal bc. of circular padding in between layers
        self.deconv_block = nn.ConvTranspose1d(
            in_channels=self.in_channels, 
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            #padding=self.padding,
            #output_padding=1,
        )

        self.sigmoid = nn.Sigmoid()


    def forward(self, z):
        """ 
        Following pseudocode from https://www.tu-chemnitz.de/etit/proaut/publications/schubert19_IV.pdf chapter III section B
        Doesnt work super-well
        """ 
        z = self.linear(z)
        # 1: Run transposed conv without padding
        z_t_conv = self.deconv_block(z) # Output shape: (width*stride + p_w), where p_w = max(0, k_w - stride)
        # 2: add the first pw left columns to the last pw right columns, and add the last pw right columns to the first pw left columns
        pad_width = int(max(0, self.kernel_size - self.stride))
        # Right side
        z_t_conv[:, :, -pad_width:] += z_t_conv[:, :, :pad_width]
        # Left side
        z_t_conv[:, :, :pad_width] += z_t_conv[:, :, -pad_width:]
        # 3: Remove the first pw/2 left columns and the last pw/2 right columns
        crop = pad_width // 2
        z_t_conv = z_t_conv[:, :, crop : -crop]

        x_hat = self.sigmoid(z_t_conv)
        return x_hat
        
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))




class Decoder_conv_deep(nn.Module):
    def __init__(self, 
                 latent_dims:int=12,
                 n_sensors:int=180, 
                 output_channels:list=[1,2,3], 
                 kernel_size:int=45,):  
        super(Decoder_conv_deep, self).__init__()

        self.n_sensors = n_sensors
        self.kernel_size = kernel_size
        self.output_channels = output_channels
        self.latent_dims = latent_dims

        len_flat = 12 # bc. of combination of input dim, kernel, stride and padding. TODO: Automate.

        self.linear = nn.Sequential(nn.Linear(self.latent_dims, len_flat), nn.ReLU())
        
        self.deconv_block = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels  = self.output_channels[0],
                out_channels = self.output_channels[1],
                kernel_size  = 3,
                stride       = 1,
                padding      = 1,
                padding_mode = 'zeros'
            ),
            nn.ReLU(),
            nn.ConvTranspose1d(
                in_channels  = self.output_channels[1],
                out_channels = self.output_channels[2],
                kernel_size  = 3,
                stride       = 1,
                padding      = 1,
                padding_mode = 'zeros'
            ),
            nn.ReLU(),
            nn.ConvTranspose1d(
                in_channels  = self.output_channels[2],
                out_channels = 1,
                kernel_size  = 45,
                stride       = 15,
                padding      = 15,
                padding_mode = 'zeros'
            ),
            nn.Sigmoid()
        )

    def forward(self, z):
        z = self.linear(z)
        x_hat = self.deconv_block(z)
        return x_hat
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))