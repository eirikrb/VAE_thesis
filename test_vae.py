"""MIDLERTIDIG TESTFIL FOR VAE"""

import torch
import torch.nn as nn
import os
import numpy as np
from utils.dataloader import load_LiDARDataset

path_x =  'data/LiDAR_MovingObstaclesNoRules.csv'

data_train, data_val, data_test, dataloader_train, dataloader_val, dataloader_test  = load_LiDARDataset(path_x,
                                                                                                        mode='max', 
                                                                                                        batch_size=16, 
                                                                                                        train_test_split=0.7,
                                                                                                        train_val_split=0.3,
                                                                                                        shuffle=True)

#en = Encoder_shallow()
#de = Decoder_shallow()
"""

data = data_train.X
print(data.shape)
z = en(data)
print(z.shape)
reconstructed = de(z)
print(reconstructed.shape)

kernel_size = 45
stride = 15

deconv_block = nn.ConvTranspose1d(
    in_channels=1, 
    out_channels=1,
    kernel_size=45,
    stride=15,
    #padding=self.padding,
    #output_padding=1,
)

z = torch.ones(16, 1, 12)

# 1: Run transposed conv without padding
z_t_conv = deconv_block(z) # Output shape: (width*stride + p_w), where p_w = max(0, k_w - stride)
# 2: add the first pw left columns to the last pw right columns, and add the last pw right columns to the first pw left columns
pad_width = int(max(0, kernel_size - stride))
# Right side
z_t_conv[:, :, -pad_width:] += z_t_conv[:, :, :pad_width]
# Left side
z_t_conv[:, :, :pad_width] += z_t_conv[:, :, -pad_width:]
# 3: Remove the first pw/2 left columns and the last pw/2 right columns
crop = pad_width // 2
x_hat = z_t_conv[:, :, crop : -crop]

print(x_hat.shape)
print(x_hat[0,1,:])
"""

