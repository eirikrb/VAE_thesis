"""MIDLERTIDIG TESTFIL FOR VAE"""

import torch
import torch.nn as nn
import os
import numpy as np
from utils.dataloader import load_LiDARDataset
from vae.encoders import Encoder_shallow
from vae.decoders import Decoder_shallow

path_x =  'data/LiDAR_MovingObstaclesNoRules.csv'
path_y = 'data/risk_MovingObstaclesNoRules.csv'

data_train, data_val, data_test, dataloader_train, dataloader_val, dataloader_test  = load_LiDARDataset(path_x, path_y, 
                                                                                                        mode='max', 
                                                                                                        batch_size=16, 
                                                                                                        train_test_split=0.7,
                                                                                                        train_val_split=0.3,
                                                                                                        shuffle=True)

en = Encoder_shallow()
de = Decoder_shallow()

data = data_train.X
print(data.shape)
z = en(data)
print(z.shape)
reconstructed = de(z)
print(reconstructed.shape)

