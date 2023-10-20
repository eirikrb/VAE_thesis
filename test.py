"""midlertidig test script"""

import numpy as np
import torch
import torch.nn as nn
from vae.encoders import Encoder_shallow
from vae.decoders import Decoder_shallow
from utils.dataloader import load_LiDARDataset

path_x =  'data/LiDAR_MovingObstaclesNoRules.csv'
path_y = 'data/risk_MovingObstaclesNoRules.csv'



data_train, data_val, data_test, dataloader_train, dataloader_val, dataloader_test  = load_LiDARDataset(path_x, path_y, 
                                                                                                        mode='max', 
                                                                                                        batch_size=16, 
                                                                                                        train_test_split=0.7,
                                                                                                        train_val_split=0.3,
                                                                                                        shuffle=True)

model = Encoder_shallow(latent_dims=12)

y = model(data_train.X)
#print(y.shape)


optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)
loss = nn.MSELoss()
training_loss = []
validation_loss = []

# Check if encoder can learn anything
for epoch in range(10):
    train_loss_epoch = 0
    num_batches = 0
    for x_batch, y_batch in dataloader_train:
        Y_pred = model(x_batch)              # Do forward pass
        loss_   = loss(Y_pred, y_batch)  
        loss_.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss = loss_.item()
        train_loss_epoch += train_loss
        num_batches += 1

    # validation step
    model.eval()
    with torch.no_grad():
        num_batches = 0
        avg_loss = 0
        for X_batch, y_batch in dataloader_val: 
            y_pred = model(X_batch)
            val_loss = loss(y_pred, y_batch)
            num_batches += 1
            avg_loss += val_loss
    avg_loss = avg_loss/num_batches
        
    model.train()
    val_loss = avg_loss.item() #val_loss.item()
    training_loss.append(train_loss_epoch/num_batches)
    validation_loss.append(val_loss)
    print('EPOCH', epoch+1, ': \tTraining loss:', train_loss_epoch/num_batches, '\tValidation loss:', val_loss)