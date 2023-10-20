import torch
from torch.optim import Adam
import torch.nn as nn
import os
import sys
from vae.VAE import VAE
from vae.encoders import Encoder_conv_shallow, Encoder_conv_deep
from vae.decoders import Decoder_conv_shallow, Decoder_conv_deep
import torch.nn.functional as F
from utils.dataloader import load_LiDARDataset
from utils.plotting import plot_reconstructions, plot_loss


# Hyper-params:
LEARNING_RATE = 0.001
N_EPOCH = 5
BATCH_SIZE = 32     
LATENT_DIMS = 12

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def loss_function(x_hat, x, mu, log_var):
    BCE_loss = F.binary_cross_entropy(x_hat, x, reduction='sum') # Reconstruction loss
    KLD_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) # KL divergence loss
    return BCE_loss + KLD_loss


def validation(model, dataloader_val):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x in dataloader_val:
            x = x.to(device)
            x_hat, mu, log_var = model(x)
            loss = loss_function(x_hat, x, mu, log_var)
            val_loss += loss.item()
    return val_loss/len(dataloader_val.dataset) # sjekk


def train(model, dataloader_train, optimizer):
    model.train()
    train_loss = 0.0
    for x_batch in dataloader_train:
        x_batch = x_batch.to(device)
        x_hat, mu, log_var = model(x_batch)
        loss = loss_function(x_hat, x_batch, mu, log_var)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss/len(dataloader_train.dataset) #sjekk


def main():
    
    # Load data
    path_x = 'data/LiDAR_MovingObstaclesNoRules.csv'
    _, _, _, dataloader_train, dataloader_val, dataloader_test  = load_LiDARDataset(path_x,  
                                                                                    mode='max', 
                                                                                    batch_size=16, 
                                                                                    train_test_split=0.7,
                                                                                    train_val_split=0.3,
                                                                                    shuffle=True,
                                                                                    extend_dataset_roll=True,
                                                                                    add_noise_to_train=True)

    # Create vae model
    encoder = Encoder_conv_shallow(latent_dims=LATENT_DIMS)
    decoder = Decoder_conv_shallow(latent_dims=LATENT_DIMS)
    vae = VAE(encoder=encoder, decoder=decoder, latent_dims=LATENT_DIMS).to(device)
    optimizer = Adam(vae.parameters(), lr=LEARNING_RATE)

    model_name_ = f'{vae.encoder.name}_latent_dims_{LATENT_DIMS}'


    # train+validate-loop
    training_loss = [] # maybe save to csv or something later so it can be compared to other configurations
    validation_loss = []
    for e in range(N_EPOCH):
        train_loss = train(vae, dataloader_train, optimizer)
        val_loss = validation(vae, dataloader_val)
        training_loss.append(train_loss)
        validation_loss.append(val_loss)
        print(f'\nEpoch: {e+1}/{N_EPOCH} \t|\t Train loss: {train_loss:.3f} \t|\t Val loss: {val_loss:.3f}')
    
    # Save model
    #vae.encoder.save(path='models/encoder_shallow.json')


    # PLOTTING
    plot_reconstructions(model=vae, dataloader=dataloader_test, model_name_=model_name_, device=device, num_examples=7, save=True)
    plot_loss(training_loss, validation_loss, save=True, model_name=model_name_)

    #create functions for plotting test error as function of beta (first check what b-VAE loss function looks like), latent dims, etc.

            






if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n\nKeyboard interrupt detected, exiting.')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
    finally:
        print('Done.')