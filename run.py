import torch
from torch.optim import Adam
import torch.nn as nn
import os
import sys
from vae.VAE import VAE
from vae.encoders import Encoder_conv_shallow, Encoder_conv_deep
from vae.decoders import Decoder_conv_shallow, Decoder_conv_deep, Decoder_circular_conv_shallow2
import torch.nn.functional as F
from utils.dataloader import load_LiDARDataset
from utils.plotting import plot_reconstructions, plot_loss
from trainer import Trainer


# HYPERPRAMETERS
LEARNING_RATE = 0.001
N_EPOCH = 40
BATCH_SIZE = 32     
LATENT_DIMS = 12

def main():
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # Load data
    datapath = 'data/LiDAR_MovingObstaclesNoRules.csv'
    _, _, _, dataloader_train, dataloader_val, dataloader_test  = load_LiDARDataset(datapath,  
                                                                                    mode='max', 
                                                                                    batch_size=16, 
                                                                                    train_test_split=0.7,
                                                                                    train_val_split=0.3,
                                                                                    shuffle=True,
                                                                                    extend_dataset_roll=True,
                                                                                    roll_degrees=[20,-20],
                                                                                    add_noise_to_train=True)
    
    # Create Variational Autoencoder(s)
    encoder = Encoder_conv_deep(latent_dims=LATENT_DIMS)
    decoder = Decoder_circular_conv_shallow2(latent_dims=LATENT_DIMS)
    vae = VAE(encoder=encoder, decoder=decoder, latent_dims=LATENT_DIMS).to(device)
    optimizer = Adam(vae.parameters(), lr=LEARNING_RATE)
    name = "ShallowConvVAE"
    
    model_name_ = f'{name}_latent_dims_{LATENT_DIMS}'
    
    # Train model
    trainer = Trainer(model=vae, 
                      epochs=N_EPOCH,
                      learning_rate=LEARNING_RATE,
                      batch_size=BATCH_SIZE,
                      dataloader_train=dataloader_train,
                      dataloader_val=dataloader_val,
                      optimizer=optimizer)
    
    trainer.train()
    
    
    # Save model
    #vae.encoder.save(path=f'models/{model_name_}.json')


    # PLOTTING
    plot_reconstructions(model=trainer.model, 
                         dataloader=dataloader_test, 
                         model_name_=model_name_, 
                         device=device, 
                         num_examples=20, 
                         save=True, 
                         loss_func=trainer.loss_function)
    
    plot_loss(training_loss=trainer.training_loss, 
              validation_loss=trainer.validation_loss, 
              save=True, 
              model_name=model_name_)

    # TODO: create functions for plotting test error as function of beta (first check what b-VAE loss function looks like), latent dims, etc.

    
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