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
from utils.plotting import plot_reconstructions, plot_loss, plot_loss_multiple_seeds, plot_separated_losses
from trainer import Trainer
import numpy as np
import matplotlib.pyplot as plt


# HYPERPRAMETERS
LEARNING_RATE = 0.001
N_EPOCH = 15
BATCH_SIZE = 16     
LATENT_DIMS = 12

def main():
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    num_seeds = 4 # set to 10 for final run for report
    
    # Load data
    datapath = 'data/LiDAR_MovingObstaclesNoRules.csv'
    """_, _, _, dataloader_train, dataloader_val, dataloader_test  = load_LiDARDataset(datapath,  
                                                                                    mode='max', 
                                                                                    batch_size=BATCH_SIZE, 
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
    """

    # CREATE MULTIPLE MODELS WITH DIFFERENT RANDOM SEEDS, TRAIN AND PLOT TOTAL, BCE AND KL LOSSES
    total_train_losses = np.zeros((num_seeds, N_EPOCH))
    total_val_losses = np.zeros((num_seeds, N_EPOCH))

    bce_train_losses = np.zeros((num_seeds, N_EPOCH))
    bce_val_losses = np.zeros((num_seeds, N_EPOCH))

    kl_train_losses = np.zeros((num_seeds, N_EPOCH))
    kl_val_losses = np.zeros((num_seeds, N_EPOCH))
    
    for i in range(num_seeds):
        print(f'Random seed {i+1}/{num_seeds}')
        # Generate different initialization of training and validation data for the new trainer
        _, _, _, dataloader_train, dataloader_val, dataloader_test  = load_LiDARDataset(datapath,  
                                                                                    mode='max', 
                                                                                    batch_size=BATCH_SIZE, 
                                                                                    train_test_split=0.7,
                                                                                    train_val_split=0.3,
                                                                                    shuffle=True,
                                                                                    extend_dataset_roll=True,
                                                                                    roll_degrees=[20,-20],
                                                                                    add_noise_to_train=True)
        # Create Variational Autoencoder for the new trainer
        encoder = Encoder_conv_deep(latent_dims=LATENT_DIMS)
        decoder = Decoder_circular_conv_shallow2(latent_dims=LATENT_DIMS)
        vae = VAE(encoder=encoder, decoder=decoder, latent_dims=LATENT_DIMS).to(device)
        optimizer = Adam(vae.parameters(), lr=LEARNING_RATE)
        
        trainer = Trainer(model=vae, 
                          epochs=N_EPOCH,
                          learning_rate=LEARNING_RATE,
                          batch_size=BATCH_SIZE,
                          dataloader_train=dataloader_train,
                          dataloader_val=dataloader_val,
                          optimizer=optimizer)
        trainer.train()

        total_train_losses[i,:] = trainer.training_loss['Total loss']
        total_val_losses[i,:] = trainer.validation_loss['Total loss']
        bce_train_losses[i,:] = trainer.training_loss['Reconstruction loss']
        bce_val_losses[i,:] = trainer.validation_loss['Reconstruction loss']
        kl_train_losses[i,:] = trainer.training_loss['KL divergence loss']
        kl_val_losses[i,:] = trainer.validation_loss['KL divergence loss']

        del trainer, vae, encoder, decoder, optimizer
    
    name = "EXAMPLE_PLOT"
    model_name_ = f'{name}_latent_dims_{LATENT_DIMS}'

    total_losses = [total_train_losses, total_val_losses]
    bce_losses = [bce_train_losses, bce_val_losses]
    kl_losses = [kl_train_losses, kl_val_losses]
    labels = ['Training loss', 'Validation loss']
    plot_separated_losses(total_losses, bce_losses, kl_losses, labels, model_name=model_name_, save=True)

    """    
    val_losses2 = val_losses + 1*np.random.randn(val_losses.shape[0], val_losses.shape[1])
    loss_trajs = [train_losses, val_losses, val_losses2]
    labels = ['Training loss', 'Validation loss', 'Validation loss + noise']
    plot_loss_multiple_seeds(loss_trajectories=loss_trajs, labels=labels, model_name=model_name_, save=True)
    """

    #plot_loss_multiple_seeds(training_losses=train_losses,validation_losses=val_losses, model_name=model_name_, save=True)

    
    # DELETE BELOW??
    """
    # TODO: create functions for plotting test error as function of beta (first check what b-VAE loss function looks like), latent dims, etc.
    # TODO: Wrap in several seeds
    betas = [0, 0.001, 0.01, 0.1, 0.25, 0.5, 1, 2] 
    _, _, _, dataloader_train, dataloader_val, dataloader_test  = load_LiDARDataset(datapath,  
                                                                            mode='max', 
                                                                            batch_size=BATCH_SIZE, 
                                                                            train_test_split=0.7,
                                                                            train_val_split=0.3,
                                                                            shuffle=True,
                                                                            extend_dataset_roll=True,
                                                                            roll_degrees=[20,-20],
                                                                            add_noise_to_train=True)
    # for each beta, create model, train for a set of epochs and get test error, plot th etest error as a function of beta
    test_losses = np.zeros(len(betas))
    for i, b in enumerate(betas):
        print(f'Training model with beta = {b}')
        # Create Variational Autoencoder for the new trainer
        encoder = Encoder_conv_deep(latent_dims=LATENT_DIMS)
        decoder = Decoder_circular_conv_shallow2(latent_dims=LATENT_DIMS)
        vae = VAE(encoder=encoder, decoder=decoder, latent_dims=LATENT_DIMS).to(device)
        optimizer = Adam(vae.parameters(), lr=LEARNING_RATE)
        trainer = Trainer(model=vae, 
                          epochs=N_EPOCH,
                          learning_rate=LEARNING_RATE,
                          batch_size=BATCH_SIZE,
                          dataloader_train=dataloader_train,
                          dataloader_val=dataloader_val,
                          optimizer=optimizer,
                          beta=b)
        trainer.train()
        
        # Get total loss from test set for the given model trained with beta = b
        test_loss = 0.0
        with torch.no_grad():
            for x in dataloader_test:
                x = x.to(device)
                x_hat, mu, log_var = trainer.model(x)
                loss = trainer.loss_function(x_hat, x, mu, log_var, b)
                test_loss += loss.item()
        
        test_losses[i] = test_loss/len(dataloader_test.dataset)

        del trainer, vae, encoder, decoder, optimizer
    
    print('Plotting test loss...')
    plt.style.use('ggplot')
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('axes', labelsize=12)
    fig, ax = plt.subplots()
    ax.plot(betas, test_losses, linewidth=2)
    ax.set_xlabel('\u03B2') # unicode beta
    ax.set_ylabel('Test loss')
    ax.legend()
    name = "ShallowConvVAE"
    model_name = f'{name}_latent_dims_{LATENT_DIMS}'
    plt.savefig(f'plots/testloss_small_betas_{model_name}.png', bbox_inches='tight')

    """
    
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