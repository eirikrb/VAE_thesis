import torch
from torch.optim import Adam
import torch.nn as nn
import os
import sys
from vae.VAE import VAE
from vae.encoders import Encoder_conv_shallow, Encoder_conv_deep
from vae.decoders import Decoder_circular_conv_shallow2, Decoder_circular_conv_deep
from utils.dataloader import load_LiDARDataset, concat_csvs
from utils.plotting import *
from trainer import Trainer
from tester import Tester
import numpy as np
import argparse


# HYPERPRAMETERS
LEARNING_RATE = 0.001
N_EPOCH = 25
BATCH_SIZE = 64     
LATENT_DIMS = 12

def main(args):
    # Set hyperparameters
    BATCH_SIZE = args.batch_size        # Default: 16
    N_EPOCH = args.epochs               # Default: 15
    LATENT_DIMS = args.latent_dims      # Default: 12
    LEARNING_RATE = args.learning_rate  # Default: 0.001
    NUM_SEEDS = args.num_seeds          # Default: 1
    BETA = args.beta                    # Default: 1
    EPS_WEIGHT = 1                      # Default: 1
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Data paths
    path_empty = "data/LiDAR_synthetic_empty.csv"
    path_moving_dense = "data/LiDAR_synthetic_onlyMovingObst_dense.csv"
    path_moving_sparse = "data/LiDAR_synthetic_onlyMovingObst_sparse.csv"
    path_static_dense = "data/LiDAR_synthetic_onlyStaticObst_dense.csv"
    path_static_moving = "data/LiDAR_synthetic_staticMovingObst.csv"
    path_moving_obs_no_rules = "data/LiDAR_MovingObstaclesNoRules.csv"

    DATA_PATHS = [path_moving_dense, path_static_dense, path_static_moving, path_moving_obs_no_rules]

    concat_path = 'data/concatinated_data.csv'
    concat_csvs(DATA_PATHS, concat_path)
    
    # Load data
    datapath = concat_path 
    rolling_degrees = [20,-20]
    num_rotations = 2000
    _, _, _, dataloader_train, dataloader_val, dataloader_test  = load_LiDARDataset(datapath,  
                                                                                mode='max', 
                                                                                batch_size=BATCH_SIZE, 
                                                                                train_test_split=0.7,
                                                                                train_val_split=0.3,
                                                                                shuffle=True,
                                                                                extend_dataset_roll=True,
                                                                                num_rotations=num_rotations,
                                                                                roll_degrees=rolling_degrees,
                                                                                add_noise_to_train=True)
    #datapath = 'data/LiDAR_MovingObstaclesNoRules.csv'
    if args.mode == 'train':
        # Set global model name 
        name = args.model_name
        model_name_ = f'{name}_latent_dims_{LATENT_DIMS}'

        # Create Variational Autoencoder(s)
        if args.model_name == 'ShallowConvVAE':
            encoder = Encoder_conv_shallow(latent_dims=LATENT_DIMS, eps_weight=EPS_WEIGHT)
            decoder = Decoder_circular_conv_shallow2(latent_dims=LATENT_DIMS)
            vae = VAE(encoder=encoder, decoder=decoder, latent_dims=LATENT_DIMS).to(device)

        if args.model_name == 'DeepConvVAE':
            encoder = Encoder_conv_deep(latent_dims=LATENT_DIMS, eps_weight=EPS_WEIGHT)
            decoder = Decoder_circular_conv_deep(latent_dims=LATENT_DIMS)
            vae = VAE(encoder=encoder, decoder=decoder, latent_dims=LATENT_DIMS).to(device)

        # Train model
        optimizer = Adam(vae.parameters(), lr=LEARNING_RATE) # TODO: make this an argument, and make it possible to choose between Adam and SGD
        """trainer = Trainer(model=vae, 
                            epochs=N_EPOCH,
                            learning_rate=LEARNING_RATE,
                            batch_size=BATCH_SIZE,
                            dataloader_train=dataloader_train,
                            dataloader_val=dataloader_val,
                            optimizer=optimizer)"""
        
        trainer = Trainer(model=vae, 
                            epochs=N_EPOCH,
                            learning_rate=LEARNING_RATE,
                            batch_size=BATCH_SIZE,
                            dataloader_train=dataloader_train,
                            dataloader_val=dataloader_val,
                            optimizer=optimizer,
                            beta=BETA)
        
        trainer.train()

        # Save model
        if args.save_model:
            print(f'Saving model to path: models/~/{model_name_}.json...\n')
            vae.encoder.save(path=f'models/encoders/{model_name_}.json')
            vae.save(path=f'models/vaes/{model_name_}.json')

        # Plotting
        if 'reconstructions' in args.plot:
            plot_reconstructions(model=trainer.model, 
                                dataloader=dataloader_test, 
                                model_name_=model_name_, 
                                device=device, 
                                num_examples=args.num_examples, 
                                save=True, 
                                loss_func=trainer.loss_function)
        
        if 'latent_distributions' in args.plot:
            model_name_latdist = f'{model_name_}_beta_{BETA}_latent_distributions'
            plot_latent_distributions(model=vae, 
                                      dataloader=dataloader_test, 
                                      model_name=model_name_latdist,
                                      device=device,
                                      num_examples=args.num_examples, 
                                      save=True)
        
        tester = Tester(model=vae, dataloader_test=dataloader_test, trainer=trainer)
        #print(f'Test loss: {tester.test()}')
        
        if any(mode in args.plot for mode in ['loss', 'separated_losses']): # Plotting modes that possibly need trainings across multiple seeds
            print("Staring multiple seeds training og plotting loss...")
            # CREATE MULTIPLE MODELS WITH DIFFERENT RANDOM SEEDS, TRAIN AND PLOT TOTAL, BCE AND KL LOSSES
            total_train_losses = np.zeros((NUM_SEEDS, N_EPOCH))
            total_val_losses = np.zeros((NUM_SEEDS, N_EPOCH))
            bce_train_losses = np.zeros((NUM_SEEDS, N_EPOCH))
            bce_val_losses = np.zeros((NUM_SEEDS, N_EPOCH))
            kl_train_losses = np.zeros((NUM_SEEDS, N_EPOCH))
            kl_val_losses = np.zeros((NUM_SEEDS, N_EPOCH))
            
            test_losses = np.zeros(NUM_SEEDS)
            for i in range(NUM_SEEDS):
                print(f'Random seed {i+1}/{NUM_SEEDS}')
                # Generate different initialization of training and validation data for the new trainer
                _, _, _, dataloader_train, dataloader_val, dataloader_test  = load_LiDARDataset(datapath,  
                                                                                            mode='max', 
                                                                                            batch_size=BATCH_SIZE, 
                                                                                            train_test_split=0.7,
                                                                                            train_val_split=0.3,
                                                                                            shuffle=True,
                                                                                            extend_dataset_roll=True,
                                                                                            roll_degrees=rolling_degrees,
                                                                                            add_noise_to_train=True)
                # Create Variational Autoencoder(s)
                if args.model_name == 'ShallowConvVAE':
                    encoder = Encoder_conv_shallow(latent_dims=LATENT_DIMS, eps_weight=EPS_WEIGHT)
                    decoder = Decoder_circular_conv_shallow2(latent_dims=LATENT_DIMS)
                if args.model_name == 'DeepConvVAE':
                    encoder = Encoder_conv_deep(latent_dims=LATENT_DIMS, eps_weight=EPS_WEIGHT)
                    decoder = Decoder_circular_conv_deep(latent_dims=LATENT_DIMS)

                vae = VAE(encoder=encoder, decoder=decoder, latent_dims=LATENT_DIMS).to(device)
                optimizer = Adam(vae.parameters(), lr=LEARNING_RATE)
                
                trainer = Trainer(model=vae, 
                            epochs=N_EPOCH,
                            learning_rate=LEARNING_RATE,
                            batch_size=BATCH_SIZE,
                            dataloader_train=dataloader_train,
                            dataloader_val=dataloader_val,
                            optimizer=optimizer,
                            beta=BETA)
                trainer.train()

                total_train_losses[i,:] = trainer.training_loss['Total loss']
                total_val_losses[i,:] = trainer.validation_loss['Total loss']
                bce_train_losses[i,:] = trainer.training_loss['Reconstruction loss']
                bce_val_losses[i,:] = trainer.validation_loss['Reconstruction loss']
                kl_train_losses[i,:] = trainer.training_loss['KL divergence loss']
                kl_val_losses[i,:] = trainer.validation_loss['KL divergence loss']
                
                # Get test error anyways
                tester = Tester(model=vae, dataloader_test=dataloader_test, trainer=trainer)
                test_loss = tester.test()
                print(f'Test loss seed {i}: {test_loss}')
                test_losses[i] = test_loss
                
                del trainer, vae, encoder, decoder, optimizer 

            total_losses = [total_train_losses, total_val_losses]
            bce_losses = [bce_train_losses, bce_val_losses]
            kl_losses = [kl_train_losses, kl_val_losses]
            labels = ['Training loss', 'Validation loss']

            if 'separated_losses' in args.plot:
                plot_separated_losses(total_losses, bce_losses, kl_losses, labels, model_name=model_name_, save=True)   

            if 'loss' in args.plot:
                if args.num_seeds == 1:
                    plot_loss(training_loss=total_train_losses[0], validation_loss=total_val_losses[0], save=True, model_name=model_name_)
                else:
                    plot_loss_multiple_seeds(loss_trajectories=total_losses, labels=labels, model_name=model_name_, save=True)
            
            if 'test_loss_report' in args.plot:
                metadata = f'Number of seeds: {NUM_SEEDS}, epochs: {N_EPOCH}, batch size: {BATCH_SIZE}, learning rate: {LEARNING_RATE}'
                tester.report_test_stats(test_losses=test_losses, model_name=model_name_, metadata=metadata)

            
        if 'latent_dims_sweep' in args.plot: 
            print("Staring latent dimension size sweep...")
            model_name_ = f'{name}_latent_dims_sweep_2'
            latent_dims_grid = [1, 2, 6, 12, 24]
            total_val_losses_for_latent_dims = [] # Fill up with val losses for each latent dim
            bce_val_losses_for_latent_dims = [] # Fill up with val losses for each latent dim
            kl_val_losses_for_latent_dims = [] # Fill up with val losses for each latent dim
            for l in latent_dims_grid:
                print(f'Latent dimension: {l}')
                total_val_losses = np.zeros((NUM_SEEDS, N_EPOCH))
                bce_val_losses = np.zeros((NUM_SEEDS, N_EPOCH))
                kl_val_losses = np.zeros((NUM_SEEDS, N_EPOCH))

                test_losses = np.zeros(NUM_SEEDS)

                for i in range(NUM_SEEDS):
                    print(f'Random seed {i+1}/{NUM_SEEDS}')
                    # Generate different initialization of training and validation data for the new trainer
                    _, _, _, dataloader_train, dataloader_val, dataloader_test  = load_LiDARDataset(datapath,  
                                                                                                mode='max', 
                                                                                                batch_size=BATCH_SIZE, 
                                                                                                train_test_split=0.7,
                                                                                                train_val_split=0.3,
                                                                                                shuffle=True,
                                                                                                extend_dataset_roll=True,
                                                                                                roll_degrees=rolling_degrees,
                                                                                                add_noise_to_train=True)
                    # Create Variational Autoencoder(s)
                    if args.model_name == 'ShallowConvVAE':
                        encoder = Encoder_conv_shallow(latent_dims=LATENT_DIMS, eps_weight=EPS_WEIGHT)
                        decoder = Decoder_circular_conv_shallow2(latent_dims=LATENT_DIMS)
                    if args.model_name == 'DeepConvVAE':
                        encoder = Encoder_conv_deep(latent_dims=LATENT_DIMS, eps_weight=EPS_WEIGHT)
                        decoder = Decoder_circular_conv_deep(latent_dims=LATENT_DIMS)

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

                    total_val_losses[i,:] = trainer.validation_loss['Total loss']
                    bce_val_losses[i,:] = trainer.validation_loss['Reconstruction loss']
                    kl_val_losses[i,:] = trainer.validation_loss['KL divergence loss']

                    # Get test error 
                    tester = Tester(model=vae, dataloader_test=dataloader_test, trainer=trainer)
                    test_loss = tester.test()
                    print(f'Test loss seed {i}: {test_loss}')
                    test_losses[i] = test_loss

                    del trainer, vae, encoder, decoder, optimizer  

                many_traj_labels = [f'Seed {i+1}' for i in range(NUM_SEEDS)]
                plot_many_loss_traj(loss_trajectories=total_val_losses, labels=many_traj_labels, model_name=f'{model_name_}_latent_dim_{l}', save=True)

                total_val_losses_for_latent_dims.append(total_val_losses)
                bce_val_losses_for_latent_dims.append(bce_val_losses)
                kl_val_losses_for_latent_dims.append(kl_val_losses)

                if 'test_loss_report' in args.plot:
                    metadata = f'Latent dim: {l}, number of seeds: {NUM_SEEDS}, epochs: {N_EPOCH}, batch size: {BATCH_SIZE}, learning rate: {LEARNING_RATE}'
                    tester.report_test_stats(test_losses=test_losses, model_name=model_name_, metadata=metadata)

            labels = [f'Latent dim = {l}' for l in latent_dims_grid]
            plot_loss_multiple_seeds(loss_trajectories=total_val_losses_for_latent_dims, labels=labels, model_name=model_name_, save=True) # vurdere å slitte opp?
            plot_separated_losses(total_val_losses_for_latent_dims, bce_val_losses_for_latent_dims, kl_val_losses_for_latent_dims, labels, model_name=model_name_, save=True)  
        




    
    if args.mode == 'test':
        """vae_path = f'models/vaes/{args.model_name_load}.json'
        encoder_path = f'models/encoders/{args.model_name_load}.json'
        vae = VAE().load(path=vae_path) 
        encoder = vae.encoder"""

        # Generate test error for all the models in vae folder for each latent dim, averaged over NUM_SEEDS
        latent_dims_grid = [1, 2, 4, 8, 12]
        seeds = list(range(1, NUM_SEEDS+1))
        test_losses = np.zeros((len(latent_dims_grid), NUM_SEEDS))
        for l in latent_dims_grid:
            for seed in seeds:
                name = "ShallowConvVAE"
                model_name_ = f'{name}_latent_dims_sweep'
                path = f'models/vaes/{model_name_}_latent_dim_{l}_seed_{seed}.json'
                encoder = Encoder_conv_shallow(latent_dims=l)
                decoder = Decoder_circular_conv_shallow2(latent_dims=l)
                vae = VAE_integrated_shallow(latent_dims=l).load(path=path)
                # Obtain test loss for this latent dim and seed combination
                for x_batch in dataloader_test:
                    x_batch = x_batch.to(device)
                    x_hat, mu, log_var = vae(x_batch)
                    loss, _, _ = trainer.loss_function(x_hat, x_batch, mu, log_var, beta=1)
                    test_losses[l, seed-1] += loss.item()
                test_losses[l, seed-1] /= len(dataloader_test.dataset)
        plot_test_error_as_function_of(test_errors=test_losses, variable=latent_dims_grid, variable_name='Latent dimension', save=True, model_name=model_name_)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',
                        help= 'Progam mode',
                        choices=['train', 'test']
    )
    parser.add_argument('--model_name',
                        help= 'Name of model to train',
                        type=str,
                        choices=['ShallowConvVAE', 'DeepConvVAE'],
                        default='ShallowConvVAE'
    )
    parser.add_argument('--model_name_load',
                        help= 'Name of model to be used for testing',
                        type=str,
                        default='ShallowConvVAE'
    )
    parser.add_argument('--plot',
                        help= 'Plotting mode',
                        type=str,
                        choices=['reconstructions', 'loss', 'separated_losses', 'latent_dims_sweep', 'latent_distributions', 'test_loss_report'],
                        nargs='+'
    )
    parser.add_argument('--save_model',
                        help= 'Save model',
                        type=bool,
                        default=False
    )
    parser.add_argument('--num_seeds',
                        help= 'Number of seeds to train and plot',
                        type=int,
                        default=1
    )
    parser.add_argument('--beta',
                    help= 'beta for beta-VAE. Default 1: vanilla VAE',
                    type=float,
                    default=1
    )
    parser.add_argument('--num_examples',
                        help= 'Number of reconstruction examples to plot',
                        type=int,
                        default=10
    )
    parser.add_argument('--batch_size', 
                        help= 'Batch size for training', 
                        type=int, 
                        default=16
    )
    parser.add_argument('--epochs',
                        help= 'Number of epochs for training', 
                        type=int, 
                        default=15
    )
    parser.add_argument('--datapath', 
                        type=str, 
                        default='data/LiDAR_MovingObstaclesNoRules.csv'
    )
    parser.add_argument('--learning_rate',
                        help= 'Learning rate for training', 
                        type=float, 
                        default=0.001
    )
    parser.add_argument('--latent_dims',
                        help= 'Number of latent dimensions', 
                        type=int, 
                        default=12
    )


    args = parser.parse_args()
    try:
        main(args)
    except KeyboardInterrupt:
        print('\n\nKeyboard interrupt detected, exiting.')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
    finally:
        print('Done.')