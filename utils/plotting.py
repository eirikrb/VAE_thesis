import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib
import pandas as pd
import os


def do_predictions(model:nn.Module, dataloader:torch.utils.data.DataLoader):
    model.eval()
    with torch.no_grad():
        for x_batch, _ in dataloader:
            x_hat = model(x_batch) # only one batch in the test set
        
        x_hat  = x_hat.detach().numpy()
        
    model.train()
    return x_hat

def predict(model:nn.Module, dataloader:torch.utils.data.DataLoader, index:int):
    sample = dataloader.dataset.X[index]
    return model(sample)


def plot_loss(training_loss, validation_loss, model_name, save=False):
    
    epochs = np.arange(len(training_loss))

    plt.figure(figsize=(15,15))
    plt.plot(epochs, training_loss, label='Training_loss')
    plt.plot(epochs, validation_loss, label='Validation loss')
    plt.legend()
    if save:
        plt.savefig(f'plots/LOSS_{model_name}.png', bbox_inches='tight')


def plot_reconstruction(index, X, X_hat, model_name, save=False):
    angles = np.linspace(0, 2*np.pi, 180) - np.pi / 2
    X = np.ones_like(X) - X # we need x = x/150 so we invert the normalization from load_LiDARDataset() since 1-x = x/150
    X_hat = np.ones_like(X_hat) - X_hat

    # Create a circular plot to represent a single top-down view of the range data
    fig = plt.figure()
    ax = fig.add_subplot(projection='polar')
    ax.scatter(angles, X, s=0.1, color='b')
    ax.scatter(angles, X_hat, s=0.1, color='r')
    ax.legend(['True', 'Reconstructed'], loc='upper right')
    if save:
        plt.savefig(f'plots/RECONSTRUCTION_{model_name}_{index}.png', bbox_inches='tight')
    else:
        plt.show()

def plot_reconstructions(model:nn.Module, dataloader:torch.utils.data.DataLoader, model_name_:str, device:str, num_examples:int=7, save=True):
    batches_to_sample_from = [i for i in range(num_examples)]
    for batch_idx, x_batch in enumerate(dataloader):
        x_batch = x_batch.to(device)
        x_hat, _, _ = model(x_batch)
        if batch_idx in batches_to_sample_from:
            x = x_batch.cpu().detach().numpy()[0,0,:]
            x_hat = x_hat.cpu().detach().numpy()[0,0,:]
            print(x.shape)
            print(x_hat.shape)
            plot_reconstruction(index=batch_idx, X=x, X_hat=x_hat, model_name=model_name_, save=True)



