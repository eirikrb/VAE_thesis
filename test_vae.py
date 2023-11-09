"""MIDLERTIDIG TESTFIL FOR VAE"""
# Currently testing tsn and umap for latent space visualization

import torch
import torch.nn as nn
import os
import numpy as np
from utils.dataloader import load_LiDARDataset
from vae.VAE import VAE
from vae.encoders import Encoder_conv_shallow, Encoder_conv_deep
from vae.decoders import Decoder_circular_conv_shallow2, Decoder_circular_conv_deep
from trainer import Trainer, TrainerInfoVAE
from torch.optim import Adam
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats


path_x =  'data/LiDAR_MovingObstaclesNoRules.csv'


LEARNING_RATE = 0.001
N_EPOCH = 40
BATCH_SIZE = 16  
LATENT_DIMS = 12
beta = 1


data_train, data_val, data_test, dataloader_train, dataloader_val, dataloader_test  = load_LiDARDataset(path_x,
                                                                                                        mode='max', 
                                                                                                        batch_size=BATCH_SIZE, 
                                                                                                        train_test_split=0.7,
                                                                                                        train_val_split=0.3,
                                                                                                        shuffle=True)

encoder = Encoder_conv_deep(latent_dims=LATENT_DIMS)
decoder = Decoder_circular_conv_deep(latent_dims=LATENT_DIMS)
vae = VAE(encoder, decoder, latent_dims=LATENT_DIMS, beta=beta)

optimizer = Adam(vae.parameters(), lr=LEARNING_RATE) # TODO: make this an argument, and make it possible to choose between Adam and SGD
trainer = TrainerInfoVAE(model=vae, 
                    epochs=N_EPOCH,
                    learning_rate=LEARNING_RATE,
                    batch_size=BATCH_SIZE,
                    dataloader_train=dataloader_train,
                    dataloader_val=dataloader_val,
                    optimizer=optimizer,
                    beta=beta)

trainer.train()

"""
df = pd.DataFrame(latent_representations)

plt.style.use('ggplot')
#plt.style.use('seaborn-poster')
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('axes', labelsize=12)

angles = np.linspace(0, 2*np.pi, 180) - np.pi / 2
X = np.ones_like(x) - x     
# Create a circular test plot to represent a single top-down view of the range data
fig = plt.figure(figsize=(17,5))
#fig, (ax1,ax2) = plt.subplots(1,2, figsize=(10,5))
ax1 = fig.add_subplot(121,projection='polar')
ax1.scatter(angles, X, s=1.2)
ax2 = fig.add_subplot(122)
#plt.savefig('plots/test_plot.pdf', bbox_inches='tight')

#fig2 = plt.figure()
x = np.linspace(min(mu) - 3*sigma[np.argmin(mu)], max(mu) + 3*sigma[np.argmax(mu)], 100)
for i in range(len(mu)):
    if i > 6:
        lin = '--'
    else:
        lin = '-'
    ax2.plot(x, stats.norm.pdf(x, mu[i], sigma[i]), label=f'z{i+1}', linestyle=lin, linewidth=1.5)
ax2.legend()

ax2.set_ylabel('Probability density')
#kde_plot = df.plot.kde()
#fig = kde_plot.get_figure()
plt.savefig('plots/latent_space.pdf', bbox_inches='tight')
"""


"""

#tsne = TSNE(n_components=2, verbose=1)
#tsne_results = tsne.fit_transform(latent_representations)
#print(tsne_results.shape)

df = pd.DataFrame()
df['labels'] = labels
df['comp-1'] = tsne_results[:,0]
df['comp-2'] = tsne_results[:,1]


plot = sns.scatterplot(
"""