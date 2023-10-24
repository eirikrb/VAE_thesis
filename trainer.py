import torch
import torch.nn as nn
import torch.nn.functional as F

class Trainer():
    """Trainer class for training VAE models"""
    def __init__(self,
                 model:nn.Module,
                 epochs:int,
                 learning_rate:float,
                 batch_size:int,
                 dataloader_train:torch.utils.data.DataLoader,
                 dataloader_val:torch.utils.data.DataLoader,
                 optimizer:torch.optim.Optimizer,
                 ) -> None:
    
        self.model = model
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.optimizer = optimizer
        
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.training_loss = []
        self.validation_loss = []

    def loss_function(self, x_hat, x, mu, log_var):
        """Calculates loss function for VAE"""
        BCE_loss = F.binary_cross_entropy(x_hat, x, reduction='sum') # Reconstruction loss
        KLD_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) # KL divergence loss
        return BCE_loss + KLD_loss
    
    def train_epoch(self):
        """Trains model for one epoch"""
        self.model.train()
        train_loss = 0.0
        for x_batch in self.dataloader_train:
            x_batch = x_batch.to(self.device)
            x_hat, mu, log_var = self.model(x_batch)
            loss = self.loss_function(x_hat, x_batch, mu, log_var)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
        return train_loss/len(self.dataloader_train.dataset) #sjekk
    
    def validation(self):
        """Calculates validation loss"""
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x in self.dataloader_val:
                x = x.to(self.device)
                x_hat, mu, log_var = self.model(x)
                loss = self.loss_function(x_hat, x, mu, log_var)
                val_loss += loss.item()
        return val_loss/len(self.dataloader_val.dataset) # sjekk


    def train(self):
        """Main training loop. Trains model for self.epochs epochs"""
        for e in range(self.epochs):
            train_loss = self.train_epoch()
            val_loss = self.validation()
            self.training_loss.append(train_loss)
            self.validation_loss.append(val_loss)
            print(f'\nEpoch: {e+1}/{self.epochs} \t|\t Train loss: {train_loss:.3f} \t|\t Val loss: {val_loss:.3f}')