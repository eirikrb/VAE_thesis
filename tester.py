import torch
import torch.nn as nn
import numpy as np
from trainer import Trainer

class Tester(): 
    """Tester class for testing VAE models"""
    def __init__(self,
                 model:nn.Module,
                 dataloader_test:torch.utils.data.DataLoader,
                 trainer:Trainer
                 ) -> None:
        self.model = model
        self.dataloader_test = dataloader_test
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.trainer = trainer # to get access to loss function
    
    def test(self) -> float:
        test_error = 0.0
        for x_batch in self.dataloader_test:
            x_batch = x_batch.to(self.device)
            x_hat, mu, log_var = self.model(x_batch)
            loss, _, _ = self.trainer.loss_function(x_hat, x_batch, mu, log_var, beta=self.trainer.beta, N=1, M=1)#, N=len(self.trainer.dataloader_train.dataset), M=len(self.trainer.dataloader_train.dataset)/self.trainer.batch_size)    
            test_error += loss.item()
        avg_test_error = test_error/len(self.dataloader_test.dataset)
        return avg_test_error
    
    def report_test_stats(self, test_losses:np.ndarray, model_name:str, metadata:str) -> None:
        mean = np.mean(test_losses)
        std = np.std(test_losses)
        ci = 1.96 * std / np.sqrt(len(test_losses))

        with open(f'tests/{model_name}_test_losses.txt', 'w') as f:
            f.write(f'Metadata: {metadata}\n')
            f.write(f'Mean: {mean}\n')
            f.write(f'Std: {std}\n')
            f.write(f'Confidence interval: {ci}\n')
            f.write(f'All test losses: {test_losses}\n')
        print(f'Mean: {mean}')
        print(f'Std: {std}')
        print(f'Confidence interval: {ci}')
        print(f'All test losses: {test_losses}')
    