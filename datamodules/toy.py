# Extend the LightningDataModule class

import torch
from pytorch_lightning import LightningModule, LightningDataModule, Trainer

import numpy as np

width = 2
margin = 1

def get_Y(x, noise = 0):
    noise_vec = np.random.normal(scale = noise, size = x.shape)
    return np.sin(3 * x) + noise_vec

def get_train_X(N = 1024):
    return np.concatenate([np.random.rand(N,) * width - (margin + width),
                           np.random.rand(N,) * width +  margin,
                           # np.array([0])
           ])

def get_val_X(N = 1024):
    return np.linspace(-5, 5, num = N)

class RegressionToyDatasetDataModule(LightningDataModule):
    
    def __init__(self):
        super().__init__()

    def setup(self, stage = None):

        tx = get_train_X()
        vx = get_val_X()

        self.Xs_train = torch.Tensor(tx)
        self.Ys_train = torch.Tensor(get_Y(tx, 0.03))

        self.Xs_val = torch.Tensor(vx)
        self.Ys_val = torch.Tensor(get_Y(vx))

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(self.Xs_train[:, None, None], self.Ys_train),
                batch_size = 128,
                shuffle = True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(self.Xs_val[:, None, None], self.Ys_val),
                batch_size = 128)

    def test_dataloader(self):
        return self.val_dataloader()

