from flash.image.classification.integrations.baal import (
    ActiveLearningDataModule,
)
from ndropALLoop import ndrop_ActiveLearningLoop
from flash.image import ImageClassifier, ImageClassificationData
from flash.core.classification import LogitsOutput

from baal.bayesian.dropout import _patch_dropout_layers
from baal.active.dataset import ActiveLearningDataset
from AdvancedHeuristics import get_heuristic_with_advanced

import os

import torch
from pytorch_lightning import LightningModule, Trainer
from torch import nn
from torch.nn import functional as F
from torchmetrics import Accuracy

import torchvision
from torchvision import datasets
from torchvision.transforms import transforms

from functools import partial

from plBaaLData import ActiveLearningDataModuleWrapper
from pl_bolts.datamodules import MNISTDataModule, CIFAR10DataModule, FashionMNISTDataModule

import pytorch_lightning as pl

from argparse import ArgumentParser
from utils import *

from pytorch_lightning.loggers import WandbLogger

#################################################

IMG_SIZE = 28

class DataModule_(ImageClassificationData):
    @property
    def num_classes(self):
        return 10

def get_data_module(heuristic, data_path):    
    # active_dm = ActiveLearningDataModuleWrapper(FashionMNISTDataModule)(
    active_dm = ActiveLearningDataModuleWrapper(MNISTDataModule)(
        data_dir = "./data",
        num_workers = 4,

        heuristic=get_heuristic_with_advanced(heuristic),
        initial_num_labels=32,
        query_size=16,
        val_split=0.01
    )
    return active_dm

#################################################################

class SimpleModel(LightningModule):
    def __init__(self, inference_iteration: int):
        super().__init__()
        
        # Trivial linear
        self.net = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(28 * 28 * 1, 8192),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
        )
        
        self.head = torch.nn.Sequential(
            torch.nn.Linear(8192, 10)
        )
        
        # https://pytorchlightning.github.io/lightning-tutorials/notebooks/lightning_examples/cifar10-baseline.html
        # resnet = torchvision.models.resnet18(pretrained = False, num_classes = 10)
        # resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        # resnet.maxpool = nn.Identity()
        # self.l1 = resnet
        
        self.accuracy = Accuracy()
        
        # TODO: Wrap and hide followings
        changed = _patch_dropout_layers(self)
        if not changed and inference_iteration > 1:
            print("The model does not contain dropout layer, inference_iteration has been set to 1.")
            inference_iteration = 1
        self.inference_iteration = inference_iteration

    def forward(self, x):
        # return torch.relu(self.l1(x.view(x.size(0), -1)))
        return self.head(self.net(x))
        # return self.l1(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.accuracy(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=False)
        self.log("val_acc", self.accuracy, prog_bar=False)
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)
    
    # def predict_step(self, batch, batch_idx):
    #     # Here we just reuse the validation_step for predicting
    #     x, y = batch
    #     return self(x)
    
    def query_step(self, batch, batch_idx):
        x, y = batch
        feat = self.net(x)
        logits = self.head(feat)
        return logits, feat
    
    # TODO: Wrap and hide this
    def predict_step(self, batch, batch_idx):
        
        # net = None
        
        with torch.no_grad():
            
            out = []
            fin = []
            
            for _ in range(self.inference_iteration):
                (logits, features) = self.query_step(batch, batch_idx)
                out.append(logits)
                fin.append(features)

        # BaaL expects a shape [num_samples, num_classes, num_iterations]
        return (
            torch.stack(out).permute((1, 2, 0)), # [N_sample, dim, N_iter]
            torch.stack(fin).permute((1, 2, 0)) # [N_sample, dim, N_iter]
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters())

#################################################################
        
def main(hparams):
    
    seed = hparams.seed * (hparams.runid + 1)
    pl.seed_everything(seed)
    
    active_dm = get_data_module(hparams.heuristic, './data')

    # Init our model
    model = SimpleModel(inference_iteration = hparams.inference_iteration)
    # model = get_model(active_dm)

    aloop = ndrop_ActiveLearningLoop(
        label_epoch_frequency = hparams.epochs_per_query,
        inference_iteration = hparams.inference_iteration
    )

    wbgroup = GetArgsStr(hparams)
    if wbgroup is None:
        wandb_logger = WandbLogger(
            project="AL-features",
            config = vars(hparams)
        )
    else:
        wandb_logger = WandbLogger(
            project="AL-features",
            config = vars(hparams),
            group = wbgroup
        )

    # Initialize a trainer
    trainer = Trainer(
        gpus=1,
        max_epochs=62, # 1024 labels in total
        progress_bar_refresh_rate=20,
        enable_checkpointing=False,

        limit_val_batches = 0.0,

        logger = wandb_logger
    )

    aloop.connect(trainer.fit_loop)
    trainer.fit_loop = aloop

    # Train the model ⚡
    trainer.fit(
        model, 
        datamodule = active_dm
    )

if __name__ == "__main__":
    
    # TODO: Use Lightning integration
    parent_parser = ArgumentParser()
    
    parent_parser = Trainer.add_argparse_args(parent_parser)
    
    parser = parent_parser.add_argument_group("Model Hyper-parameters")
    
    parser = parent_parser.add_argument_group("Active Learning related")
    parser.add_argument("--heuristic", type=str, default="random")
    parser.add_argument("--epochs_per_query", type=int, default=25)
    parser.add_argument("--inference_iteration", type=int, default=20)
    
    parser = parent_parser.add_argument_group("Run metadata / WandB sweeps")
    parser.add_argument('--keyargs', type=str, default="", help='Key variables in HP tune, splitted in commas')
    parser.add_argument('--aaarunid', type=int, default=0, help='Run ID.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed. will be multiplied with runid+1 to ensure different RNGs for different runs.')
    
    args = parent_parser.parse_args()
    args.runid = args.aaarunid
    del(args.aaarunid)

    main(args)
    
