# from flash.image.classification.integrations.baal import (
#     ActiveLearningDataModule,
# )
from ndropALLoop import ndrop_ActiveLearningLoop
# from flash.image import ImageClassifier, ImageClassificationData
# from flash.core.classification import LogitsOutput

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

from opacus.grad_sample import GradSampleModule

from resnet import resnet18

#################################################

IMG_SIZE = 28

# class DataModule_(ImageClassificationData):
#     @property
#     def num_classes(self):
#         return 10

def get_data_module(heuristic, data_path, budget=16, initial=32):    
    active_dm = ActiveLearningDataModuleWrapper(FashionMNISTDataModule)(
#     active_dm = ActiveLearningDataModuleWrapper(MNISTDataModule)(
        data_dir = "./data",
        num_workers = 0,
        
        batch_size = 96,

        heuristic=get_heuristic_with_advanced(heuristic),
        initial_num_labels=initial,
        query_size=budget,
        val_split=0.01
    )
    
    return active_dm

#################################################################

class SimpleModel(LightningModule):
    def __init__(self, args, heuristic = None, inference_iteration: int = 1, variance_val_loader_getter = None, perEpoch = False, perEpisode = True):
        super().__init__()
        
        # Trivial linear
        self.net, self.net_no_dropout, self.head, self.key_layers = self.getNets()
        
        if args.loss == 'mse':
            self.loss = lambda x, y: F.mse_loss(x, F.one_hot(y, 10).detach().float())
        elif args.loss == 'cent':
            self.loss = nn.CrossEntropyLoss()

        # https://pytorchlightning.github.io/lightning-tutorials/notebooks/lightning_examples/cifar10-baseline.html
        # resnet = torchvision.models.resnet18(pretrained = False, num_classes = 10)
        # resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        # resnet.maxpool = nn.Identity()
        # self.l1 = resnet
        
        self.accuracy = Accuracy()
        self.heuristic = heuristic

        # TODO: Wrap and hide followings
        changed = _patch_dropout_layers(self)
        if not changed and inference_iteration > 1:
            print("The model does not contain dropout layer, inference_iteration has been set to 1.")
            inference_iteration = 1
        self.inference_iteration = inference_iteration
        
        self.varval_loader_cache = True
        self.varval_loader_cached = None
        
        self.varval_loader_getter = variance_val_loader_getter
        self.perEpoch = perEpoch
        self.perEpisode = perEpisode
        
        self.hidden_dim = 1024

    def getNets(self):
        
#         net = torch.nn.Sequential(
#             torch.nn.Flatten(),
#             torch.nn.Linear(28 * 28 * 1, self.hidden_dim),
#             torch.nn.ReLU(),
#             torch.nn.Dropout(),
#         )
        
#         net_no_dropout = torch.nn.Sequential(
#             net[0],
#             net[1],
#             net[2]
#         )
        
        net = resnet18(
            num_classes = self.hidden_dim,
            zero_init_residual = True,
            conv1_type = "cifar",
            no_maxpool = True
        )
        
        net_no_dropout = net
        
        head = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim, 10)
        )
        
        key_layers = [net.fc, head[0]]
        
        return net, net_no_dropout, head, key_layers
        
    def forward(self, x):
        # return torch.relu(self.l1(x.view(x.size(0), -1)))
        return self.head(self.net(x))
        # return self.l1(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        # loss = F.cross_entropy(self(x), y)
        loss = self.loss(self(x), y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        # loss = F.cross_entropy(logits, y)
        loss = self.loss(self(x), y)
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
        
        if self.heuristic is not None and hasattr(self.heuristic, "custom_prediction_step"):
            return self.heuristic.custom_prediction_step(self, batch, batch_idx)

        else:
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

    def checkStddev(self):
        
        if self.varval_loader_getter is not None:
            with torch.no_grad():
                
                if self.varval_loader_cached is not None:
                    varval = self.varval_loader_getter()
                else:
                    varval = self.varval_loader_getter()
                    if self.varval_loader_cache:
                        self.varval_loader_cached = varval
                
                N = len(varval.dataset)
                
                # Dropout
                std_dropout = 0
                for batch in varval:
                    batch = [x.to(self.device) for x in batch]
                    out = []
                    for _ in range(self.inference_iteration):
                        (logits, features) = self.query_step(batch, 0)
                        # logits: [Nb, C]
                        out.append(logits)
                    out = torch.stack(out, dim = 0) # out: [I, Nb, C]
                    std_dropout += torch.std(out, dim = 0).sum()
#                     logits_dropout.append(out)
                    
#                 logits_dropout = torch.cat(logits_dropout, dim = 1) # [I, Nx, C]
                
                # Non-dropout
                std_nondropout = 0
                width = 8192 * 0.5
                for batch in varval:
                    out = []
                    batch = [x.to(self.device) for x in batch]
                    x, _ = batch
                    feat = self.net_no_dropout(x)

                    # Activation
                    assert len(self.head) == 1
                    feat_aligned = feat.unsqueeze(-1) # [bs, hidden_dim, 1]
                    last_layer = self.head[-1]
                    activation = feat_aligned * last_layer.weight.permute(1, 0).unsqueeze(0) + last_layer.bias[None, None, :] # [bs, hidden_dim, out_dim]
                    
                    # activation: [Nb, I, C]
                    std = torch.sqrt(((activation - activation.mean(dim = 1, keepdims = True)) ** 2) / width).sum()
                    std_nondropout += std
                    
#                 self.log("Test set stddev norm", {"Dropout": std_dropout / N, "Monte-Carlo": std_nondropout / N})
        return std_dropout / N, std_nondropout / N
        
    def EvalStddevEpisode(self):
    
        if not self.perEpisode:
            return None, None
        
        training = self.training
        self.eval()
        
        stddrop, stdndrop = self.checkStddev()
#         self.log("stddev-Episode", {"Dropout": stddrop, "Monte-Carlo": stdndrop})

        if training:
            self.train()
            
        return stddrop, stdndrop
        
    def on_train_epoch_end(self):
        
        if not self.perEpoch:
            return
        
        self.eval()
        
        stddrop, stdndrop = self.checkStddev()
        self.log("stddev-Epoch", {"Dropout": stddrop, "Monte-Carlo": stdndrop})
                
        self.train()
        
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters())

#################################################################
        
def main(hparams):
    
    seed = hparams.seed * (hparams.runid + 1)
    pl.seed_everything(seed)
    
    active_dm = get_data_module(hparams.heuristic, './data', hparams.budget, hparams.initial)
    heuristic = active_dm.heuristic

    # Init our model
    model = SimpleModel(
        args,
        heuristic = heuristic,
        inference_iteration = hparams.inference_iteration,
        variance_val_loader_getter = lambda: active_dm.test_dataloader()
    )
    # model = get_model(active_dm)

    if hasattr(heuristic, "register_model"):
        heuristic.register_model(model)

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
        max_epochs=hparams.rounds, # 1024 labels in total
        progress_bar_refresh_rate=20,
        enable_checkpointing=False,


        # limit_val_batches = 0.0,

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
    parser.add_argument("--loss", type=str, default="cent") # cent, mse
    parser.add_argument("--epochs_per_query", type=int, default=25)
    parser.add_argument("--inference_iteration", type=int, default=20)
    
    parser.add_argument('--budget', type=int, default=16, help="Budget per query")
    parser.add_argument('--initial', type=int, default=32, help="Initial labels")
    parser.add_argument('--total_queries', type=int, default=1024, help="Total queries after training finished")
    
    parser = parent_parser.add_argument_group("Run metadata / WandB sweeps")
    parser.add_argument('--keyargs', type=str, default="", help='Key variables in HP tune, splitted in commas')
    parser.add_argument('--aaarunid', type=int, default=0, help='Run ID.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed. will be multiplied with runid+1 to ensure different RNGs for different runs.')
    
    args = parent_parser.parse_args()
    args.runid = args.aaarunid
    del(args.aaarunid)

    # Compute rounds
    args.rounds = (args.total_queries) // args.budget
    print("Actual #total labels at end: %d" % (args.initial + args.budget * args.rounds))

    print("Args: %s" % vars(args))

    main(args)
    
