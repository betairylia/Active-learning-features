import os

import torch
from pytorch_lightning import LightningModule, Trainer
from torch import nn
from torch.nn import functional as F
from torchmetrics import Accuracy, AUROC, AveragePrecision

import torchvision
from torchvision import datasets
from torchvision.transforms import transforms
import torchvision.transforms as transforms

from functools import partial

from pl_bolts.datamodules import MNISTDataModule, CIFAR10DataModule, FashionMNISTDataModule
from datamodules import SVHNDataModule

import pytorch_lightning as pl

from argparse import ArgumentParser
from utils import *

from pytorch_lightning.loggers import WandbLogger

from data_uncertainty import MNIST_UncertaintyDM, CIFAR10_UncertaintyDM, SVHN_UncertaintyDM, ImageNet_Validation_UncertaintyDM #, FashionMNIST_UncertaintyDM
from recorder import Recorder

from nets import net_dict

import math
# from resnet import resnet18
# from weight_drop import *

import copy
import ot

class SimpleModel(LightningModule):
    def __init__(self, args, input_shape, output_dim = 10):
        super().__init__()

        self.args = args
        
        # Construct networks
        self.hidden_dim = 2048
        self.output_dim = output_dim
        self.visualized = False

        self.net_factory = net_dict[args.net]()
        self.net, self.head = self.net_factory.getNets(
            input_shape, 
            [output_dim],
            hidden_dim = args.hidden_dim,
            dropout_rate = args.dropout_rate
        )
        
        if args.loss == 'mse':
            if args.binary:
                self.loss = lambda x, y: F.mse_loss(x, y.detach().float())
            else:
                self.loss = lambda x, y: F.mse_loss(x, F.one_hot(y, output_dim).detach().float())
        elif args.loss == 'cent':
            if args.binary:
                self.loss = lambda x, y: F.binary_cross_entropy_with_logits(x, y.detach().float())
            else:
                self.loss = nn.CrossEntropyLoss()

        # https://pytorchlightning.github.io/lightning-tutorials/notebooks/lightning_examples/cifar10-baseline.html
        # resnet = torchvision.models.resnet18(pretrained = False, num_classes = 10)
        # resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        # resnet.maxpool = nn.Identity()
        # self.l1 = resnet
        
        self.accuracy = lambda p, y: (p == y).float().mean()
        self.accuracy_seen = lambda p, y, o: ((torch.logical_and(p == y, o == 0)).float().sum() / (o == 0).float().sum()) if ((o == 0).float().sum() > 0) else None

        # TODO: Check which one fits the task better
        # self.uncertainty_auroc = AUROC(task="binary")
        self.uncertainty_auroc = AveragePrecision(task="binary")
        self.uncertainty_acc = BestAccuracySweep()

        self.val_uncertainty_scores = []
        self.val_uncertainty_labels = []

        # Store the initialized network
        self.net_init = copy.deepcopy(self.net)
        self.head_init = copy.deepcopy(self.head)

        print("Model initialized")
        print(self)
        
    def forward(self, x):

        self.disable_dropout(self.net)
        self.disable_dropout(self.head)

        logits = self.head(self.net(x))
        pred_prob = F.softmax(logits, dim = 1)
        entropy = -torch.sum(pred_prob * torch.log(pred_prob + 1e-8), dim = 1)

        # shape of logits: [batch_size, num_classes]
        # shape of entropy: [batch_size]

        return logits, entropy

    def scale_output(self, raw_logits, x):
        
        # init_logits = self(x)
        # return (raw_logits - init_logits) * 
        return raw_logits * self.args.lazy_scaling

    def training_step(self, batch, batch_nb):
        i, x, y, o = batch

        logits = self.scale_output(self(x)[0], x)

        # Handle binary classification
        if self.args.binary:
            logits = logits.squeeze()

        loss = self.loss(logits, y)

        self.log("train_loss", loss)

        return loss
    
    def validation_step(self, batch, batch_idx):
        
        i, x, y, o = batch
        # logits = self(x)
        # loss = F.cross_entropy(logits, y)

        self.val_index = i
        self.val_full_batch = batch

        logits, uncertainty = self(x)
        logits = self.scale_output(logits, x)

        # Handle logit and predictions (for binary / multi-class classification)
        if self.args.binary:
            logits = logits.squeeze()
            preds = (logits > 0.5).long()
        else:
            preds = torch.argmax(logits, dim=1)

        self.val_index = None
        self.val_full_batch = None

        y[y >= self.output_dim] = -100
        loss = self.loss(logits, y)
        acc = self.accuracy(preds, y)
        acc_seen = self.accuracy_seen(preds, y, o)

        if not self.visualized:
            self.visualized = True
            self.logger.log_image(
                key = "test-set",
                images = [ImageMosaicSQ(x)],
                caption = ["".join(["1" if _o == 1 else "0" for _o in o.detach().cpu()])]
            )

        self.val_uncertainty_scores.append(uncertainty)
        self.val_uncertainty_labels.append(o)

        logit_norm_seen = torch.norm(logits[o.squeeze() == 0], dim = -1).mean()
        logit_norm_unseen = torch.norm(logits[o.squeeze() == 1], dim = -1).mean()

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=False)
        self.log("val_acc", acc, prog_bar=False)
        
        self.log("val_logit_norm_seen", logit_norm_seen, prog_bar=False)
        self.log("val_logit_norm_unseen", logit_norm_unseen, prog_bar=False)

        if acc_seen is not None:
            self.log("val_acc_seen", acc_seen, prog_bar=True)
        
        return loss

    def on_validation_epoch_end(self):
        self.val_uncertainty_scores = torch.cat(self.val_uncertainty_scores, dim = 0)
        self.val_uncertainty_labels = torch.cat(self.val_uncertainty_labels, dim = 0)
        
        best_uncertain_acc, best_uncertain_th = self.uncertainty_acc(self.val_uncertainty_scores, self.val_uncertainty_labels.squeeze())
        self.log("val_uncertain_acc", best_uncertain_acc, prog_bar=True)

        self.uncertainty_auroc(self.val_uncertainty_scores, self.val_uncertainty_labels.squeeze())
        self.log("val_auroc", self.uncertainty_auroc, prog_bar=False)

        mean_uncertainty_seen = self.val_uncertainty_scores[self.val_uncertainty_labels.squeeze() == 0].mean()
        mean_uncertainty_unseen = self.val_uncertainty_scores[self.val_uncertainty_labels.squeeze() == 1].mean()
        self.log("val_uncertain_seen", mean_uncertainty_seen, prog_bar=False)
        self.log("val_uncertain_unseen", mean_uncertainty_unseen, prog_bar=False)

        # Parameter statistics
        current_params = {k:v for k, v in self.net.named_parameters()}
        current_params.update({k:v for k, v in self.head.named_parameters()})

        init_params = {k:v for k, v in self.net_init.named_parameters()}
        init_params.update({k:v for k, v in self.head_init.named_parameters()})

        param_diff_avg = 0
        param_norm_avg = 0
        param_init_norm_avg = 0

        for k in current_params.keys():

            param_norm = current_params[k].abs().mean()
            param_init_norm = init_params[k].abs().mean()
            param_diff = (current_params[k] - init_params[k]).abs().mean() / param_init_norm

            param_diff_avg += param_diff
            param_norm_avg += param_norm
            param_init_norm_avg += param_init_norm

        param_diff_avg /= len(current_params)
        param_norm_avg /= len(current_params)
        param_init_norm_avg /= len(current_params)

        self.log("param_diff_avg", param_diff_avg, prog_bar=False)
        self.log("param_norm_avg", param_norm_avg, prog_bar=False)
        self.log("param_init_norm_avg", param_init_norm_avg, prog_bar=False)

        self.val_uncertainty_scores = []
        self.val_uncertainty_labels = []

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        # Here we just reuse the on_validation_epoch_end for testing
        return self.on_validation_epoch_end()

    def on_test_epoch_start(self):
        # Here we just reuse the on_validation_epoch_start for testing
        return self.on_validation_epoch_start()
        
    def configure_optimizers(self):
        # return torch.optim.AdamW(self.parameters())

        if self.args.optim == 'sgd':
            return torch.optim.SGD(self.parameters(), lr = 1e-3, momentum = 0.9, weight_decay = 1e-5)
        elif self.args.optim == 'adamw':
            return torch.optim.AdamW(self.parameters(), lr = 3e-4, weight_decay = 1e-5)
        else:
            print("UNSUPPORTED OPTIM TYPE!")

    def enable_dropout(self, m):
        for each_module in m.modules():
            if each_module.__class__.__name__.startswith('Dropout'):
    #             print("Enabling dropout")
                each_module.train()
                
    def disable_dropout(self, m):
        for each_module in m.modules():
            if each_module.__class__.__name__.startswith('Dropout'):
    #             print("Disabling dropout")
                each_module.eval()
                
    def record(self, m):
        for each_module in m.modules():
            if each_module.__class__.__name__.startswith('Recorder'):
                each_module.switchToRecord()
                
    def replay(self, m):
        for each_module in m.modules():
            if each_module.__class__.__name__.startswith('Recorder'):
                each_module.switchToReplay()
                
    def recorder_identity(self, m):
        for each_module in m.modules():
            if each_module.__class__.__name__.startswith('Recorder'):
                each_module.switchToIdentity()