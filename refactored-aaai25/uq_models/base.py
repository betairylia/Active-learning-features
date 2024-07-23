import os

import torch
from lightning import LightningModule, Trainer
from torch import nn
from torch.nn import functional as F
from torchmetrics import Accuracy, AUROC, AveragePrecision

import torchvision
from torchvision import datasets
from torchvision.transforms import transforms
import torchvision.transforms as transforms

from functools import partial

import lightning as pl

from argparse import ArgumentParser
import utils
import nets

# from lightning.loggers import WandbLogger

from nets import net_dict

import math
# from resnet import resnet18
# from weight_drop import *

import copy
# import ot

def BestAccuracySweep(num_sweeps = 256):
    
    def foo(scores, labels):

        min_score = scores.min()
        max_score = scores.max()
        
        if torch.is_tensor(min_score):
            min_score = min_score.item()
        if torch.is_tensor(max_score):
            max_score = max_score.item()

        best_acc = 0
        best_threshold = 0
        for threshold in torch.linspace(min_score, max_score, num_sweeps):
            acc = ((scores >= threshold) == labels).float().mean()
            if acc > best_acc:
                best_acc = acc
                best_threshold = threshold
        
        return best_acc, best_threshold
    
    return foo

class RuntimeDataCache:
    pass

class SimpleModel(LightningModule):

    '''
    dm_header => { input_dim: [X,Y,Z,...]; output_dim: [X,Y,Z,...] }
    Seperated as individual modules:
     - accuracy
     - UQ visualization (and in general, all visualization)
    '''
    def __init__(self, args, dm_header, loss, metrics):

        super().__init__()

        self.args = args
        self.dm_header = dm_header

        # Construct networks
        self.net_factory = net_dict[args.net]()

        # Placeholders
        # TODO: Change `nets` so we get new interface as follows:
        # self.net = self.net_factory.getNets(
        #     args,
        #     dm_header,
        # )

        self.net = nn.Sequential(
            *self.net_factory.getNets(
                dm_header['input_dim'],
                dm_header['output_dim'],
                hidden_dim = args.hidden_dim,
                dropout_rate = args.dropout_rate
            )
        )

        self.net_init = copy.deepcopy(self.net)
        self.try_load_ckpt(args.load_ckpt, args.load_ckpt_init)

        # Loss related
        self.loss = loss

        # Metrics
        self.metrics = metrics

        # Not sure how to let lightning record model outputs correctly
        # So we use this workaround instead.
        # This is a literally empty object that will hold any data provided.
        self.cache = RuntimeDataCache()

        # UQ Metrics
        self.uncertainty_auroc = AveragePrecision(task="binary")
        self.uncertainty_acc = BestAccuracySweep()

        self.val_uncertainty_scores = []
        self.val_uncertainty_labels = []

        utils.log("Model initialized")
        utils.log(self)
    
    def filter_rename_ckpt(self, state_dict, prefix):
        filtered_dict = {}
        for k in state_dict:
            if k.startswith("%s." % prefix):
                filtered_dict[k.replace("%s." % prefix, "", 1)] = state_dict[k]
        return filtered_dict

    def try_load_ckpt(self, path, path_init):
        
        if path == None:
            return

        loaded = torch.load(path)["state_dict"]
        self.net.load_state_dict(self.filter_rename_ckpt(loaded, "net"))

        if path_init == None:
            self.net_init.load_state_dict(self.filter_rename_ckpt(loaded, "net_init"))
            return

        loaded = torch.load(path_init)["state_dict"]
        self.net_init.load_state_dict(self.filter_rename_ckpt(loaded, "net"))

        return
    
    def forward(self, x):

        logits = self.net(x)
        pred_prob = F.softmax(logits, dim = 1)
        entropy = -torch.sum(pred_prob * torch.log(pred_prob + 1e-8), dim = 1)

        # shape of logits: [batch_size, num_classes]
        # shape of entropy: [batch_size]

        # Returns: outputs, uncertainty score
        return logits, entropy
    
    # TODO: Scale_output for lazy regime training?
    #    -> maybe go to nets implementation?

    def training_step(self, batch, batch_idx):

        i, x, y, o = batch

        logits = self(x)[0]

        loss = self.loss(logits, y)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):

        i, x, y, o = batch

        # Some info that we pass to forward
        self.val_index = i
        self.val_full_batch = batch

        logits, uncertainty = self(x)
        self.metric_eval = self.metrics(logits, y)

        # Disable temporal info
        self.val_index = None
        self.val_full_batch = None

        # Compute val loss 
        loss = self.loss(logits, y)

        # Log metrics
        self.log("val_loss", loss, prog_bar=False)
        self.log("val_%s" % self.metrics.logger_name, self.metric_eval, prog_bar=False)

        # Store uncertainty scores
        self.val_uncertainty_scores.append(uncertainty)
        self.val_uncertainty_labels.append(o)

        # Capture current model outputs so on_val_batch_end callbacks can work properly
        self.cache.logits = logits
        self.cache.uncertainty = uncertainty
    
    def on_validation_epoch_end(self):

        # Collect all results for UQ
        self.val_uncertainty_scores = torch.cat(self.val_uncertainty_scores, dim = 0)
        self.val_uncertainty_labels = torch.cat(self.val_uncertainty_labels, dim = 0)
        
        # Evaluation + Logging
        best_uncertain_acc, best_uncertain_th = self.uncertainty_acc(self.val_uncertainty_scores, self.val_uncertainty_labels.squeeze())
        self.log("val_uncertain_acc", best_uncertain_acc, prog_bar=True)

        self.uncertainty_auroc(self.val_uncertainty_scores, self.val_uncertainty_labels.squeeze())
        self.log("val_auroc", self.uncertainty_auroc, prog_bar=False)

        mean_uncertainty_seen = self.val_uncertainty_scores[self.val_uncertainty_labels.squeeze() == 0].mean()
        mean_uncertainty_unseen = self.val_uncertainty_scores[self.val_uncertainty_labels.squeeze() == 1].mean()
        self.log("val_uncertain_seen", mean_uncertainty_seen, prog_bar=False)
        self.log("val_uncertain_unseen", mean_uncertainty_unseen, prog_bar=False)

        # Reset them to empty lists so we can use them again
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

        if self.args.optim == 'sgd':
            return torch.optim.SGD(self.parameters(), lr = 1e-3, momentum = 0.9, weight_decay = 1e-5)
        elif self.args.optim == 'adamw':
            return torch.optim.AdamW(self.parameters(), lr = 3e-4, weight_decay = 1e-5)
        else:
            print("UNSUPPORTED OPTIM TYPE!")

    # Helper functions, maybe not used anymore?

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
