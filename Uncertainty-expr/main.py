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

from data_uncertainty import MNIST_UncertaintyDM, CIFAR10_UncertaintyDM #, FashionMNIST_UncertaintyDM, SVHN_UncertaintyDM
from recorder import Recorder

from nets import net_dict

import math
# from resnet import resnet18
# from weight_drop import *

#################################################

# class DataModule_(ImageClassificationData):
#     @property
#     def num_classes(self):
#         return 10

input_size_dict = {
    'mnist': [1, 32, 32], # Resized
    'cifar10': [3, 32, 32],
    'cifar100': [3, 32, 32],
    'svhn': [3, 32, 32],
    'fashionmnist': [1, 32, 32], # Resized
    'imagenet': [3, 224, 224],
    'tinyimagenet': [3, 64, 64],
    'stl10': [3, 96, 96],
    'lsun': [3, 256, 256],
    'celeba': [3, 64, 64],
    'cub200': [3, 224, 224],
}

def get_data_module(dataset_name, batch_size, data_augmentation=True, num_workers=16, data_dir='./data', do_partial_train = True, do_contamination = True):
    
    if dataset_name == 'mnist':
        main_dm = MNIST_UncertaintyDM(data_dir = data_dir, batch_size = batch_size, num_workers = num_workers, do_partial_train = do_partial_train, do_contamination = do_contamination)
    
    elif dataset_name == 'cifar10':
        main_dm = CIFAR10_UncertaintyDM(data_dir = data_dir, batch_size = batch_size, num_workers = num_workers, do_partial_train = do_partial_train, do_contamination = do_contamination)
    
    return main_dm, input_size_dict[dataset_name]

def BestAccuracySweep(num_sweeps = 256):
    def foo(scores, labels):
        min_score = scores.min()
        max_score = scores.max()
        best_acc = 0
        best_threshold = 0
        for threshold in torch.linspace(min_score, max_score, num_sweeps):
            acc = ((scores >= threshold) == labels).float().mean()
            if acc > best_acc:
                best_acc = acc
                best_threshold = threshold
        return best_acc, best_threshold
    return foo

#################################################################

class SimpleModel(LightningModule):
    def __init__(self, args, input_shape, output_dim = 10):
        super().__init__()

        self.args = args
        
        # Construct networks
        self.hidden_dim = 2048
        self.output_dim = output_dim

        self.net_factory = net_dict[args.net]()
        self.net, self.head = self.net_factory.getNets(
            input_shape, 
            [output_dim],
            hidden_dim = args.hidden_dim,
            dropout_rate = args.dropout_rate
        )
        
        if args.loss == 'mse':
            self.loss = lambda x, y: F.mse_loss(x, F.one_hot(y, 10).detach().float())
        elif args.loss == 'cent':
            self.loss = nn.CrossEntropyLoss()

        # https://pytorchlightning.github.io/lightning-tutorials/notebooks/lightning_examples/cifar10-baseline.html
        # resnet = torchvision.models.resnet18(pretrained = False, num_classes = 10)
        # resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        # resnet.maxpool = nn.Identity()
        # self.l1 = resnet
        
        self.accuracy = lambda p, y: (p == y).float().mean()

        # TODO: Check which one fits the task better
        # self.uncertainty_auroc = AUROC(task="binary")
        self.uncertainty_auroc = AveragePrecision(task="binary")
        self.uncertainty_acc = BestAccuracySweep()

        self.val_uncertainty_scores = []
        self.val_uncertainty_labels = []

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

    def training_step(self, batch, batch_nb):
        x, y, o = batch
        # loss = F.cross_entropy(self(x), y)
        loss = self.loss(self(x)[0], y)

        self.log("train_loss", loss)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y, o = batch
        logits = self(x)
        # loss = F.cross_entropy(logits, y)
        logits, uncertainty = self(x)

        y[y >= self.output_dim] = -100
        loss = self.loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)

        self.val_uncertainty_scores.append(uncertainty)
        self.val_uncertainty_labels.append(o)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=False)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        self.val_uncertainty_scores = torch.cat(self.val_uncertainty_scores, dim = 0)
        self.val_uncertainty_labels = torch.cat(self.val_uncertainty_labels, dim = 0)
        
        best_uncertain_acc, best_uncertain_th = self.uncertainty_acc(self.  val_uncertainty_scores, self.val_uncertainty_labels.squeeze())
        self.log("val_uncertain_acc", best_uncertain_acc, prog_bar=True)

        self.uncertainty_auroc(self.val_uncertainty_scores, self.val_uncertainty_labels.squeeze())
        self.log("val_auroc", self.uncertainty_auroc, prog_bar=False)

        self.val_uncertainty_scores = []
        self.val_uncertainty_labels = []

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        # Here we just reuse the on_validation_epoch_end for testing
        return self.on_validation_epoch_end()
        
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters())
#         return torch.optim.AdamW(self.parameters(), lr = 3e-4, weight_decay = 1e-5)

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

class MCDropoutModel(SimpleModel):
    
    def forward(self, x):
    
        self.enable_dropout(self.net)
        self.enable_dropout(self.head)

        # self.disable_dropout(self.net)
        # self.disable_dropout(self.head)

        if self.training:
            logits = self.head(self.net(x))
            return logits, None
        else:
            logits = []
            probs = []
            for i in range(self.args.dropout_iters):
                logits.append(self.head(self.net(x)))
                pred_prob = F.softmax(logits[-1], dim = 1)
                probs.append(pred_prob)
            
            logits = torch.stack(logits, dim = 0)
            probs = torch.stack(probs, dim = 0)
            
            # Compute the MI between prediction and parameters
            # probs_marginal = torch.mean(probs, dim = 0)
            # entropy_marginal = -torch.sum(probs_marginal * torch.log(probs_marginal + 1e-8), dim = 1)
            # entropy_conditional = -torch.sum(probs * torch.log(probs + 1e-8), dim = 2).mean(dim = 0)
            # bald_objective = entropy_marginal - entropy_conditional

            uncertainty = torch.std(logits, dim = 0).mean(dim = 1)

            return logits.mean(dim = 0), uncertainty

class TestTimeOnly_ApproximateDropoutModel(SimpleModel):

    dropout_iters: 10

    def forward(self, x):

        if self.training:
            self.disable_dropout(self.net)
            self.disable_dropout(self.head)
            logits = self.head(self.net(x))
            return logits, None
        else:

            # Record
            self.disable_dropout(self.net)
            self.disable_dropout(self.head)

            self.record(self.net)
            self.record(self.head)

            _logits = self.head(self.net(x))

            # Replay

            self.enable_dropout(self.net)
            self.enable_dropout(self.head)

            self.replay(self.net)
            self.replay(self.head)

            probs = []
            logits = []
            for i in range(self.args.dropout_iters):
                logits.append(self.head(self.net(x)))
                pred_prob = F.softmax(logits[-1], dim = 1)
                probs.append(pred_prob)

            logits = torch.stack(logits, dim = 0)
            probs = torch.stack(probs, dim = 0)
            
            # Compute the MI between prediction and parameters
            # probs_marginal = torch.mean(probs, dim = 0)
            # entropy_marginal = -torch.sum(probs_marginal * torch.log(probs_marginal + 1e-8), dim = 1)
            # entropy_conditional = -torch.sum(probs * torch.log(probs + 1e-8), dim = 2).mean(dim = 0)
            # bald_objective = entropy_marginal - entropy_conditional
            uncertainty = torch.std(logits, dim = 0).mean(dim = 1)

            self.recorder_identity(self.net)
            self.recorder_identity(self.head)

            return logits.mean(dim = 0), uncertainty

models_dict =\
{
    "default": SimpleModel,
    "mcdropout": MCDropoutModel,
    "tt_approx": TestTimeOnly_ApproximateDropoutModel,
}

#################################################################
        
def main(hparams):
    
    seed = hparams.seed * (hparams.runid + 1)
    pl.seed_everything(seed)
    
    main_datamodule, input_dim = get_data_module(hparams.dataset, hparams.batch_size, data_augmentation = (hparams.dataAug > 0), num_workers=hparams.num_workers, do_partial_train = hparams.do_partial_train, do_contamination = hparams.do_contamination)
    main_datamodule.setup()

    # Init our model
    model = models_dict[hparams.model](hparams, input_dim, main_datamodule.n_classes)

    wbgroup = GetArgsStr(hparams)
    if wbgroup is None:
        wandb_logger = WandbLogger(
            project="TT-uncertainty-estimation",
            config = vars(hparams),
        )
    else:
        wandb_logger = WandbLogger(
            project="TT-uncertainty-estimation",
            config = vars(hparams),
            group = wbgroup
        )

    # Initialize a trainer
    trainer = Trainer(
        gpus=1,
        max_epochs=hparams.epochs,
        progress_bar_refresh_rate=20,
        enable_checkpointing=False,

        # limit_val_batches = 0.0,

        logger = wandb_logger
    )

    # Train the model ⚡
    trainer.fit(
        model, 
        datamodule = main_datamodule
    )

if __name__ == "__main__":
    
    # TODO: Use Lightning integration
    # TODO: NTK - https://pytorch.org/functorch/stable/notebooks/neural_tangent_kernels.html
    parent_parser = ArgumentParser()
    
    parent_parser = Trainer.add_argparse_args(parent_parser)
    
    parser = parent_parser.add_argument_group("Model Hyper-parameters")
    
    parser = parent_parser.add_argument_group("Active Learning related")
    parser.add_argument("--heuristic", type=str, default="random")
    parser.add_argument("--loss", type=str, default="cent") # cent, mse
    parser.add_argument("--epochs_per_query", type=int, default=25)
    parser.add_argument("--inference_iteration", type=int, default=20)
    parser.add_argument("--model", type=str, default='default')
    parser.add_argument("--net", type=str, default='mlp')
    parser.add_argument("--dataset", type=str, default='mnist')
    parser.add_argument("--do_partial_train", type=int, default=1)
    parser.add_argument("--do_contamination", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=16)
    
    # Model
    parser.add_argument("--hidden_dim", type=int, default=2048)
    parser.add_argument("--dropout_rate", type=float, default=0.85)
    parser.add_argument("--dropout_iters", type=int, default=10)

    # Training
    parser.add_argument('--dataAug', type=int, default=0, help="Data augmentation on(1) / off(0).")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for training.")
    parser.add_argument('--epochs', type=int, default=50, help="Number of epochs for training.")
    parser.add_argument('--augTrials', type=int, default=4, help="Trials per augmentation (ignored if augmentation disabled)")
    
    parser = parent_parser.add_argument_group("Run metadata / WandB sweeps")
    parser.add_argument('--keyargs', type=str, default="", help='Key variables in HP tune, splitted in commas')
    parser.add_argument('--aaarunid', type=int, default=0, help='Run ID.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed. will be multiplied with runid+1 to ensure different RNGs for different runs.')

    args = parent_parser.parse_args()

    args.dataset = args.dataset.lower()
    
    # Process arguments a bit
    args.runid = args.aaarunid
    del(args.aaarunid)
    
    if args.dataAug == 0:
        args.augTrials = 1

    print("Args: %s" % vars(args))

    main(args)
    
