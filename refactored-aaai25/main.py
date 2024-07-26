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

# from pl_bolts.datamodules import MNISTDataModule, CIFAR10DataModule, FashionMNISTDataModule
# from dataloaders import SVHNDataModule

import lightning as pl

from argparse import ArgumentParser
from utils import *

from lightning.pytorch.loggers import WandbLogger

import dataloaders as dl
import lossmetrics

from nets import net_dict

import math
# from resnet import resnet18
# from weight_drop import *

import copy

import uq_models as uq
import extensions as exs
import random

from lightning.pytorch.callbacks import ModelCheckpoint

###################################################################################
# UTILS

def has_func(obj, func_name):
    return hasattr(obj, func_name) and callable(getattr(obj, func_name))

###################################################################################
# CHECKPOINTING UTILS

class InitialCheckpointsCallback(ModelCheckpoint):

    def on_train_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)
        self.save_checkpoint(trainer)

    def on_validation_end(self, trainer, pl_module):
        super().on_validation_end(trainer, pl_module)
        if(trainer.current_epoch + 1) < 10:
            self.save_checkpoint(trainer)

###################################################################################
# MAIN and ARGS

def main(hparams):

    wbgroup = GetArgsStr(hparams)
    if wbgroup is None:
        wandb_logger = WandbLogger(
            project="TTUQ-AAAI25",
            config = vars(hparams),
        )
    else:
        wandb_logger = WandbLogger(
            project="TTUQ-AAAI25",
            config = vars(hparams),
            group = wbgroup
        )

    seed = hparams.seed * (hparams.runid + 1)
    pl.seed_everything(seed)
    
    # TODO: Refactor here
    main_datamodule, dm_header = dl.get_data_module(
        data_augmentation = (hparams.dataAug > 0),
        **vars(hparams))
    main_datamodule.setup()

    loss, metrics = lossmetrics.get_loss_and_metrics(hparams, dm_header)

    # Prepare optional calibration dataset for reference
    ref_data = None
    if hparams.use_reference_dataset:
        if hparams.contaminate_ref:
            ref_data = torch.utils.data.Subset(
                main_datamodule.test_dataset,
                np.random.choice(len(main_datamodule.test_dataset), size = (hparams.reference_data_count,))
            )
        else:
            ref_data = torch.utils.data.Subset(
                main_datamodule.ref_dataset,
                np.random.choice(len(main_datamodule.ref_dataset), size = (hparams.reference_data_count,))
            )

    # Init our model
    model = uq.models_dict[hparams.model](hparams, dm_header, loss, metrics, ref_data)

    # Checkpointing
    checkpoint_callback = InitialCheckpointsCallback(
        dirpath  = "checkpoints/%s" % hparams.ckpt_path,
        filename = "{epoch:04d}-{val_acc_seen:.2f}",
        every_n_epochs = hparams.ckpt_interval,
        save_top_k = -1
    )

    all_callbacks = []
    if hparams.ckpt:
        all_callbacks.append(checkpoint_callback)

    all_callbacks += exs.get_callbacks(hparams)

    # TODO: Visualization callbacks

    # TODO: NTK callbacks
    
    # Initialize a trainer
    trainer = Trainer(
        # hparams,
        devices=1,
        max_epochs=hparams.epochs,
        enable_checkpointing=hparams.ckpt,

        # limit_val_batches = 0.0,

        logger = wandb_logger,
        callbacks = all_callbacks
    )

    if not hparams.no_train:
        # Train the model ⚡
        trainer.fit(
            model, 
            datamodule = main_datamodule
        )
    
    # Test the model 🔍
    trainer.test(
        model,
        datamodule = main_datamodule
    )

if __name__ == "__main__":
    
    # TODO: Use Lightning integration
    # TODO: NTK - https://pytorch.org/functorch/stable/notebooks/neural_tangent_kernels.html
    parent_parser = ArgumentParser()
    
    # parent_parser = Trainer.add_argparse_args(parent_parser)
    
    parser = parent_parser.add_argument_group("Model Hyper-parameters")
    
    parser = parent_parser.add_argument_group("Active Learning related")
    parser.add_argument("--heuristic", type=str, default="random")
    parser.add_argument("--loss", type=str, default="cent") # cent, mse
    parser.add_argument("--epochs_per_query", type=int, default=25)
    parser.add_argument("--inference_iteration", type=int, default=20)
    parser.add_argument("--model", type=str, default='default')
    parser.add_argument("--net", type=str, default='mlp')
    parser.add_argument("--dataset", type=str, default='mnist')
    parser.add_argument("--do_partial_train", type=int, default=0)
    parser.add_argument("--use_full_trainset", type=int, default=1)
    parser.add_argument("--use_reference_dataset", type=int, default=0)
    parser.add_argument("--do_contamination", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--test_set_max", type=int, default=-1)

    parser.add_argument("--binary", type=int, default=0, help="Convert the problem to a binary classification. Splits the dataset into halves.")
    parser.add_argument("--ckpt", type=int, default=0, help="Save checkpoints")
    parser.add_argument("--ckpt_path", type=str, default="Untitled", help="Checkpoint save path")
    parser.add_argument("--ckpt_interval", type=int, default=5, help="Checkpoint save interval (epochs)")

    parser.add_argument("--load_ckpt", type=str, default=None, help="Checkpoint to load")
    parser.add_argument("--load_ckpt_init", type=str, default=None, help="Checkpoint to load (as initialization t = t_s)")
    
    # Model
    parser.add_argument("--hidden_dim", type=int, default=2048)
    parser.add_argument("--dropout_rate", type=float, default=0.0)
    parser.add_argument("--dropout_iters", type=int, default=10)
    parser.add_argument("--lazy_scaling", type=float, default=1)
    parser.add_argument("--pointwise_linearization", type=int, default=1)

    parser.add_argument("--reference_data_count", type=int, default=64)
    parser.add_argument("--random_multidim", type=int, default=1)
    parser.add_argument("--num_multidim", type=int, default=32)

    parser.add_argument("--noise_pattern", type=str, default='prop', help = "prop | indep | inv")
    parser.add_argument("--perturb_power", type=float, default=-1, help = "Overrides perturb_min / max if set to value above 0")
    parser.add_argument("--perturb_min", type=float, default=0.1, help = "Perturb noise norm for 1st layer")
    parser.add_argument("--perturb_max", type=float, default=0.1, help = "Perturb noise norm for last layer")
    parser.add_argument("--perturb_ex", type=float, default=0.1, help = "Perturb noise norm for subtract multiplicative")
    parser.add_argument("--perturb_nonlinear", type=float, default=0.0, help = "Perturb noise norm curve nonlinearity; >0 => more change towards last layer | <0 => more change towards first layer")

    parser.add_argument("--add_temp", type=float, default=0.0, help = "Additive temperature for Indep-Det-Posterior")
    parser.add_argument("--mul_temp", type=float, default=0.1, help = "Multiplicative temperature for Indep-Det-Posterior")
    parser.add_argument("--indepdet_mode", type=str, default='posterior', help = "pure-fluctuation | posterior")

    # Visualization
    # TODO: FIXME:  Visalization is wrong. The r.v. is a dropout mask, then we need to plot (grad for parameter #j, hessian for parameter #j)
    #               for this particular mask, which is a loop thru entire dataset per mask.
    # parser.add_argument("--independence_check_layers", nargs="+", type=str)
    # parser.add_argument("--independence_check_dataid", nargs="+", type=int)

    # Training
    parser.add_argument('--no_train', type=int, default=0, help="Don't train the model.")
    parser.add_argument('--dataAug', type=int, default=0, help="Data augmentation on(1) / off(0).")
    parser.add_argument('--contaminate_ref', type=int, default=0, help="Use comtaminated data in reference dataset.")
    parser.add_argument('--noise', type=float, default=0.3, help="Noise std for outliers.")
    parser.add_argument('--blur', type=float, default=2.0, help="Gaussian blur sigma for outliers. ImageNet-C: [1, 2, 3, 4, 6]")
    parser.add_argument('--optim', type=str, default='sgd', help="Optimizer type: ['sgd', 'adamw'].")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for training.")
    parser.add_argument('--epochs', type=int, default=50, help="Number of epochs for training.")
    parser.add_argument('--augTrials', type=int, default=4, help="Trials per augmentation (ignored if augmentation disabled)")

    parser = parent_parser.add_argument_group("Extensions")
    parser.add_argument('--dataset-vis', type=int, default=0, help="Visualize dataset from the first batch")
    parser.add_argument('--seen-class-acc', type=int, default=1, help="Compute accuracy for seen (IN) classes")
    parser.add_argument('--exact-ntk-inf', type=int, default=0, help="Compute exact NTKs for uncertainty infimum upperbounds, with reference training set")
    parser.add_argument('--ntk-batchsize', type=int, default=4, help="batchsize for NTK computation")
    
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

    if args.perturb_power > 0:
        args.perturb_min = args.perturb_power
        args.perturb_max = args.perturb_power

    # if args.no_train:
        # args.batch_size = 1

    print("Args: %s" % vars(args))

    main(args)
