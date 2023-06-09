import os

import torch
from pytorch_lightning import LightningModule, Trainer
from torch import nn
from torch.nn import functional as F
from torchmetrics import Accuracy

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

from functorch import make_functional, vmap, vjp, jvp, jacrev

import math
# from resnet import resnet18
# from weight_drop import *

import numpy as np

#################################################

# class DataModule_(ImageClassificationData):
#     @property
#     def num_classes(self):
#         return 10

input_size_dict = {
    'mnist': [1, 28, 28],
    'cifar10': [3, 32, 32],
    'cifar100': [3, 32, 32],
    'svhn': [3, 32, 32],
    'fashionmnist': [1, 28, 28],
    'imagenet': [3, 224, 224],
    'tinyimagenet': [3, 64, 64],
    'stl10': [3, 96, 96],
    'lsun': [3, 256, 256],
    'celeba': [3, 64, 64],
    'cub200': [3, 224, 224],
}

def get_data_module(dataset_name, batch_size, data_augmentation=True):    
    # main_dm = SVHNDataModule(
#     main_dm = CIFAR10DataModule(
#     main_dm = FashionMNISTDataModule(

    if dataset_name == 'mnist':
        main_dm = MNISTDataModule(
            data_dir = "./data",
            num_workers = 16,
            batch_size = batch_size,
        )
    
    elif dataset_name == 'cifar10':
        main_dm = CIFAR10DataModule(
            data_dir = "./data",
            num_workers = 16,
            batch_size = batch_size,
        )
    
    # CIFAR-10 / 32x32x3 images
    if data_augmentation == True:
        
        main_dm.train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(32, (0.4, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
            )
        ])
        
        main_dm.val_transforms = transforms.Compose([
            transforms.Resize(40),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
            )
        ])
        
        main_dm.test_transforms = transforms.Compose([
            transforms.Resize(40),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
            )
        ])
    
    return main_dm, input_size_dict[dataset_name]

#################################################################

class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, y):
        return F.mse_loss(x, F.one_hot(y, 10).detach().float())

class SimpleModel(LightningModule):
    def __init__(self, args, input_shape):
        super().__init__()

        self.args = args
        
        # Construct networks
        self.hidden_dim = args.hidden_dim
        self.net, self.head = self.getNets(input_shape)

        self.evaluated_NTKs = []
        
        if args.loss == 'mse':
            self.loss = MSELoss()
        elif args.loss == 'cent':
            self.loss = nn.CrossEntropyLoss()

        # https://pytorchlightning.github.io/lightning-tutorials/notebooks/lightning_examples/cifar10-baseline.html
        # resnet = torchvision.models.resnet18(pretrained = False, num_classes = 10)
        # resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        # resnet.maxpool = nn.Identity()
        # self.l1 = resnet
        
        self.accuracy = Accuracy()
        
    def initNets(self, net):

        def weights_init_wrapper(scale = 1.0):
            def weights_init(m):
                if isinstance(m, nn.Linear):
                    torch.nn.init.normal_(m.weight, mean = 0.0, std = (1 / math.sqrt(m.weight.shape[0])) * scale)
                    torch.nn.init.constant_(m.bias, 0)
            return weights_init

        net.apply(weights_init_wrapper(scale = self.args.initialization_scale))

    def getNets(self, input_shape):
        
        # Compute the input size from input_shape
        flatten_size = 1
        for dim in input_shape:
            flatten_size *= dim

        def getblock(d_in, d_out):
            return [
                torch.nn.Linear(d_in, d_out),
                # torch.nn.ReLU(),
                torch.nn.Tanh(),
                # torch.nn.Dropout(),
            ]

        net = torch.nn.Sequential(
            torch.nn.Flatten(),
            *getblock(flatten_size, self.hidden_dim),
            *getblock(self.hidden_dim, self.hidden_dim),
            *getblock(self.hidden_dim, self.hidden_dim),
        )
        
        head = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim, 10)
        )

        # self.initNets(net)
        # self.initNets(head)
        
        return net, head
        
    def forward(self, x):
        return self.head(self.net(x))

    def training_step(self, batch, batch_nb):
        x, y = batch
        # loss = F.cross_entropy(self(x), y)
        loss = self.loss(self(x), y)

        self.log("train_loss", loss)

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
        
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters())
#         return torch.optim.AdamW(self.parameters(), lr = 3e-4, weight_decay = 1e-5)

# Create a model which is an ensemble of smaller networks (summate their outputs)
class EnsembleNets(nn.Module):
    def __init__(self, nets, sums = True, split_input = False, reduce_method = 'sum'):
        super().__init__()
        self.nets = nn.ParameterList(nets)
        self.sums = sums
        self.split_input = split_input

        self.reduce = reduce_method
        if self.reduce == 'sum':
            self.reduce = torch.sum
        elif self.reduce == 'mean':
            self.reduce = torch.mean
        else:
            print("Ensemble Nets ERROR: Unsupported reduce method %s" % reduce_method)

    def forward(self, x, override_return_individuals = False):

        if self.split_input:
            # shape of x: (num_nets, batch_size, input_size)
            outputs = [net(x_i) for net, x_i in zip(self.nets, x)]
        else:
            # shape of x: (batch_size, input_size)
            outputs = [net(x) for net in self.nets]
        
        if self.sums and not override_return_individuals:
            return self.reduce(torch.stack(outputs), dim = 0)
        else:
            return torch.stack(outputs, dim = 0)

# TODO: Initialization scaling
class EnsembleModel(SimpleModel):

    def __init__(self, args, input_shape):
        super().__init__(args, input_shape)

        self.hidden_dim = self.hidden_dim * self.args.ensemble_expansion // self.args.ensemble_size
        self.net, self.head = self.getNets_ensemble(input_shape)
        print(self)

    def getNets_ensemble(self, input_shape):
        nets_and_heads = [self.getNets(input_shape) for _ in range(self.args.ensemble_size)]
        net = EnsembleNets([net for net, _ in nets_and_heads], sums = False, reduce_method = self.args.ensemble_reduce)
        head = EnsembleNets([head for _, head in nets_and_heads], sums = True, split_input = True, reduce_method = self.args.ensemble_reduce)
        return net, head
    
    def forward(self, x):
        individuals = self.head(self.net(x), override_return_individuals = True)
        ensemble_std = torch.std(individuals, dim = 0).mean()
        self.log("ensemble_stddev", ensemble_std)
        return self.head.reduce(individuals, dim = 0)

def obtain_NTK_data(main_datamodule):

    main_datamodule.setup()
    train_set = main_datamodule.train_dataloader().dataset
    val_set = main_datamodule.val_dataloader().dataset

    rng = np.random.default_rng(42)
    indices_train = rng.choice(len(train_set), size = 100, replace = False)
    indices_val = rng.choice(len(val_set), size = 100, replace = False)
    train_NTK_data = torch.stack([train_set[i][0] for i in indices_train])
    val_NTK_data = torch.stack([val_set[i][0] for i in indices_val])

    return train_NTK_data, val_NTK_data

# https://pytorch.org/functorch/stable/notebooks/neural_tangent_kernels.html
def eval_NTK(net, data_A, data_B):

    # prev_device = next(iter(net.parameters())).device
    # net = net.to(data_A.device)
    fnet, params = make_functional(net)

    # Obtain device by a parameter from net
    device = next(iter(net.parameters())).device

    data_A_dev = data_A.to(device)
    data_B_dev = data_B.to(device)

    def fnet_single(params, x):
        return fnet(params, x.unsqueeze(0)).squeeze(0)
    
    def empirical_ntk_jacobian_contraction(fnet_single, params, x1, x2, compute='full'):
        
        # Compute J(x1)
        jac1 = vmap(jacrev(fnet_single), (None, 0))(params, x1)
        jac1 = [j.flatten(2) for j in jac1]
        
        # Compute J(x2)
        jac2 = vmap(jacrev(fnet_single), (None, 0))(params, x2)
        jac2 = [j.flatten(2) for j in jac2]
        
        # Compute J(x1) @ J(x2).T
        einsum_expr = None
        if compute == 'full':
            einsum_expr = 'Naf,Mbf->NMab'
        elif compute == 'trace':
            einsum_expr = 'Naf,Maf->NM'
        elif compute == 'diagonal':
            einsum_expr = 'Naf,Maf->NMa'
        else:
            assert False
            
        result = torch.stack([torch.einsum(einsum_expr, j1, j2) for j1, j2 in zip(jac1, jac2)]).detach()
        result = result.sum(0)
        return result
    
    NTK_batchsize = 10
    result = torch.zeros((data_A.shape[0], data_B.shape[0]))

    batchesA = data_A.shape[0] // NTK_batchsize
    batchesB = data_B.shape[0] // NTK_batchsize

    for NTK_i in range(batchesA):
        for NTK_j in range(batchesB):

            si = NTK_i * NTK_batchsize
            sj = NTK_j * NTK_batchsize
            ei = si + NTK_batchsize
            ej = sj + NTK_batchsize
            
            result[si:ei, sj:ej] = empirical_ntk_jacobian_contraction(
                fnet_single,
                params,
                data_A_dev[si:ei],
                data_B_dev[sj:ej],
                'trace'
            )

    print("Evaluated empirical NTK with shape: %s" % repr(result.shape))

    # Return net to previous device
    # net = net.to(prev_device)

    return result

def min_max_normalize(tensor):
    return (tensor - tensor.min()) / (tensor.max() - tensor.min())


models_dict =\
{
    "default": SimpleModel,
    "ensemble": EnsembleModel,
}

#################################################################

import types

def main(hparams):
    
    seed = hparams.seed * (hparams.runid + 1)
    pl.seed_everything(seed)
    
    main_datamodule, input_dim = get_data_module(hparams.dataset, hparams.batch_size, data_augmentation = (hparams.dataAug > 0))
    train_NTK_data, val_NTK_data = obtain_NTK_data(main_datamodule)
    NTK_data = [train_NTK_data, val_NTK_data]

    def on_validation_epoch_end(self):
        self.evaluated_NTKs.append(eval_NTK(
            nn.Sequential(self.net, self.head),
            NTK_data[0], NTK_data[1]
        ))
        # self.evaluated_NTKs.append(torch.normal(0, 1, size = (100, 100)))

    all_model_NTKs = []
    model_list = [hparams.model, hparams.modelB]

    wbgroup = GetArgsStr(hparams)
    if wbgroup is None:
        wandb_logger = WandbLogger(
            project="NTK-ensemble",
            config = vars(hparams),
        )
    else:
        wandb_logger = WandbLogger(
            project="NTK-ensemble",
            config = vars(hparams),
            group = wbgroup
        )

    model_ensemble = models_dict["ensemble"](hparams, input_dim)
    model_ensemble = model_ensemble.to('cuda')
    NTK_ensemble = eval_NTK(
        nn.Sequential(model_ensemble.net, model_ensemble.head),
        NTK_data[0], NTK_data[1]
    ).detach().cpu()

    if args.min_max_normalize:
        NTK_ensemble = min_max_normalize(NTK_ensemble)

    NTK_sum = torch.zeros_like(NTK_ensemble)
    NTK_sum_2 = torch.zeros_like(NTK_ensemble)
    NTK_wide_sum = torch.zeros_like(NTK_ensemble)

    # Compute sum2
    for i in range(args.ensemble_size):

        # Init our model
        # model = models_dict["default"](hparams, input_dim)
        # model = model.to('cuda')
        
        NTK_curr_model = eval_NTK(
            # nn.Sequential(model.net, model.head),
            nn.Sequential(model_ensemble.net.nets[i], model_ensemble.head.nets[i]),
            NTK_data[0], NTK_data[1]
        ).detach().cpu()

        NTK_sum_2 += NTK_curr_model
        
        NTK_mean_2 = NTK_sum_2 / (i+1)
        if args.min_max_normalize:
            NTK_mean_2 = min_max_normalize(NTK_mean_2)

        NTK_diff_2 = (NTK_ensemble - NTK_mean_2).abs().mean()
        
        wandb_logger._prefix = ""
        wandb_logger.log_metrics({"NTK_diff_EnsembleIndividuals": NTK_diff_2}, step = i+1)

        # del model

    del model_ensemble

    import copy
    hparams_wide = copy.deepcopy(hparams)
    hparams_wide.hidden_dim *= int(hparams.wide)

    # Compute wide sum
    for i in range(args.ensemble_size):

        # Init our model
        model = models_dict["default"](hparams_wide, input_dim)
        model = model.to('cuda')
        
        NTK_curr_model = eval_NTK(
            nn.Sequential(model.net, model.head),
            NTK_data[0], NTK_data[1]
        ).detach().cpu()

        NTK_wide_sum += NTK_curr_model
    
    NTK_wide_sum = NTK_wide_sum / args.ensemble_size
    NTK_wide_sum = min_max_normalize(NTK_wide_sum)

    all_NTKs = []

    # Compute sum1
    for i in range(args.network_samples):

        # Init our model
        model = models_dict["default"](hparams, input_dim)
        model = model.to('cuda')
        
        NTK_curr_model = eval_NTK(
            nn.Sequential(model.net, model.head),
            NTK_data[0], NTK_data[1]
        ).detach().cpu()

        all_NTKs.append(NTK_curr_model)
        NTK_sum += NTK_curr_model
        
        NTK_mean = NTK_sum / (i+1)
        if args.min_max_normalize:
            NTK_mean = min_max_normalize(NTK_mean)

        NTK_diff = (NTK_ensemble - NTK_mean).abs().mean()
        NTK_diff_2 = (NTK_mean_2 - NTK_mean).abs().mean()
        NTK_diff_wide = (NTK_wide_sum - NTK_mean).abs().mean()
        
        wandb_logger._prefix = ""
        wandb_logger.log_metrics({
            "NTK_diff_EnsembleBias": NTK_diff,
            "NTK_diff_RegularNetwork": NTK_diff_2,
            "NTK_diff_WideNetwork": NTK_diff_wide
        }, step = i+1)

        del model
    
    # Plot NTK mean - variance relationship
    all_NTKs = torch.stack(all_NTKs, 0) # [N_samples, N_data0, N_data1]
    print(all_NTKs.shape)
    all_NTK_means = all_NTKs.mean(0).flatten()
    all_NTK_stds = all_NTKs.std(0).flatten()

    print(all_NTK_means.shape)
    print(all_NTK_means)

    from matplotlib import pyplot as plt
    plt.scatter(all_NTK_means, all_NTK_stds, s = 8, alpha = 0.5)
    plt.xlabel("means")
    plt.ylabel("stddevs")
    plt.savefig("NTK-mean-stddev.png")

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
    parser.add_argument("--modelB", type=str, default='default')
    parser.add_argument("--dataset", type=str, default='mnist')
    
    # Model
    parser.add_argument("--hidden_dim", type=int, default=2048)
    parser.add_argument("--ensemble_size", type=int, default=16)
    parser.add_argument("--network_samples", type=int, default=2048)
    parser.add_argument("--ensemble_expansion", type=int, default=1)
    parser.add_argument("--ensemble_reduce", type=str, default=torch.sum)
    parser.add_argument("--initialization_scale", type=float, default=1.0)
    parser.add_argument("--wide", type=float, default=4.0)

    # Training
    parser.add_argument('--dataAug', type=int, default=0, help="Data augmentation on(1) / off(0).")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for training.")
    parser.add_argument('--epochs', type=int, default=50, help="Number of epochs for training.")
    parser.add_argument('--augTrials', type=int, default=4, help="Trials per augmentation (ignored if augmentation disabled)")

    parser.add_argument("--min_max_normalize", "-mm", action = 'store_true')
    
    parser = parent_parser.add_argument_group("Run metadata / WandB sweeps")
    parser.add_argument('--keyargs', type=str, default="", help='Key variables in HP tune, splitted in commas')
    parser.add_argument('--aaarunid', type=int, default=0, help='Run ID.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed. will be multiplied with runid+1 to ensure different RNGs for different runs.')

    args = parent_parser.parse_args()

    args.dataset = args.dataset.lower()

    # Force the size of the nets for ensemble
    args.ensemble_expansion = args.ensemble_size
    args.ensemble_reduce = 'mean'

    print(args)
    
    # Process arguments a bit
    args.runid = args.aaarunid
    del(args.aaarunid)
    
    if args.dataAug == 0:
        args.augTrials = 1

    print("Args: %s" % vars(args))

    main(args)
    
