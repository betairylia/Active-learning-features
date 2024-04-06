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

import uq_models as uq
from uq_models import SimpleModel

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

def get_data_module(
    dataset_name,
    batch_size,
    data_augmentation=True,
    num_workers=16,
    data_dir='./data',
    do_partial_train = False,
    do_contamination = True,
    use_full_trainset = True,
    test_set_max = -1,
    is_binary = 0,
    noise_std = 0.3,
    blur_sigma = 2.0):
    
    args = {
        "data_dir": data_dir,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "do_partial_train": do_partial_train,
        "do_contamination": do_contamination,
        "use_full_trainset": use_full_trainset,
        "test_set_max": test_set_max,
        "is_binary": is_binary,
        "noise_std": noise_std,
        "blur_sigma": blur_sigma,
    }

    if dataset_name == 'mnist':
        main_dm = MNIST_UncertaintyDM(**args)
    
    elif dataset_name == 'cifar10':
        main_dm = CIFAR10_UncertaintyDM(**args)
    
    elif dataset_name == 'svhn':
        main_dm = SVHN_UncertaintyDM(**args)

    elif dataset_name == 'imagenet':
        main_dm = ImageNet_Validation_UncertaintyDM(**args)
    
    return main_dm, input_size_dict[dataset_name]


#################################################################

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

class NaiveEnsembleSummationModel_HeadOnly(SimpleModel):

    def __init__(self, args, input_shape, output_dim=10, ensemble_size = 4):

        super().__init__(args, input_shape, output_dim)

        self.ensemble_size = ensemble_size
        self.ensemble_reduce = 'sum'
        self.net, self.head = self.getNets_ensemble(input_shape, output_dim)
        print(self)

        # Store the initialized network
        self.net_init = copy.deepcopy(self.net)
        self.head_init = copy.deepcopy(self.head)

    def getNets_ensemble(self, input_shape, output_dim):
        heads = [
            self.net_factory.getNets(
                input_shape,
                [output_dim],
                hidden_dim = self.args.hidden_dim,
                dropout_rate = self.args.dropout_rate
            )[1] for _ in range(self.ensemble_size)
        ]
        head = EnsembleNets(heads, sums = True, split_input = False, reduce_method = self.ensemble_reduce)
        return self.net, head

    def forward(self, x):

        self.disable_dropout(self.net)
        self.disable_dropout(self.head)

        z = self.net(x)
        individuals = self.head(z, override_return_individuals = True)
        individual_probs = F.softmax(individuals, dim = 2)

        # ensemble_std = torch.std(individual_probs, dim = 0).mean(1)
        model_prediction = individual_probs.mean(0)
        entropy = -torch.sum(model_prediction * torch.log(model_prediction + 1e-8), dim = 1)
        uncertainty = entropy

        return self.head.reduce(individuals, dim = 0), uncertainty # ensemble_std

class NaiveEnsembleSummationModel(SimpleModel):

    def __init__(self, args, input_shape, output_dim=10, ensemble_size = 4):

        super().__init__(args, input_shape, output_dim)

        self.ensemble_size = ensemble_size
        self.ensemble_reduce = 'sum'
        self.net, self.head = self.getNets_ensemble(input_shape, output_dim)
        print(self)

        # Store the initialized network
        self.net_init = copy.deepcopy(self.net)
        self.head_init = copy.deepcopy(self.head)

    def getNets_ensemble(self, input_shape, output_dim):
        nets_and_heads = [
            self.net_factory.getNets(
                input_shape,
                [output_dim],
                hidden_dim = self.args.hidden_dim,
                dropout_rate = self.args.dropout_rate
            ) for _ in range(self.ensemble_size)
        ]
        net = EnsembleNets([net for net, _ in nets_and_heads], sums = False, reduce_method = self.ensemble_reduce)
        head = EnsembleNets([head for _, head in nets_and_heads], sums = True, split_input = True, reduce_method = self.ensemble_reduce)
        return net, head

    def forward(self, x):

        self.disable_dropout(self.net)
        self.disable_dropout(self.head)

        individuals = self.head(self.net(x), override_return_individuals = True)
        individual_probs = F.softmax(individuals, dim = 2)

        # ensemble_std = torch.std(individual_probs, dim = 0).mean(1)
        model_prediction = individual_probs.mean(0)
        entropy = -torch.sum(model_prediction * torch.log(model_prediction + 1e-8), dim = 1)
        uncertainty = entropy

        return self.head.reduce(individuals, dim = 0), uncertainty # ensemble_std

class NaiveEnsembleModel(NaiveEnsembleSummationModel):

    def forward(self, x):

        self.disable_dropout(self.net)
        self.disable_dropout(self.head)

        individuals = self.head(self.net(x), override_return_individuals = True)
        individual_probs = F.softmax(individuals, dim = 2)

        # ensemble_std = torch.std(individual_probs, dim = 0).mean(1)
        model_prediction = individual_probs.mean(0)
        entropy = -torch.sum(model_prediction * torch.log(model_prediction + 1e-8), dim = 1)
        uncertainty = entropy

        if self.training:
            return individuals, uncertainty # ensemble_std
        else:
            return self.head.reduce(individuals, dim = 0), uncertainty # ensemble_std

    def training_step(self, batch, batch_nb):

        i, x, y, o = batch

        logits = self.scale_output(self(x)[0], x)

        # loss = F.cross_entropy(self(x), y)
        loss = 0
        for i in range(logits.shape[0]):
            loss += self.loss(logits[i, :, :], y)
        
        # loss /= logits.shape[0]

        self.log("train_loss", loss)

        return loss

class NaiveEnsembleModel_KernelNoiseOnly(NaiveEnsembleModel):

    def forward(self, x):

        self.disable_dropout(self.net)
        self.disable_dropout(self.head)

        individuals = self.head(self.net(x), override_return_individuals = True)
        individuals_init = self.head_init(self.net_init(x), override_return_individuals = True)
        individuals = individuals - individuals_init

        individual_probs = F.softmax(individuals, dim = 2)

        # ensemble_std = torch.std(individual_probs, dim = 0).mean(1)
        model_prediction = individual_probs.mean(0)
        entropy = -torch.sum(model_prediction * torch.log(model_prediction + 1e-8), dim = 1)
        uncertainty = entropy

        if self.training:
            return individuals, uncertainty # ensemble_std
        else:
            return self.head.reduce(individuals, dim = 0), uncertainty # ensemble_std
        
class NaiveEnsembleSummationModel_KernelNoiseOnly(NaiveEnsembleSummationModel):

    def forward(self, x):

        self.disable_dropout(self.net)
        self.disable_dropout(self.head)

        individuals = self.head(self.net(x), override_return_individuals = True)
        individuals_init = self.head_init(self.net_init(x), override_return_individuals = True)
        individuals = individuals - individuals_init

        individual_probs = F.softmax(individuals, dim = 2)

        # ensemble_std = torch.std(individual_probs, dim = 0).mean(1)
        model_prediction = individual_probs.mean(0)
        entropy = -torch.sum(model_prediction * torch.log(model_prediction + 1e-8), dim = 1)
        uncertainty = entropy

        if self.training:
            return individuals, uncertainty # ensemble_std
        else:
            return self.head.reduce(individuals, dim = 0), uncertainty # ensemble_std

class RandomModel(SimpleModel):
    
    def forward(self, x):

        self.disable_dropout(self.net)
        self.disable_dropout(self.head)
        logits = self.head(self.net(x))

        return logits, torch.normal(torch.zeros(logits.shape[0], device = logits.device), torch.ones(logits.shape[0], device = logits.device))

class NaiveDropoutModel(SimpleModel):
    
    def forward(self, x):

        if self.training:
            self.disable_dropout(self.net)
            self.disable_dropout(self.head)
            logits = self.head(self.net(x))
            return logits, None
        else:
            self.enable_dropout(self.net)
            self.enable_dropout(self.head)
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

            # uncertainty = torch.std(logits, dim = 0).mean(dim = 1)
            # uncertainty = probs.std(0).mean(1)
            model_prediction = probs.mean(0)
            entropy = -torch.sum(model_prediction * torch.log(model_prediction + 1e-8), dim = 1)
            uncertainty = entropy

            return logits.mean(dim = 0), uncertainty

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

            # uncertainty = torch.std(logits, dim = 0).mean(dim = 1)
            # uncertainty = probs.std(0).mean(1)
            model_prediction = probs.mean(0)
            entropy = -torch.sum(model_prediction * torch.log(model_prediction + 1e-8), dim = 1)
            uncertainty = entropy

            return logits.mean(dim = 0), uncertainty

class TestTimeOnly_ApproximateDropoutModel(SimpleModel):

    def forward(self, x):

        if self.training:
            self.disable_dropout(self.net)
            self.disable_dropout(self.head)
            # self.enable_dropout(self.net)
            # self.enable_dropout(self.head)
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
            # uncertainty = torch.std(logits, dim = 0).mean(dim = 1)
            model_prediction = probs.mean(0)
            entropy = -torch.sum(model_prediction * torch.log(model_prediction + 1e-8), dim = 1)
            uncertainty = entropy

            self.recorder_identity(self.net)
            self.recorder_identity(self.head)

            return logits.mean(dim = 0), uncertainty

class TestTimeOnly_GroundTruthInit_PlainTrainingDifference(SimpleModel):

    def __init__(self, args, input_shape, output_dim = 10):
        
        super().__init__(args, input_shape, output_dim)

    def forward(self, x):

        if self.training:

            self.disable_dropout(self.net)
            self.disable_dropout(self.head)
            logits = self.head(self.net(x))
            return logits, None

        else:

            ##################################################
            # INITIAL MODEL
            ##################################################

            self.disable_dropout(self.net_init)
            self.disable_dropout(self.head_init)

            logits_init = self.head_init(self.net_init(x))

            ##################################################
            # TRAINED MODEL
            ##################################################

            self.disable_dropout(self.net)
            self.disable_dropout(self.head)

            logits = self.head(self.net(x))

            distance = (logits - logits_init).abs().sum(-1)
            uncertainty = -distance

            return logits, uncertainty.to(logits.device)

def solve_one_dim_ot(source, target, dim = -1):
    sorted_source, _ = torch.sort(source, dim = dim)
    sorted_target, _ = torch.sort(target, dim = dim)
    return (sorted_target - sorted_source).abs().sum(dim)

class TestTimeOnly_GroundTruthInit_WassersteinModel(SimpleModel):

    def __init__(self, args, input_shape, output_dim = 10):
        
        super().__init__(args, input_shape, output_dim)

    def forward(self, x):

        if self.training:

            self.disable_dropout(self.net)
            self.disable_dropout(self.head)
            # self.enable_dropout(self.net)
            # self.enable_dropout(self.head)
            logits = self.head(self.net(x))
            return logits, None

        else:

            ##################################################
            # INITIAL MODEL
            ##################################################

            # Record
            
            self.disable_dropout(self.net_init)
            self.disable_dropout(self.head_init)

            self.record(self.net_init)
            self.record(self.head_init)

            _logits_init = self.head_init(self.net_init(x))

            # Replay

            self.enable_dropout(self.net_init)
            self.enable_dropout(self.head_init)

            self.replay(self.net_init)
            self.replay(self.head_init)

            logits_init = []
            probs_init = []
            for i in range(self.args.dropout_iters):
                logits_init.append(self.head_init(self.net_init(x)))
                pred_prob_init = F.softmax(logits_init[-1], dim = 1)
                probs_init.append(pred_prob_init)

            logits_init = torch.stack(logits_init, dim = 0)
            probs_init = torch.stack(probs_init, dim = 0)

            ##################################################
            # TRAINED MODEL
            ##################################################

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

            logits = []
            probs = []
            for i in range(self.args.dropout_iters):
                logits.append(self.head(self.net(x)))
                pred_prob = F.softmax(logits[-1], dim = 1)
                probs.append(pred_prob)

            logits = torch.stack(logits, dim = 0)
            probs = torch.stack(probs, dim = 0)
            
            # Compute OT for each dimension
            # N_POINTS, BATCH_SIZE, DIM = logits.shape
            # margin_a = np.ones((N_POINTS,)) / N_POINTS
            # margin_b = np.ones((N_POINTS,)) / N_POINTS

            # distance = torch.zeros(BATCH_SIZE)

            # for b in range(BATCH_SIZE):
            #     for d in range(DIM):
            #         M = (logits_init[:, b, d].unsqueeze(1) - logits[:, b, d].unsqueeze(0)).abs().detach().cpu().numpy()
            #         distance[b] += ot.emd2(margin_a, margin_b, M)

            distance = solve_one_dim_ot(logits, logits_init, dim = 0).sum(-1)

            model_prediction = probs.mean(0)
            # entropy = -torch.sum(model_prediction * torch.log(model_prediction + 1e-8), dim = 1)
            uncertainty = -distance

            self.recorder_identity(self.net)
            self.recorder_identity(self.head)

            return logits.mean(dim = 0), uncertainty.to(logits.device)

class TestTimeOnly_GroundTruthInit_VelocityModel(SimpleModel):

    def __init__(self, args, input_shape, output_dim = 10):
        
        super().__init__(args, input_shape, output_dim)

    def forward(self, x):

        if self.training:

            self.disable_dropout(self.net)
            self.disable_dropout(self.head)
            # self.enable_dropout(self.net)
            # self.enable_dropout(self.head)
            logits = self.head(self.net(x))
            return logits, None

        else:

            ##################################################
            # INITIAL MODEL
            ##################################################

            # Record
            
            self.disable_dropout(self.net_init)
            self.disable_dropout(self.head_init)

            self.record(self.net_init)
            self.record(self.head_init)

            _logits_init = self.head_init(self.net_init(x))

            # Replay

            self.enable_dropout(self.net_init)
            self.enable_dropout(self.head_init)

            self.replay(self.net_init)
            self.replay(self.head_init)

            logits_init = []
            probs_init = []
            for i in range(self.args.dropout_iters):
                logits_init.append(self.head_init(self.net_init(x)))
                pred_prob_init = F.softmax(logits_init[-1], dim = 1)
                probs_init.append(pred_prob_init)

            logits_init = torch.stack(logits_init, dim = 0)
            probs_init = torch.stack(probs_init, dim = 0)

            ##################################################
            # TRAINED MODEL
            ##################################################

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

            logits = []
            probs = []
            for i in range(self.args.dropout_iters):
                logits.append(self.head(self.net(x)))
                pred_prob = F.softmax(logits[-1], dim = 1)
                probs.append(pred_prob)

            logits = torch.stack(logits, dim = 0)
            probs = torch.stack(probs, dim = 0)
            
            # Compute OT for each dimension
            # N_POINTS, BATCH_SIZE, DIM = logits.shape
            # margin_a = np.ones((N_POINTS,)) / N_POINTS
            # margin_b = np.ones((N_POINTS,)) / N_POINTS

            # distance = torch.zeros(BATCH_SIZE)

            # for b in range(BATCH_SIZE):
            #     for d in range(DIM):
            #         M = (logits_init[:, b, d].unsqueeze(1) - logits[:, b, d].unsqueeze(0)).abs().detach().cpu().numpy()
            #         distance[b] += ot.emd2(margin_a, margin_b, M)

            move_dirc = torch.sort(logits, dim = -1)[0] - torch.sort(logits_init, dim = -1)[0]
            moved_distance = torch.sqrt(1e-8 + (logits.mean(0, keepdims = True) - logits_init.mean(0, keepdims = True)) ** 2)
            normed_velocity = move_dirc / moved_distance
            uncertainty = normed_velocity.std(0).mean(-1)

            model_prediction = probs.mean(0)
            # entropy = -torch.sum(model_prediction * torch.log(model_prediction + 1e-8), dim = 1)

            self.recorder_identity(self.net)
            self.recorder_identity(self.head)

            return logits.mean(dim = 0), uncertainty.to(logits.device)

class TestTimeOnly_GaussianInit_WassersteinModel(SimpleModel):

    def __init__(self, args, input_shape, output_dim = 10):
        
        super().__init__(args, input_shape, output_dim)

    def forward(self, x):

        if self.training:

            self.disable_dropout(self.net)
            self.disable_dropout(self.head)
            # self.enable_dropout(self.net)
            # self.enable_dropout(self.head)
            logits = self.head(self.net(x))
            return logits, None

        else:

            ##################################################
            # TRAINED MODEL
            ##################################################

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

            logits = []
            probs = []
            for i in range(self.args.dropout_iters):
                logits.append(self.head(self.net(x)))
                pred_prob = F.softmax(logits[-1], dim = 1)
                probs.append(pred_prob)

            logits = torch.stack(logits, dim = 0)
            probs = torch.stack(probs, dim = 0)

            logits_mean = logits.mean(0).unsqueeze(0).repeat(logits.shape[0], 1, 1)
            logits_std = logits.std(0).unsqueeze(0).repeat(logits.shape[0], 1, 1)
            logits_gaussian_init = torch.normal(logits_mean, logits_std)
            
            # Compute OT for each dimension
            # N_POINTS, BATCH_SIZE, DIM = logits.shape
            # margin_a = np.ones((N_POINTS,)) / N_POINTS
            # margin_b = np.ones((N_POINTS,)) / N_POINTS

            # distance = torch.zeros(BATCH_SIZE)

            # for b in range(BATCH_SIZE):
            #     for d in range(DIM):
            #         M = (logits_init[:, b, d].unsqueeze(1) - logits[:, b, d].unsqueeze(0)).abs().detach().cpu().numpy()
            #         distance[b] += ot.emd2(margin_a, margin_b, M)

            distance = solve_one_dim_ot(logits, logits_gaussian_init, dim = 0).sum(-1)

            model_prediction = probs.mean(0)
            # entropy = -torch.sum(model_prediction * torch.log(model_prediction + 1e-8), dim = 1)
            uncertainty = -distance

            self.recorder_identity(self.net)
            self.recorder_identity(self.head)

            return logits.mean(dim = 0), uncertainty.to(logits.device)

class TestTimeOnly_NTK(SimpleModel):

    def __init__(self, args, input_shape, output_dim = 10):
        
        super().__init__(args, input_shape, output_dim)

        self.combined_net = nn.Sequential(self.net, self.head)
        self.y_index = 0

    def extra_init(self, reference_dl):

        self.reference_dl = reference_dl 

    def fnet_single_hvp(self, x, y_index = -1):

        # Torch 2.0 / CUDA 11.7
        # def foo(params):
        #     return torch.func.functional_call(self, params, (x,))[0, y_index]

        # Torch 1.13 / FuncTorch
        def foo(params):

            result = self.fnet(params, self.fbuffer, x)[0]

            resolved_y_index = y_index
            if resolved_y_index < 0:
                resolved_y_index = torch.argmax(result)

            return result[resolved_y_index]
        
        return foo

    def on_validation_epoch_start(self):

        torch.set_grad_enabled(True)

        # 1. Extract parameters and make functional version of the network

        # Torch 1.13 / FuncTorch
        funcresult = make_functional_with_buffers(nn.Sequential(self.net, self.head))
        self.fnet, _, self.fbuffer = funcresult
        self.fparams = dict(self.combined_net.named_parameters())

        # Torch 2.0 / CUDA 11.7
        # self.fparams = dict(self.combined_net.named_parameters())

        self.gradResults = DropoutHessianRecorder()

        # Torch 2.0 / CUDA 11.7
        # def fnet_single(params, x):
        #     return torch.func.functional_call(self, params, (x,))

        # print("Created functionalized network")

        # LeNet-5 parameters:
        # ipdb> print("\n".join([repr((p[0], p[1].shape)) for p in self.fparams.items()]))
        # ('0.0.0.weight', torch.Size([6, 1, 5, 5]))
        # ('0.0.0.bias', torch.Size([6]))
        # ('0.0.3.weight', torch.Size([16, 6, 5, 5]))
        # ('0.0.3.bias', torch.Size([16]))
        # ('0.2.0.weight', torch.Size([120, 400]))
        # ('0.2.0.bias', torch.Size([120]))
        # ('0.2.2.weight', torch.Size([84, 120]))
        # ('0.2.2.bias', torch.Size([84]))
        # ('1.weight', torch.Size([10, 84]))
        # ('1.bias', torch.Size([10]))

        # Loop thru all reference data-points
        for batch in self.reference_dl:
            
            i, x, y, o = batch
            x = x.to(self.device)

            # Feed-forward

            # Torch 2.0 / CUDA 11.7
            # hvp_result = torch.func.vhp(fnet_single_vhp(x, 0), self.fparams, masked_parameters).t()
            # hvp_result = torch.func.vjp(torch.func.grad(fnet_single_hvp(x, self.y_index)), self.fparams)[1](masked_parameters).t()

            # self.hvpResults.record(hvp_result)

            # Torch 1.13 / FuncTorch
            grad_result = grad(self.fnet_single_hvp(x, self.y_index))(to_unnamed(self.fparams))

            # TODO: Multiply with D (:= \Theta^{-1}(x, X) ∂f L(X), i.e., NTK-normalized training direction)
            # or simply unnormalized might be enough

            self.gradResults.record(unnamed_tuple_to_named(self.fparams, grad_result))

            # print("-", end = '')

        # Store the running var / mean
        # The results are stored in self.hvpResults and is accessible via self.hvpResults.mean() / self.hvpResults.variance().

        # breakpoint()

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        torch.set_grad_enabled(False)

    def forward(self, x):

        if self.training:

            self.disable_dropout(self.net)
            self.disable_dropout(self.head)
            logits = self.head(self.net(x))
            return logits, None

        else:

            # proxy
            self.disable_dropout(self.net)
            self.disable_dropout(self.head)

            logits = self.head(self.net(x))

            # Comment if 3b; Enable if 3a
            # self.enable_dropout(self.net)
            # self.enable_dropout(self.head)

            len_x = x.shape[0]
            full_x = x

            ex_uncertainties = torch.zeros(len_x, device = x.device)

            for j in range(len_x):

                # print("%5d S" % j, end='')

                x = full_x[j].unsqueeze(0)

                # 3. Collect non-dropout statstics at datapoint x
                grad_at_x = grad(self.fnet_single_hvp(x, self.y_index))(to_unnamed(self.fparams))
                grad_tuple = unnamed_tuple_to_named(self.fparams, grad_at_x)

                # 4. Dot-product the hvp var / mean with regular gradients @ evaluating data-points
                ntk_prenorm = params_multiply(self.gradResults.mean(), grad_tuple)
                ntk_norm = params_l2norm(ntk_prenorm)
                ntk_eval = params_sum(ntk_prenorm)

                if self.val_full_batch is not None:

                    o = self.val_full_batch[3][j]

                    if o == 1:
                        self.log("NTK_eval_unseen", ntk_eval, on_epoch = True)
                        self.log("NTK_2norm_unseen", ntk_norm, on_epoch = True)
                        self.log("grad_2norm_unseen", params_l2norm(grad_tuple), on_epoch = True)
                    else:
                        self.log("NTK_eval_seen", ntk_eval, on_epoch = True)
                        self.log("NTK_2norm_seen", ntk_norm, on_epoch = True)
                        self.log("grad_2norm_seen", params_l2norm(grad_tuple), on_epoch = True)

                extra_ntk_uncertainty =\
                    ntk_norm
                    # Tvar_Hvar +\
                    # Tvar_Hepc +\
                    # Tepc_Hvar
                
                ex_uncertainties[j] = ntk_norm.detach()

                # print(".")

            return logits, ex_uncertainties

class TestTimeOnly_NTK_withInv(SimpleModel):

    def __init__(self, args, input_shape, output_dim = 10):
        
        super().__init__(args, input_shape, output_dim)

        self.combined_net = nn.Sequential(self.net, self.head)
        self.y_index = 0

    def extra_init(self, reference_dl):

        self.reference_dl = reference_dl

    def fnet_single_hvp(self, x, y_index = -1):

        # Torch 2.0 / CUDA 11.7
        # def foo(params):
        #     return torch.func.functional_call(self, params, (x,))[0, y_index]

        # Torch 1.13 / FuncTorch
        def foo(params):

            result = self.fnet(params, self.fbuffer, x)[0]

            resolved_y_index = y_index
            if resolved_y_index < 0:
                resolved_y_index = torch.argmax(result)

            return result[resolved_y_index]
        
        return foo

    def on_validation_epoch_start(self):

        torch.set_grad_enabled(True)

        # 1. Extract parameters and make functional version of the network

        # Torch 1.13 / FuncTorch
        funcresult = make_functional_with_buffers(nn.Sequential(self.net, self.head))
        self.fnet, _, self.fbuffer = funcresult
        self.fparams = dict(self.combined_net.named_parameters())

        # Torch 2.0 / CUDA 11.7
        # self.fparams = dict(self.combined_net.named_parameters())

        self.gradResults = DropoutHessianRecorder()

        # Torch 2.0 / CUDA 11.7
        # def fnet_single(params, x):
        #     return torch.func.functional_call(self, params, (x,))

        # print("Created functionalized network")

        # LeNet-5 parameters:
        # ipdb> print("\n".join([repr((p[0], p[1].shape)) for p in self.fparams.items()]))
        # ('0.0.0.weight', torch.Size([6, 1, 5, 5]))
        # ('0.0.0.bias', torch.Size([6]))
        # ('0.0.3.weight', torch.Size([16, 6, 5, 5]))
        # ('0.0.3.bias', torch.Size([16]))
        # ('0.2.0.weight', torch.Size([120, 400]))
        # ('0.2.0.bias', torch.Size([120]))
        # ('0.2.2.weight', torch.Size([84, 120]))
        # ('0.2.2.bias', torch.Size([84]))
        # ('1.weight', torch.Size([10, 84]))
        # ('1.bias', torch.Size([10]))

        # Loop thru all reference data-points
        for batch in self.reference_dl:
            
            i, x, y, o = batch
            x = x.to(self.device)

            # Feed-forward

            # Torch 2.0 / CUDA 11.7
            # hvp_result = torch.func.vhp(fnet_single_vhp(x, 0), self.fparams, masked_parameters).t()
            # hvp_result = torch.func.vjp(torch.func.grad(fnet_single_hvp(x, self.y_index)), self.fparams)[1](masked_parameters).t()

            # self.hvpResults.record(hvp_result)

            # Torch 1.13 / FuncTorch
            grad_result = grad(self.fnet_single_hvp(x, self.y_index))(to_unnamed(self.fparams))
            grad_result = unnamed_tuple_to_named(self.fparams, grad_result)

            # NTK Normalization
            l2n_sq = params_l2norm_sq(grad_result)
            grad_result = params_scale(grad_result, 1.0 / l2n_sq)

            self.gradResults.record(grad_result)

            # print("-", end = '')

        # Store the running var / mean
        # The results are stored in self.hvpResults and is accessible via self.hvpResults.mean() / self.hvpResults.variance().

        # breakpoint()

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        torch.set_grad_enabled(False)

    def forward(self, x):

        if self.training:

            self.disable_dropout(self.net)
            self.disable_dropout(self.head)
            logits = self.head(self.net(x))
            return logits, None

        else:

            # proxy
            self.disable_dropout(self.net)
            self.disable_dropout(self.head)

            logits = self.head(self.net(x))

            # Comment if 3b; Enable if 3a
            # self.enable_dropout(self.net)
            # self.enable_dropout(self.head)

            len_x = x.shape[0]
            full_x = x

            ex_uncertainties = torch.zeros(len_x, device = x.device)

            for j in range(len_x):

                # print("%5d S" % j, end='')

                x = full_x[j].unsqueeze(0)

                # 3. Collect non-dropout statstics at datapoint x
                grad_at_x = grad(self.fnet_single_hvp(x, self.y_index))(to_unnamed(self.fparams))
                grad_tuple = unnamed_tuple_to_named(self.fparams, grad_at_x)

                # 4. Dot-product the hvp var / mean with regular gradients @ evaluating data-points
                ntk_prenorm = params_multiply(self.gradResults.mean(), grad_tuple)
                ntk_norm = params_l2norm(ntk_prenorm)
                ntk_eval = params_sum(ntk_prenorm)

                if self.val_full_batch is not None:

                    o = self.val_full_batch[3][j]

                    if o == 1:
                        self.log("NTK_eval_unseen", ntk_eval, on_epoch = True)
                        self.log("NTK_2norm_unseen", ntk_norm, on_epoch = True)
                        self.log("grad_2norm_unseen", params_l2norm(grad_tuple), on_epoch = True)
                    else:
                        self.log("NTK_eval_seen", ntk_eval, on_epoch = True)
                        self.log("NTK_2norm_seen", ntk_norm, on_epoch = True)
                        self.log("grad_2norm_seen", params_l2norm(grad_tuple), on_epoch = True)

                extra_ntk_uncertainty =\
                    ntk_norm
                    # Tvar_Hvar +\
                    # Tvar_Hepc +\
                    # Tepc_Hvar
                
                ex_uncertainties[j] = ntk_norm.detach()

                # print(".")

            return logits, ex_uncertainties

class TestTimeOnly_NTK_initialization(SimpleModel):

    def __init__(self, args, input_shape, output_dim = 10):
        
        super().__init__(args, input_shape, output_dim)

        self.combined_net = nn.Sequential(self.net, self.head)
        self.combined_net_init = nn.Sequential(self.net_init, self.head_init)

        self.y_index = 0
        self.fnet_summary = None
        self.grad_only_mode = False

    def fnet_single_hvp(self, x, y_index = -1, summary = None):

        # Torch 2.0 / CUDA 11.7
        # def foo(params):
        #     return torch.func.functional_call(self, params, (x,))[0, y_index]

        # Torch 1.13 / FuncTorch
        def foo(params):

            result = self.fnet(params, self.fbuffer, x)[0]

            resolved_y_index = y_index
            if resolved_y_index < 0:
                resolved_y_index = torch.argmax(result)

            if summary == "softmax":
                # result = F.softmax(result, dim = 0)
                result = result *\
                    torch.clamp(softmax_gradient_at_idx(result, resolved_y_index).detach(), 0.2, 5.0)
            
            elif summary == "mean":
                result = torch.mean(result, dim = 0, keepdims = True)
                resolved_y_index = 0

            return result[resolved_y_index]
        
        return foo

    def get_params_init_diff(self, params):
        return params_substract(params, dict(self.combined_net_init.named_parameters()))

    def on_validation_epoch_start(self):

        # Torch 1.13 / FuncTorch
        funcresult = make_functional_with_buffers(nn.Sequential(self.net, self.head))
        self.fnet, _, self.fbuffer = funcresult
        self.fparams = dict(self.combined_net.named_parameters())

        self.params_diff = self.get_params_init_diff(self.fparams)

        self.stats_recorder = DropoutHessianRecorder()

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        torch.set_grad_enabled(False)

        means = self.stats_recorder.mean()
        variances = self.stats_recorder.variance()

        for k in means.keys():
            self.log("Gradient_means/" + k, means[k])

        for k in variances.keys():
            self.log("Gradient_variances/" + k, variances[k])

    def obtain_uncertainty_score(self, x):

        # 3. Collect non-dropout statstics at datapoint x
        grad_at_x = grad(self.fnet_single_hvp(x, self.y_index, self.fnet_summary))(to_unnamed(self.fparams))
        grad_tuple = unnamed_tuple_to_named(self.fparams, grad_at_x)

        self.stats_recorder.record(grad_tuple)

        # 4. Dot-product the hvp var / mean with regular gradients @ evaluating data-points
        if self.grad_only_mode == False:
            ntk_prenorm = params_multiply(self.params_diff, grad_tuple)
        else:
            ntk_prenorm = grad_tuple

        ntk_norm = params_l2norm(ntk_prenorm)
        # ntk_eval = params_sum(ntk_prenorm)

        return ntk_norm

    def forward(self, x):

        if self.training:

            self.disable_dropout(self.net)
            self.disable_dropout(self.head)
            logits = self.head(self.net(x))
            return logits, None

        else:

            # proxy
            self.disable_dropout(self.net)
            self.disable_dropout(self.head)

            logits = self.head(self.net(x))

            # Comment if 3b; Enable if 3a
            # self.enable_dropout(self.net)
            # self.enable_dropout(self.head)

            len_x = x.shape[0]
            full_x = x

            ex_uncertainties = torch.zeros(len_x, device = x.device)

            for j in range(len_x):

                # print("%5d S" % j, end='')

                x = full_x[j].unsqueeze(0)
                ntk_norm = self.obtain_uncertainty_score(x)

                if self.val_full_batch is not None:

                    o = self.val_full_batch[3][j]

                    if o == 1:
                        # self.log("NTK_eval_unseen", ntk_eval, on_epoch = True)
                        self.log("NTK_2norm_unseen", ntk_norm, on_epoch = True)
                        # self.log("grad_2norm_unseen", params_l2norm(grad_tuple), on_epoch = True)
                    else:
                        # self.log("NTK_eval_seen", ntk_eval, on_epoch = True)
                        self.log("NTK_2norm_seen", ntk_norm, on_epoch = True)
                        # self.log("grad_2norm_seen", params_l2norm(grad_tuple), on_epoch = True)

                extra_ntk_uncertainty =\
                    ntk_norm
                    # Tvar_Hvar +\
                    # Tvar_Hepc +\
                    # Tepc_Hvar
                
                ex_uncertainties[j] = ntk_norm.detach()

                # print(".")

            return logits, ex_uncertainties

class TestTimeOnly_NTK_initialization_classification(TestTimeOnly_NTK_initialization):

    def __init__(self, args, input_shape, output_dim = 10):
        
        super().__init__(args, input_shape, output_dim)

        self.fnet_summary = "softmax"
        self.y_index = -1

class TestTimeOnly_NTK_zero_initialization(TestTimeOnly_NTK_initialization):

    def get_params_init_diff(self, params):
        return params

class TestTimeOnly_NTK_zero_initialization_classification(TestTimeOnly_NTK_initialization):

    def __init__(self, args, input_shape, output_dim = 10):
        
        super().__init__(args, input_shape, output_dim)

        self.fnet_summary = "softmax"
        self.y_index = -1

class TestTimeOnly_NTK_zero_initialization_classification_nosoftmax(TestTimeOnly_NTK_initialization):

    def __init__(self, args, input_shape, output_dim = 10):
        
        super().__init__(args, input_shape, output_dim)

        self.fnet_summary = None
        self.y_index = -1

class TestTimeOnly_NTK_zero_initialization_mean(TestTimeOnly_NTK_initialization):

    def __init__(self, args, input_shape, output_dim = 10):
        
        super().__init__(args, input_shape, output_dim)

        self.fnet_summary = "mean"
        self.y_index = -1

# Multi dimension variants
class TestTimeOnly_NTK_zero_initialization_simple_multidim(TestTimeOnly_NTK_zero_initialization):

    def __init__(self, args, input_shape, output_dim = 10):

        super().__init__(args, input_shape, output_dim)

        self.fnet_summary = None
        self.y_index = -1

        # parser.add_argument("--random_multidim", type=int, default=1)
        # parser.add_argument("--num_multidim", type=int, default=32)

        if args.random_multidim > 0:
            self.multidim_idx = np.random.choice(output_dim, args.num_multidim, replace = False)
        else:
            self.multidim_idx = np.array([i for i in range(min(output_dim, args.num_multidim))])

    def obtain_uncertainty_score(self, x):

        scores = []

        for dix in self.multidim_idx:
            self.y_index = dix
            scores.append(super().obtain_uncertainty_score(x))
        
        scores_mean = torch.stack(scores, dim = 0).mean(dim = 0)
        return scores_mean


from functorch import make_functional, make_functional_with_buffers, jvp, grad
from func_dropout import *
# ImportError: cannot import name 'hvp' from 'functorch'

# TODO: FIXME:  Visalization is wrong. The r.v. is a dropout mask, then we need to plot (grad for parameter #j, hessian for parameter #j)
#               for this particular mask, which is a loop thru entire dataset per mask.
# class TestTimeOnly_HessianVariance_IndependenceTestVisualizer():
   
#     def __init__(self, args):

#         self.keys = args.independence_check_layers
#         self.ids = args.independence_check_dataid
#         self.do_check = (self.self.independence_check_layers is not None and self.independence_check_dataid is not None)

#         self.vis_sizeX = 3
#         self.vis_sizeY = 3

#         self.buffer = []
    
#     def visualize(self, data_id, row_header, data_dict_with_id, data_dict_general):

#         if self.do_check == false:
#             return

#         if data_id is in self.ids:

#             fig, axs = plt.subplots(1, 1 + len(self.keys), figsize = (self.vis_sizeX * len(self.keys), self.vis_sizeY))
#             plt.tight_layout()
            
#             for key, with_id, general in params_zip(data_dict_with_id, data_dict_general):
                
#                 if key is in self.keys:
                    
#                     key_id = self.keys.index(key)

#                     data_x = with_id.detach().cpu().numpy().flatten()
#                     data_y = general.detach().cpu().numpy().flatten()

#                     axs[1+key_id].scatter(data_x, data_y)
#                     plt.axis('off')

#             im = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
#             self.buffer.append(im)

#             plt.clf()
#             plt.close()
    
#     def flush(self):

#         if len(self.buffer) <= 0:
#             return None

#         dst = Image.new('RGB', (self.buffer[0].width, self.buffer[0].height * len(self.buffer)))
#         for i in range(len(self.buffer)):
#             dst.paste(self.buffer[i], (0, i * self.buffer[0].height))
        
#         self.buffer = []
#         return dst

class TestTimeOnly_HessianVariance(SimpleModel):

    def __init__(self, args, input_shape, output_dim = 10):
        
        super().__init__(args, input_shape, output_dim)

        self.combined_net = nn.Sequential(self.net, self.head)

        self.indep_test_vis = TestTimeOnly_HessianVariance_IndependenceTestVisualizer(args)
        
        self.y_index = 0

    def extra_init(self, reference_dl):

        self.reference_dl = reference_dl

    def fnet_single_hvp(self, x, y_index = -1):

        # Torch 2.0 / CUDA 11.7
        # def foo(params):
        #     return torch.func.functional_call(self, params, (x,))[0, y_index]

        # Torch 1.13 / FuncTorch
        def foo(params):

            result = self.fnet(params, self.fbuffer, x)[0]

            resolved_y_index = y_index
            if resolved_y_index < 0:
                resolved_y_index = torch.argmax(result)

            return result[resolved_y_index]
        
        return foo

    def on_validation_epoch_start(self):

        torch.set_grad_enabled(True)

        # 1. Extract parameters and make functional version of the network

        # Torch 1.13 / FuncTorch
        funcresult = make_functional_with_buffers(nn.Sequential(self.net, self.head))
        self.fnet, _, self.fbuffer = funcresult
        self.fparams = dict(self.combined_net.named_parameters())

        # Torch 2.0 / CUDA 11.7
        # self.fparams = dict(self.combined_net.named_parameters())

        self.hvpResults = DropoutHessianRecorder()

        # Torch 2.0 / CUDA 11.7
        # def fnet_single(params, x):
        #     return torch.func.functional_call(self, params, (x,))

        # print("Created functionalized network")

        # LeNet-5 parameters:
        # ipdb> print("\n".join([repr((p[0], p[1].shape)) for p in self.fparams.items()]))
        # ('0.0.0.weight', torch.Size([6, 1, 5, 5]))
        # ('0.0.0.bias', torch.Size([6]))
        # ('0.0.3.weight', torch.Size([16, 6, 5, 5]))
        # ('0.0.3.bias', torch.Size([16]))
        # ('0.2.0.weight', torch.Size([120, 400]))
        # ('0.2.0.bias', torch.Size([120]))
        # ('0.2.2.weight', torch.Size([84, 120]))
        # ('0.2.2.bias', torch.Size([84]))
        # ('1.weight', torch.Size([10, 84]))
        # ('1.bias', torch.Size([10]))

        # Loop thru all reference data-points
        for batch in self.reference_dl:
            
            i, x, y, o = batch
            x = x.to(self.device)

            # Feed-forward
            # Generate dropout mask
            # masked_parameters = apply_dropout(self.fparams)
            masked_parameters = params_apply_p_minus_1_dropout(self.fparams, 0.5)
            # TODO: Use dropout-p from arguments

            # 2. hvp and accumulate running variance / mean

            # Torch 2.0 / CUDA 11.7
            # hvp_result = torch.func.vhp(fnet_single_vhp(x, 0), self.fparams, masked_parameters).t()
            # hvp_result = torch.func.vjp(torch.func.grad(fnet_single_hvp(x, self.y_index)), self.fparams)[1](masked_parameters).t()

            # self.hvpResults.record(hvp_result)

            # Torch 1.13 / FuncTorch
            hvp_result = jvp(grad(self.fnet_single_hvp(x, self.y_index)), (to_unnamed(self.fparams),), (to_unnamed(masked_parameters),))[0]

            # TODO: Multiply with D (:= \Theta^{-1}(x, X) ∂f L(X), i.e., NTK-normalized training direction)
            # or simply unnormalized might be enough

            self.hvpResults.record(unnamed_tuple_to_named(self.fparams, hvp_result))

            # print("-", end = '')

        # Store the running var / mean
        # The results are stored in self.hvpResults and is accessible via self.hvpResults.mean() / self.hvpResults.variance().

        # breakpoint()

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        torch.set_grad_enabled(False)

    def forward(self, x):

        if self.training:

            self.disable_dropout(self.net)
            self.disable_dropout(self.head)
            logits = self.head(self.net(x))
            return logits, None

        else:

            # proxy
            self.disable_dropout(self.net)
            self.disable_dropout(self.head)

            logits = self.head(self.net(x))

            # Comment if 3b; Enable if 3a
            # self.enable_dropout(self.net)
            # self.enable_dropout(self.head)

            len_x = x.shape[0]
            full_x = x

            ex_uncertainties = torch.zeros(len_x, device = x.device)

            for j in range(len_x):

                # print("%5d S" % j, end='')

                x = full_x[j].unsqueeze(0)

                testPointStats = DropoutHessianRecorder()

                # 3a. Collect statstics at the datapoint x
                # for i in range(self.args.dropout_iters):

                #     # Torch 1.13 / FuncTorch
                #     # TODO: Change back to self.fparams (instead of masked_parameters; as dropout has already been enabled)
                #     # masked_parameters = params_apply_dropout(self.fparams, 0.5)
                #     grad_at_x = grad(self.fnet_single_hvp(x, self.y_index))(to_unnamed(self.fparams))
                #     testPointStats.record(unnamed_tuple_to_named(self.fparams, grad_at_x))

                #     # print("d", end='')

                # test_var = testPointStats.variance()
                # test_mean2 = params_pow(testPointStats.mean(), 2.0)

                # 3b. Collect non-dropout statstics at datapoint x
                grad_at_x = grad(self.fnet_single_hvp(x, self.y_index))(to_unnamed(self.fparams))
                grad_tuple = unnamed_tuple_to_named(self.fparams, grad_at_x)
                
                test_var = grad_tuple
                test_mean2 = grad_tuple

                # 4. Dot-product the hvp var / mean with regular gradients @ evaluating data-points
                hvp_var = self.hvpResults.variance()
                hvp_mean2 = params_pow(self.hvpResults.mean(), 2.0)

                # print("P", end='')

                # Precision seems super rough ...
                Tvar_Hvar = params_sum(params_multiply(test_var, hvp_var))
                Tvar_Hepc = params_sum(params_multiply(test_var, hvp_mean2))
                Tepc_Hvar = params_sum(params_multiply(test_mean2, hvp_var))

                Tvar_L0   = params_l0norm(test_var)
                Tvar_norm = params_l1norm(test_var)
                Tepc_norm = params_l1norm(test_mean2)

                # Visualization
                # TODO: Visualization module ?
                # TODO: FIXME:  Visalization is wrong. The r.v. is a dropout mask, then we need to plot (grad for parameter #j, hessian for parameter #j)
                #               for this particular mask, which is a loop thru entire dataset per mask.

                # if self.val_index is not None:
                    # self.indep_test_vis.visualize(self.self.val_index[j], x, grad_tuple, self.hvpResults.mean())

                self.log("Tvar_Hvar", Tvar_Hvar, on_epoch = True)
                self.log("Tvar_Hepc", Tvar_Hepc, on_epoch = True)
                self.log("Tepc_Hvar", Tepc_Hvar, on_epoch = True)

                self.log("Tvar_norm", Tvar_norm, on_epoch = True)
                self.log("Tepc_norm", Tepc_norm, on_epoch = True)

                self.log("Tvar_L0", Tvar_L0, on_epoch = True)

                extra_hvp_uncertainty =\
                    Tvar_norm
                    # Tvar_Hvar +\
                    # Tvar_Hepc +\
                    # Tepc_Hvar
                
                ex_uncertainties[j] = extra_hvp_uncertainty.detach()

                # print(".")

            return logits, ex_uncertainties

class TestTimeOnly_HessianVariance_Negate(TestTimeOnly_HessianVariance):

    def __init__(self, args, input_shape, output_dim = 10):
        
        super().__init__(args, input_shape, output_dim)
    
    def forward(self, x):

        l, u = super().forward(x)

        if u is not None:
            u = -u
        
        return l, u

# from laplace import Laplace

# Create a custom dataset for the training set
class PickTwoDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, idxA, idxB):
        self.dataset = dataset
        self.idxA = idxA
        self.idxB = idxB

    def __getitem__(self, index):
        d = self.dataset[index]
        if type(d) is tuple:
            return d[self.idxA], d[self.idxB]
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.dataset)

class LaplaceRedux(SimpleModel):

    def extra_init(self, reference_dl):

        self.reference_dl = torch.utils.data.DataLoader(
            PickTwoDataset(reference_dl.dataset, 1, 2),
            batch_size = 16,
            shuffle = False,
            num_workers = 0
        )

    def forward(self, x):

        if self.la is not None:
            pred = self.la(x, link_approx='probit')
        else:
            print("[WARNING] LaplaceRedux in running without LA (currently using --model=default as a fallback)!")
            logits = self.head(self.net(x))
            pred = F.softmax(logits, dim = 1)

        entropy = -torch.sum(pred_prob * torch.log(pred_prob + 1e-8), dim = 1)

        return logits, entropy

    def on_validation_epoch_start(self):

        torch.set_grad_enabled(True)

        # Assemble net & head
        model = nn.Sequential(self.net, self.head)

        # Fit the LA with self.reference_dl
        self.la = Laplace(model, 'classification',
             subset_of_weights='all',
             hessian_structure='diag')
        self.la.fit(self.reference_dl)

        # Perhaps we don't optimize the prior here
        # la.optimize_prior_precision(method='CV', val_loader=val_loader)

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        torch.set_grad_enabled(False)

models_dict =\
{
    "default": SimpleModel,
    "random": RandomModel,
    "mcdropout": MCDropoutModel,
    "tt_approx": TestTimeOnly_ApproximateDropoutModel,
    "naive-ensemble": NaiveEnsembleModel,
    "naive-ensemble-sum": NaiveEnsembleSummationModel,
    "naive-ensemble-sum-head": NaiveEnsembleSummationModel_HeadOnly,
    "kernel-ensemble": NaiveEnsembleModel_KernelNoiseOnly,
    "kernel-ensemble-sum": NaiveEnsembleSummationModel_KernelNoiseOnly,
    "naive-dropout": NaiveDropoutModel,
    "train-diff": TestTimeOnly_GroundTruthInit_PlainTrainingDifference,

    "wasserstein-GTinit": TestTimeOnly_GroundTruthInit_WassersteinModel,
    "wasserstein-GaussianInit": TestTimeOnly_GaussianInit_WassersteinModel,

    "velocity-std": TestTimeOnly_GroundTruthInit_VelocityModel,

    "hessian-variance": TestTimeOnly_HessianVariance,
    "hessian-variance-negate": TestTimeOnly_HessianVariance_Negate,

    "plain-ntk": TestTimeOnly_NTK,

    "plain-ntk-withNTKinv": TestTimeOnly_NTK_withInv,
    "plain-ntk-init": TestTimeOnly_NTK_initialization,
    "plain-ntk-zero-init": TestTimeOnly_NTK_zero_initialization,
    "plain-ntk-init-cls": TestTimeOnly_NTK_initialization_classification,
    "plain-ntk-zero-init-cls": TestTimeOnly_NTK_zero_initialization_classification,
    "plain-ntk-zero-init-cls-nosmax": TestTimeOnly_NTK_zero_initialization_classification_nosoftmax,
    "plain-ntk-zero-init-mean": TestTimeOnly_NTK_zero_initialization_mean,

    "plain-ntk-zero-init-simple-multidim": TestTimeOnly_NTK_zero_initialization_simple_multidim,

    "la": LaplaceRedux,

    "inject-test": uq.InjectTest,
}

#################################################################

def has_func(obj, func_name):
    return hasattr(obj, func_name) and callable(getattr(obj, func_name))

def main(hparams):

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

    seed = hparams.seed * (hparams.runid + 1)
    pl.seed_everything(seed)
    
    main_datamodule, input_dim = get_data_module(
        hparams.dataset,
        hparams.batch_size,
        data_augmentation = (hparams.dataAug > 0),
        num_workers=hparams.num_workers,
        do_partial_train = hparams.do_partial_train,
        do_contamination = hparams.do_contamination,
        test_set_max = hparams.test_set_max,
        is_binary = hparams.binary,
        noise_std = hparams.noise,
        blur_sigma = hparams.blur)
    main_datamodule.setup()

    # Init our model
    model = models_dict[hparams.model](hparams, input_dim, main_datamodule.n_classes if not hparams.binary else 1)

    # Dirty workaround
    if has_func(model, "extra_init"):

        if hparams.contaminate_ref:
            ref_data = torch.utils.data.Subset(
                main_datamodule.test_dataset,
                np.random.choice(len(main_datamodule.test_dataset), size = (hparams.reference_data_count,))
            )
        else:
            ref_data = torch.utils.data.Subset(
                main_datamodule.train_dataset,
                np.random.choice(len(main_datamodule.train_dataset), size = (hparams.reference_data_count,))
            )

        model.extra_init(
            torch.utils.data.DataLoader(
                ref_data,
                batch_size = 1,
                shuffle = False,
                num_workers = 0
            )
        )

    # Initialize a trainer
    trainer = Trainer.from_argparse_args(
        hparams,
        gpus=1,
        max_epochs=hparams.epochs,
        enable_checkpointing=False,

        # limit_val_batches = 0.0,

        logger = wandb_logger
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
    parser.add_argument("--do_partial_train", type=int, default=0)
    parser.add_argument("--use_full_trainset", type=int, default=1)
    parser.add_argument("--do_contamination", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--test_set_max", type=int, default=-1)

    parser.add_argument("--binary", type=int, default=0, help="Convert the problem to a binary classification. Splits the dataset into halves.")
    
    # Model
    parser.add_argument("--hidden_dim", type=int, default=2048)
    parser.add_argument("--dropout_rate", type=float, default=0.85)
    parser.add_argument("--dropout_iters", type=int, default=10)
    parser.add_argument("--lazy_scaling", type=float, default=1)
    parser.add_argument("--pointwise_linearization", type=int, default=1)

    parser.add_argument("--reference_data_count", type=int, default=64)
    parser.add_argument("--random_multidim", type=int, default=1)
    parser.add_argument("--num_multidim", type=int, default=32)

    parser.add_argument("--perturb_power", type=float, default=-1, help = "Overrides perturb_min / max if set to value above 0")
    parser.add_argument("--perturb_min", type=float, default=0.1, help = "Perturb noise norm for 1st layer")
    parser.add_argument("--perturb_max", type=float, default=0.1, help = "Perturb noise norm for last layer")
    parser.add_argument("--perturb_nonlinear", type=float, default=0.0, help = "Perturb noise norm curve nonlinearity; >0 => more change towards last layer | <0 => more change towards first layer")

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

    # if args.no_train:
        # args.batch_size = 1

    print("Args: %s" % vars(args))

    main(args)
    
