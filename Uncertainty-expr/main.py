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

from data_uncertainty import MNIST_UncertaintyDM, CIFAR10_UncertaintyDM, SVHN_UncertaintyDM #, FashionMNIST_UncertaintyDM
from recorder import Recorder

from nets import net_dict

import math
# from resnet import resnet18
# from weight_drop import *

import copy
import ot

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
    use_full_trainset = True):
    
    args = {
        "data_dir": data_dir,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "do_partial_train": do_partial_train,
        "do_contamination": do_contamination,
        "use_full_trainset": use_full_trainset
    }

    if dataset_name == 'mnist':
        main_dm = MNIST_UncertaintyDM(**args)
    
    elif dataset_name == 'cifar10':
        main_dm = CIFAR10_UncertaintyDM(**args)
    
    elif dataset_name == 'svhn':
        main_dm = SVHN_UncertaintyDM(**args)
    
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
        self.visualized = False

        self.net_factory = net_dict[args.net]()
        self.net, self.head = self.net_factory.getNets(
            input_shape, 
            [output_dim],
            hidden_dim = args.hidden_dim,
            dropout_rate = args.dropout_rate
        )
        
        if args.loss == 'mse':
            self.loss = lambda x, y: F.mse_loss(x, F.one_hot(y, output_dim).detach().float())
        elif args.loss == 'cent':
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
        x, y, o = batch

        logits = self.scale_output(self(x)[0], x)

        # loss = F.cross_entropy(self(x), y)
        loss = self.loss(logits, y)

        self.log("train_loss", loss)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y, o = batch
        # logits = self(x)
        # loss = F.cross_entropy(logits, y)
        logits, uncertainty = self(x)
        logits = self.scale_output(logits, x)

        y[y >= self.output_dim] = -100
        loss = self.loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        acc_seen = self.accuracy_seen(preds, y, o)

        if not self.visualized:
            self.visualized = True
            self.logger.log_image(key = "test-set", images = [ImageMosaicSQ(x)], caption = ["".join(["1" if _o == 1 else "0" for _o in o.detach().cpu()])])

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

        self.val_uncertainty_scores = []
        self.val_uncertainty_labels = []

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        # Here we just reuse the on_validation_epoch_end for testing
        return self.on_validation_epoch_end()
        
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

class NaiveEnsembleModel(SimpleModel):

    def __init__(self, args, input_shape, output_dim=10, ensemble_size = 4):

        super().__init__(args, input_shape, output_dim)

        self.ensemble_size = ensemble_size
        self.ensemble_reduce = 'mean'
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

models_dict =\
{
    "default": SimpleModel,
    "mcdropout": MCDropoutModel,
    "tt_approx": TestTimeOnly_ApproximateDropoutModel,
    "naive-ensemble": NaiveEnsembleModel,
    "naive-dropout": NaiveDropoutModel,
    "train-diff": TestTimeOnly_GroundTruthInit_PlainTrainingDifference,

    "wasserstein-GTinit": TestTimeOnly_GroundTruthInit_WassersteinModel,
    "wasserstein-GaussianInit": TestTimeOnly_GaussianInit_WassersteinModel,
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
    parser.add_argument("--do_partial_train", type=int, default=0)
    parser.add_argument("--use_full_trainset", type=int, default=1)
    parser.add_argument("--do_contamination", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=16)
    
    # Model
    parser.add_argument("--hidden_dim", type=int, default=2048)
    parser.add_argument("--dropout_rate", type=float, default=0.85)
    parser.add_argument("--dropout_iters", type=int, default=10)
    parser.add_argument("--lazy_scaling", type=float, default=1)

    # Training
    parser.add_argument('--dataAug', type=int, default=0, help="Data augmentation on(1) / off(0).")
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

    print("Args: %s" % vars(args))

    main(args)
    
