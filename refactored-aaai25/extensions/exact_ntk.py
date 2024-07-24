import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics import Accuracy, AUROC, AveragePrecision

from matplotlib import pyplot as plt

# import torchvision
# from torchvision import datasets
# from torchvision.transforms import transforms
# import torchvision.transforms as transforms

import lightning as L

from ntk_utils import NTKHelper
import scipy
import utils
import wandb

class ExactNTKComputation(L.Callback):

    def __init__(self, args):
        super().__init__()
        self.args = args

        self.batchsize = args.ntk_batchsize
        self.downsample_by = 30
        self.per_epochs = 10
        self.pre_epochs = 5

        # UQ Metrics
        self.uncertainty_auroc_inf = AveragePrecision(task="binary")
        self.uncertainty_auroc_avg = AveragePrecision(task="binary")
        self.uncertainty_acc = utils.BestAccuracySweep()

    def setup(self, trainer, plm, stage):
        self.ntk = NTKHelper(plm.net)
        self.ref_dl = torch.utils.data.DataLoader(
            plm.ref_data,
            batch_size = self.batchsize,
            shuffle = False,
            num_workers = 0
        )
    
    def check_epoch(self, ep):
        return (ep > self.pre_epochs and ep % self.per_epochs == 0) or (ep <= self.pre_epochs)

    def on_validation_epoch_start(self, trainer, plm):
        
        if not self.check_epoch(trainer.current_epoch):
            return

        # Result containers
        self.inf_bounds = []
        self.avg_bounds = []
        self.ood_labels = []

        # Refresh NTK Helper to use latest weights
        self.ntk.refresh(plm.net)

        self.Oxx = self.ntk.compute_ntk(
            self.ref_dl, self.ref_dl,
            batch_mode = '1to1',
            x1_map = lambda batch : batch[1], # i, x, y, o
            x2_map = lambda batch : batch[1], 
        )
    
    def on_validation_batch_end(self, trainer, plm, outputs, batch, bid, did = 0):

        if not self.check_epoch(trainer.current_epoch):
            return

        # Downsample the # of datapoints by self.downsample_by
        if bid % self.downsample_by != 0:
            return

        i, x, y, o = batch

        self.Ozz = self.ntk.compute_ntk(
            x, x,
            batch_mode = '1to1',
        )

        self.Ozx = self.ntk.compute_ntk(
            x, self.ref_dl,
            x2_map = lambda batch : batch[1], # i, x, y, o
        )

        # Oxx: Tensor[|X|]
        # Ozz: Tensor[|Z|]
        # Ozx: Tensor[|Z|, |X|]

        inf_bound = self.Ozz + torch.min(self.Oxx.unsqueeze(0) - 2 * self.Ozx, dim = 1)[0]
        avg_bound = self.Ozz + torch.mean(self.Oxx.unsqueeze(0), dim = 1) - 2 * torch.mean(torch.abs(self.Ozx), dim = 1)

        self.inf_bounds.append(inf_bound.detach().cpu())
        self.avg_bounds.append(avg_bound.detach().cpu())
        self.ood_labels.append(o.cpu())
    
    def on_validation_epoch_end(self, trainer, plm):

        if not self.check_epoch(trainer.current_epoch):
            return

        self.inf_bounds = torch.cat(self.inf_bounds)
        self.avg_bounds = torch.cat(self.avg_bounds)
        self.ood_labels = torch.cat(self.ood_labels).squeeze()

        uacc_inf, uacc_th_inf = self.uncertainty_acc(self.inf_bounds, self.ood_labels)
        uacc_avg, uacc_th_avg = self.uncertainty_acc(self.avg_bounds, self.ood_labels)

        uauroc_inf = self.uncertainty_auroc_inf(self.inf_bounds, self.ood_labels)
        uauroc_avg = self.uncertainty_auroc_avg(self.avg_bounds, self.ood_labels)

        infavgR = scipy.stats.pearsonr(self.inf_bounds, self.avg_bounds).statistic

        plm.log("val_NTKinf_uAcc", uacc_inf, prog_bar = False)
        plm.log("val_NTKavg_uAcc", uacc_avg, prog_bar = False)
        plm.log("val_NTKinf_uAUROC", uauroc_inf, prog_bar = False)
        plm.log("val_NTKavg_uAUROC", uauroc_avg, prog_bar = False)
        plm.log("val_NTK_inf-avg_R", infavgR, prog_bar = False)

        fig, ax = plt.subplots()
        ax.scatter(self.inf_bounds, self.avg_bounds)
        ax.set_xlabel("infimum")
        ax.set_ylabel("average-abs")
        ax.set_title("R: %f" % infavgR)
        wandb.log({"Empirical NTK - inf bound (X) vs. avg-abs bound (Y)": wandb.Image(fig)})
        plt.close('all')
