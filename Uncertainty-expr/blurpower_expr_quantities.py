import torch
from pytorch_lightning import LightningModule, Trainer
from torch import nn
from torch.nn import functional as F

import numpy as np

class QuantityBase():

    def __init__(self, args):

        self.args = args

    def preprocess(self, net, head):

        return net, head

    def evaluate(self, net, head, batch):

        i, x, y, o = batch
        
        logits = head(net(x))
        pred_prob = F.softmax(logits, dim = 1)
        entropy = -torch.sum(pred_prob * torch.log(pred_prob + 1e-8), dim = 1)

        # shape of logits: [batch_size, num_classes]
        # shape of entropy: [batch_size]

        return entropy

    def summary(self, results):

        return {"Q": np.asarray(results).mean()}

Qdict =\
{
    "trivial-uq": QuantityBase,
    "injection-uq": None,
    "grad-param-prod": None,
    "cal-NTK-eval": None,
    "grad-norm": None,
}

