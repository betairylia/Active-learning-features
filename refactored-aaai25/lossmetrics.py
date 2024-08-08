import torch
from torch import nn
from torch.nn import functional as F

# TODO: torchmetrics ??
class MulticlassAccuracy():

    logger_name = 'acc'

    def __call__(self, p, y):

        return (torch.argmax(p, dim=1) == y).float().mean() 

def get_loss_and_metrics(args, dm_header):

    # TODO: out-of-ranged classes disabled?
    # y[y >= self.output_dim] = -100

    loss = None
    if args.loss == 'mse':
        if args.binary:
            loss = lambda x, y: F.mse_loss(x, y.detach().float())
        else:
            loss = lambda x, y: F.mse_loss(x, F.one_hot(y, output_dim).detach().float())
    elif args.loss == 'cent':
        if args.binary:
            loss = lambda x, y: F.binary_cross_entropy_with_logits(x, y.detach().float())
        else:
            loss = nn.CrossEntropyLoss()

    # TODO: Binary?
    metrics = MulticlassAccuracy()

    return loss, metrics
