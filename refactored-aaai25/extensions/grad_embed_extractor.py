import torch
from torch import nn
from torch.nn import functional as F

import lightning as L

from .commons import Extension

class GradEmbedExtractor(Extension):

    def __init__(self, args):
        super().__init__()

        self.results = []
        self.extract_rate = 0.1
        self.extract_max = 8192

    def on_train_epoch_start(self, trainer, plm):

        self.results = []
    
    def on_train_batch_start(self, trainer, plm, outputs, batch, bid, did = 0):

        pass
