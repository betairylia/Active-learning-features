
import torch
from torch import nn
from torch.nn import functional as F

from ntk_utils import NTKHelper
from .base import SimpleModel

class NTKTest(SimpleModel):
    
    def __init__(self, args, dm_header, l, m):
        
        super().__init__(args, dm_header, l, m)

        self.ntk = NTKHelper(self.net)

    def on_validation_epoch_start(self):
        self.ntk.refresh(self.net)
    
    def forward(self, x):

        l, u = super().forward(x)

        if not self.training:
            self.ntk.compute_full_ntk(x, x)
        
        return l, u
