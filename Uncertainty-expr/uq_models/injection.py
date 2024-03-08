import torch
from .base import SimpleModel

from .param_inject import *

class InjectTest(SimpleModel):

    def __init__(self, args, input_shape, output_dim = 10):
        
        super().__init__(args, input_shape, output_dim)

        self.combined_net = nn.Sequential(self.net, self.head)
        self.combined_net = InjectNet(self.combined_net)

        self.combined_net_init = nn.Sequential(self.net_init, self.head_init)

        breakpoint()
