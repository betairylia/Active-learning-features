from typing import Callable, Optional

import torch
from torch import nn

def Inject(module, *args, **kwargs):

    if isinstance(module, nn.Linear):
        return Linear_ParameterInjector(module)

    else:
        print("[Injector] Unsupported layer: %s, Ignoring!" % str(module))
        return None

def InjectNet(net, *args, **kwargs):
    # TODO
    # ref: https://github.com/baal-org/baal/blob/b9435080b7bdbd1c75722370ac833e97380d38c0/baal/bayesian/common.py#L52

    for name, child in net.named_children():
        new_module: Optional[nn.Module] = Inject(child, *args, **kwargs)

        if new_module is not None:
            new_module.train(mode = child.training)
            net.add_module(name, new_module)
    
    return net

####################################
## ↓↓ Injector implementations ↓↓ ##
####################################

class Linear_ParameterInjector(nn.Module):

    def __init__(self, moduleToWrap):
        
        super().__init__()
        self.module = moduleToWrap

        # weight
        self.weight_inject = torch.zeros_like(self.module.weight)
        self.weight_original = self.module.weight

        # bias
        self.bias_inject = torch.zeros_like(self.module.bias)
        self.bias_original = self.module.bias
    
    def forward(self, x):

        self.module.weight = self.weight_original + self.weight_inject
        self.module.bias = self.bias_original + self.bias_inject
        return self.module(x)
