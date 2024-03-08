import torch
from torch import nn

def Inject(module):

    if isinstance(module, nn.Linear):
        return Linear_ParameterInjector(module)

    else:
        print("[Injector] Unsupported layer: %s, Ignoring!" % str(module))

def InjectNet(net):
    # TODO

    breakpoint() 

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
