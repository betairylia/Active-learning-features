from typing import Callable, Optional

import torch
from torch import nn

def Inject(module, *args, **kwargs):

    if isinstance(module, nn.Linear):
        print("[Injector] Supported layer: nn.Linear")
        return Linear_ParameterInjector(module)

    elif isinstance(module, nn.Conv2d):
        print("[Injector] Supported layer: nn.Conv2d")
        return Linear_ParameterInjector(module)

    else:
        print("[Injector] Unsupported layer: %s, Ignoring!" % module.__class__.__name__)
        return None

def InjectNet(net, depth = 0, *args, **kwargs):
    # TODO
    # ref: https://github.com/baal-org/baal/blob/b9435080b7bdbd1c75722370ac833e97380d38c0/baal/bayesian/common.py#L52

    if depth > 100:
        print("[INJECTOR] MAX RECURSION DEPTH")
        return False

    for name, child in net.named_children():
        new_module: Optional[nn.Module] = Inject(child, *args, **kwargs)
        
        if new_module is not None:
            new_module.train(mode = child.training)
            net.add_module(name, new_module)
    
        # Do it recursively
        InjectNet(child, depth+1, *args, **kwargs)

    if depth == 0:
        print(net)

    return True

def enable_perturb(module, *args, **kwargs):
    for each_module in module.modules():
        if isinstance(each_module, ParameterInjector):
            each_module.enable(*args, **kwargs)

def disable_perturb(module, *args, **kwargs):
    for each_module in module.modules():
        if isinstance(each_module, ParameterInjector):
            each_module.disable(*args, **kwargs)

def resample_perturb(module, *args, **kwargs):
    for each_module in module.modules():
        if isinstance(each_module, ParameterInjector):
            each_module.sample(*args, **kwargs)

####################################
## ↓↓ Injector implementations ↓↓ ##
####################################

class ParameterInjector(nn.Module):

    def enable(self, *args, **kwargs):
        pass
    
    def disable(self, *args, **kwargs):
        pass
    
    def sample(self, *args, **kwargs):
        pass

class Linear_ParameterInjector(ParameterInjector):

    def __init__(self, moduleToWrap):
        
        super().__init__()
        self.module = moduleToWrap

        # weight
        self.weight_inject = torch.zeros_like(self.module.weight)
        self.weight_original = self.module.weight

        # bias
        if self.module.bias is not None:
            self.bias_inject = torch.zeros_like(self.module.bias)
            self.bias_original = self.module.bias

        self.enabled = False

    def enable(self, *args, **kwargs):
        # print("Enabled Linear_PI")
        self.enabled = True
    
    def disable(self, *args, **kwargs):
        # print("Disabled Linear_PI")
        self.enabled = False
    
    def sample(self, *args, **kwargs):
        # self.weight_inject = torch.normal(
        #     mean = torch.ones_like(self.module.weight),
        #     std = self.module.weight * 0.1,
        #     device = self.module.weight.device)
        
        self.weight_inject = torch.randn(*self.module.weight.shape, device = self.module.weight.device)
        self.weight_inject = 0.1 * self.weight_inject * self.module.weight

        # TODO: bias

    def forward(self, x):

        if self.enabled:
            self.module.weight = nn.Parameter(self.weight_original + self.weight_inject)
            # TODO: bias
            # self.module.bias = nn.Parameter(self.bias_original + self.bias_inject)
            return self.module(x)

        else:
            self.module.weight = self.weight_original
            # self.module.bias = self.bias_original
            return self.module(x)

