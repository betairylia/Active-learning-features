from typing import Callable, Optional

import torch
from torch import nn

import math
import wandb

import utils

def Inject(module, module_init, *args, **kwargs):

    if isinstance(module, nn.Linear):
        utils.log("[Injector] Supported layer: nn.Linear")
        return Linear_ParameterInjector(module, module_init, *args, **kwargs)

    elif isinstance(module, nn.Conv2d):
        utils.log("[Injector] Supported layer: nn.Conv2d")
        return Linear_ParameterInjector(module, module_init, *args, **kwargs)

    else:
        utils.log("[Injector] Unsupported layer: %s, Ignoring!" % module.__class__.__name__)
        return None

# In-place modification that injects noise perturbation functions to `net`
def InjectNet(
        net,
        net_init = None,
        depth = 0,
        layers = [],

        perturb_adaptive = 'none',

        perturb_nonlinear = 0.0,
        perturb_min = 0.1,
        perturb_max = 0.1,
        *args, **kwargs):
    # TODO
    # ref: https://github.com/baal-org/baal/blob/b9435080b7bdbd1c75722370ac833e97380d38c0/baal/bayesian/common.py#L52

    if depth == 0:
        layers = []

    if depth > 100:
        utils.log("[INJECTOR] MAX RECURSION DEPTH")
        return False

    for name, child in net.named_children():
        utils.log("[INJECTOR] Current layer: %s" % name)
        child_init = net_init.get_submodule(name) if net_init is not None else None
        new_module: Optional[nn.Module] = Inject(child, child_init, *args, **kwargs)
        
        if new_module is not None:

            new_module.train(mode = child.training)
            net.add_module(name, new_module)
            
            # Maintain a list of all injected layers, with their order in `net.named_children()`
            layers.append(new_module)
    
        # Do it recursively
        InjectNet(child, child_init, depth+1, layers, *args, **kwargs)

    if depth == 0:
        utils.log(net)
        utils.log(len(layers))
        
        layer_norms = []
        if perturb_adaptive == 'none':
            layer_norms = [
                (((i / (len(layers) - 1)) if len(layers) > 1 else 1) ** math.exp(perturb_nonlinear)) * (perturb_max - perturb_min) + perturb_min
                for i in range(len(layers))
            ]
        
        elif perturb_adaptive == 'inv-param-norm':
            param_norm = torch.Tensor([l.get_param_norm() for l in layers])
            raw_param_norm = param_norm
            # param_norm = 1 / param_norm
            param_norm = param_norm / param_norm.max()
            layer_norms = param_norm * perturb_max

            utils.log("Adaptive Layer-wise scaling info")
            utils.log("%5s %12s %12s" % ("No.", "param_norm", "perturb_norm"))
            utils.log("\n".join(["%5d %12.7f %12.7f" % (i, p, l) for i, (l, p) in enumerate(zip(layer_norms, raw_param_norm))]))
        
        # Apply layer-wise scaling
        for i, layer in enumerate(layers):
            layer.set_norm(
                layer_norms[i]
            )
            # breakpoint()
            # wandb.log({"noise_norm": layer.noise_norm, "layer_index": i})

    return layers

def get_states(module):
    states = []
    for each_module in module.modules():
        if isinstance(each_module, ParameterInjector):
            states.append(each_module.get_state())
    return states

def set_states(module, states):
    current_index = 0
    for each_module in module.modules():
        if isinstance(each_module, ParameterInjector):
            # TODO: Rename
            each_module.set_norm(**states[current_index])
            current_index = current_index + 1

def set_perturb_norm(module, *args, **kwargs):
    for each_module in module.modules():
        if isinstance(each_module, ParameterInjector):
            each_module.set_norm(*args, **kwargs)

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

    def get_state(self):
        pass

    def set_norm(self, noise_norm):
        pass

    def enable(self, *args, **kwargs):
        pass
    
    def disable(self, *args, **kwargs):
        pass
    
    def sample(self, *args, **kwargs):
        pass
    
    def get_param_norm(self, *args, **kwargs):
        return 0

class Linear_ParameterInjector(ParameterInjector):

    '''
    noise_norm: float
    noise_pattern: str | 'prop', 'indep', 'inv', 'subtract'
    '''
    def __init__(self, moduleToWrap, moduleToWrap_init, *args, **kwargs):
        
        super().__init__()
        self.module = moduleToWrap
        self.module_init = moduleToWrap_init
        
        self.noise_norm = 0.1
        if 'noise_norm' in kwargs:
            self.noise_norm = kwargs['noise_norm']

        self.noise_pattern = 'prop'
        if 'noise_pattern' in kwargs:
            self.noise_pattern = kwargs['noise_pattern']

        self.noise_norm_ex = 1.0 
        if 'noise_norm_ex' in kwargs:
            self.noise_norm_ex = kwargs['noise_norm_ex']

        # weight
        self.weight_inject = torch.zeros_like(self.module.weight)
        self.weight_original = self.module.weight

        # bias
        if self.module.bias is not None:
            self.bias_inject = torch.zeros_like(self.module.bias)
            self.bias_original = self.module.bias

        self.enabled = False

    def get_state(self):
        return {
            "noise_norm": self.noise_norm,
            "noise_pattern": self.noise_pattern,
            "noise_norm_ex": self.noise_norm_ex
        }

    def set_norm(self, noise_norm, noise_pattern = None, noise_norm_ex = None):

        if noise_norm is not None:
            self.noise_norm = noise_norm

        if noise_pattern is not None:
            self.noise_pattern = noise_pattern

        if noise_norm_ex is not None:
            self.noise_norm_ex = noise_norm_ex

    def enable(self, *args, **kwargs):
        # utils.log("Enabled Linear_PI")
        self.enabled = True
    
    def disable(self, *args, **kwargs):
        # utils.log("Disabled Linear_PI")
        self.enabled = False
    
    def sample(self, *args, **kwargs):
        # self.weight_inject = torch.normal(
        #     mean = torch.ones_like(self.module.weight),
        #     std = self.module.weight * 0.1,
        #     device = self.module.weight.device)
        
        self.weight_inject = torch.randn(*self.module.weight.shape, device = self.module.weight.device)

        if self.noise_pattern == 'prop':
            self.weight_inject = self.noise_norm * self.weight_inject * torch.abs(self.module.weight)

        elif self.noise_pattern == 'indep':
            self.weight_inject = self.noise_norm * self.weight_inject 

        elif self.noise_pattern == 'inv':
            self.weight_inject = self.noise_norm * self.weight_inject * self.module.weight

        elif self.noise_pattern == 'subtract':
            self.weight_inject = (self.noise_norm - self.noise_norm * self.noise_norm_ex * torch.abs(self.module.weight)) * self.weight_inject

        elif self.noise_pattern == 'prop-deterministic':
            if self.module_init is not None:
                self.weight_inject = self.noise_norm * self.noise_norm_ex * (self.module.weight - self.module_init.weight)
                # self.weight_inject = self.noise_norm * self.module.weight 
            else:
                self.weight_inject = self.noise_norm * self.noise_norm_ex * self.module.weight 

        # TODO: bias

    def get_param_norm(self, *args, **kwargs):
        if self.module_init is not None:
            utils.log("module_init is not None")
            return (self.module.weight - self.module_init.weight).abs().mean()
        else:
            # return self.module.weight.abs().mean()
            return self.module.weight.norm() / math.sqrt(torch.numel(self.module.weight))

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

