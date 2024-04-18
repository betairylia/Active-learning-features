import torch
from pytorch_lightning import LightningModule, Trainer
from torch import nn
from torch.nn import functional as F

import numpy as np

from uq_models.param_inject import *
from func_dropout import *

from tqdm import tqdm

class QuantityBase():

    def __init__(self, args):

        self.args = args
        self.q_eval_ref = 1.0

    def preprocess(self, net, head):

        return net, head

    def evaluate(self, net, head, batch):

        i, x, y, o = batch
        
        logits = head(net(x))
        pred_prob = F.softmax(logits, dim = 1)
        entropy = -torch.sum(pred_prob * torch.log(pred_prob + 1e-8), dim = 1)

        # shape of logits: [batch_size, num_classes]
        # shape of entropy: [batch_size]

        return entropy.detach().cpu()

    def summary(self, is_ref, results):

        q_eval = torch.cat(results).mean()
        if is_ref:
            self.q_eval_ref = q_eval

        return {"Q": q_eval, "Q (Normalized)": q_eval / self.q_eval_ref}

def VisualizeHessianBound(gradResults):
    
    means = gradResults.mean()
    varis = gradResults.variance()

    print("Layer keys:")
    print([k for k in means])

    all_data = []
    for i, k in enumerate(means):
        data = {"Hessian-mean-mean": means[k].mean(), "Hessian-mean-norm": means[k].norm(), "Hessian-var-mean": varis[k].mean(), "Hessian-var-norm": varis[k].norm(), "param_index": i, "param_name": k}
        all_data.append(data)
        wandb.log(data)

    table = wandb.Table(columns = list(all_data[0].keys()), data = [list(d.values()) for d in all_data])
    wandb.log({"Hessian-bound-table": table})

class HessianBoundCalculator(QuantityBase):

    def fnet_single_hvp(self, x, y_index = 0):

        # Torch 2.0 / CUDA 11.7
        def foo(params):
            return torch.func.functional_call(self.combined_net, params, (x,))[0, y_index]
        
        return foo

    def preprocess(self, net, head):
        
        # self.perturb_power = self.args.perturb_power 
        
        # InjectNet(net, noise_norm = self.perturb_power)
        # InjectNet(head, noise_norm = self.perturb_power)

        self.combined_net = nn.Sequential(net, head)
        self.combined_net.eval()
        self.y_index = 0

        # Calculate hessian bound
        self.fparams = dict(self.combined_net.named_parameters())
        self.gradResults = DropoutHessianRecorder()

        self.device = torch.device('cuda')

        for batch_id in tqdm(range(1024), desc = "Hessian bound"):

            # Generate data
            # x = torch.randn(self.args.batch_size, 3, 224, 224, device = self.device)
            x = torch.randn(1, 3, 224, 224, device = self.device)

            grad = torch.func.grad(self.fnet_single_hvp(x, 0))
            hvp_result = torch.func.jvp(grad, (self.fparams,), (params_randn_like(self.fparams),))[1]

            self.gradResults.record(hvp_result)

        VisualizeHessianBound(self.gradResults)

        return net, head

class InjectUncertainty(QuantityBase):

    def preprocess(self, net, head):
        
        self.perturb_power = self.args.perturb_power 
        
        InjectNet(net, noise_norm = self.perturb_power)
        InjectNet(head, noise_norm = self.perturb_power)

        return net, head
    
    def get_predictions(self, net, head, batch):

        i, x, y, o = batch
        
        resample_perturb(net)
        resample_perturb(head)

        enable_perturb(net)
        enable_perturb(head)

        logits = []
        probs = []

        for i in range(self.args.dropout_iters):

            resample_perturb(net)
            resample_perturb(head)

            logits.append(head(net(x)).detach())
            pred_prob = F.softmax(logits[-1], dim = 1)
            probs.append(pred_prob)

        logits = torch.stack(logits, dim = 0)
        probs = torch.stack(probs, dim = 0)

        return logits, probs

    def evaluate(self, net, head, batch):

        logits, probs = self.get_predictions(net, head, batch)

        model_prediction = probs.mean(0)
        entropy = -torch.sum(model_prediction * torch.log(model_prediction + 1e-8), dim = 1)

        return entropy.detach().cpu()

class InjectFluctuation(InjectUncertainty):

    def evlauate(self, net, head, batch):

        logits, probs = self.get_predictions(net, head, batch)

        fluctuation = logits.std(dim = 0)

        return fluctuation.detach().cpu()

class InjectFluctuation_normalized(InjectUncertainty):

    def evlauate(self, net, head, batch):

        logits, probs = self.get_predictions(net, head, batch)

        # Pop state
        cache = (get_states(net), get_states(head))

        # Switch to indep perturbation
        set_perturb_norm(net, noise_norm = 0.001, noise_pattern = 'indep')
        set_perturb_norm(head, noise_norm = 0.001, noise_pattern = 'indep')

        # Obtain indep result
        logits_indep, probs_indep = self.get_predictions(net, head, batch)

        # Push state
        set_states(net, cache[0])
        set_states(head, cache[1])

        fluctuation_normed = logits.std(dim = 0) / logits_indep.std(dim = 0)

        return fluctuation.detach().cpu()

Qdict =\
{
    "trivial-uq": QuantityBase,
    "injection-uq": InjectUncertainty,
    "injection-fluc": InjectFluctuation,
    "injection-fluc-normed": InjectFluctuation_normalized,
    "grad-param-prod": None,
    "cal-NTK-eval": None,
    "grad-norm": None,
    
    "hessian-bound": HessianBoundCalculator,
}

