import torch
from torch.nn import functional as F
from .base import SimpleModel

from .param_inject import *

class TTUQBase(SimpleModel):

    def __init__(self, args, dm_header, loss, metrics, reference_dataset = None):
        
        super().__init__(args, dm_header, loss, metrics, reference_dataset)

        self.K = args.ttuq_K
        self.lambd = args.ttuq_lambda
        self.delta = args.ttuq_delta

        self.perturb_nonlinear = args.perturb_nonlinear
        self.perturb_min = args.perturb_min
        self.perturb_max = args.perturb_max
        self.noise_pattern = args.noise_pattern

        self.inject(args)
    
    def inject(self, args):

        InjectNet(
            self.net,
            self.net_init,
            perturb_nonlinear = self.perturb_nonlinear,
            perturb_min = self.perturb_min,
            perturb_max = self.perturb_max,
            noise_pattern = self.noise_pattern
        )

    def get_predictions(self, x, times = -1):

        if times <= 0:
            times = self.args.dropout_iters # TODO: Change my name?

        resample_perturb(self.net)
        enable_perturb(self.net)
        logits = []

        for i in range(times):
            logits.append(self.net(x))
            resample_perturb(self.net)

        logits = torch.stack(logits, dim = 0)

        return logits
    
    def entropy(self, logits):

        probs = F.softmax(logits, dim = -1)
        model_prediction = probs.mean(0)
        entropy = -torch.sum(model_prediction * torch.log(model_prediction + 1e-8), dim = 1)
        return entropy

    def eval_forward(self, x):

        logits = self.get_predictions(x)

        if abs(self.lambd - 1) > 1e-2:

            # Pop state
            cache = get_states(self.net)

            # Compute original logits
            set_perturb_norm(self.net, noise_norm = 0, noise_pattern = 'prop-deterministic')
            logits_original = self.get_predictions(x, times = 1)

            # Push state
            set_states(self.net, cache)

            logits_diff = logits - logits_original
            logits_scaled_diff = logits_diff * self.lambd
            logits_new = logits_original + logits_scaled_diff

            logits = logits_new

        uncertainty = self.entropy(logits)

        return logits.mean(dim = 0), uncertainty

    def forward(self, x):

        if self.training:
            disable_perturb(self.net)
            logits = self.net(x)
            return logits, None
        
        else:

            logits, uncertainty = self.eval_forward(x)
            return logits, uncertainty

class TTUQAdaptiveScaling(TTUQBase):

    def inject(self, args):
        
        # perturb norm should be approx. 15% of weight norm (L1)

        all_layers = InjectNet(
            self.net,
            self.net_init,
            perturb_nonlinear = 0.0,
            perturb_min = self.args.perturb_power,
            perturb_max = self.args.perturb_power,
            perturb_adaptive = 'inv-param-norm',
            noise_pattern = self.noise_pattern
        )

class TTUQComplete(TTUQAdaptiveScaling):

    def get_uncertainty_from_ub(self, x, logits, logits_perturbed, Ozz, grad_param_product):

        return Ozz - grad_param_product

    def eval_forward(self, x):

        logits = self.get_predictions(x)

        # Pop state
        cache = get_states(self.net)

        # Compute <g,p>
        set_perturb_norm(self.net, noise_norm = None, noise_norm_ex = self.delta, noise_pattern = 'prop-deterministic')
        logits_det = self.get_predictions(x, times = 1) # [1, bs, outdim]

        # Compute original logits
        set_perturb_norm(self.net, noise_norm = 0, noise_pattern = 'prop-deterministic')
        logits_original = self.get_predictions(x, times = 1) # [1, bs, outdim]

        # Push state
        set_states(self.net, cache)

        det_diff = logits_det - logits_original
        Ozz = logits.std(dim = 0).sum(dim = -1)
        Oxz = self.K * torch.norm(det_diff, dim = -1).squeeze()

        # breakpoint()

        self.log("Ozz value", Ozz.mean())
        self.log("gpp value", Oxz.mean())

        uq = self.get_uncertainty_from_ub(x, logits_original.squeeze(0), logits, Ozz, Oxz)

        # Classification accuracy of logits_det
        # delta =  0.0 | 75.932
        # delta =  2.0 | 75.098
        # delta = 10.0 | 67.866

        # return logits_det.squeeze(0), uq
        # return logits_original.squeeze(0), uq
        return self.ref_l.mean(dim = 0), uq

class TTUQCompletePosterior(TTUQComplete):

    def get_uncertainty_from_ub(self, x, l, lp, Ozz, gpp):

        # This method provides bad posterior that is basically useless.
        # Consider scaling the original perturbed posterior? But how do we justify them?

        ub = (Ozz - gpp) * self.lambd # hparam
        self.log("UB value", ub.mean())

        ########################################################
        # Perturbed logits scaling (Seems good)
        # test subset with 8192 samples, AUROC
        ##############
        # PARAMS
        # net           resnet50-imagenet
        # noise_pattern indep
        # perturb_power 0.018
        # delta         2.0
        # lambda        0.005
        ##############
        # RESULTS
        # With GPP term: 97.05
        # Without GPP term: 96.5x
        ########################################################

        logit_differences = lp - l
        logit_differences =\
            torch.sign(logit_differences) *\
            torch.maximum(
                torch.abs(logit_differences) -
                gpp[None, :, None] * self.lambd,
                torch.zeros_like(logit_differences)
            )
        new_lp = l + logit_differences
        self.ref_l = new_lp

        self.log("logit_diff value", (lp - l).abs().mean())
        self.log("gpp value (scaled)", (gpp * self.lambd).mean())

        return self.entropy(self.ref_l) + ub

        # Code below this line has no effect but I'm too lazy to comment them out

        ########################################################
        # Score mixture (SO-SO EFFECTIVE)
        ########################################################

        self.ref_l = l.unsqueeze(0)

        return self.entropy(self.ref_l) + ub

        ########################################################
        # Gaussian injection (BAD)
        ########################################################

        # Sample random vector with L2norm = 1
        rv = torch.randn(*lp.shape, device = lp.device, dtype = lp.dtype)
        rv = F.normalize(rv, p = 2.0, dim = -1)
        rv = torch.einsum('iNd,N->iNd', rv, ub) 

        refined_logits = l.unsqueeze(0) + rv
        self.ref_l = refined_logits

        return self.entropy(refined_logits)
