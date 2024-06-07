import torch
from torch.nn import functional as F
from .base import SimpleModel

from .param_inject import *

class InjectTest(SimpleModel):

    def __init__(self, args, input_shape, output_dim = 10):
        
        super().__init__(args, input_shape, output_dim)
        
        self.perturb_nonlinear = args.perturb_nonlinear
        self.perturb_min = args.perturb_min
        self.perturb_max = args.perturb_max
        self.noise_pattern = args.noise_pattern
        
        self.combined_net = nn.Sequential(self.net, self.head)
        InjectNet(
            self.combined_net,
            perturb_nonlinear = self.perturb_nonlinear,
            perturb_min = self.perturb_min,
            perturb_max = self.perturb_max,
            noise_pattern = self.noise_pattern
        )

        self.combined_net_init = nn.Sequential(self.net_init, self.head_init)

        # breakpoint()

    def get_predictions(self, x, times = self.args.dropout_iters):

        resample_perturb(self.combined_net)
        enable_perturb(self.combined_net)
        logits = []
        probs = []

        for i in range(times):
            logits.append(self.combined_net(x))
            pred_prob = F.softmax(logits[-1], dim = 1)
            probs.append(pred_prob)
            resample_perturb(self.combined_net)

        logits = torch.stack(logits, dim = 0)
        probs = torch.stack(probs, dim = 0)

        return logits, probs

    def forward(self, x):

        if self.training:
            disable_perturb(self.combined_net)
            logits = self.combined_net(x)
            return logits, None
        
        else:

            logits, probs = self.get_predictions(x)

            model_prediction = probs.mean(0)
            entropy = -torch.sum(model_prediction * torch.log(model_prediction + 1e-8), dim = 1)
            uncertainty = entropy

            return logits.mean(dim = 0), uncertainty

class InjectTest_Fluc(InjectTest):

    def forward(self, x):

        if self.training:
            return super().forward(x)
        
        else:

            logits, probs = self.get_predictions(x)

            fluctuation_normed = logits.std(dim = 0).mean(dim = -1)

            return logits.mean(dim = 0), fluctuation_normed

class InjectTest_NormalizedFluc(InjectTest):

    def forward(self, x):

        if self.training:
            return super().forward(x)
        
        else:

            logits, probs = self.get_predictions(x)

            # Pop state
            cache = get_states(self.combined_net)

            # Switch to indep perturbation
            set_perturb_norm(self.combined_net, noise_norm = 0.001, noise_pattern = 'indep')

            # Obtain indep result
            logits_indep, probs_indep = self.get_predictions(x)

            # Push state
            set_states(self.combined_net, cache)

            fluctuation_normed = (logits.std(dim = 0) / logits_indep.std(dim = 0)).mean(dim = -1)

            return logits.mean(dim = 0), fluctuation_normed

class InjectTest_Subtract(InjectTest):

    def __init__(self, args, input_shape, output_dim = 10):
        
        super().__init__(args, input_shape, output_dim)

        # Switch to subtract perturbation
        set_perturb_norm(self.combined_net, noise_norm = None, noise_pattern = 'subtract', noise_norm_ex = args.perturb_ex)

class InjectTest_IndepDet(InjectTest):

    def __init__(self, args, input_shape, output_dim = 10):
        
        if args.noise_pattern != "indep":
            print("[WARNING] noise_pattern != 'indep' for InjectTest_IndepDet, result will not align much!")

        super().__init__(args, input_shape, output_dim)

        self.lambda_det = args.perturb_ex

    def forward(self, x):

        if self.training:
            return super().forward(x)
        
        else:

            logits, probs = self.get_predictions(x) # indep, [n_iters, batch_size, output_dim]

            # Pop state
            cache = get_states(self.combined_net)

            # Compute original logits
            set_perturb_norm(self.combined_net, noise_norm = 0, noise_pattern = 'prop-deterministic')
            logits_original, probs_original = self.get_predictions(x, times = 1)

            # Compute <g,p>
            set_perturb_norm(self.combined_net, noise_norm = 0.00001, noise_pattern = 'prop-deterministic')
            logits_det, probs_det = self.get_predictions(x, times = 1)

            # Push state
            set_states(self.combined_net, cache)

            det_diff = logits_det - logits_original
            ub = logits.std(dim = 0) - self.lambda_det * det_diff.squeeze()

            ub = ub - torch.min(ub, dim = -1, keepdim = True)

            upperbound_sum = torch.sum(ub * probs_original.squeeze(), dim = -1)
            fluctuation_mean = (logits.std(dim = 0)).mean(dim = -1)

            return logits.mean(dim = 0), upperbound_sum


