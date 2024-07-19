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
        self.mul_temp = args.mul_temp
        
        self.combined_net = nn.Sequential(self.net, self.head)
        self.combined_net_init = nn.Sequential(self.net_init, self.head_init)

        InjectNet(
            self.combined_net,
            self.combined_net_init,
            perturb_nonlinear = self.perturb_nonlinear,
            perturb_min = self.perturb_min,
            perturb_max = self.perturb_max,
            noise_pattern = self.noise_pattern
        )

        # breakpoint()

    def get_predictions(self, x, times = -1):

        if times <= 0:
            times = self.args.dropout_iters

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

            if abs(self.mul_temp - 1) > 1e-2:

                # Pop state
                cache = get_states(self.combined_net)

                # Compute original logits
                set_perturb_norm(self.combined_net, noise_norm = 0, noise_pattern = 'prop-deterministic')
                logits_original, probs_original = self.get_predictions(x, times = 1)

                # Push state
                set_states(self.combined_net, cache)

                logits_diff = logits - logits_original
                logits_scaled_diff = logits_diff * self.mul_temp
                logits_new = logits_original + logits_scaled_diff
                probs_new = F.softmax(logits_new, dim = -1)

                logits = logits_new
                probs = probs_new

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

# TODO
class InjectTest_NormalizedSubtract(InjectTest):

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
        self.add_temp = args.add_temp
        self.mul_temp = args.mul_temp

        # self.mode = "pure-fluctuation"
        # self.mode = "posterior"
        self.mode = args.indepdet_mode

    def forward(self, x):

        if self.training:
            return super().forward(x)
        
        else:

            logits, probs = self.get_predictions(x) # indep, [n_iters, batch_size, output_dim]

            # Pop state
            cache = get_states(self.combined_net)

            # Compute original logits
            set_perturb_norm(self.combined_net, noise_norm = 0, noise_pattern = 'prop-deterministic')
            logits_original, probs_original = self.get_predictions(x, times = 1) # [1, bs, outdim]

            # Compute <g,p>
            set_perturb_norm(self.combined_net, noise_norm = 0.005, noise_pattern = 'prop-deterministic')
            logits_det, probs_det = self.get_predictions(x, times = 1) # [1, bs, outdim]

            # Push state
            set_states(self.combined_net, cache)

            if self.mode == "pure-fluctuation":

                # det_diff = logits_det - logits_original
                # ub = logits.std(dim = 0) - self.lambda_det * torch.abs(det_diff).squeeze()
                # print("noise %f | det %f" % (logits.std(dim = 0).mean(), self.lambda_det * torch.abs(det_diff).mean()))

                # # ub = ub - torch.min(ub, dim = -1, keepdim = True)[0]

                # upperbound_sum = torch.sum(ub * probs_original.squeeze(), dim = -1)
                # fluctuation_mean = (logits.std(dim = 0)).mean(dim = -1)

                # return logits.mean(dim = 0), upperbound_sum

                det_diff = logits_det - logits_original
                Ozz = logits.std(dim = 0).sum(dim = -1)
                Oxz = self.lambda_det * torch.norm(det_diff, dim = -1).squeeze()
                ub = Ozz - Oxz
                print("noise %f | det %f" % (Ozz.mean(), Oxz.mean()))

                # ub = ub - torch.min(ub, dim = -1, keepdim = True)[0]

                # upperbound_sum = torch.sum(ub * probs_original.squeeze(), dim = -1)
                # fluctuation_mean = (logits.std(dim = 0)).mean(dim = -1)

                # ub = -Ozz

                return logits.mean(dim = 0), ub

            elif self.mode == "mixed":

                det_diff = logits_det - logits_original # [1, bs, outdim]
                ub = logits.std(dim = 0) - self.lambda_det * torch.abs(det_diff).squeeze() # [bs, outdim]

                upperbound_sum = torch.sum(ub * probs_original.squeeze(), dim = -1) # [bs]
                fluctuation_mean = (logits.std(dim = 0)).mean(dim = -1)

                logits_refined = logits_original.squeeze() - (upperbound_sum.unsqueeze(-1) * probs_original.squeeze()) * self.mul_temp # [bs, outdim]
                probs_refined = F.softmax(logits_refined, dim = -1)
                entropy = -torch.sum(probs_refined * torch.log(probs_refined + 1e-8), dim = 1)
                uncertainty = entropy

                # entropy = -torch.sum(probs.mean(0) * torch.log(probs.mean(0) + 1e-8), dim = 1)
                # uncertainty = entropy + self.mul_temp * upperbound_sum

                print("entropy %f | logits %f [Max %f] -> logits %f [Max %f] | noise %f | det %f" % (
                    entropy.mean(),
                    logits_original.mean(),
                    logits_original.max(dim = -1)[0].mean(),
                    logits_refined.mean(),
                    logits_refined.max(dim = -1)[0].mean(),
                    logits.std(dim = 0).mean(),
                    self.lambda_det * torch.abs(det_diff).mean()
                ))
 
                return logits_refined, uncertainty

            elif self.mode == "posterior":

                logits_diff = logits - logits_original
                
                det_diff = torch.abs(logits_det - logits_original)
                logits_scale = (torch.ones_like(det_diff) - self.lambda_det * det_diff)
                # print("Logits scale: %s" % repr(logits_scale))

                # missed a sqrt here

                logits_scaled_diff = logits_diff * torch.exp(self.mul_temp * (logits_scale + self.add_temp))
                # print("Actual logits scale: %s" % repr(logits_scaled_diff / logits_diff))
                logits_new = logits_original + logits_scaled_diff
                probs_new = F.softmax(logits_new, dim = -1)

                model_prediction = probs_new.mean(0)
                entropy = -torch.sum(model_prediction * torch.log(model_prediction + 1e-8), dim = 1)
                uncertainty = entropy

                return logits_new.mean(dim = 0), uncertainty

