import torch
from .base import SimpleModel

from .param_inject import *

class InjectTest(SimpleModel):

    def __init__(self, args, input_shape, output_dim = 10):
        
        super().__init__(args, input_shape, output_dim)

        self.combined_net = nn.Sequential(self.net, self.head)
        self.combined_net = InjectNet(self.combined_net)

        self.combined_net_init = nn.Sequential(self.net_init, self.head_init)

        # breakpoint()

    def forward(self, x):

        if self.training:
            disable_perturb(self.combined_net)
            logits = self.combined_net(x)
            return logits, None
        
        else:
            resample_perturb(self.combined_net)
            enable_perturb(self.combined_net)
            logits = []
            probs = []

            for i in range(self.args.dropout_iters):
                logits.append(self.head(self.net(x)))
                pred_prob = F.softmax(logits[-1], dim = 1)
                probs.append(pred_prob)

            logits = torch.stack(logits, dim = 0)
            probs = torch.stack(probs, dim = 0)

            model_prediction = probs.mean(0)
            entropy = -torch.sum(model_prediction * torch.log(model_prediction + 1e-8), dim = 1)
            uncertainty = entropy

            return logits.mean(dim = 0), uncertainty
