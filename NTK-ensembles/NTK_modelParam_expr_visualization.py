
from param_inject import *

'''
net: a torch.Model that can forward test_set
test_set: a dataloader for all test points
raw_NTK_eval: Tensor[N_train, N_test] given the NTK(X,Z). N_test's order must be the same as test_set

output: None
behavior: will upload the following datatable to wandb run:
    - E[NTK(X,z)]; E[|NTK(X,z)|]; Maximum[NTK(X,z)]; <d_\theta f(z), \theta_T>
'''
def visualize(net, test_set, raw_NTK_eval):

    # Create noise-injected model
    injected_net = Inject(net, noise_pattern = 'prop')

    # Collect output variance -> <grad, param>
    resample_perturb(injected_net)
    enable_perturb(injected_net)
    outputs = []

    for i in range(self.args.dropout_iters):
        outputs.append(injected_net(x))
        resample_perturb(injected_net)

    outputs = torch.stack(outputs, dim = 0)
    
    # Output to datatables

