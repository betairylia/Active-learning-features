
from param_inject import *

'''
net: a torch.Model that can forward test_set
test_set: enumerable containing test data
raw_NTK_eval: Tensor[N_train, N_test] given the NTK(X,Z). N_test's order must be the same as test_set

output: None
behavior: will upload the following datatable to wandb run:
    - E[NTK(X,z)]; E[|NTK(X,z)|]; Maximum[NTK(X,z)]; <d_\theta f(z), \theta_T>
'''
def visualize(net, test_set, raw_NTK_eval, outdim = -1):

    # Create noise-injected model
    injected_net = Inject(net, noise_pattern = 'prop')

    # Collect output variance -> <grad, param>
    resample_perturb(injected_net)
    enable_perturb(injected_net)

    outputs = []
    batch_size = 16
    batches = len(test_set) // batch_size

    for bi in range(batches):

        bs = bi * batch_size
        be = bs + batch_size

        batch_outputs = []

        for i in range(10):
            outputs.append(injected_net(x[bs:be]))
            resample_perturb(injected_net)

        batch_outputs = torch.stack(outputs, dim = 0)

        # Use all dimensions
        if outdim < 0:
            product_result = batch_outputs.std(dim = 0).sum(-1)
        else:
            product_result = batch_outputs[:, :, outdim].std(dim=0)

        outputs.append(product_result)

    outputs = torch.cat(outputs).detach().cpu()
    
    # Result: outputs : Tensor of [batch_size]

    # Output to datatables
    all_data = []
    E_O = raw_NTK_eval.mean(dim = 0)
    E_abs_O = torch.abs(raw_NTK_eval).mean(dim = 0)
    E_max = torch.max(raw_NTK_eval, 0)[0]

    # Looks dirty but okay
    for i in range(outputs.shape[0]):
        data = {
            "E[O(X,z)]": E_O[i],
            "E[|O(X,z)|]": E_abs_O[i],
            "Max[O(X,z)]": E_max[i],
            "<df(z), param>": outputs[i],
        }
        all_data.append(data)
        wandb.log(data)

    table = wandb.Table(columns = list(all_data[0].keys()), data = [list(d.values()) for d in all_data])
    wandb.log({"Table" % prefix: table})
    

