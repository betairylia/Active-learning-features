import seaborn as sns
from matplotlib import pyplot as plt
from param_inject import *

import numpy as np
import scipy

'''
net: a torch.Model that can forward test_set
test_set: enumerable containing test data
raw_NTK_eval: Tensor[N_train, N_test] given the NTK(X,Z). N_test's order must be the same as test_set

output: None
behavior: will upload the following datatable to wandb run:
    - E[NTK(X,z)]; E[|NTK(X,z)|]; Maximum[NTK(X,z)]; <d_\theta f(z), \theta_T>
'''
def visualize(net, test_set, raw_NTK_eval, raw_NTK_eval_val_val, outdim = -1):

    # Create noise-injected model
    InjectNet(net, noise_pattern = 'prop')

    # Collect output variance -> <grad, param>
    resample_perturb(net)
    enable_perturb(net)

    outputs = []
    batch_size = 16
    batches = len(test_set) // batch_size

    for bi in range(batches):

        bs = bi * batch_size
        be = bs + batch_size

        batch_outputs = []

        batch = test_set[bs:be]
        batch = batch.to(torch.device('cuda'))

        for i in range(10):
            batch_outputs.append(net(batch))
            resample_perturb(net)

        batch_outputs = torch.stack(batch_outputs, dim = 0)

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
    O_zz = torch.diagonal(raw_NTK_eval_val_val, 0)
    E_abs_O = torch.abs(raw_NTK_eval).mean(dim = 0)
    E_max = torch.max(raw_NTK_eval, 0)[0]

    # Looks dirty but okay
    for i in range(outputs.shape[0]):
        data = {
            "E[O(X,z)]": E_O[i],
            "O(z,z)": O_zz[i],
            "O(z,z)-E[O(X,z)]": O_zz[i] - E_O[i],
            "E[|O(X,z)|]": E_abs_O[i],
            "Max[O(X,z)]": E_max[i],
            "<df(z), param>": outputs[i],
        }
        all_data.append(data)
        wandb.log(data)

    table = wandb.Table(columns = list(all_data[0].keys()), data = [list(d.values()) for d in all_data])
    wandb.log({"NTK-experiment table": table})
    
    fig, ax = plt.subplots()
    sns.scatterplot(data = table.get_dataframe(), x = "<df(z), param>", y = "E[O(X,z)]", ax = ax)
    wandb.log({"NTK-expr plot E_O": wandb.Image(fig)})

    R_E_O = scipy.stats.pearsonr(outputs, E_O[:outputs.shape[0]]).statistic
    R_zz_sub_E_O = scipy.stats.pearsonr(outputs, O_zz[:outputs.shape[0]] - E_O[:outputs.shape[0]]).statistic
    R_E_abs_O = scipy.stats.pearsonr(outputs, E_abs_O[:outputs.shape[0]]).statistic
    R_E_E_abs = scipy.stats.pearsonr(E_O, E_abs_O).statistic
    wandb.log({
        "PearsonR E[O(X,z)] || <,>": R_E_O,
        "PearsonR O(z,z) - E[O(X,z)] || <,>": R_E_O,
        "PearsonR E[|O(X,z)] || <,>": R_E_abs_O,
        "PearsonR E[] || |E[]|": R_E_E_abs
    })

