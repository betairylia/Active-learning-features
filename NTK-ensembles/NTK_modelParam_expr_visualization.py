import seaborn as sns
from matplotlib import pyplot as plt
from param_inject import *

import numpy as np
import scipy

def get_outputs_std(net, dataset, outdim = -1, num_iters = 30, device = torch.device('cuda')):

    # Collect output variance -> <grad, param>
    resample_perturb(net)
    enable_perturb(net)

    outputs = []
    raw_outputs = []
    batch_size = 16
    batches = len(dataset) // batch_size

    for bi in range(batches):

        bs = bi * batch_size
        be = bs + batch_size

        batch_outputs = []

        batch = dataset[bs:be]
        batch = batch.to(torch.device('cuda'))

        for i in range(num_iters):
            batch_outputs.append(net(batch))
            resample_perturb(net)

        batch_outputs = torch.stack(batch_outputs, dim = 0)

        # Use all dimensions
        if outdim < 0:
            product_result = batch_outputs.std(dim = 0).sum(-1)
        else:
            product_result = batch_outputs[:, :, outdim].std(dim=0)

        outputs.append(product_result)
        raw_outputs.append(batch_outputs)

    outputs = torch.cat(outputs).detach().cpu()
    raw_outputs = torch.cat(raw_outputs, dim=1).detach().cpu()

    return outputs, raw_outputs
'''
net: a torch.Model that can forward test_set
test_set: enumerable containing test data
raw_NTK_eval: Tensor[N_train, N_test] given the NTK(X,Z). N_test's order must be the same as test_set

output: None
behavior: will upload the following datatable to wandb run:
    - E[NTK(X,z)]; E[|NTK(X,z)|]; Maximum[NTK(X,z)]; <d_\theta f(z), \theta_T>
'''
def visualize(net, test_set, raw_NTK_eval, raw_NTK_eval_val_val, grad_param_dot = None, grad_paramDiff_dot = None, grad_diff = None, test_y = None, vis_uncertainty = False, outdim = -1):

    _, outputs_original = get_outputs_std(net, test_set, outdim = outdim, num_iters = 2)
    outputs_original = outputs_original[0, :, :]

    # Create noise-injected model <df(z), param>
    InjectNet(net, noise_pattern = 'prop')

    # Result: outputs : Tensor of [batch_size]
    outputs, _ = get_outputs_std(net, test_set, outdim = outdim)    

    # Indep noise pattern result 
    # Switch to indep perturbation ||df(z)||
    set_perturb_norm(net, noise_norm = 0.001, noise_pattern = 'indep')
    outputs_indep, _ = get_outputs_std(net, test_set, outdim = outdim)

    # Subtract noise pattern result
    set_perturb_norm(net, noise_norm = 0.001, noise_pattern = 'subtract')
    outputs_subtract, _ = get_outputs_std(net, test_set, outdim = outdim)

    # Deterministic injection result
    set_perturb_norm(net, noise_norm = 0.00001, noise_pattern = 'prop-deterministic')
    _, outputs_det = get_outputs_std(net, test_set, outdim = outdim, num_iters = 2)
    print(outputs_det.shape)
    outputs_det = (outputs_det[0, :, :] - outputs_original).sum(-1)
    print(outputs_det.shape)

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
            "<df(z)^2, param^2>": outputs[i],
            "<df(z), param>-det": outputs_det[i],
            "df(z) norm": outputs_indep[i],
            "subtract": outputs_subtract[i],
        }

        if grad_param_dot is not None:
            data["<df(z), param>-exact"] = grad_param_dot[i]
            data["<df(z), paramDiff>-exact"] = grad_paramDiff_dot[i]

        all_data.append(data)
        wandb.log(data)

    # breakpoint()
    table = wandb.Table(columns = list(all_data[0].keys()), data = [list(d.values()) for d in all_data])
    wandb.log({"NTK-experiment table": table})

    fig, ax = plt.subplots()
    sns.histplot(data = raw_NTK_eval.flatten(), bins = 64)
    wandb.log({"Histogram of O(z, X)": wandb.Image(fig)})
    plt.close('all')
    
    fig, ax = plt.subplots()
    sns.scatterplot(data = table.get_dataframe(), x = "<df(z)^2, param^2>", y = "E[O(X,z)]", ax = ax).invert_yaxis()
    wandb.log({"NTK-expr plot prop || E_O": wandb.Image(fig)})
    plt.close('all')

    fig, ax = plt.subplots()
    sns.scatterplot(data = table.get_dataframe(), x = "<df(z), param>-det", y = "E[O(X,z)]", ax = ax).invert_yaxis()
    wandb.log({"NTK-expr plot det || E_O": wandb.Image(fig)})
    plt.close('all')

    if grad_param_dot is not None:
        fig, ax = plt.subplots()
        sns.scatterplot(data = table.get_dataframe(), x = "<df(z), param>-det", y = "<df(z), param>-exact", ax = ax).invert_yaxis()
        wandb.log({"<grad, param>: det || exact": wandb.Image(fig)})
        plt.close('all')

        fig, ax = plt.subplots()
        sns.scatterplot(data = table.get_dataframe(), x = "<df(z)^2, param^2>", y = "<df(z), param>-exact", ax = ax).invert_yaxis()
        wandb.log({"<grad, param>: prop || exact": wandb.Image(fig)})
        plt.close('all')

        fig, ax = plt.subplots()
        sns.scatterplot(data = table.get_dataframe(), x = "<df(z), param>-exact", y = "<df(z), paramDiff>-exact", ax = ax).invert_yaxis()
        wandb.log({"<grad, param>: exact || diff-exact": wandb.Image(fig)})
        plt.close('all')

        fig, ax = plt.subplots()
        sns.scatterplot(data = table.get_dataframe(), x = "<df(z), paramDiff>-exact", y = "E[O(X,z)]", ax = ax).invert_yaxis()
        wandb.log({"diff-exact || E_O": wandb.Image(fig)})
        plt.close('all')

    R_E_O = scipy.stats.pearsonr(outputs, E_O[:outputs.shape[0]]).statistic
    R_zz_sub_E_O = scipy.stats.pearsonr(outputs, O_zz[:outputs.shape[0]] - E_O[:outputs.shape[0]]).statistic
    R_E_abs_O = scipy.stats.pearsonr(outputs, E_abs_O[:outputs.shape[0]]).statistic
    R_E_E_abs = scipy.stats.pearsonr(E_O, E_abs_O).statistic

    def compare_PearsonR(value, name):
        R_ub = scipy.stats.pearsonr(outputs, value).statistic
        R_ub_indep = scipy.stats.pearsonr(outputs_indep, value).statistic
        R_ub_subt = scipy.stats.pearsonr(outputs_subtract, value).statistic
        R_ub_final = scipy.stats.pearsonr(outputs_indep - outputs, value).statistic
        R_ub_det = scipy.stats.pearsonr(outputs_det, value).statistic
        R_ub_zz = scipy.stats.pearsonr(O_zz[:outputs.shape[0]], value).statistic
        R_ub_indep_det = scipy.stats.pearsonr(outputs_indep - outputs_det, value).statistic

        o = {
            "PearsonR prop || %s" % name: R_ub,
            "PearsonR indep || %s" % name: R_ub_indep,
            "PearsonR indep-prop || %s" % name: R_ub_final,
            "PearsonR subtract || %s" % name: R_ub_subt,
            "PearsonR deterministic || %s" % name: R_ub_det,
            "PearsonR indep-det || %s" % name: R_ub_indep_det,
            "PearsonR zz || %s" % name: R_ub_zz,
        }

        if grad_param_dot is not None:
            o["PearsonR <g,p>Exct || %s" % name] = scipy.stats.pearsonr(grad_param_dot[:outputs.shape[0]], value).statistic
            o["PearsonR <g,dp>Exct || %s" % name] = scipy.stats.pearsonr((grad_param_dot - grad_paramDiff_dot)[:outputs.shape[0]], value).statistic

        return o

    R_ubs = compare_PearsonR(O_zz[:outputs.shape[0]] - E_max[:outputs.shape[0]], "UB")
    R_EOs = compare_PearsonR(E_O[:outputs.shape[0]], "E[O(X,z)]")
    R_MOs = compare_PearsonR(E_max[:outputs.shape[0]], "Max[O(X,z)]")
    # TODO: Max[O(x, x) - 2*O(x, z)]

    R_gdiff = {}
    if grad_diff is not None:
        minimum_gradDiff = torch.min(grad_diff, 0)[0]
        R_gdiff = compare_PearsonR(minimum_gradDiff[:outputs.shape[0]], "Ozz + Oxx - 2Ozx")

    def visualize_uncertainty(value, name):
        fig, ax = plt.subplots()
        ax.plot(test_set.squeeze(), outputs_original.squeeze())
        if test_y is not None:
            ax.plot(test_set.squeeze(), test_y.squeeze(), 'r--')
        norm_val = value / torch.max(torch.abs(value))
        ax.fill_between(test_set.squeeze(), outputs_original.squeeze() - norm_val.squeeze(), outputs_original.squeeze() + norm_val.squeeze(), alpha = 0.3)
        wandb.log({"Uncertainty %s" % name: wandb.Image(fig)})
        plt.close('all')

    if vis_uncertainty == True:
        visualize_uncertainty(minimum_gradDiff[:outputs.shape[0]], "Ozz + Oxx - 2Ozx")
        visualize_uncertainty(O_zz[:outputs.shape[0]] - E_max[:outputs.shape[0]], "UB")
        visualize_uncertainty(O_zz[:outputs.shape[0]] - E_O[:outputs.shape[0]], "zz-EOzx")
        visualize_uncertainty(outputs_indep, "indep")
        visualize_uncertainty(outputs_det, "det")

    wandb.log({
        "PearsonR O(z,z) - E[O(X,z)] || <,>": R_E_O,
        "PearsonR E[|O(X,z)] || <,>": R_E_abs_O,
        "PearsonR E[] || |E[]|": R_E_E_abs,

        **R_ubs,
        **R_EOs,
        **R_MOs,
        **R_gdiff,
    })

