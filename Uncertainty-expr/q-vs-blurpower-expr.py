import main as main_expr

import torch
from pytorch_lightning import LightningModule, Trainer
from torch import nn
from torch.nn import functional as F

import numpy as np

from argparse import ArgumentParser

from nets import net_dict
from blurpower_expr_quantities import Qdict

import wandb

from tqdm import tqdm

def parse_args():

    parent_parser = ArgumentParser()
    
    parser = parent_parser.add_argument_group("Model Hyper-parameters")
    
    parser = parent_parser.add_argument_group("Active Learning related")
    parser.add_argument("--heuristic", type=str, default="random")
    parser.add_argument("--loss", type=str, default="cent") # cent, mse
    parser.add_argument("--epochs_per_query", type=int, default=25)
    parser.add_argument("--inference_iteration", type=int, default=20)
    parser.add_argument("--model", type=str, default='default')
    parser.add_argument("--net", type=str, default='mlp')
    parser.add_argument("--dataset", type=str, default='mnist')
    parser.add_argument("--do_partial_train", type=int, default=0)
    parser.add_argument("--use_full_trainset", type=int, default=1)
    parser.add_argument("--do_contamination", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--test_set_max", type=int, default=-1)

    parser.add_argument("--binary", type=int, default=0, help="Convert the problem to a binary classification. Splits the dataset into halves.")
    
    # Model
    parser.add_argument("--hidden_dim", type=int, default=2048)
    parser.add_argument("--dropout_rate", type=float, default=0.85)
    parser.add_argument("--dropout_iters", type=int, default=10)
    parser.add_argument("--lazy_scaling", type=float, default=1)
    parser.add_argument("--pointwise_linearization", type=int, default=1)

    parser.add_argument("--reference_data_count", type=int, default=1024)
    parser.add_argument("--random_multidim", type=int, default=1)
    parser.add_argument("--num_multidim", type=int, default=32)

    parser.add_argument("--perturb_power", type=float, default=0.1)

    # Visualization
    # TODO: FIXME:  Visalization is wrong. The r.v. is a dropout mask, then we need to plot (grad for parameter #j, hessian for parameter #j)
    #               for this particular mask, which is a loop thru entire dataset per mask.
    # parser.add_argument("--independence_check_layers", nargs="+", type=str)
    # parser.add_argument("--independence_check_dataid", nargs="+", type=int)

    # Training
    parser.add_argument('--no_train', type=int, default=0, help="Don't train the model.")
    parser.add_argument('--dataAug', type=int, default=0, help="Data augmentation on(1) / off(0).")
    parser.add_argument('--contaminate_ref', type=int, default=0, help="Use comtaminated data in reference dataset.")
    parser.add_argument('--noise', type=float, default=0.3, help="Noise std for outliers.")
    parser.add_argument('--blur', type=float, default=2.0, help="Gaussian blur sigma for outliers. ImageNet-C: [1, 2, 3, 4, 6]")
    parser.add_argument('--optim', type=str, default='sgd', help="Optimizer type: ['sgd', 'adamw'].")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for training.")
    parser.add_argument('--epochs', type=int, default=50, help="Number of epochs for training.")
    parser.add_argument('--augTrials', type=int, default=4, help="Trials per augmentation (ignored if augmentation disabled)")
    
    parser = parent_parser.add_argument_group("Run metadata / WandB sweeps")
    parser.add_argument('--keyargs', type=str, default="", help='Key variables in HP tune, splitted in commas')
    parser.add_argument('--aaarunid', type=int, default=0, help='Run ID.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed. will be multiplied with runid+1 to ensure different RNGs for different runs.')

    parser = parent_parser.add_argument_group("Dataloader")

    parser.add_argument("--blur_min", type=float, default=0.0)
    parser.add_argument("--blur_max", type=float, default=10.0) # cent, mse
    parser.add_argument("--blur_steps", type=int, default=64) # cent, mse
    
    parser.add_argument("--q", type=str, default='trivial-uq')

    args = parent_parser.parse_args()

    return args

def main():

    args = parse_args()

    # Start of experiment

    wandb_logger = wandb.init(
        project = "q-vs-blurpower-experiment",
        config = args
    )

    Q = Qdict[args.q](args)

    indices = None

    # Loop through all blur strengths
    for blur_power in np.linspace(args.blur_min, args.blur_max, args.blur_steps):

        print("Blur sigma = %f" % blur_power)

        ##############################
        # Prepare dataloader
        ##############################

        main_datamodule, input_dim = main_expr.get_data_module(
            args.dataset,
            args.batch_size,
            data_augmentation = (args.dataAug > 0),
            num_workers=args.num_workers,
            do_partial_train = args.do_partial_train,
            do_contamination = args.do_contamination,
            test_set_max = args.test_set_max,
            is_binary = args.binary,
            noise_std = args.noise,
            blur_sigma = blur_power)

        main_datamodule.setup()

        if indices is None:
            indices = np.random.choice(len(main_datamodule.test_dataset), size = (args.reference_data_count,))

        calibration_data = torch.utils.data.Subset(
            main_datamodule.test_dataset,
            np.random.choice(len(main_datamodule.test_dataset), size = (args.reference_data_count,))
        )

        calibration_dl = torch.utils.data.DataLoader(
            calibration_data,
            batch_size = args.batch_size,
            shuffle = False,
            num_workers = 0
        )

        ##############################
        # Get Net
        ##############################

        output_dim = main_datamodule.n_classes if not args.binary else 1

        net_factory = net_dict[args.net]()
        net, head = net_factory.getNets(
            input_dim, 
            [output_dim],
            hidden_dim = args.hidden_dim,
            dropout_rate = args.dropout_rate
        )
       
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        net = net.to(device)
        head = head.to(device)

        net, head = Q.preprocess(net, head)

        ##############################
        # Feed the beast
        ##############################
       
        q_evals = []

        for batch in tqdm(calibration_dl):
            batch = [b.to(device) for b in batch]
            q_eval = Q.evaluate(net, head, batch)
            q_evals.append(q_eval)

        summary_dict = Q.summary(blur_power == 0.0, q_evals)
        summary_dict["blur_power"] = blur_power
        wandb.log(summary_dict)

if __name__ == "__main__":
    main()

