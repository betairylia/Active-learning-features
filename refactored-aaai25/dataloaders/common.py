from .data_uncertainty import MNIST_UncertaintyDM, CIFAR10_UncertaintyDM, SVHN_UncertaintyDM, ImageNet_Validation_UncertaintyDM #, FashionMNIST_UncertaintyDM

# dataset format: batch => [i, x, y, o] where
# i: List[Number]       data index in dataset.
# x: Tensor             input X
# y: Tensor             input Y (usually labels)
# o: LongTensor, 1D     OOD label (0 for IN, 1 for OOD)

input_size_dict = {
    'mnist': [1, 32, 32], # Resized
    'cifar10': [3, 32, 32],
    'cifar100': [3, 32, 32],
    'svhn': [3, 32, 32],
    'fashionmnist': [1, 32, 32], # Resized
    'imagenet': [3, 224, 224],
    'tinyimagenet': [3, 64, 64],
    'stl10': [3, 96, 96],
    'lsun': [3, 256, 256],
    'celeba': [3, 64, 64],
    'cub200': [3, 224, 224],
}

def get_data_module(
    dataset,
    batch_size,
    data_augmentation=True,
    num_workers=16,
    data_dir='./data',
    do_partial_train = False,
    do_contamination = True,
    use_full_trainset = True,
    test_set_max = -1,
    binary = 0,
    noise = 0.3,
    blur = 2.0,
    **kwargs):
    
    args = {
        "data_dir": data_dir,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "do_partial_train": do_partial_train,
        "do_contamination": do_contamination,
        "use_full_trainset": use_full_trainset,
        "test_set_max": test_set_max,
        "is_binary": binary,
        "noise_std": noise,
        "blur_sigma": blur,
    }

    if dataset == 'mnist':
        main_dm = MNIST_UncertaintyDM(**args)
    
    elif dataset == 'cifar10':
        main_dm = CIFAR10_UncertaintyDM(**args)
    
    elif dataset == 'svhn':
        main_dm = SVHN_UncertaintyDM(**args)

    elif dataset == 'imagenet':
        main_dm = ImageNet_Validation_UncertaintyDM(**args)
    
    return main_dm, {
        "input_dim" : input_size_dict[dataset],
        "output_dim": [main_dm.n_classes,]
    }
    # TODO: Binary
    # main_datamodule.n_classes if not hparams.binary else 1
