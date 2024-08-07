import torch
from pytorch_lightning import LightningModule, Trainer
from torch import nn
import pytorch_lightning as pl

import torchvision
from torchvision import datasets
from torchvision.transforms import transforms
import torchvision.transforms as transforms

import numpy as np
import math

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class BinaryDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, n_classes):
        self.dataset = dataset
        self.n_classes = n_classes
        self.half_split = n_classes // 2

    def convert_to_binary(self, y):
        return 0 if y < self.half_split else 1

    def __getitem__(self, index):

        item = self.dataset[index]

        # x, y
        if len(item) == 2:
            return (item[0], self.convert_to_binary(item[1]))

        # x, y, o
        elif len(item) == 3:
            return (item[0], self.convert_to_binary(item[1]), item[2])

        # i, x, y, o
        elif len(item) == 4:
            return (item[0], item[1], self.convert_to_binary(item[2]), item[3])

    def __len__(self):
        return len(self.dataset)

class IndicedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        x = self.dataset[index]
        return (index, *x)

    def __len__(self):
        return len(self.dataset)

class UncertaintyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, is_contamination = 0):
        self.dataset = dataset
        self.is_contamination = is_contamination

    def __getitem__(self, index):
        d, l = self.dataset[index]
        return d, l, self.is_contamination

    def __len__(self):
        return len(self.dataset)

class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_1, dataset_2):
        self.dataset_1 = dataset_1
        self.d1_len = len(self.dataset_1)
        self.dataset_2 = dataset_2
        self.d2_len = len(self.dataset_2)
    
    def __getitem__(self, index):
        if index < self.d1_len:
            d, l, is_o = self.dataset_1[index]
        else:
            d, l, is_o = self.dataset_2[index - self.d1_len]
        return d, l, is_o
    
    def __len__(self):
        return self.d1_len + self.d2_len

class UncertaintyDataModule(pl.LightningDataModule):

    def __init__(
            self,
            batch_size = 128,
            n_labeled = 16384,
            num_workers = 16,
            data_dir = './data',
            do_partial_train = False,
            do_contamination = True,
            use_full_trainset = True,
            blur_sigma = 2.0,
            noise_std = 0.3,
            test_set_max = -1,
            is_binary = 0
        ):

        super().__init__()

        self.batch_size = batch_size
        self.n_labeled = n_labeled
        self.num_workers = num_workers
        self.data_dir = data_dir

        self.inited = False

        self.do_partial_train = do_partial_train
        self.do_contamination = do_contamination
        self.use_full_trainset = use_full_trainset
        self.test_set_max = test_set_max

        self.blur_sigma = blur_sigma
        self.noise_std = noise_std

        self.is_binary = is_binary
    
    def prepare_data(self):
        pass

    def get_standard_transforms(self):
        pass
    
    def get_standard_transforms_train(self):
        return get_standard_transforms()

    def get_train_dataset(self, transform):
        pass

    def get_test_dataset(self, transform):
        pass

    def setup(self, stage = None):
        if self.inited == False:
            self.inited = True
            standard_transform = self.get_standard_transforms()
            standard_transform_train = self.get_standard_transforms_train()

            print("Loading raw datasets...")
            self.train_dataset = self.get_train_dataset(transform = standard_transform_train)
            self.test_dataset = self.get_test_dataset(transform = standard_transform)
    
            if self.do_partial_train:
                print("Removing classes from training set...")
                self.remove_classes_from_train_set()
    
            if self.do_contamination:
                print("Adding contamination...")
                self.add_contaminate_dataset()
    
            if not self.use_full_trainset:
                print("Pruning training set...")
                self.prune_train_set(self.n_labeled)
            
            # print("Balancing test set...")
            # self.balance_test_set()

            if(self.test_set_max > 0):
                print("Capping test set...")
                self.cap_test_set(self.test_set_max)

            print("Wrapping datasets...")
            self.train_dataset = IndicedDataset(self.train_dataset)
            self.test_dataset = IndicedDataset(self.test_dataset)

            if self.is_binary:
                print("Converting to binary classification...")
                self.train_dataset = BinaryDataset(self.train_dataset, self.n_classes or 10)
                self.test_dataset = BinaryDataset(self.test_dataset, self.n_classes or 10)

    def prune_train_set(self, n_labeled):
        self.train_dataset_full = self.train_dataset
        self.train_dataset = torch.utils.data.Subset(self.train_dataset, np.random.choice(len(self.train_dataset), size = (n_labeled,)))

    def balance_test_set(self):
        inlier_indices_test = [i for i, (x, y, o) in enumerate(self.test_dataset) if o == 0]
        outlier_indices_test = [i for i, (x, y, o) in enumerate(self.test_dataset) if o == 1]
        self.n_raw_inliers = len(inlier_indices_test)
        self.n_raw_outliers = len(self.test_dataset) - self.n_raw_inliers

        assert self.n_raw_outliers > 0 and self.n_raw_inliers > 0 and len(self.test_dataset) == self.n_raw_inliers + len(outlier_indices_test)

        picked_inliers = np.random.choice(inlier_indices_test, size = (self.n_raw_outliers,))
        self.test_dataset = torch.utils.data.Subset(self.test_dataset, [*picked_inliers, *outlier_indices_test])
    
    def cap_test_set(self, capacity):
        if capacity > 0:
            self.test_dataset = torch.utils.data.Subset(self.test_dataset, np.random.choice(len(self.test_dataset), size = (capacity,)))

    def remove_classes_from_train_set(self):
        
        # Define classes to remove
        class_to_remove_1 = 3
        class_to_remove_2 = 5

        class_map_inv = [i for i in range(self.n_classes) if i != class_to_remove_1 and i != class_to_remove_2]
        class_map_inv.append(class_to_remove_1)
        class_map_inv.append(class_to_remove_2)

        # from:      [0, 1, ~2~, 3, 4]
        # to:        [0, 1, 3, 4, 2]
        # class_map: [0, 1, 4, 2, 3]

        class_map = [0 for i in range(self.n_classes)]
        for i in range(self.n_classes):
            class_map[class_map_inv[i]] = i

        print(class_map)

        # Create a custom dataset for the training set
        class LabelRemapDataset(torch.utils.data.Dataset):
            def __init__(self, dataset, label_map, inlier_classes = 8):
                self.dataset = dataset
                self.label_map = torch.LongTensor(label_map)
                self.inlier_classes = inlier_classes

            def __getitem__(self, index):
                d, l = self.dataset[index]
                new_l = self.label_map[l]
                return d, new_l, 1 if new_l >= self.inlier_classes else 0

            def __len__(self):
                return len(self.dataset)

        filtered_indices_train = [i for i, (x, y) in enumerate(self.train_dataset) if y != class_to_remove_1 and y != class_to_remove_2]
        print(len(filtered_indices_train))
        self.train_dataset = torch.utils.data.Subset(self.train_dataset, filtered_indices_train)
        self.train_dataset = LabelRemapDataset(self.train_dataset, class_map)
        self.test_dataset = LabelRemapDataset(self.test_dataset, class_map)

        self.n_classes -= 2

    def add_contaminate_dataset(self):

        n_contaminate = len(self.test_dataset)
        dset_self = self

        self.train_dataset = UncertaintyDataset(self.train_dataset, is_contamination = 0)

        ###################################
        # Create a contaminated test set
        ###################################

        contaminated_transforms = self.get_standard_transforms().transforms
        idx_toTensor = [i for i, t in enumerate(contaminated_transforms) if isinstance(t, transforms.ToTensor)][0]
        
        # Insert Gaussian Blur before ToTensor
        if self.blur_sigma > 0:
            kernel_size = math.ceil(8 * self.blur_sigma - 1) // 2 * 2 + 1
            contaminated_transforms.insert(idx_toTensor, transforms.GaussianBlur(kernel_size = max(3, kernel_size), sigma = self.blur_sigma))

        # Insert Gaussian Noise at the end
        contaminated_transforms.append(AddGaussianNoise(mean = 0.0, std = self.noise_std))

        # Compose the transform
        contaminated_transform = transforms.Compose(contaminated_transforms)

        self.test_dataset_contaminate = self.get_test_dataset(transform = contaminated_transform)

        ###################################
        # Combine them
        ###################################
            
        self.test_dataset = UncertaintyDataset(self.test_dataset, is_contamination = 0)
        self.test_dataset_contaminate = UncertaintyDataset(self.test_dataset_contaminate, is_contamination = 1)
        self.test_dataset_uncontaminated = self.test_dataset

        assert len(self.test_dataset) == len(self.test_dataset_contaminate)
        print("%d test samples loaded." % len(self.test_dataset))
        
        self.test_dataset = CombinedDataset(self.test_dataset, self.test_dataset_contaminate)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle = True, num_workers = self.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size = self.batch_size, shuffle = True, num_workers = self.num_workers)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size = self.batch_size, shuffle = True, num_workers = self.num_workers)

class MNIST_UncertaintyDM(UncertaintyDataModule):
    
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_classes = 10

    def prepare_data(self):
        datasets.MNIST(root = self.data_dir, train = True, download = True)
        datasets.MNIST(root = self.data_dir, train = False, download = True)

    def get_standard_transforms(self):
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def get_train_dataset(self, transform):
        train_dataset = datasets.MNIST(root = self.data_dir, train = True, download = True, transform = transform)
        return train_dataset
    
    def get_test_dataset(self, transform):
        test_dataset = datasets.MNIST(root = self.data_dir, train = False, download = True, transform = transform)
        return test_dataset

class CIFAR10_UncertaintyDM(UncertaintyDataModule):
    
    n_classes: 10

    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_classes = 10

    def prepare_data(self):
        datasets.CIFAR10(root = self.data_dir, train = True, download = True)
        datasets.CIFAR10(root = self.data_dir, train = False, download = True)

    def get_standard_transforms(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
    def get_standard_transforms_train(self):
        return transforms.Compose([
            transforms.Resize((32,32)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    
    def get_train_dataset(self, transform):
        return datasets.CIFAR10(root = self.data_dir, train = True, download = True, transform = transform)
    
    def get_test_dataset(self, transform):
        return datasets.CIFAR10(root = self.data_dir, train = False, download = True, transform = transform)

class SVHN_UncertaintyDM(UncertaintyDataModule):
    
    n_classes: 10

    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_classes = 10

    def prepare_data(self):
        datasets.SVHN(root = self.data_dir, split = 'train', download = True)
        datasets.SVHN(root = self.data_dir, split = 'test', download = True)

    def get_standard_transforms(self):
        return transforms.Compose([
            transforms.ToTensor(),
            
            # https://deepobs.readthedocs.io/en/v1.2.0-beta0_a/_modules/deepobs/pytorch/datasets/svhn.html
            # Not sure where those numbers come from but should be fine
            transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
        ])
    
    def get_train_dataset(self, transform):
        return datasets.SVHN(root = self.data_dir, split = 'train', download = True, transform = transform)
    
    def get_test_dataset(self, transform):
        return datasets.SVHN(root = self.data_dir, split = 'test', download = True, transform = transform)

class ImageNet_Validation_UncertaintyDM(UncertaintyDataModule):
    
    n_classes: 1000

    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_classes = 1000

    def prepare_data(self):
        print("Loading raw datasets for the first time...")
        datasets.ImageNet(root = './data/ILSVRC2012', split = 'val')

    def get_standard_transforms(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    
    def get_train_dataset(self, transform):
        return None
    
    def get_test_dataset(self, transform):
        return datasets.ImageNet(root = './data/ILSVRC2012', split = 'val', transform = transform)
