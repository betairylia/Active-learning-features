import torch
from pytorch_lightning import LightningModule, Trainer
from torch import nn
import pytorch_lightning as pl

import torchvision
from torchvision import datasets
from torchvision.transforms import transforms
import torchvision.transforms as transforms

import numpy as np

class UncertaintyDataModule(pl.LightningDataModule):

    def __init__(self, batch_size = 128, n_labeled = 16384, num_workers = 16, data_dir = './data', do_partial_train = True, do_contamination = True):

        super().__init__()

        self.batch_size = batch_size
        self.n_labeled = n_labeled
        self.num_workers = num_workers
        self.data_dir = data_dir

        self.inited = False

        self.do_partial_train = do_partial_train
        self.do_contamination = do_contamination
    
    def prepare_data(self):
        pass

    def get_standard_transforms(self):
        pass

    def get_datasets(self):
        pass

    def setup(self, stage = None):
        if self.inited == False:
            self.inited = True
            self.train_dataset, self.test_dataset, self.n_classes = self.get_dataset()
    
            if self.do_partial_train:
                self.remove_classes_from_train_set()
    
            if self.do_contamination:
                self.add_contaminate_dataset()
    
            # self.prune_train_set(self.n_labeled)
            self.balance_test_set()

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

        class UncertaintyDataset(torch.utils.data.Dataset):
            def __init__(self, dataset, is_contamination = 0):
                self.dataset = dataset
                self.is_contamination = is_contamination

            def __getitem__(self, index):
                d, l = self.dataset[index]
                return d, l, self.is_contamination

            def __len__(self):
                return len(self.dataset)

        class ContaminateDataset(torch.utils.data.Dataset):

            def __init__(self, dataset):
                self.dataset = dataset
                self.additional_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.GaussianBlur(kernel_size = 7, sigma = 2.0),
                    transforms.ToTensor()
                ])

            def __getitem__(self, index):
                d, l = self.dataset[index]
                return self.additional_transform(d), torch.LongTensor([l])[0]

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

        self.train_dataset = UncertaintyDataset(self.train_dataset, is_contamination = 0)

        self.test_dataset_contaminate = UncertaintyDataset(ContaminateDataset(self.test_dataset), is_contamination = 1)
        self.test_dataset = UncertaintyDataset(self.test_dataset, is_contamination = 0)
        self.test_dataset_uncontaminated = self.test_dataset
        
        self.test_dataset = CombinedDataset(self.test_dataset, self.test_dataset_contaminate)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle = True, num_workers = self.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size = self.batch_size, shuffle = False, num_workers = self.num_workers)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size = self.batch_size, shuffle = False, num_workers = self.num_workers)

class MNIST_UncertaintyDM(UncertaintyDataModule):
    
    def __init__(self, batch_size = 64, n_labeled = 16384, num_workers = 16, data_dir = "../data", do_partial_train = True, do_contamination = True):
        super().__init__(batch_size, n_labeled, num_workers, data_dir, do_partial_train, do_contamination)

    def prepare_data(self):
        datasets.MNIST(root = self.data_dir, train = True, download = True)
        datasets.MNIST(root = self.data_dir, train = False, download = True)

    def get_standard_transforms(self):
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def get_dataset(self):
        transform = self.get_standard_transforms()
        train_dataset = datasets.MNIST(root = self.data_dir, train = True, download = True, transform = transform)
        test_dataset = datasets.MNIST(root = self.data_dir, train = False, download = True, transform = transform)
        n_classes = 10
        return train_dataset, test_dataset, n_classes
    
class CIFAR10_UncertaintyDM(UncertaintyDataModule):
    
    def __init__(self, batch_size = 64, n_labeled = 16384, num_workers = 16, data_dir = "../data", do_partial_train = True, do_contamination = True):
        super().__init__(batch_size, n_labeled, num_workers, data_dir, do_partial_train, do_contamination)

    def prepare_data(self):
        datasets.CIFAR10(root = self.data_dir, train = True, download = True)
        datasets.CIFAR10(root = self.data_dir, train = False, download = True)

    def get_standard_transforms(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    def get_dataset(self):
        transform = self.get_standard_transforms()
        train_dataset = datasets.CIFAR10(root = self.data_dir, train = True, download = True, transform = transform)
        test_dataset = datasets.CIFAR10(root = self.data_dir, train = False, download = True, transform = transform)
        n_classes = 10
        return train_dataset, test_dataset, n_classes
