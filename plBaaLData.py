# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import warnings
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.utils.data import DataLoader, Dataset, random_split

# from flash.core.data.data_module import DataModule
# from flash.core.data.io.input import InputBase
# from flash.core.data.io.input_transform import create_worker_input_transform_processor
# from flash.core.utilities.imports import _BAAL_AVAILABLE, requires
# from flash.core.utilities.stages import RunningStage

from baal.active.dataset import ActiveLearningDataset
from baal.active.heuristics import AbstractHeuristic, BALD
from AdvancedHeuristics import AdvancedAbstractHeuristic

from functools import partial
import copy

def dataset_to_non_labelled_tensor(dataset: Dataset) -> torch.tensor:
    return np.zeros(len(dataset))


def filter_unlabelled_data(dataset: Dataset) -> Dataset:
    return dataset


def train_val_split(dataset: Dataset, val_size: float = 0.1):
    L = len(dataset)
    train_size = int(L * (1 - val_size))
    val_size = L - train_size
    return random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

def ChangeLoaderDataset(loader: DataLoader, dataset: Dataset, shuffle: bool = False):
    return DataLoader(
        dataset,
        batch_size = loader.batch_size,
        # sampler = loader.sampler,
        shuffle = shuffle,
        num_workers = loader.num_workers,
        collate_fn = loader.collate_fn,
        pin_memory = loader.pin_memory,
        drop_last = loader.drop_last,
        timeout = loader.timeout,
        worker_init_fn = loader.worker_init_fn,
        prefetch_factor = loader.prefetch_factor,
        persistent_workers = loader.persistent_workers
    )

def ActiveLearningDataModuleWrapper(base: pl.LightningDataModule):
    
    class ActiveLearningLightningDataModule(base):
        def __init__(
            self, 

            heuristic: "AbstractHeuristic" = BALD(),
            map_dataset_to_labelled: Optional[Callable] = dataset_to_non_labelled_tensor,
            filter_unlabelled_data: Optional[Callable] = filter_unlabelled_data,
            initial_num_labels: Optional[int] = None,
            query_size: int = 1,

            *args: Any,
            **kwargs: Any,
        ):
            super().__init__(*args, **kwargs)

            self.heuristic = heuristic
            self.map_dataset_to_labelled = map_dataset_to_labelled
            self.filter_unlabelled_data = filter_unlabelled_data
            self.initial_num_labels = initial_num_labels
            self.query_size = query_size

            self._dataset = None
            self._original_train_set = None
            
            # self.is_setup = False
            
        def setup(self, stage: Optional[str] = None) -> None:
            
            super().setup(stage)

        def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:

            loader = super().train_dataloader(*args, **kwargs)

            if self._dataset is None or loader.dataset is not self._original_train_set:

                self._original_train_set = loader.dataset

                self._dataset = ActiveLearningDataset(
                    self._original_train_set, labelled=self.map_dataset_to_labelled(self._original_train_set)
                )

                if not self.initial_num_labels:
                    warnings.warn(
                        "No labels provided for the initial step," "the estimated uncertainties are unreliable!", UserWarning
                    )
                else:
                    self._dataset.label_randomly(self.initial_num_labels)

            al_loader = ChangeLoaderDataset(loader, self._dataset, True)

            return al_loader

        def predict_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
            test_loader = copy.copy(super().test_dataloader(*args, **kwargs))
            al_loader = ChangeLoaderDataset(test_loader, self.filter_unlabelled_data(self._dataset.pool), False)
            return al_loader
        
        @property
        def has_test(self) -> bool:
            return True # TODO ?

        @property
        def has_labelled_data(self) -> bool:
            if self._dataset is None:
                self.train_dataloader()
            return self._dataset.n_labelled > 0

        @property
        def has_unlabelled_data(self) -> bool:
            if self._dataset is None:
                self.train_dataloader()
            return self._dataset.n_unlabelled > 0

        # TODO: Add net here.
        def label(self, evidences: List[Any]=None, net: torch.nn.Sequential=None, indices=None):
            
            if evidences is not None and indices:
                raise MisconfigurationException(
                    "The `evidences` and `indices` are mutually exclusive, pass only of one them."
                )
            
            if evidences is not None:
                
                probabilities, features = list(zip(*evidences))
                
                probabilities = torch.cat([p for p in probabilities], dim=0)
                features = torch.cat([f for f in features], dim=0)
                
                if isinstance(self.heuristic, AdvancedAbstractHeuristic):
                    uncertainties = self.heuristic.get_uncertainties(probabilities, features=features, net=net)
                else:
                    uncertainties = self.heuristic.get_uncertainties(probabilities)
                
                # TODO: Sampling w.r.t. Gibbs distribution instead of argsort
                indices = np.argsort(uncertainties)
                
            if self._dataset is not None:
                self._dataset.label(indices[-self.query_size :])

        def state_dict(self) -> Dict[str, torch.Tensor]:
            return self._dataset.state_dict()

        def load_state_dict(self, state_dict) -> None:
            return self._dataset.load_state_dict(state_dict)
        
    return ActiveLearningLightningDataModule

# class ActiveLearningDataModule(pl.LightningDataModule):
#     @requires("baal")
#     def __init__(
#         self,
        
#         labelled: Optional[pl.LightningDataModule] = None,
        
#         # train_set: Optional[Dataset] = None,
#         # test_set: Optional[Dataset] = None,
        
#         heuristic: "AbstractHeuristic" = BALD(),
#         map_dataset_to_labelled: Optional[Callable] = dataset_to_non_labelled_tensor,
#         filter_unlabelled_data: Optional[Callable] = filter_unlabelled_data,
#         initial_num_labels: Optional[int] = None,
#         query_size: int = 1,
        
#         # train_loader: Optional[Callable] = partial(DataLoader, shuffle = True),
#         # test_loader: Optional[Callable] = DataLoader,
#         # pool_loader: Optional[Callable] = DataLoader,
        
#         # batch_size: int = 32,
#         # num_workers: int = 0,
#     ):
#         """The `ActiveLearningDataModule` handles data manipulation for ActiveLearning.

#         Args:
#             labelled: DataModule containing labelled train data for research use-case.
#                 The labelled data would be masked.
#             heuristic: Sorting algorithm used to rank samples on how likely they can help with model performance.
#             map_dataset_to_labelled: Function used to emulate masking on labelled dataset.
#             filter_unlabelled_data: Function used to filter the unlabelled data while computing uncertainties.
#             initial_num_labels: Number of samples to randomly label to start the training with.
#             query_size: Number of samples to be labelled at each Active Learning loop based on the fed heuristic.
#             val_split: Float to split train dataset into train and validation set.
#         """
#         super().__init__()
        
#         self.labelled = data_module
#         self.heuristic = heuristic
#         self.map_dataset_to_labelled = map_dataset_to_labelled
#         self.filter_unlabelled_data = filter_unlabelled_data
#         self.initial_num_labels = initial_num_labels
#         self.query_size = query_size
#         # self.val_split = val_split
#         self._dataset: Optional[ActiveLearningDataset] = None

#         if not self.labelled:
#             raise MisconfigurationException("The labelled `LightningDataModule` should be provided.")
    
#     def prepare_data(self, *args: Any, **kwargs: Any) -> None:
#         self.labelled.prepare_data(*args, **kwargs)
    
#     def setup(self, stage: Optional[str] = None) -> None:
        
#         self.labelled.setup(stage)
        
#         if stage == "fit" or stage is None:
            
#             self._dataset = ActiveLearningDataset(
#                 self.labelled.dataset_train, labelled=self.map_dataset_to_labelled(self.labelled.dataset_train)
#             )

#             if not self.initial_num_labels:
#                 warnings.warn(
#                     "No labels provided for the initial step," "the estimated uncertainties are unreliable!", UserWarning
#                 )
#             else:
#                 self._dataset.label_randomly(self.initial_num_labels)
            
#             # if hasattr(self.labelled, "on_after_batch_transfer"):
#                 # self.on_after_batch_transfer = self.labelled.on_after_batch_transfer

#     @property
#     def has_test(self) -> bool:
#         return bool(self.labelled._test_input)

#     @property
#     def has_labelled_data(self) -> bool:
#         return self._dataset.n_labelled > 0

#     @property
#     def has_unlabelled_data(self) -> bool:
#         return self._dataset.n_unlabelled > 0

#     @property
#     def num_classes(self) -> Optional[int]:
#         return getattr(self.labelled, "num_classes", None) or getattr(self.unlabelled, "num_classes", None)

#     def train_dataloader(self) -> "DataLoader":
#         if self.val_split:
#             self.labelled._train_input = train_val_split(self._dataset, self.val_split)[0]
#         else:
#             self.labelled._train_input = self._dataset

#         if self.has_labelled_data and self.val_split:
#             self.val_dataloader = self._val_dataloader

#         if self.has_labelled_data:
#             return self.labelled.train_dataloader()
#         # Return a dummy dataloader, will be replaced by the loop
#         return DataLoader(["dummy"])

#     def _val_dataloader(self) -> "DataLoader":
#         self.labelled._val_input = train_val_split(self._dataset, self.val_split)[1]
#         dataloader = self.labelled._val_dataloader()
#         dataloader.collate_fn = create_worker_input_transform_processor(
#             RunningStage.TRAINING, self.labelled.input_transform
#         )
#         return dataloader

#     def _test_dataloader(self) -> "DataLoader":
#         return self.labelled.test_dataloader()

#     def predict_dataloader(self) -> "DataLoader":
#         self.labelled._predict_input = self.filter_unlabelled_data(self._dataset.pool)
#         dataloader = self.labelled._predict_dataloader()
#         dataloader.collate_fn = create_worker_input_transform_processor(
#             RunningStage.TRAINING, self.labelled.input_transform
#         )
#         return dataloader

#     def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
#         current_stage = self.trainer.state.stage
#         if current_stage == RunningStage.VALIDATING or current_stage == RunningStage.PREDICTING:
#             self.trainer.state.stage = RunningStage.TRAINING
#         batch = super().on_after_batch_transfer(batch, dataloader_idx)
#         self.trainer.state.stage = current_stage
#         return batch

#     def label(self, probabilities: List[torch.Tensor] = None, indices=None):
#         if probabilities is not None and indices:
#             raise MisconfigurationException(
#                 "The `probabilities` and `indices` are mutually exclusive, pass only of one them."
#             )
#         if probabilities is not None:
#             probabilities = torch.cat([p[0].unsqueeze(0) for p in probabilities], dim=0)
#             uncertainties = self.heuristic.get_uncertainties(probabilities)
#             indices = np.argsort(uncertainties)
#             if self._dataset is not None:
#                 self._dataset.label(indices[-self.query_size :])

#     def state_dict(self) -> Dict[str, torch.Tensor]:
#         return self._dataset.state_dict()

#     def load_state_dict(self, state_dict) -> None:
#         return self._dataset.load_state_dict(state_dict)
