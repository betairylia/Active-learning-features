from argparse import ArgumentParser
from typing import Any, Callable, Optional, Sequence, Union

from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from pl_bolts.datasets import TrialCIFAR10
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pl_bolts.utils import _TORCHVISION_AVAILABLE
# from pl_bolts.utils.stability import under_review
from pl_bolts.utils.warnings import warn_missing_pkg

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms as transform_lib
    from torchvision.datasets import SVHN
else:  # pragma: no cover
    warn_missing_pkg("torchvision")
    SVHN = None

def WrappedSVHN(_, datadir, train = True, **kwargs):
    if train == True:
        return SVHN(datadir, split = 'train', **kwargs)
    else:
        return SVHN(datadir, split = 'test', **kwargs)

class SVHNDataModule(VisionDataModule):
    """
    SVHN dataset from pytorch
    """

    name = "svhn"
    dataset_cls = WrappedSVHN
    dims = (3, 32, 32)

    def __init__(
        self,
        data_dir: Optional[str] = None,
        val_split: Union[int, float] = 0.2,
        num_workers: int = 0,
        normalize: bool = False,
        batch_size: int = 32,
        seed: int = 42,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            data_dir: Where to save/load the data
            val_split: Percent (float) or number (int) of samples to use for the validation split
            num_workers: How many workers to use for loading data
            normalize: If true applies image normalize
            batch_size: How many samples per batch to load
            seed: Random seed to be used for train/val/test splits
            shuffle: If true shuffles the train data every epoch
            pin_memory: If true, the data loader will copy Tensors into CUDA pinned memory before
                        returning them
            drop_last: If true drops the last incomplete batch
        """
        super().__init__(  # type: ignore[misc]
            data_dir=data_dir,
            val_split=val_split,
            num_workers=num_workers,
            normalize=normalize,
            batch_size=batch_size,
            seed=seed,
            shuffle=shuffle,
            pin_memory=pin_memory,
            drop_last=drop_last,
            *args,
            **kwargs,
        )

    @property
    def num_samples(self) -> int:
        train_len, _ = self._get_splits(len_dataset=50_000)
        return train_len

    @property
    def num_classes(self) -> int:
        """
        Return:
            10
        """
        return 10

    def default_transforms(self) -> Callable:
        if self.normalize:
            cf10_transforms = transform_lib.Compose([transform_lib.ToTensor(), cifar10_normalization()])
        else:
            cf10_transforms = transform_lib.Compose([transform_lib.ToTensor()])

        return cf10_transforms

    @staticmethod
    def add_dataset_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--data_dir", type=str, default=".")
        parser.add_argument("--num_workers", type=int, default=0)
        parser.add_argument("--batch_size", type=int, default=32)

        return parser
