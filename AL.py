from flash.image.classification.integrations.baal import (
    ActiveLearningDataModule,
)
from ndropALLoop import ndrop_ActiveLearningLoop
from flash.image import ImageClassifier, ImageClassificationData
from flash.core.classification import LogitsOutput

from baal.active.dataset import ActiveLearningDataset
from baal.active import get_heuristic

import os

import torch
from pytorch_lightning import LightningModule, Trainer
from torch import nn
from torch.nn import functional as F
from torchmetrics import Accuracy

from torchvision import datasets
from torchvision.transforms import transforms

from functools import partial

from plBaaLData import ActiveLearningDataModuleWrapper
from pl_bolts.datamodules import CIFAR10DataModule

#################################################

IMG_SIZE = 32
# https://lightning-flash.readthedocs.io/en/latest/reference/image_classification.html#custom-transformations

# train_transforms = transforms.Compose(
#     [
#         transforms.Resize((IMG_SIZE, IMG_SIZE)),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomRotation(30),
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#     ]
# )
# test_transforms = transforms.Compose(
#     [
#         transforms.Resize((IMG_SIZE, IMG_SIZE)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#     ]
# )

class DataModule_(ImageClassificationData):
    @property
    def num_classes(self):
        return 10

def get_data_module(heuristic, data_path):
    # train_set = datasets.CIFAR10(data_path, train=True, download=True)
    # test_set = datasets.CIFAR10(data_path, train=False, download=True)
    # dm = DataModule_.from_datasets(
    #     train_dataset=train_set,
    #     test_dataset=test_set,
    #     # train_transform=train_transforms,
    #     # test_transform=test_transforms,
    #     # Do not forget to set `predict_transform`,
    #     # this is what we will use for uncertainty estimation!
    #     # predict_transform=test_transforms,
    #     transform_kwargs=dict(image_size=(32, 32)),
    #     batch_size=64,
    # )
    # active_dm = ActiveLearningDataModule(
    #     dm,
    #     heuristic=get_heuristic(heuristic),
    #     initial_num_labels=1024,
    #     query_size=100,
    #     val_split=0.0,
    # )
    # assert active_dm.has_test, "No test set?"
    # return active_dm
    
    active_dm = ActiveLearningDataModuleWrapper(CIFAR10DataModule)(
        data_dir = "./data",
        
        heuristic=get_heuristic(heuristic),
        initial_num_labels=1024,
        query_size=100,
        val_split=0.01
    )
    return active_dm

#################################################################

def get_model(dm):
    loss_fn = nn.CrossEntropyLoss()
    head = nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(512, 512),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(512, dm.num_classes),
    )
    LR = 0.001
    model = ImageClassifier(
        num_classes=dm.num_classes,
        head=head,
        backbone="vgg16",
        pretrained=True,
        loss_fn=loss_fn,
        optimizer=partial(torch.optim.SGD, momentum=0.9, weight_decay=5e-4),
        learning_rate=LR,
        # serializer=Logits(),  # Note the serializer to Logits to be able to estimate uncertainty.
    )
    model.output = LogitsOutput()
    return model

#################################################################

class SimpleModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(32 * 32 * 3, 10)
        self.accuracy = Accuracy()

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.accuracy(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.accuracy, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)
    
    def predict_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for predicting
        x, y = batch
        return self(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

#################################################################
        
active_dm = get_data_module('random', './data')

# Init our model
model = SimpleModel()
# model = get_model(active_dm)

aloop = ndrop_ActiveLearningLoop(
    label_epoch_frequency = 1,
    inference_iteration = 1
)

# Initialize a trainer
trainer = Trainer(
    gpus=1,
    max_epochs=10,
    progress_bar_refresh_rate=20,
    
    limit_val_batches = 0.0
)

aloop.connect(trainer.fit_loop)
trainer.fit_loop = aloop

# Train the model ⚡
trainer.fit(
    model, 
    datamodule = active_dm
)