"""
Tooth Model Fancy
Improved model class for training and inference
With validation steps and TensorBoard logging
Andreas Werdich
Center for Computational Biomedicine
Harvard Medical School
"""

import logging
from typing import Any

# Pytorch imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights

# Lightning module
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler, STEP_OUTPUT
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS

logger = logging.getLogger(name=__name__)


class ResNet50Model:
    """ This is the ResNet50 model from torchvision.models
    We use this model with nn.CrossEntropyLoss() which does NOT require a softmax at the output
    """

    def __init__(self, n_outputs=4):
        self.n_outputs = n_outputs

    def create_model(self):
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        model.fc = nn.Sequential(
            nn.Linear(in_features=model.fc.in_features, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=self.n_outputs)
        )
        return model


class ToothModel(LightningModule):
    def __init__(self,
                 train_dataset,
                 val_dataset,
                 test_dataset,
                 batch_size,
                 num_workers=1,
                 lr=1.0e-3,
                 model=None):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lr = lr
        self.decimals = 5
        # Model architecture
        if model is None:
            self.model = ResNet50Model().create_model()
        else:
            self.model = model
        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        dl = DataLoader(self.train_dataset,
                        batch_size=self.batch_size,
                        num_workers=self.num_workers,
                        shuffle=True,
                        pin_memory=True)
        return dl

    def val_dataloader(self) -> EVAL_DATALOADERS:
        dl = DataLoader(self.val_datase,
                        batch_size=self.batch_size,
                        num_workers=self.num_workers,
                        shuffle=False,
                        pin_memory=True)
        return dl

    def test_dataloader(self) -> EVAL_DATALOADERS:
        dl = DataLoader(self.test_dataset,
                        batch_size=self.batch_size,
                        num_workers=self.num_workers,
                        shuffle=False,
                        pin_memory=True)
        return dl

    def forward(self, x, *args: Any, **kwargs: Any) -> Any:
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        image, label = batch
        pred = self.forward(image)
        loss = self.criterion(pred, label)
        return loss

    def predict_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> Any:
        image, label = batch
        output = self.forward(image)
        return output

    def configure_optimizers(self) -> OptimizerLRScheduler:
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return opt
