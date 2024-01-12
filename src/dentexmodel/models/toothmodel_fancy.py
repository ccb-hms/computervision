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
# Torchmetrics
import torchmetrics.classification as tmc

logger = logging.getLogger(name=__name__)


def performance_metrics(metric_dict, logits, target, metric_prefix='train'):
    """
    Calculate performance metrics for a given set of logits and target labels.
    """
    preds = nn.Softmax(dim=1)(logits)
    performance_dict = {}
    for metric_name, metric in metric_dict.items():
        performance_dict.update({f'{metric_prefix}_{metric_name}': metric(preds=preds, target=target)})
    return performance_dict


def average_performance_metrics(step_metrics_list, decimals=3):
    """
    Calculate the average performance from a list of dictionaries
    """
    average_metrics = {}
    for metric in step_metrics_list[0].keys():
        metric_value = torch.stack([x.get(metric) for x in step_metrics_list])
        # Remove any zero values before averaging
        metric_value = metric_value[metric_value.nonzero().squeeze()]
        metric_value = metric_value.mean().detach().cpu().numpy().round(decimals)
        average_metrics[metric] = metric_value
    return average_metrics


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
                 num_classes,
                 num_workers=1,
                 lr=1.0e-3,
                 model=None):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_classes = num_classes
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
        # Performance metrics
        self.metrics = nn.ModuleDict({
            'accuracy': tmc.MulticlassAccuracy(num_classes=num_classes, average='micro'),
            'precision': tmc.MulticlassPrecision(num_classes=num_classes, average='macro'),
            'recall': tmc.MulticlassRecall(num_classes=num_classes, average='macro'),
            'f1': tmc.MulticlassF1Score(num_classes=num_classes, average='macro'),
            'auroc': tmc.MulticlassAUROC(num_classes=num_classes, average='macro')
        })
        self.train_step_metrics_list = []
        self.val_step_metrics_list = []
        self.test_step_metrics_list = []

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        dl = DataLoader(self.train_dataset,
                        batch_size=self.batch_size,
                        num_workers=self.num_workers,
                        shuffle=True,
                        pin_memory=True)
        return dl

    def val_dataloader(self) -> EVAL_DATALOADERS:
        dl = DataLoader(self.val_dataset,
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
        image, target = batch
        pred = self.forward(image)
        loss = self.criterion(pred, target)
        train_step_metrics = {'train_loss': loss}
        performance_dict = performance_metrics(metric_dict=self.metrics,
                                               logits=pred,
                                               target=target,
                                               metric_prefix='train')
        train_step_metrics.update(performance_dict)
        self.train_step_metrics_list.append(train_step_metrics)
        return loss

    def validation_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        image, target = batch
        pred = self.forward(image)
        loss = self.criterion(pred, target)
        val_step_metrics = {'val_loss': loss}
        performance_dict = performance_metrics(metric_dict=self.metrics,
                                               logits=pred,
                                               target=target,
                                               metric_prefix='val')
        val_step_metrics.update(performance_dict)
        self.val_step_metrics_list.append(val_step_metrics)
        return loss

    def predict_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> Any:
        image, label = batch
        output = self.forward(image)
        return output

    def configure_optimizers(self) -> OptimizerLRScheduler:
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return opt

    def on_train_epoch_end(self) -> None:
        epoch_train_metrics = average_performance_metrics(step_metrics_list=self.train_step_metrics_list,
                                                          decimals=self.decimals)
        self.log_dict(epoch_train_metrics, prog_bar=True)
        self.train_step_metrics_list.clear()

    def on_validation_epoch_end(self) -> None:
        epoch_val_metrics = average_performance_metrics(step_metrics_list=self.val_step_metrics_list,
                                                        decimals=self.decimals)
        self.log_dict(epoch_val_metrics, prog_bar=True)
        self.val_step_metrics_list.clear()

