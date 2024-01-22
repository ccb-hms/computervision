"""
Train the dentex toothmodel_facny version
"""

# Imports
import os
from pathlib import Path
import pandas as pd

# PyTorch packages
import torch

# Lightning library
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, LearningRateFinder
from lightning.pytorch.loggers import TensorBoardLogger

# Albumentations library
import albumentations as alb

# Package imports
from dentexmodel.models.toothmodel_fancy import ToothModel
from dentexmodel.imageproc import ImageData
from dentexmodel.torchdataset import DatasetFromDF

# %% Directories and files
# dentex_dir = os.path.join(os.environ['HOME'], 'data', 'dentex')
dentex_dir = os.path.join(os.environ['data_dir'], 'dentex')
data_dir = os.path.join(dentex_dir, 'dentex_disease')
image_dir = os.path.join(data_dir, 'quadrant-enumeration-disease', 'xrays', 'crop')
data_file_name = 'dentex_disease_datasplit.parquet'
data_file = os.path.join(dentex_dir, data_file_name)

# Model parameters and name
seed = 234
max_im_size = 550
im_size = 224
model_name = 'dtx240122'
model_version = 1
max_epochs = 300
num_classes = 4
num_workers = 12
batch_size = 64
check_val_every_n_epoch = 1
checkpoint_every_n_epoch = 2
save_top_k = 10
initial_learning_rate = 5.0e-4

# Logs
log_dir = os.path.join(dentex_dir, 'log')
checkpoint_dir = os.path.join(log_dir, model_name, f'version_{model_version}', 'checkpoints')
Path(log_dir).mkdir(parents=True, exist_ok=True)
Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

# Logger
logger = TensorBoardLogger(save_dir=log_dir,
                           name=model_name,
                           version=model_version)

# %% Select device

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(torch.cuda.is_available())
print(f'Number of GPUs found:  {torch.cuda.device_count()}')
print(f'Current device ID:     {torch.cuda.current_device()}')
print(f'GPU device name:       {torch.cuda.get_device_name(0)}')
print(f'CUDNN version:         {torch.backends.cudnn.version()}')
torch.set_float32_matmul_precision(precision='high')

# %% Load the data frame with the file names
data_df = pd.read_parquet(data_file)
# Select the samples for training, validation and testing from our data frame
train_df = data_df.loc[data_df['dataset'] == 'train']
val_df = data_df.loc[data_df['dataset'] == 'val']
test_df = data_df.loc[data_df['dataset'] == 'test']

train_samples = sorted(list(train_df['box_name'].unique()))
print(f'Found {len(train_samples)} samples in the training set.')
val_samples = sorted(list(val_df['box_name'].unique()))
print(f'Found {len(val_samples)} samples in the validation set.')
test_samples = sorted(list(test_df['box_name'].unique()))
print(f'Found {len(test_samples)} samples in the test set.')

# %% Augmentations
train_transform = alb.Compose([
    alb.Resize(im_size + 32, im_size + 32),
    alb.RandomCrop(im_size, im_size),
    alb.HorizontalFlip(),
    alb.ShiftScaleRotate(),
    alb.Blur(),
    alb.RandomGamma(),
    alb.Sharpen(),
    alb.GaussNoise(),
    alb.CoarseDropout(16, 32, 32),
    alb.CLAHE(),
    alb.Normalize(mean=ImageData().image_net_mean,
                  std=ImageData().image_net_std)])

val_transform = alb.Compose([
    alb.Resize(im_size, im_size),
    alb.Normalize(mean=ImageData().image_net_mean,
                  std=ImageData().image_net_std)])

# %% PyTorch data sets from the data frames

train_dataset = DatasetFromDF(data=train_df,
                              file_col='box_file',
                              label_col='cl',
                              max_image_size=max_im_size,
                              transform=train_transform,
                              validate=True)

val_dataset = DatasetFromDF(data=val_df,
                            file_col='box_file',
                            label_col='cl',
                            max_image_size=max_im_size,
                            transform=val_transform,
                            validate=True)

test_dataset = DatasetFromDF(data=test_df,
                             file_col='box_file',
                             label_col='cl',
                             max_image_size=max_im_size,
                             transform=val_transform,
                             validate=True)

# %% Initialize the model

model = ToothModel(train_dataset=train_dataset,
                   val_dataset=val_dataset,
                   test_dataset=test_dataset,
                   batch_size=batch_size,
                   num_classes=num_classes,
                   num_workers=num_workers,
                   lr=initial_learning_rate)

# %% Callbacks

checkpoint_callback = ModelCheckpoint(
    dirpath=checkpoint_dir,
    filename='{epoch}',
    monitor='val_loss',
    mode='min',
    save_last=True,
    every_n_epochs=checkpoint_every_n_epoch,
    save_on_train_epoch_end=True,
    save_top_k=save_top_k)

lr_monitor = LearningRateMonitor(logging_interval='epoch',
                                 log_momentum=True)

# %% Trainer instance
seed_everything(seed=seed, workers=True)
trainer = Trainer(accelerator='gpu',
                  max_epochs=max_epochs,
                  check_val_every_n_epoch=check_val_every_n_epoch,
                  logger=logger,
                  callbacks=[checkpoint_callback, lr_monitor])

# Run training
trainer.fit(model)
