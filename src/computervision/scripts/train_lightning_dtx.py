
#%% Imports
import sys
import os
import json
import pandas as pd
import albumentations as alb
from pathlib import Path

from computervision.fileutils import FileOP
import computervision as cv
from computervision.imageproc import ImageData, is_image
from computervision.transformations import AugmentationTransform
from computervision.datasets import DatasetFromDF
from computervision.inference import get_gpu_info
from computervision.models.lightningmodel import ToothModel

# Lightning module
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

#%% Directories and files
# Main data directory (defined as environment variable in docker-compose.yml)
dataset_name = 'dentex_classification'
data_dir = os.path.join(os.environ.get('DATA'), 'dentex', dataset_name)

# Download directory (change as needed)
model_dir = os.path.join(data_dir, 'model')

# This image directory is where the xrays are in the archive, so should be left as-is
raw_image_dir = os.path.join(data_dir, 'quadrant-enumeration-disease', 'xrays')
image_dir = os.path.join(raw_image_dir, 'crop')

data_file_name = 'dentex_disease_datasplit.parquet'
data_file = os.path.join(data_dir, data_file_name)

#%% Load annotations
annotations_file_name = 'dentex_disease_datasplit.parquet'
annotations_file = os.path.join(data_dir, annotations_file_name)
df = pd.read_parquet(annotations_file)

file_col = 'file_path'
bbox_col= 'bbox'
label_col = 'label'
dset_col = 'dataset'

labels = sorted(list(df[label_col].unique()))
label2id = dict(zip(labels, range(len(labels))))
id2label = {category_id: label for label, category_id in label2id.items()}

# Now we can add a category id to the data frame
df = df.assign(category=df[label_col].apply(lambda label: label2id.get(label)))

# Check the images
file_list = [os.path.join(image_dir, file_name) for file_name in df[file_col].unique()]
checked = [is_image(file) for file in file_list]
assert len(file_list) == sum(checked), f'WARNING: Could not open all {len(file_list)} images at: {image_dir}'
print(f'Image directory:        {image_dir}')
print(f'Total number of images: {len(file_list)}')
print(f'Annotations:            {df.shape[0]}')

#%% Augmentations
# Initial scaling and padding for the bigger dimension
max_image_size = 550

# Model input size
im_width, im_height = 224, 224
train_transforms = AugmentationTransform(im_width=im_width, im_height=im_height).\
                get_transforms(name='train_transform')

# Resize and then normalize
# with ImageNet mean and standard deviation for ResNet50
image_net_mean = ImageData().image_net_mean
image_net_std = ImageData().image_net_std

# This transform is essential and needs to be applied for both training and validation
resize_and_normalize = [alb.Resize(width=im_width, height=im_height),
                        alb.Normalize(mean=image_net_mean, std=image_net_std)]
train_transforms.extend(resize_and_normalize)
train_transform = alb.Compose(train_transforms)

# However, for validation and testing, we don't want the augmentations,
# so we just resize and normalize the data
test_transform = alb.Compose(resize_and_normalize)

#%% Datasets
train_dataset = DatasetFromDF(data=df.loc[df[dset_col] == 'train'],
                              image_dir=image_dir,
                              file_name_col=file_col,
                              label_id_col='category',
                              max_image_size=max_image_size,
                              transform=train_transform,
                              validate=True)


val_dataset = DatasetFromDF(data=df.loc[df[dset_col] == 'val'],
                            image_dir=image_dir,
                            file_name_col=file_col,
                            label_id_col='category',
                            max_image_size=max_image_size,
                            transform=test_transform,
                            validate=True)

test_dataset = DatasetFromDF(data=df.loc[df[dset_col] == 'test'],
                             image_dir=image_dir,
                             file_name_col=file_col,
                             label_id_col='category',
                             max_image_size=max_image_size,
                             transform=test_transform,
                             validate=True)

#%% Model parameters
device_number = 0
device, device_str = get_gpu_info(device_number=device_number)

model_name = 'dentexmodel'
model_version = 2
model_version_str = str(model_version).zfill(2)

model_name_dir = os.path.join(model_dir, model_name)
checkpoint_dir = os.path.join(model_name_dir, f'{model_name}_{model_version_str}')
Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

model_info = {'model_version': model_version_str,
              'device_number': device_number,
              'project_version': cv.__version__,
              'model_name': model_name,
              'image_dir': image_dir,
              'model_dir': checkpoint_dir,
              'im_width': im_width,
              'im_height': im_height,
              'max_image_size': max_image_size}

training_args = {'max_epochs': 5,
                 'num_classes': 6,
                 'num_workers': 2,
                 'batch_size': 16,
                 'initial_lr': 1.0e-3,
                 'check_val_every_n_epoch': 1,
                 'checkpoint_very_n_epoch': 2,
                 'save_top_k': 3}

# Save the model parameters
parameters = {'model_info': model_info,
              'id2label': id2label,
              'training_args': training_args}

json_file = os.path.join(checkpoint_dir, f'{model_name}.json')
with open(json_file, 'w') as f:
    json.dump(parameters, f, indent=4)

#%% Create the model
model = ToothModel(train_dataset=train_dataset,
                   val_dataset=val_dataset,
                   test_dataset=test_dataset,
                   batch_size=training_args.get('batch_size'),
                   num_classes=training_args.get('num_classes'),
                   num_workers=training_args.get('num_workers'),
                   lr=training_args.get('initial_lr'))

#%% Logger and checkpoints
# Directory to save checkpoints and logs
chk_callback = ModelCheckpoint(dirpath=checkpoint_dir,
                               filename='model-{epoch}',
                               monitor='val_loss',
                               mode='min',
                               save_last=True,
                               every_n_epochs=training_args.get('checkpoint_every_n_epoch'),
                               save_on_train_epoch_end=True,
                               save_top_k=training_args.get('save_top_k'))

# Setup logger
logger = TensorBoardLogger(save_dir=checkpoint_dir,
                           name='log')

lr_monitor = LearningRateMonitor(logging_interval='epoch',
                                 log_momentum=True)

#%% Training
seed_everything(42)
tr = Trainer(max_epochs=training_args.get('max_epochs'),
             default_root_dir=checkpoint_dir,
             callbacks=[chk_callback, lr_monitor],
             logger=logger,
             check_val_every_n_epoch=training_args.get('check_val_every_n_epoch'))

tr.fit(model)