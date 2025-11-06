""" Run predictions for the HSDM CLIP prototype """
import sys
import os
import glob
import json
import logging
import datetime
import timeit
import copy
import pandas as pd
import numpy as np
from pathlib import Path

import computervision
from computervision.inference import DETRinference, get_gpu_info
from computervision.imageproc import validate_image_data
from computervision.transformations import AugmentationTransform
from computervision.datasets import DETRdataset
from computervision.fileutils import chunks, Flag

#%% GPU availability
device, device_str = get_gpu_info()

#%% Directories and files
data_dir = os.path.join(os.environ.get('DATA_DIR'), 'projects', 'hsdm')
print(data_dir)

image_dir = os.path.join(data_dir, 'image_output_250710')
# Data frame with image directories and file names
df_file_name = 'Images_250710_exported_wFlag.parquet'
df_file = os.path.join(image_dir, df_file_name)

# Model name and checkpoint
model_name = 'rtdetr_dtx_rbf_hsdm_251018_02'
model_dir = os.path.join(data_dir, 'toothmodel', model_name)
checkpoint = 6240
checkpoint_dir = os.path.join(model_dir, f'checkpoint-{checkpoint}')
model_config_file = os.path.join(model_dir, f'{model_name}.json')

# Prediction threshold and other information about the process
threshold = 0.1
n_files_per_chunk = 2000
file_base_name = 'prototype_251030'

# Let's create an output directory to save some examples
output_dir = os.path.join(model_dir, 'output')
output_images_dir = os.path.join(output_dir, 'examples')
Path(output_images_dir).mkdir(parents=True, exist_ok=True)
flg = Flag(flag_dir=output_dir)
# And we also we want to save some logs for the preciction process
log_dir = os.path.join(output_dir, 'log')
Path(log_dir).mkdir(parents=True, exist_ok=True)

# Load the model configuration
with open(model_config_file, mode='r') as file:
    model_config = json.load(file)
print(*list(model_config.keys()), sep='\n')

# Load the model
dtr = DETRinference(device_name='cuda:0',
                    checkpoint_path=checkpoint_dir,
                    batch_size=256)
transforms = AugmentationTransform().get_transforms(name='val')
bbox_format = model_config.get('bbox_format')

#%% Log file
date_str = datetime.date.today().strftime('%y%m%d')
log_file_name = f'predict_log_{date_str}.log'
log_file = os.path.join(log_dir, log_file_name)
dtfmt = '%y%m%d-%H:%M'
logfmt = '%(asctime)s-%(name)s-%(levelname)s-%(message)s'

logging.basicConfig(filename=log_file,
                    filemode='w',
                    level=logging.INFO,
                    format=logfmt,
                    datefmt=dtfmt)

logger = logging.getLogger(name=__name__)

#%% Data filescolumn_list = ['file_hash', 'file_dir', 'prototype_flag']
column_list = ['file_hash', 'file_dir', 'prototype_flag']
df = pd.read_parquet(df_file, columns=column_list)

dset_col = 'dset'
label_col = 'pos'
file_col = 'file_name'
bbox_col = 'bbox'
score_col = 'score'

# Create a file_name column
df[file_col] = df['file_dir'] + '/' + df['file_hash'] + '.png'

print(df.shape)
print(len(df['file_hash'].unique()))

# Filter the prototype flag
df = df.loc[df['prototype_flag']]
print(df.shape)

# File list
# n_test = 100
# rng = np.random.default_rng(seed=123)
# file_list = rng.choice(df[file_col].unique(), size=100)
file_list = df[file_col].unique()

#%% Helper function for running the predicitons
def predict_df(data_df, dtr_model, score_threshold, augmentation, boxformat):
    data_df_validated = copy.deepcopy(data_df)
    data_df_validated = validate_image_data(data_df_validated, file_col=file_col, image_dir=image_dir)
    dataset = DETRdataset(data=data_df_validated,
                          image_processor=dtr_model.processor,
                          image_dir=image_dir,
                          file_name_col=file_col,
                          label_id_col=None,
                          bbox_col=None,
                          bbox_format=boxformat,
                          transforms=augmentation)
    # Convert the output image number id to the file name
    file_hash_list = [os.path.splitext(os.path.basename(file_f))[0] for file_f in dataset.file_list]
    id2hash = dict(zip(range(len(file_hash_list)), file_hash_list))
    # Convert the output category to the label (position)
    id2label = {int(category_id): int(label) for category_id, label in model_config.get('id2label').items()}
    # Run the model on the images
    pred = dtr_model.predict_on_dataset(dataset, threshold=score_threshold)
    pred = pred.assign(model=model_name,
                       checkpoint=checkpoint,
                       threshold=threshold)
    pred['file_hash'] = pred['image_id'].apply(lambda image_id: id2hash.get(image_id))
    pred[label_col] = pred['category_id'].apply(lambda category_id: id2label.get(category_id))
    # Filter out rows with images that do not have predictions
    pred = pred.loc[~pred[label_col].isnull()]
    # Merge with the original data frame
    df_output = data_df.merge(pred, on='file_hash', how='left')
    df_output = df_output.drop(columns=['image_id', 'batch']).reset_index(drop=True)
    return df_output

#%% Run predictions
file_chunk_list = list(chunks(file_list, n=n_files_per_chunk))
n_files = len(file_chunk_list)
mag = int(np.ceil(np.log(n_files + 1) / np.log(10)))
start_time = timeit.default_timer()
for c, file_chunk in enumerate(file_chunk_list):
    c_name = f'{file_base_name}_{str(c).zfill(mag)}'
    flag_exist_list = flg.find_flags()
    if c_name not in flag_exist_list:
        flg.set_flag_file(flag_base=c_name, flag='started')
        if (c + 1) % 20 == 0:
            dt = np.round((timeit.default_timer() - start_time) / 60)
            print(f'Starting file {c + 1} / {n_files} @ time: {dt} minutes.')
        # Running chunk c
        df_test = df.loc[df[file_col].isin(file_chunk)].\
            reset_index(drop=True)
        df_pred = predict_df(data_df=df_test,
                             dtr_model=dtr,
                             score_threshold=threshold,
                             augmentation=transforms, boxformat=bbox_format)
        if len(df_pred) > 0:
            output_file = os.path.join(output_dir, f'{c_name}.parquet')
            df_pred.to_parquet(output_file)
            n_files_input = len(df_test[file_col].unique())
            n_files_output = len(df_pred[file_col].unique())
            assert n_files_input == n_files_output
            flg.set_flag_file(flag_base=c_name, flag='success')
        else:
            logger.warning(f'No predictions for chunk {c + 1} / {len(file_chunk_list)}')
    else:
        print(f'File {c_name} processing or completed. Skipping.')