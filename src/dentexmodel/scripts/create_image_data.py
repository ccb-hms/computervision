"""
Create Dentex data set for baseline performance testing
Andreas Werdich
Center for Computational Biomedicine
"""

import os
import glob
import pandas as pd
from pathlib import Path
import tarfile
import json
import cv2

from dentexmodel.fileutils import FileOP
from dentexmodel.imageproc import ImageData, crop_image

# Directories and settings
dentex_dir = os.path.join(os.environ['HOME'], 'data', 'dentex')
data_dir = os.path.join(dentex_dir, 'dentex_disease')
Path(data_dir).mkdir(parents=True, exist_ok=True)
url = 'https://dsets.s3.amazonaws.com/dentex/dentex-quadrant-enumeration-disease.tar.gz'

def create_image_data(dataset_url=url):

    """ Download and create the dentex annotated data set """

    # Download data
    data_tar_file = FileOP().download_from_url(dataset_url, download_dir=data_dir)
    if data_tar_file is not None and os.path.exists(data_tar_file):
        with tarfile.open(data_tar_file, 'r') as tar:
            tar.extractall(data_dir)
    image_dir = os.path.join(data_dir, 'quadrant-enumeration-disease', 'xrays')
    file_list = glob.glob(os.path.join(image_dir, '*.png'))

    # Create the data frame with the file paths
    file_name_list = [os.path.basename(file) for file in file_list]
    im_number_list = [int(os.path.splitext(file)[0].rsplit('_', maxsplit=1)[-1]) for file in file_name_list]
    files = pd.DataFrame({'image_number': im_number_list,
                          'file_name': file_name_list,
                          'file_path': file_list}).\
                    sort_values(by='image_number', ascending=True).reset_index(drop=True)

    # Annotation file
    annotation_file = glob.glob(os.path.join(data_dir, 'quadrant-enumeration-disease', '*.json'))
    if len(annotation_file) > 0:
        annotation_file = annotation_file[0]
        print(f'Annotation data file: {annotation_file}')
        with open(annotation_file, 'r') as file:
            js = json.load(file)
        js_im = js.get('images')
        js_an = js.get('annotations')
        print(f'Found {len(js_an)} annotations for {len(js_im)} images.')
    else:
        annotation_file = None
    # Add image ids to the files data frame
    js_im_df = pd.DataFrame(js_im).\
                    merge(files, on='file_name', how='inner').\
                    sort_values(by='id', ascending=True).\
                    reset_index(drop=True)

    # Tooth locations and disease classifications
    quadrant_df = pd.DataFrame(js.get('categories_1'))
    num_df = pd.DataFrame(js.get('categories_2'))
    cl_df = pd.DataFrame(js.get('categories_3'))

    # Transfer of the annotations from the json file into the data frame
    an_df_list = []
    for idx, an_dict in enumerate(js_an):
        if (idx + 1) % 500 == 0:
            print(f'Annotation {idx + 1} / {len(js_an)}')
        id = an_dict.get('image_id')
        id_df = js_im_df.loc[js_im_df['id'] == id]
        # Extract the annotations for this annotation id
        quadrant = quadrant_df.loc[quadrant_df['id'] == an_dict.get('category_id_1'), 'name'].values[0]
        position = num_df.loc[num_df['id'] == an_dict.get('category_id_2'), 'name'].values[0]
        id_df = id_df.assign(quadrant=quadrant,
                             position=position,
                             label=cl_df.loc[cl_df['id'] == an_dict.get('category_id_3'), 'name'].values[0],
                             area=[an_dict.get('area')],
                             bbox=[an_dict.get('bbox')],
                             box_name=(f'{os.path.splitext(id_df["file_name"].values[0])[0]}_'
                                       f'{quadrant}_{position}'))
        an_df_list.append(id_df)
    an_df = pd.concat(an_df_list, axis=0, ignore_index=True)

    # Add the number of annotations to each image
    n_annotations = an_df[['file_name', 'label']].\
                    groupby('file_name').count().\
                    reset_index(drop=False).\
                    rename(columns={'label': 'annotations'})
    an_df = an_df.merge(n_annotations, on='file_name', how='inner').\
                    sort_values(by='id', ascending=True).\
                    reset_index(drop=True)

    # Save the data frame with the file paths and annotations
    df_file_name = 'dentex_disease_dataset.parquet'
    df_file = os.path.join(dentex_dir, df_file_name)
    an_df.to_parquet(df_file)
    print(f'Annotation data frame saved: {df_file}')

    # Start a list of new data frames
    data_df_list = []

    # Loop over the panoramic x-rays
    cropped_image_dir = os.path.join(image_dir, 'crop')
    Path(cropped_image_dir).mkdir(parents=True, exist_ok=True)
    print(f'Cropping images. Please wait.')
    file_name_list = sorted(an_df['file_name'].unique())
    for f, file_name in enumerate(file_name_list):
        box_name_list = an_df.loc[an_df['file_name'] == file_name, 'box_name'].values
        if (f + 1) % 50 == 0:
            print(f'Processing image {f + 1} / {len(file_name_list)}')
        # Loop over the bounding boxes for this file
        for b, box_name in enumerate(box_name_list):
            box_file = os.path.join(cropped_image_dir, f'{box_name}.png')
            # Get the row in the data frame
            box_df = an_df.loc[(an_df['file_name'] == file_name) & (an_df['box_name'] == box_name)]. \
                assign(box_file=box_file)
            box = box_df['bbox'].values[0]
            bbox = box[0], box[1], box[0] + box[2], box[1] + box[3]
            file = os.path.join(image_dir, file_name)
            # We can skip this part if the data already exists
            if not os.path.exists(box_file):
                im = ImageData().load_image(file)
                im_crop = crop_image(im, bbox)
                # Some contrast enhancement
                im_crop_enhanced = ImageData().hist_eq(im_crop)
                # Save the image
                cv2.imwrite(box_file, cv2.cvtColor(im_crop_enhanced, cv2.COLOR_RGB2BGR))
            # Add the image size to the data frame
            box_file_size = ImageData().image_size(box_file)
            box_df = box_df.assign(im_width=box_file_size[1],
                                   im_height=box_file_size[0])
            # Add the data frame for this image to the list
            data_df_list.append(box_df)
    # Concatenate the data frames
    data_df = pd.concat(data_df_list, axis=0, ignore_index=True)

    # Save the data frame
    df_box_file = df_file.replace('dataset', 'cropped_dataset')
    data_df.to_parquet(df_box_file)

    return data_df



