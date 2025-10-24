""" PyTorch datsets for object detection and image classification """

# Imports
import os
import numpy as np
import pandas as pd
import torch
import cv2
import logging
from torch.utils.data import Dataset
import albumentations as alb
from albumentations.augmentations.geometric.resize import LongestMaxSize
from albumentations.augmentations.geometric.transforms import PadIfNeeded
from computervision.imageproc import ImageData, is_image, clipxywh
from computervision.transformations import DETRansform

logger = logging.getLogger(__name__)

def load_and_process_image(image_file_path, max_image_size=550):
    """
    Image preprocessing
    """
    # For the albumentations transformation, max_image_size needs to be of type 'int'
    if not isinstance(max_image_size, int):
        max_image_size = int(max_image_size)
    transform = alb.Compose([LongestMaxSize(max_size=max_image_size),
                             PadIfNeeded(min_height=max_image_size,
                                         min_width=max_image_size,
                                         border_mode=cv2.BORDER_CONSTANT,
                                         value=0)])
    im_raw = ImageData().load_image(image_file_path)
    im_output = transform(image=im_raw)['image']
    return im_output

class DETRdataset(Dataset):
    def __init__(self,
                 data: pd.DataFrame,
                 image_processor,
                 image_dir: str,
                 file_name_col: str,
                 label_id_col: str = None,
                 bbox_col: str = None,
                 bbox_format: dict = None,
                 transforms: list = None):

        self.data = data
        self.image_processor = image_processor
        self.image_dir = image_dir
        self.file_name_col = file_name_col
        self.label_id_col = label_id_col
        self.bbox_col = bbox_col
        self.transforms = transforms
        self.bbox_format = bbox_format
        if transforms is None:
            self.transforms = [alb.NoOp()]
        self.file_list = [os.path.join(image_dir, file) for file in list(data[file_name_col].unique())]
        assert self.validate()
        if bbox_format is None:
            self.bbox_format = {'format': 'coco',
                                'label_fields': ['tooth_position'],
                                'clip': True}
        assert self.bbox_format['format'] == 'coco', 'Only "coco" format is supported.'

    def validate(self):
        """ Making sure all images can be read """
        validated = np.sum([is_image(file) for file in self.file_list])
        output = False
        try:
            assert np.sum(validated) == len(self.file_list)
        except AssertionError:
            logger.warning(f'Could not validate all images: loaded {validated} / {len(self.file_list)} images.')
        else:
            logger.info(f'Validated {validated} images.')
            output = True
        return output

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        idx %= self.__len__()
        file = self.file_list[idx]
        file_name = os.path.basename(file)
        image = ImageData().load_image(file)
        # Convert to RGB
        if len(image.shape) == 2:
            image = ImageData().np2color(image)
        if any(var is None for var in [self.label_id_col, self.bbox_col]):
            transformation = alb.Compose(self.transforms)
            # Apply image transform
            transformed_im = transformation(image=image)['image']
            transformed_annotations = {'image_id': idx, 'annotations': []}
        else:
            bboxes = self.data.loc[self.data[self.file_name_col] == file_name, self.bbox_col].tolist()
            bboxes = [clipxywh(list(box), xlim=(0, image.shape[1]), ylim=(0, image.shape[0]), decimals=0) \
                      for box in bboxes]
            labels = self.data.loc[self.data[self.file_name_col] == file_name, self.label_id_col].tolist()
            # Apply image transform
            detr = DETRansform(bbox_format=self.bbox_format, transforms=self.transforms)
            transformed_im, transformed_annotations = detr. \
                format_transform(image=image, image_id=idx, bboxes=bboxes, labels=labels)

        # Apply the image processor to the augmentation transform
        processed = self.image_processor(images=transformed_im,
                                         annotations=transformed_annotations,
                                         return_tensors='pt')

        # The processor returns lists for "pixel_values" and annotations,
        # but we need only one image and the annotations for that image
        output = {k: v[0] for k, v in processed.items()}

        return output