""" Methods for transforming images and bounding boxes """
import numpy as np
import logging
import cv2
import albumentations as alb
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class AugmentationTransform:
    im_width: int = None
    im_height: int = None

    def get_transforms(self, name: str) -> list:

        if name == 'train_roboflow':
            crop_transforms = [alb.NoOp()]

            image_transforms = [
                alb.Affine(translate_percent=(-0.01, 0.01),
                           rotate=(-15, 15),
                           interpolation=cv2.INTER_LINEAR,
                           border_mode=cv2.BORDER_CONSTANT,
                           keep_ratio=True,
                           rotate_method='largest_box',
                           balanced_scale=True,
                           p=0.5),
                alb.CoarseDropout(num_holes_range=(1, 32),
                                  hole_height_range=(4, 50),
                                  hole_width_range=(4, 50),
                                  p=0.5),
                alb.RandomBrightnessContrast(p=0.5),
                alb.Sharpen(p=0.5),
                alb.CLAHE(p=0.5)]

        elif name == 'train_dentex':

            assert self.im_width is not None, 'im_width must be specified for training'
            assert self.im_height is not None, 'im_height must be specified for training'

            crop_transforms = [
                alb.RandomCropFromBorders(crop_left=0.25,
                                          crop_right=0.25,
                                          p=1.0),
                alb.CenterCrop(height=self.im_height,
                               width=self.im_width,
                               pad_if_needed=True, p=1.0)]

            image_transforms = [
                alb.Affine(translate_percent=(-0.01, 0.01),
                           rotate=(-15, 15),
                           interpolation=cv2.INTER_LINEAR,
                           border_mode=cv2.BORDER_CONSTANT,
                           keep_ratio=True,
                           rotate_method='largest_box',
                           balanced_scale=True,
                           p=0.5),
                alb.CoarseDropout(num_holes_range=(1, 50),
                                  hole_height_range=(4, 32),
                                  hole_width_range=(4, 32),
                                  p=0.5),
                alb.RandomBrightnessContrast(p=0.5),
                alb.Sharpen(p=0.5),
                alb.CLAHE(p=0.5)]

        elif name == 'val':
            crop_transforms = [alb.NoOp(p=1)]
            image_transforms = [alb.AutoContrast(p=1), alb.CLAHE(p=1)]

        elif name == 'test_set':

            # Augmentations for creating a test set from the Dentex dataset

            assert self.im_width is not None, 'im_width must be specified for creating test augmentations'
            assert self.im_height is not None, 'im_height must be specified for creating test augmentations'

            crop_transforms = [
                alb.RandomCropFromBorders(crop_left=0.25,
                                          crop_right=0.25,
                                          p=1.0),
                alb.CenterCrop(height=self.im_height,
                               width=self.im_width,
                               pad_if_needed=True, p=1.0)]

            image_transforms = [
                alb.Affine(translate_percent=(-0.01, 0.01),
                           rotate=(-15, 15),
                           interpolation=cv2.INTER_LINEAR,
                           border_mode=cv2.BORDER_CONSTANT,
                           keep_ratio=True,
                           rotate_method='largest_box',
                           balanced_scale=True,
                           p=0.5),
                alb.RandomBrightnessContrast(p=1.0)]

        else:
            logger.error('Transformation "{}" not implemented'.format(name))
            print('Transformation "{}" not implemented'.format(name))
            crop_transforms = [alb.NoOp(p=1)]
            image_transforms = [alb.NoOp(p=1)]

        transforms = crop_transforms + image_transforms

        return transforms


class DETRansform:
    """
    Class to handle transformations and formatting for object detection tasks.

    This class is primarily designed for transforming image data and bounding boxes to
    a format suitable for machine learning models like the RT-DETR model. It supports
    applying a set of transformations to images, as well as handling bounding boxes and
    label fields according to a specified format. Additionally, it produces annotated
    inputs required by specific the RT-DETR object detection model.

    Attributes:
        transformations: List of transformations to be applied to the images.
                         If not provided, a default transformation is applied.
        bbox_format: Dictionary specifying the format of bounding boxes and
                     related label fields. It includes configurations such as
                     the format type, label fields, whether to clip bounding
                     boxes, and a minimum area threshold.

    Methods:
        transform(image, bboxes: list, label_fields: list):
            Applies transformations to the input image, bounding boxes,
            and label fields. Maintains consistency with the specified
            bounding box format and label fields. Returns the transformed
            image, bounding boxes, and updated related fields.

        format_transform(image, image_id, bboxes: list, labels: list):
            Formats the transformed outputs to be compatible with the RT-DETR
            model. Converts bounding boxes and labels to the required format,
            ensures input consistency, and generates annotations suitable
            for the model.
    """
    def __init__(self, transforms: list = None, bbox_format: dict = None):
        self.transforms = transforms
        self.bbox_format = bbox_format
        if transforms is None:
            self.transforms = [alb.NoOp()]
        if bbox_format is None:
            self.bbox_format = {'format': 'coco',
                                'label_fields': ['quadrants', 'positions'],
                                'clip': True,
                                'filter_invalid_bboxes': True,
                                'min_area': 5000}

    def transform(self, image, bboxes: list, label_fields: list):

        # The label_fields should be one list for each field
        try:
            assert len(label_fields) == len(self.bbox_format.get('label_fields'))
            labels = dict(zip(self.bbox_format.get('label_fields'), label_fields))
        except Exception as e:
            print(
                f'The argument "label_fields" must be a list of lists with labels: {self.bbox_format.get("label_fields")}')

        # The bboxes variable must be a list: convert to (N x 4) numpy array
        assert isinstance(bboxes, list)
        bbox_array = np.array(bboxes).reshape(len(bboxes), 4)

        # Set up the transformation
        transformation = alb.Compose(self.transforms, bbox_params=alb.BboxParams(**self.bbox_format))
        transformed = transformation(image=image, bboxes=bbox_array, **labels)

        # Create the output
        output = {'image': transformed['image'], 'bboxes': list(transformed['bboxes'].astype(int))}
        output.update({field: transformed[field] for field in self.bbox_format.get('label_fields')})

        return output

    def format_transform(self, image, image_id, bboxes: list, labels: list):
        """ This method produces the formatted input for the RT-DETR model. """

        # Consistency checks
        assert len(bboxes) == len(labels), 'We need as many labels as bounding boxes: len(bboxes) == len(labels)!'
        assert len(self.bbox_format.get('label_fields')) == 1, 'We can only use one set of labels.'
        assert self.bbox_format.get(
            'format') == 'coco', f'Bounding box format must be "coco", but is: {self.bbox_format.get("format")}!'
        assert isinstance(image_id, int), 'Image ID must be of type int.'
        assert all(isinstance(l, int) for l in labels), 'All labels must be class IDs (int).'

        # Transform the image
        output = self.transform(image=image, bboxes=bboxes, label_fields=[labels])
        output_image = output['image']
        output_bboxes = output['bboxes']
        output_labels = output.get(self.bbox_format.get('label_fields')[0])

        # Annotations for the model using the transformed image, bounding boxes and labels
        annotation_list = []
        # This list can only contain data if the transformed output contains at least one bounding box
        if len(output_labels) > 0:
            for bbox, label in zip(output_bboxes, output_labels):
                assert len(bbox) == 4, f'Incompatible bounding box: {bbox}'
                annotation = {'image_id': image_id,
                              'category_id': int(label),
                              'bbox': list(bbox),
                              'iscrowd': 0,
                              'area': bbox[2] * bbox[3]}
                annotation_list.append(annotation)
        output_annotations = {'image_id': image_id, 'annotations': annotation_list}
        return output_image, output_annotations