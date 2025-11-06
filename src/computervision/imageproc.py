"""
Methods for image processing
Andreas Werdich
Core for Computational Biomedicine
Harvard Medical School, Boston, MA, USA
"""

import os
import copy
import numpy as np
import cv2
import logging
import itertools
from skimage import io
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

logger = logging.getLogger(name=__name__)

def plot_boxes(image,
               box_list,
               ax,
               label_list=None,
               color=None,
               linewidth=1.5,
               fontsize=8,
               cmap='grey',
               offset_xy=(0, 0)):
    offset_xy = (10 + offset_xy[0], 100 + offset_xy[1])
    # Take a ratio that looks good
    offset = (image.shape[1]*offset_xy[0]/2500,
              image.shape[0]*offset_xy[1]/1250)
    if color is None:
        # If no color is provided, color each box in a different color
        color_list = list(plt.cm.rainbow(np.linspace(0, 1, len(box_list))))
    else:
        color_list = [color]*len(box_list)
    ax.set(xticks=[], yticks=[])
    ax.imshow(image, cmap=cmap)
    # Loop over the bounding boxes
    for b, box in enumerate(box_list):
        rect = Rectangle(xy=(box[0], box[1]),
                         width=box[2],
                         height=box[3],
                         linewidth=linewidth,
                         edgecolor=color_list[b],
                         facecolor='none',
                         alpha=0.7)
        ax.add_patch(rect)
        if label_list is not None:
            ax.text(x=box[0]+offset[0], y=box[1]+offset[1], s=label_list[b], color=color_list[b], fontsize=fontsize)
    return ax


def clip_range(r, min_val=0, max_val=1):
    return max(min(r, max_val), min_val)

def flatten(list_of_lists: list) -> list:
    """
    Flattens a list of lists into a single list.
    """
    return list(itertools.chain.from_iterable(list_of_lists))

def transform_box(box_padded, img, pad_pixels=0):
    """
    Transform bounding box from padded to original image
    """
    img_h, img_w = img.shape[:2]
    x, y, w, h = box_padded - float(pad_pixels)
    output_box = tuple([clip_range(x, min_val=0, max_val=img_w),
                        clip_range(y, min_val=0, max_val=img_h),
                        clip_range(w, min_val=0, max_val=img_w),
                        clip_range(h, min_val=0, max_val=img_h)])
    return output_box

def xywh2xyxy(xywh):
    assert isinstance(xywh, list) and len(xywh)==4, 'input must be a bounding box [x, y, width, height]'
    return [xywh[0], xywh[1], xywh[0] + xywh[2], xywh[1] + xywh[3]]

def xyxy2xywh(xyxy):
    assert isinstance(xyxy, list) and len(xyxy)==4, 'input must be a bounding box [x_min, y_min, x_max, y_max]'
    return [xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]]

def clipxyxy(xyxy, xlim, ylim, decimals=0):
    assert isinstance(xyxy, list) and len(xyxy)==4, 'xyxy must be a bounding box [x_min, y_min, x_max, y_max]'
    assert len(xlim)==len(ylim)==2, 'xlim and xlim must be lists [min, max]'
    xyxy_clipped = [clip_range(xyxy[0], min_val=min(xlim), max_val=max(xlim)),
                    clip_range(xyxy[1], min_val=min(ylim), max_val=max(ylim)),
                    clip_range(xyxy[2], min_val=min(xlim), max_val=max(xlim)),
                    clip_range(xyxy[3], min_val=min(ylim), max_val=max(ylim))]

    if decimals is None:
        output = xyxy_clipped

    elif decimals==0:
        # Convert the output bounding box coordinates into integer values
        output = [np.int64(np.floor(r)) for r in xyxy_clipped]

    else:
        output = [round(r, ndigits=decimals) for r in xyxy_clipped]

    return output

def clipxywh(xywh, xlim, ylim, decimals=0):
    """
    Clips a bounding box defined in [x_min, y_min, width, height] format to specified limits.

    This function ensures that the bounding box defined in xywh format is clipped to the
    given x and y-axis limits. The returned bounding box coordinates remain within the specified bounds.
    """
    assert isinstance(xywh, list) and len(xywh)==4, 'xywh must be a bounding box [x_min, y_min, width, height]'
    assert len(xlim)==len(ylim)==2, 'xlim and xlim must be lists [min, max]'
    xyxy = xywh2xyxy(xywh)
    xyxy_clipped = clipxyxy(xyxy=xyxy, xlim=xlim, ylim=ylim, decimals=decimals)
    return xyxy2xywh(xyxy_clipped)


def yolo2xywh(yolo_bbox: list, image_width: int, image_height: int):
    # Consistency checks
    assert isinstance(yolo_bbox, list), 'input_box must be a list'
    assert isinstance(image_height, int), 'image_height must be int'
    assert isinstance(image_width, int), 'image_width must be int'
    assert all((c <= 1) & (c >= 0) for c in yolo_bbox)
    center_x = yolo_bbox[0] * image_width
    center_y = yolo_bbox[1] * image_height
    width = yolo_bbox[2] * image_width
    height = yolo_bbox[3] * image_height
    x_min = center_x - (width / 2)
    y_min = center_y - (height / 2)
    output_box = [x_min, y_min, width, height]
    return output_box


def xywh2yolo(coco_bbox: list, image_width: int, image_height: int, clip=True):
    # Consistency checks
    assert isinstance(coco_bbox, list), 'input_box must be a list'
    assert isinstance(image_height, int), 'image_height must be int'
    assert isinstance(image_width, int), 'image_width must be int'
    x_min, y_min, width, height = map(float, coco_bbox)
    if image_width <= 0 or image_height <= 0:
        raise ValueError("img_w and img_h must be positive.")
    # center in pixels
    center_x = x_min + (width / 2.0)
    center_y = y_min + (height / 2.0)
    # normalize
    x_rel = center_x / image_width
    y_rel = center_y / image_height
    w_rel = width / image_width
    h_rel = height / image_height
    if clip:
        x_rel = min(max(x_rel, 0.0), 1.0)
        y_rel = min(max(y_rel, 0.0), 1.0)
        w_rel = min(max(w_rel, 0.0), 1.0)
        h_rel = min(max(h_rel, 0.0), 1.0)
    output_box = [x_rel, y_rel, w_rel, h_rel]
    return output_box

def enclosing_box(bbox_list_xywh: list, offset:int) -> list:
    """
    Calculate and return a bounding box encapsulating all input bounding boxes with an
    optional offset.

    This function takes a list of bounding boxes in `xywh` format and computes a new
    bounding box that encompasses all of them with an additional offset. The conversion
    to and from `xywh` and `xyxy` formats is handled internally to facilitate calculations.

    Parameters:
    - bbox_list_xywh (list): A list of bounding boxes, each represented in the format [x, y, w, h].
    - offset (int): The amount to expand the final bounding box outward on all sides.

    Returns:
    - list: A single bounding box in `xywh` format representing the encompassing box
      of all input bounding boxes with the specified offset applied.
    """
    assert isinstance(bbox_list_xywh, list)
    bbox_list_xyxy = [xywh2xyxy(bbox) for bbox in bbox_list_xywh]
    bbox_list_x = flatten([[bbox[0], bbox[2]] for bbox in bbox_list_xyxy])
    bbox_list_y = flatten([[bbox[1], bbox[3]] for bbox in bbox_list_xyxy])
    quadrant_bbox_xywh = xyxy2xywh([min(bbox_list_x)-offset,
                                    min(bbox_list_y)-offset,
                                    max(bbox_list_x)+offset,
                                    max(bbox_list_y)+offset])
    return quadrant_bbox_xywh

def crop_image(image, box):
    """
    Crops an object in an image by bounding box
    :Parameters:
        image: (np.ndarray) image data
        box: (tuple) (x_min, y_min, width, height)
    :returns:
        crop_img: (np.ndarray) cropped image
    """
    x, y, w, h = [int(np.round(c)) for c in box]
    return image[y:y+h, x:x+w, :]

def is_image(image_file_path, min_area=500):
    """ Use the PIL package to check if a file is an image """
    file_is_image = False
    if os.path.exists(image_file_path):
        try:
            img = Image.open(image_file_path)
            width, height = img.size
            area = width * height
            assert area >= min_area
        except (IOError, AssertionError) as ex:
            logger.warning(f'File: {image_file_path} is not an image.')
        except Exception as ex:
            logger.warning(f'File: {image_file_path} is not an image. Exception: {ex}')
        else:
            file_is_image = True
    else:
        logger.warning(f'File: {image_file_path} does not exist.')
    return file_is_image

def validate_image_data(data_df, file_col, image_dir=None):
    output_df = copy.deepcopy(data_df)
    n_start = len(output_df)
    output_df = output_df.loc[output_df[file_col].\
        apply(lambda f: is_image(os.path.join(image_dir, f), min_area=500))]
    n_dropped = n_start - len(output_df)
    if n_dropped > 0:
        warning_msg = f'Warning: Dropped {n_dropped} rows!'
        logger.warning(warning_msg)
    else:
        logger.info('All files validated.')
    return output_df

def determine_bbox_format(bbox):
    """
    This is just a consistency check for the bounding boxes.
    May not be conclusive.
    Parameters:
        bbox (list or tuple): Bounding box, list of four numbers.
    Returns:
        str: 'xywh' if it's in COCO format, 'xyxy' if it's in Pascal VOC format, None if undetermined.
    """
    output = None
    if isinstance(bbox, (list, tuple, np.ndarray)) and len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        if all(x>=0 for x in bbox):
            if x2 > x1 and y2 > y1:
                # PASCAL: Here x2, y2 are max values, implying it represents bottom-right; x1, y1 as top-left
                output = 'xyxy'
            elif x2 > 0 and y2 > 0:
                # COCO: Here x2, y2 are width and height, but those should be larger than zero
                output = 'xywh'
    return output

class ImageData:
    """
    Represents image data processing functionality.

    This class provides various utilities for image manipulation and transformation, including
    image loading, padding, converting color schemes, resizing, and histogram equalization.
    The tools are designed to handle images represented as numpy arrays and perform image
    preprocessing tasks efficiently.
    """

    def __init__(self, resize=None):
        self.resize = resize
        self.image_net_mean = [0.485, 0.456, 0.406]
        self.image_net_std = [0.229, 0.224, 0.225]

    def load_image(self, image_path):
        """ Load image as np.ndarray
        Parameters:
            image_path: (str) complete path to image file
        Returns:
            output_array: (np.ndarray) (uint8)
        """
        output_array = None
        if os.path.exists(image_path):
            try:
                with open(image_path, mode='rb') as fl:
                    img = io.imread(fl)
            except Exception as ex:
                read_error_msg = f'Unable to read: {image_path}'
                print(read_error_msg)
                logger.error(read_error_msg)
            else:
                output_array = img.astype(np.uint8)
        else:
            error_msg = f'Image file: {image_path} does not exist.'
            print(error_msg)
            logger.error(error_msg)
        return output_array

    def np_square_pad(self, im_array, pad_pixels, pad_number=0):
        """ Pad 2D image
        Parameters:
            im_array: (np.ndarray) 2D numpy array
            pad_pixels: (int) pixels to add on each side
            pad_number: (int) gray value in [0, 256]
        """
        assert len(im_array.shape) == 2, f'Require 2D grayscale image.'

        def pad_with(vector, pad_width, iaxis, kwargs):
            pad_value = kwargs.get('padder', 10)
            vector[:pad_width[0]] = pad_value
            vector[-pad_width[1]:] = pad_value

        output = np.pad(im_array, pad_pixels, pad_with, padder=pad_number)
        return output

    def np2color(self, im_array, color_scheme='RGB'):
        """ Convert np.ndarray into color image
        Parameters:
            im_array: (np.ndarray) 2D or 3D numpy array
            color_scheme: (str) 'RGB', 'BGR' or 'GRAY'
        """
        assert color_scheme in ['RGB', 'BGR', 'GRAY'], f'color_scheme must be RGB, BGR or GRAY'
        if len(im_array.shape) == 2:
            color_code_str = f'cv2.COLOR_GRAY2{color_scheme}'
            image = cv2.cvtColor(im_array, code=eval(color_code_str))
        elif len(im_array.shape) == 3:
            if color_scheme == 'RGB':
                image = im_array.copy()
            else:
                color_code_str = f'cv2.COLOR_RGB2{color_scheme}'
                image = cv2.cvtColor(im_array, code=eval(color_code_str))
        else:
            raise NotImplementedError('input array must be 2D or 3D.')
        return image

    def convert_transparent_png(self, img):
        """ Decomposing the alpha channel for 4 channel png file """
        # image_4channel = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        output_image = img
        if len(img.shape) == 3 and img.shape[2] == 4:
            alpha_channel = img[:, :, 3]
            rgb_channels = img[:, :, :3]
            white_background_image = np.ones_like(rgb_channels, dtype=np.uint8) * 255
            alpha_factor = alpha_channel[:, :, np.newaxis].astype(np.float32) / 255.0
            alpha_factor = np.concatenate((alpha_factor, alpha_factor, alpha_factor), axis=2)
            base = rgb_channels.astype(np.float32) * alpha_factor
            white = white_background_image.astype(np.float32) * (1 - alpha_factor)
            output_image = (base + white).astype(np.uint8)
        return output_image

    def resize_image_list(self, image_list, output_size):
        """ Match size for multiple inputs before augmentation
        Parameters:
            image_list, list of images
            output_size, int or tuple, size of output image
        """
        assert any([isinstance(output_size, int),
                    isinstance(output_size, tuple)]), 'output_size must be of type int or tuple.'

        if isinstance(output_size, int):
            dim = (output_size, output_size)
        else:
            dim = output_size

        output_image_list = [cv2.resize(im, dim, interpolation=cv2.INTER_AREA) for im in image_list]
        return output_image_list

    def hist_eq(self, img, color_space='RGB'):
        """
        Enhances the contrast of an image using Contrast Limited Adaptive Histogram 
        Equalization (CLAHE) in the LAB color space. The method supports output in 
        different color spaces such as RGB, BGR, or grayscale.

        Parameters
        ----------
        img : numpy.ndarray
            Input image to be processed. It can be either grayscale or RGB.
        color_space : str, optional
            Specifies the desired output color space. Must be one of 'RGB', 'BGR', 
            or 'GRAY'. Defaults to 'RGB'.

        Returns
        -------
        numpy.ndarray
            Image with enhanced contrast in the specified color space.

        Raises
        ------
        ValueError
            If an unsupported color space is passed to the `color_space` parameter.
        """
        # If the image is grayscale, convert it to RGB
        if len(img.shape) != 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)
        lim_g = cv2.merge((cl, a, b))
        if color_space == 'RGB':
            enhanced_img = cv2.cvtColor(lim_g, cv2.COLOR_LAB2RGB)
        elif color_space == 'BGR':
            enhanced_img = cv2.cvtColor(lim_g, cv2.COLOR_LAB2BGR)
        elif color_space == 'GRAY':
            enhanced_img = cv2.cvtColor(lim_g, cv2.COLOR_LAB2RGB)
            enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2GRAY)
        else:
            raise ValueError('Unsupported color space')
        return enhanced_img

