""" Inference methods for computer vision """

import numpy as np
import pandas as pd
import logging
import torch
from torch.utils.data import DataLoader, Dataset
from computervision.imageproc import xyxy2xywh, clipxywh
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor

logger = logging.getLogger(name=__name__)

# GPU checks
def get_gpu_info(device_number: int = None):

    if device_number is None:
        is_cuda = torch.cuda.is_available()
        print(f'CUDA available: {is_cuda}')
        print(f'Number of GPUs found:  {torch.cuda.device_count()}')
        if is_cuda:
            print(f'Current device ID: {torch.cuda.current_device()}')
            print(f'GPU device name:   {torch.cuda.get_device_name(0)}')
            print(f'PyTorch version:   {torch.__version__}')
            print(f'CUDA version:      {torch.version.cuda}')
            print(f'CUDNN version:     {torch.backends.cudnn.version()}')
            device_str = 'cuda:0'
            torch.cuda.empty_cache()
        else:
            device_str = 'cpu'
    else:
        device_str = f'cuda:{device_number}'

    info_msg = f'Current device:    {device_str}'
    print(info_msg)
    device = torch.device(device_str)
    logger.info(info_msg)

    return device, device_str

class DETRinference:
    """DETR inference class"""
    def __init__(self, device_name=None, checkpoint_path=None, batch_size=4):
        if device_name is None:
            device, device_name = get_gpu_info()
        self.device = torch.device(device_name)
        if checkpoint_path is None:
            raise ValueError('checkpoint_path must be provided')
        self.checkpoint_path = checkpoint_path
        self.batch_size = batch_size
        self.model, self.processor = self.load_model(checkpoint_path)

    def collate_fn(self, batch):
        data = {"pixel_values": torch.stack([x["pixel_values"] for x in batch]).to(self.device),
                "labels": [x["labels"].to(self.device) for x in batch]}
        return data

    def load_model(self, checkpoint_path):
        """Load DETR model"""
        model = RTDetrV2ForObjectDetection.from_pretrained(checkpoint_path, device_map=self.device)
        processor = RTDetrImageProcessor.from_pretrained(checkpoint_path)
        return model, processor

    def create_dataloader(self, dataset: Dataset) -> DataLoader:
        """Create a dataloader for inference"""
        dataloader = DataLoader(dataset,
                                batch_size=self.batch_size,
                                collate_fn=self.collate_fn)
        return dataloader

    def predict_on_image(self, image, threshold):
        """ Predict bounding boxes and labels for an image """
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            model_outputs = self.model(**inputs)
        output = self.processor.\
            post_process_object_detection(model_outputs,
                                          target_sizes=torch.tensor([image.shape[:2]]),
                                          threshold=threshold)
        output_boxes = output[0].get('boxes').cpu().numpy()
        if len(output_boxes) > 0:
            bboxes = [clipxywh(xyxy2xywh(list(box)),
                               xlim=(0, image.shape[1]),
                               ylim=(0, image.shape[0]),
                               decimals=0) for box in output_boxes]
            areas = [int(box[2] * box[3]) for box in bboxes]
            categories = list(output[0].get('labels').cpu().numpy())
            scores = output[0].get('scores').cpu().numpy()
            categories = [int(c) for c in categories]
            scores = [np.around(s, decimals=4) for s in scores]
            predictions = {'bbox': bboxes, 'area': areas, 'category_id': categories, 'score': scores}
        else:
            logger.warning(f'No predictions for threshold {threshold}.')
            predictions = None
        return predictions

    def predict_on_batch(self, batch, threshold):
        """ Predict bounding boxes and labels for a batch of images """
        size_list = [label['orig_size'] for label in batch['labels']]
        target_sizes = torch.stack(size_list)
        output_batch = None
        with torch.no_grad():
            outputs = self.model(**batch)
            output_batch = self.processor.\
                post_process_object_detection(outputs=outputs,
                                              target_sizes=target_sizes,
                                              threshold=threshold)
        if len(output_batch) == 0:
            logger.warning(f'No predictions for threshold {threshold}.')
        return output_batch

    def predict_dataloader(self, dataloader, threshold, report=10):
        """ Predict bounding boxes and labels for a dataloader """
        pred_df_list = []
        for b, batch in enumerate(dataloader):
            if (b +1) % report == 0:
                print(f'Predicting batch {b+1} of {len(dataloader)}.')
            image_id_list = [int(label.get('image_id').cpu().numpy()[0]) for label in batch['labels']]
            image_size_list = [tuple(label['orig_size'].cpu().numpy()) for label in batch['labels']]
            output_batch = self.predict_on_batch(batch=batch, threshold=threshold)
            for image_id, image_size, output in zip(image_id_list, image_size_list, output_batch):
                height, width = image_size
                x_lim, y_lim = (0, width), (0, height)
                scores = list(output.get('scores').cpu().numpy())
                labels = list(output.get('labels').cpu().numpy())
                box_list = list(output.get('boxes').cpu().numpy())
                box_list = [clipxywh(xyxy2xywh(list(box)), xlim=x_lim, ylim=y_lim, decimals=0) for box in box_list]
                area_list = [box[2] * box[3] for box in box_list]
                pred = {'category_id': labels, 'bbox': box_list, 'score': scores, 'area': area_list}
                pred_df = pd.DataFrame(pred)
                if len(pred_df) == 0:
                    image_id, width, height = [image_id], [width], [height]
                pred_df.insert(loc=0, column='image_id', value=image_id)
                pred_df.insert(loc=1, column='image_width', value=width)
                pred_df.insert(loc=2, column='image_height', value=height)
                pred_df.insert(loc=3, column='batch', value=b)
                pred_df = pred_df.astype('object')
                pred_df_list.append(pred_df)
        pred_df = pd.concat(pred_df_list, axis=0, ignore_index=True)
        return pred_df

    def predict_on_dataset(self, dataset, threshold):
        """ Predict bounding boxes and labels for a dataset """
        dataloader = self.create_dataloader(dataset)
        pred_df = self.predict_dataloader(dataloader, threshold)
        return pred_df