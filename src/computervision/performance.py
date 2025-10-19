""" Performance metrics for computer vision models """

import numpy as np
import pandas as pd
import torch
from torchvision import ops
from computervision.imageproc import xywh2xyxy, clipxywh

class DetectionMetrics:
    def __init__(self, im_width: int = None, im_height: int = None, bbox_format: str = 'xywh'):
        if im_width is not None and im_height is not None:
            self.x_lim = (0, im_width)
            self.y_lim = (0, im_height)
        else:
            self.x_lim = None
            self.y_lim = None
        self.bbox_format = bbox_format
        assert bbox_format == 'xywh', 'bbox_format should be in COCO "xywh" format'

    def classify_predictions(self,
                             true_labels: list,
                             true_bboxes: list,
                             pred_labels: list,
                             pred_bboxes: list,
                             iou_threshold: float = 0.5) -> pd.DataFrame:

        # Make sure that the true and pred labels are lists
        assert all([isinstance(true_labels, list), isinstance(pred_labels, list)])
        assert all([isinstance(true_bboxes, list), isinstance(pred_bboxes, list)])

        # Make sure that the true and pred labels are the same length
        assert len(true_labels) == len(true_bboxes), 'labels and bboxes (true) must be the same length'
        assert len(pred_labels) == len(pred_bboxes), 'labels and bboxes (pred) must be the same length'

        # Clip bounding boxes to image dimensions if image sizes are provided
        if self.x_lim is not None and self.y_lim is not None:
            true_bboxes = [clipxywh(bbox, xlim=self.x_lim, ylim=self.y_lim, decimals=0) for bbox in true_bboxes]
            pred_bboxes = [clipxywh(bbox, xlim=self.x_lim, ylim=self.y_lim, decimals=0) for bbox in pred_bboxes]
        else:
            true_bboxes = [list(np.int64(box)) for box in true_bboxes]
            pred_bboxes = [list(np.int64(box)) for box in pred_bboxes]

        # Total number of predictions that were missed
        missed = sorted(list(set(true_labels).difference(pred_labels)))

        # Classify predictions (TP:1, FP:0)
        iou_list = []
        prediction_list = []
        for p, p_label in enumerate(pred_labels):
            p_bbox = pred_bboxes[p]
            p_prediction = 0  # FP
            # Check if the predicted label is in the ground truth labels
            # If a prediction does not have a ground truth label, the iou is NaN
            p_iou = np.nan
            pt_iou_list = []
            for t, t_label in enumerate(true_labels):
                if p_label == t_label:
                    t_bbox = true_bboxes[t]
                    pt_iou = DetectionMetrics.compute_iou(p_bbox, t_bbox, bbox_format='xywh', method='pt')
                    pt_iou_list.append(pt_iou)
            if len(pt_iou_list) > 0:
                p_iou = np.max(pt_iou_list)
                if p_iou >= iou_threshold:
                    p_prediction = 1  # TP
            prediction_list.append(p_prediction)
            iou_list.append(p_iou)

        pred_df = pd.DataFrame({'pred_label': pred_labels,
                                'TP': prediction_list,
                                'IoU': iou_list})

        pred_df = pred_df.assign(n_missed=len(missed),
                                 duplicate_TP=False)

        output_df = pred_df.copy()
        # Flip duplicate TP predictions for the same label with FP
        output_df.loc[(pred_df.duplicated(subset=['pred_label', 'TP'])) & (pred_df['TP'] == 1), 'TP'] = 0
        output_df.loc[(pred_df.duplicated(subset=['pred_label', 'TP'])) & (pred_df['TP'] == 1), 'duplicate_TP'] = True

        return output_df


    @staticmethod
    def compute_iou(bbox_1: list, bbox_2: list, bbox_format: str = 'xywh', method: str = 'np') -> float:
        assert method in ['np', 'pt'], 'method should be either "np" or "pt"'
        assert bbox_format in ['xyxy', 'xywh'], 'bbox_format should be either "xyxy" or "xywh"'
        iou = None
        if bbox_format == 'xywh':
            bbox_1, bbox_2 = xywh2xyxy(bbox_1), xywh2xyxy(bbox_2)
        if method == 'np':
            ix1 = np.maximum(bbox_1[0], bbox_2[0])
            iy1 = np.maximum(bbox_1[1], bbox_2[1])
            ix2 = np.minimum(bbox_1[2], bbox_2[2])
            iy2 = np.minimum(bbox_1[3], bbox_2[3])
            # Intersection height and width.
            i_height = np.maximum(iy2 - iy1 + 1, np.array(0.))
            i_width = np.maximum(ix2 - ix1 + 1, np.array(0.))
            area_of_intersection = i_height * i_width
            # Ground Truth dimensions.
            gt_height = bbox_1[3] - bbox_1[1] + 1
            gt_width = bbox_1[2] - bbox_1[0] + 1
            # Prediction dimensions.
            pd_height = bbox_2[3] - bbox_2[1] + 1
            pd_width = bbox_2[2] - bbox_2[0] + 1
            area_of_union = gt_height * gt_width + pd_height * pd_width - area_of_intersection
            iou = area_of_intersection / area_of_union
        elif method == 'pt':
            bbox_tensors = [torch.tensor([bbox_1], dtype=torch.float),
                            torch.tensor([bbox_2], dtype=torch.float)]
            iou = ops.box_iou(bbox_tensors[0], bbox_tensors[1]).item()
        return iou