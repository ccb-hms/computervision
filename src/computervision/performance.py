""" Performance metrics for computer vision models """

import numpy as np
import pandas as pd
import torch
from torchvision import ops
from sklearn import metrics
from computervision.imageproc import xywh2xyxy, clipxywh

class DetectionMetrics:
    def __init__(self,
                 true_df: pd.DataFrame,
                 pred_df: pd.DataFrame,
                 file_col: str,
                 label_col: str,
                 bbox_col: str,
                 score_col: str,
                 im_width: int = None,
                 im_height: int = None,
                 bbox_format: str = 'xywh'):
        self.true_df = true_df
        self.pred_df = pred_df
        self.file_col = file_col
        self.label_col = label_col
        self.bbox_col = bbox_col
        self.score_col = score_col
        if im_width is not None and im_height is not None:
            self.x_lim = (0, im_width)
            self.y_lim = (0, im_height)
        else:
            self.x_lim = None
            self.y_lim = None
        self.bbox_format = bbox_format
        assert bbox_format == 'xywh', 'bbox_format should be in COCO "xywh" format'

    def classify_predictions_df(self, iou_threshold=0.5):
        pred_df = self.pred_df.loc[~self.pred_df[self.label_col].isnull()]
        file_list = sorted(list(set(self.true_df[self.file_col].tolist()). \
                                intersection(pred_df[self.file_col].tolist())))
        classifications_df_list = []
        missed_df_list = []
        for f, file in enumerate(file_list):
            true_bboxes = self.true_df.loc[self.true_df[self.file_col] == file, self.bbox_col].tolist()
            pred_bboxes = pred_df.loc[pred_df[self.file_col] == file, self.bbox_col].tolist()
            true_bboxes = [list(np.int64(box)) for box in true_bboxes]
            pred_bboxes = [list(np.int64(box)) for box in pred_bboxes]
            true_labels = self.true_df.loc[self.true_df[self.file_col] == file, self.label_col].tolist()
            pred_labels = pred_df.loc[pred_df[self.file_col] == file, self.label_col].tolist()
            pred_scores = pred_df.loc[pred_df[self.file_col] == file, self.score_col].tolist()

            pred_cl = self.\
                classify_predictions(true_labels=true_labels,
                                     true_bboxes=true_bboxes,
                                     pred_labels=pred_labels,
                                     pred_bboxes=pred_bboxes,
                                     iou_threshold=iou_threshold). \
                rename(columns={'pred_label': self.label_col})

            pred_cl.insert(loc=0, column=self.file_col, value=file)
            pred_cl.insert(loc=1, column='iou_threshold', value=iou_threshold)
            pred_cl.insert(loc=2, column=self.score_col, value=pred_scores)
            pred_cl.insert(loc=3, column=self.bbox_col, value=pred_bboxes)

            classifications_df_list.append(pred_cl)

            # False negatives: Labels in the ground truth data that were not detected
            missed_label_list = sorted(list(set(true_labels).difference(pred_labels)))

            if len(missed_label_list) > 0:
                missed_cl = pd.DataFrame({self.label_col: missed_label_list})
                missed_cl.insert(loc=0, column=self.file_col, value=file)
                missed_df_list.append(missed_cl)

        if len(classifications_df_list) > 0:
            classifications = pd.concat(classifications_df_list, axis=0, ignore_index=True)
            classifications = classifications. \
                sort_values(by=[self.label_col, self.score_col], ascending=True). \
                reset_index(drop=True)
        else:
            classifications = None

        if len(missed_df_list) > 0:
            missed = pd.concat(missed_df_list, axis=0, ignore_index=True)
            missed = missed. \
                sort_values(by=self.label_col, ascending=True). \
                reset_index(drop=True)
        else:
            missed = None

        return classifications, missed

    def ap_from_classifications(self, classifications):
        """ Calculate Precision - Recall curves and AP for each label """

        pr_df_list = []
        label_list = sorted(list(classifications[self.label_col].unique()))

        for label in label_list:
            # Total number of positives in the data set (TP + FN)
            n_labels = len(self.true_df.loc[self.true_df[self.label_col] == label])

            # Predictions for this class sorted by score in descending order
            classifications_label = classifications. \
                loc[classifications[self.label_col] == label]. \
                sort_values(by='score', ascending=False). \
                reset_index(drop=True)

            # Calculate precision and recall for each row
            correct = classifications_label['TP'].tolist()

            # precision = true positives / all detections
            precision = [sum(correct[:i + 1]) / (i + 1) for i in range(len(correct))]

            # recall = true positives / samples with this label in ground truth data
            recall = [sum(correct[:i + 1]) / n_labels for i in range(len(correct))]

            # Add precision and recall to the data frame for this label
            classifications_label = classifications_label. \
                assign(precision=precision, recall=recall)

            # Calculate precision and recall independent from the bounding box
            # We count every prediction that is in the image as positive
            # Detections that were not in the image did not get an iou value (FP)

            # TP + FP
            n_detections = len(classifications_label)
            # TP: all detections for that class with a ground truth label, so IoU >= 0
            n_detections_with_iou = len(classifications_label.loc[~classifications_label['IoU'].isnull()])
            # We can add a precision and recall value that is just for this class, indepdendent from the bounding box
            precision_label = n_detections_with_iou / n_detections
            recall_label = n_detections_with_iou / n_labels

            # Calculate the AUC
            auc = metrics.auc(x=recall, y=precision)

            # Add the class-level precision/recall values to the data frame
            classifications_label = classifications_label. \
                assign(precision_label=precision_label,
                       recall_label=recall_label,
                       auc=auc)

            pr_df_list.append(classifications_label)

        pr_df = pd.concat(pr_df_list, axis=0, ignore_index=True)

        return pr_df

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