## Instance segmentation pipeline ##

Detectron2-based instance segmentation on the [Dentex Challenge dataset](./data.md): trains a
Faster R-CNN detection baseline and then a Mask R-CNN instance-segmentation model. Notebooks live
in `notebooks/segmentation/`.

### Steps

| # | Notebook | What it does |
|---|---|---|
| 01 | `01_download_segmentation.ipynb` | Downloads the Dentex quadrant-enumeration dataset via `DentexData`, builds a file/annotation DataFrame, saves `dentex_detection_dataset.parquet`. |
| 02 | `02_train_val_test_split_segmentation.ipynb` | Stratified train/val/test split (`val_test_split`, 50 examples per class for val/test), saves `dentex_detection_datasplit.parquet`. |
| 03 | `03_annotations.ipynb` | Converts raw Dentex JSON into per-split COCO-style annotation files (`create_rcnn_anntations`) for Detectron2; verifies by displaying sample annotated images. |
| 04 | `04_train_detection.ipynb` | Registers Detectron2 datasets and trains a Faster R-CNN (`faster_rcnn_R_101_FPN_3x`) object-detection baseline via a custom `Trainer`. |
| 05 | `05_train_segmentation.ipynb` | Same setup, trains a Mask R-CNN (`mask_rcnn_X_101_32x8d_FPN_3x`) instance-segmentation model. |
| 06 | `06_test_segmentation.ipynb` | Loads a saved checkpoint (or falls back to a public S3 checkpoint, `toothsegmentation_1K.pth`), runs predictions/metrics on the registered test set, displays example predictions. |
| 07 | `07_predict_segmentation.ipynb` | Near-identical to 06, framed as applying the trained model to new/unseen images rather than formal test-set evaluation. |

### Key library code

- `computervision.detector` — Detectron2 training/evaluation support (`Trainer`, `COCOEvaluator` hooks).
- `computervision.dentexdata.DentexData` — dataset download and annotation parsing, shared with the detection pipeline.

See [architecture.md](./architecture.md) for the full module map and [data.md](./data.md) for
dataset sources and the on-disk data layout.
