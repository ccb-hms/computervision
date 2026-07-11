## Object detection pipeline ##

Dental enumeration on full panoramic X-rays for the [Dentex Challenge 2023](./data.md): detect
each tooth and classify its FDI/ADA position. Notebooks live in `notebooks/detection/` and
fine-tune a Hugging Face **RT-DETRv2** model (`PekingU/rtdetr_v2_r101vd`) — this pipeline does not
use Detectron2 (that's the [segmentation pipeline](./segmentation.md)).

### Steps

| # | Notebook | What it does |
|---|---|---|
| 01 | `01_download_dataset.ipynb` | Downloads the Dentex quadrant-enumeration data (and a pre-cropped RT-DETR training set), builds the FDI↔ADA tooth-position mapping, saves as parquet. |
| 02 | `02_check_image_data.ipynb` | Sanity-checks image counts and plots a sample X-ray with ground-truth boxes by quadrant. |
| 03 | `03_crop_images.ipynb` | Crops full panoramic X-rays into quadrant-combination sub-images to reduce input resolution, recomputes/saves cropped annotations. |
| 04 | `04_augmentations.ipynb` | Defines `DETRansform`, a custom Albumentations `BboxParams` (COCO format) wrapper tailored to DETR's label format. |
| 05 | `05_datasplit.ipynb` | Train/val/test split (deterministic crops for val/test), saved as parquet. |
| 06 | `06_dataset.ipynb` | `DTRdataset` PyTorch dataset built around Hugging Face `RTDetrImageProcessor`; visualizes a sample batch. |
| 07 | `07_train_dentex.ipynb` | Fine-tunes `RTDetrV2ForObjectDetection` from the pretrained checkpoint via HF `Trainer` with a custom `collate_fn`. |
| 08 | `08_inference.ipynb` | Runs inference with a trained checkpoint on test images; also compares predictions against Claude Opus's own bounding-box predictions on the same X-ray. |
| 09 | `09_performance.ipynb` | Loads all training checkpoints, classifies TP/FP/FN, computes precision/recall and average precision at IoU 0.25/0.5/0.75 (`computervision.performance.DetectionMetrics`), plots AP bar charts and PR curves. |

### Key library code

- `computervision.transformations.DETRansform` — bbox-aware augmentation for the DETR label format.
- `computervision.inference` — RT-DETR inference helpers (`RTDetrV2ForObjectDetection`, `RTDetrImageProcessor`).
- `computervision.performance.DetectionMetrics` — precision/recall/AP computation.
- `computervision.mapeval` — mean-average-precision utilities built on `torchmetrics`.

See [architecture.md](./architecture.md) for the full module map and [data.md](./data.md) for
dataset sources and the on-disk data layout.
