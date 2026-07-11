## Repository architecture: `src/computervision` ##

The installable package lives at `src/computervision/` (entry point `computervision.main`,
declared in `pyproject.toml`). It splits into shared library modules used by the notebooks, a
`models/` subpackage, standalone training `scripts/`, and static Label Studio interface configs.

### Core modules

| Module | Purpose |
|---|---|
| `fileutils.py` | Generic file operations — downloading from a URL, extracting tar/gzip archives (`FileOP`). |
| `imageproc.py` | Image processing utilities: loading/checking images (`ImageData`, `is_image`), bounding-box format conversions (`xywh2xyxy`, `clipxywh`, etc.). |
| `transformations.py` | Augmentation wrappers: `AugmentationTransform` (classification, resize/pad + Albumentations) and `DETRansform` (detection, bbox-aware for the DETR label format). |
| `datasets.py` | PyTorch `Dataset` classes for classification and detection. |
| `dentexdata.py` | Core Dentex Challenge data handling: download, JSON annotation parsing, FDI/ADA tooth-position mapping, train/val/test splitting (`DentexData`, `val_test_split`). |
| `dentex.py` | Thin `Dentex` helper for downloading the detection dataset via `DT_URL`. |
| `detector.py` | Detectron2 training/evaluation support — custom `Trainer`, `COCOEvaluator` hooks, used by the segmentation pipeline. |
| `inference.py` | Inference helpers for the RT-DETRv2 detection model (`RTDetrV2ForObjectDetection`, `RTDetrImageProcessor`), plus GPU info utilities. |
| `mapeval.py` | Prototype mean-average-precision evaluation built on `torchmetrics.detection.mean_ap`. |
| `performance.py` | `DetectionMetrics` — precision/recall/AP computation and TP/FP/FN classification for the detection pipeline. |
| `cudacheck.py` | Small script/module to verify CUDA/GPU availability inside the environment. |

### `models/`

- `lightningmodel.py` — the `ToothModel` PyTorch Lightning module: a ResNet50 backbone
  (`torchvision.models.resnet50`) with automated learning-rate scheduling
  (`ReduceLROnPlateau`) and TensorBoard logging. Used throughout the classification pipeline.

### `scripts/`

Standalone, non-notebook Python scripts mirroring parts of the notebook pipelines for
command-line / batch use, mainly around Dentex and Roboflow training:

- `train_dentex_01.py` … `train_dentex_08.py` — successive stages of the Dentex classification
  training pipeline (numbering roughly mirrors `notebooks/classification/dentex/`).
- `train_dentex_hsdm.py`, `predict_hsdm.py`, `predict_hsdm_prototype.py` — a related "HSDM"
  training/prediction variant.
- `train_lightning.py`, `train_lightning_dtx.py` — PyTorch Lightning training entry points for
  the Roboflow and Dentex classification models, respectively.
- `train_roboflow_01.py` — Roboflow classification training script.
- `cudacheck.py` — CLI-runnable GPU/CUDA check.

These are convenience scripts, not part of the automated test suite — treat the notebooks as the
source of truth for the documented pipeline steps.

### `labelinterface/`

Static HTML label-interface configurations for [Label Studio](./labelstudio.md)
(`label_interface1.html` … `label_interface_classification.html`, `sam_example_*.html` for
Segment-Anything–assisted labeling). These are imported into Label Studio project configs, not
executed directly.

### Package metadata

- `AUTHORS` — attribution (Center for Computational Biomedicine, HMS).
- `__init__.py` — exposes `__version__` (via installed package metadata) and `__authors__`, and
  a `main()` entry point used by the `computervision` console script.
