## Image classification pipeline ##

Single-tooth image classification: crop individual teeth out of annotated dental X-rays, then
train a ResNet50 classifier (wrapped in a project `ToothModel` PyTorch Lightning class) to
classify each cropped tooth.

The pipeline exists in two parallel forms that share the same 10-step structure but target
different data sets:

- `notebooks/classification/*.ipynb` — the generic [Roboflow dental dataset](./data.md) (`CL_URL`).
- `notebooks/classification/dentex/*.ipynb` — the [Dentex Challenge dataset](./data.md) (`DT_URL`),
  using `computervision.dentexdata.DentexData` instead of the generic download/crop helpers, and
  adding `torchmetrics`-based evaluation in the later steps.

### Steps (both variants)

| # | Notebook | What it does |
|---|---|---|
| 01 | `01_download.ipynb` / `dentex/01_dentex_download.ipynb` | Downloads the dataset tarball and parses annotations into a DataFrame. |
| 02 | `02_prepare_dataset.ipynb` / `dentex/02_create_dataset.ipynb` | Crops individual tooth bounding boxes out of the full images into per-tooth classification samples. |
| 03 | `03_train_val_test_split.ipynb` / `dentex/03_train_val_test_split.ipynb` | Stratified train/val/test split by class label. |
| 04 | `04_augmentation.ipynb` / `dentex/04_augmentation.ipynb` | Resize-and-pad + Albumentations augmentation pipeline (`AugmentationTransform`). |
| 05 | `05_dataloaders.ipynb` / `dentex/05_dataloaders.ipynb` | Wraps the split data in a `DatasetFromDF` and `DataLoader` for PyTorch. |
| 06 | `06_train.ipynb` / `dentex/06_train_dentexmodel.ipynb` | Trains a ResNet50 model via PyTorch Lightning (`ResNet50Model` / `ToothModel`). |
| 07 | `07_train_model_B.ipynb` / `dentex/07_model_training_fancy.ipynb` | A more advanced training run: TensorBoard logging, `ModelCheckpoint`, `LearningRateMonitor`/`LearningRateFinder`. |
| 08 | `08_test_binary.ipynb` / `dentex/08_evaluation_metrics.ipynb` | Loads a checkpoint and evaluates as binary classification (ROC AUC, F1, confusion matrix; dentex variant adds `torchmetrics`). |
| 09 | `09_test_roc.ipynb` / `dentex/09_performance_testdata.ipynb` | Per-class (one-vs-rest) ROC curves on held-out test data. |
| 10 | `10_explainable_ai.ipynb` (+ near-duplicate `10_explainable_ai-2.ipynb`) / `dentex/10_explainable_ai.ipynb` | Grad-CAM visualization (`pytorch_grad_cam`) of what the trained model attends to. |

### Key library code

- `computervision.datasets` / `computervision.torchdataset` — `DatasetFromDF` and image-loading utilities.
- `computervision.transformations.AugmentationTransform` — the shared augmentation wrapper.
- `computervision.models.lightningmodel` — the `ToothModel` Lightning module (ResNet50 backbone).
- `computervision.dentexdata.DentexData` — Dentex-specific download/annotation parsing.

See [architecture.md](./architecture.md) for the full module map and [data.md](./data.md) for
dataset sources and the on-disk data layout.
