## Data sets ##

This repository uses two public dental-imaging data sets, downloaded on demand by the
`01_download*` notebook in each pipeline rather than committed to the repository.

### Roboflow dental dataset (classification)

- Source URL: `CL_URL` (see [environment-variables.md](./environment-variables.md)), an S3-hosted
  tarball (`dataset_dental_roboflow.tar.gz`) of individually annotated tooth images.
- Used by the top-level `notebooks/classification/` pipeline (single-tooth image classification).

### Dentex Challenge 2023 dataset (detection, segmentation, and `classification/dentex/`)

- Source URL: `DT_URL`, an S3 mirror of the
  [Dentex Challenge](https://dentex.grand-challenge.org/) quadrant-enumeration release, which is
  also published on [Zenodo](https://zenodo.org/records/7812323#.ZDQE1uxBwUG).
- Panoramic dental X-rays with COCO-style JSON annotations for quadrant, tooth enumeration
  (FDI/ADA numbering), and diagnosis.
- Downloaded and parsed via `computervision.dentexdata.DentexData` / `computervision.dentex.Dentex`.
- Used by `notebooks/detection/`, `notebooks/segmentation/`, and `notebooks/classification/dentex/`.

### `./data` directory layout

`DATA_DIR` (set to `/app/data` inside the Docker container, mapped to `./data` on the host via the
Compose volume mount) is the root for everything the pipelines download or produce:

| Subdirectory | Contents |
|---|---|
| `data/classification` | Roboflow classification dataset (raw + cropped tooth images). |
| `data/dentex` | Dentex Challenge raw and cropped images/annotations. |
| `data/computervision_data` | Processed dataframes/parquet files shared between notebook steps. |
| `data/training_examples` | Sample batches saved for inspection (e.g. augmentation/dataloader checks). |
| `data/output` | Model checkpoints, training logs, and evaluation outputs. |
| `data/hub` | Hugging Face Hub cache (`HF_HOME`) — pretrained/fine-tuned model weights. |
| `data/xet` | Cache used by the `xet`-backed Hugging Face Hub downloader. |

### The parquet convention

Across all three pipelines, each notebook that produces a dataset (download, crop, split) saves
its output as a `.parquet` file under `DATA_DIR`, which the next notebook in the sequence loads.
This is the hand-off mechanism between pipeline steps — e.g.
`dentex_detection_dataset.parquet` → `dentex_detection_datasplit.parquet` in the segmentation
pipeline. If you re-run an early notebook with different parameters, re-run the downstream
notebooks too, since they read the parquet file rather than recomputing it.
