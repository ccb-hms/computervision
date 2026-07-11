[![Python 3.12](
https://img.shields.io/badge/python-3.12-blue.svg)](
https://www.python.org/downloads/release/python-31210/)
[![pytest](https://github.com/ccb-hms/computervision/actions/workflows/pytest.yml/badge.svg?branch=main)](https://github.com/ccb-hms/computervision/actions/workflows/pytest.yml)
[![docker](
https://github.com/ccb-hms/computervision/actions/workflows/docker.yml/badge.svg?branch=main)](https://github.com/ccb-hms/computervision/actions/workflows/docker.yml)
[![GHCR](
https://img.shields.io/badge/ghcr.io-ccb--hms%2Fcomputervision-blue?logo=docker)](https://github.com/ccb-hms/computervision/pkgs/container/computervision)

<p float="left">
    <img style="vertical-align: top" src="./images/train_248_boxes.png" width="40%" />
</p>

## The CCB Computer Vision Code Repository #

This repository contains template code that can be used as
a starting point for computer vision projects.
All frameworks, libraries, and data sets are open source and publicly available.
Some common tasks included here are:

- [Image Classification](./notebooks/classification) — see [docs/classification.md](./docs/classification.md)
- [Object Detection](./notebooks/detection) — see [docs/detection.md](./docs/detection.md)
- [Instance Segmentation](./notebooks/segmentation) — see [docs/segmentation.md](./docs/segmentation.md)
- Gradient-weighted Class Activation Mapping (Grad-CAM), included as the final step of the classification pipeline

## The Dentex Challenge 2023

The Dentex Challenge 2023 aims to provide insights into the effectiveness of AI in
dental radiology analysis and its potential to improve dental practice by comparing
frameworks that simultaneously point out abnormal teeth with dental enumeration and
associated diagnosis on panoramic dental X-rays.
The dataset comprises panoramic dental X-rays obtained from three
different institutions using standard clinical conditions but varying equipment and imaging protocols,
resulting in diverse image quality reflecting heterogeneous clinical practice.
It includes X-rays from patients aged 12 and above,
randomly selected from the hospital's database to ensure patient privacy and confidentiality.
A detailed description of the data and the annotation protocol
can be found on the [Dentex Challenge](https://dentex.grand-challenge.org/) website.
The data set is publicly available for download from the [Zenodo](https://zenodo.org/records/7812323#.ZDQE1uxBwUG)
open-access data repository.

## Quick start

The fastest way to get a working environment is Docker:

```bash
git clone git@github.com:ccb-hms/computervision.git
cd computervision
cp env .env          # then fill in real values, see docs/environment-variables.md
./compose-up.sh
```

`compose-up.sh` auto-detects whether an NVIDIA GPU/runtime is available and starts the
container with or without GPU support accordingly. Once running, open
`http://localhost:8888` for Jupyter Lab (notebooks live in `./notebooks`) and
`http://localhost:6006` for TensorBoard.

See [docs/docker.md](./docs/docker.md) for details on all four Compose files and the
plain `docker run` alternative.

## Getting started with this repository ##

- [Install on your local machine](./docs/install_local.md)
- [Install on the HMS O2 cluster](./docs/install_O2.md)
- [Docker setup reference](./docs/docker.md)
- [Environment variables](./docs/environment-variables.md)
- [Data sets](./docs/data.md)
- [Classification pipeline](./docs/classification.md)
- [Object detection pipeline](./docs/detection.md)
- [Instance segmentation pipeline](./docs/segmentation.md)
- [Repository architecture (`src/computervision`)](./docs/architecture.md)
- [Testing and CI](./docs/testing.md)
- [Label Studio](./docs/labelstudio.md)

## Label Studio ##

Label Studio is an open-source data labeling tool designed for labeling, annotating, and exploring various data types. The tool also features a robust machine learning interface, which can be utilized for training new models, active learning, supervised learning, and various other training techniques.

For more information on how to use Label Studio, please refer to the [Label Studio documentation](https://labelstud.io/guide/). You can find installation instructions [here](https://labelstud.io/guide/install.html) and in the documentation of this repository [here](./docs/labelstudio.md).
