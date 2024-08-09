FROM nvcr.io/nvidia/pytorch:24.07-py3 AS base

ARG DEV_computervision

ENV \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    PIP_SRC=/src \
    PIPENV_HIDE_EMOJIS=true \
    NO_COLOR=true \
    PIPENV_NOSPIN=true

# Port for JupyterLab server
EXPOSE 8888

RUN mkdir -p /app
WORKDIR /app

# System dependencies
RUN apt-get update -y && \
        apt-get install -y \
        'libsndfile1' \
        'libgl1-mesa-glx' \
        'ffmpeg' \
        'libsm6' \
        'libxext6' \
        'ninja-build'

# Pip and pipenv
RUN pip install --upgrade pip
RUN pip install pipenv

# Some package stuff
COPY setup.py ./
COPY src/computervision/__init__.py src/computervision/__init__.py

# Install additional packages into system python
RUN --mount=source=.git,target=.git,type=bind \
    pip install --no-cache-dir -e .
RUN python -m pip install -U \
    jupyterlab \
    scikit-learn \
    scikit-image \
    grad-cam \
    albumentations \
    lightning

# We need to replace some versions installed in the container
RUN python -m pip install \
    "opencv-python==4.7.*" \
    "opencv-python-headless==4.7.*" \
    "numpy>=1.24,<2.0" \
    --force-reinstall

# Detectron2
RUN python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Run the jupyter lab server
RUN mkdir -p /run_scripts
COPY /bash_scripts/docker_entry /run_scripts
RUN chmod +x /run_scripts/*
CMD ["/bin/bash", "/run_scripts/docker_entry"]