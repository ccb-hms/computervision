FROM python:3.12 AS base

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

# Ports for jupyter and tensorboard
EXPOSE 8888
EXPOSE 6006

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

# Pipenv dependencies
COPY Pipfile Pipfile.lock ./
RUN --mount=source=.git,target=.git,type=bind \
    pipenv install --system --deploy --ignore-pipfile --dev

# Copy bash scripts set executable flags
RUN mkdir -p /run_scripts
COPY /bash_scripts/* /run_scripts
RUN chmod +x /run_scripts/*

# Install Detectron2
# RUN python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Run the jupyter server
CMD ["/bin/bash", "/run_scripts/docker_entry"]