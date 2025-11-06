FROM nvcr.io/nvidia/pytorch:25.08-py3 AS base

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

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
    NO_COLOR=true \
    UV_COMPILE_BYTECODE=1 \
    UV_SYSTEM_PYTHON=true \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON_PREFERENCE=only-system \
    UV_LINK_MODE=copy \
    UV_TOOL_BIN_DIR=/usr/bin \
    UV_PROJECT_ENVIRONMENT=/usr

# Ports for jupyter and tensorboard
EXPOSE 8888
EXPOSE 6006

RUN mkdir -p /app
WORKDIR /app

RUN apt-get -y update && \
    apt-get -y install libgl1

# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=.git,target=.git \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
     uv sync --inexact

# Install the project
COPY src/computervision/__init__.py \
    src/computervision/VERSION src/computervision
COPY pyproject.toml uv.lock ./
RUN uv sync --inexact

# Dependencies that depend on the container's libraries
RUN python -m pip install -U \
    "numpy<2.0" \
    torchmetrics \
    timm \
    accelerate \
    lightning \
    opencv-python \
    grad-cam

RUN python -c "from accelerate.utils import write_basic_config; write_basic_config(mixed_precision='fp16')"

# Detectron2 library
RUN python -m pip install "git+https://github.com/facebookresearch/detectron2.git"

# Copy bash scripts and set executable flags
RUN mkdir -p /run_scripts
COPY /bash_scripts/* /run_scripts
RUN chmod +x /run_scripts/*

# Run the jupyter server
CMD ["/bin/bash", "/run_scripts/docker_entry"]
