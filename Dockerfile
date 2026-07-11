FROM python:3.12.13-slim

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    NO_COLOR=true \
    UV_COMPILE_BYTECODE=1 \
    UV_SYSTEM_PYTHON=true \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON_PREFERENCE=only-system \
    UV_PYTHON=3.12 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/usr/local

# Ports for jupyter and tensorboard
EXPOSE 8888
EXPOSE 6006

WORKDIR /app

RUN apt-get -y update && \
    apt-get -y install --no-install-recommends libgl1 libglib2.0-0 git build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project

# Install the project
COPY src/computervision/__init__.py src/computervision
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

RUN python -c "from accelerate.utils import write_basic_config; write_basic_config(mixed_precision='fp16')"

# Detectron2 library
RUN python -m pip install --no-build-isolation "git+https://github.com/facebookresearch/detectron2.git"

# Link the image automatically to the repository on GitHub
LABEL org.opencontainers.image.source=https://github.com/ccb-hms/computervision

# Copy bash scripts and set executable flags
COPY /bash_scripts/* /run_scripts/
RUN chmod +x /run_scripts/*

# Run the jupyter server
CMD ["/bin/bash", "/run_scripts/docker_entry"]
