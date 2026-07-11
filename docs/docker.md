## Docker reference ##

This repository ships a single Docker image (built from the `Dockerfile` at the repo root)
that runs a Jupyter Lab server for all three pipelines (classification, detection, segmentation).

### The image

`Dockerfile` builds on `python:3.12.13-slim` and:

- Installs [`uv`](https://docs.astral.sh/uv/) and syncs dependencies from `uv.lock`/`pyproject.toml`.
- Installs `detectron2` directly from GitHub (`facebookresearch/detectron2`) via pip, since it
  isn't published to PyPI.
- Writes a default Hugging Face `accelerate` config (`mixed_precision='fp16'`).
- Copies `bash_scripts/*` into `/run_scripts/` inside the image.
- Exposes port `8888` (Jupyter Lab) and `6006` (TensorBoard).
- Default `CMD` runs `/run_scripts/docker_entry`, which starts:
  ```bash
  jupyter lab --ip=0.0.0.0 --allow-root --port=8888 \
    --ContentsManager.allow_hidden=True \
    --notebook-dir=/app/notebooks \
    --ServerApp.token='' --ServerApp.password=''
  ```
  **Note:** Jupyter runs with no token and no password. This is fine for local use with the
  ports bound to `localhost`, but do not expose port 8888 on a public interface without adding
  authentication.

The published image is `ghcr.io/ccb-hms/computervision:latest`, built by the
`docker` GitHub Actions workflow.

### Compose files

All four Compose files define one `app` service (container name `computervision`), set
`ipc: host`, publish ports `8888:8888` and `6006:6006`, mount `.:/app` and `./data:/app/data`,
and load `./.env`. They differ only in image source and GPU reservation:

| File | Image source | GPU reservation |
|---|---|---|
| `docker-compose.yml` | pulls `ghcr.io/ccb-hms/computervision:latest` | yes (1 NVIDIA GPU) |
| `docker-compose.gpu.yml` | pulls `ghcr.io/ccb-hms/computervision:latest` | yes (1 NVIDIA GPU) — identical to the default file, meant to be layered on top of `docker-compose.yml` |
| `docker-compose.cpu.yml` | pulls `ghcr.io/ccb-hms/computervision:latest` | no |
| `docker-compose.build.yml` | builds locally from `./Dockerfile` as `computervision:0.0.1` | yes (1 NVIDIA GPU) |

Use `docker-compose.build.yml` (or `docker compose build`) when you've changed the `Dockerfile`
or `pyproject.toml`/`uv.lock` and need a fresh local image instead of the published one.

### Recommended entry point: `compose-up.sh`

```bash
./compose-up.sh
```

This script auto-detects whether an NVIDIA GPU/runtime is usable (`nvidia-smi` present *and*
Docker reports an `nvidia` runtime) and runs:

- **GPU detected:** `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --remove-orphans`
- **No GPU:** `docker compose -f docker-compose.yml up --remove-orphans`

Override the detection manually:

```bash
FORCE_GPU=1 ./compose-up.sh   # force GPU
FORCE_GPU=0 ./compose-up.sh   # force CPU-only
```

To run CPU-only directly without the script:

```bash
docker compose -f docker-compose.cpu.yml up
```

### Alternative: plain `docker run`

`bash_scripts/docker_run` runs the same image without Compose:

```bash
docker run \
  --name computervision \
  --gpus 1 \
  --ipc=host \
  -e DATA_DIR=/app/data \
  --env-file ./env \
  -p 8888:8888 \
  -v "$(pwd)":/app \
  -v "$(pwd)/data":/app/data \
  ghcr.io/ccb-hms/computervision:latest
```

Note this variant hard-codes `--gpus 1` (edit the script for CPU-only hosts) and only publishes
port 8888 (no TensorBoard port).

### Environment variables inside the container

`HF_HOME`, `TORCH_HOME`, and `DATA_DIR` are all set to `/app/data` directly in the Compose files
so Hugging Face caches, Torch Hub caches, and the pipelines' data directory all land under the
mounted `./data` volume. See [environment-variables.md](./environment-variables.md) for the
remaining variables loaded from `.env`.
