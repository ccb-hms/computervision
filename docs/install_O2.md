### Install on O2 at Harvard Medical School ###
<h3><img align="center" width="25%" src=../images/cloud_computing_640_3.jpg></h3>

O2 is a high-performance computing platform based on Linux, located at Harvard Medical School.
The platform is managed by the Research Computing Group, part of HMS IT. See the
[O2 documentation website](https://harvardmed.atlassian.net/wiki/spaces/O2/overview?homepageId=1586790623)
for platform-wide details (accounts, SLURM scheduling, storage quotas).

O2 does not support Docker, so this repository must be installed with the
[non-Docker `uv` workflow](./install_local.md#install-without-docker) described in
`install_local.md`, adapted to O2's module system and job scheduler:

1. Log in to O2 and start an interactive session with a GPU if you need one for training
   (see the O2 docs for current `srun`/`sbatch` GPU-partition syntax).
2. Load a Python 3.12-compatible toolchain via `module load`, then install `uv` per the
   [`uv` installation guide](https://docs.astral.sh/uv/getting-started/installation/).
3. Clone the repository and sync dependencies:
   ```bash
   git clone git@github.com:ccb-hms/computervision.git
   cd computervision
   uv sync --frozen
   ```
4. Install Detectron2 the same way as in `install_local.md`, from source:
   ```bash
   uv run python -m pip install --no-build-isolation "git+https://github.com/facebookresearch/detectron2.git"
   ```
   This requires a working CUDA toolchain in your loaded modules; consult O2's GPU documentation
   if the build fails.
5. Copy `env` to `.env` and set `DATA_DIR` to a path on O2 storage you have write access to and
   sufficient quota for (dental X-ray datasets and model checkpoints are large) — see
   [environment-variables.md](./environment-variables.md).
6. Run Jupyter Lab through O2's [Open OnDemand](https://o2portal.rc.hms.harvard.edu/) portal, or
   start it manually in your interactive session with `uv run jupyter lab` and tunnel the port
   over SSH, per O2's Jupyter documentation.

Because module names, GPU partitions, and portal URLs change over time, treat the O2 platform
docs linked above as authoritative for anything not specific to this repository.
