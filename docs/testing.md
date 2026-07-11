## Testing and CI ##

### Running tests locally

```bash
uv run --frozen pytest
```

(or, inside the Docker container / an environment with the project installed, plain `pytest`).

### `tests/`

- `test_download_roboflow.py` — integration test that downloads the Roboflow dental dataset
  tarball from S3 into a temporary directory via `computervision.fileutils.FileOP` and asserts
  the file exists. Requires network access.
- `test_template.py` — empty placeholder/conftest template; no active tests.

Test coverage is currently minimal (one integration test) — most of the pipeline logic is
exercised interactively through the notebooks rather than through the test suite.

### CI workflows (`.github/workflows/`)

| Workflow | Trigger | What it does |
|---|---|---|
| `pytest.yml` | push to `main` | Sets up `uv` and Python (version from `pyproject.toml`), runs `uv run --frozen pytest`. |
| `docker.yml` | push to `main` | Cleans up unrelated preinstalled tooling on the runner to free disk space, runs `docker compose build`, then runs the test suite inside the CPU image: `docker compose -f ./docker-compose.cpu.yml run app python -m pytest`. |

Neither workflow currently publishes the built image to `ghcr.io/ccb-hms/computervision` as part
of its steps shown here — check the GitHub Actions history if you need to confirm how/when the
published `:latest` tag is updated.
