## Environment variables ##

Copy the template file to `.env` and fill in real values before running anything:

```bash
cp env .env
```

`.env` is loaded by Docker Compose (`env_file: ./.env`) and by `python-dotenv` in local,
non-Docker use. It is git-ignored — never commit real tokens.

| Variable | Set in | Used by | Purpose |
|---|---|---|---|
| `DATA_DIR` | `env` / `.env` (local use) or directly in the Compose files (`/app/data`, container use) | notebooks, `computervision.fileutils`, `computervision.dentexdata` | Root directory for downloaded datasets, cropped images, checkpoints, and other pipeline outputs. |
| `CL_URL` | `env` / `.env` | classification download notebooks (`notebooks/classification/01_download.ipynb`) | S3 URL of the Roboflow dental classification dataset tarball. |
| `DT_URL` | `env` / `.env` | `computervision.dentex.Dentex`, Dentex download notebooks | S3 URL of the Dentex quadrant-enumeration dataset tarball (mirrors the Zenodo release). |
| `HF_TOKEN` | `env` / `.env` | Hugging Face Hub downloads (RT-DETR checkpoints, `transformers`, `accelerate`) | Hugging Face authentication token. |
| `CLAUDE_CODE_OAUTH_TOKEN` | `env` / `.env` | Claude Code CLI, used interactively in `notebooks/detection/08_inference.ipynb` to compare model predictions against Claude's own bounding-box predictions | OAuth token for Claude Code. |
| `HF_HOME` | set directly to `/app/data` in all Compose files | Hugging Face Hub cache | Keeps downloaded HF models/datasets under the mounted `./data` volume instead of the container's home directory. |
| `TORCH_HOME` | set directly to `/app/data` in all Compose files | Torch Hub cache | Same idea, for `torch.hub` downloads. |

### Notes

- Without Docker, `DATA_DIR` also determines where `computervision` looks for/writes data on your
  local machine (e.g. `/home/andreas/data/cv_data` in one example configuration) — set it to
  wherever you want datasets and checkpoints to live.
- `HF_TOKEN` and `CLAUDE_CODE_OAUTH_TOKEN` are secrets. The tracked `env` file only contains
  placeholder values (`hf_token`, `claude_token`); real values belong only in your local `.env`.
