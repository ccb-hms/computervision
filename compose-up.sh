#!/usr/bin/env bash

set -euo pipefail

# Pick docker compose command (Docker Desktop uses "docker compose")
if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
  DC=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
  DC=(docker-compose)
else
  echo "Error: docker compose not found." >&2
  exit 1
fi

# Allow manual override:
#   FORCE_GPU=1 ./compose-up.sh   -> force GPU override
#   FORCE_GPU=0 ./compose-up.sh   -> force CPU-only
FORCE_GPU="${FORCE_GPU:-}"

has_nvidia() {
  # Fast checks that don't pull images:
  # 1) nvidia-smi exists (installed NVIDIA drivers/toolkit)
  # 2) Docker knows about the 'nvidia' runtime (typical on GPU Linux hosts)
  command -v nvidia-smi >/dev/null 2>&1 || return 1
  docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -qi '"nvidia"' || return 1
  return 0
}

use_gpu=0
if [[ "${FORCE_GPU}" == "1" ]]; then
  use_gpu=1
elif [[ "${FORCE_GPU}" == "0" ]]; then
  use_gpu=0
elif has_nvidia; then
  use_gpu=1
fi

if [[ ${use_gpu} -eq 1 ]]; then
  echo "✅ NVIDIA runtime detected. Starting with GPU override..."
  "${DC[@]}" -f docker-compose.yml -f docker-compose.gpu.yml up --remove-orphans
else
  echo "ℹ️  No usable NVIDIA runtime found. Starting CPU-only..."
  "${DC[@]}" -f docker-compose.yml up  --remove-orphans
fi
