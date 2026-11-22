#!/usr/bin/env bash
# NVIDIA DGX Spark (GB10 Grace Blackwell ARM64 + 128 GB Unified Memory)
# One-command environment bootstrapper & JupyterLab launcher for DCPA training.
#
# Usage:
#   bash scripts/setup_dgx_spark.sh --ngc      # (Recommended) Launch official NVIDIA NGC PyTorch container + JupyterLab
#   bash scripts/setup_dgx_spark.sh --native   # Create native ARM64 cu128 venv on DGX OS + launch JupyterLab
#   bash scripts/setup_dgx_spark.sh --native --no-jupyter  # Install native environment only (for tmux CLI training)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODE="ngc"
START_JUPYTER=1
PORT="${DCPA_JUPYTER_PORT:-8888}"
NGC_IMAGE="${DCPA_NGC_IMAGE:-nvcr.io/nvidia/pytorch:25.04-py3}"
VENV_DIR="${REPO_ROOT}/.venv_spark"

for arg in "$@"; do
  case "${arg}" in
    --ngc) MODE="ngc" ;;
    --native) MODE="native" ;;
    --no-jupyter) START_JUPYTER=0 ;;
    --port=*) PORT="${arg#*=}" ;;
    --image=*) NGC_IMAGE="${arg#*=}" ;;
    -h|--help)
      sed -n '2,10p' "$0"
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: ${arg}" >&2
      exit 1
      ;;
  esac
done

echo "======================================================================"
echo "  DCPA DGX Spark (GB10 Grace Blackwell) Bootstrapper"
echo "  Repo : ${REPO_ROOT}"
echo "  Arch : $(uname -m) | OS: $(uname -s) | Mode: ${MODE}"
echo "======================================================================"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
else
  echo "[WARN] nvidia-smi not found in PATH; continuing..."
fi

if [[ "${MODE}" == "ngc" ]]; then
  if ! command -v docker >/dev/null 2>&1; then
    echo "[ERROR] docker not found. Re-run with: bash scripts/setup_dgx_spark.sh --native" >&2
    exit 1
  fi
  echo "[NGC] Launching ${NGC_IMAGE} with GPU access & JupyterLab on port ${PORT}..."
  echo "[Tip] From your Mac, forward port via: ssh -L ${PORT}:localhost:${PORT} <user>@<dgx-spark-ip>"
  echo "[Tip] Then open: http://localhost:${PORT}/lab/tree/notebooks/DGX_Spark_DCPA_Train.ipynb"
  exec docker run --rm -it \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -p "${PORT}:${PORT}" \
    -v "${REPO_ROOT}:/workspace/dcpa" \
    -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
    -w /workspace/dcpa \
    -e HF_HUB_ENABLE_HF_TRANSFER=1 \
    -e HF_XET_HIGH_PERFORMANCE=1 \
    -e PYTHONPATH=/workspace/dcpa:/workspace/dcpa/dcpa-train-py \
    "${NGC_IMAGE}" \
    bash -lc "pip install -q safetensors datasets hf_transfer tree-sitter tree-sitter-python tree-sitter-rust pyyaml pytest matplotlib jupyterlab && \
              pip install -q --no-deps --no-build-isolation -e /workspace/dcpa/dcpa-train-py && \
              jupyter lab --ip=0.0.0.0 --port=${PORT} --no-browser --allow-root"
fi

# Native DGX OS venv mode (ARM64 + CUDA 12.8 Blackwell wheels)
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ ! -d "${VENV_DIR}" ]]; then
  echo "[Native] Creating isolated virtualenv at ${VENV_DIR}..."
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m pip install -q --upgrade pip setuptools wheel

if ! python -c "import torch; assert torch.cuda.is_available()" >/dev/null 2>&1; then
  echo "[Native] Installing PyTorch CUDA 12.8 wheels for DGX Spark (Blackwell sm_121)..."
  pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu128
fi

echo "[Native] Installing DCPA training, AST filtering & Notebook dependencies..."
pip install -q \
  safetensors \
  datasets \
  hf_transfer \
  tree-sitter \
  tree-sitter-python \
  tree-sitter-rust \
  pyyaml \
  pytest \
  matplotlib \
  jupyterlab \
  ipykernel

pip install -q --no-deps --no-build-isolation -e "${REPO_ROOT}/dcpa-train-py"
python -m ipykernel install --user --name dcpa-spark --display-name "Python (DCPA DGX Spark)" >/dev/null

python -c "
import torch
print('[OK] PyTorch:', torch.__version__, '| CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f'[OK] GPU: {p.name} | cc={p.major}.{p.minor} | bf16={torch.cuda.is_bf16_supported()}')
"

if [[ "${START_JUPYTER}" -eq 1 ]]; then
  echo "[Native] Starting JupyterLab on http://0.0.0.0:${PORT}"
  echo "[Tip] From your Mac: ssh -L ${PORT}:localhost:${PORT} <user>@<dgx-spark-ip>"
  echo "[Tip] Open notebooks/DGX_Spark_DCPA_Train.ipynb and select kernel 'Python (DCPA DGX Spark)'."
  export HF_HUB_ENABLE_HF_TRANSFER=1
  export HF_XET_HIGH_PERFORMANCE=1
  export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/dcpa-train-py"
  exec jupyter lab --ip=0.0.0.0 --port="${PORT}" --no-browser
else
  echo "[DONE] Native DGX Spark environment ready. Activate with: source ${VENV_DIR}/bin/activate"
fi
