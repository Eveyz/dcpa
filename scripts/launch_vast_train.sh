#!/usr/bin/env bash
# Vast.ai One-Command Multi-GPU Launcher for DCPA (Data -> Stage 1..4 DDP -> Stage 5 SFT DDP)
#
# Examples:
#   # Recommended 1-week run on 4x A100 SXM4 80GB (~2.68B MoE-64, ~12.1 GB EN/ZH + Py/JS + Reasoning/Instruct/ToolCall shards):
#   bash scripts/launch_vast_train.sh --tmux=dcpa
#
#   # Faster 1.8B MoE-64 variant or larger 3.5B MoE-64 variant:
#   bash scripts/launch_vast_train.sh --preset=moe_1_8b --tmux=dcpa
#   bash scripts/launch_vast_train.sh --preset=moe_3_5b --total-steps=150000 --tmux=dcpa
#
#   # Instant <30s dataset cold start from a pre-packed .tar.zst URL or local snapshot:
#   bash scripts/launch_vast_train.sh --from-archive="https://.../dcpa_v2_10g.tar.zst" --tmux=dcpa
#
#   # Dry-run offline inspection (prints all resolved paths & torchrun commands without executing):
#   bash scripts/launch_vast_train.sh --dry-run
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Load Vast.ai optimal NCCL / NVMe / HF_TRANSFER environment if setup_vast_ai.sh was run
if [[ -f "${REPO_ROOT}/.vast_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.vast_env.sh"
fi

CONFIG_PATH="configs/dcpa_v2_4xa100.yaml"
PRESET="moe_2_6b"
DATA_PROFILE="v2_stem_10g"
NVME_DIR="${DCPA_NVME_ROOT:-}"
BACKUP_DIR=""
FROM_ARCHIVE=""
PACK_ARCHIVE=""
NGPU=""
TOTAL_STEPS=""
SFT_STEPS="10000"
SKIP_DATA=0
SKIP_SFT=0
RESUME_MODE="auto"
TMUX_SESSION=""
DRY_RUN=0
PYTHON_BIN="${PYTHON_BIN:-python3}"
EXTRA_TRAIN_ARGS=()

ORIG_ARGS=("$@")
for arg in "$@"; do
  case "${arg}" in
    --config=*) CONFIG_PATH="${arg#*=}" ;;
    --preset=*) PRESET="${arg#*=}" ;;
    --data-profile=*) DATA_PROFILE="${arg#*=}" ;;
    --nvme-dir=*) NVME_DIR="${arg#*=}" ;;
    --backup-dir=*) BACKUP_DIR="${arg#*=}" ;;
    --from-archive=*) FROM_ARCHIVE="${arg#*=}" ;;
    --pack-archive=*) PACK_ARCHIVE="${arg#*=}" ;;
    --gpus=*) NGPU="${arg#*=}" ;;
    --total-steps=*) TOTAL_STEPS="${arg#*=}" ;;
    --sft-steps=*) SFT_STEPS="${arg#*=}" ;;
    --resume=*) RESUME_MODE="${arg#*=}" ;;
    --no-resume) RESUME_MODE="none" ;;
    --skip-data) SKIP_DATA=1 ;;
    --skip-sft) SKIP_SFT=1 ;;
    --tmux) TMUX_SESSION="dcpa" ;;
    --tmux=*) TMUX_SESSION="${arg#*=}" ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help)
      sed -n '2,17p' "$0"
      exit 0
      ;;
    *)
      EXTRA_TRAIN_ARGS+=("${arg}")
      ;;
  esac
done

# Re-exec inside a persistent tmux session if requested and not already inside tmux
if [[ -n "${TMUX_SESSION}" && -z "${TMUX:-}" && "${DRY_RUN}" -eq 0 ]]; then
  if ! command -v tmux >/dev/null 2>&1; then
    echo "[WARN] tmux not installed; continuing in foreground."
  else
    FILTERED_ARGS=()
    for a in "${ORIG_ARGS[@]}"; do
      case "${a}" in
        --tmux|--tmux=*) ;;
        *) FILTERED_ARGS+=("${a}") ;;
      esac
    done
    if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
      echo "[tmux] Session '${TMUX_SESSION}' already exists. Attach with: tmux attach -t ${TMUX_SESSION}"
      exit 0
    fi
    CMD_STR="cd $(printf '%q' "${REPO_ROOT}") && bash scripts/launch_vast_train.sh"
    for fa in "${FILTERED_ARGS[@]}"; do
      CMD_STR+=" $(printf '%q' "${fa}")"
    done
    echo "[tmux] Launching detached session '${TMUX_SESSION}'..."
    tmux new-session -d -s "${TMUX_SESSION}" "${CMD_STR}; exec bash"
    echo "[tmux] Started! Attach live logs anytime with: tmux attach -t ${TMUX_SESSION}"
    exit 0
  fi
fi

# Resolve NVMe root directory
if [[ -z "${NVME_DIR}" ]]; then
  if [[ -d "/workspace" && -w "/workspace" ]]; then
    NVME_DIR="/workspace/nvme_cache"
  elif [[ -d "/raid" && -w "/raid" ]]; then
    NVME_DIR="/raid/dcpa_nvme"
  else
    NVME_DIR="${REPO_ROOT}/artifacts/nvme_cache"
  fi
fi

# Detect GPU count automatically
if [[ -z "${NGPU}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    NGPU="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)"
  fi
  if [[ -z "${NGPU}" || "${NGPU}" -lt 1 ]]; then
    NGPU=1
  fi
fi

# Ensure high-throughput NCCL / Allocator / HF Rust transfer env vars
export DCPA_NVME_ROOT="${NVME_DIR}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/dcpa-train-py${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_XET_HIGH_PERFORMANCE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

DATA_DIR="${NVME_DIR}/data"
CAP_DIR="${NVME_DIR}/cap_data"
SFT_DIR="${NVME_DIR}/sft_data"
CKPT_DIR="${NVME_DIR}/checkpoints"
LOGS_DIR="${NVME_DIR}/logs"
EXPORT_DIR="${NVME_DIR}/export"

PREP_CMD=(
  "${PYTHON_BIN}" "${REPO_ROOT}/data/prepare_vast_corpus.py"
  "--profile" "${DATA_PROFILE}"
  "--nvme-root" "${NVME_DIR}"
)
[[ -n "${FROM_ARCHIVE}" ]] && PREP_CMD+=("--from-archive" "${FROM_ARCHIVE}")
[[ -n "${PACK_ARCHIVE}" ]] && PREP_CMD+=("--pack-archive" "${PACK_ARCHIVE}")

if [[ "${NGPU}" -gt 1 ]]; then
  LAUNCHER=("${PYTHON_BIN}" "-m" "torch.distributed.run" "--standalone" "--nnodes=1" "--nproc_per_node=${NGPU}")
else
  LAUNCHER=("${PYTHON_BIN}")
fi

TRAIN_CMD=(
  "${LAUNCHER[@]}"
  "${REPO_ROOT}/train/train_dcpa.py"
  "--config" "${CONFIG_PATH}"
  "--preset" "${PRESET}"
  "--data-dir" "${DATA_DIR}"
  "--cap-data-dir" "${CAP_DIR}"
  "--local-cache-dir" "${NVME_DIR}"
  "--ckpt-dir" "${CKPT_DIR}"
  "--logs-dir" "${LOGS_DIR}"
  "--export-dir" "${EXPORT_DIR}"
  "--resume" "${RESUME_MODE}"
)
[[ -n "${TOTAL_STEPS}" ]] && TRAIN_CMD+=("--total-steps" "${TOTAL_STEPS}")
if [[ -n "${BACKUP_DIR}" ]]; then
  TRAIN_CMD+=(
    "--drive-sync-dir" "${BACKUP_DIR}/checkpoints"
    "--drive-logs-dir" "${BACKUP_DIR}/logs"
    "--drive-export-dir" "${BACKUP_DIR}/export"
  )
fi
if [[ "${#EXTRA_TRAIN_ARGS[@]}" -gt 0 ]]; then
  TRAIN_CMD+=("${EXTRA_TRAIN_ARGS[@]}")
fi

SFT_GLOBAL_BYTES=$(( 65536 * NGPU ))
SFT_CMD=(
  "${LAUNCHER[@]}"
  "${REPO_ROOT}/train/train_stage5_sft.py"
  "--weights" "${EXPORT_DIR}/dcpa_final.safetensors"
  "--config" "${EXPORT_DIR}/config.json"
  "--data-dir" "${SFT_DIR}"
  "--cap-data-dir" "${CAP_DIR}"
  "--export-dir" "${EXPORT_DIR}"
  "--total-steps" "${SFT_STEPS}"
  "--target-tokens-bytes-per-step" "${SFT_GLOBAL_BYTES}"
)
[[ -n "${BACKUP_DIR}" ]] && SFT_CMD+=("--drive-export-dir" "${BACKUP_DIR}/export")

echo "======================================================================"
echo "  DCPA Multi-GPU Training Pipeline"
echo "  Config      : ${CONFIG_PATH} (preset=${PRESET})"
echo "  GPUs        : ${NGPU} | NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL}"
echo "  Data Profile: ${DATA_PROFILE} -> ${NVME_DIR}"
echo "  Checkpoints : ${CKPT_DIR} (resume=${RESUME_MODE})"
echo "  Export      : ${EXPORT_DIR}"
echo "======================================================================"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "[DRY-RUN] Step 1 (Parallel NVMe Shards):"
  echo "  ${PREP_CMD[*]}"
  echo "[DRY-RUN] Step 2 (Stage 1..4 Developmental Training):"
  echo "  ${TRAIN_CMD[*]}"
  echo "[DRY-RUN] Step 3 (Stage 5 Capability & Pragmatic SFT):"
  echo "  ${SFT_CMD[*]}"
  exit 0
fi

mkdir -p "${CKPT_DIR}" "${LOGS_DIR}" "${EXPORT_DIR}"

if [[ "${SKIP_DATA}" -eq 0 ]]; then
  echo "[1/3] Preparing & verifying NVMe binary shards (profile=${DATA_PROFILE})..."
  "${PREP_CMD[@]}"
else
  echo "[1/3] Skipping dataset preparation (--skip-data)."
fi

echo "[2/3] Starting Stage 1..4 multi-GPU developmental training on ${NGPU} GPU(s)..."
"${TRAIN_CMD[@]}"

if [[ "${SKIP_SFT}" -eq 0 ]]; then
  echo "[3/3] Starting Stage 5 multi-GPU pragmatic & tool-call SFT (${SFT_STEPS} steps)..."
  "${SFT_CMD[@]}"
else
  echo "[3/3] Skipping Stage 5 SFT (--skip-sft)."
fi

echo "======================================================================"
echo "  [ALL DONE] Final Rust-ready artifacts exported to: ${EXPORT_DIR}"
ls -lh "${EXPORT_DIR}" || true
echo "======================================================================"
