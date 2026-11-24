#!/usr/bin/env bash
# Unified Single-GPU <-> Multi-GPU Training Launcher for DCPA (Data -> Stage 1..4 -> Stage 5 SFT)
#
# Same script, same config, same checkpoints across 1 GPU, N GPUs, DGX Spark, Vast.ai, or Mac:
#   # 1) Auto-detect all available GPUs (1 GPU -> single-process; 2/4/8 GPUs -> multi-GPU DDP):
#   bash scripts/train.sh
#
#   # 2) Force Single-GPU training (even on a multi-GPU server; optionally pick a specific GPU):
#   bash scripts/train.sh --gpus=1 --preset=moe_650m
#   bash scripts/train.sh --gpus=1 --gpu-ids=2 --preset=moe_650m
#
#   # 3) Switch to 2 / 4 / 8 GPUs anytime (resumes seamlessly from the same checkpoint directory!):
#   bash scripts/train.sh --gpus=4 --preset=moe_2_6b --tmux=dcpa
#
#   # 4) Preview resolved commands offline without executing:
#   bash scripts/train.sh --gpus=1 --dry-run
#   bash scripts/train.sh --gpus=4 --dry-run
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f "${REPO_ROOT}/.vast_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.vast_env.sh"
fi

CONFIG_PATH="configs/v1/dcpa_unified.yaml"
PRESET=""
DATA_PROFILE=""
GPUS="auto"
GPU_IDS=""
NVME_DIR="${DCPA_NVME_ROOT:-}"
BACKUP_DIR=""
FROM_ARCHIVE=""
PACK_ARCHIVE=""
TOTAL_STEPS=""
SFT_STEPS=""
SKIP_DATA=0
SKIP_SFT=0
RESUME_MODE="auto"
TMUX_SESSION=""
DRY_RUN=0
PYTHON_BIN="${PYTHON_BIN:-}"
EXTRA_TRAIN_ARGS=()

if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x "${REPO_ROOT}/dcpa-train-py/.venv/bin/python" ]]; then
    PYTHON_BIN="${REPO_ROOT}/dcpa-train-py/.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

ORIG_ARGS=("$@")
for arg in "$@"; do
  case "${arg}" in
    --config=*) CONFIG_PATH="${arg#*=}" ;;
    --v2-cognitive|--cognitive-v2) CONFIG_PATH="configs/v2/dcpa_210m_v2_cognitive.yaml" ;;
    --v2-cognitive-48m|--cognitive-v2-48m) CONFIG_PATH="configs/v2/dcpa_48m_v2_cognitive.yaml" ;;
    --preset=*) PRESET="${arg#*=}" ;;
    --data-profile=*) DATA_PROFILE="${arg#*=}" ;;
    --gpus=*) GPUS="${arg#*=}" ;;
    --gpu-ids=*) GPU_IDS="${arg#*=}" ;;
    --single-gpu|--1gpu) GPUS="1" ;;
    --nvme-dir=*) NVME_DIR="${arg#*=}" ;;
    --backup-dir=*) BACKUP_DIR="${arg#*=}" ;;
    --from-archive=*) FROM_ARCHIVE="${arg#*=}" ;;
    --pack-archive=*) PACK_ARCHIVE="${arg#*=}" ;;
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
      sed -n '2,18p' "$0"
      exit 0
      ;;
    *)
      EXTRA_TRAIN_ARGS+=("${arg}")
      ;;
  esac
done

# Re-exec inside a persistent tmux session if requested
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
    CMD_STR="cd $(printf '%q' "${REPO_ROOT}") && bash scripts/train.sh"
    for fa in "${FILTERED_ARGS[@]}"; do
      CMD_STR+=" $(printf '%q' "${fa}")"
    done
    echo "[tmux] Launching detached session '${TMUX_SESSION}'..."
    tmux new-session -d -s "${TMUX_SESSION}" "${CMD_STR}; exec bash"
    echo "[tmux] Started! Attach live logs anytime with: tmux attach -t ${TMUX_SESSION}"
    exit 0
  fi
fi

# Resolve effective GPU count for smart default preset / data-profile selection
DETECTED_GPUS=0
if [[ -n "${GPU_IDS}" ]]; then
  IFS=',' read -r -a _IDS_ARR <<< "${GPU_IDS}"
  DETECTED_GPUS="${#_IDS_ARR[@]}"
elif command -v nvidia-smi >/dev/null 2>&1; then
  DETECTED_GPUS="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)"
fi

if [[ "${GPUS}" == "auto" || "${GPUS}" == "all" ]]; then
  EFF_GPUS=$(( DETECTED_GPUS > 0 ? DETECTED_GPUS : 1 ))
else
  EFF_GPUS="${GPUS}"
fi

# Smart defaults when --preset / --data-profile / --sft-steps are omitted
IS_V2_COGNITIVE=0
IS_V2_COGNITIVE_48M=0
if [[ "${CONFIG_PATH}" == *"dcpa_48m_v2_cognitive"* ]]; then
  IS_V2_COGNITIVE=1
  IS_V2_COGNITIVE_48M=1
elif [[ "${CONFIG_PATH}" == *"dcpa_210m_v2_cognitive"* ]]; then
  IS_V2_COGNITIVE=1
fi

if [[ -z "${PRESET}" ]]; then
  if [[ "${IS_V2_COGNITIVE_48M}" -eq 1 ]]; then
    PRESET="proto_48m"
  elif [[ "${IS_V2_COGNITIVE}" -eq 1 ]]; then
    PRESET="moe_210m"
  elif [[ "${EFF_GPUS}" -ge 2 ]]; then
    PRESET="moe_2_6b"
  elif [[ "${DETECTED_GPUS}" -ge 1 ]]; then
    PRESET="moe_650m"
  else
    PRESET="moe_210m"
  fi
fi

if [[ -z "${DATA_PROFILE}" ]]; then
  if [[ "${IS_V2_COGNITIVE_48M}" -eq 1 ]]; then
    if [[ "${DETECTED_GPUS}" -ge 1 ]]; then
      DATA_PROFILE="v2_cognitive_3g"
    else
      DATA_PROFILE="v2_cognitive_smoke"
    fi
  elif [[ "${IS_V2_COGNITIVE}" -eq 1 ]]; then
    if [[ "${DETECTED_GPUS}" -ge 1 ]]; then
      DATA_PROFILE="v2_cognitive_5g"
    else
      DATA_PROFILE="v2_cognitive_smoke"
    fi
  elif [[ "${EFF_GPUS}" -ge 2 ]]; then
    DATA_PROFILE="v2_stem_10g"
  elif [[ "${DETECTED_GPUS}" -ge 1 ]]; then
    DATA_PROFILE="v1_fast"
  else
    DATA_PROFILE="smoke"
  fi
fi

if [[ -z "${SFT_STEPS}" ]]; then
  if [[ "${IS_V2_COGNITIVE_48M}" -eq 1 ]]; then
    SFT_STEPS="10000"
  elif [[ "${IS_V2_COGNITIVE}" -eq 1 ]]; then
    SFT_STEPS="15000"
  else
    SFT_STEPS="8000"
  fi
fi

if [[ -z "${NVME_DIR}" ]]; then
  if [[ -d "/workspace" && -w "/workspace" ]]; then
    NVME_DIR="/workspace/nvme_cache"
  elif [[ -d "/raid" && -w "/raid" ]]; then
    NVME_DIR="/raid/dcpa_nvme"
  else
    NVME_DIR="${REPO_ROOT}/artifacts/nvme_cache"
  fi
fi

export DCPA_NVME_ROOT="${NVME_DIR}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/dcpa-train-py${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_XET_HIGH_PERFORMANCE=1
export TOKENIZERS_PARALLELISM=false

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

# Both train_dcpa.py and train_stage5_sft.py natively handle --gpus=1 (in-process) and --gpus=N (auto-torchrun)!
GPU_FLAGS=("--gpus" "${EFF_GPUS}")
[[ -n "${GPU_IDS}" ]] && GPU_FLAGS+=("--gpu-ids" "${GPU_IDS}")

TRAIN_CMD=(
  "${PYTHON_BIN}" "${REPO_ROOT}/train/train_dcpa.py"
  "${GPU_FLAGS[@]}"
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

SFT_CMD=(
  "${PYTHON_BIN}" "${REPO_ROOT}/train/train_stage5_sft.py"
  "${GPU_FLAGS[@]}"
  "--weights" "${EXPORT_DIR}/dcpa_final.safetensors"
  "--config" "${EXPORT_DIR}/config.json"
  "--train-config" "${CONFIG_PATH}"
  "--data-dir" "${SFT_DIR}"
  "--cap-data-dir" "${CAP_DIR}"
  "--export-dir" "${EXPORT_DIR}"
  "--total-steps" "${SFT_STEPS}"
)
[[ -n "${BACKUP_DIR}" ]] && SFT_CMD+=("--drive-export-dir" "${BACKUP_DIR}/export")

MODE_LABEL="$( [[ "${EFF_GPUS}" -gt 1 ]] && echo "Multi-GPU (${EFF_GPUS} GPUs)" || echo "Single-GPU / Single-Process" )"
echo "======================================================================"
echo "  DCPA Unified Training Pipeline — ${MODE_LABEL}"
echo "  Config      : ${CONFIG_PATH} (preset=${PRESET})"
echo "  GPU Spec    : --gpus=${EFF_GPUS}${GPU_IDS:+ --gpu-ids=${GPU_IDS}}"
echo "  Data Profile: ${DATA_PROFILE} -> ${NVME_DIR}"
echo "  Checkpoints : ${CKPT_DIR} (resume=${RESUME_MODE}, elastic 1<->N GPUs)"
echo "  Export      : ${EXPORT_DIR}"
echo "======================================================================"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "[DRY-RUN] Step 1 (Parallel Shards):"
  echo "  ${PREP_CMD[*]}"
  echo "[DRY-RUN] Step 2 (Stage 1..4 Developmental Training):"
  echo "  ${TRAIN_CMD[*]}"
  echo "[DRY-RUN] Step 3 (Stage 5 Capability & Pragmatic SFT):"
  echo "  ${SFT_CMD[*]}"
  exit 0
fi

mkdir -p "${CKPT_DIR}" "${LOGS_DIR}" "${EXPORT_DIR}"

if [[ "${SKIP_DATA}" -eq 0 ]]; then
  echo "[1/3] Preparing & verifying binary shards (profile=${DATA_PROFILE})..."
  "${PREP_CMD[@]}"
else
  echo "[1/3] Skipping dataset preparation (--skip-data)."
fi

echo "[2/3] Starting Stage 1..4 training (${MODE_LABEL})..."
"${TRAIN_CMD[@]}"

if [[ "${SKIP_SFT}" -eq 0 ]]; then
  echo "[3/3] Starting Stage 5 SFT (${MODE_LABEL}, ${SFT_STEPS} steps)..."
  "${SFT_CMD[@]}"
else
  echo "[3/3] Skipping Stage 5 SFT (--skip-sft)."
fi

echo "======================================================================"
echo "  [ALL DONE] Final Rust-ready artifacts exported to: ${EXPORT_DIR}"
ls -lh "${EXPORT_DIR}" || true
echo "======================================================================"
