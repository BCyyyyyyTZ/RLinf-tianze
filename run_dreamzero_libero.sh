#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# User-editable settings
###############################################################################

# Supported modes:
#   eval_2gpu   - LIBERO + DreamZero SFT rollout eval smoke test
# training methods:
#   collocated  - sync PPO through train_embodied_agent.py, actor/env/rollout all selected GPUs
#   hybrid      - sync PPO through train_embodied_agent.py, actor all local GPUs,
#                 env lower half, rollout upper half
#   async       - async PPO through train_async.py, actor/env/rollout all local GPUs
#   ours        - async PPO through train_async.py with ours scheduling, actor/env/rollout all local GPUs,
#                 pipeline=2, CPU-pinned latency-balanced LIBERO rollout
DREAMZERO_RUN_MODE="${DREAMZERO_RUN_MODE:-ours}"

# Paths to update on a new machine after:
#   git clone https://github.com/mi150/RLinf.git
#   cd RLinf
#   bash requirements/install.sh embodied  --model dreamzero  --env maniskill_libero  --venv your_venv_name
TIANZE_ROOT="${TIANZE_ROOT:-/data1/gaobowen/tianze}"
REPO_PATH="${REPO_PATH:-${DREAMZERO_RLINF_REPO:-/data1/gaobowen/RLinf_dreamzero}}"
DREAMZERO_VENV_PATH="${DREAMZERO_VENV_PATH:-${DREAMZERO_PYTHON_ENV:-${REPO_PATH}/.venv-libero-dreamzero}}"
DREAMZERO_SOURCE_PATH="${DREAMZERO_SOURCE_PATH:-${DREAMZERO_SOURCE_ROOT:-${TIANZE_ROOT}/dreamzero}}"
export DREAMZERO_MODEL_PATH="${DREAMZERO_MODEL_PATH:-${TIANZE_ROOT}/model/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Step18000}"
export DREAMZERO_WAN_PATH="${DREAMZERO_WAN_PATH:-${TIANZE_ROOT}/model/Wan2.2-TI2V-5B}"
# Wan2.2-TI2V-5B does not ship CLIP; keep Wan2.1 only for the image encoder.
export DREAMZERO_WAN21_CLIP_PATH="${DREAMZERO_WAN21_CLIP_PATH:-${TIANZE_ROOT}/model/Wan2.1-I2V-14B-480P}"
export DREAMZERO_TOKENIZER_PATH="${DREAMZERO_TOKENIZER_PATH:-${DREAMZERO_WAN_PATH}/google/umt5-xxl}"
export DREAMZERO_EMBODIMENT_TAG="${DREAMZERO_EMBODIMENT_TAG:-libero_sim}"
LOG_ROOT="${LOG_ROOT:-${TIANZE_ROOT}/logs/4gpu}"
RUNTIME_MODEL_ROOT="${RUNTIME_MODEL_ROOT:-${TIANZE_ROOT}/runtime_models}"
RAY_TMPDIR_ROOT="${RAY_TMPDIR_ROOT:-${TIANZE_ROOT}/raytmp}"
DREAMZERO_DEFAULT_GPUS="${DREAMZERO_DEFAULT_GPUS:-0,1,2,3}"
DREAMZERO_DEFAULT_PLACEMENT="${DREAMZERO_DEFAULT_PLACEMENT:-0-3}"

# Common runtime knobs.
DREAMZERO_DRY_RUN="${DREAMZERO_DRY_RUN:-0}"
DREAMZERO_RUNTIME_MODEL_STAGING="${DREAMZERO_RUNTIME_MODEL_STAGING:-auto}" # auto|hardlink|symlink|copy
DREAMZERO_DISABLE_TORCH_COMPILE="${DREAMZERO_DISABLE_TORCH_COMPILE:-true}"
DREAMZERO_RELEASE_ROLLOUT_CACHE_AFTER_GENERATE="${DREAMZERO_RELEASE_ROLLOUT_CACHE_AFTER_GENERATE:-true}"
DREAMZERO_ACTION_HEAD_TRAIN_ARCHITECTURE="${DREAMZERO_ACTION_HEAD_TRAIN_ARCHITECTURE:-lora}"
DREAMZERO_ACTION_HEAD_DEBUG_LOGS="${DREAMZERO_ACTION_HEAD_DEBUG_LOGS:-false}"
if [ "${DREAMZERO_ACTION_HEAD_TRAIN_ARCHITECTURE}" = "lora" ]; then
  DREAMZERO_ACTION_HEAD_DEFER_LORA_INJECTION="${DREAMZERO_ACTION_HEAD_DEFER_LORA_INJECTION:-true}"
else
  DREAMZERO_ACTION_HEAD_DEFER_LORA_INJECTION="${DREAMZERO_ACTION_HEAD_DEFER_LORA_INJECTION:-false}"
fi
DREAMZERO_START_RAY_HEAD="${DREAMZERO_START_RAY_HEAD:-0}"
RAY_NODE_IP_ADDRESS="${RAY_NODE_IP_ADDRESS:-127.0.0.1}"
RAY_PORT="${RAY_PORT:-$((20000 + RANDOM % 20000))}"
RAY_NODE_MANAGER_PORT="${RAY_NODE_MANAGER_PORT:-$((40000 + RANDOM % 1000))}"
RAY_OBJECT_MANAGER_PORT="${RAY_OBJECT_MANAGER_PORT:-$((41000 + RANDOM % 1000))}"
RAY_MIN_WORKER_PORT="${RAY_MIN_WORKER_PORT:-42000}"
RAY_MAX_WORKER_PORT="${RAY_MAX_WORKER_PORT:-42199}"
RAY_STOP_BEFORE_RUN="${RAY_STOP_BEFORE_RUN:-0}"
DREAMZERO_START_MPS="${DREAMZERO_START_MPS:-auto}" # auto|1|0; auto enables MPS for ours.
DREAMZERO_MPS_PIPE_ROOT="${DREAMZERO_MPS_PIPE_ROOT:-/tmp}"
DREAMZERO_MPS_LOG_ROOT="${DREAMZERO_MPS_LOG_ROOT:-/tmp}"
# On this host the global /tmp/nvidia-mps/control socket can make early CUDA
# initialization hang. By default, runs that do not start their own MPS daemon
# use an isolated, empty pipe directory so CUDA does not attach to that socket.
DREAMZERO_AVOID_GLOBAL_MPS_HANG="${DREAMZERO_AVOID_GLOBAL_MPS_HANG:-1}"

# Eval defaults. Keep max steps >= actor.model.num_action_chunks.
DREAMZERO_EVAL_TOTAL_NUM_ENVS="${DREAMZERO_EVAL_TOTAL_NUM_ENVS:-32}"
DREAMZERO_EVAL_ROLLOUT_EPOCH="${DREAMZERO_EVAL_ROLLOUT_EPOCH:-1}"
DREAMZERO_EVAL_MAX_STEPS="${DREAMZERO_EVAL_MAX_STEPS:-80}"
DREAMZERO_EVAL_MAX_EPISODE_STEPS="${DREAMZERO_EVAL_MAX_EPISODE_STEPS:-80}"
DREAMZERO_EVAL_SPECIFIC_RESET_ID="${DREAMZERO_EVAL_SPECIFIC_RESET_ID:-0}"
DREAMZERO_EVAL_AUTO_RESET="${DREAMZERO_EVAL_AUTO_RESET:-false}"
DREAMZERO_EVAL_IGNORE_TERMINATIONS="${DREAMZERO_EVAL_IGNORE_TERMINATIONS:-false}"
DREAMZERO_EVAL_SAVE_VIDEO="${DREAMZERO_EVAL_SAVE_VIDEO:-false}"

# Training defaults. Per-mode defaults below match the checked-in configs unless
# these variables are already set by the user.
DREAMZERO_TRAIN_MAX_EPOCHS="${DREAMZERO_TRAIN_MAX_EPOCHS:-${MAX_EPOCHS:-2}}"
DREAMZERO_TRAIN_MAX_STEPS="${DREAMZERO_TRAIN_MAX_STEPS:-${MAX_STEPS:--1}}"
DREAMZERO_TRAIN_SAVE_INTERVAL="${DREAMZERO_TRAIN_SAVE_INTERVAL:--1}"
DREAMZERO_TRAIN_MAX_EPISODE_STEPS="${DREAMZERO_TRAIN_MAX_EPISODE_STEPS:-128}"
DREAMZERO_TRAIN_MAX_STEPS_PER_ROLLOUT_EPOCH_USER_SET="${DREAMZERO_TRAIN_MAX_STEPS_PER_ROLLOUT_EPOCH+x}"
DREAMZERO_TRAIN_MAX_STEPS_PER_ROLLOUT_EPOCH="${DREAMZERO_TRAIN_MAX_STEPS_PER_ROLLOUT_EPOCH:-128}"
DREAMZERO_TRAIN_EVAL_NUM_ENVS="${DREAMZERO_TRAIN_EVAL_NUM_ENVS:-32}"
DREAMZERO_TRAIN_AUTO_RESET="${DREAMZERO_TRAIN_AUTO_RESET:-false}"
DREAMZERO_TRAIN_IGNORE_TERMINATIONS="${DREAMZERO_TRAIN_IGNORE_TERMINATIONS:-false}"
DREAMZERO_TRAIN_SAVE_VIDEO_USER_SET="${DREAMZERO_TRAIN_SAVE_VIDEO+x}"
DREAMZERO_TRAIN_SAVE_VIDEO="${DREAMZERO_TRAIN_SAVE_VIDEO:-false}"
DREAMZERO_EVAL_DURING_TRAIN_SAVE_VIDEO="${DREAMZERO_EVAL_DURING_TRAIN_SAVE_VIDEO:-false}"
DREAMZERO_ACTOR_GLOBAL_BATCH_SIZE="${DREAMZERO_ACTOR_GLOBAL_BATCH_SIZE:-${DREAMZERO_GLOBAL_BATCH_SIZE:-64}}"
DREAMZERO_ACTOR_MICRO_BATCH_SIZE="${DREAMZERO_ACTOR_MICRO_BATCH_SIZE:-${DREAMZERO_MICRO_BATCH_SIZE:-1}}"
DREAMZERO_ACTOR_GRADIENT_CHECKPOINTING="${DREAMZERO_ACTOR_GRADIENT_CHECKPOINTING:-false}"
DREAMZERO_FSDP_USE_ORIG_PARAMS="${DREAMZERO_FSDP_USE_ORIG_PARAMS:-true}"
DREAMZERO_FSDP_SHARDING_STRATEGY="${DREAMZERO_FSDP_SHARDING_STRATEGY:-full_shard}"
DREAMZERO_ROLLOUT_PIPELINE_STAGE_NUM="${DREAMZERO_ROLLOUT_PIPELINE_STAGE_NUM:-}"
DREAMZERO_TRAIN_CHUNK_STEP_MODE="${DREAMZERO_TRAIN_CHUNK_STEP_MODE:-}"
DREAMZERO_TRAIN_LOSS_TYPE="${DREAMZERO_TRAIN_LOSS_TYPE:-}"
DREAMZERO_ACTION_NOISE_METHOD="${DREAMZERO_ACTION_NOISE_METHOD:-flow_sde}"
DREAMZERO_ACTION_DENOISE_STEPS_USER_SET="${DREAMZERO_ACTION_DENOISE_STEPS+x}"
DREAMZERO_ACTION_DENOISE_STEPS="${DREAMZERO_ACTION_DENOISE_STEPS:-4}"
DREAMZERO_STALENESS_THRESHOLD="${DREAMZERO_STALENESS_THRESHOLD:-}"
DREAMZERO_CLIP_LOG_RATIO_MIN="${DREAMZERO_CLIP_LOG_RATIO_MIN:-}"
DREAMZERO_CLIP_LOG_RATIO_MAX="${DREAMZERO_CLIP_LOG_RATIO_MAX:-}"
DREAMZERO_ROLLOUT_ENABLE_OFFLOAD="${DREAMZERO_ROLLOUT_ENABLE_OFFLOAD:-false}"

###############################################################################
# Argument parsing and mode defaults
###############################################################################

if [ "$#" -gt 0 ]; then
  case "$1" in
    eval_2gpu|collocated|hybrid|async|ours)
      DREAMZERO_RUN_MODE="$1"
      shift
      ;;
  esac
fi

case "${DREAMZERO_RUN_MODE}" in
  eval_2gpu)
    CONFIG_NAME="${DREAMZERO_CONFIG_NAME:-libero_spatial_eval_dreamzero}"
    ENTRYPOINT="examples/embodiment/eval_embodied_agent.py"
    RUN_KIND="eval"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${DREAMZERO_DEFAULT_GPUS}}"
    DREAMZERO_ACTOR_PLACEMENT="${DREAMZERO_ACTOR_PLACEMENT:-${DREAMZERO_DEFAULT_PLACEMENT}}"
    DREAMZERO_ENV_PLACEMENT="${DREAMZERO_ENV_PLACEMENT:-${DREAMZERO_DEFAULT_PLACEMENT}}"
    DREAMZERO_ROLLOUT_PLACEMENT="${DREAMZERO_ROLLOUT_PLACEMENT:-${DREAMZERO_DEFAULT_PLACEMENT}}"
    ;;
  async)
    CONFIG_NAME="${DREAMZERO_CONFIG_NAME:-dreamzero_libero_async_ppo}"
    ENTRYPOINT="examples/embodiment/train_async.py"
    RUN_KIND="train"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${DREAMZERO_DEFAULT_GPUS}}"
    DREAMZERO_TRAIN_TOTAL_NUM_ENVS="${DREAMZERO_TRAIN_TOTAL_NUM_ENVS:-32}"
    DREAMZERO_ACTOR_PLACEMENT="${DREAMZERO_ACTOR_PLACEMENT:-${DREAMZERO_DEFAULT_PLACEMENT}}"
    DREAMZERO_ENV_PLACEMENT="${DREAMZERO_ENV_PLACEMENT:-${DREAMZERO_DEFAULT_PLACEMENT}}"
    DREAMZERO_ROLLOUT_PLACEMENT="${DREAMZERO_ROLLOUT_PLACEMENT:-${DREAMZERO_DEFAULT_PLACEMENT}}"
    ;;
  collocated)
    CONFIG_NAME="${DREAMZERO_CONFIG_NAME:-dreamzero_libero_collocated_ppo}"
    ENTRYPOINT="examples/embodiment/train_embodied_agent.py"
    RUN_KIND="train"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${DREAMZERO_DEFAULT_GPUS}}"
    DREAMZERO_TRAIN_TOTAL_NUM_ENVS="${DREAMZERO_TRAIN_TOTAL_NUM_ENVS:-32}"
    DREAMZERO_ACTOR_PLACEMENT="${DREAMZERO_ACTOR_PLACEMENT:-${DREAMZERO_DEFAULT_PLACEMENT}}"
    DREAMZERO_ENV_PLACEMENT="${DREAMZERO_ENV_PLACEMENT:-${DREAMZERO_DEFAULT_PLACEMENT}}"
    DREAMZERO_ROLLOUT_PLACEMENT="${DREAMZERO_ROLLOUT_PLACEMENT:-${DREAMZERO_DEFAULT_PLACEMENT}}"
    ;;
  hybrid)
    CONFIG_NAME="${DREAMZERO_CONFIG_NAME:-dreamzero_libero_hybrid_ppo}"
    ENTRYPOINT="examples/embodiment/train_embodied_agent.py"
    RUN_KIND="train"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${DREAMZERO_DEFAULT_GPUS}}"
    DREAMZERO_TRAIN_TOTAL_NUM_ENVS="${DREAMZERO_TRAIN_TOTAL_NUM_ENVS:-32}"
    DREAMZERO_ACTOR_PLACEMENT="${DREAMZERO_ACTOR_PLACEMENT:-${DREAMZERO_DEFAULT_PLACEMENT}}"
    DREAMZERO_ENV_PLACEMENT="${DREAMZERO_ENV_PLACEMENT:-0-1}"
    DREAMZERO_ROLLOUT_PLACEMENT="${DREAMZERO_ROLLOUT_PLACEMENT:-2-3}"
    ;;
  ours)
    CONFIG_NAME="${DREAMZERO_CONFIG_NAME:-dreamzero_libero_ours_ppo}"
    ENTRYPOINT="examples/embodiment/train_async.py"
    RUN_KIND="train"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${DREAMZERO_DEFAULT_GPUS}}"
    DREAMZERO_TRAIN_TOTAL_NUM_ENVS="${DREAMZERO_TRAIN_TOTAL_NUM_ENVS:-32}"
    if [ -z "${DREAMZERO_TRAIN_SAVE_VIDEO_USER_SET}" ]; then
      DREAMZERO_TRAIN_SAVE_VIDEO=false
    fi
    DREAMZERO_ACTOR_PLACEMENT="${DREAMZERO_ACTOR_PLACEMENT:-${DREAMZERO_DEFAULT_PLACEMENT}}"
    DREAMZERO_ENV_PLACEMENT="${DREAMZERO_ENV_PLACEMENT:-${DREAMZERO_DEFAULT_PLACEMENT}}"
    DREAMZERO_ROLLOUT_PLACEMENT="${DREAMZERO_ROLLOUT_PLACEMENT:-${DREAMZERO_DEFAULT_PLACEMENT}}"
    DREAMZERO_RESOURCE_POOL_GPU_DEVICES="${DREAMZERO_RESOURCE_POOL_GPU_DEVICES:-${DREAMZERO_DEFAULT_PLACEMENT}}"
    DREAMZERO_ROLLOUT_PIPELINE_STAGE_NUM="${DREAMZERO_ROLLOUT_PIPELINE_STAGE_NUM:-1}"
    DREAMZERO_TRAIN_CHUNK_STEP_MODE="${DREAMZERO_TRAIN_CHUNK_STEP_MODE:-latency_balanced_pair}"
    if [ -z "${DREAMZERO_ACTION_DENOISE_STEPS_USER_SET}" ]; then
      DREAMZERO_ACTION_DENOISE_STEPS=4
    fi
    DREAMZERO_STALENESS_THRESHOLD="${DREAMZERO_STALENESS_THRESHOLD:-2}"
    DREAMZERO_CLIP_LOG_RATIO_MIN="${DREAMZERO_CLIP_LOG_RATIO_MIN:--20}"
    DREAMZERO_CLIP_LOG_RATIO_MAX="${DREAMZERO_CLIP_LOG_RATIO_MAX:-20}"
    ;;
  *)
    echo "[DreamZero-LIBERO] Unsupported DREAMZERO_RUN_MODE=${DREAMZERO_RUN_MODE}" >&2
    echo "[DreamZero-LIBERO] Use one of: eval_2gpu, collocated, hybrid, async, ours." >&2
    exit 1
    ;;
esac
export CUDA_VISIBLE_DEVICES
export DREAMZERO_ACTION_HEAD_TRAIN_ARCHITECTURE
export DREAMZERO_ACTION_HEAD_DEFER_LORA_INJECTION
export DREAMZERO_ACTION_HEAD_DEBUG_LOGS
export DREAMZERO_ACTION_NOISE_METHOD
export DREAMZERO_ACTION_DENOISE_STEPS
export DREAMZERO_RELEASE_ROLLOUT_CACHE_AFTER_GENERATE

###############################################################################
# Runtime setup
###############################################################################

cd "${REPO_PATH}"
: "${PYTHONPATH:=}"
if [ ! -f "${DREAMZERO_VENV_PATH}/bin/activate" ]; then
  echo "[DreamZero-LIBERO] Missing Python env: ${DREAMZERO_VENV_PATH}/bin/activate" >&2
  echo "[DreamZero-LIBERO] Install with: MODEL=dreamzero ENV_NAME=libero bash requirements/install.sh --venv $(basename "${DREAMZERO_VENV_PATH}") --no-root" >&2
  exit 1
fi
source "${DREAMZERO_VENV_PATH}/bin/activate"

export REPO_PATH="${REPO_PATH}"
export EMBODIED_PATH="${REPO_PATH}/examples/embodiment"
export DREAMZERO_PATH="${DREAMZERO_SOURCE_PATH}"
export PYTHONPATH="${REPO_PATH}:${DREAMZERO_SOURCE_PATH}:${PYTHONPATH}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export ROBOT_PLATFORM="${ROBOT_PLATFORM:-LIBERO}"
export LIBERO_TYPE="${LIBERO_TYPE:-standard}"
export NO_ALBUMENTATIONS_UPDATE=1
export OMEGACONF_MAX_YAML_EXPANDED_NODES="${OMEGACONF_MAX_YAML_EXPANDED_NODES:-500000}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export DREAMZERO_COMPILE_SCHEDULER="${DREAMZERO_COMPILE_SCHEDULER:-0}"
export DREAMZERO_DISABLE_TORCH_COMPILE
export DREAMZERO_IMAGE_ENCODER_PATH="${DREAMZERO_IMAGE_ENCODER_PATH:-${DREAMZERO_WAN21_CLIP_PATH}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth}"
export DREAMZERO_TEXT_ENCODER_PATH="${DREAMZERO_TEXT_ENCODER_PATH:-${DREAMZERO_WAN_PATH}/models_t5_umt5-xxl-enc-bf16.pth}"
export DREAMZERO_VAE_PATH="${DREAMZERO_VAE_PATH:-${DREAMZERO_WAN_PATH}/Wan2.2_VAE.pth}"
export DREAMZERO_METADATA_JSON_PATH="${DREAMZERO_METADATA_JSON_PATH:-${DREAMZERO_MODEL_PATH}/experiment_cfg/metadata.json}"

if [ "${DREAMZERO_DISABLE_TORCH_COMPILE}" = "true" ] \
  || [ "${DREAMZERO_DISABLE_TORCH_COMPILE}" = "1" ]; then
  export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
  export TORCHINDUCTOR_CUDAGRAPHS="${TORCHINDUCTOR_CUDAGRAPHS:-0}"
fi

require_file() {
  if [ ! -f "$1" ]; then
    echo "[DreamZero-LIBERO] Missing file: $1" >&2
    exit 1
  fi
}

require_dir() {
  if [ ! -d "$1" ]; then
    echo "[DreamZero-LIBERO] Missing directory: $1" >&2
    exit 1
  fi
}

require_dir "${DREAMZERO_SOURCE_PATH}/groot"
require_dir "${DREAMZERO_MODEL_PATH}"
require_file "${DREAMZERO_MODEL_PATH}/config.json"
require_file "${DREAMZERO_MODEL_PATH}/experiment_cfg/metadata.json"
if [ ! -f "${DREAMZERO_MODEL_PATH}/model.safetensors.index.json" ] \
  && [ ! -f "${DREAMZERO_MODEL_PATH}/model.safetensors" ]; then
  echo "[DreamZero-LIBERO] No complete model safetensors manifest found under ${DREAMZERO_MODEL_PATH}" >&2
  exit 1
fi

require_dir "${DREAMZERO_WAN_PATH}"
require_dir "${DREAMZERO_WAN21_CLIP_PATH}"
require_file "${DREAMZERO_IMAGE_ENCODER_PATH}"
require_file "${DREAMZERO_TEXT_ENCODER_PATH}"
require_file "${DREAMZERO_VAE_PATH}"
require_file "${DREAMZERO_WAN_PATH}/diffusion_pytorch_model.safetensors.index.json"
require_file "${DREAMZERO_TOKENIZER_PATH}/tokenizer.json"
require_file "${REPO_PATH}/${ENTRYPOINT}"

RUN_ID="${DREAMZERO_RUN_ID:-$(date +'%Y%m%d-%H%M%S')-${DREAMZERO_RUN_MODE}-${CONFIG_NAME}}"
LOG_DIR="${LOG_ROOT%/}/${RUN_ID}"
LOG_FILE="${LOG_DIR}/run_dreamzero_libero.log"
RUNTIME_MODEL_PATH="${RUNTIME_MODEL_ROOT%/}/${RUN_ID}/DreamZero-LIBERO-runtime"
MPS_ENABLED=0
MPS_WATCHDOG_PID=""
export RLINF_DREAMZERO_LOG_DIR="${LOG_DIR}"
# Ray creates Unix sockets below this directory; keep the root path short and
# put it on a filesystem with free space. Override RAY_TMPDIR_ROOT on new hosts.
export RAY_TMPDIR="${RAY_TMPDIR:-${RAY_TMPDIR_ROOT%/}}"
export TMPDIR="${TMPDIR:-${LOG_DIR}/tmp}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${LOG_DIR}/matplotlib}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${TMPDIR}/torchinductor_${USER:-user}}"

mkdir -p "${LOG_DIR}" "${RAY_TMPDIR}" "${TMPDIR}" "${MPLCONFIGDIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${RUNTIME_MODEL_ROOT}"

should_start_mps() {
  case "${DREAMZERO_START_MPS}" in
    1|true|yes|on)
      return 0
      ;;
    0|false|no|off)
      return 1
      ;;
    auto)
      [ "${DREAMZERO_RUN_MODE}" = "ours" ]
      return
      ;;
    *)
      echo "[DreamZero-LIBERO] Unsupported DREAMZERO_START_MPS=${DREAMZERO_START_MPS}" >&2
      echo "[DreamZero-LIBERO] Use one of: auto, 1, 0." >&2
      exit 1
      ;;
  esac
}

check_mps_control() {
  echo ps | nvidia-cuda-mps-control >/dev/null 2>&1
}

start_mps() {
  if ! command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
    echo "[DreamZero-LIBERO] nvidia-cuda-mps-control not found; cannot start MPS." >&2
    exit 1
  fi

  export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-${DREAMZERO_MPS_PIPE_ROOT%/}/nvidia-mps-${USER:-user}-${RUN_ID}}"
  export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-${DREAMZERO_MPS_LOG_ROOT%/}/nvidia-mps-log-${USER:-user}-${RUN_ID}}"
  mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"

  {
    echo "[DreamZero-LIBERO] starting NVIDIA MPS control daemon"
    echo "[DreamZero-LIBERO] mps_pipe_dir: ${CUDA_MPS_PIPE_DIRECTORY}"
    echo "[DreamZero-LIBERO] mps_log_dir: ${CUDA_MPS_LOG_DIRECTORY}"
  } | tee -a "${LOG_FILE}"
  nvidia-cuda-mps-control -d

  if ! check_mps_control; then
    echo "[DreamZero-LIBERO] MPS control daemon did not become available." >&2
    exit 1
  fi
  MPS_ENABLED=1
}

stop_mps() {
  if [ "${MPS_ENABLED}" = "1" ]; then
    echo "[DreamZero-LIBERO] stopping NVIDIA MPS control daemon" | tee -a "${LOG_FILE}"
    echo quit | nvidia-cuda-mps-control >/dev/null 2>&1 || true
    MPS_ENABLED=0
  fi
}

mps_watchdog() {
  local target_pid="$1"
  while kill -0 "${target_pid}" >/dev/null 2>&1; do
    if ! check_mps_control; then
      echo "[DreamZero-LIBERO] MPS control daemon is unavailable; terminating run." | tee -a "${LOG_FILE}" >&2
      kill -TERM "${target_pid}" >/dev/null 2>&1 || true
      return
    fi
    sleep 10
  done
}

configure_cuda_mps_pipe() {
  case "${DREAMZERO_AVOID_GLOBAL_MPS_HANG}" in
    1|true|yes|on)
      ;;
    0|false|no|off)
      return
      ;;
    *)
      echo "[DreamZero-LIBERO] Unsupported DREAMZERO_AVOID_GLOBAL_MPS_HANG=${DREAMZERO_AVOID_GLOBAL_MPS_HANG}" >&2
      echo "[DreamZero-LIBERO] Use one of: 1, 0." >&2
      exit 1
      ;;
  esac

  if should_start_mps; then
    return
  fi
  if [ -n "${CUDA_MPS_PIPE_DIRECTORY:-}" ]; then
    return
  fi

  export CUDA_MPS_PIPE_DIRECTORY="${TMPDIR%/}/cuda-mps-pipe-isolated"
  mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}"
}

stage_model_tree() {
  local staging_mode="$1"
  rm -rf "${RUNTIME_MODEL_PATH}"
  mkdir -p "${RUNTIME_MODEL_PATH}"

  case "${staging_mode}" in
    auto)
      if ln "${DREAMZERO_MODEL_PATH}/config.json" "${RUNTIME_MODEL_PATH}/.hardlink-test" 2>/dev/null; then
        rm -f "${RUNTIME_MODEL_PATH}/.hardlink-test"
        stage_model_tree hardlink
      else
        stage_model_tree symlink
      fi
      return
      ;;
    hardlink)
      cp -dRl --no-preserve=mode,ownership,timestamps,xattr,context \
        "${DREAMZERO_MODEL_PATH}/." "${RUNTIME_MODEL_PATH}/"
      ;;
    symlink)
      cp -a -s "${DREAMZERO_MODEL_PATH}/." "${RUNTIME_MODEL_PATH}/"
      ;;
    copy)
      cp -R --no-preserve=mode,ownership,timestamps \
        "${DREAMZERO_MODEL_PATH}/." "${RUNTIME_MODEL_PATH}/"
      ;;
    *)
      echo "[DreamZero-LIBERO] Unsupported DREAMZERO_RUNTIME_MODEL_STAGING=${DREAMZERO_RUNTIME_MODEL_STAGING}" >&2
      echo "[DreamZero-LIBERO] Use one of: auto, hardlink, symlink, copy." >&2
      exit 1
      ;;
  esac

  rm -f "${RUNTIME_MODEL_PATH}/config.json"
  cp --no-preserve=mode,ownership,timestamps \
    "${DREAMZERO_MODEL_PATH}/config.json" "${RUNTIME_MODEL_PATH}/config.json"
}

cleanup_runtime_model() {
  if [ "${DREAMZERO_KEEP_RUNTIME_MODEL:-0}" != "1" ] \
    && [ -n "${RUNTIME_MODEL_PATH:-}" ] \
    && [ -d "${RUNTIME_MODEL_PATH}" ]; then
    rm -rf "${RUNTIME_MODEL_PATH}"
    rmdir --ignore-fail-on-non-empty "$(dirname "${RUNTIME_MODEL_PATH}")" 2>/dev/null || true
  fi
}

patch_runtime_config() {
  python - "${RUNTIME_MODEL_PATH}/config.json" "${DREAMZERO_WAN_PATH}" <<'PY'
import json
import os
import sys

config_path, wan_path = sys.argv[1], sys.argv[2]
with open(config_path, "r") as f:
    config = json.load(f)

head_cfg = config["action_head_cfg"]["config"]
head_cfg["diffusion_model_cfg"]["diffusion_model_pretrained_path"] = wan_path
head_cfg["text_encoder_cfg"]["text_encoder_pretrained_path"] = os.environ[
    "DREAMZERO_TEXT_ENCODER_PATH"
]
head_cfg["image_encoder_cfg"]["image_encoder_pretrained_path"] = os.environ[
    "DREAMZERO_IMAGE_ENCODER_PATH"
]
head_cfg["vae_cfg"]["vae_pretrained_path"] = os.environ["DREAMZERO_VAE_PATH"]
head_cfg["train_architecture"] = os.environ.get(
    "DREAMZERO_ACTION_HEAD_TRAIN_ARCHITECTURE", "lora"
)
head_cfg["defer_lora_injection"] = os.environ.get(
    "DREAMZERO_ACTION_HEAD_DEFER_LORA_INJECTION", "false"
).lower() in ("1", "true", "yes", "on")
head_cfg["debug_logging"] = os.environ.get(
    "DREAMZERO_ACTION_HEAD_DEBUG_LOGS", "false"
).lower() in ("1", "true", "yes", "on")

for key, value in {
    "diffusion_model_pretrained_path": wan_path,
    "image_encoder_pretrained_path": os.environ["DREAMZERO_IMAGE_ENCODER_PATH"],
    "text_encoder_pretrained_path": os.environ["DREAMZERO_TEXT_ENCODER_PATH"],
    "vae_pretrained_path": os.environ["DREAMZERO_VAE_PATH"],
    "tokenizer_path": os.environ["DREAMZERO_TOKENIZER_PATH"],
    "metadata_json_path": os.environ["DREAMZERO_METADATA_JSON_PATH"],
}.items():
    config[key] = value

with open(config_path, "w") as f:
    json.dump(config, f, indent=2)
    f.write("\n")
PY
}

COMMON_OVERRIDES=(
  "runner.logger.log_path=${LOG_DIR}"
  "actor.model.model_path=${RUNTIME_MODEL_PATH}"
  "rollout.model.model_path=${RUNTIME_MODEL_PATH}"
  "actor.model.metadata_json_path=${RUNTIME_MODEL_PATH}/experiment_cfg/metadata.json"
  "actor.model.tokenizer_path=${DREAMZERO_TOKENIZER_PATH}"
  "actor.model.diffusion_model_pretrained_path=${DREAMZERO_WAN_PATH}"
  "actor.model.image_encoder_pretrained_path=${DREAMZERO_IMAGE_ENCODER_PATH}"
  "actor.model.text_encoder_pretrained_path=${DREAMZERO_TEXT_ENCODER_PATH}"
  "actor.model.vae_pretrained_path=${DREAMZERO_VAE_PATH}"
  "actor.model.embodiment_tag=${DREAMZERO_EMBODIMENT_TAG}"
  "~cluster.component_placement"
  "+cluster.component_placement.actor=${DREAMZERO_ACTOR_PLACEMENT}"
  "+cluster.component_placement.env=${DREAMZERO_ENV_PLACEMENT}"
  "+cluster.component_placement.rollout=${DREAMZERO_ROLLOUT_PLACEMENT}"
)

if [ "${RUN_KIND}" = "eval" ]; then
  HYDRA_OVERRIDES=(
    "${COMMON_OVERRIDES[@]}"
    "runner.max_epochs=1"
    "runner.only_eval=true"
    "algorithm.eval_rollout_epoch=${DREAMZERO_EVAL_ROLLOUT_EPOCH}"
    "algorithm.group_size=1"
    "env.train.total_num_envs=${DREAMZERO_EVAL_TOTAL_NUM_ENVS}"
    "env.eval.total_num_envs=${DREAMZERO_EVAL_TOTAL_NUM_ENVS}"
    "env.eval.max_steps_per_rollout_epoch=${DREAMZERO_EVAL_MAX_STEPS}"
    "env.eval.max_episode_steps=${DREAMZERO_EVAL_MAX_EPISODE_STEPS}"
    "env.eval.specific_reset_id=${DREAMZERO_EVAL_SPECIFIC_RESET_ID}"
    "env.eval.auto_reset=${DREAMZERO_EVAL_AUTO_RESET}"
    "env.eval.ignore_terminations=${DREAMZERO_EVAL_IGNORE_TERMINATIONS}"
    "env.eval.video_cfg.save_video=${DREAMZERO_EVAL_SAVE_VIDEO}"
  )
else
  HYDRA_OVERRIDES=(
    "${COMMON_OVERRIDES[@]}"
    "runner.max_epochs=${DREAMZERO_TRAIN_MAX_EPOCHS}"
    "runner.max_steps=${DREAMZERO_TRAIN_MAX_STEPS}"
    "runner.save_interval=${DREAMZERO_TRAIN_SAVE_INTERVAL}"
    "actor.global_batch_size=${DREAMZERO_ACTOR_GLOBAL_BATCH_SIZE}"
    "actor.micro_batch_size=${DREAMZERO_ACTOR_MICRO_BATCH_SIZE}"
    "actor.model.gradient_checkpointing=${DREAMZERO_ACTOR_GRADIENT_CHECKPOINTING}"
    "actor.model.is_lora=$([ "${DREAMZERO_ACTION_HEAD_TRAIN_ARCHITECTURE}" = "lora" ] && echo true || echo false)"
    "actor.model.action_head_cfg.config.train_architecture=${DREAMZERO_ACTION_HEAD_TRAIN_ARCHITECTURE}"
    "actor.model.action_head_cfg.config.defer_lora_injection=${DREAMZERO_ACTION_HEAD_DEFER_LORA_INJECTION}"
    "actor.model.action_head_cfg.config.debug_logging=${DREAMZERO_ACTION_HEAD_DEBUG_LOGS}"
    "actor.fsdp_config.gradient_checkpointing=${DREAMZERO_ACTOR_GRADIENT_CHECKPOINTING}"
    "actor.fsdp_config.use_orig_params=${DREAMZERO_FSDP_USE_ORIG_PARAMS}"
    "actor.fsdp_config.sharding_strategy=${DREAMZERO_FSDP_SHARDING_STRATEGY}"
    "env.train.total_num_envs=${DREAMZERO_TRAIN_TOTAL_NUM_ENVS}"
    "env.eval.total_num_envs=${DREAMZERO_TRAIN_EVAL_NUM_ENVS}"
    "env.train.max_episode_steps=${DREAMZERO_TRAIN_MAX_EPISODE_STEPS}"
    "env.eval.max_episode_steps=${DREAMZERO_TRAIN_MAX_EPISODE_STEPS}"
    "env.train.max_steps_per_rollout_epoch=${DREAMZERO_TRAIN_MAX_STEPS_PER_ROLLOUT_EPOCH}"
    "env.eval.max_steps_per_rollout_epoch=${DREAMZERO_TRAIN_MAX_STEPS_PER_ROLLOUT_EPOCH}"
    "env.train.auto_reset=${DREAMZERO_TRAIN_AUTO_RESET}"
    "env.train.ignore_terminations=${DREAMZERO_TRAIN_IGNORE_TERMINATIONS}"
    "env.train.video_cfg.save_video=${DREAMZERO_TRAIN_SAVE_VIDEO}"
    "env.eval.video_cfg.save_video=${DREAMZERO_EVAL_DURING_TRAIN_SAVE_VIDEO}"
    "actor.model.add_value_head=true"
    "actor.model.noise_method=${DREAMZERO_ACTION_NOISE_METHOD}"
    "actor.model.num_steps=${DREAMZERO_ACTION_DENOISE_STEPS}"
    "actor.model.safe_get_logprob=false"
    "actor.model.joint_logprob=false"
    "algorithm.group_size=1"
    "algorithm.rollout_epoch=1"
    "algorithm.eval_rollout_epoch=1"
    "algorithm.update_epoch=1"
    "rollout.recompute_logprobs=false"
    "rollout.enable_offload=${DREAMZERO_ROLLOUT_ENABLE_OFFLOAD}"
  )
  if [ -n "${DREAMZERO_TRAIN_LOSS_TYPE}" ]; then
    HYDRA_OVERRIDES+=("algorithm.loss_type=${DREAMZERO_TRAIN_LOSS_TYPE}")
  fi
  if [ -n "${DREAMZERO_STALENESS_THRESHOLD}" ]; then
    HYDRA_OVERRIDES+=(
      "algorithm.staleness_threshold=${DREAMZERO_STALENESS_THRESHOLD}"
    )
  fi
  if [ -n "${DREAMZERO_CLIP_LOG_RATIO_MIN}" ]; then
    HYDRA_OVERRIDES+=(
      "+algorithm.clip_log_ratio_min=${DREAMZERO_CLIP_LOG_RATIO_MIN}"
    )
  fi
  if [ -n "${DREAMZERO_CLIP_LOG_RATIO_MAX}" ]; then
    HYDRA_OVERRIDES+=(
      "+algorithm.clip_log_ratio_max=${DREAMZERO_CLIP_LOG_RATIO_MAX}"
    )
  fi
  if [ -n "${DREAMZERO_ROLLOUT_PIPELINE_STAGE_NUM}" ]; then
    HYDRA_OVERRIDES+=(
      "rollout.pipeline_stage_num=${DREAMZERO_ROLLOUT_PIPELINE_STAGE_NUM}"
    )
  fi
  if [ "${DREAMZERO_RUN_MODE}" = "ours" ]; then
    HYDRA_OVERRIDES+=(
      "cluster.resource_pool.gpu.pools.gpu_pool.devices=${DREAMZERO_RESOURCE_POOL_GPU_DEVICES}"
    )
  fi
  if [ -n "${DREAMZERO_TRAIN_CHUNK_STEP_MODE}" ]; then
    HYDRA_OVERRIDES+=(
      "env.train.chunk_step_mode=${DREAMZERO_TRAIN_CHUNK_STEP_MODE}"
    )
  fi
fi

if [ "$#" -gt 0 ]; then
  HYDRA_OVERRIDES+=("$@")
fi

configure_cuda_mps_pipe

COMMAND=(python "${ENTRYPOINT}" --config-path "${REPO_PATH}/examples/embodiment/config" --config-name "${CONFIG_NAME}" "${HYDRA_OVERRIDES[@]}")

if [ "${DREAMZERO_START_RAY_HEAD}" = "1" ]; then
  export RAY_ADDRESS="127.0.0.1:${RAY_PORT}"
else
  export RAY_ADDRESS="${RAY_ADDRESS:-local}"
fi

{
  echo "[DreamZero-LIBERO] run_mode: ${DREAMZERO_RUN_MODE}"
  echo "[DreamZero-LIBERO] log_dir: ${LOG_DIR}"
  echo "[DreamZero-LIBERO] repo: ${REPO_PATH}"
  echo "[DreamZero-LIBERO] python_env: ${DREAMZERO_VENV_PATH}"
  echo "[DreamZero-LIBERO] config: ${CONFIG_NAME}"
  echo "[DreamZero-LIBERO] train_loss_type: ${DREAMZERO_TRAIN_LOSS_TYPE}"
  echo "[DreamZero-LIBERO] action_noise_method: ${DREAMZERO_ACTION_NOISE_METHOD}"
  echo "[DreamZero-LIBERO] action_denoise_steps: ${DREAMZERO_ACTION_DENOISE_STEPS}"
  echo "[DreamZero-LIBERO] staleness_threshold: ${DREAMZERO_STALENESS_THRESHOLD:-config-default}"
  echo "[DreamZero-LIBERO] clip_log_ratio: ${DREAMZERO_CLIP_LOG_RATIO_MIN:-config-default}/${DREAMZERO_CLIP_LOG_RATIO_MAX:-config-default}"
  echo "[DreamZero-LIBERO] entrypoint: ${ENTRYPOINT}"
  echo "[DreamZero-LIBERO] visible_gpus: ${CUDA_VISIBLE_DEVICES}"
  echo "[DreamZero-LIBERO] placement actor/env/rollout: ${DREAMZERO_ACTOR_PLACEMENT}/${DREAMZERO_ENV_PLACEMENT}/${DREAMZERO_ROLLOUT_PLACEMENT}"
  echo "[DreamZero-LIBERO] resource_pool_gpu_devices: ${DREAMZERO_RESOURCE_POOL_GPU_DEVICES:-none}"
  echo "[DreamZero-LIBERO] model: ${DREAMZERO_MODEL_PATH}"
  echo "[DreamZero-LIBERO] wan: ${DREAMZERO_WAN_PATH}"
  echo "[DreamZero-LIBERO] wan21_clip: ${DREAMZERO_WAN21_CLIP_PATH}"
  echo "[DreamZero-LIBERO] runtime_model: ${RUNTIME_MODEL_PATH}"
  echo "[DreamZero-LIBERO] runtime_model_staging: ${DREAMZERO_RUNTIME_MODEL_STAGING}"
  echo "[DreamZero-LIBERO] disable_torch_compile: ${DREAMZERO_DISABLE_TORCH_COMPILE}"
  echo "[DreamZero-LIBERO] release_rollout_cache_after_generate: ${DREAMZERO_RELEASE_ROLLOUT_CACHE_AFTER_GENERATE}"
  echo "[DreamZero-LIBERO] rollout_enable_offload: ${DREAMZERO_ROLLOUT_ENABLE_OFFLOAD}"
  echo "[DreamZero-LIBERO] avoid_global_mps_hang: ${DREAMZERO_AVOID_GLOBAL_MPS_HANG}"
  echo "[DreamZero-LIBERO] cuda_mps_pipe_directory: ${CUDA_MPS_PIPE_DIRECTORY:-unset}"
  echo "[DreamZero-LIBERO] ray_address: ${RAY_ADDRESS}"
  echo "[DreamZero-LIBERO] ray head ports: ${RAY_NODE_IP_ADDRESS}:${RAY_PORT} node=${RAY_NODE_MANAGER_PORT} object=${RAY_OBJECT_MANAGER_PORT} workers=${RAY_MIN_WORKER_PORT}-${RAY_MAX_WORKER_PORT}"
  echo "[DreamZero-LIBERO] start_mps: ${DREAMZERO_START_MPS}"
  if should_start_mps; then
    echo "[DreamZero-LIBERO] start_mps_resolved: yes"
  else
    echo "[DreamZero-LIBERO] start_mps_resolved: no"
  fi
  echo "[DreamZero-LIBERO] command: ${COMMAND[*]}"
} | tee "${LOG_FILE}"

if [ "${DREAMZERO_DRY_RUN}" = "1" ] || [ "${DREAMZERO_DRY_RUN}" = "true" ]; then
  echo "[DreamZero-LIBERO] dry run complete; no Ray process or training/eval job was started." | tee -a "${LOG_FILE}"
  exit 0
fi

stage_model_tree "${DREAMZERO_RUNTIME_MODEL_STAGING}"
patch_runtime_config

cleanup_ray() {
  if [ "${DREAMZERO_START_RAY_HEAD}" = "1" ]; then
    echo "[DreamZero-LIBERO] stopping isolated Ray head at ${RAY_ADDRESS}"
    ray stop --force >/dev/null 2>&1 || true
  fi
  cleanup_runtime_model
}

cleanup_all() {
  if [ -n "${MPS_WATCHDOG_PID}" ]; then
    kill "${MPS_WATCHDOG_PID}" >/dev/null 2>&1 || true
  fi
  cleanup_ray
  stop_mps
}
trap cleanup_all EXIT

if [ "${RAY_STOP_BEFORE_RUN}" = "1" ]; then
  ray stop --force >/dev/null 2>&1 || true
fi

if should_start_mps; then
  start_mps
fi

if [ "${DREAMZERO_START_RAY_HEAD}" = "1" ]; then
  ray start --head \
    --node-ip-address="${RAY_NODE_IP_ADDRESS}" \
    --port="${RAY_PORT}" \
    --node-manager-port="${RAY_NODE_MANAGER_PORT}" \
    --object-manager-port="${RAY_OBJECT_MANAGER_PORT}" \
    --min-worker-port="${RAY_MIN_WORKER_PORT}" \
    --max-worker-port="${RAY_MAX_WORKER_PORT}" \
    --num-gpus="$(python - <<'PY'
import os
print(len([x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]))
PY
)" \
    --temp-dir="${RAY_TMPDIR}" \
    --include-dashboard=false \
    --disable-usage-stats
fi

if [ "${MPS_ENABLED}" = "1" ]; then
  set +e
  "${COMMAND[@]}" > >(tee -a "${LOG_FILE}") 2>&1 &
  job_pid=$!
  mps_watchdog "${job_pid}" &
  MPS_WATCHDOG_PID=$!
  wait "${job_pid}"
  status=$?
  kill "${MPS_WATCHDOG_PID}" >/dev/null 2>&1 || true
  MPS_WATCHDOG_PID=""
  set -e
  exit "${status}"
fi

"${COMMAND[@]}" 2>&1 | tee -a "${LOG_FILE}"
