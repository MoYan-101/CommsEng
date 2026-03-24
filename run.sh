#!/usr/bin/env bash
set -e
PY=""
if command -v conda >/dev/null 2>&1; then
  if [ -z "$CONDA_ENV" ]; then
    CONDA_ENV="comms310"
  fi
  if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
    PY="conda run --no-capture-output -n $CONDA_ENV python -u"
  fi
fi
if [ -z "$PY" ]; then
  PY_BIN="./.venv/bin/python3"
  if [ -x "$PY_BIN" ]; then
    PY="$PY_BIN -u"
  else
    PY="python3 -u"
  fi
fi

export CONFIG_FILE="${CONFIG_FILE:-configs/config.yaml}"
if [ ! -f "$CONFIG_FILE" ]; then
  echo "[ERROR] Config file not found: $CONFIG_FILE"
  exit 1
fi

if [ "${ALLOW_CONCURRENT_PIPELINE:-0}" != "1" ] && command -v flock >/dev/null 2>&1; then
  LOCK_FILE="${LOCK_FILE:-.run_pipeline.lock}"
  exec 9>"$LOCK_FILE"
  if ! flock -n 9; then
    echo "[ERROR] Another run.sh pipeline is already running (lock: $LOCK_FILE)."
    echo "[HINT] If this is intentional, set ALLOW_CONCURRENT_PIPELINE=1."
    exit 1
  fi
fi

CPU_TOTAL="${CPU_TOTAL:-$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN || echo 1)}"
if ! [ "$CPU_TOTAL" -ge 1 ] 2>/dev/null; then
  CPU_TOTAL=1
fi

# Parallel trials for Optuna (auto heuristic by CPU count).
if [ -z "$OPTUNA_N_JOBS" ]; then
  if [ "$CPU_TOTAL" -ge 16 ]; then
    OPTUNA_N_JOBS=4
  elif [ "$CPU_TOTAL" -ge 8 ]; then
    OPTUNA_N_JOBS=2
  else
    OPTUNA_N_JOBS=1
  fi
fi
export OPTUNA_N_JOBS

# Model-level parallelism (use all logical CPUs by default).
export MODEL_N_JOBS="${MODEL_N_JOBS:-$CPU_TOTAL}"
export RF_N_JOBS="${RF_N_JOBS:-$MODEL_N_JOBS}"
export CATBOOST_THREAD_COUNT="${CATBOOST_THREAD_COUNT:-$CPU_TOTAL}"
export XGB_N_JOBS="${XGB_N_JOBS:-$CPU_TOTAL}"
export SVM_N_JOBS="${SVM_N_JOBS:-$CPU_TOTAL}"

# Inference-stage model parallelism (can be tuned independently if needed).
export INFER_MODEL_N_JOBS="${INFER_MODEL_N_JOBS:-$MODEL_N_JOBS}"
export INFER_RF_N_JOBS="${INFER_RF_N_JOBS:-$RF_N_JOBS}"
export INFER_XGB_N_JOBS="${INFER_XGB_N_JOBS:-$XGB_N_JOBS}"
export INFER_CATBOOST_THREAD_COUNT="${INFER_CATBOOST_THREAD_COUNT:-$CATBOOST_THREAD_COUNT}"
export INFER_SVM_N_JOBS="${INFER_SVM_N_JOBS:-$SVM_N_JOBS}"
export INFER_MODEL_WORKERS="${INFER_MODEL_WORKERS:-1}"
export INFER_GRID_WORKERS="${INFER_GRID_WORKERS:-1}"
export INFER_GRID_CHUNK_ROWS="${INFER_GRID_CHUNK_ROWS:-8}"

# Avoid nested BLAS/OpenMP over-subscription when model-level n_jobs is high.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export BLIS_NUM_THREADS="${BLIS_NUM_THREADS:-1}"

# Torch CPU threading (for ANN and torch inference on CPU).
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-$CPU_TOTAL}"
TORCH_INTEROP_DEFAULT=$((CPU_TOTAL / 4))
if [ "$TORCH_INTEROP_DEFAULT" -lt 1 ]; then
  TORCH_INTEROP_DEFAULT=1
fi
if [ "$TORCH_INTEROP_DEFAULT" -gt 4 ]; then
  TORCH_INTEROP_DEFAULT=4
fi
export TORCH_NUM_INTEROP_THREADS="${TORCH_NUM_INTEROP_THREADS:-$TORCH_INTEROP_DEFAULT}"
export INFER_TORCH_NUM_THREADS="${INFER_TORCH_NUM_THREADS:-$TORCH_NUM_THREADS}"
export INFER_TORCH_NUM_INTEROP_THREADS="${INFER_TORCH_NUM_INTEROP_THREADS:-$TORCH_NUM_INTEROP_THREADS}"
if [ -z "$RUN_ID" ]; then
  BASE_RUN_ID=$(date +"%Y%m%d_%H%M%S")
else
  BASE_RUN_ID="$RUN_ID"
fi

if [ -z "$OVERFIT_ALPHA_LIST" ]; then
  echo -n "Enter overfit_penalty_alpha values (comma-separated, blank = use config): "
  read -r OVERFIT_ALPHA_LIST || true
fi

ALPHAS=()
if [ -z "$OVERFIT_ALPHA_LIST" ]; then
  ALPHAS+=("")
else
  ALPHAS_RAW=$(echo "$OVERFIT_ALPHA_LIST" | tr ',' ' ')
  for a in $ALPHAS_RAW; do
    ALPHAS+=("$a")
  done
fi

export PYTHONUNBUFFERED=1
if [ "${ALLOW_CONCURRENT_TRAIN:-0}" != "1" ]; then
  EXISTING_TRAIN=$(pgrep -af "python -u train.py" 2>/dev/null | sed '/pgrep -af "python -u train.py"/d' || true)
  if [ -n "$EXISTING_TRAIN" ]; then
    echo "[ERROR] Existing train.py process(es) detected:"
    echo "$EXISTING_TRAIN"
    echo "[HINT] Stop old jobs first, or set ALLOW_CONCURRENT_TRAIN=1 to bypass."
    exit 1
  fi
fi
echo "[INFO] CPU_TOTAL=$CPU_TOTAL OPTUNA_N_JOBS=$OPTUNA_N_JOBS MODEL_N_JOBS=$MODEL_N_JOBS RF_N_JOBS=$RF_N_JOBS CATBOOST_THREAD_COUNT=$CATBOOST_THREAD_COUNT XGB_N_JOBS=$XGB_N_JOBS SVM_N_JOBS=$SVM_N_JOBS INFER_MODEL_N_JOBS=$INFER_MODEL_N_JOBS INFER_RF_N_JOBS=$INFER_RF_N_JOBS INFER_CATBOOST_THREAD_COUNT=$INFER_CATBOOST_THREAD_COUNT INFER_XGB_N_JOBS=$INFER_XGB_N_JOBS INFER_SVM_N_JOBS=$INFER_SVM_N_JOBS INFER_MODEL_WORKERS=$INFER_MODEL_WORKERS INFER_GRID_WORKERS=$INFER_GRID_WORKERS INFER_GRID_CHUNK_ROWS=$INFER_GRID_CHUNK_ROWS TORCH_NUM_THREADS=$TORCH_NUM_THREADS TORCH_NUM_INTEROP_THREADS=$TORCH_NUM_INTEROP_THREADS INFER_TORCH_NUM_THREADS=$INFER_TORCH_NUM_THREADS INFER_TORCH_NUM_INTEROP_THREADS=$INFER_TORCH_NUM_INTEROP_THREADS OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "[INFO] CONFIG_FILE=$CONFIG_FILE"
for alpha in "${ALPHAS[@]}"; do
  if [ -n "$alpha" ]; then
    export OVERFIT_ALPHA="$alpha"
    RUN_ID="${BASE_RUN_ID}_alpha${alpha}"
  else
    unset OVERFIT_ALPHA
    RUN_ID="${BASE_RUN_ID}"
  fi
  export RUN_ID
  echo "[INFO] RUN_ID=$RUN_ID OVERFIT_ALPHA=${OVERFIT_ALPHA:-<config>}"
  $PY train.py --config "$CONFIG_FILE"
  $PY inference.py --config "$CONFIG_FILE"
  $PY visualization.py --config "$CONFIG_FILE"
done

