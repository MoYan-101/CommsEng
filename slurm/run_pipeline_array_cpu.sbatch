#!/bin/bash
#SBATCH --job-name=MY-CE1
#SBATCH --output=logs/slurm-%x-%A_%a.out
#SBATCH --error=logs/slurm-%x-%A_%a.err
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=120G
#SBATCH --time=120:00:00
#SBATCH --array=0-4%5
#SBATCH --exclusive

set -euo pipefail

# Default alpha list (requested):
#   0.0, 0.01, 0.03, 0.05, 0.07
# You may override with ALPHA_LIST (supports comma / semicolon / space separators).
ALPHA_LIST="${ALPHA_LIST:-0.0,0.01,0.03,0.05,0.07}"

PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
cd "$PROJECT_DIR"
mkdir -p logs

# Optional module loading (adapt for your cluster).
# module purge
# module load anaconda/2024.02

# Default env for this project.
CONDA_ENV="${CONDA_ENV:-mlcpu}"
ENV_MANAGER=""
MICROMAMBA_BIN=""

# Prefer micromamba first.
if command -v micromamba >/dev/null 2>&1; then
  MICROMAMBA_BIN="$(command -v micromamba)"
elif [ -x "$HOME/bin/micromamba" ]; then
  MICROMAMBA_BIN="$HOME/bin/micromamba"
elif [ -x "$HOME/.local/bin/micromamba" ]; then
  MICROMAMBA_BIN="$HOME/.local/bin/micromamba"
fi

if [ -n "$MICROMAMBA_BIN" ]; then
  eval "$("$MICROMAMBA_BIN" shell hook --shell=bash)"
  if micromamba activate "$CONDA_ENV" >/dev/null 2>&1; then
    ENV_MANAGER="micromamba"
  else
    echo "[WARN] micromamba activate '$CONDA_ENV' failed; fallback to conda/.venv."
  fi
fi

if [ -z "$ENV_MANAGER" ]; then
  # Fallback to conda for non-interactive shells.
  if ! command -v conda >/dev/null 2>&1; then
    for conda_sh in \
      "$HOME/miniconda3/etc/profile.d/conda.sh" \
      "$HOME/anaconda3/etc/profile.d/conda.sh" \
      "/opt/conda/etc/profile.d/conda.sh"
    do
      if [ -f "$conda_sh" ]; then
        # shellcheck source=/dev/null
        source "$conda_sh"
        break
      fi
    done
  fi

  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
    ENV_MANAGER="conda"
  elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    ENV_MANAGER=".venv"
  else
    echo "[ERROR] Python env not found: env='$CONDA_ENV' unavailable via micromamba/conda and .venv missing."
    exit 1
  fi
fi
echo "[INFO] ENV_MANAGER=$ENV_MANAGER CONDA_ENV=$CONDA_ENV PYTHON=$(command -v python)"

# Parse alpha list.
ALPHA_CLEAN="$(echo "$ALPHA_LIST" | tr ',;' '  ')"
read -r -a ALPHAS <<< "$ALPHA_CLEAN"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$TASK_ID" -lt 0 ] || [ "$TASK_ID" -ge "${#ALPHAS[@]}" ]; then
  echo "[ERROR] SLURM_ARRAY_TASK_ID=$TASK_ID out of range for ALPHA_LIST='$ALPHA_LIST'"
  echo "[HINT] Adjust --array range to 0-$((${#ALPHAS[@]} - 1))."
  exit 1
fi

ALPHA="${ALPHAS[$TASK_ID]}"
if [ -z "$ALPHA" ]; then
  echo "[ERROR] Empty alpha at index $TASK_ID from ALPHA_LIST='$ALPHA_LIST'"
  exit 1
fi

# Required for concurrent array tasks in the same project directory.
export ALLOW_CONCURRENT_PIPELINE=1
export ALLOW_CONCURRENT_TRAIN=1

# Each task runs one alpha via run.sh.
export OVERFIT_ALPHA_LIST="$ALPHA"
export CONFIG_FILE="${CONFIG_FILE:-configs/config.yaml}"
if [ ! -f "$CONFIG_FILE" ]; then
  echo "[WARN] Config '$CONFIG_FILE' not found, fallback to configs/config.yaml"
  export CONFIG_FILE="configs/config.yaml"
fi

CPU_TOTAL="${SLURM_CPUS_PER_TASK:-8}"
export CPU_TOTAL
export MODEL_N_JOBS="${MODEL_N_JOBS:-$CPU_TOTAL}"
export RF_N_JOBS="${RF_N_JOBS:-$CPU_TOTAL}"
export XGB_N_JOBS="${XGB_N_JOBS:-$CPU_TOTAL}"
export CATBOOST_THREAD_COUNT="${CATBOOST_THREAD_COUNT:-$CPU_TOTAL}"
export SVM_N_JOBS="${SVM_N_JOBS:-$CPU_TOTAL}"
export INFER_MODEL_N_JOBS="${INFER_MODEL_N_JOBS:-$MODEL_N_JOBS}"
export INFER_RF_N_JOBS="${INFER_RF_N_JOBS:-$RF_N_JOBS}"
export INFER_XGB_N_JOBS="${INFER_XGB_N_JOBS:-$XGB_N_JOBS}"
export INFER_CATBOOST_THREAD_COUNT="${INFER_CATBOOST_THREAD_COUNT:-$CATBOOST_THREAD_COUNT}"
export INFER_SVM_N_JOBS="${INFER_SVM_N_JOBS:-$SVM_N_JOBS}"
if [ -z "${INFER_MODEL_WORKERS:-}" ]; then
  if [ "$CPU_TOTAL" -ge 24 ]; then
    export INFER_MODEL_WORKERS=2
  else
    export INFER_MODEL_WORKERS=1
  fi
fi
if [ -z "${INFER_GRID_WORKERS:-}" ]; then
  if [ "$CPU_TOTAL" -ge 24 ]; then
    export INFER_GRID_WORKERS=2
  else
    export INFER_GRID_WORKERS=1
  fi
fi
export INFER_GRID_CHUNK_ROWS="${INFER_GRID_CHUNK_ROWS:-8}"

if [ -z "${OPTUNA_N_JOBS:-}" ]; then
  if [ "$CPU_TOTAL" -ge 24 ]; then
    export OPTUNA_N_JOBS=4
  elif [ "$CPU_TOTAL" -ge 12 ]; then
    export OPTUNA_N_JOBS=2
  else
    export OPTUNA_N_JOBS=1
  fi
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export BLIS_NUM_THREADS="${BLIS_NUM_THREADS:-1}"

export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-$CPU_TOTAL}"
export TORCH_NUM_INTEROP_THREADS="${TORCH_NUM_INTEROP_THREADS:-4}"
export INFER_TORCH_NUM_THREADS="${INFER_TORCH_NUM_THREADS:-$TORCH_NUM_THREADS}"
export INFER_TORCH_NUM_INTEROP_THREADS="${INFER_TORCH_NUM_INTEROP_THREADS:-$TORCH_NUM_INTEROP_THREADS}"
export PYTHONUNBUFFERED=1

echo "[INFO] JOB_ID=$SLURM_JOB_ID ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID HOST=$(hostname)"
echo "[INFO] PROJECT_DIR=$PROJECT_DIR"
echo "[INFO] ALPHA_LIST='$ALPHA_LIST' -> selected alpha=$ALPHA"
echo "[INFO] CONFIG_FILE=$CONFIG_FILE"
echo "[INFO] CPU_TOTAL=$CPU_TOTAL OPTUNA_N_JOBS=$OPTUNA_N_JOBS MODEL_N_JOBS=$MODEL_N_JOBS INFER_MODEL_N_JOBS=$INFER_MODEL_N_JOBS INFER_MODEL_WORKERS=$INFER_MODEL_WORKERS INFER_GRID_WORKERS=$INFER_GRID_WORKERS INFER_GRID_CHUNK_ROWS=$INFER_GRID_CHUNK_ROWS"

# Optional explicit RUN_ID (usually not needed; run.sh auto-generates).
# export RUN_ID="slurm_${SLURM_JOB_ID}_task${SLURM_ARRAY_TASK_ID}_a${ALPHA}"

bash run.sh
