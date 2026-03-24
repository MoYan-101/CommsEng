# SLURM Guide

This folder contains the SLURM entry scripts for the pipeline:

- `run.sh -> train.py -> inference.py -> visualization.py`
- `slurm/run_pipeline_cpu.sbatch`: one SLURM job, optional multiple alphas in sequence
- `slurm/run_pipeline_array_cpu.sbatch`: job array, one alpha per task

## Setup

Run this once per session after login:

```bash
cd /public/home/user2/CommsEng
chmod +x run.sh slurm/*.sbatch
mkdir -p logs
```

## Environment

Recommended environment name: `mlcpu`.

```bash
cd /public/home/user2/CommsEng
eval "$($HOME/bin/micromamba shell hook --shell=bash)"
micromamba create -n mlcpu python=3.10 -y
micromamba activate mlcpu
python -m pip install -U pip setuptools wheel
python -m pip install --prefer-binary -r requirements.txt
```

Before submitting jobs, sanity-check Python:

```bash
which python
python -V
python -c "import sys; print(sys.executable)"
python -m pip -V
```

## Quick Check

```bash
cd /public/home/user2/CommsEng
grep -n "path:\\|trials:" "${CONFIG_FILE:-configs/config.yaml}"
ls -lh data/Main_20260128_cleansed.csv
```

## Submit One Job

```bash
cd /public/home/user2/CommsEng
mkdir -p logs
sbatch --export=ALL,CONDA_ENV=mlcpu,OVERFIT_ALPHA_LIST=0.0 slurm/run_pipeline_cpu.sbatch
```

Multiple alphas in one job:

```bash
sbatch --export=ALL,CONDA_ENV=mlcpu,OVERFIT_ALPHA_LIST=0.0,0.03,0.05 slurm/run_pipeline_cpu.sbatch
```

## Submit a Job Array

Default alpha list:

- `0.0,0.01,0.03,0.05,0.07`

Submit:

```bash
cd /public/home/user2/CommsEng
mkdir -p logs
sbatch --export=ALL,CONDA_ENV=mlcpu slurm/run_pipeline_array_cpu.sbatch
```

Custom alpha list:

```bash
export ALPHA_LIST="0.0,0.02,0.04"
sbatch --array=0-2%3 --export=ALL,CONDA_ENV=mlcpu slurm/run_pipeline_array_cpu.sbatch
```

## Monitor Jobs

```bash
squeue -u "$USER"
squeue -u "$USER" -o "%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R"
```

Logs:

```bash
tail -f logs/slurm-MY-CE1-<jobid>.out
tail -f logs/slurm-MY-CE1-<jobid>_<taskid>.out
```

## Outputs

- `models/<csv_name>/<run_id>/...`
- `postprocessing/<csv_name>/<run_id>/...`
- `evaluation/figures/<csv_name>/<run_id>/...`

## Notes

- Always pass `OVERFIT_ALPHA_LIST` in SLURM. Otherwise `run.sh` may enter interactive mode.
- If you run multiple non-array jobs in parallel, use a unique `RUN_ID` to avoid overwriting outputs.

```bash
sbatch --export=ALL,CONDA_ENV=mlcpu,OVERFIT_ALPHA_LIST=0.0,RUN_ID=slurm_$(date +%Y%m%d_%H%M%S) slurm/run_pipeline_cpu.sbatch
```

- The array script already enables concurrent pipeline/training mode.
