# This repository is for the paper we submitted to Comms.Eng

This repository provides an end-to-end machine learning pipeline for catalyst-performance prediction. It covers data preprocessing, model training, hyperparameter tuning, inference, and figure generation in a single workflow.

## Pipeline

- `data_preprocessing/`: builds model-ready features from the raw CSV.
- `train.py`: trains RF, DT, CatBoost, XGB, ANN, and SVM models.
- `inference.py`: runs heatmap and confusion-style inference workflows.
- `visualization.py`: generates training, SHAP, Optuna, and inference figures.

## Quick Start

Use the single project config:

```bash
bash run.sh
```

Or run with an explicit config path:

```bash
CONFIG_FILE=configs/config.yaml bash run.sh
```

To skip the interactive alpha prompt:

```bash
OVERFIT_ALPHA_LIST="0.0,0.03" bash run.sh
```

## Main Config

The canonical config file is [configs/config.yaml](/home/mo/Documents/MY/CommsEng/CommsEng_Main/configs/config.yaml).

Common settings:

- `data.path`: input CSV path
- `model.types`: models to train
- `evaluation.*`: figure and report switches
- `inference.heatmap_axes`: axes used for 2D/3D heatmaps
- `optuna.*`: tuning controls

## Outputs

- `models/<csv_name>/<run_id>/`: trained models, scalers, metadata
- `postprocessing/<csv_name>/<run_id>/`: arrays and intermediate artifacts
- `evaluation/figures/<csv_name>/<run_id>/`: plots and summaries

## Notes

- 3D heatmaps can be expensive; use `skip_3d_models` and `max_combinations` to control runtime.
- SHAP for SVM can be slow because it relies on kernel-based explainers.
- For the complete dataset, please email the first author or the corresponding author of Yan, M., Yao, C., Wu, S. et al. A machine learning perspective on three decades of methanol synthesis: research framework and experimental operation insights. Commun Eng (2026). https://doi.org/10.1038/s44172-026-00706-4.
