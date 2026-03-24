#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train.py

Training entrypoint for regression models (RF/DT/CatBoost/XGB/SVM/ANN),
including preprocessing, optional Optuna tuning, evaluation, and artifact export.
"""

import argparse
import yaml
import os
import numpy as np
import torch
import joblib
import copy
import shap
from catboost import Pool
import optuna
import shutil  # Remove stale evaluation outputs before regeneration.
import json
from typing import cast
from data_preprocessing.my_dataset import MyDataset

from data_preprocessing.data_loader_modified import (
    load_smart_data_simple,
    load_raw_data_for_correlation,
    extract_data_statistics,
    build_group_value_vectors
)
from data_preprocessing.data_split import split_data
from data_preprocessing.scaler_utils import (
    standardize_data, inverse_transform_output, save_scaler
)

from models.model_ann import ANNRegression
from models.model_rf import RFRegression
from models.model_dt import DTRegression
from models.model_catboost import CatBoostRegression
from models.model_xgb import XGBRegression
from models.model_svm import SVMRegression

from losses.torch_losses import get_torch_loss_fn
from trainers.train_torch import train_torch_model_dataloader
from trainers.train_sklearn import train_sklearn_model
from evaluation.metrics import compute_regression_metrics, compute_mixed_metrics

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pandas as pd
from utils import (
    ensure_dir,
    get_model_dir,
    get_root_model_dir,
    get_postprocess_dir,
    get_eval_dir,
    get_run_id,
)
import re


def _resolve_config_path(config_path_arg: str | None = None) -> str:
    default_path = os.path.join(os.path.dirname(__file__), "configs", "config.yaml")
    candidate = (
        config_path_arg
        or os.environ.get("CONFIG_FILE")
        or os.environ.get("CONFIG_PATH")
        or default_path
    )
    if os.path.isabs(candidate):
        resolved = candidate
    else:
        cwd_candidate = os.path.abspath(candidate)
        if os.path.exists(cwd_candidate):
            resolved = cwd_candidate
        else:
            resolved = os.path.abspath(os.path.join(os.path.dirname(__file__), candidate))
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"Config file not found: {resolved}")
    return resolved


# -----------------------------------------------------------------------
# runtime tuning
# -----------------------------------------------------------------------
def _read_env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return None
    try:
        value = int(raw)
    except ValueError:
        print(f"[WARN] Invalid {name}='{raw}', expected integer; ignore.")
        return None
    if value < 1:
        print(f"[WARN] Invalid {name}='{raw}', expected >= 1; ignore.")
        return None
    return value


def _apply_runtime_overrides(config: dict) -> None:
    model_cfg = config.setdefault("model", {})
    rf_cfg = model_cfg.setdefault("rf_params", {})
    xgb_cfg = model_cfg.setdefault("xgb_params", {})
    cat_cfg = model_cfg.setdefault("catboost_params", {})
    svm_cfg = model_cfg.setdefault("svm_params", {})
    optuna_cfg = config.setdefault("optuna", {})

    model_n_jobs = _read_env_int("MODEL_N_JOBS")
    rf_n_jobs = _read_env_int("RF_N_JOBS")
    xgb_n_jobs = _read_env_int("XGB_N_JOBS")
    cat_threads = _read_env_int("CATBOOST_THREAD_COUNT")
    svm_n_jobs = _read_env_int("SVM_N_JOBS")
    optuna_n_jobs = _read_env_int("OPTUNA_N_JOBS")
    cpu_total = _read_env_int("CPU_TOTAL")
    if cpu_total is None:
        cpu_total = max(1, int(os.cpu_count() or 1))
    is_slurm = bool(os.environ.get("SLURM_JOB_ID"))

    # On SLURM, if user did not explicitly provide model n_jobs overrides,
    # default to full allocated CPU resources.
    if is_slurm and all(v is None for v in [model_n_jobs, rf_n_jobs, xgb_n_jobs, cat_threads, svm_n_jobs]):
        model_n_jobs = cpu_total

    # On SLURM, if Optuna parallelism is not explicitly set and config is conservative,
    # raise it to a practical high-throughput default.
    if is_slurm and optuna_n_jobs is None:
        try:
            cfg_optuna_jobs = int(optuna_cfg.get("n_jobs", 1))
        except Exception:
            cfg_optuna_jobs = 1
        if cfg_optuna_jobs <= 1:
            if cpu_total >= 24:
                optuna_n_jobs = 4
            elif cpu_total >= 12:
                optuna_n_jobs = 2
            else:
                optuna_n_jobs = 1

    if model_n_jobs is not None:
        rf_cfg["n_jobs"] = model_n_jobs
        xgb_cfg["n_jobs"] = model_n_jobs
        cat_cfg["thread_count"] = model_n_jobs
        svm_cfg["n_jobs"] = model_n_jobs
    if rf_n_jobs is not None:
        rf_cfg["n_jobs"] = rf_n_jobs
    if xgb_n_jobs is not None:
        xgb_cfg["n_jobs"] = xgb_n_jobs
    if cat_threads is not None:
        cat_cfg["thread_count"] = cat_threads
    if svm_n_jobs is not None:
        svm_cfg["n_jobs"] = svm_n_jobs
    if optuna_n_jobs is not None:
        optuna_cfg["n_jobs"] = optuna_n_jobs
    optuna_cfg.setdefault("n_jobs", 1)

    print(
        "[INFO] Model threads => "
        f"RF n_jobs={rf_cfg.get('n_jobs', -1)}, "
        f"XGB n_jobs={xgb_cfg.get('n_jobs', -1)}, "
        f"CatBoost thread_count={cat_cfg.get('thread_count', -1)}, "
        f"SVM n_jobs={svm_cfg.get('n_jobs', -1)}"
    )
    print(
        f"[INFO] Optuna parallel trials => n_jobs={optuna_cfg.get('n_jobs', 1)} "
        f"(cpu_total={cpu_total}, slurm={is_slurm})"
    )


def _configure_torch_runtime() -> None:
    torch_threads = _read_env_int("TORCH_NUM_THREADS")
    if torch_threads is not None:
        try:
            torch.set_num_threads(torch_threads)
        except Exception as e:
            print(f"[WARN] Failed to set TORCH_NUM_THREADS={torch_threads}: {e}")

    interop_threads = _read_env_int("TORCH_NUM_INTEROP_THREADS")
    if interop_threads is not None:
        try:
            torch.set_num_interop_threads(interop_threads)
        except Exception as e:
            print(f"[WARN] Failed to set TORCH_NUM_INTEROP_THREADS={interop_threads}: {e}")

    print(
        "[INFO] Torch threads => "
        f"num_threads={torch.get_num_threads()}, "
        f"num_interop_threads={torch.get_num_interop_threads()}"
    )


def _resolve_worker_threads(raw_threads, optuna_jobs, cpu_total):
    try:
        threads = int(raw_threads)
    except Exception:
        threads = -1
    if threads <= 0:
        threads = max(1, int(cpu_total))
    optuna_jobs = max(1, int(optuna_jobs))
    if optuna_jobs > 1:
        threads = max(1, threads // optuna_jobs)
    return threads


# -----------------------------------------------------------------------
# helper ranges
# -----------------------------------------------------------------------
def _safe_float_range(cfg, lo_key, hi_key, min_bump=1.01):
    low, high = float(cfg[lo_key]), float(cfg[hi_key])
    if low > high:
        low, high = high, low
    if low == high:
        high = low + (1e-8 if low == 0 else low * (min_bump - 1))
    return low, high

def _safe_int_range(cfg, lo_key, hi_key):
    low, high = int(cfg[lo_key]), int(cfg[hi_key])
    if low > high:
        low, high = high, low
    if low == high:
        high = low + 1
    return low, high

def _suggest_float_auto(trial, name, low, high, log_threshold=50.0):
    if low <= 0:
        return trial.suggest_float(name, low, high)
    if high / max(low, 1e-12) >= log_threshold:
        return trial.suggest_float(name, low, high, log=True)
    return trial.suggest_float(name, low, high)

def _to_2d(arr: np.ndarray) -> np.ndarray:
    """Ensure arrays are 2-D for consistent metric/scaler behavior."""
    return arr.reshape(-1, 1) if arr.ndim == 1 else arr

def _resolve_scale_cols_idx(config, model_type, numeric_cols_idx, n_features):
    """
    Choose input columns to standardize:
    - If preprocessing.standardize_all_features_for is configured, only models
      in that list scale all columns (including one-hot/embeddings).
    - Otherwise fall back to preprocessing.standardize_all_features.
    """
    pre_cfg = config.get("preprocessing", {})
    models_all = pre_cfg.get("standardize_all_features_for", None)
    if models_all is not None:
        use_all = model_type in set(models_all)
    else:
        use_all = pre_cfg.get("standardize_all_features", False)
    return list(range(n_features)) if use_all else numeric_cols_idx

def _apply_log_transform(X, x_col_names, cols, eps, numeric_cols_idx=None, tag=None):
    """Apply ln transform on selected numeric feature columns."""
    if not cols:
        return X
    eps = float(eps) if eps is not None else 1e-8
    X_new = X.copy()
    idx_map = {name: i for i, name in enumerate(x_col_names)}
    numeric_set = set(numeric_cols_idx or [])
    for c in cols:
        if c not in idx_map:
            print(f"[WARN] log_transform col '{c}' not found in features; skip.")
            continue
        idx = idx_map[c]
        if numeric_set and idx not in numeric_set:
            print(f"[WARN] log_transform col '{c}' not numeric; skip.")
            continue
        col = X_new[:, idx].astype(float)
        if np.any(col <= 0):
            print(f"[WARN] log_transform '{c}' has non-positive values; clamping to {eps}."
                  f"{' (' + str(tag) + ')' if tag else ''}")
        X_new[:, idx] = np.log(np.clip(col, eps, None))
    return X_new


def _restore_display_features_from_model_input(
    X_model: np.ndarray,
    scaler_x,
    scale_cols_idx,
    x_col_names,
    log_transform_cols,
    log_transform_eps,
) -> np.ndarray:
    """
    Restore model-input-domain features to display domain:
    1) inverse standardization on scaled columns
    2) exp() on configured log columns
    """
    X_disp = np.array(X_model, dtype=float, copy=True)

    if scaler_x is not None and scale_cols_idx:
        idx_list = list(scale_cols_idx)
        X_disp[:, idx_list] = scaler_x.inverse_transform(X_disp[:, idx_list])

    idx_map = {name: i for i, name in enumerate(x_col_names)}
    for cname in list(log_transform_cols or []):
        idx = idx_map.get(cname)
        if idx is None:
            continue
        # model domain stores ln(x), restore to original physical scale
        X_disp[:, idx] = np.exp(X_disp[:, idx])

    return X_disp

# -----------------------------------------------------------------------
# main tuner
# -----------------------------------------------------------------------
def tune_model(model_type, config, X, Y,
               numeric_cols_idx, x_col_names, y_col_names,
               random_seed):

    # 1) split & standardize --------------------------------------------------
    scale_cols_idx = _resolve_scale_cols_idx(config, model_type, numeric_cols_idx, X.shape[1])
    X_train, X_val, Y_train, Y_val = split_data(
        X, Y, test_size=config["data"]["test_size"], random_state=random_seed
    )
    (X_train_s, X_val_s, sx), (Y_train_s, Y_val_s, sy) = standardize_data(
        X_train, X_val, Y_train, Y_val,
        do_input = config["preprocessing"]["standardize_input"],
        do_output= config["preprocessing"]["standardize_output"],
        numeric_cols_idx=numeric_cols_idx,
        scale_cols_idx=scale_cols_idx
    )
    cpu_total = _read_env_int("CPU_TOTAL")
    if cpu_total is None:
        cpu_total = max(1, int(os.cpu_count() or 1))
    optuna_jobs = max(1, int(config.get("optuna", {}).get("n_jobs", 1)))
    tune_rf_n_jobs = _resolve_worker_threads(
        config.get("model", {}).get("rf_params", {}).get("n_jobs", -1),
        optuna_jobs, cpu_total
    )
    tune_xgb_n_jobs = _resolve_worker_threads(
        config.get("model", {}).get("xgb_params", {}).get("n_jobs", -1),
        optuna_jobs, cpu_total
    )
    tune_cat_threads = _resolve_worker_threads(
        config.get("model", {}).get("catboost_params", {}).get("thread_count", -1),
        optuna_jobs, cpu_total
    )
    tune_svm_n_jobs = _resolve_worker_threads(
        config.get("model", {}).get("svm_params", {}).get("n_jobs", -1),
        optuna_jobs, cpu_total
    )

    print(
        f"[INFO] {model_type} tuning plan => trials={config['optuna']['trials']}, "
        f"optuna_n_jobs={optuna_jobs}, "
        f"RF={tune_rf_n_jobs}, XGB={tune_xgb_n_jobs}, "
        f"CatBoost={tune_cat_threads}, SVM={tune_svm_n_jobs}"
    )

    # 2) Optuna objective -----------------------------------------------------
    def objective(trial):

        # -------- build & train each model -----------------------------------
        if model_type == "ANN":
            base = config["optuna"]["ann_params"]
            dims_choices = [",".join(map(str, d)) for d in base["hidden_dims_choices"]]
            hidden_dims = tuple(int(s) for s in trial.suggest_categorical(
                                "hidden_dims", dims_choices).split(","))

            d_lo,d_hi   = _safe_float_range(base, "dropout_min","dropout_max")
            lr_lo,lr_hi = _safe_float_range(base, "learning_rate_min","learning_rate_max")
            wd_lo,wd_hi = _safe_float_range(base, "weight_decay_min","weight_decay_max")

            dropout      = trial.suggest_float("dropout", d_lo, d_hi)
            lr           = trial.suggest_float("learning_rate",  lr_lo, lr_hi, log=True)
            weight_decay = trial.suggest_float("weight_decay",   wd_lo, wd_hi, log=True)
            batch_sz     = trial.suggest_categorical("batch_size", base["batch_size_choices"])
            optim        = trial.suggest_categorical("optimizer",  base["optimizer_choices"])
            actv         = trial.suggest_categorical("activation", base["activation_choices"])
            epochs       = base.get("tuning_epochs", 100)

            model_instance = ANNRegression(
                input_dim=X_train_s.shape[1],
                # output_dim=config["data"].get("output_len", 4),   #changed
                output_dim=config["data"]["output_len"],
                hidden_dims=hidden_dims,
                dropout=dropout,
                activation=actv,
                random_seed=random_seed
            )

            loss_fn = get_torch_loss_fn(config["loss"]["type"])

            train_ds, val_ds = MyDataset(X_train_s, Y_train_s), MyDataset(X_val_s, Y_val_s)

            model_instance, _, _ = train_torch_model_dataloader(
                model_instance, train_ds, val_ds,
                loss_fn=loss_fn,
                epochs=epochs,
                batch_size=batch_sz,
                lr=lr, weight_decay=weight_decay,
                checkpoint_path=None,
                log_interval=base.get("log_interval", 5),
                early_stopping=base.get("early_stopping", True),
                patience=base.get("patience", 10),
                optimizer_name=optim
            )

            model_instance.eval()
            dev = next(model_instance.parameters()).device
            with torch.no_grad():
                pred_val   = model_instance(torch.tensor(X_val_s,   dtype=torch.float32, device=dev)).cpu().numpy()
                pred_train = model_instance(torch.tensor(X_train_s, dtype=torch.float32, device=dev)).cpu().numpy()

        elif model_type == "RF":
            base = config["optuna"]["rf_params"]
            n_lo,n_hi   = _safe_int_range  (base,"n_estimators_min","n_estimators_max")
            d_lo,d_hi   = _safe_int_range  (base,"max_depth_min",   "max_depth_max")
            c_lo,c_hi   = _safe_float_range(base,"ccp_alpha_min",   "ccp_alpha_max")
            l_lo,l_hi   = _safe_int_range  (base,"min_samples_leaf_min","min_samples_leaf_max")

            model_instance = RFRegression(
                n_estimators    = trial.suggest_int ("n_estimators", n_lo, n_hi),
                max_depth       = trial.suggest_int ("max_depth",    d_lo, d_hi),
                ccp_alpha       = trial.suggest_float("ccp_alpha",   c_lo, c_hi),
                min_samples_leaf= trial.suggest_int ("min_samples_leaf", l_lo, l_hi),
                random_state=random_seed,
                n_jobs=tune_rf_n_jobs
            )
            model_instance = train_sklearn_model(model_instance, X_train_s, Y_train_s)
            pred_train, pred_val = model_instance.predict(X_train_s), model_instance.predict(X_val_s)

        elif model_type == "DT":
            base = config["optuna"]["dt_params"]
            d_lo,d_hi = _safe_int_range  (base,"max_depth_min","max_depth_max")
            c_lo,c_hi = _safe_float_range(base,"ccp_alpha_min","ccp_alpha_max")

            model_instance = DTRegression(
                max_depth   = trial.suggest_int  ("max_depth", d_lo, d_hi),
                ccp_alpha   = trial.suggest_float("ccp_alpha", c_lo, c_hi),
                random_state= config["model"]["dt_params"]["random_state"]
            )
            model_instance = train_sklearn_model(model_instance, X_train_s, Y_train_s)
            pred_train, pred_val = model_instance.predict(X_train_s), model_instance.predict(X_val_s)

        elif model_type == "CatBoost":
            base = config["optuna"]["catboost_params"]
            it_lo,it_hi = _safe_int_range  (base,"iterations_min","iterations_max")
            lr_lo,lr_hi = _safe_float_range(base,"learning_rate_min","learning_rate_max")
            dep_lo,dep_hi = _safe_int_range(base,"depth_min","depth_max")
            l2_lo,l2_hi = _safe_float_range(base,"l2_leaf_reg_min","l2_leaf_reg_max")
            rs_lo, rs_hi = _safe_float_range(
                base, "random_strength_min", "random_strength_max"
            ) if "random_strength_min" in base and "random_strength_max" in base else (1e-8, 1.0)
            bt_lo, bt_hi = _safe_float_range(
                base, "bagging_temperature_min", "bagging_temperature_max"
            ) if "bagging_temperature_min" in base and "bagging_temperature_max" in base else (0.0, 1.0)
            rsm_lo, rsm_hi = _safe_float_range(
                base, "rsm_min", "rsm_max"
            ) if "rsm_min" in base and "rsm_max" in base else (0.6, 1.0)
            es_rounds = config["model"]["catboost_params"].get("early_stopping_rounds", 50)

            model_instance = CatBoostRegression(
                iterations   = trial.suggest_int  ("iterations",   it_lo, it_hi),
                learning_rate= trial.suggest_float("learning_rate", lr_lo, lr_hi, log=True),
                depth        = trial.suggest_int  ("depth",        dep_lo, dep_hi),
                l2_leaf_reg  = trial.suggest_float("l2_leaf_reg",  l2_lo, l2_hi, log=True),
                random_strength = trial.suggest_float("random_strength", rs_lo, rs_hi, log=True),
                bagging_temperature = trial.suggest_float("bagging_temperature", bt_lo, bt_hi),
                rsm = trial.suggest_float("rsm", rsm_lo, rsm_hi),
                random_seed  = config["model"]["catboost_params"]["random_seed"],
                thread_count = tune_cat_threads
            )
            model_instance.fit(X_train_s, Y_train_s,
                               eval_set=(X_val_s, Y_val_s),
                               early_stopping_rounds=es_rounds,
                               use_best_model=True,
                               verbose=False)
            pred_train, pred_val = model_instance.predict(X_train_s), model_instance.predict(X_val_s)

        elif model_type == "XGB":
            base = config["optuna"]["xgb_params"]
            n_lo,n_hi   = _safe_int_range  (base,"n_estimators_min","n_estimators_max")
            lr_lo,lr_hi = _safe_float_range(base,"learning_rate_min","learning_rate_max")
            d_lo,d_hi   = _safe_int_range  (base,"max_depth_min","max_depth_max")
            a_lo,a_hi   = _safe_float_range(base,"reg_alpha_min","reg_alpha_max")
            l_lo,l_hi   = _safe_float_range(base,"reg_lambda_min","reg_lambda_max")
            mcw_lo, mcw_hi = _safe_float_range(
                base, "min_child_weight_min", "min_child_weight_max"
            ) if "min_child_weight_min" in base and "min_child_weight_max" in base else (1.0, 10.0)
            gm_lo, gm_hi = _safe_float_range(
                base, "gamma_min", "gamma_max"
            ) if "gamma_min" in base and "gamma_max" in base else (1e-8, 1.0)
            ss_lo, ss_hi = _safe_float_range(
                base, "subsample_min", "subsample_max"
            ) if "subsample_min" in base and "subsample_max" in base else (0.6, 1.0)
            cs_lo, cs_hi = _safe_float_range(
                base, "colsample_bytree_min", "colsample_bytree_max"
            ) if "colsample_bytree_min" in base and "colsample_bytree_max" in base else (0.6, 1.0)
            es_rounds = config["model"]["xgb_params"].get("early_stopping_rounds", 50)

            model_instance = XGBRegression(
                n_estimators = trial.suggest_int  ("n_estimators", n_lo, n_hi),
                learning_rate= trial.suggest_float("learning_rate", lr_lo, lr_hi, log=True),
                max_depth    = trial.suggest_int  ("max_depth", d_lo, d_hi),
                reg_alpha    = trial.suggest_float("reg_alpha", a_lo, a_hi, log=True),
                reg_lambda   = trial.suggest_float("reg_lambda", l_lo, l_hi, log=True),
                min_child_weight = _suggest_float_auto(trial, "min_child_weight", mcw_lo, mcw_hi),
                gamma        = trial.suggest_float("gamma", gm_lo, gm_hi, log=True),
                subsample    = trial.suggest_float("subsample", ss_lo, ss_hi),
                colsample_bytree = trial.suggest_float("colsample_bytree", cs_lo, cs_hi),
                random_state = config["model"]["xgb_params"]["random_seed"],
                n_jobs = tune_xgb_n_jobs
            )
            model_instance = train_sklearn_model(model_instance, X_train_s, Y_train_s,
                                                 X_val=X_val_s, Y_val=Y_val_s,
                                                 enable_early_stop=True, es_rounds=es_rounds)
            pred_train, pred_val = model_instance.predict(X_train_s), model_instance.predict(X_val_s)

        elif model_type == "SVM":
            base = config["optuna"]["svm_params"]
            c_lo, c_hi = _safe_float_range(base, "C_min", "C_max")
            e_lo, e_hi = _safe_float_range(base, "epsilon_min", "epsilon_max")
            g_lo, g_hi = _safe_float_range(base, "gamma_min", "gamma_max")
            max_iter = int(base.get("max_iter", 20000))
            k_choices = base.get("kernel_choices", None)
            if k_choices is None:
                k_cfg = base.get("kernel", "rbf")
                k_choices = list(k_cfg) if isinstance(k_cfg, (list, tuple)) else [k_cfg]
            kernel = trial.suggest_categorical("kernel", k_choices)

            if kernel in ["rbf", "poly", "sigmoid"]:
                gamma = _suggest_float_auto(trial, "gamma", g_lo, g_hi)
            else:
                gamma = "scale"

            degree = None
            coef0 = None
            if kernel == "poly":
                d_lo, d_hi = _safe_int_range(base, "degree_min", "degree_max")
                degree = trial.suggest_int("degree", d_lo, d_hi)
            if kernel in ["poly", "sigmoid"]:
                c0_lo, c0_hi = _safe_float_range(base, "coef0_min", "coef0_max")
                coef0 = _suggest_float_auto(trial, "coef0", c0_lo, c0_hi)

            model_instance = SVMRegression(
                kernel=kernel,
                C=_suggest_float_auto(trial, "C", c_lo, c_hi),
                epsilon=_suggest_float_auto(trial, "epsilon", e_lo, e_hi),
                gamma=gamma,
                degree=degree if degree is not None else 3,
                coef0=coef0 if coef0 is not None else 0.0,
                max_iter=max_iter,
                n_jobs=tune_svm_n_jobs
            )
            model_instance = train_sklearn_model(model_instance, X_train_s, Y_train_s)
            pred_train, pred_val = model_instance.predict(X_train_s), model_instance.predict(X_val_s)

        else:
            raise ValueError(f"Tuning for {model_type} not implemented.")

        # -------- metrics & objective  ---------------------------------------
        m_train = compute_regression_metrics(Y_train_s, pred_train)
        m_val   = compute_regression_metrics(Y_val_s,   pred_val)

        ratio  = m_val["MSE"] / max(m_train["MSE"], 1e-8)
        alpha  = float(config.get("optuna", {}).get("overfit_penalty_alpha", 1.0))
        obj    = m_val["MSE"] + alpha * ratio

        trial.set_user_attr("MSE_STD_VAL",  m_val["MSE"])
        trial.set_user_attr("MSE_STD_TRAIN",m_train["MSE"])
        trial.set_user_attr("Penalty",      alpha * ratio)

        # R2 in raw output domain.
        trial.set_user_attr(
            "R2_RAW",
            compute_regression_metrics(Y_val, inverse_transform_output(pred_val, sy))["R2"]
        )

        return obj

    # 3) run Optuna -----------------------------------------------------------
    sampler = optuna.samplers.TPESampler(seed=random_seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    restore_torch_threads = None
    restore_torch_interop = None
    if model_type == "ANN" and optuna_jobs > 1:
        restore_torch_threads = torch.get_num_threads()
        restore_torch_interop = torch.get_num_interop_threads()
        ann_tune_threads = _resolve_worker_threads(restore_torch_threads, optuna_jobs, cpu_total)
        try:
            torch.set_num_threads(ann_tune_threads)
        except Exception as e:
            print(f"[WARN] Failed to adjust ANN tune torch threads: {e}")
        try:
            torch.set_num_interop_threads(1)
        except Exception as e:
            print(f"[WARN] Failed to adjust ANN tune interop threads: {e}")
        print(
            "[INFO] ANN tune torch threads => "
            f"num_threads={torch.get_num_threads()}, "
            f"num_interop_threads={torch.get_num_interop_threads()}"
        )
    try:
        study.optimize(
            objective,
            n_trials=config["optuna"]["trials"],
            n_jobs=optuna_jobs,
            gc_after_trial=True
        )
    finally:
        if restore_torch_threads is not None:
            try:
                torch.set_num_threads(int(restore_torch_threads))
            except Exception as e:
                print(f"[WARN] Failed to restore torch num_threads: {e}")
        if restore_torch_interop is not None:
            try:
                torch.set_num_interop_threads(int(restore_torch_interop))
            except Exception as e:
                print(f"[WARN] Failed to restore torch num_interop_threads: {e}")

    best_params = study.best_params
    print(f"[{model_type}] Best Obj={study.best_value:.6f}, params={best_params}")

    csv_name = os.path.splitext(os.path.basename(config["data"]["path"]))[0]
    run_id = get_run_id(config)
    optuna_dir = get_postprocess_dir(csv_name, run_id, "optuna", model_type)
    ensure_dir(optuna_dir)
    # study.trials_dataframe().to_csv(os.path.join(optuna_dir, "trials.csv"), index=False)

    return study, best_params


def create_model_by_type(model_type, config, random_seed=42, input_dim=None):
    csv_name = os.path.splitext(os.path.basename(config["data"]["path"]))[0]
    run_id = get_run_id(config)
    best_params = None
    optuna_dir = get_postprocess_dir(csv_name, run_id, "optuna", model_type)
    best_params_path = os.path.join(optuna_dir, "best_params.pkl")
    if os.path.exists(best_params_path):
        best_params = joblib.load(best_params_path)
        if best_params and "hidden_dims" in best_params:
            if isinstance(best_params["hidden_dims"], str):
                best_params["hidden_dims"] = tuple(int(x.strip()) for x in best_params["hidden_dims"].split(','))
        print(f"[INFO] Using tuned hyperparameters for {model_type}: {best_params}")

    if model_type == "ANN":
        ann_cfg = config["model"].get("ann_params", {}).copy()
        ckpt_path = os.path.join(get_model_dir(csv_name, "ANN", run_id=run_id), "best_ann.pt")
        ensure_dir(os.path.dirname(ckpt_path))
        if best_params:
            ann_cfg.update(best_params)
        ann_cfg.setdefault("checkpoint_path", ckpt_path)
        # baseline defaults when optuna is disabled
        ann_cfg.setdefault("hidden_dims", (64, 64))
        ann_cfg.setdefault("dropout", 0.0)
        ann_cfg.setdefault("learning_rate", 1e-3)
        ann_cfg.setdefault("weight_decay", 0.0)
        ann_cfg.setdefault("batch_size", 128)
        ann_cfg.setdefault("optimizer", "AdamW")
        ann_cfg.setdefault("activation", "ReLU")
        ann_cfg.setdefault("random_seed", random_seed)
        out_dim = config["data"]["output_len"]
        actual_dim = input_dim if input_dim is not None else ann_cfg.get("input_dim", 14)
        model = ANNRegression(
            input_dim=actual_dim,
            # output_dim=ann_cfg.get("output_dim", 4),
            output_dim=out_dim,
            hidden_dims=ann_cfg["hidden_dims"],
            dropout=ann_cfg.get("dropout", 0.0),
            activation=ann_cfg.get("activation", "ReLU"),
            random_seed=ann_cfg.get("random_seed", random_seed)
        )
        return model, ann_cfg
    elif model_type == "RF":
        rf_cfg = config["model"].get("rf_params", {}).copy()
        if best_params:
            rf_cfg.update(best_params)
        rf_cfg.setdefault("n_estimators", 200)
        rf_cfg.setdefault("max_depth", 12)
        rf_cfg.setdefault("random_state", random_seed)
        rf_cfg.setdefault("ccp_alpha", 0.0)
        rf_cfg.setdefault("min_samples_leaf", 1)
        rf_cfg.setdefault("n_jobs", -1)
        return RFRegression(
            n_estimators=rf_cfg["n_estimators"],
            max_depth=rf_cfg["max_depth"],
            random_state=rf_cfg["random_state"],
            ccp_alpha=rf_cfg.get("ccp_alpha", 0.0),
            min_samples_leaf=rf_cfg.get("min_samples_leaf", 1),
            n_jobs=rf_cfg.get("n_jobs", -1)
        )
    elif model_type == "DT":
        dt_cfg = config["model"].get("dt_params", {}).copy()
        if best_params:
            dt_cfg.update(best_params)
        dt_cfg.setdefault("max_depth", 12)
        dt_cfg.setdefault("random_state", random_seed)
        dt_cfg.setdefault("ccp_alpha", 0.0)
        return DTRegression(
            max_depth=dt_cfg["max_depth"],
            random_state=dt_cfg["random_state"],
            ccp_alpha=dt_cfg.get("ccp_alpha", 0.0)
        )
    elif model_type == "CatBoost":
        cat_cfg = config["model"].get("catboost_params", {}).copy()
        if best_params:
            cat_cfg.update(best_params)
        cat_cfg.setdefault("iterations", 500)
        cat_cfg.setdefault("learning_rate", 0.1)
        cat_cfg.setdefault("depth", 8)
        cat_cfg.setdefault("random_seed", random_seed)
        cat_cfg.setdefault("l2_leaf_reg", 3.0)
        cat_cfg.setdefault("random_strength", 1.0)
        cat_cfg.setdefault("bagging_temperature", 0.0)
        cat_cfg.setdefault("rsm", 1.0)
        cat_cfg.setdefault("thread_count", -1)
        return CatBoostRegression(
            iterations=cat_cfg["iterations"],
            learning_rate=cat_cfg["learning_rate"],
            depth=cat_cfg["depth"],
            random_seed=cat_cfg["random_seed"],
            l2_leaf_reg=cat_cfg.get("l2_leaf_reg", 3.0),
            random_strength=cat_cfg.get("random_strength", 1.0),
            bagging_temperature=cat_cfg.get("bagging_temperature", 0.0),
            rsm=cat_cfg.get("rsm", 1.0),
            thread_count=cat_cfg.get("thread_count", -1)
        )
    elif model_type == "XGB":
        xgb_cfg = config["model"].get("xgb_params", {}).copy()
        if best_params:
            xgb_cfg.update(best_params)
        xgb_cfg.setdefault("n_estimators", 300)
        xgb_cfg.setdefault("learning_rate", 0.1)
        xgb_cfg.setdefault("max_depth", 6)
        xgb_cfg.setdefault("random_seed", random_seed)
        xgb_cfg.setdefault("reg_alpha", 0.0)
        xgb_cfg.setdefault("reg_lambda", 1.0)
        xgb_cfg.setdefault("min_child_weight", 1.0)
        xgb_cfg.setdefault("gamma", 0.0)
        xgb_cfg.setdefault("subsample", 1.0)
        xgb_cfg.setdefault("colsample_bytree", 1.0)
        xgb_cfg.setdefault("n_jobs", -1)
        return XGBRegression(
            n_estimators=xgb_cfg["n_estimators"],
            learning_rate=xgb_cfg["learning_rate"],
            max_depth=xgb_cfg["max_depth"],
            random_state=xgb_cfg["random_seed"],
            reg_alpha=xgb_cfg.get("reg_alpha", 0.0),
            reg_lambda=xgb_cfg.get("reg_lambda", 1.0),
            min_child_weight=xgb_cfg.get("min_child_weight", 1.0),
            gamma=xgb_cfg.get("gamma", 0.0),
            subsample=xgb_cfg.get("subsample", 1.0),
            colsample_bytree=xgb_cfg.get("colsample_bytree", 1.0),
            n_jobs=xgb_cfg.get("n_jobs", -1)
        )
    elif model_type == "SVM":
        svm_cfg = config["model"].get("svm_params", {}).copy()
        if best_params:
            svm_cfg.update(best_params)
        svm_cfg.setdefault("kernel", "rbf")
        svm_cfg.setdefault("C", 10.0)
        svm_cfg.setdefault("epsilon", 0.1)
        svm_cfg.setdefault("gamma", "scale")
        svm_cfg.setdefault("degree", 3)
        svm_cfg.setdefault("coef0", 0.0)
        svm_cfg.setdefault("max_iter", 20000)
        svm_cfg.setdefault("n_jobs", -1)
        return SVMRegression(
            kernel=svm_cfg["kernel"],
            C=svm_cfg["C"],
            epsilon=svm_cfg["epsilon"],
            gamma=svm_cfg["gamma"],
            degree=svm_cfg["degree"],
            coef0=svm_cfg["coef0"],
            max_iter=svm_cfg["max_iter"],
            n_jobs=svm_cfg["n_jobs"]
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")




def train_main(config_path: str | None = None):
    config_path = _resolve_config_path(config_path)
    print(f"[INFO] Using config => {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    _configure_torch_runtime()
    _apply_runtime_overrides(config)
    env_alpha = os.environ.get("OVERFIT_ALPHA")
    if env_alpha not in (None, ""):
        try:
            config.setdefault("optuna", {})["overfit_penalty_alpha"] = float(env_alpha)
            print(f"[INFO] OVERFIT_ALPHA override => {config['optuna']['overfit_penalty_alpha']}")
        except ValueError:
            print(f"[WARN] Invalid OVERFIT_ALPHA='{env_alpha}', using config value.")

    csv_path = config["data"]["path"]
    if not os.path.isabs(csv_path):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(config_path), ".."))
        csv_path = os.path.join(repo_root, csv_path)
        config["data"]["path"] = csv_path
    csv_name = os.path.splitext(os.path.basename(csv_path))[0]
    run_id = get_run_id(config)
    root_model_dir = get_root_model_dir(csv_name, run_id=run_id)
    ensure_dir(root_model_dir)

    base_outdir = get_postprocess_dir(csv_name, run_id, "train")
    ensure_dir(base_outdir)
    eval_dir = get_eval_dir(csv_name, run_id)
    ensure_dir(eval_dir)
    config_snapshot_path = os.path.join(eval_dir, "config_snapshot.yaml")
    try:
        with open(config_snapshot_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
        print(f"[INFO] Config snapshot saved => {config_snapshot_path}")
    except Exception as e:
        print(f"[WARN] Failed to save config snapshot: {e}")

    dl_cfg = config.get("data_loader", {})
    element_cols = tuple(dl_cfg.get("element_cols", ["Promoter 1", "Promoter 2"]))
    text_cols = tuple(dl_cfg.get("text_cols", ["Type of sysnthesis procedure"]))
    y_cols = dl_cfg.get("y_cols", None)
    element_embedding = dl_cfg.get("element_embedding", "advanced")
    drop_metadata_cols = tuple(dl_cfg.get("drop_metadata_cols", ["DOI", "Name", "Year"]))
    fill_numeric = dl_cfg.get("fill_numeric", "median")
    missing_text_token = dl_cfg.get("missing_text_token", "__MISSING__")
    impute_missing = dl_cfg.get("impute_missing", True)
    impute_method = dl_cfg.get("impute_method", "simple")
    impute_seed = dl_cfg.get("impute_seed", 42)
    preserve_null = dl_cfg.get("preserve_null", True)
    impute_type_substring = dl_cfg.get("impute_type_substring", "Type")
    impute_skip_substring = dl_cfg.get("impute_skip_substring", "ame")
    log_transform_cols = dl_cfg.get("log_transform_cols", None)
    log_transform_cols_extra_for = dl_cfg.get("log_transform_cols_extra_for", {}) or {}
    log_transform_eps = dl_cfg.get("log_transform_eps", 1e-8)
    if log_transform_cols_extra_for:
        print("[INFO] log_transform_cols_extra_for is ignored. "
              "Only data_loader.log_transform_cols are applied.")
    (X, Y, numeric_cols_idx, x_col_names, y_col_names,
     observed_values, observed_value_counts, observed_value_ratios,
     onehot_groups, feature_group_map) = cast(tuple, load_smart_data_simple(
        csv_path=csv_path,
        element_cols=element_cols,
        text_cols=text_cols,
        y_cols=y_cols,
        promoter_ratio_cols=dl_cfg.get("promoter_ratio_cols", None),
        promoter_onehot=dl_cfg.get("promoter_onehot", True),
        promoter_interaction_features=dl_cfg.get("promoter_interaction_features", True),
        promoter_pair_onehot=dl_cfg.get("promoter_pair_onehot", True),
        promoter_pair_onehot_min_count=dl_cfg.get("promoter_pair_onehot_min_count", 2),
        promoter_pair_onehot_max_categories=dl_cfg.get("promoter_pair_onehot_max_categories", 64),
        promoter_interaction_eps=dl_cfg.get("promoter_interaction_eps", 1e-8),
        log_transform_cols=log_transform_cols,
        log_transform_eps=log_transform_eps,
        element_embedding=element_embedding,
        drop_metadata_cols=drop_metadata_cols,
        fill_numeric=fill_numeric,
        missing_text_token=missing_text_token,
        impute_missing=impute_missing,
        impute_method=impute_method,
        impute_seed=impute_seed,
        preserve_null=preserve_null,
        impute_type_substring=impute_type_substring,
        impute_skip_substring=impute_skip_substring,
        return_dataframe=False
    ))

    X_base = X


    model_types = config.get("model", {}).get("types", [])
    log_base = list(log_transform_cols or [])
    X_by_model = {m: X for m in model_types}

    scale_cols_all = list(range(X_base.shape[1]))
    scale_cols_numeric = numeric_cols_idx
    scale_cols_idx_by_model = {
        m: _resolve_scale_cols_idx(config, m, numeric_cols_idx, X_base.shape[1])
        for m in model_types
    }
    # legacy default (kept for backward compatibility)
    scale_all_features = config.get("preprocessing", {}).get("standardize_all_features", False)
    scale_cols_idx = list(range(X_base.shape[1])) if scale_all_features else numeric_cols_idx

    np.save(os.path.join(base_outdir, "X_features.npy"), X_base)
    np.save(os.path.join(base_outdir, "x_feature_colnames.npy"), x_col_names)

    in_len = config["data"].get("input_len", None)
    out_len = config["data"].get("output_len", None)
    df_raw_14 = load_raw_data_for_correlation(
        csv_path,
        drop_nan=True,
        input_len=in_len,
        output_len=out_len,
        fill_same_as_train=True,
        element_cols=element_cols,
        promoter_ratio_cols=dl_cfg.get("promoter_ratio_cols", None),
        text_cols=text_cols,
        drop_metadata_cols=drop_metadata_cols,
        impute_seed=impute_seed,
        impute_type_substring=impute_type_substring,
        impute_skip_substring=impute_skip_substring,
        missing_text_token=missing_text_token,
        impute_method=impute_method,
        preserve_null=preserve_null
    )
    raw_csv_path = os.path.join(base_outdir, "df_raw_14.csv")
    df_raw_14.to_csv(raw_csv_path, index=False)
    print(f"[INFO] Saved raw 14-col CSV => {raw_csv_path}")

    stats_dict = extract_data_statistics(X_base, x_col_names, numeric_cols_idx, Y=Y, y_col_names=y_col_names)
    stats_dict["numeric_cols_idx"] = numeric_cols_idx
    stats_dict["scale_cols_idx"] = scale_cols_idx
    stats_dict["scale_cols_idx_by_model"] = scale_cols_idx_by_model
    stats_dict["onehot_groups"] = onehot_groups
    stats_dict["oh_index_map"] = feature_group_map
    stats_dict["observed_onehot_combos"] = observed_values
    stats_dict["observed_values"] = observed_values
    stats_dict["observed_value_counts"] = observed_value_counts
    stats_dict["observed_value_ratios"] = observed_value_ratios
    stats_dict["group_names"] = list(element_cols) + list(text_cols)
    stats_dict["group_value_vectors"] = build_group_value_vectors(
        observed_values=observed_values,
        observed_value_counts=observed_value_counts,
        observed_value_ratios=observed_value_ratios,
        element_cols=element_cols,
        text_cols=text_cols,
        element_embedding=element_embedding,
        promoter_onehot=dl_cfg.get("promoter_onehot", True)
    )
    stats_dict["feature_means"] = X_base.mean(axis=0)
    stats_dict["x_col_names"] = x_col_names
    stats_dict["y_col_names"] = y_col_names
    stats_dict["loader_config"] = {
        "element_cols": list(element_cols),
        "text_cols": list(text_cols),
        "y_cols": list(y_cols) if y_cols is not None else None,
        "promoter_ratio_cols": dl_cfg.get("promoter_ratio_cols", None),
        "promoter_onehot": dl_cfg.get("promoter_onehot", True),
        "promoter_interaction_features": dl_cfg.get("promoter_interaction_features", True),
        "promoter_pair_onehot": dl_cfg.get("promoter_pair_onehot", True),
        "promoter_pair_onehot_min_count": dl_cfg.get("promoter_pair_onehot_min_count", 2),
        "promoter_pair_onehot_max_categories": dl_cfg.get("promoter_pair_onehot_max_categories", 64),
        "promoter_interaction_eps": dl_cfg.get("promoter_interaction_eps", 1e-8),
        "log_transform_cols": log_base,
        "log_transform_cols_extra_for": {},
        "log_transform_cols_extra_for_requested": log_transform_cols_extra_for,
        "log_transform_eps": log_transform_eps,
        "element_embedding": element_embedding,
        "drop_metadata_cols": list(drop_metadata_cols),
        "fill_numeric": fill_numeric,
        "missing_text_token": missing_text_token,
        "impute_missing": impute_missing,
        "impute_method": impute_method,
        "impute_seed": impute_seed,
        "preserve_null": preserve_null,
        "impute_type_substring": impute_type_substring,
        "impute_skip_substring": impute_skip_substring,
    }

    meta_path = os.path.join(root_model_dir, "metadata.pkl")
    joblib.dump(stats_dict, meta_path)
    print(f"[INFO] metadata saved => {meta_path}")

    stats_dict_by_model = {}
    for m in model_types:
        stats_m = copy.deepcopy(stats_dict)
        stats_m["scale_cols_idx"] = scale_cols_idx_by_model.get(m, numeric_cols_idx)
        stats_m["loader_config"]["log_transform_cols"] = log_base
        stats_dict_by_model[m] = stats_m

    random_seed = config["data"].get("random_seed", 42)
    X_train, X_val, Y_train, Y_val = split_data(
        X_base, Y, test_size=config["data"]["test_size"], random_state=random_seed
    )
    bounded_cols = config["preprocessing"].get("bounded_output_columns", None)
    if bounded_cols is not None:
        bounded_indices = []
        for col in bounded_cols:
            if col in y_col_names:
                bounded_indices.append(y_col_names.index(col))
            else:
                print(f"[WARN] {col} not found in y_col_names")
    else:
        bounded_indices = None

    np.save(os.path.join(base_outdir, "Y_train.npy"), Y_train)
    np.save(os.path.join(base_outdir, "Y_val.npy"), Y_val)

    if config["optuna"].get("enable", False):
        for mtype in config["optuna"]["models"]:
            print(f"\n[INFO] Tuning hyperparameters for {mtype} ...")
            study, best_params = tune_model(
                mtype, config, X_by_model.get(mtype, X_base), Y,
                numeric_cols_idx, x_col_names, y_col_names, random_seed
            )
            optuna_dir = get_postprocess_dir(csv_name, run_id, "optuna", mtype)
            ensure_dir(optuna_dir)
            joblib.dump(study, os.path.join(optuna_dir, "study.pkl"))
            joblib.dump(best_params, os.path.join(optuna_dir, "best_params.pkl"))
            print(f"[INFO] Best params for {mtype} saved => {optuna_dir}")

    # --------------------------------------------------------------
    if config["evaluation"].get("do_cross_validation", False):
        kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
        cv_metrics = {}
        cv_metrics_path = os.path.join(base_outdir, "cv_metrics.pkl")

        for mtype in config["model"]["types"]:
            print(f"[INFO] Running 5-fold CV for model: {mtype}")
            scale_cols_idx_cv = _resolve_scale_cols_idx(config, mtype, numeric_cols_idx, X_base.shape[1])
            X_m = X_by_model.get(mtype, X_base)

            mse_tr, mse_va = [], []
            mae_tr, mae_va = [], []
            r2_tr, r2_va = [], []

            fold_id = 1
            for train_idx, val_idx in kf.split(X_m):
                print(f"  Fold {fold_id}: train={len(train_idx)}, val={len(val_idx)}")
                # -------------------------------- split / scale
                X_tr, X_va = X_m[train_idx], X_m[val_idx]
                Y_tr, Y_va = Y[train_idx], Y[val_idx]

                (X_tr_s, X_va_s, _), (Y_tr_s, Y_va_s, sy_fold) = standardize_data(
                    X_tr, X_va, Y_tr, Y_va,
                    do_input=config["preprocessing"]["standardize_input"],
                    do_output=config["preprocessing"]["standardize_output"],
                    numeric_cols_idx=numeric_cols_idx,
                    scale_cols_idx=scale_cols_idx_cv,
                    do_output_bounded=(bounded_indices is not None) or
                                      config["preprocessing"].get("bounded_output", False),
                    bounded_output_cols_idx=bounded_indices
                )

                # -------------------------------- build & train model_cv
                if mtype == "ANN":
                    model_cv, ann_cfg = cast(
                        tuple[ANNRegression, dict],
                        create_model_by_type("ANN", config, random_seed, input_dim=X_tr_s.shape[1])
                    )
                    if "epochs" not in ann_cfg:
                        ann_cfg["epochs"] = config["model"].get("ann_params", {}).get("epochs", 6000)

                    loss_fn = get_torch_loss_fn(config["loss"]["type"])
                    train_ds = MyDataset(X_tr_s, Y_tr_s)
                    val_ds = MyDataset(X_va_s, Y_va_s)

                    model_cv, _, _ = train_torch_model_dataloader(
                        model_cv, train_ds, val_ds,
                        loss_fn=loss_fn,
                        epochs=ann_cfg["epochs"],
                        batch_size=ann_cfg["batch_size"],
                        lr=float(ann_cfg["learning_rate"]),
                        weight_decay=float(ann_cfg.get("weight_decay", 0.0)),
                        checkpoint_path=None,
                        log_interval=config["training"]["log_interval"],
                        early_stopping=ann_cfg.get("early_stopping", True),
                        patience=ann_cfg.get("patience", 5),
                        optimizer_name=ann_cfg.get("optimizer", "Adam")
                    )
                    model_cv.eval().to("cpu")
                else:
                    model_cv = create_model_by_type(mtype, config, random_seed,
                                                    input_dim=X_tr_s.shape[1])
                    es_flag = mtype in ["CatBoost", "XGB"]
                    es_round = config["model"][f"{mtype.lower()}_params"].get(
                        "early_stopping_rounds", 50)
                    model_cv = train_sklearn_model(
                        model_cv, X_tr_s, Y_tr_s,
                        X_val=X_va_s, Y_val=Y_va_s,
                        enable_early_stop=es_flag, es_rounds=es_round
                    )

                if hasattr(model_cv, "eval") and hasattr(model_cv, "forward"):
                    with torch.no_grad():
                        pred_tr = model_cv(torch.tensor(X_tr_s, dtype=torch.float32)).cpu().numpy()
                        pred_va = model_cv(torch.tensor(X_va_s, dtype=torch.float32)).cpu().numpy()
                else:
                    pred_tr = model_cv.predict(X_tr_s)
                    pred_va = model_cv.predict(X_va_s)

                m_tr_std = compute_regression_metrics(Y_tr_s, pred_tr)
                m_va_std = compute_regression_metrics(Y_va_s, pred_va)

                mse_tr.append(m_tr_std["MSE"]);
                mse_va.append(m_va_std["MSE"])
                mae_tr.append(m_tr_std["MAE"]);
                mae_va.append(m_va_std["MAE"])
                r2_tr.append(m_tr_std["R2"]);
                r2_va.append(m_va_std["R2"])

                print(f"    [CV] Fold {fold_id}  MSE={m_va_std['MSE']:.4f}  MAE={m_va_std['MAE']:.4f}  R2={m_va_std['R2']:.4f}")
                fold_id += 1

            mse_ratio = [v / t if t != 0 else np.inf for v, t in zip(mse_va, mse_tr)]
            r2_diff = [t - v for v, t in zip(r2_va, r2_tr)]

            cv_metrics[mtype] = {
                "MSE": float(np.mean(mse_va)),
                "MAE": float(np.mean(mae_va)),
                "R2": float(np.mean(r2_va)),
                "folds": {
                    "MSE_train": mse_tr,
                    "MSE_val": mse_va,
                    "MAE_train": mae_tr,
                    "MAE_val": mae_va,
                    "R2_train": r2_tr,
                    "R2_val": r2_va,
                    "MSE_ratio": mse_ratio,
                    "R2_diff": r2_diff,
                },
            }
            print(f"[INFO] Finished 5-fold CV for {mtype}")

        joblib.dump(cv_metrics, cv_metrics_path)
        print(f"[INFO] 5-fold CV metrics (detail) saved => {cv_metrics_path}")

    metrics_rows: list[dict[tuple[str, str, str] | str, float | str]] = []
    for mtype in model_types:
        val_losses: list[float] | None = None
        print(f"\n=== Train model: {mtype} ===")
        X_m = X_by_model.get(mtype, X_base)
        X_train, X_val, Y_train, Y_val = split_data(
            X_m, Y, test_size=config["data"]["test_size"], random_state=random_seed
        )
        scale_cols_idx_m = _resolve_scale_cols_idx(config, mtype, numeric_cols_idx, X_base.shape[1])
        (X_train_s, X_val_s, sx), (Y_train_s, Y_val_s, sy) = standardize_data(
            X_train, X_val, Y_train, Y_val,
            do_input=config["preprocessing"]["standardize_input"],
            do_output=config["preprocessing"]["standardize_output"],
            numeric_cols_idx=numeric_cols_idx,
            scale_cols_idx=scale_cols_idx_m,
            do_output_bounded=(bounded_indices is not None) or config["preprocessing"].get("bounded_output", False),
            bounded_output_cols_idx=bounded_indices
        )
        outdir_m = os.path.join(base_outdir, mtype)
        ensure_dir(outdir_m)
        if mtype == "ANN":
            model, ann_cfg = cast(
                tuple[ANNRegression, dict],
                create_model_by_type("ANN", config, random_seed, input_dim=X_train_s.shape[1])
            )
            if "epochs" not in ann_cfg:
                ann_cfg["epochs"] = config["model"].get("ann_params",{}).get("epochs", 6000)
            loss_fn = get_torch_loss_fn(config["loss"]["type"])
            train_ds = MyDataset(X_train_s, Y_train_s)
            val_ds = MyDataset(X_val_s, Y_val_s)
            if os.path.exists(ann_cfg["checkpoint_path"]):
                print(f"[INFO] Found checkpoint for {mtype}, loading weights from {ann_cfg['checkpoint_path']}")
                model.load_state_dict(torch.load(ann_cfg["checkpoint_path"]))
            model, train_losses, val_losses = train_torch_model_dataloader(
                model, train_ds, val_ds,
                loss_fn=loss_fn,
                epochs=ann_cfg["epochs"],
                batch_size=ann_cfg["batch_size"],
                lr=float(ann_cfg["learning_rate"]),
                weight_decay=float(ann_cfg.get("weight_decay", 0.0)),
                checkpoint_path=ann_cfg["checkpoint_path"],
                log_interval=config["training"]["log_interval"],
                early_stopping=ann_cfg.get("early_stopping", True),
                patience=ann_cfg.get("patience", 5),
                optimizer_name=ann_cfg.get("optimizer", "AdamW")
            )
            model.eval()
            model.to("cpu")
            np.save(os.path.join(outdir_m, "train_losses.npy"), train_losses)
            np.save(os.path.join(outdir_m, "val_losses.npy"), val_losses)
        else:
            model = create_model_by_type(mtype, config, random_seed, input_dim=X_train_s.shape[1])
            early = mtype in ["CatBoost", "XGB"]
            es_rounds = config["model"][f"{mtype.lower()}_params"].get("early_stopping_rounds", 50)
            model = train_sklearn_model(
                model, X_train_s, Y_train_s,
                X_val=X_val_s, Y_val=Y_val_s,
                enable_early_stop=early, es_rounds=es_rounds
            )
        if hasattr(model, 'eval') and hasattr(model, 'forward'):
            with torch.no_grad():
                train_pred_std = model(torch.tensor(X_train_s, dtype=torch.float32)).cpu().numpy()
                val_pred_std = model(torch.tensor(X_val_s, dtype=torch.float32)).cpu().numpy()
        else:
            train_pred_std = model.predict(X_train_s)
            val_pred_std = model.predict(X_val_s)
        train_pred_std = _to_2d(train_pred_std)
        val_pred_std = _to_2d(val_pred_std)
        # ------------------------------------------------------------------------

        train_pred_raw = inverse_transform_output(train_pred_std, sy) \
            if config["preprocessing"]["standardize_output"] else train_pred_std
        val_pred_raw = inverse_transform_output(val_pred_std, sy) \
            if config["preprocessing"]["standardize_output"] else val_pred_std
        train_pred_raw = _to_2d(train_pred_raw)
        val_pred_raw = _to_2d(val_pred_raw)
        # ---------------------------------------------------
        Y_train_s = _to_2d(Y_train_s)
        Y_val_s = _to_2d(Y_val_s)
        Y_train = _to_2d(Y_train)
        Y_val = _to_2d(Y_val)
        # -------------------------------------------------------------

        std_tr = compute_regression_metrics(Y_train_s, train_pred_std)
        std_va = compute_regression_metrics(Y_val_s, val_pred_std)

        raw_tr = compute_regression_metrics(Y_train, train_pred_raw)
        raw_va = compute_regression_metrics(Y_val, val_pred_raw)

        mix_tr = {"MSE": std_tr["MSE"], "MAE": std_tr["MAE"], "R2": raw_tr["R2"]}
        mix_va = {"MSE": std_va["MSE"], "MAE": std_va["MAE"], "R2": raw_va["R2"]}

        print(f"   => train MIX={mix_tr},  valid MIX={mix_va}")

        np.save(os.path.join(outdir_m, "train_pred_std.npy"), train_pred_std)
        np.save(os.path.join(outdir_m, "val_pred_std.npy"), val_pred_std)
        np.save(os.path.join(outdir_m, "train_pred_raw.npy"), train_pred_raw)
        np.save(os.path.join(outdir_m, "val_pred_raw.npy"), val_pred_raw)

        joblib.dump(
            {
                "mixed": {"train": mix_tr, "val": mix_va},
                "std": {"train": std_tr, "val": std_va},
                "raw": {"train": raw_tr, "val": raw_va}
            },
            os.path.join(outdir_m, "metrics.pkl")
        )

        out_names = y_col_names or [f"output{i}" for i in range(train_pred_raw.shape[1])]
        row_dict: dict[tuple[str, str, str] | str, float | str] = {"model": mtype}

        for d, name in enumerate(out_names):
            r2_tr = r2_score(Y_train[:, d], train_pred_raw[:, d])
            r2_va = r2_score(Y_val[:, d], val_pred_raw[:, d])

            mse_tr = mean_squared_error(Y_train_s[:, d], train_pred_std[:, d])
            mse_va = mean_squared_error(Y_val_s[:, d], val_pred_std[:, d])

            mae_tr = mean_absolute_error(Y_train_s[:, d], train_pred_std[:, d])
            mae_va = mean_absolute_error(Y_val_s[:, d], val_pred_std[:, d])

            row_dict[("train", name, "r2")] = r2_tr
            row_dict[("train", name, "mse")] = mse_tr
            row_dict[("train", name, "mae")] = mae_tr
            row_dict[("valid", name, "r2")] = r2_va
            row_dict[("valid", name, "mse")] = mse_va
            row_dict[("valid", name, "mae")] = mae_va
        # ---------------------------------------------------------------------
        metrics_rows.append(row_dict)

        model_dir = get_model_dir(csv_name, mtype, run_id=run_id)
        ensure_dir(model_dir)

        if mtype == "ANN":
            assert val_losses is not None
            torch.save(model.state_dict(),
                       os.path.join(model_dir, "best_ann.pt"))
            ann_meta = {
                "input_dim": X_train.shape[1],
                "best_val_loss": float(min(val_losses)),
                "epoch": int(np.argmin(val_losses))
            }
            with open(os.path.join(model_dir, "ann_meta.json"), "w") as f:
                json.dump(ann_meta, f, indent=2)
        else:
            joblib.dump(model,
                        os.path.join(model_dir, "trained_model.pkl"))

        save_scaler(sx, os.path.join(model_dir, f"scaler_x_{mtype}.pkl"))
        save_scaler(sy, os.path.join(model_dir, f"scaler_y_{mtype}.pkl"))
        np.save(os.path.join(model_dir, f"scale_cols_idx_{mtype}.npy"),
                np.array(scale_cols_idx_m, dtype=int))
        stats_m = stats_dict_by_model.get(mtype, stats_dict)
        joblib.dump(stats_m, os.path.join(model_dir, "metadata.pkl"))

        np.save(os.path.join(model_dir, "x_col_names.npy"),
                np.array(x_col_names, dtype=object))
        np.save(os.path.join(model_dir, "y_col_names.npy"),
                np.array(y_col_names, dtype=object))

        #shap
        if config["evaluation"].get("save_shap", False):
            shap_dir = get_eval_dir(csv_name, run_id, "model_comparison", mtype, "shap")
            ensure_dir(shap_dir)
            X_full_model = np.concatenate([X_train_s, X_val_s], axis=0)
            try:
                if mtype == "ANN":
                    model.eval()
                    background = torch.tensor(X_train_s[:100], dtype=torch.float32)
                    explainer = shap.DeepExplainer(model, background)
                    shap_values = explainer.shap_values(torch.tensor(X_full_model, dtype=torch.float32))
                elif mtype == "CatBoost":
                    shap_values = model.get_shap_values(X_full_model)
                elif mtype in ["RF", "DT", "XGB"]:
                    base_model = model.model if hasattr(model, "model") else model
                    try:
                        explainer = shap.TreeExplainer(base_model)
                        shap_values = explainer.shap_values(X_full_model)
                    except Exception:
                        shap_values = None
                        if mtype == "XGB" and hasattr(base_model, "get_booster"):
                            try:
                                explainer = shap.TreeExplainer(base_model.get_booster())
                                shap_values = explainer.shap_values(X_full_model)
                            except Exception:
                                shap_values = None
                        if shap_values is None:
                            bg = shap.sample(X_train_s, 50, random_state=random_seed)
                            X_explain = shap.sample(X_full_model, 200, random_state=random_seed)
                            explainer = shap.KernelExplainer(base_model.predict, bg)
                            shap_values = explainer.shap_values(X_explain, nsamples=100)
                            X_full_model = X_explain
                elif mtype == "SVM":
                    bg = shap.sample(X_train_s, 50, random_state=random_seed)
                    X_explain = shap.sample(X_full_model, 200, random_state=random_seed)
                    base_model = model.model if hasattr(model, "model") else model
                    explainer = shap.KernelExplainer(base_model.predict, bg)
                    shap_values = explainer.shap_values(X_explain, nsamples=100)
                    X_full_model = X_explain
                else:
                    print(f"[WARN] SHAP not supported for {mtype} => skip.")
                    shap_values = None
                if shap_values is None:
                    raise RuntimeError("SHAP values not available.")
                loader_cfg_m = stats_m.get("loader_config", {})
                log_cols_m = list(loader_cfg_m.get("log_transform_cols", []) or [])
                log_eps_m = loader_cfg_m.get("log_transform_eps", log_transform_eps)
                X_full_display = _restore_display_features_from_model_input(
                    X_full_model,
                    scaler_x=sx,
                    scale_cols_idx=scale_cols_idx_m,
                    x_col_names=x_col_names,
                    log_transform_cols=log_cols_m,
                    log_transform_eps=log_eps_m,
                )
                shap_save = {
                    "shap_values": shap_values,
                    "X_full": X_full_display,
                    "X_full_model": X_full_model,
                    "x_col_names": x_col_names,
                    "y_col_names": y_col_names,
                    "log_transform_cols": log_cols_m,
                    "log_transform_eps": log_eps_m
                }
                shap_save_path = os.path.join(shap_dir, "shap_data.pkl")
                joblib.dump(shap_save, shap_save_path)
                print(f"[INFO] SHAP data saved for model {mtype} => {shap_save_path}")
            except Exception as e:
                print(f"[WARN] SHAP computation failed for {mtype}: {e}")
        print("\n[INFO] train_main => done.")
    # ------------------------------------------------------------------
    if metrics_rows:
        df = pd.DataFrame(metrics_rows)

        def _safe_name(text: str) -> str:
            text = str(text).strip()
            text = re.sub(r"[^\w]", "_", text)
            return text.strip("_")

        flat_cols = []
        for col in df.columns:
            if isinstance(col, tuple):
                flat_cols.append("_".join(_safe_name(c) for c in col if c))
            else:
                flat_cols.append(_safe_name(col))
        df.columns = flat_cols

        dest_dir = get_eval_dir(csv_name, run_id)
        ensure_dir(dest_dir)
        excel_path = os.path.join(dest_dir, "metrics_summary.xlsx")

        with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="metrics")

            worksheet = writer.sheets["metrics"]
            for i in range(len(df.columns)):
                worksheet.set_column(i, i, 12 if i == 0 else 18)

        print(f"[INFO] Excel metrics summary saved => {excel_path}")
    # ------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models with YAML config.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config. Overrides CONFIG_FILE/CONFIG_PATH.",
    )
    args = parser.parse_args()
    train_main(args.config)
