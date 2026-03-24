"""
visualization.py

1) Read from postprocessing/<csv_name>[/<run_id>]/train:
   - df_raw_14.csv => standard correlation + data analysis
   - X_features.npy => correlation_heatmap_one_hot
   - Y_train.npy, Y_val.npy => scatter/residual plots
   - for each model => read train_pred.npy, val_pred.npy, metrics.pkl, train_losses.npy, val_losses.npy
     => draw scatter, residual, MAE, MSE, loss curve, and feature importance (if available)
2) Read from postprocessing/<csv_name>[/<run_id>]/inference/<model_type>:
   - heatmap_pred.npy, grid_x.npy, grid_y.npy => 2D heatmap
   - confusion_pred_norm.npy => confusion-like plots
3) Save figures to ./evaluation/figures/<csv_name>[/<run_id>]/...
4) K-Fold logic has been removed.
"""

import os
import re
import argparse
import yaml
import numpy as np
import pandas as pd
import joblib
import optuna

from utils import (
    ensure_dir,
    get_root_model_dir,
    get_model_dir,
    get_postprocess_dir,
    get_eval_dir,
    get_run_id,
    # mic correlation
    plot_mic_network_heatmap,
    safe_filename,
    # data analysis
    plot_kde_distribution,
    # model metrics
    plot_cv_metrics,
    plot_cv_boxplot,
    plot_overfitting_horizontal,
    plot_loss_curve,
    plot_joint_scatter_with_marginals,
    merge_onehot_shap,
    plot_shap_combined,
    plot_shap_importance_multi_output,
    plot_local_shap_force,
    plot_local_shap_lines,
    plot_shap_heatmap_local,
    plot_multi_model_residual_distribution_single_dim,
    # inference
    plot_2d_heatmap_from_npy,
    plot_confusion_from_npy,
    plot_3d_surface_from_heatmap,
    plot_3d_bars_from_confusion,
    plot_3d_surface_from_3d_heatmap,
    # optuna
    plot_optuna_tuning_curve,
    plot_optuna_summary_curve,
    plot_optuna_slice,
    plot_optuna_param_importances
)


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


def _find_group_idx(keyword, groups, colnames):
    kw = keyword.lower()
    for idx, grp in enumerate(groups):
        if any(kw in colnames[c].lower() for c in grp):
            return idx
    return None


def _find_group_idx_by_name(keyword, group_names):
    kw = keyword.lower()
    for idx, name in enumerate(group_names):
        if kw in str(name).lower():
            return idx
    return None


def _find_latest_run_id(csv_name):
    def _scan(base, marker=None):
        if not os.path.isdir(base):
            return []
        candidates = []
        for name in os.listdir(base):
            path = os.path.join(base, name)
            if not os.path.isdir(path):
                continue
            if not re.match(r"^\d{8}_\d{6}(?:_.*)?$", name):
                continue
            if marker and not os.path.exists(os.path.join(path, marker)):
                continue
            candidates.append(name)
        return candidates

    candidates = _scan(os.path.join("models", csv_name), "metadata.pkl")
    if not candidates:
        candidates = _scan(os.path.join("evaluation", "figures", csv_name))
    return max(candidates) if candidates else None


def _resolve_viz_run_id(csv_name, config=None):
    rid = get_run_id(config)
    if rid:
        return rid
    rid = _find_latest_run_id(csv_name)
    if rid:
        print(f"[INFO] RUN_ID not set; using latest run_id => {rid}")
    return rid


def _normalize_shap_values(shap_data):
    """
    Normalize SHAP values to list of 2D arrays: [(n_samples, n_features), ...].
    Supports 2D array, 3D array, or list of 2D arrays.
    """
    sv = shap_data.get("shap_values", None)
    if sv is None:
        return shap_data
    if isinstance(sv, list):
        return shap_data
    if not isinstance(sv, np.ndarray):
        return shap_data
    if sv.ndim == 2:
        shap_data["shap_values"] = [sv]
        return shap_data
    if sv.ndim != 3:
        return shap_data

    n_samples = shap_data.get("X_full", None)
    n_samples = n_samples.shape[0] if isinstance(n_samples, np.ndarray) else None
    n_features = len(shap_data.get("x_col_names", []))
    a, b, c = sv.shape

    if c == n_features:
        if n_samples is not None and a == n_samples:
            shap_data["shap_values"] = [sv[:, i, :] for i in range(b)]
            return shap_data
        if n_samples is not None and b == n_samples:
            shap_data["shap_values"] = [sv[i, :, :] for i in range(a)]
            return shap_data

    if b == n_features:
        if n_samples is not None and a == n_samples:
            shap_data["shap_values"] = [sv[:, :, i] for i in range(c)]
            return shap_data
        if n_samples is not None and c == n_samples:
            shap_data["shap_values"] = [sv[i, :, :].T for i in range(a)]
            return shap_data

    # Fallback: assume outputs on axis=1
    if n_samples is not None and a == n_samples:
        shap_data["shap_values"] = [sv[:, i, :] for i in range(b)]
    else:
        shap_data["shap_values"] = [sv[i, :, :] for i in range(a)]
    return shap_data


def generate_shap_plots(csv_name, model_types, top_n, config=None):
    """
    Generate SHAP figures from shap_data.pkl, with one-hot merge logic included.
    """
    run_id = _resolve_viz_run_id(csv_name, config)
    # ==== Load metadata once ====
    meta_path = os.path.join(get_root_model_dir(csv_name, run_id=run_id), "metadata.pkl")
    meta_data = joblib.load(meta_path)
    onehot_groups = meta_data["onehot_groups"]
    orig_case_map = {c.lower(): c for c in meta_data["x_col_names"]}

    for mtype in model_types:
        shap_dir = get_eval_dir(csv_name, run_id, "model_comparison", mtype, "shap")
        ensure_dir(shap_dir)
        local_dir = os.path.join(shap_dir, "local"); ensure_dir(local_dir)
        shap_data_path = os.path.join(shap_dir, "shap_data.pkl")
        if not os.path.exists(shap_data_path):
            print(f"[WARN] shap_data not found => {shap_data_path}")
            continue

        # ----- 1) Load and merge one-hot features -----
        raw_shap_data = joblib.load(shap_data_path)
        raw_shap_data = _normalize_shap_values(raw_shap_data)
        shap_data = merge_onehot_shap(raw_shap_data,
                                      onehot_groups=onehot_groups,
                                      case_map=orig_case_map,
                                      aggregate_interactions_to_original=True)   # Pass None if case restoration is not needed.

        # ----- 2) Global plots -----
        try:
            plot_shap_combined(shap_data, shap_dir,
                               top_n_features=top_n, plot_width=12, plot_height=8)
            print(f"[INFO] SHAP combined plotted for {mtype}")
        except Exception as e:
            print(f"[WARN] combined plot failed ({mtype}): {e}")

        try:
            out_jpg = os.path.join(shap_dir, "multi_output_shap_importance.jpg")
            plot_shap_importance_multi_output(shap_data, output_path=out_jpg,
                                              top_n_features=top_n, plot_width=12,
                                              plot_height=8)
            print(f"[INFO] stacked bar plotted for {mtype}")
        except Exception as e:
            print(f"[WARN] stacked bar failed ({mtype}): {e}")

        # ----- 3) Local plots -----
        outputs = (shap_data["shap_values"]
                   if isinstance(shap_data["shap_values"], list)
                   else [shap_data["shap_values"]])
        y_names = shap_data.get("y_col_names",
                                [f"Output{i}" for i in range(len(outputs))])

        sample_indices = list(range(5))
        for i, sv in enumerate(outputs):
            local_sd = shap_data.copy()
            local_sd["shap_values"] = sv
            local_sd["y_col_names"] = [y_names[i]]

            out_sub = local_dir if len(outputs) == 1 else \
                      os.path.join(local_dir, safe_filename(y_names[i]))
            ensure_dir(out_sub)

            # force
            for idx in sample_indices:
                fn = os.path.join(out_sub, f"local_force_{idx}.jpg")
                try:
                    plot_local_shap_force(local_sd, idx, fn, top_n_features=top_n-6,
                                          outputID=i)
                except Exception as e:
                    print(f"[WARN] force fail ({mtype}-{i}-{idx}): {e}")

            # decision line
            fn = os.path.join(out_sub, "local_line.jpg")
            try:
                plot_local_shap_lines(local_sd, sample_indices, fn,
                                      top_n_features=top_n-6, outputID=i)
            except Exception as e:
                print(f"[WARN] line fail ({mtype}-{i}): {e}")

            # heatmap
            fn = os.path.join(out_sub, "local_heatmap.jpg")
            try:
                plot_shap_heatmap_local(local_sd, fn, sample_count=100,
                                        max_display=14, outputID=i)
            except Exception as e:
                print(f"[WARN] heatmap fail ({mtype}-{i}): {e}")



def load_optuna_trials_df(data_name, mtype, run_id=None):
    """
    Load study.pkl for a given dataset name and model type, and return a preprocessed trials DataFrame.
    Remove the prefix from all column names that start with "params_".

    Parameters:
      data_name: dataset name, matching the filename without extension
      mtype: model type, matching an item in config["optuna"]["models"]

    Returns:
      The preprocessed DataFrame if the study file exists, otherwise None.
    """
    study_path = get_postprocess_dir(data_name, run_id, "optuna", mtype, "study.pkl")
    if os.path.exists(study_path):
        try:
            study_obj = joblib.load(study_path)
        except Exception as e:
            print(f"[WARN] Failed to load optuna study for {mtype}: {e}")
            return None
        trials_df = study_obj.trials_dataframe()
        trials_df.rename(
            columns=lambda x: x.replace("params_", "") if x.startswith("params_") else x,
            inplace=True
        )
        return trials_df
    else:
        print(f"[WARN] No study file for {mtype}")
        return None


def plot_optuna_results(config):
    """
    For each model in config["optuna"]["models"], load the tuning study object from
    postprocessing/<csv_name>[/<run_id>]/optuna/<model>/study.pkl,
    generate Optuna figures with the helper functions, save them under
    evaluation/figures/<csv_name>[/<run_id>]/optuna/<model>/,
    and also generate a summary plot.
    """
    import os, joblib
    csv_name = os.path.splitext(os.path.basename(config["data"]["path"]))[0]
    run_id = _resolve_viz_run_id(csv_name, config)
    dest_optuna = get_eval_dir(csv_name, run_id, "optuna")
    ensure_dir(dest_optuna)

    # Keep a trials_df dictionary for per-model plots and the final summary plot.
    trials_dict = {}

    for mtype in config["optuna"]["models"]:
        trials_df = load_optuna_trials_df(csv_name, mtype, run_id=run_id)
        if trials_df is not None:
            trials_dict[mtype] = trials_df
            model_optuna_dir = os.path.join(dest_optuna, mtype)
            ensure_dir(model_optuna_dir)

            # Optimization history.
            out_history = os.path.join(model_optuna_dir, "optimization_history.jpg")
            plot_optuna_tuning_curve(trials_df, out_history)

            # Get slice parameters and generate the slice plot.
            model_slice_params = config["optuna"].get("slice_params", {}).get(mtype, [])
            if model_slice_params:
                out_slice = os.path.join(model_optuna_dir, "slice.jpg")
                plot_optuna_slice(trials_df, model_slice_params, out_slice)

            # Parameter importance plot.
            out_importance = os.path.join(model_optuna_dir, "param_importances.jpg")
            plot_optuna_param_importances(trials_df, out_importance)

    if not trials_dict:
        print("[WARN] No optuna trials found => skip optuna plots.")
        return

    # Generate the summary plot from preloaded trials data.
    out_summary = os.path.join(dest_optuna, "summary.jpg")
    plot_optuna_summary_curve(trials_dict, out_summary)


def visualize_main(config_path: str | None = None):
    config_path = _resolve_config_path(config_path)
    print(f"[INFO] Using config => {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    csv_path = config["data"]["path"]
    if not os.path.isabs(csv_path):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(config_path), ".."))
        csv_path = os.path.join(repo_root, csv_path)
        config["data"]["path"] = csv_path
    csv_name = os.path.splitext(os.path.basename(csv_path))[0]
    run_id = _resolve_viz_run_id(csv_name, config)

    # Generate Optuna-related plots if enabled in the config.
    if config["evaluation"].get("save_optuna", False):
        plot_optuna_results(config)

    base_train = get_postprocess_dir(csv_name, run_id, "train")
    if not os.path.isdir(base_train):
        print(f"[WARN] train folder not found => {base_train}")
        return

    # ========== 1.1) df_raw_14.csv & data_corr_dir ==========
    raw_csv_path = os.path.join(base_train, "df_raw_14.csv")
    data_corr_dir = get_eval_dir(csv_name, run_id, "DataCorrelation")
    ensure_dir(data_corr_dir)

    if os.path.exists(raw_csv_path):
        df_raw_14 = pd.read_csv(raw_csv_path)

        if config["evaluation"].get("save_correlation", False):
            fn1 = os.path.join(data_corr_dir, "correlation_heatmap.jpg")
            plot_mic_network_heatmap(
                df_raw_14,
                filename=fn1,
                method="mic",
                dpi=700
            )

        # Data analysis plots.
        if config["evaluation"].get("save_data_analysis_plots", False):
            df_plot = df_raw_14.copy()
            # Plot-only transform: convert Calcination time (h) to LN scale without changing raw data.
            if ("Calcination time (h) (LN scale)" not in df_plot.columns
                    and "Calcination time (h)" in df_plot.columns):
                log_eps = 1e-8
                ct = pd.to_numeric(df_plot["Calcination time (h)"], errors="coerce")
                ct = ct.clip(lower=log_eps)
                df_plot["Calcination time (h) (LN scale)"] = np.log(ct)
            # Plot-only transform: convert Molar ratio (Zn:Cu) to LN scale without changing raw data.
            if ("Molar ratio (Zn:Cu) (LN scale)" not in df_plot.columns
                    and "Molar ratio (Zn:Cu)" in df_plot.columns):
                log_eps = 1e-8
                mr = pd.to_numeric(df_plot["Molar ratio (Zn:Cu)"], errors="coerce")
                mr = mr.clip(lower=log_eps)
                df_plot["Molar ratio (Zn:Cu) (LN scale)"] = np.log(mr)
            possible_cols = [
                "Molar ratio (Zn:Cu) (LN scale)",
                "Promoter 1 ratio (Promoter 1:Cu)",
                "Promoter 2 ratio (Promoter 2:Cu)",
                "Catalyst surface area (m2/g) (LN scale)",
                "Calcination temperature (°C)",
                "Calcination time (h) (LN scale)",
                "Temperature (°C)",
                "Pressure (bar)",
                "H2/CO2 ratio (-)",
                "GHSV (mL/g.h) (LN scale)",
                "GHSV (mL/g.h)",
                "Catalyst loading (g)",
                "CO selectivity (%)",
                "Methanol selectivity (%)",
                "STY_CH3OH (g/kg·h) (LN scale)",
                "CO2 conversion efficiency (%)",
            ]
            existing_cols = [c for c in possible_cols if c in df_plot.columns]
            if existing_cols:
                out_kde = os.path.join(data_corr_dir, "kde_distribution.jpg")
                plot_kde_distribution(df_plot, existing_cols, filename=out_kde)
    else:
        print(f"[WARN] df_raw_14.csv not found => {raw_csv_path}")

    # ========== Extra: draw K-Fold CV metric comparison plots ==========
    cv_metrics_path = os.path.join(base_train, "cv_metrics.pkl")
    if os.path.exists(cv_metrics_path):
        cv_metrics = joblib.load(cv_metrics_path)

        # Original 4-panel plot.
        out_cv = os.path.join(data_corr_dir, "cv_metrics_comparison.jpg")
        plot_cv_metrics(cv_metrics, save_name=out_cv, show_label=False)

        # Added box-style CV plots for validation performance and overfitting.
        out_box_mse = os.path.join(data_corr_dir, "cv_box_MSE.jpg")
        plot_cv_boxplot(cv_metrics, metric="MSE", save_name=out_box_mse)
        out_box_r2 = os.path.join(data_corr_dir, "cv_box_R2.jpg")
        plot_cv_boxplot(cv_metrics, metric="R2", save_name=out_box_r2)
        print(f"[INFO] CV box plots saved => {out_box_mse}, {out_box_r2}")

    # ========== 1.3) Y_train.npy, Y_val.npy ==========
    y_train_path = os.path.join(base_train, "Y_train.npy")
    y_val_path = os.path.join(base_train, "Y_val.npy")
    Y_train = np.load(y_train_path) if os.path.exists(y_train_path) else None
    Y_val = np.load(y_val_path) if os.path.exists(y_val_path) else None

    # ========== 1.4) Per-model plots ==========
    model_types = config["model"]["types"]
    for mtype in model_types:
        model_subdir = os.path.join(base_train, mtype)
        if not os.path.isdir(model_subdir):
            print(f"[WARN] no train folder for model type => {model_subdir}")
            continue

        metrics_pkl = os.path.join(model_subdir, "metrics.pkl")
        train_pred_path_raw = os.path.join(model_subdir, "train_pred_raw.npy")
        val_pred_path_raw = os.path.join(model_subdir, "val_pred_raw.npy")
        train_pred_path_std = os.path.join(model_subdir, "train_pred_std.npy")
        val_pred_path_std = os.path.join(model_subdir, "val_pred_std.npy")
        train_loss_path = os.path.join(model_subdir, "train_losses.npy")
        val_loss_path = os.path.join(model_subdir, "val_losses.npy")

        train_metrics = None
        val_metrics = None
        if os.path.exists(metrics_pkl):
            data_ = joblib.load(metrics_pkl)
            train_metrics = data_.get("mixed", {}).get("train", None)
            val_metrics = data_.get("mixed", {}).get("val", None)
            print(f"[{mtype}] train_metrics={train_metrics}, val_metrics={val_metrics}")

        # ---- Load predictions in inverse-transformed raw scale ----
        train_pred = np.load(train_pred_path_raw) if os.path.exists(train_pred_path_raw) else None
        val_pred = np.load(val_pred_path_raw) if os.path.exists(val_pred_path_raw) else None

        train_losses = np.load(train_loss_path) if os.path.exists(train_loss_path) else None
        val_losses = np.load(val_loss_path) if os.path.exists(val_loss_path) else None

        # Load y_col_names if available.
        model_dir = get_model_dir(csv_name, mtype, run_id=run_id)
        ycol_path = os.path.join(model_dir, "y_col_names.npy")
        if os.path.exists(ycol_path):
            y_cols = list(np.load(ycol_path, allow_pickle=True))
        else:
            y_cols = None

        model_comp_dir = get_eval_dir(csv_name, run_id, "model_comparison", mtype)
        ensure_dir(model_comp_dir)

        # (a) Draw the loss curve.
        if train_losses is not None and val_losses is not None and config["evaluation"].get("save_loss_curve", False):
            out_lc = os.path.join(model_comp_dir, f"{mtype}_loss_curve.jpg")
            plot_loss_curve(train_losses, val_losses, filename=out_lc)
        # (b) Draw scatter and residual-related plots.
        if (Y_train is not None and Y_val is not None) and (train_pred is not None and val_pred is not None):
            if config["evaluation"].get("save_scatter_with_marginals_plot", False):
                out_mae_tr = os.path.join(model_comp_dir, "full", "train", f"{mtype}_scatter_with_marginals_train.jpg")
                ensure_dir(os.path.dirname(out_mae_tr))
                plot_joint_scatter_with_marginals(Y_train, train_pred, y_labels=y_cols, filename=out_mae_tr)
                out_mae_val = os.path.join(model_comp_dir, "full", "valid", f"{mtype}_scatter_with_marginals_valid.jpg")
                ensure_dir(os.path.dirname(out_mae_val))
                plot_joint_scatter_with_marginals(Y_val, val_pred, y_labels=y_cols, filename=out_mae_val)
        # ========== 2) Collect metrics across models ==========
        # Comparison plots are generated after the loop.
        # cv_metrics.pkl is saved by train.py; plotting is triggered here.
    # (d) Draw multi-model comparison plots.
    train_metrics_dict = {}
    val_metrics_dict = {}
    for mtype in model_types:
        mdir = os.path.join(base_train, mtype)
        mpkl = os.path.join(mdir, "metrics.pkl")
        if os.path.exists(mpkl):
            data_ = joblib.load(mpkl)
            train_metrics_dict[mtype] = data_.get("mixed", {}).get("train", {})
            val_metrics_dict[mtype] = data_.get("mixed", {}).get("val", {})

    if train_metrics_dict or val_metrics_dict:
        if train_metrics_dict:
            out_3train = os.path.join(data_corr_dir, "three_metrics_horizontal_train.jpg")
            plot_cv_metrics(train_metrics_dict, save_name=out_3train,show_label=False)
            print(f"[INFO] train metrics plot saved => {out_3train}")
        if val_metrics_dict:
            out_3val = os.path.join(data_corr_dir, "three_metrics_horizontal_val.jpg")
            plot_cv_metrics(val_metrics_dict, save_name=out_3val,show_label=False)
            print(f"[INFO] valid metrics plot saved => {out_3val}")
        if config["evaluation"].get("save_models_evaluation_bar", False):
            if train_metrics_dict and val_metrics_dict:
                overfit_data = {}
                for m in train_metrics_dict:
                    trm = train_metrics_dict[m]
                    vam = val_metrics_dict[m]
                    ms_ratio = float("inf") if trm["MSE"] == 0 else vam["MSE"] / trm["MSE"]
                    r2_diff = trm["R2"] - vam["R2"]
                    overfit_data[m] = {"MSE_ratio": ms_ratio, "R2_diff": r2_diff}
                out_of = os.path.join(data_corr_dir, "overfitting_single.jpg")
                plot_overfitting_horizontal(overfit_data, save_name=out_of)
        # ========== 3) Multi-model residual visualization ==========
        if config["evaluation"].get("save_multi_model_residual_plot", False):
            if (Y_train is not None) and (Y_val is not None):
                y_cols = None
                for mtype in model_types:
                    model_dir = get_model_dir(csv_name, mtype, run_id=run_id)
                    ycol_path = os.path.join(model_dir, "y_col_names.npy")
                    if os.path.exists(ycol_path):
                        y_cols = list(np.load(ycol_path, allow_pickle=True))
                        break
                if y_cols is None:
                    if Y_train.ndim == 2:
                        out_dim = Y_train.shape[1]
                        y_cols = [f"Output_{i}" for i in range(out_dim)]
                    else:
                        y_cols = ["Output_0"]
                out_dim = len(y_cols)
                train_residuals_dicts = [dict() for _ in range(out_dim)]
                val_residuals_dicts = [dict() for _ in range(out_dim)]
                for mtype in model_types:
                    model_subdir = os.path.join(base_train, mtype)
                    train_pred_path = os.path.join(model_subdir, "train_pred_raw.npy")
                    val_pred_path = os.path.join(model_subdir, "val_pred_raw.npy")
                    if os.path.exists(train_pred_path) and os.path.exists(val_pred_path):
                        train_pred = np.load(train_pred_path)
                        val_pred = np.load(val_pred_path)
                        for d in range(out_dim):
                            train_residuals_dicts[d][mtype] = Y_train[:, d] - train_pred[:, d]
                            val_residuals_dicts[d][mtype] = Y_val[:, d] - val_pred[:, d]
                for d in range(out_dim):
                    col_name = y_cols[d]
                    # out_tr_fig = os.path.join("./evaluation/figures", csv_name, "model_comparison",
                    #                            f"multi_residual_{col_name}_train.jpg")
                    out_tr_fig = get_eval_dir(
                        csv_name,
                        run_id,
                        "model_comparison",
                        f"multi_residual_{safe_filename(col_name)}_train.jpg"
                    )

                    plot_multi_model_residual_distribution_single_dim(
                        residuals_dict=train_residuals_dicts[d],
                        out_label=f"{col_name} (Train)",
                        bins=6,
                        filename=out_tr_fig,
                        rug_negative_space=0.13,
                        show_zero_line_arrow=False
                    )
                    out_val_fig = get_eval_dir(
                        csv_name,
                        run_id,
                        "model_comparison",
                        f"multi_residual_{safe_filename(col_name)}_valid.jpg"
                    )

                    plot_multi_model_residual_distribution_single_dim(
                        residuals_dict=val_residuals_dicts[d],
                        out_label=f"{col_name} (Valid)",
                        bins=6,
                        filename=out_val_fig,
                        rug_negative_space=0.13,
                        show_zero_line_arrow=False
                    )
                print("[INFO] => Multi-model residual distribution plots done.")
            else:
                print("[WARN] => Y_train / Y_val not found, skip multi-model residual distribution.")
        # ========== 4) SHAP interpretability plots ==========
        if config["evaluation"].get("save_shap", False):
            inp_len = int(config["data"]["input_len"])
            generate_shap_plots(csv_name, model_types, top_n=inp_len, config=config)

    # ========== 5) Inference visualization (heatmap + confusion) ==========
    base_inf = get_postprocess_dir(csv_name, run_id, "inference")
    inf_models = config["inference"].get("models", [])
    metadata_path = os.path.join(get_root_model_dir(csv_name, run_id=run_id), "metadata.pkl")
    if os.path.exists(metadata_path):
        meta_data = joblib.load(metadata_path)
        stats_dict = meta_data.get("continuous_cols", {})
    else:
        stats_dict = {}
    axes_names = config["inference"].get("heatmap_axes", [])
    heatmap_x_label = axes_names[0] if len(axes_names) >= 1 else "X-axis"
    heatmap_y_label = axes_names[1] if len(axes_names) >= 2 else "Y-axis"
    heatmap_z_label = axes_names[2] if len(axes_names) >= 3 else "Z-axis"

    conf_default = config["inference"]["confusion_axes"]
    conf_by_model = config["inference"].get("confusion_axes_by_model", {})
    # Re-index 2D combinations from 0 for each model.

    for mtype in inf_models:
        combo_id = -1
        inf_dir = os.path.join(base_inf, mtype)
        if not os.path.isdir(inf_dir):
            print(f"[WARN] no inference dir => {inf_dir}")
            continue
        # These three lines are the key fix.
        base_out = get_eval_dir(csv_name, run_id, "inference", mtype)
        ensure_dir(base_out)
        confusion_path = os.path.join(inf_dir, "confusion_pred_norm.npy")
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # --------------------------------------------------------------
        # A) Process all 2D files matching heatmap_pred_*.
        # --------------------------------------------------------------
        file_list = os.listdir(inf_dir)
        for fname in file_list:
            if not (fname.startswith("heatmap_pred_") and fname.endswith(".npy")
                    and "3d" not in fname):
                continue

            # ---------- Load the grid ----------
            tag = fname[len("heatmap_pred_"):-4]  # May be "Temp__GHSV" or some irregular tag.
            gx = os.path.join(inf_dir, f"grid_x_{tag}.npy")
            gy = os.path.join(inf_dir, f"grid_y_{tag}.npy")
            if not (os.path.exists(gx) and os.path.exists(gy)):
                continue

            heatmap_pred = np.load(os.path.join(inf_dir, fname))
            grid_x, grid_y = np.load(gx), np.load(gy)

            # ---------- Create the combination directory ----------
            combo_id += 1
            folder_name = f"comb{combo_id}"
            tag_dir = os.path.join(base_out, "2d_heatmap", folder_name)
            map_dir = os.path.join(tag_dir, "map")
            surf_dir = os.path.join(tag_dir, "surface")
            ensure_dir(map_dir);
            ensure_dir(surf_dir)

            # ---------- Parse axis labels ----------
            parts = tag.split("__")
            if len(parts) == 2 and all(parts):
                xlab_raw, ylab_raw = parts
            else:  # Fall back to config defaults if parsing fails.
                xlab_raw, ylab_raw = heatmap_x_label, heatmap_y_label
            xlab = xlab_raw.replace("_", " ")
            ylab = ylab_raw.replace("_", " ")

            # Write the actual axis info to info.txt for traceability.
            info_txt = os.path.join(tag_dir, "info.txt")
            if not os.path.exists(info_txt):
                with open(info_txt, "w", encoding="utf8") as f:
                    f.write(f"X-axis : {xlab_raw}\nY-axis : {ylab_raw}\n")

            # ---------- Output variable names ----------
            y_cols = None
            yp = os.path.join(get_model_dir(csv_name, mtype, run_id=run_id), "y_col_names.npy")
            if os.path.exists(yp):
                y_cols = list(np.load(yp, allow_pickle=True))

            # ---------- (a) 2D heatmap ----------
            plot_2d_heatmap_from_npy(
                grid_x, grid_y, heatmap_pred,
                out_dir=map_dir,
                x_label=xlab, y_label=ylab,
                y_col_names=y_cols,
                colorbar_extend_ratio=0.02)

            # ---------- (b) Local 3D surface ----------
            plot_3d_surface_from_heatmap(
                grid_x, grid_y, heatmap_pred,
                out_dir=surf_dir,
                x_label=xlab, y_label=ylab,
                y_col_names=y_cols,
                colorbar_extend_ratio=0.02,
                cmap_name="GnBu")

        # --------------------------------------------------------------
        # B) Full 3-variable 3D: transparent surface slices
        # --------------------------------------------------------------
        hp3d = os.path.join(inf_dir, "heatmap_pred_3d.npy")
        gx3d = os.path.join(inf_dir, "grid_x_3d.npy")
        gy3d = os.path.join(inf_dir, "grid_y_3d.npy")
        gz3d = os.path.join(inf_dir, "grid_z_3d.npy")

        if all(os.path.exists(p) for p in [hp3d, gx3d, gy3d, gz3d]):
            heatmap_pred_3d = np.load(hp3d)
            grid_x_3d, grid_y_3d, grid_z_3d = (
                np.load(gx3d), np.load(gy3d), np.load(gz3d))

            # Load y_col_names if available.
            y_cols = None
            yp = os.path.join(get_model_dir(csv_name, mtype, run_id=run_id), "y_col_names.npy")
            if os.path.exists(yp):
                y_cols = list(np.load(yp, allow_pickle=True))

            overall_dir = os.path.join(base_out, "3d_surface_overall")
            ensure_dir(overall_dir)

            out_dim = heatmap_pred_3d.shape[-1]
            for d in range(out_dim):
                # --- Transparent slice surfaces ---
                plot_3d_surface_from_3d_heatmap(
                    grid_x_3d, grid_y_3d, grid_z_3d, heatmap_pred_3d,
                    out_dir=overall_dir,
                    axes_labels=(heatmap_x_label, heatmap_y_label, heatmap_z_label),
                    y_col_names=y_cols,
                    out_idx=d,
                    alpha_mode="value",  # Or "inverse"
                    alpha_gamma=2  # Larger gamma emphasizes high values.
                )

        # --------------------------------------------------------------
        # Read confusion_pred_norm.npy and draw confusion-like plots
        # --------------------------------------------------------------
        if os.path.exists(confusion_path):
            confusion_pred = np.load(confusion_path)

            # 1) Read metadata: one-hot groups and x column names.
            meta_path = os.path.join(get_root_model_dir(csv_name, run_id=run_id), "metadata.pkl")
            meta = joblib.load(meta_path)
            oh_groups = meta["onehot_groups"]  # List[List[int]]
            group_names = meta.get("group_names", [])
            group_value_vectors = meta.get("group_value_vectors", {})

            xcol_path = os.path.join(get_model_dir(csv_name, mtype, run_id=run_id), "x_col_names.npy")
            xcols = list(np.load(xcol_path, allow_pickle=True))

            if len(oh_groups) >= 2:  # Keep this check.
                # 2) Determine which groups to use for rows and columns from config keywords.
                conf_m = conf_by_model.get(mtype, {})
                row_kw = conf_m.get("row_name", conf_default["row_name"])
                col_kw = conf_m.get("col_name", conf_default["col_name"])

                row_idx = _find_group_idx_by_name(row_kw, group_names) if group_names else None
                col_idx = _find_group_idx_by_name(col_kw, group_names) if group_names else None

                if (row_idx is None) or (col_idx is None):
                    row_idx = _find_group_idx(row_kw, oh_groups, xcols)
                    col_idx = _find_group_idx(col_kw, oh_groups, xcols)

                if (row_idx is None) or (col_idx is None):
                    print("[WARN] Could not find matching one-hot groups for the keywords; falling back to the first two groups.")
                    grpA, grpB = oh_groups[:2]
                    row_name = group_names[0] if group_names else None
                    col_name = group_names[1] if group_names else None
                elif row_idx == col_idx:
                    raise ValueError("row_name and col_name fall into the same one-hot group; please check the config.")
                else:
                    grpA, grpB = oh_groups[row_idx], oh_groups[col_idx]
                    row_name = group_names[row_idx] if row_idx < len(group_names) else None
                    col_name = group_names[col_idx] if col_idx < len(group_names) else None

                # 3) Build row and column labels.
                row_labels_path = os.path.join(inf_dir, "confusion_row_labels.npy")
                col_labels_path = os.path.join(inf_dir, "confusion_col_labels.npy")

                if os.path.exists(row_labels_path) and os.path.exists(col_labels_path):
                    row_labels = list(np.load(row_labels_path, allow_pickle=True))
                    col_labels = list(np.load(col_labels_path, allow_pickle=True))
                elif row_name in group_value_vectors and col_name in group_value_vectors:
                    row_labels = group_value_vectors[row_name]["values"]
                    col_labels = group_value_vectors[col_name]["values"]
                else:
                    row_labels = [xcols[cid] for cid in grpA]
                    col_labels = [xcols[cid] for cid in grpB]

                # 4) Load output labels if available.
                ycol_path = os.path.join(get_model_dir(csv_name, mtype, run_id=run_id), "y_col_names.npy")
                y_cols = list(np.load(ycol_path, allow_pickle=True)) if os.path.exists(ycol_path) else None

                # 5) Generate the plots.
                out_conf = os.path.join(base_out, "confusion_matrix")
                ensure_dir(out_conf)

                plot_confusion_from_npy(
                    confusion_pred,
                    row_labels, col_labels,
                    out_dir=out_conf,
                    y_col_names=y_cols,
                    cell_scale=0.25,
                    row_axis_name=row_kw,
                    col_axis_name=col_kw
                )

                plot_3d_bars_from_confusion(
                    confusion_pred,
                    row_labels, col_labels,
                    out_dir=out_conf,
                    y_col_names=y_cols,
                    colorbar_extend_ratio=0.02,
                    cmap_name="GnBu"
                )
            else:
                # Fewer than two one-hot groups.
                print("[WARN] Not enough one-hot groups ⇒ skip confusion matrix.")
    print("\n[INFO] visualize_main => done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate figures with YAML config.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config. Overrides CONFIG_FILE/CONFIG_PATH.",
    )
    args = parser.parse_args()
    visualize_main(args.config)
