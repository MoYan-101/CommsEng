#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
inference.py

- Load trained models (`trained_model.pkl` / `best_ann.pt`) and metadata.
- Rebuild model inputs from saved feature statistics and preprocessing artifacts.
- Run weighted 2D/3D inference grids and confusion-style predictions.
- Save outputs such as `heatmap_pred*.npy` and `confusion_pred_norm.npy`.
"""

import argparse
import yaml
import os
import re
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import torch
import joblib
from itertools import product
import json

from data_preprocessing.scaler_utils import load_scaler, inverse_transform_output
from utils import get_model_dir, get_root_model_dir, get_postprocess_dir, get_run_id

from models.model_ann import ANNRegression
from models.model_rf import RFRegression
from models.model_dt import DTRegression
from models.model_catboost import CatBoostRegression
from models.model_xgb import XGBRegression

from itertools import combinations


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


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


def _read_first_env_int(names: list[str]) -> int | None:
    for name in names:
        value = _read_env_int(name)
        if value is not None:
            return value
    return None


def _resolve_inference_parallel_plan(inf_models: list[str]) -> dict[str, int]:
    cpu_total = _read_env_int("CPU_TOTAL")
    if cpu_total is None:
        cpu_total = max(1, int(os.cpu_count() or 1))

    requested_model_workers = _read_env_int("INFER_MODEL_WORKERS") or 1
    model_workers = max(1, min(requested_model_workers, max(1, len(inf_models)), cpu_total))

    requested_grid_workers = _read_env_int("INFER_GRID_WORKERS") or 1
    per_model_budget = max(1, cpu_total // model_workers)
    grid_workers = max(1, min(requested_grid_workers, per_model_budget))

    chunk_rows = _read_env_int("INFER_GRID_CHUNK_ROWS")
    if chunk_rows is None:
        chunk_rows = 8
    chunk_rows = max(1, chunk_rows)

    # Per-predict backend threads to keep total CPU near budget:
    # model_workers * grid_workers * per_predict_threads <= cpu_total
    per_predict_threads = max(1, per_model_budget // grid_workers)

    return {
        "cpu_total": cpu_total,
        "model_workers": model_workers,
        "grid_workers": grid_workers,
        "chunk_rows": chunk_rows,
        "per_model_budget": per_model_budget,
        "per_predict_threads": per_predict_threads,
    }


def _iter_row_chunks(total_rows: int, chunk_rows: int):
    chunk_rows = max(1, int(chunk_rows))
    for start in range(0, int(total_rows), chunk_rows):
        yield start, min(int(total_rows), start + chunk_rows)


def _run_row_chunks(total_rows: int, chunk_rows: int, worker_fn, max_workers: int, task_name: str):
    ranges = list(_iter_row_chunks(total_rows, chunk_rows))
    if max_workers <= 1 or len(ranges) <= 1:
        return [worker_fn(s, e) for s, e in ranges]

    try:
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(worker_fn, s, e) for s, e in ranges]
            for fut in as_completed(futures):
                results.append(fut.result())
        return results
    except Exception as e:
        print(f"[WARN] {task_name} parallel failed ({e}); fallback to sequential.")
        return [worker_fn(s, e) for s, e in ranges]


def _configure_torch_runtime() -> None:
    torch_threads = _read_first_env_int(["INFER_TORCH_NUM_THREADS", "TORCH_NUM_THREADS"])
    if torch_threads is not None:
        try:
            torch.set_num_threads(torch_threads)
        except Exception as e:
            print(f"[WARN] Failed to set torch num_threads={torch_threads}: {e}")

    interop_threads = _read_first_env_int(["INFER_TORCH_NUM_INTEROP_THREADS", "TORCH_NUM_INTEROP_THREADS"])
    if interop_threads is not None:
        try:
            torch.set_num_interop_threads(interop_threads)
        except Exception as e:
            print(f"[WARN] Failed to set torch num_interop_threads={interop_threads}: {e}")

    print(
        "[INFO] Inference torch threads => "
        f"num_threads={torch.get_num_threads()}, "
        f"num_interop_threads={torch.get_num_interop_threads()}"
    )


def _apply_model_runtime_overrides(model, model_type: str, runtime_overrides: dict[str, int] | None = None):
    model_n_jobs = _read_first_env_int(["INFER_MODEL_N_JOBS", "MODEL_N_JOBS"])
    rf_n_jobs = _read_first_env_int(["INFER_RF_N_JOBS", "RF_N_JOBS"])
    xgb_n_jobs = _read_first_env_int(["INFER_XGB_N_JOBS", "XGB_N_JOBS"])
    cat_threads = _read_first_env_int(["INFER_CATBOOST_THREAD_COUNT", "CATBOOST_THREAD_COUNT"])
    svm_n_jobs = _read_first_env_int(["INFER_SVM_N_JOBS", "SVM_N_JOBS"])

    if runtime_overrides:
        model_n_jobs = runtime_overrides.get("model_n_jobs", model_n_jobs)
        rf_n_jobs = runtime_overrides.get("rf_n_jobs", rf_n_jobs)
        xgb_n_jobs = runtime_overrides.get("xgb_n_jobs", xgb_n_jobs)
        cat_threads = runtime_overrides.get("cat_threads", cat_threads)
        svm_n_jobs = runtime_overrides.get("svm_n_jobs", svm_n_jobs)

    changed = None
    try:
        if model_type == "RF":
            n = rf_n_jobs if rf_n_jobs is not None else model_n_jobs
            if n is not None and hasattr(model, "model") and hasattr(model.model, "set_params"):
                model.model.set_params(n_jobs=n)
                changed = f"RF n_jobs={n}"

        elif model_type == "XGB":
            n = xgb_n_jobs if xgb_n_jobs is not None else model_n_jobs
            if n is not None and hasattr(model, "set_params"):
                model.set_params(n_jobs=n)
                changed = f"XGB n_jobs={n}"

        elif model_type == "CatBoost":
            n = cat_threads if cat_threads is not None else model_n_jobs
            if n is not None:
                setattr(model, "_infer_thread_count", int(n))
                changed = f"CatBoost predict(thread_count)={n}"

        elif model_type == "SVM":
            n = svm_n_jobs if svm_n_jobs is not None else model_n_jobs
            if n is not None and hasattr(model, "model") and hasattr(model.model, "set_params"):
                model.model.set_params(n_jobs=n)
                changed = f"SVM n_jobs={n}"
    except Exception as e:
        print(f"[WARN] Failed to apply inference runtime override for {model_type}: {e}")

    if changed:
        print(f"[INFO] Inference runtime override => {changed}")


def _write_inference_error(outdir, model_type, err):
    try:
        ensure_dir(outdir)
        err_path = os.path.join(outdir, "error.log")
        with open(err_path, "w", encoding="utf-8") as f:
            f.write(f"[ERROR] Inference failed for {model_type}:\n")
            f.write(str(err) + "\n\n")
            f.write(traceback.format_exc())
    except Exception:
        # If logging fails, fall back to console only.
        pass


def _find_latest_run_id(csv_name: str) -> str | None:
    base_dir = os.path.join("models", csv_name)
    if not os.path.isdir(base_dir):
        return None
    pattern = re.compile(r"^\d{8}_\d{6}(?:_.*)?$")
    candidates = []
    for name in os.listdir(base_dir):
        path = os.path.join(base_dir, name)
        if not os.path.isdir(path):
            continue
        if not pattern.match(name):
            continue
        if not os.path.exists(os.path.join(path, "metadata.pkl")):
            continue
        candidates.append(name)
    return max(candidates) if candidates else None



# Resolve group index by matching keyword against grouped feature names.
def find_group_idx(keyword, groups, colnames):
    kw = keyword.lower()
    for idx, grp in enumerate(groups):
        if any(kw in colnames[c].lower() for c in grp):
            return idx
    return None


def find_group_idx_by_name(keyword, group_names):
    kw = keyword.lower()
    for idx, name in enumerate(group_names):
        if kw in str(name).lower():
            return idx
    return None

def _get_model_input_dim(model):
    if hasattr(model, "n_features_in_"):
        val = int(model.n_features_in_)
        return val if val > 0 else None
    if hasattr(model, "model"):
        if hasattr(model.model, "n_features_in_"):
            val = int(model.model.n_features_in_)
            return val if val > 0 else None
        if hasattr(model.model, "feature_count_"):
            val = int(model.model.feature_count_)
            return val if val > 0 else None
        if hasattr(model.model, "feature_names_"):
            names = getattr(model.model, "feature_names_", None)
            if names:
                return len(names)
    if hasattr(model, "feature_names_"):
        names = getattr(model, "feature_names_", None)
        if names:
            return len(names)
    if hasattr(model, "net"):
        for layer in model.net:
            if hasattr(layer, "in_features"):
                return int(layer.in_features)
    return None


def _assert_feature_dim(model, expected_dim, model_type):
    actual_dim = _get_model_input_dim(model)
    if actual_dim is not None and actual_dim != expected_dim:
        raise RuntimeError(
            f"[ERROR] Feature dimension mismatch for {model_type}: "
            f"model expects {actual_dim}, current input has {expected_dim}. "
            "Please retrain the model."
        )


# --------------------------------------------------
#                Model Loading
# --------------------------------------------------
def load_inference_model(model_type, config, run_id=None, runtime_overrides: dict[str, int] | None = None):
    csv_name = os.path.splitext(os.path.basename(config["data"]["path"]))[0]
    rid = get_run_id(config) if run_id is None else run_id
    model_dir = get_model_dir(csv_name, model_type, run_id=rid)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"[ERROR] Directory not found => {model_dir}")

    x_col_path = os.path.join(model_dir, "x_col_names.npy")
    y_col_path = os.path.join(model_dir, "y_col_names.npy")
    if not (os.path.exists(x_col_path) and os.path.exists(y_col_path)):
        raise FileNotFoundError("[ERROR] x_col_names.npy or y_col_names.npy not found.")

    x_col_names = list(np.load(x_col_path, allow_pickle=True))
    y_col_names = list(np.load(y_col_path, allow_pickle=True))

    # ---------- ANN ----------
    if model_type == "ANN":
        ann_cfg = config["model"]["ann_params"].copy()

        if "hidden_dims" not in ann_cfg:
            best_params = None
            if config.get("optuna", {}).get("enable", False):
                optuna_dir = get_postprocess_dir(csv_name, rid, "optuna", "ANN")
                best_params_path = os.path.join(optuna_dir, "best_params.pkl")
                if os.path.exists(best_params_path):
                    best_params = joblib.load(best_params_path)
                    if isinstance(best_params.get("hidden_dims"), str):
                        best_params["hidden_dims"] = tuple(int(x) for x in best_params["hidden_dims"].split(","))
                    ann_cfg.update(best_params)
                    print(f"[INFO] Updated ann_params from optuna: {ann_cfg}")
                else:
                    print(f"[WARN] best_params not found for ANN => {best_params_path}, using defaults.")

            # fallback defaults for inference
            ann_cfg.setdefault("hidden_dims", (64, 64))
            ann_cfg.setdefault("dropout", 0.0)
            ann_cfg.setdefault("activation", "ReLU")
            ann_cfg.setdefault("random_seed", 42)

        net = ANNRegression(
            input_dim=len(x_col_names),
            output_dim=len(y_col_names),
            hidden_dims=ann_cfg["hidden_dims"],
            dropout=ann_cfg.get("dropout", 0.0),
            activation=ann_cfg.get("activation", "ReLU"),
            random_seed=ann_cfg.get("random_seed", 42)
        )

        ckpt_path = os.path.join(model_dir, "best_ann.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"[ERROR] {ckpt_path} not found.")
        try:
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except TypeError:  # Older torch versions may not support weights_only.
            state_dict = torch.load(ckpt_path, map_location="cpu")
        net.load_state_dict(state_dict)
        net.eval()
        _apply_model_runtime_overrides(net, model_type, runtime_overrides=runtime_overrides)
        _assert_feature_dim(net, len(x_col_names), model_type)
        return net, x_col_names, y_col_names

    # ---------- Non-ANN models ----------
    else:
        pkl_path = os.path.join(model_dir, "trained_model.pkl")
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"[ERROR] {pkl_path} not found.")
        model = joblib.load(pkl_path)
        _apply_model_runtime_overrides(model, model_type, runtime_overrides=runtime_overrides)
        _assert_feature_dim(model, len(x_col_names), model_type)
        return model, x_col_names, y_col_names


def model_predict(model, X_2d):
    """Unified prediction interface for Torch / sklearn / boosting wrappers."""
    if hasattr(model, "eval") and hasattr(model, "forward"):
        with torch.no_grad():
            out = model(torch.tensor(X_2d, dtype=torch.float32)).cpu().numpy()
    else:
        infer_thread_count = getattr(model, "_infer_thread_count", None)
        if infer_thread_count is not None:
            try:
                out = model.predict(X_2d, thread_count=int(infer_thread_count))
            except TypeError:
                out = model.predict(X_2d)
        else:
            out = model.predict(X_2d)

    # Always normalize to 2D: (n_samples, n_outputs).
    if out.ndim == 1:
        out = out.reshape(-1, 1)

    return out


def _get_base_vector(stats_dict, x_col_names):
    base = stats_dict.get("feature_means", None)
    if base is not None:
        base = np.asarray(base, dtype=float)
        if base.shape[0] == len(x_col_names):
            return base.copy()

    base_vec = np.zeros(len(x_col_names), dtype=float)
    for cname, cstat in stats_dict.get("continuous_cols", {}).items():
        if cname in x_col_names:
            base_vec[x_col_names.index(cname)] = float(cstat.get("mean", 0.0))
    return base_vec


def _get_group_entries(stats_dict, x_col_names):
    groups = []
    onehot_groups = stats_dict.get("onehot_groups", [])
    group_names = stats_dict.get("group_names", [])
    group_value_vectors = stats_dict.get("group_value_vectors", {})

    for gid, grp in enumerate(onehot_groups):
        name = group_names[gid] if gid < len(group_names) else f"group_{gid}"
        info = group_value_vectors.get(name)
        if not info:
            continue
        vecs = np.asarray(info.get("vectors", []), dtype=float)
        if vecs.size == 0:
            continue
        if vecs.shape[1] != len(grp):
            continue
        weights = info.get("weights")
        if weights is not None and len(weights) == vecs.shape[0]:
            weights = np.asarray(weights, dtype=float)
            if weights.sum() > 0:
                weights = weights / weights.sum()
            else:
                weights = None
        else:
            weights = None
        values = info.get("values", list(range(vecs.shape[0])))
        groups.append(
            {
                "gid": gid,
                "name": name,
                "indices": np.asarray(grp, dtype=int),
                "vectors": vecs,
                "weights": weights,
                "values": values,
            }
        )
    return groups


def _build_combo_templates(base_vec, groups, fixed=None, max_combos=None, seed=42):
    fixed = fixed or {}
    base = base_vec.copy()
    for g in groups:
        if g["gid"] in fixed:
            base[g["indices"]] = fixed[g["gid"]]

    iter_groups = [g for g in groups if g["gid"] not in fixed]
    if not iter_groups:
        return base.reshape(1, -1), np.array([1.0], dtype=float)

    sizes = [len(g["vectors"]) for g in iter_groups]
    total = 1
    for s in sizes:
        total *= s

    rng = np.random.default_rng(seed)
    if max_combos and total > max_combos:
        n = int(max_combos)
        templates = np.repeat(base.reshape(1, -1), n, axis=0)
        for g in iter_groups:
            w = g["weights"]
            if w is None:
                idxs = rng.integers(0, len(g["vectors"]), size=n)
            else:
                idxs = rng.choice(len(g["vectors"]), size=n, p=w)
            templates[:, g["indices"]] = g["vectors"][idxs]
        weights = np.ones(n, dtype=float)
        return templates, weights

    from itertools import product

    templates = []
    weights = []
    for combo in product(*[range(len(g["vectors"])) for g in iter_groups]):
        vec = base.copy()
        w = 1.0
        for g, idx in zip(iter_groups, combo):
            vec[g["indices"]] = g["vectors"][idx]
            if g["weights"] is not None:
                w *= float(g["weights"][idx])
        templates.append(vec)
        weights.append(w)

    templates = np.vstack(templates)
    weights = np.asarray(weights, dtype=float)
    if weights.sum() <= 0:
        weights = np.ones_like(weights)
    return templates, weights


def _parse_log_transform_config(stats_dict, x_col_names):
    loader_cfg = stats_dict.get("loader_config", {}) or {}
    raw_cols = list(loader_cfg.get("log_transform_cols", []) or [])
    try:
        log_eps = float(loader_cfg.get("log_transform_eps", 1e-8))
    except (TypeError, ValueError):
        log_eps = 1e-8
    if log_eps <= 0:
        log_eps = 1e-8
    idx_map = {name: i for i, name in enumerate(x_col_names)}
    cols = [c for c in raw_cols if c in idx_map]
    idxs = [idx_map[c] for c in cols]
    return set(cols), idxs, log_eps


def _to_display_domain(v, use_log):
    return np.exp(v) if use_log else v


def _prepare_model_input(batch, scaler_x, scale_cols_idx, log_cols_idx=None, log_eps=1e-8):
    X = batch.copy()
    if log_cols_idx:
        X[:, log_cols_idx] = np.log(np.clip(X[:, log_cols_idx], log_eps, None))
    if scaler_x is not None:
        X[:, scale_cols_idx] = scaler_x.transform(X[:, scale_cols_idx])
    return X


def _weighted_predict(model, batch, weights, scaler_x, scaler_y, scale_cols_idx,
                      log_cols_idx=None, log_eps=1e-8):
    X = _prepare_model_input(batch, scaler_x, scale_cols_idx, log_cols_idx=log_cols_idx, log_eps=log_eps)
    pred = model_predict(model, X)
    pred = inverse_transform_output(pred, scaler_y)
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
    w = np.asarray(weights, dtype=float).reshape(-1, 1)
    w_sum = float(w.sum())
    if w_sum <= 0:
        w = np.ones_like(w)
        w_sum = float(w.sum())
    return (pred * w).sum(axis=0) / w_sum



def get_onehot_global_col_index(local_oh_index, oh_index_map):
    return oh_index_map[local_oh_index]

# ==============================================================
#                 2D Heatmap Inference
# ==============================================================

def heatmap_2d_inference(model, x_name, y_name,
                         stats_dict, x_col_names, scale_cols_idx,
                         scaler_x, scaler_y,
                         outdir_m, n_points=50,
                         group_templates=None, group_weights=None,
                         log_cols=None, log_cols_idx=None, log_eps=1e-8,
                         grid_workers=1, chunk_rows=8):
    """Generate weighted 2D heatmap for two continuous axes."""
    if (x_name not in stats_dict["continuous_cols"]
            or y_name not in stats_dict["continuous_cols"]):
        print(f"[WARN] {x_name}/{y_name} is not in continuous_cols; skip this 2D heatmap.")
        return

    xinfo = stats_dict["continuous_cols"][x_name]
    yinfo = stats_dict["continuous_cols"][y_name]

    x_is_log = x_name in (log_cols or set())
    y_is_log = y_name in (log_cols or set())
    xv_model = np.linspace(xinfo["min"], xinfo["max"], n_points)
    yv_model = np.linspace(yinfo["min"], yinfo["max"], n_points)
    xv = _to_display_domain(xv_model, x_is_log)
    yv = _to_display_domain(yv_model, y_is_log)
    grid_x, grid_y = np.meshgrid(xv, yv)

    # Build baseline in model feature domain, then convert selected log cols to display domain.
    base_vec_model = _get_base_vector(stats_dict, x_col_names)
    base_vec = base_vec_model.copy()
    if log_cols_idx:
        base_vec[log_cols_idx] = np.exp(base_vec[log_cols_idx])
    if group_templates is None or group_weights is None:
        group_templates = base_vec.reshape(1, -1)
        group_weights = np.ones(1, dtype=float)

    tmp = base_vec.reshape(1, -1)
    tmp_model = _prepare_model_input(
        tmp, scaler_x, scale_cols_idx, log_cols_idx=log_cols_idx, log_eps=log_eps
    )
    out_dim = model_predict(model, tmp_model).shape[-1]

    H, W = grid_x.shape
    heatmap_pred = np.zeros((H, W, out_dim))
    x_idx = x_col_names.index(x_name)
    y_idx = x_col_names.index(y_name)

    def _compute_chunk(start: int, end: int):
        local = np.zeros((end - start, W, out_dim), dtype=float)
        for local_i, i in enumerate(range(start, end)):
            for j in range(W):
                batch = group_templates.copy()
                batch[:, x_idx] = grid_x[i, j]
                batch[:, y_idx] = grid_y[i, j]
                real = _weighted_predict(
                    model, batch, group_weights,
                    scaler_x, scaler_y, scale_cols_idx,
                    log_cols_idx=log_cols_idx, log_eps=log_eps
                )
                local[local_i, j, :] = np.maximum(real.reshape(-1), 0)
        return start, local

    results = _run_row_chunks(
        total_rows=H,
        chunk_rows=chunk_rows,
        worker_fn=_compute_chunk,
        max_workers=max(1, int(grid_workers)),
        task_name=f"2D({x_name},{y_name})",
    )
    for start, local in results:
        heatmap_pred[start:start + local.shape[0], :, :] = local

    tag = f"{x_name}__{y_name}".replace(" ", "_").replace("/", "_")
    np.save(os.path.join(outdir_m, f"grid_x_{tag}.npy"), grid_x)
    np.save(os.path.join(outdir_m, f"grid_y_{tag}.npy"), grid_y)
    np.save(os.path.join(outdir_m, f"heatmap_pred_{tag}.npy"), heatmap_pred)
    print(f"[INFO] 2D heatmap ({x_name},{y_name}) saved => {outdir_m}")


# ==============================================================
#                 3D Heatmap Inference
# ==============================================================

def heatmap_3d_inference(model, axes_names, stats_dict,
                         x_col_names, scale_cols_idx,
                         scaler_x, scaler_y,
                         outdir_m, n_points=40,
                         group_templates=None, group_weights=None,
                         log_cols=None, log_cols_idx=None, log_eps=1e-8,
                         grid_workers=1, chunk_rows=8):
    """axes_names = [x_name, y_name, z_name]"""
    x_name, y_name, z_name = axes_names

    def _mm(col):
        info = stats_dict["continuous_cols"][col]
        return info["min"], info["max"]

    x_is_log = x_name in (log_cols or set())
    y_is_log = y_name in (log_cols or set())
    z_is_log = z_name in (log_cols or set())
    xv_model = np.linspace(*_mm(x_name), n_points)
    yv_model = np.linspace(*_mm(y_name), n_points)
    zv_model = np.linspace(*_mm(z_name), n_points)
    xv = _to_display_domain(xv_model, x_is_log)
    yv = _to_display_domain(yv_model, y_is_log)
    zv = _to_display_domain(zv_model, z_is_log)
    grid_x, grid_y, grid_z = np.meshgrid(xv, yv, zv, indexing="ij")

    base_vec_model = _get_base_vector(stats_dict, x_col_names)
    base_vec = base_vec_model.copy()
    if log_cols_idx:
        base_vec[log_cols_idx] = np.exp(base_vec[log_cols_idx])
    if group_templates is None or group_weights is None:
        group_templates = base_vec.reshape(1, -1)
        group_weights = np.ones(1, dtype=float)

    tmp = base_vec.reshape(1, -1)
    tmp_model = _prepare_model_input(
        tmp, scaler_x, scale_cols_idx, log_cols_idx=log_cols_idx, log_eps=log_eps
    )
    out_dim = model_predict(model, tmp_model).shape[-1]

    H, W, D = grid_x.shape
    heatmap_pred = np.zeros((H, W, D, out_dim))
    x_idx = x_col_names.index(x_name)
    y_idx = x_col_names.index(y_name)
    z_idx = x_col_names.index(z_name)

    def _compute_chunk(start: int, end: int):
        local = np.zeros((end - start, W, D, out_dim), dtype=float)
        for local_i, i in enumerate(range(start, end)):
            for j in range(W):
                for k in range(D):
                    batch = group_templates.copy()
                    batch[:, x_idx] = grid_x[i, j, k]
                    batch[:, y_idx] = grid_y[i, j, k]
                    batch[:, z_idx] = grid_z[i, j, k]
                    real = _weighted_predict(
                        model, batch, group_weights,
                        scaler_x, scaler_y, scale_cols_idx,
                        log_cols_idx=log_cols_idx, log_eps=log_eps
                    )
                    local[local_i, j, k, :] = np.maximum(real.reshape(-1), 0)
        return start, local

    results = _run_row_chunks(
        total_rows=H,
        chunk_rows=chunk_rows,
        worker_fn=_compute_chunk,
        max_workers=max(1, int(grid_workers)),
        task_name="3DHeatmap-X",
    )
    for start, local in results:
        heatmap_pred[start:start + local.shape[0], :, :, :] = local

    np.save(os.path.join(outdir_m, "grid_x_3d.npy"), grid_x)
    np.save(os.path.join(outdir_m, "grid_y_3d.npy"), grid_y)
    np.save(os.path.join(outdir_m, "grid_z_3d.npy"), grid_z)
    np.save(os.path.join(outdir_m, "heatmap_pred_3d.npy"), heatmap_pred)
    print(f"[INFO] 3D heatmap saved => {outdir_m}")

# --------------------------------------------------
#                     Main Entry
# --------------------------------------------------
def inference_main(config_path: str | None = None):
    config_path = _resolve_config_path(config_path)
    print(f"[INFO] Using config => {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    _configure_torch_runtime()

    csv_path = config["data"]["path"]
    if not os.path.isabs(csv_path):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(config_path), ".."))
        csv_path = os.path.join(repo_root, csv_path)
        config["data"]["path"] = csv_path

    inf_models = config["inference"].get("models", [])
    if not inf_models:
        print("[INFO] No inference models => exit.")
        return

    csv_path = config["data"]["path"]
    csv_name = os.path.splitext(os.path.basename(csv_path))[0]
    run_id = get_run_id(config)
    if not run_id:
        run_id = _find_latest_run_id(csv_name)
        if run_id:
            print(f"[INFO] RUN_ID not set; using latest run_id => {run_id}")
        else:
            print("[ERROR] RUN_ID not set and no previous run found. Please train first or set RUN_ID.")
            return
    root_model_dir = get_root_model_dir(csv_name, run_id=run_id)
    meta_path = os.path.join(root_model_dir, "metadata.pkl")

    if not os.path.exists(meta_path):
        print(f"[ERROR] metadata => {meta_path} missing. Please retrain the model.")
        return

    base_stats_dict = joblib.load(meta_path)
    random_seed = config.get("data", {}).get("random_seed", 42)
    max_combos = config.get("inference", {}).get("max_combinations", None)

    base_inf = get_postprocess_dir(csv_name, run_id, "inference")
    ensure_dir(base_inf)

    parallel_plan = _resolve_inference_parallel_plan(inf_models)
    print(
        "[INFO] Inference parallel plan => "
        f"cpu_total={parallel_plan['cpu_total']}, "
        f"model_workers={parallel_plan['model_workers']}, "
        f"grid_workers={parallel_plan['grid_workers']}, "
        f"chunk_rows={parallel_plan['chunk_rows']}, "
        f"per_predict_threads={parallel_plan['per_predict_threads']}"
    )

    runtime_overrides = {
        "model_n_jobs": parallel_plan["per_predict_threads"],
        "rf_n_jobs": parallel_plan["per_predict_threads"],
        "xgb_n_jobs": parallel_plan["per_predict_threads"],
        "cat_threads": parallel_plan["per_predict_threads"],
        "svm_n_jobs": parallel_plan["per_predict_threads"],
    }

    def _run_single_model(mtype: str):
        outdir_m = os.path.join(base_inf, mtype)
        ensure_dir(outdir_m)
        err_path = os.path.join(outdir_m, "error.log")
        if os.path.exists(err_path):
            os.remove(err_path)
        print(f"\n=== Inference => {mtype} ===")
        try:
            model, x_col_names, y_col_names = load_inference_model(
                mtype, config, run_id=run_id, runtime_overrides=runtime_overrides
            )

            # --- scaler & per-model metadata ---
            model_dir = get_model_dir(csv_name, mtype, run_id=run_id)
            stats_dict = base_stats_dict
            meta_m = os.path.join(model_dir, "metadata.pkl")
            if os.path.exists(meta_m):
                stats_dict = joblib.load(meta_m)

            meta_x_cols = stats_dict.get("x_col_names")
            if meta_x_cols is not None and len(meta_x_cols) != len(x_col_names):
                raise RuntimeError(
                    f"Feature dimension mismatch between metadata and model for {mtype}: "
                    f"metadata has {len(meta_x_cols)}, model has {len(x_col_names)}. "
                    "Please retrain the model."
                )

            numeric_cols_idx = stats_dict["numeric_cols_idx"]
            scale_cols_idx_default = stats_dict.get("scale_cols_idx", numeric_cols_idx)
            scale_cols_idx_by_model = stats_dict.get("scale_cols_idx_by_model", {})
            onehot_groups = stats_dict.get("onehot_groups", [])
            group_names = stats_dict.get("group_names", [])
            group_value_vectors = stats_dict.get("group_value_vectors", {})
            log_cols, log_cols_idx, log_eps = _parse_log_transform_config(stats_dict, x_col_names)

            sx_path = os.path.join(model_dir, f"scaler_x_{mtype}.pkl")
            sy_path = os.path.join(model_dir, f"scaler_y_{mtype}.pkl")
            scaler_x = load_scaler(sx_path) if os.path.exists(sx_path) else None
            scaler_y = load_scaler(sy_path) if os.path.exists(sy_path) else None
            scale_cols_idx = scale_cols_idx_by_model.get(mtype, scale_cols_idx_default)
            scale_idx_path = os.path.join(model_dir, f"scale_cols_idx_{mtype}.npy")
            if os.path.exists(scale_idx_path):
                scale_cols_idx = np.load(scale_idx_path).tolist()

            if scaler_x and len(scale_cols_idx) != scaler_x.n_features_in_:
                raise RuntimeError(
                    f"scale_cols_idx length={len(scale_cols_idx)} does not match "
                    f"scaler_x.n_features_in_={scaler_x.n_features_in_}"
                )

            # ANN on CPU uses a global torch pool; avoid concurrent chunk workers.
            effective_grid_workers = parallel_plan["grid_workers"] if mtype != "ANN" else 1
            chunk_rows = parallel_plan["chunk_rows"]

            axes_names = config["inference"].get("heatmap_axes", [])
            dim_axes = len(axes_names)
            n_points = config["inference"].get("n_points", 50)
            enable_3d = config["inference"].get("enable_3d_heatmap", True)
            skip_3d_models = set(config["inference"].get("skip_3d_models", []))

            base_vec = _get_base_vector(stats_dict, x_col_names)
            if log_cols_idx:
                base_vec = base_vec.copy()
                base_vec[log_cols_idx] = np.exp(base_vec[log_cols_idx])
            group_entries = _get_group_entries(stats_dict, x_col_names)
            group_templates, group_weights = _build_combo_templates(
                base_vec,
                group_entries,
                fixed=None,
                max_combos=max_combos,
                seed=random_seed
            )

            if dim_axes == 2:
                heatmap_2d_inference(
                    model,
                    axes_names[0], axes_names[1],
                    stats_dict, x_col_names, scale_cols_idx,
                    scaler_x, scaler_y,
                    outdir_m, n_points,
                    group_templates=group_templates,
                    group_weights=group_weights,
                    log_cols=log_cols, log_cols_idx=log_cols_idx, log_eps=log_eps,
                    grid_workers=effective_grid_workers, chunk_rows=chunk_rows,
                )
            elif dim_axes == 3:
                for x_name, y_name in combinations(axes_names, 2):
                    heatmap_2d_inference(
                        model,
                        x_name, y_name,
                        stats_dict, x_col_names, scale_cols_idx,
                        scaler_x, scaler_y,
                        outdir_m, n_points,
                        group_templates=group_templates,
                        group_weights=group_weights,
                        log_cols=log_cols, log_cols_idx=log_cols_idx, log_eps=log_eps,
                        grid_workers=effective_grid_workers, chunk_rows=chunk_rows,
                    )
                if enable_3d and mtype not in skip_3d_models:
                    heatmap_3d_inference(
                        model,
                        axes_names,
                        stats_dict, x_col_names, scale_cols_idx,
                        scaler_x, scaler_y,
                        outdir_m, n_points,
                        group_templates=group_templates,
                        group_weights=group_weights,
                        log_cols=log_cols, log_cols_idx=log_cols_idx, log_eps=log_eps,
                        grid_workers=effective_grid_workers, chunk_rows=chunk_rows,
                    )
                else:
                    print(f"[INFO] Skip 3D heatmap for {mtype}.")
            else:
                print(f"[WARN] heatmap_axes={axes_names} (dim={dim_axes}) is invalid; expected 2 or 3 continuous axes.")

            if len(onehot_groups) < 2:
                print("[WARN] Not enough groups => skip confusion.")
                return

            conf_default = config["inference"]["confusion_axes"]
            conf_by_model = config["inference"].get("confusion_axes_by_model", {})
            conf_m = conf_by_model.get(mtype, {})
            row_kw = conf_m.get("row_name", conf_default["row_name"])
            col_kw = conf_m.get("col_name", conf_default["col_name"])

            row_idx = find_group_idx_by_name(row_kw, group_names) if group_names else None
            col_idx = find_group_idx_by_name(col_kw, group_names) if group_names else None

            if (row_idx is None) or (col_idx is None):
                row_idx = find_group_idx(row_kw, onehot_groups, x_col_names)
                col_idx = find_group_idx(col_kw, onehot_groups, x_col_names)

            if (row_idx is None) or (col_idx is None):
                print("[WARN] Invalid confusion row/col axes; fallback to first two groups.")
                row_idx, col_idx = 0, 1

            if row_idx == col_idx:
                print("[WARN] row_name and col_name resolved to the same group => skip confusion.")
                return

            grpA = onehot_groups[row_idx]
            grpB = onehot_groups[col_idx]

            row_name = group_names[row_idx] if row_idx < len(group_names) else f"group_{row_idx}"
            col_name = group_names[col_idx] if col_idx < len(group_names) else f"group_{col_idx}"

            row_info = group_value_vectors.get(row_name)
            col_info = group_value_vectors.get(col_name)
            if not row_info or not col_info:
                print("[WARN] Missing group value vectors => skip confusion.")
                return

            row_vals = row_info["values"]
            row_vecs = np.asarray(row_info["vectors"])
            col_vals = col_info["values"]
            col_vecs = np.asarray(col_info["vectors"])

            if row_vecs.shape[1] != len(grpA) or col_vecs.shape[1] != len(grpB):
                print("[WARN] Group vector dim mismatch => skip confusion.")
                return

            if max_combos and (len(row_vals) * len(col_vals) > max_combos):
                cap = max(1, int(np.floor(np.sqrt(max_combos))))
                row_vals = row_vals[:cap]
                row_vecs = row_vecs[:cap]
                col_vals = col_vals[:cap]
                col_vecs = col_vecs[:cap]

            base_vec_conf = _get_base_vector(stats_dict, x_col_names)
            other_groups = [g for g in group_entries if g["gid"] not in {row_idx, col_idx}]
            other_templates, other_weights = _build_combo_templates(
                base_vec_conf,
                other_groups,
                fixed=None,
                max_combos=max_combos,
                seed=random_seed
            )

            tmp = base_vec_conf.reshape(1, -1)
            tmp_model = _prepare_model_input(
                tmp, scaler_x, scale_cols_idx, log_cols_idx=log_cols_idx, log_eps=log_eps
            )
            outdim = model_predict(model, tmp_model).shape[-1]
            confusion_pred = np.zeros((len(row_vals), len(col_vals), outdim), dtype=float)

            def _compute_confusion_chunk(start: int, end: int):
                local = np.zeros((end - start, len(col_vals), outdim), dtype=float)
                for local_i, i in enumerate(range(start, end)):
                    for j in range(len(col_vals)):
                        batch = other_templates.copy()
                        batch[:, grpA] = row_vecs[i]
                        batch[:, grpB] = col_vecs[j]
                        real_pred = _weighted_predict(
                            model, batch, other_weights, scaler_x, scaler_y, scale_cols_idx,
                            log_cols_idx=log_cols_idx, log_eps=log_eps
                        )
                        local[local_i, j, :] = real_pred.reshape(-1)
                return start, local

            conf_results = _run_row_chunks(
                total_rows=len(row_vals),
                chunk_rows=chunk_rows,
                worker_fn=_compute_confusion_chunk,
                max_workers=max(1, int(effective_grid_workers)),
                task_name=f"Confusion({mtype})",
            )
            for start, local in conf_results:
                confusion_pred[start:start + local.shape[0], :, :] = local

            np.save(os.path.join(outdir_m, "confusion_row_labels.npy"), np.array(row_vals, dtype=object))
            np.save(os.path.join(outdir_m, "confusion_col_labels.npy"), np.array(col_vals, dtype=object))

            v_min = confusion_pred.min()
            v_max = confusion_pred.max()
            eps = 1e-12
            confusion_norm = (confusion_pred - v_min) / (v_max - v_min + eps)
            np.save(os.path.join(outdir_m, "confusion_pred_norm.npy"), confusion_norm)
            print(f"[INFO] confusion saved => {outdir_m}")

        except Exception as e:
            print(f"[ERROR] Inference failed for {mtype}: {e}")
            _write_inference_error(outdir_m, mtype, e)

    if parallel_plan["model_workers"] > 1 and len(inf_models) > 1:
        print(f"[INFO] Running model-level parallel inference with workers={parallel_plan['model_workers']}")
        with ThreadPoolExecutor(max_workers=parallel_plan["model_workers"]) as ex:
            futures = {ex.submit(_run_single_model, m): m for m in inf_models}
            for fut in as_completed(futures):
                mtype = futures[fut]
                try:
                    fut.result()
                except Exception as e:
                    # _run_single_model already writes error logs; keep console fallback.
                    print(f"[ERROR] Unexpected parallel failure for {mtype}: {e}")
    else:
        for mtype in inf_models:
            _run_single_model(mtype)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with YAML config.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config. Overrides CONFIG_FILE/CONFIG_PATH.",
    )
    args = parser.parse_args()
    inference_main(args.config)
