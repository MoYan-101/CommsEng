"""
utils.py

Includes all plotting functions and a few helpers:
1) correlation_heatmap (standard and one-hot)
2) training visualization: loss curve, scatter (MAE/MSE), residuals, feature importance, etc.
3) raw data analysis (KDE, scatter, boxplot)
4) inference visualization (2D heatmap + confusion matrix)
5) confusion matrices with values inside triangles, extended colorbar range, and square layout.

K-Fold logic has been removed; comments are kept.
"""

from __future__ import annotations
import os
import re
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.collections import PolyCollection
from matplotlib.artist import Artist
from matplotlib.projections.polar import PolarAxes
import pandas as pd
import math
from matplotlib.patches import Patch, Polygon, Rectangle
from sklearn.metrics import r2_score
from matplotlib.ticker import MaxNLocator, FormatStrFormatter  # Can be omitted if already imported above
import matplotlib.ticker as ticker
import shap  # type: ignore[reportMissingImports]
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d import Axes3D  # type: ignore[reportUnusedImport]
from matplotlib.lines import Line2D
# from matplotlib import colors
from scipy.ndimage import zoom
import warnings
from typing import cast
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Global font settings for all plots.
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

#save pt
def _resolve_run_id(run_id: str | None) -> str | None:
    rid = run_id if run_id not in (None, "") else os.environ.get("RUN_ID")
    if rid is None:
        return None
    rid = str(rid).strip()
    if not rid or rid.lower() in {"none", "null"}:
        return None
    return rid

def get_run_id(config: dict | None = None) -> str | None:
    rid = os.environ.get("RUN_ID")
    if rid:
        return rid
    if config:
        cfg_rid = config.get("run_id") or config.get("data", {}).get("run_id")
        if cfg_rid:
            return str(cfg_rid).strip()
    return None

def get_model_dir(csv_name: str, model_type: str, run_id: str | None = None) -> str:
    """Return the unified directory ./models/<csv_name>[/<run_id>]/<model_type>."""
    rid = _resolve_run_id(run_id)
    parts = ["./models", csv_name]
    if rid:
        parts.append(rid)
    parts.append(model_type)
    return os.path.join(*parts)

def get_root_model_dir(csv_name: str, run_id: str | None = None) -> str:
    """Return the root directory ./models/<csv_name>[/<run_id>] for metadata and model subdirs."""
    rid = _resolve_run_id(run_id)
    parts = ["./models", csv_name]
    if rid:
        parts.append(rid)
    return os.path.join(*parts)

def get_postprocess_dir(csv_name: str, run_id: str | None = None, *parts: str) -> str:
    rid = _resolve_run_id(run_id)
    base = ["postprocessing", csv_name]
    if rid:
        base.append(rid)
    base.extend(parts)
    return os.path.join(*base)

def get_eval_dir(csv_name: str, run_id: str | None = None, *parts: str) -> str:
    rid = _resolve_run_id(run_id)
    base = ["evaluation", "figures", csv_name]
    if rid:
        base.append(rid)
    base.extend(parts)
    return os.path.join(*base)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def ensure_dir_for_file(filepath):
    dir_ = os.path.dirname(filepath)
    if dir_:
        os.makedirs(dir_, exist_ok=True)

def normalize_data(data, vmin, vmax):
    """Normalize data to the [0, 1] range."""
    return (data - vmin) / (vmax - vmin) if vmax > vmin else data

def safe_filename(name):
    return re.sub(r'[^a-zA-Z0-9_.-]', '_', name)

# --------------- correlation ---------------
def short_label(s: str) -> str:
    """
    Split a string by underscores, keep the last segment, and then:
      - return it unchanged if it is fully uppercase (for example, "CO", "OH")
      - apply special handling for chemical symbols (for example, "cu(oh)2" -> "Cu(OH)2")
      - otherwise capitalize only the first letter and keep the rest unchanged
    """
    special_chemicals = {
        # --- Pure chemical formulas: keep upright with \mathrm ---
        "cu":  r"$\mathrm{Cu}$",
        "cu(oh)2": r"$\mathrm{Cu(OH)_{2}}$",
        "cuxo":    r"$\mathrm{Cu_{X}O}$",
        "cu2s":    r"$\mathrm{Cu_{2}S}$",
        "cu2(oh)2co3": r"$\mathrm{Cu_{2}(OH)_{2}CO_{3}}$",
        "c2+": r"$\mathrm{C_{2+}}$",
        "c1":  r"$\mathrm{C_{1}}$",
        "h2":  r"$\mathrm{H_{2}}$",

        # --- Long text also stays in \mathrm{}, with spaces as '\ ' ---
        "catalyst surface area (m2/g) (ln scale)":
            r"$\mathrm{Catalyst\ surface\ area\ (m^{2}/g)\ (LN\ scale)}$",

        "h2/co2 ratio (-)":
            r"$\mathrm{H_{2}/CO_{2}\ ratio\ (-)}$",

        "ch3oh (g/kg·h) (ln scale)":
            r"$\mathrm{STY\_CH_{3}OH\ (g/kg\!\cdot\!h)\ (LN\ scale)}$",

        "co2 conversion efficiency (%)":
            r"$\mathrm{CO_{2}\ conversion\ efficiency\ (\%)}$"
}


    s = str(s)
    parts = s.split('_')
    last_part = parts[-1]  # Keep the last segment.

    # Handle empty trailing segment.
    if not last_part:
        return s  # Avoid empty labels.

    # Check special chemical mappings first.
    lower_last_part = last_part.lower()  # Match in lowercase.
    if lower_last_part in special_chemicals:
        return special_chemicals[lower_last_part]

    # Return uppercase labels as-is.
    if last_part.isupper():
        return last_part

    # Otherwise capitalize only the first character.
    return last_part[0].upper() + last_part[1:]

def only_positive_formatter(x, pos):
    """
    Custom formatter:
      - return an empty string when x <= 0 so the tick label is hidden
      - show x as a float when x > 0
    """
    if x <= 0:
        return ""
    else:
        return f"{x:.2f}"

# ------------------------------------------------------------------------------
# Correlation‑Network Heatmap  (feature × feature, MIC / distance‑corr)
# ------------------------------------------------------------------------------

def plot_mic_network_heatmap(feature_df: pd.DataFrame,
                             filename: str,
                             method: str = "mic",
                             dpi: int = 700) -> None:
    """
    Parameters
    ----------
    feature_df : pd.DataFrame
    filename   : str
    method     : {"mic", "distance"}
        - "mic"      → Maximal Information Coefficient
        - "distance" → distance-correlation (dcor)
    dpi        : int
        Default dpi is 700.
    """

    # ------------------------------------------------------------------
    # 0. Internal helpers
    # ------------------------------------------------------------------
    def _load_style() -> None:
        try:
            plt.style.use("chartlab.mplstyle")
        except Exception:
            pass
        plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        plt.rcParams["xtick.major.size"] = 0
        plt.rcParams["ytick.major.size"] = 0

    def _gradient_color(min_v: float, max_v: float,
                        palette: list[str], v: float) -> str:
        """Linearly interpolate colors and return a HEX string."""
        if max_v == min_v:
            return palette[len(palette) // 2]
        t = np.clip((v - min_v) / (max_v - min_v), 0, 1)
        i = int(t * (len(palette) - 1))
        c0 = mcolors.to_rgb(palette[i])
        c1 = mcolors.to_rgb(palette[min(i + 1, len(palette) - 1)])
        blend = tuple((1 - (t % 1)) * s + (t % 1) * e for s, e in zip(c0, c1))
        return mcolors.to_hex(blend)

    def _mic(x: np.ndarray, y: np.ndarray) -> float:
        """Compute MIC; fall back to distance-correlation or Pearson if minepy is unavailable."""
        try:
            from minepy import MINE  # type: ignore[reportMissingImports]
            mine = MINE(alpha=0.6, c=15)
            mine.compute_score(x, y)
            return float(mine.mic())
        except Exception:
            try:
                import dcor  # type: ignore[reportMissingImports]
                return float(dcor.distance_correlation(x, y))
            except Exception:
                val = float(np.corrcoef(x, y)[0, 1])
                return 0.0 if np.isnan(val) else abs(val)

    def _corr_matrix(df: pd.DataFrame, _method: str = "mic") -> np.ndarray:
        """Compute an n x n nonlinear correlation matrix using MIC or distance-correlation."""
        n = df.shape[1]
        mat = np.eye(n)
        to_num = lambda s: s.to_numpy(float) if pd.api.types.is_numeric_dtype(s) \
                           else s.astype("category").cat.codes.to_numpy(float)
        for i in range(n):
            xi = to_num(df.iloc[:, i])
            for j in range(i + 1, n):
                xj = to_num(df.iloc[:, j])
                m  = min(len(xi), len(xj))
                if _method == "mic":
                    score = _mic(xi[:m], xj[:m])
                else:
                    # Use distance-correlation; fall back to |Pearson| if dcor is unavailable.
                    try:
                        import dcor  # type: ignore[reportMissingImports]
                        score = abs(dcor.distance_correlation(xi[:m], xj[:m]))
                    except Exception:
                        score = abs(np.corrcoef(xi[:m], xj[:m])[0, 1])
                mat[i, j] = mat[j, i] = np.clip(score, 0, 1)
        return mat

    # ------------------------------------------------------------------
    # 1. Prepare data and colors
    # ------------------------------------------------------------------
    _load_style()
    palette = ["#515a85", "#c0627a"]

    C         = _corr_matrix(feature_df, method)
    feat_full = feature_df.columns.to_list()
    feat_lbl = []
    for c in feat_full:
        lbl = short_label(c)
        feat_lbl.append(lbl)
    n         = len(feat_full)

    # ------------------------------------------------------------------
    # 2. Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(max(8, .55 * n) + 3,  # Leave extra space on the right.
                                    max(6, .55 * n)),
                           dpi=dpi)

    # Draw squares once for the upper triangle, including the diagonal.
    for i in range(n):
        for j in range(i, n):
            s   = C[i, j]
            col = _gradient_color(0, 1, palette, s)
            # Background frame.
            ax.add_patch(mpatches.Rectangle((n - i - 1, j), 1, 1,
                                            edgecolor="#999999", facecolor="#ffffff",
                                            linewidth=.25))
            # Proportional square.
            ax.add_patch(mpatches.Rectangle((n - i - 0.5 - s / 2, j + 0.5 - s / 2),
                                            s, s,
                                            edgecolor="#999999", facecolor=col,
                                            linewidth=.5))

        # ---------------------------- Axis labels ----------------------------
        # Keep only right-side vertical labels and rotate them by 30 degrees.
        for k, lab in enumerate(feat_lbl):
            ax.text(n + 0.6,  # Shift slightly right from 0.5 to 0.6.
                    n - 0.5 - k,
                    lab,
                    ha="left",  # Start text from the left edge.
                    va="center",
                    fontsize=10)

    # Colorbar.
    # cmap  = mcolors.LinearSegmentedColormap.from_list("mic_cmap", palette)
    # cb_ax = fig.add_axes([0.09, 0.15, 0.03, 0.25])
    # cb    = plt.colorbar(cm.ScalarMappable(cmap=cmap,
    #                                        norm=mcolors.Normalize(0, 1)),
    #                      cax=cb_ax)
    # cb.ax.set_title(f"{method.upper()}\nCorr", fontsize=9, pad=8)

        # -------------------- Colorbar: place it at the lower left --------------------
        # Use a small horizontal bar in the lower-left corner.
        cb_ax = fig.add_axes((0.15, 0.25, 0.25, 0.03))  # [left, bottom, width, height]

        # Horizontal colorbar.
        cb = plt.colorbar(
            cm.ScalarMappable(
                cmap=mcolors.LinearSegmentedColormap.from_list("mic_cmap", palette),
                norm=mcolors.Normalize(0, 1)
            ),
            cax=cb_ax,
            orientation="horizontal"
        )

        # Rotate tick labels by 45 degrees.
        # Label.
        # cb.set_label(f"{method.upper()} Corr", labelpad=4,
        #              fontsize=9, ha="left", va="center")
        cb.set_label(f"{method.upper()} Corr", fontsize=11)
        # Tick labels.
        for t in cb.ax.get_xticklabels():
            t.set_rotation(45)
            t.set_horizontalalignment("right")  # Keep labels tucked close to the ticks.
            t.set_fontsize(11)

        # Remove the colorbar outline for a cleaner look.
        for spine in cb.ax.spines.values():
            spine.set_visible(False)

    # Final styling and save.
    ax.set_xticklabels([]); ax.set_yticklabels([])
    ax.xaxis.tick_top();    ax.yaxis.tick_right()
    ax.axis("equal")
    ax.set_xlim(-2, n + 1.5)
    ax.set_ylim(-1, n + 1)
    ax.set_title("Correlation Network Heatmap",
                 fontsize=14, pad=18)

    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    # Remove the outer frame of the plot.
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout(rect=(0, 0, 0.88, 1))
    plt.savefig(filename, dpi=dpi)
    plt.close()
    print(f"[plot_mic_network_heatmap] → {filename}")


# --------------- Training visualization: loss, scatter, residuals, etc. ---------------
def plot_loss_curve(train_losses, val_losses, filename):
    ensure_dir_for_file(filename)
    plt.figure()
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Val Loss', linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    # plt.title("Training/Validation Loss")
    plt.savefig(filename, dpi=700, format='jpg')
    plt.close()


def plot_joint_scatter_with_marginals(y_true, y_pred, y_labels=None, filename="joint_scatter_with_marginals.jpg"):
    """
    Create joint scatter plots with marginal gradient-filled KDE curves for each output dimension.

    For each output dimension:
      - Display a scatter plot of true vs. predicted values, colored by 2D kernel density.
      - Draw a reference line (dashed red) for perfect prediction (True = Predicted).
      - Add marginal plots above and to the right showing the KDE of true and predicted values.
        The marginal plots are filled with a gradient color that reflects the local KDE value using
        a segment-wise PolyCollection:
           * Top marginal: using the 'Blues' colormap for true values, colored by the KDE density.
           * Right marginal: using the 'Reds' colormap for predicted values, colored by the KDE density.
      - Annotate the main plot with the R² score.

    Parameters:
      y_true : numpy array of true values, shape (N, out_dim)
      y_pred : numpy array of predicted values, shape (N, out_dim)
      y_labels : Optional list of labels for each output dimension.
      filename : Name of the file to save the plot.
    """
    ensure_dir_for_file(filename)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 2 or y_pred.ndim != 2:
        raise ValueError("y_true and y_pred must be 2-dimensional arrays, shaped as (N, out_dim)")

    _, out_dim = y_pred.shape
    fig, axes = plt.subplots(1, out_dim, figsize=(5.3 * out_dim, 5.15), squeeze=False)

    # Define colormaps for main and marginal plots.
    cmap_main = cm.get_cmap('cividis')
    cmap_x = cm.get_cmap('Blues')
    cmap_y = cm.get_cmap('Reds')

    for i in range(out_dim):
        # Extract data for current dimension.
        x = y_true[:, i]
        y = y_pred[:, i]
        ax = axes[0, i]

        # Compute R² score.
        r2_val = r2_score(x, y)

        # Main scatter plot: colored by 2D kernel density.
        xy = np.vstack([x, y])
        kde_xy = gaussian_kde(xy)
        density_xy = kde_xy(xy)
        sc = ax.scatter(x, y, c=density_xy, cmap=cmap_main, alpha=0.5, edgecolor='none')

        # Draw perfect prediction line using data min and max.
        min_val = float(np.min([np.min(x), np.min(y)]))
        max_val = float(np.max([np.max(x), np.max(y)]))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5)

        # Set axis labels; keep font sizes consistent.
        if y_labels and i < len(y_labels):
            # ax.set_title(f"{y_labels[i]} (MAE)", fontsize=16)
            ax.set_xlabel(f"True {y_labels[i]}", fontsize=17)
            ax.set_ylabel(f"Predicted {y_labels[i]}", fontsize=17)

        else:
            # ax.set_title(f"Out {i} (MAE)", fontsize=16)
            ax.set_xlabel("True Value", fontsize=17)
            ax.set_ylabel("Predicted Value", fontsize=17)
        # Keep this after set_xlabel / set_ylabel.
        ax.tick_params(axis="both", labelsize=17)
        # Annotate R² in the main plot.
        ax.text(0.05, 0.95, f"R² = {r2_val:.3f}", transform=ax.transAxes,
                fontsize=16, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))

        # Hide the main plot's top and right borders.
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Get main axis data coordinate range.
        current_xlim = ax.get_xlim()
        current_ylim = ax.get_ylim()

        # Add marginal axes for KDE plots.
        divider = make_axes_locatable(ax)
        ax_histx = divider.append_axes("top", size="20%", pad=0, sharex=ax)
        ax_histy = divider.append_axes("right", size="20%", pad=0, sharey=ax)
        # Force marginal axes to use the same coordinate range as the main plot.
        ax_histx.set_xlim(current_xlim)
        ax_histy.set_ylim(current_ylim)

        # --- Top marginal (x): gradient-filled KDE for true values ---
        x_vals = np.linspace(current_xlim[0], current_xlim[1], 200)
        if len(x) > 1:
            kde_x = gaussian_kde(x)
            kde_x_vals = kde_x(x_vals)
        else:
            kde_x_vals = np.zeros_like(x_vals)
        segments_x = []
        colors_x = []
        # Use KDE values (density) for color mapping.
        norm_kde_x = mcolors.Normalize(vmin=np.min(kde_x_vals), vmax=np.max(kde_x_vals))
        for j in range(len(x_vals) - 1):
            x0, x1 = x_vals[j], x_vals[j + 1]
            d0, d1 = kde_x_vals[j], kde_x_vals[j + 1]
            segments_x.append([[x0, 0], [x0, d0], [x1, d1], [x1, 0]])
            mid_density = 0.5 * (d0 + d1)
            colors_x.append(cmap_x(norm_kde_x(mid_density)))
        pc_x = PolyCollection(segments_x, facecolors=colors_x, edgecolors='none', alpha=0.8)
        ax_histx.plot(x_vals, kde_x_vals, color='darkblue', linewidth=1.2, alpha=0.5)
        ax_histx.add_collection(pc_x)
        ax_histx.set_ylim(0, np.max(kde_x_vals))
        ax_histx.axis('off')

        # --- Right marginal (y): gradient-filled KDE for predicted values ---
        y_vals = np.linspace(current_ylim[0], current_ylim[1], 200)
        if len(y) > 1:
            kde_y_obj = gaussian_kde(y)
            kde_y_vals = kde_y_obj(y_vals)
        else:
            kde_y_vals = np.zeros_like(y_vals)
        segments_y = []
        colors_y = []
        # Use KDE values (density) for color mapping.
        norm_kde_y = mcolors.Normalize(vmin=np.min(kde_y_vals), vmax=np.max(kde_y_vals))
        for j in range(len(y_vals) - 1):
            y0, y1 = y_vals[j], y_vals[j + 1]
            d0, d1 = kde_y_vals[j], kde_y_vals[j + 1]
            segments_y.append([[0, y0], [d0, y0], [d1, y1], [0, y1]])
            mid_density = 0.5 * (d0 + d1)
            colors_y.append(cmap_y(norm_kde_y(mid_density)))
        pc_y = PolyCollection(segments_y, facecolors=colors_y, edgecolors='none', alpha=0.8)
        ax_histy.plot(kde_y_vals, y_vals, color='darkred', linewidth=1.2, alpha=0.5)
        ax_histy.add_collection(pc_y)
        ax_histy.set_xlim(0, np.max(kde_y_vals))
        ax_histy.axis('off')

    plt.tight_layout()
    plt.savefig(filename, dpi=700)
    plt.close()


class MyScalarFormatter(ticker.ScalarFormatter):
    def __init__(self, useMathText=True):
        super().__init__(useMathText=useMathText)
        # You can also call set_powerlimits((0, 0)) externally to force scientific notation.

    def _set_format(self):
        # Show only one decimal place.
        self.format = '%.1f'



# --------------- Raw data analysis ---------------
def plot_kde_distribution(df, columns, filename):
    """
    Draw KDE plots for each column in `columns`, up to 4 per row, wrapping to new rows
    as needed. Figure size adapts to the grid dimensions.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list[str]
        List of column names to plot.
    filename : str
        Path to save the figure.
    """
    ensure_dir_for_file(filename)

    n = len(columns)
    if n == 0:
        raise ValueError("No columns provided to plot.")

    # determine grid: max 4 columns per row
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)

    # let figure size grow with grid (approx 4" per subplot)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4 * ncols, 4 * nrows),
                             squeeze=False)
    axes = np.ravel(axes)

    for i, col in enumerate(columns):
        ax = axes[i]
        if col not in df.columns:
            ax.text(0.5, 0.5, f"'{col}' not in df", ha='center', va='center')
            ax.set_axis_off()
            continue

        # basic KDE
        sns.kdeplot(df[col], ax=ax, fill=False, color="black",
                    clip=(df[col].min(), df[col].max()))

        lines = ax.get_lines()
        if not lines:
            ax.set_title(f"No Data for {col}")
            ax.set_axis_off()
            continue

        line = lines[-1]
        x_plot, y_plot = line.get_xdata(), line.get_ydata()
        idxsort = np.argsort(x_plot)
        x_plot, y_plot = x_plot[idxsort], y_plot[idxsort]

        # build gradient fill under curve
        vmin, vmax = float(np.min(x_plot)), float(np.max(x_plot))
        cmap = cm.get_cmap("coolwarm")
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        for j in range(len(x_plot) - 1):
            x0, x1 = x_plot[j], x_plot[j + 1]
            y0, y1 = y_plot[j], y_plot[j + 1]
            color = cmap(norm((x0 + x1) * 0.5))
            verts = np.array([[x0, 0], [x0, y0], [x1, y1], [x1, 0]])
            poly = PolyCollection([verts],
                                  facecolors=[color],
                                  edgecolor='none',
                                  alpha=0.6)
            ax.add_collection(poly)

        # labels (slightly larger for data analysis plots)
        label_size = 14
        tick_size = 11
        cb_label_size = 12
        cb_tick_size = 10
        offset_size = 10

        ax.set_xlabel(col, fontsize=label_size)
        ax.set_ylabel("Density", fontsize=label_size)
        ax.tick_params(axis="both", labelsize=tick_size)
        ax.set_xlim(vmin, vmax)

        # y-axis formatting
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
        fmt = MyScalarFormatter(useMathText=True)
        fmt.set_powerlimits((0, 0))
        ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.get_offset_text().set_fontsize(offset_size)

        # colorbar
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cb = plt.colorbar(sm, ax=ax, pad=0.02)
        cb.set_label("Value", fontsize=cb_label_size)
        cb.ax.tick_params(labelsize=cb_tick_size)

    # hide any unused axes
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(filename, dpi=700)
    plt.close()
    print(f"[plot_kde_distribution] => {filename}")

#####################################################
# Custom one-hot merge helpers
#####################################################
import copy


def _normalize_feature_key(name: str) -> str:
    return re.sub(r"\s+", " ", str(name).strip()).lower()


def _parse_pair_interaction_feature(col_name: str) -> tuple[str, str] | None:
    """
    Parse pair-interaction names generated by data_loader_modified, e.g.
    'Promoter 1__x__Promoter 2__embed_cosine' or
    'Promoter 1__x__Promoter 2__pair__A__PAIR__B'.
    """
    m = re.match(r"^\s*(.*?)\s*__x__\s*(.*?)\s*__.+$", str(col_name))
    if not m:
        return None
    left = m.group(1).strip()
    right = m.group(2).strip()
    if not left or not right:
        return None
    return left, right


def _aggregate_pair_interactions_to_original(
    shap_list: list[np.ndarray],
    data_arr: np.ndarray | None,
    col_names: list[str],
    split_ratio: float = 0.5,
) -> tuple[list[np.ndarray], np.ndarray | None, list[str]]:
    """
    Fold pairwise interaction SHAP columns back to their two original features.
    This keeps model training untouched and only changes SHAP display space.
    """
    if not shap_list or not col_names:
        return shap_list, data_arr, col_names

    # Clamp split ratio and keep additivity by default (0.5 + 0.5 = 1.0).
    split = float(split_ratio)
    if split < 0.0:
        split = 0.0
    if split > 1.0:
        split = 1.0

    n_samples = shap_list[0].shape[0]
    names = list(col_names)
    key_to_idx = {_normalize_feature_key(n): i for i, n in enumerate(names)}

    def _ensure_base_feature(feature_name: str) -> int:
        nonlocal data_arr
        key = _normalize_feature_key(feature_name)
        if key in key_to_idx:
            return key_to_idx[key]
        # If a base feature was previously merged away, re-create it as zero column.
        new_idx = len(names)
        names.append(feature_name)
        key_to_idx[key] = new_idx
        for i in range(len(shap_list)):
            zeros = np.zeros((n_samples, 1), dtype=shap_list[i].dtype)
            shap_list[i] = np.hstack([shap_list[i], zeros])
        if data_arr is not None:
            zeros_d = np.zeros((n_samples, 1), dtype=data_arr.dtype)
            data_arr = np.hstack([data_arr, zeros_d])
        return new_idx

    interaction_idx: list[int] = []
    base_col_count = len(col_names)
    for idx in range(base_col_count):
        parsed = _parse_pair_interaction_feature(names[idx])
        if parsed is None:
            continue
        left_name, right_name = parsed
        left_idx = _ensure_base_feature(left_name)
        right_idx = _ensure_base_feature(right_name)

        # Distribute SHAP interaction contribution back to original features.
        for i in range(len(shap_list)):
            contrib = shap_list[i][:, idx]
            if left_idx == right_idx:
                shap_list[i][:, left_idx] += contrib
            else:
                shap_list[i][:, left_idx] += contrib * split
                shap_list[i][:, right_idx] += contrib * (1.0 - split)

        # Keep data matrix aligned for beeswarm coloring.
        if data_arr is not None:
            d = data_arr[:, idx]
            if left_idx == right_idx:
                data_arr[:, left_idx] += d
            else:
                data_arr[:, left_idx] += d * split
                data_arr[:, right_idx] += d * (1.0 - split)

        interaction_idx.append(idx)

    if not interaction_idx:
        return shap_list, data_arr, names

    drop = set(interaction_idx)
    keep_idx = [i for i in range(len(names)) if i not in drop]
    shap_list = [sv[:, keep_idx] for sv in shap_list]
    if data_arr is not None:
        data_arr = data_arr[:, keep_idx]
    names = [names[i] for i in keep_idx]
    return shap_list, data_arr, names


def merge_onehot_shap(
    shap_data,
    onehot_groups,
    case_map=None,
    aggregate_interactions_to_original=False,
    interaction_split_ratio=0.5,
):
    """
    Merge one-hot dummy columns from the same category into a single column and return a new shap_data dict.
    - shap_data: dict saved by train.py and loaded by visualization.py
                 must include "shap_values", "X_full", and "x_col_names"
    - onehot_groups: [[7,8,9], [10,11], ...], each sublist is a global index group of dummy columns
    - case_map: {lower_name: OriginalName}, used when original casing should be restored
    - aggregate_interactions_to_original:
        When True, aggregate interaction features such as
        "Promoter 1__x__Promoter 2__..." back to the original features
        so they no longer appear as separate entries in final SHAP ranking/display.
    """
    shap_values = shap_data["shap_values"]
    X_full      = shap_data["X_full"]
    col_names   = shap_data["x_col_names"]

    # ------------- 1) Normalize to a list -------------
    shap_is_list = isinstance(shap_values, list)
    shap_values = shap_values if shap_is_list else [shap_values]

    # ------------- 2) Build indices for retained columns -------------
    flat_oh = {i for g in onehot_groups for i in g}
    keep_idx = [i for i in range(len(col_names)) if i not in flat_oh]

    # ------------- 3) Build new column names -------------
    new_col_names = [col_names[i] for i in keep_idx]
    for g in onehot_groups:
        pref = col_names[g[0]].split("__", 1)[0]  # Use the prefix as the category name.
        if case_map is not None:
            pref = case_map.get(pref.lower(), pref)
        new_col_names.append(pref)

    # ------------- 4) Merge SHAP values and X_full -------------
    new_shap_list, new_data = [], []
    for sv in shap_values:                        # sv: (n_samples, n_features)
        parts = [sv[:, keep_idx]]
        for g in onehot_groups:
            parts.append(sv[:, g].sum(axis=1, keepdims=True))
        new_shap_list.append(np.hstack(parts))

    if X_full is not None:
        parts_d = [X_full[:, keep_idx]]
        for g in onehot_groups:
            # Use the argmax column index as the category marker.
            chosen = (X_full[:, g].argmax(axis=1)).reshape(-1, 1)
            parts_d.append(chosen)
        new_data = np.hstack(parts_d)
    else:
        new_data = None

    # ------------- 5) Optionally fold interaction features back to original features -------------
    if aggregate_interactions_to_original:
        new_shap_list, new_data, new_col_names = _aggregate_pair_interactions_to_original(
            shap_list=new_shap_list,
            data_arr=new_data,
            col_names=new_col_names,
            split_ratio=interaction_split_ratio,
        )

    # ------------- 6) Pack and return -------------
    new_sd = copy.deepcopy(shap_data)
    new_sd["shap_values"] = new_shap_list if shap_is_list else new_shap_list[0]
    new_sd["X_full"]      = new_data
    new_sd["x_col_names"] = new_col_names
    return new_sd

#####################################################
# 1) Custom plot_shap_importance function
#####################################################
def plot_shap_importance(
    shap_data,
    output_path,
    top_n_features=15,
    plot_width=12,
    plot_height=8
):
    """
    Draw a custom SHAP feature-importance bar chart:
      - compute feature importance as mean(|SHAP|)
      - show only the top_n_features
      - use the mean of those top features as a threshold:
        blue for above mean, red for below or equal
      - mark the threshold with shading and a dashed line
      - support multi-output cases when shap_values is a list

    shap_data must contain:
        "shap_values": array or list<array> with shape (n_samples, n_features)
        "X_full":      shape (n_samples, n_features); only the column count needs to match
        "x_col_names": feature names, length n_features
        "y_col_names": output names for multi-output SHAP values
    """
    ensure_dir_for_file(os.path.join(output_path, "dummy.txt"))  # Ensure the output directory exists.

    shap_values = shap_data["shap_values"]
    X_full = np.asarray(shap_data["X_full"])
    x_col_names = shap_data["x_col_names"]
    y_col_names = shap_data["y_col_names"]

    # Simplify labels with short_label.
    x_col_names = [short_label(col) for col in x_col_names]
    y_col_names = [short_label(y) for y in y_col_names]

    # Convert to a list for unified handling in single-output cases.
    multi_output = True
    if not isinstance(shap_values, list):
        shap_values = [shap_values]
        multi_output = False

    # Plot each output separately.
    for idx, sv in enumerate(shap_values):
        sv_arr = np.asarray(sv)
        # Compute feature importance as mean(|SHAP|).
        # sv shape: (n_samples, n_features)
        mean_abs_shap = np.mean(np.abs(sv_arr), axis=0)  # (n_features, )

        # Select the top features.
        sorted_idx = np.argsort(mean_abs_shap)[::-1]  # Descending order.
        top_idx = sorted_idx[:top_n_features]
        top_imps = mean_abs_shap[top_idx]
        top_feats = [x_col_names[i] for i in top_idx]

        # Compute the threshold as the mean importance.
        threshold = top_imps.mean()

        # Colors: blue above mean, red below or equal to mean.
        colors = ["blue" if imp > threshold else "red" for imp in top_imps]

        fig, ax = plt.subplots(figsize=(plot_width, plot_height))
        ax.barh(range(len(top_imps)), top_imps, align='center', color=colors)
        ax.set_yticks(range(len(top_imps)))
        ax.set_yticklabels(top_feats, fontsize=10)
        ax.invert_yaxis()

        # X-axis label and title handling.
        ax.set_xlabel("Mean(|SHAP|)", fontsize=12)
        if multi_output:
            out_label = y_col_names[idx] if idx < len(y_col_names) else f"Output{idx}"
            safe_out_label = safe_filename(out_label)  # Sanitize the output label for filenames.
            out_name = f"shap_importance_{safe_out_label}.jpg"
            # In multi-output mode, use the matching y_col_names entry.
            # ax.set_title(f"Mean |SHAP| (Top-{top_n_features}) - {out_label}",
            #              fontsize=14)
        else:
            # ax.set_title(f"Mean |SHAP| (Top-{top_n_features})", fontsize=14)
            out_name = "shap_importance.jpg"

        # Draw threshold shading and a vertical line.
        ax.axvspan(0, threshold, facecolor='lightgray', alpha=0.5)
        ax.axvline(threshold, color='gray', linestyle='dashed', linewidth=2)

        # Legend.
        legend_e = [
            Patch(facecolor="blue", label="Above Mean"),
            Patch(facecolor="red", label="Below/Equal Mean")
        ]
        ax.legend(handles=legend_e, loc="lower right", fontsize=12)

        # Keep plot borders visible.
        for spine in ax.spines.values():
            spine.set_visible(True)

        plt.tight_layout()
        save_path = os.path.join(output_path, out_name)
        plt.savefig(save_path, dpi=700)
        plt.close()
        print(f"[INFO] SHAP importance (custom) saved => {save_path}")


#####################################################
# 2) Custom plot_shap_beeswarm function (with visible frame)
#####################################################
def plot_shap_beeswarm(
    shap_data,
    output_path,
    top_n_features=15,
    plot_width=12,
    plot_height=8
):
    """
    Draw a beeswarm plot with shap.summary_plot(..., plot_type='dot'/default)
    and then manually make the outer frame visible.

    Parameters:
    -------
    shap_data : dict
        Contains "shap_values", "X_full", "x_col_names", and "y_col_names"
    top_n_features : int
        Maximum number of displayed features
    plot_width, plot_height : float
        Control the figure size
    """
    ensure_dir_for_file(os.path.join(output_path, "dummy.txt"))  # Ensure the output directory exists.

    shap_values = shap_data["shap_values"]
    X_full = shap_data["X_full"]
    x_col_names = shap_data["x_col_names"]
    y_col_names = shap_data["y_col_names"]

    # Simplify feature and output names.
    x_col_names = [short_label(col) for col in x_col_names]
    y_col_names = [short_label(y) for y in y_col_names]

    # Detect multi-output mode.
    multi_output = True
    if not isinstance(shap_values, list):
        shap_values = [shap_values]
        multi_output = False

    for idx, sv in enumerate(shap_values):
        if multi_output:
            out_label = y_col_names[idx] if idx < len(y_col_names) else f"Output{idx}"
            safe_out_label = safe_filename(out_label)  # Sanitize the output label for filenames.
            out_name = f"shap_beeswarm_{safe_out_label}.jpg"
        else:
            out_name = "shap_beeswarm.jpg"

        sv_arr = np.asarray(sv)
        X_full_arr = np.asarray(X_full)
        # Generate the beeswarm plot via shap.summary_plot.
        shap.summary_plot(
            sv_arr,
            features=X_full_arr,
            feature_names=x_col_names,
            show=False,
            max_display=top_n_features,
            plot_size=(plot_width, plot_height)
        )
        # shap.summary_plot creates or switches to SHAP's default figure/axes.
        ax = plt.gca()

        # Make the outer frame visible.
        for spine in ax.spines.values():
            spine.set_visible(True)

        plt.tight_layout()
        save_path = os.path.join(output_path, out_name)

        plt.savefig(save_path, dpi=700)
        plt.close()
        print(f"[INFO] SHAP beeswarm saved => {save_path}")

STYLE_COLORS = ["#24345C", "#279DE1", "#36CDCB", "#FF7F4C"]


def plot_shap_importance_multi_output(
    shap_data,
    output_path,
    top_n_features=14,
    plot_width=12,
    plot_height=8
):
    """
    Draw a stacked SHAP feature-importance bar chart for multi-output (MIMO) models:
      - compute mean(|SHAP|) separately for each output
      - merge them into a matrix of shape (n_features, n_outputs)
      - rank features by the row-wise sum and keep the Top-N
      - draw horizontal stacked bars, one row per feature
      - use a different color for each output and show output names in the legend

    Parameters
    ----
    shap_data : dict
        {
          "shap_values": list of arrays or a single array;
                        a list means multi-output, with
                        shap_values[i].shape = (n_samples, n_features)
          "x_col_names": list of feature names
          "y_col_names": list of output names
          ...
        }
    output_path : str
        Path to save the figure. The directory is created internally.
    top_n_features : int
        Show only the top N features by total contribution
    plot_width, plot_height : float
        Figure size
    """

    ensure_dir_for_file(output_path)

    shap_values = shap_data["shap_values"]  # May be a list or a single array.
    x_col_names = shap_data["x_col_names"]
    y_col_names = shap_data.get("y_col_names", None)  # May be missing.

    # Convert to a list for unified multi-output handling.
    if not isinstance(shap_values, list):
        shap_values = [shap_values]
    n_outputs = len(shap_values)

    # Repeat colors if there are more outputs than base colors.
    if n_outputs > len(STYLE_COLORS):
        color_palette = STYLE_COLORS * (n_outputs // len(STYLE_COLORS) + 1)
    else:
        color_palette = STYLE_COLORS[:n_outputs]

    # Simplify feature names.
    x_col_names = [short_label(f) for f in x_col_names]
    # Fall back to generic output names if needed.
    if not y_col_names or len(y_col_names) < n_outputs:
        y_col_names = [f"Output{i+1}" for i in range(n_outputs)]
    else:
        y_col_names = [short_label(y) for y in y_col_names]

    # ============ 1) Compute mean(|SHAP|) for each output ============
    # shap_values[i] shape = (n_samples, n_features)
    # mean_abs_shap[i, :] => shape=(n_features,)
    # Final shap_matrix shape = (n_features, n_outputs).
    n_features = len(x_col_names)
    shap_matrix = np.zeros((n_features, n_outputs), dtype=np.float64)

    for i in range(n_outputs):
        sv_i = np.asarray(shap_values[i])  # (n_samples, n_features)
        mean_abs_i = np.mean(np.abs(sv_i), axis=0)  # (n_features,)
        shap_matrix[:, i] = mean_abs_i

    # ============ 2) Sum rows and pick Top-N features ============
    sum_importances = np.sum(shap_matrix, axis=1)  # (n_features,)
    # Sort by descending total importance.
    sorted_idx = np.argsort(sum_importances)[::-1]
    top_idx = sorted_idx[:top_n_features]

    # ============ 3) Stacked horizontal bar chart ============
    # Extract the top features.
    top_features = [x_col_names[i] for i in top_idx]
    # shap_matrix_top shape = (top_n, n_outputs)
    shap_matrix_top = shap_matrix[top_idx, :]
    # sum_top shape = (top_n,)
    sum_top = sum_importances[top_idx]

    # top_idx is already sorted by descending total importance.

    fig, ax = plt.subplots(figsize=(plot_width, plot_height))

    # Draw stacked horizontal bars for each ranked feature.
    for rank in range(top_n_features):
        f_idx = top_idx[rank]
        # segments = shap_matrix_top[rank, :]
        segments = shap_matrix[f_idx, :]
        left_acc = 0.0
        for i in range(n_outputs):
            val_i = segments[i]
            ax.barh(
                y=rank,
                width=val_i,
                left=left_acc,
                color=color_palette[i],  # Publication-style palette.
                alpha=0.7
            )
            left_acc += val_i

    # Y ticks => top features.
    ax.set_yticks(np.arange(top_n_features))
    ax.set_yticklabels([top_features[i] for i in range(top_n_features)], fontsize=10)
    ax.invert_yaxis()

    ax.set_xlabel("Sum of mean(|SHAP|) across outputs", fontsize=12)
    # ax.set_title(f"Multi-output SHAP Feature Importance (Top-{top_n_features})", fontsize=14)

    # X axis spans the maximum summed importance.
    ax.set_xlim(0, np.max(sum_top)*1.05)

    # ============ 4) Legend: output name + matching color ============
    legend_patches = []
    for i in range(n_outputs):
        patch = Patch(facecolor=color_palette[i], label=y_col_names[i], alpha=0.7)
        legend_patches.append(patch)

    ax.legend(handles=legend_patches, loc="lower right", fontsize=10, frameon=False)

    # Styling.
    for spine in ax.spines.values():
        spine.set_visible(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=700)
    plt.close()
    print(f"[plot_shap_importance_multi_output] => {output_path}")


# ---------------------- Combined SHAP helper ---------------------
def plot_shap_combined(
    shap_data,
    output_path,
    top_n_features=15,
    plot_width=12,
    plot_height=8
):
    """
    Draw the following in one figure:
      - a shared left y-axis for features
      - a bottom x-axis beeswarm plot showing per-sample SHAP value distribution
      - a top x-axis horizontal bar chart of mean(|SHAP|), with threshold shading and a dashed line
      - a hidden right axis
    Output is saved as JPG at 700 dpi.

    Parameters
    ----------
    shap_data : dict
        Must contain:
          - "shap_values": np.array with shape (n_samples, n_features)
          - "X_full":      np.array with shape (n_samples, n_features)
          - "x_col_names": feature name list of length n_features
        Optional:
          - "y_col_names": handle multi-output externally; this example mainly targets single output
    output_path : str
        Output directory path
    top_n_features : int
        Draw only the top_n_features most important features
    plot_width, plot_height : float
        Figure width and height in inches
    """
    ensure_dir_for_file(os.path.join(output_path, "dummy.txt"))
    # Load data.
    shap_values = shap_data["shap_values"]
    X_full = shap_data["X_full"]
    x_col_names = shap_data["x_col_names"]

    # Simplify feature names.
    x_col_names = [short_label(col) for col in x_col_names]

    # Convert to a list if shap_values is not already one.
    if not isinstance(shap_values, list):
        shap_values = [shap_values]
        multi_output = False
    else:
        multi_output = True

    for idx, sv in enumerate(shap_values):
        sv_arr = np.asarray(sv)
        # -------------------------------
        # 1. Compute the mean absolute SHAP value for each feature
        # -------------------------------
        mean_abs_shap = np.squeeze(np.mean(np.abs(sv_arr), axis=0))  # (n_features,)
        sorted_idx = np.argsort(mean_abs_shap)[::-1]             # Descending order.
        top_idx = sorted_idx[:top_n_features]
        top_idx = [int(i) for i in top_idx]  # Convert to standard Python ints.

        top_imps = mean_abs_shap[top_idx]
        top_feats = [x_col_names[i] for i in top_idx]

        # Reorder SHAP values and data so the beeswarm uses the same feature order.
        sv_sorted = sv_arr[:, top_idx]
        X_sorted = X_full[:, top_idx]
        feat_sorted = top_feats[:]  # Copy.

        # Build an Explanation object (recommended in newer SHAP versions).
        explanation = shap.Explanation(
            values=sv_sorted,
            base_values=None,
            data=X_sorted,
            feature_names=feat_sorted
        )

        # Threshold and colors: blue above mean, red below or equal.
        threshold = top_imps.mean()
        colors = ["blue" if imp > threshold else "red" for imp in top_imps]

        # -------------------------------
        # 2. Create a single-axis base layout
        # -------------------------------
        fig, ax_bottom = plt.subplots(figsize=(plot_width, plot_height), dpi=700)
        try:
            plt.sca(ax_bottom)
            shap.summary_plot(
                sv_sorted,
                features=X_sorted,
                feature_names=feat_sorted,
                max_display=top_n_features,
                show=False,
                plot_size=None,
                plot_type="dot",
                sort=False
            )
        except TypeError:
            plt.sca(ax_bottom)
            shap.summary_plot(
                sv_sorted,
                features=X_sorted,
                feature_names=feat_sorted,
                max_display=top_n_features,
                show=False,
                plot_type="dot",
                sort=False
            )
        ax_bottom.set_xlabel("SHAP Value", fontsize=12)
        ax_bottom.spines['right'].set_visible(False)
        # Keep the beeswarm ordering consistent with highest-importance features on top.
        # ax_bottom.invert_yaxis()

        # -------------------------------
        # 3. Add a top x-axis with twiny and draw the bar chart
        # -------------------------------
        ax_top = ax_bottom.twiny()  # Share the y-axis.
        ax_top.set_ylim(ax_bottom.get_ylim())
        y_labels = [t.get_text() for t in ax_bottom.get_yticklabels()]
        y_ticks = ax_bottom.get_yticks()
        imp_map = {feat_sorted[i]: float(top_imps[i]) for i in range(len(top_imps))}
        bar_imps = [imp_map.get(lbl, 0.0) for lbl in y_labels]
        bar_colors = ["blue" if v > threshold else "red" for v in bar_imps]
        ax_top.barh(
            y=y_ticks,
            width=bar_imps,
            color=bar_colors,
            alpha=0.15,
            align='center'
        )
        ax_top.axvspan(0, threshold, facecolor='lightgray', alpha=0.3)
        ax_top.axvline(threshold, color='gray', linestyle='dashed', linewidth=2)
        ax_top.xaxis.set_label_position('top')
        ax_top.xaxis.tick_top()
        ax_top.set_xlabel("Feature Importance", fontsize=12)
        ax_top.set_yticks(y_ticks)
        ax_bottom.set_yticks(y_ticks)
        ax_bottom.set_yticklabels(y_labels)
        # Do not call invert_yaxis() on ax_top.

        # Add the legend based on bar colors.
        legend_e = [
            Patch(facecolor=to_rgba("blue", alpha=0.15), label="Above Mean"),
            Patch(facecolor=to_rgba("red", alpha=0.15), label="Below/Equal Mean")
        ]
        ax_top.legend(handles=legend_e, loc="lower right", fontsize=12)

        # -------------------------------
        # 4. Finalize and save
        # -------------------------------
        if multi_output:
            out_label = shap_data["y_col_names"][idx] if idx < len(shap_data["y_col_names"]) else f"Output{idx}"
            safe_out_label = safe_filename(out_label)
            out_file = f"shap_combined_{safe_out_label}.jpg"
            # fig.suptitle(f"SHAP Combined Plot - {out_label}", fontsize=16)
        else:
            out_file = "shap_combined.jpg"
            # fig.suptitle("SHAP Combined Plot", fontsize=16)

        plt.tight_layout(rect=(0, 0, 1, 0.95))
        save_path = os.path.join(output_path, out_file)
        plt.savefig(save_path, dpi=700, format='jpg')
        plt.close(fig)
        print(f"[INFO] SHAP combined figure saved => {save_path}")
# ============ Main functions ============
# ------------------------------------------------------------------
# -----------------------------------------------------------------
def plot_local_shap_force(shap_data, sample_index, output_path,
                          top_n_features=8, outputID=0,
                          pos_color="#d9534f", neg_color="#1766b5",
                          bar_height=1.3, dpi=700):
    """
    Pure‑Matplotlib local SHAP force‑plot:
    • Top‑N + Other, no feature names
    • Baseline & f(x) arrows
    • Arrow shapes follow shap/_force_matplotlib.py (v0.44)
    """

    # ========== 1. Data ==========
    sv = shap_data["shap_values"]
    vals = sv[outputID][sample_index] if isinstance(sv, list) else sv[sample_index]
    vals = np.nan_to_num(np.asarray(vals, float))

    bv = shap_data.get("base_values", 0.)
    base = float(bv[sample_index]) if isinstance(bv, (list, np.ndarray, pd.Series)) else float(bv)

    # ========== 2. Top-N + Other ==========
    idx_sorted = np.argsort(np.abs(vals))[::-1]
    top_idx, rest_idx = idx_sorted[:top_n_features], idx_sorted[top_n_features:]
    top_vals = vals[top_idx]
    if rest_idx.size:
        top_vals = np.append(top_vals, vals[rest_idx].sum())

    # Sort the top part again by |v| in descending order and keep "Other" last.
    if rest_idx.size:
        sort_part = np.argsort(np.abs(top_vals[:-1]))[::-1]
        top_vals = np.concatenate([top_vals[:-1][sort_part], top_vals[-1:]])

    pos_vals = top_vals[top_vals > 0]
    neg_vals = top_vals[top_vals < 0]

    # ========== 3. Arrow settings ==========
    total_neg, total_pos = neg_vals.sum(), pos_vals.sum()
    x_span = abs(total_neg) + total_pos
    head_len_const = max(x_span / 200.0, 0.02)   # pixels in data‑coords

    def head_len(v):
        """Avoid letting small arrows be fully occupied by the head; close to SHAP's default behavior."""
        h = min(abs(v) * 0.4, head_len_const)
        return max(h, 0.3 * abs(v))              # Keep it within <= 0.7 * |v|.

    # ========== 4. Plot ==========
    fig, ax = plt.subplots(figsize=(13, 1.8), dpi=dpi)

    # ---- 4A. Negative side: accumulate left from 0 (reverse order keeps bars near the baseline last) ----
    p = 0.0
    for v in sorted(neg_vals, key=abs):          # Small |v| to large |v|.
        h = head_len(v)
        # Rectangle.
        rect_w = abs(v) - h
        rect_start = p - rect_w
        ax.add_patch(Rectangle((rect_start, -bar_height/2),
                               rect_w, bar_height,
                               color=neg_color, lw=0))
        # Triangle.
        tri = [[rect_start,  bar_height/2],
               [rect_start - h, 0],
               [rect_start, -bar_height/2]]
        ax.add_patch(Polygon(tri, closed=True, color=neg_color, lw=0))
        # Text.
        center = p - abs(v)/2
        ax.text(center, 0, f"{v:+.2f}", color="white",
                ha="center", va="center", fontsize=8, rotation=90)
        p -= abs(v)                               # Update the pointer.

    # ---- 4B. Positive side: accumulate right from 0 ----
    p = 0.0
    for v in sorted(pos_vals, key=abs, reverse=True):  # Large |v| to small |v|.
        h = head_len(v)
        rect_w = v - h
        ax.add_patch(Rectangle((p, -bar_height/2),
                               rect_w, bar_height,
                               color=pos_color, lw=0))
        tri = [[p + rect_w,  bar_height/2],
               [p + rect_w + h, 0],
               [p + rect_w, -bar_height/2]]
        ax.add_patch(Polygon(tri, closed=True, color=pos_color, lw=0))
        center = p + v/2
        ax.text(center, 0, f"{v:+.2f}", color="white",
                ha="center", va="center", fontsize=8, rotation=90)
        p += v                                    # Update the pointer.

    # ========== 5. baseline & f(x) ==========
    ax.axvline(0, ls="--", lw=1.4, color="gray", zorder=0)
    fx = base + vals.sum()
    arrow_y = bar_height * 1.25
    ax.annotate("",
        xy=(fx, arrow_y), xytext=(0, arrow_y),
        arrowprops=dict(arrowstyle="<->", lw=1.5, color="black"))
    ax.text(0,  arrow_y + 0.12, f"base={base:.2f}",
            ha="left", va="bottom", fontsize=10, color="gray")
    ax.text(fx, arrow_y + 0.12, f"f(x)={fx:.2f}",
            ha="right", va="bottom", fontsize=10)

    # ========== 6. Axes and save ==========
    pad = 0.05 * x_span
    ax.set_xlim(total_neg - pad, total_pos + pad)
    ax.set_ylim(-1.2, 1.8)
    ax.set_yticks([])
    ax.set_xlabel("SHAP contribution (positive → right, negative → left)",
                  fontsize=11)
    # ax.set_title(f"Local SHAP Force Plot (sample {sample_index})",
    #              fontsize=14, color="#1f77b4", pad=6)
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(6))
    ax.grid(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Local SHAP force‑plot saved → {output_path}")


def plot_local_shap_lines(shap_data, sample_indices, output_path, top_n_features=8, link="identity", lineID=0, outputID=0):
    """
    Draw a local SHAP explanation plot in a decision-plot style for multiple samples.
    Multiple decision curves are shown in one figure to compare contribution paths,
    with an added "Other" contribution block such as "85 other features".
    "Other" is forced to stay last so it appears at the bottom of the final plot.
    """
    shap_values = shap_data["shap_values"]
    X_full = shap_data["X_full"]
    feature_names = shap_data["x_col_names"]
    # Simplify feature names.
    feature_names = [short_label(f) for f in feature_names]
    base_values = shap_data.get("base_values", None)

    if isinstance(shap_values, list):
        shap_value = shap_values[outputID]
    else:
        shap_value = shap_values

    # Collect SHAP values and input data for the selected samples.
    selected_shap = []
    selected_data = []
    for idx in sample_indices:
        s_shap = np.array(shap_value[idx], dtype=float)
        s_shap = np.nan_to_num(s_shap, nan=0.0)
        selected_shap.append(s_shap)
        s_data = X_full[idx]
        if isinstance(s_data, np.ndarray) and np.issubdtype(s_data.dtype, np.number):
            s_data = np.nan_to_num(s_data, nan=0.0)
        selected_data.append(s_data)
    selected_shap = np.array(selected_shap)  # shape: (n_samples, n_features)
    selected_data = np.array(selected_data)

    # Use the first sample to pick top_n_features by absolute SHAP value, excluding "Other".
    abs_shap = np.abs(selected_shap[0])
    sorted_idx = np.argsort(abs_shap)[::-1]    # Descending order: most important first.
    top_idx = sorted_idx[:top_n_features]
    # top_idx_sorted_desc keeps descending order, with the most important feature first.
    top_idx_sorted_desc = sorted(top_idx, key=lambda i: abs_shap[i], reverse=True)
    # Reverse the top-feature order so "Other" ends up at the bottom in the final decision plot.
    top_idx_sorted = list(reversed(top_idx_sorted_desc))

    # Extract top-feature contributions and data in low-to-high importance order.
    shap_values_top = selected_shap[:, top_idx_sorted]
    data_top = selected_data[:, top_idx_sorted]
    feature_names_top = [feature_names[i] for i in top_idx_sorted]

    # Compute the "Other" contribution as total contribution minus the top-feature contribution.
    others = np.array([np.sum(s) - np.sum(s[top_idx_sorted]) for s in selected_shap]).reshape(-1, 1)
    # Put the "Other" column first so that after the internal reversal it stays at the bottom.
    shap_values_top = np.hstack([others, shap_values_top])
    data_other = np.zeros((selected_data.shape[0], 1))
    data_top = np.hstack([data_other, data_top])
    # Count hidden features and build a label such as "85 other features".
    others_count = len(feature_names) - top_n_features
    feature_names_top = [f"{others_count} other features"] + feature_names_top

    if base_values is None:
        base_value = 0.0
    else:
        base_value = base_values[0] if isinstance(base_values, list) else base_values

    plt.figure(figsize=(12, 6))
    # Leave feature_order unset so decision_plot preserves the current order.
    shap.decision_plot(
        base_value=base_value,
        shap_values=shap_values_top,
        features=data_top,
        feature_names=feature_names_top,
        link=link,
        show=False,
        feature_order=None,
    )
    ax = plt.gca()
    ax.spines['left'].set_visible(True)
    ax.spines['right'].set_visible(True)

    # Emphasize the first line.
    lines = ax.get_lines()
    lineID += 9
    if lines and lineID < len(lines):
        lines[lineID].set_color('black')
        lines[lineID].set_linewidth(3.0)
        lines[lineID].set_linestyle('dashdot')

    # Hide text labels in the plot.
    for txt in ax.texts:
        txt.set_visible(False)

    # plt.title("Local SHAP (Decision Plot) for samples ", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=700)
    plt.close()
    print(f"[INFO] Local SHAP lines plot saved => {output_path}")
# ---------------------- SHAP heatmap helper ----------------------
# ---------------------- SHAP heatmap helper ----------------------
def plot_shap_heatmap_local(shap_data, output_path, sample_count=100, max_display=12, figsize=(14,8), outputID=0):
    """
    Draw a SHAP heatmap:
      - samples on the X axis
      - features on the Y axis
      - color encodes SHAP values
      - use the first sample_count samples by default
      - max_display controls the number of shown features
      - choose the output with outputID in multi-output cases
    Output is saved as JPG at 700 dpi.
    """
    # If shap_values is a list, select the requested output; otherwise use it directly.
    if isinstance(shap_data["shap_values"], list):
        heatmap_values = shap_data["shap_values"][outputID]
    else:
        heatmap_values = shap_data["shap_values"]

    # Keep only the first sample_count samples.
    heatmap_values = heatmap_values[:sample_count]

    # Simplify feature names.
    simplified_feature_names = [short_label(f) for f in shap_data["x_col_names"]]

    # Convert the NumPy array to a SHAP Explanation object.
    expl = shap.Explanation(values=heatmap_values,
                            feature_names=simplified_feature_names,
                            data=None)

    plt.figure(figsize=figsize, dpi=700)
    shap.plots.heatmap(expl, max_display=max_display, show=False, plot_width=figsize[0])
    plt.savefig(output_path, dpi=700, format='jpg')
    plt.close()
    print(f"[INFO] SHAP heatmap saved => {output_path}")

# --------------------------------------------------------------
def plot_cv_metrics(cv_metrics: dict,
                    save_name: str = "combined_cv_metrics.jpg",
                    show_label: bool = True):
    """
    4‑panel figure: 3 horizontal bar charts + 1 radar chart.

    show_label -> whether to show a./b./c./d. in the upper-left corner of each subplot.
    """

    ensure_dir_for_file(save_name)

    # ------------ Data ------------
    model_names = list(cv_metrics.keys())
    mse_vals = [cv_metrics[m]["MSE"] for m in model_names]
    mae_vals = [cv_metrics[m]["MAE"] for m in model_names]
    r2_vals  = [cv_metrics[m]["R2"]  for m in model_names]

    # (best, worst, ordinary) colours
    colors_mse = ("#2ca02c", "#d62728", "#1f77b4")
    colors_mae = ("#17becf", "#e377c2", "#bcbd22")
    colors_r2  = ("#ff7f0e", "#9467bd", "#8c564b")

    # ---------- Normalize for the radar chart ----------
    metrics = ["MSE", "MAE", "R2"]
    norm_data = {m: [] for m in model_names}
    for metric in metrics:
        col = [cv_metrics[m][metric] for m in model_names]
        mn, mx = min(col), max(col)
        span = mx - mn if mx != mn else 1
        for i, model in enumerate(model_names):
            norm_data[model].append((col[i] - mn) / span)

    # ---------- Figure & Grid ----------
    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 2, figure=fig,
                            left=0.08, right=0.96,
                            top=0.93, bottom=0.07,
                            wspace=0.25, hspace=0.25)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = cast(PolarAxes, fig.add_subplot(gs[1, 1], polar=True))

    subplot_font = {"size": 14}

    # ---------- Horizontal bar helper ----------
    def hbar_with_mean(ax, names, vals, label_char,
                       bigger_is_better, color_triplet):

        vals = np.asarray(vals)
        best = vals.argmax() if bigger_is_better else vals.argmin()
        worst = vals.argmin() if bigger_is_better else vals.argmax()

        bar_colors = [color_triplet[0] if i == best else
                      color_triplet[1] if i == worst else
                      color_triplet[2] for i in range(len(vals))]

        y = np.arange(len(vals))[::-1]
        ax.barh(y, vals[y], color=np.array(bar_colors)[y],
                height=0.4, alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels(np.array(names)[y], fontsize=12)
        ax.invert_yaxis()

        if show_label:
            ax.text(-0.08, 1.05, f"{label_char}.",
                    transform=ax.transAxes,
                    ha="left", va="top", fontdict=subplot_font)

        # Value annotations.
        right_lim = 1.5 * float(np.max(vals))           # Use a fixed 0-based axis up to 1.5 * max.
        shift = 0.03 * right_lim
        for xv, yv in zip(vals, y[::-1]):
            ax.text(xv + shift, yv, f"{xv:.2f}",
                    va="center", ha="left", fontsize=10)

        # Mean line and shaded region.
        m = vals.mean()
        ax.axvline(m, color="gray", ls="--", lw=2)
        ax.axvspan(0, m, color="gray", alpha=0.2)

        ax.set_xlim(0, right_lim)              # Use a unified 0-based axis.
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))

        legend_handles = [
            Patch(facecolor=color_triplet[0], label="Best"),
            Patch(facecolor=color_triplet[1], label="Worst"),
            Patch(facecolor=color_triplet[2], label="Ordinary"),
            Patch(facecolor="gray", alpha=0.2, label="Under Mean")
        ]
        ax.legend(handles=legend_handles, fontsize=9, loc="lower right")

    # ---------- Draw the three bar charts ----------
    hbar_with_mean(ax_a, model_names, mse_vals, "a", False, colors_mse)
    hbar_with_mean(ax_b, model_names, mae_vals, "b", False, colors_mae)
    hbar_with_mean(ax_c, model_names, r2_vals,  "c", True,  colors_r2)

    # ---------- Radar chart ----------
    N = len(metrics)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    radar_colors = ["#E64B35", "#4DBBD5", "#00A087",
                    "#3C5488", "#F39B7F", "#8491B4"]

    for idx, model in enumerate(model_names):
        vals = np.asarray(norm_data[model] + norm_data[model][:1], dtype=float)
        color = radar_colors[idx % len(radar_colors)]
        ax_d.plot(angles, vals, lw=2, color=color, label=model)
        ax_d.fill(angles, vals, color=color, alpha=0.25)

    ax_d.set_thetagrids(np.degrees(angles[:-1]), metrics, fontsize=12)
    if show_label:
        ax_d.text(-0.12, 1.1, "d.", transform=ax_d.transAxes,
                  ha="left", va="top", fontdict=subplot_font)

    ax_d.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02),
                ncol=len(model_names), frameon=False,
                prop={"size": 11})

    # ---------- Save ----------
    plt.savefig(save_name, dpi=700)
    plt.close()
    print(f"[plot_cv_metrics_combined] → {save_name}")

# =============================================================
#  K-Fold cross-validation boxplot
# =============================================================
def plot_cv_boxplot(
    cv_metrics: dict,
    metric: str = "MSE",
    save_name: str = "cv_boxplot_MSE.jpg",
    show_overfit: bool = True
):
    """
    Enhanced box‑plot for 5‑fold CV.

    • metric  : "MSE" | "MAE" | "R2"
    • If show_overfit=True, also draw the overfitting metric on the right twin axis
      ─ "MSE_ratio"  = Val_MSE / Train_MSE
      ─ "R2_diff"    = Train_R2 − Val_R2
    """

    ensure_dir_for_file(save_name)
    model_names, data_train, data_val, ovf_vals = [], [], [], []

    # -------- 1. Collect data --------
    for m, rec in cv_metrics.items():
        folds = rec.get("folds", {})
        tr_key, va_key = f"{metric}_train", f"{metric}_val"

        if not (tr_key in folds and va_key in folds):
            continue
        tr, va = folds[tr_key], folds[va_key]
        if len(tr) != len(va):
            continue

        model_names.append(m)
        data_train.append(tr)
        data_val.append(va)

        if show_overfit:
            if metric == "MSE":
                ovf = folds.get("MSE_ratio", [])
            elif metric == "R2":
                ovf = folds.get("R2_diff", [])
            else:
                ovf = []
            ovf_vals.append(ovf)

    if not model_names:
        print("[plot_cv_boxplot] – no valid data.")
        return

    # -------- 2. Parameters --------
    n_models   = len(model_names)
    box_width  = 0.25
    group_gap  = 1.0
    color_train, color_val = "#0072B2", "#D55E00"

    # -------- 3. Draw the main axis --------
    fig, ax = plt.subplots(figsize=(1.5 * n_models, 6))

    positions_tr = [i * group_gap - box_width / 2 for i in range(n_models)]
    positions_va = [i * group_gap + box_width / 2 for i in range(n_models)]

    bp_tr = ax.boxplot(
        data_train, positions=positions_tr, widths=box_width,
        patch_artist=True, showfliers=False
    )
    bp_va = ax.boxplot(
        data_val,   positions=positions_va, widths=box_width,
        patch_artist=True, showfliers=False
    )

    # Color the boxes.
    for p in bp_tr["boxes"]:
        p.set_facecolor(color_train)
        p.set_alpha(0.55)
    for p in bp_va["boxes"]:
        p.set_facecolor(color_val)
        p.set_alpha(0.55)

    plt.setp(bp_tr["medians"], color="black", linewidth=2)
    plt.setp(bp_va["medians"], color="black", linewidth=2)

    # Mean markers.
    for pos, vals in zip(positions_tr, data_train):
        ax.scatter(pos, np.mean(vals), marker="o", color=color_train, s=65, zorder=3)
    for pos, vals in zip(positions_va, data_val):
        ax.scatter(pos, np.mean(vals), marker="o", color=color_val,   s=65, zorder=3)

    # Jittered scatter points.
    rng = np.random.default_rng(0)
    for pos_c, dlist, col in [(positions_tr, data_train, color_train),
                              (positions_va, data_val,   color_val)]:
        for p, series in zip(pos_c, dlist):
            jitter = (rng.random(len(series)) - 0.5) * box_width * 0.5
            ax.scatter(p + jitter, series, color=col, alpha=0.3, s=28, zorder=2)

    # x‑tick
    ax.set_xticks([i * group_gap for i in range(n_models)])
    ax.set_xticklabels(model_names, rotation=30, ha="right",
                       fontsize=11)

    # -------- 4. Left y-axis --------
    ax.set_ylabel(metric, fontsize=13)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    y_vals_flat = [*sum(data_train, []), *sum(data_val, [])]  # Flatten to a single list.
    y_min_main = min(y_vals_flat)
    y_max_main = max(y_vals_flat)
    span_main = y_max_main - y_min_main
    ax.set_ylim(y_min_main - 0.10 * span_main,
                y_max_main + 0.10 * span_main)  # Add 10% padding above and below.

    # Legend
    ax.scatter([], [], color=color_train, label="Train",      s=65)
    ax.scatter([], [], color=color_val,   label="Validation", s=65)
    ax.legend(frameon=False, loc="upper left")

    # -------- 5. Right-side overfitting metric --------
    if show_overfit and metric in ("MSE", "R2") and ovf_vals:
        ax2 = ax.twinx()
        mean_ovf = [np.mean(v) if v else np.nan for v in ovf_vals]

        ax2.plot([i * group_gap for i in range(n_models)],
                 mean_ovf, marker="^", markersize=8,
                 linewidth=2, color="purple", label="Over-fit")

        # Y label.
        label_text = "MSE ratio (Val/Train)" if metric == "MSE" \
                     else "R$^2$ diff (Train - Val)"
        ax2.set_ylabel(label_text)

        # Keep one decimal place here as well.
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        y_min_ovf, y_max_ovf = min(mean_ovf), max(mean_ovf)
        span_ovf = y_max_ovf - y_min_ovf
        ax2.set_ylim(y_min_ovf - 0.10 * span_ovf,
                     y_max_ovf + 0.10 * span_ovf)

        ax2.grid(False)
        ax2.legend(frameon=False, loc="upper right")

    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(save_name, dpi=700)
    plt.close()
    print(f"[plot_cv_boxplot] → {save_name}")

def plot_overfitting_horizontal(overfit_data,
                                save_name="overfitting_horizontal.jpg"):
    """
    Draw two horizontal lollipop plots to compare overfitting metrics:
      • MSE_ratio (Val / Train) - lower is better
      • R2_diff (Train - Val) - lower is better
    Colors and shaded regions match the original bar-chart version,
    but bars are replaced by lines and dots.
    """

    # Nature Reviews style base colors (colorblind-friendly).
    NATURE_RED   = "#D55E00"
    NATURE_BLUE  = "#0072B2"
    NATURE_GREEN = "#009E73"
    GRAY         = "#BEBEBE"
    LIGHT_RED    = "#E69F9F"  # Slightly darker, still colorblind-friendly.

    ensure_dir_for_file(save_name)
    model_names = list(overfit_data.keys())
    msr_vals = [overfit_data[m]["MSE_ratio"] for m in model_names]
    r2d_vals = [overfit_data[m]["R2_diff"] for m in model_names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --------------------------------------------------------------
    def lollipop(ax, names, vals, metric_label,
                 threshold_h, threshold_l):
        vals = np.asarray(vals)
        best, worst = vals.argmin(), vals.argmax()

        # Per-model colors.
        colors = [NATURE_RED if i == best else
                  NATURE_BLUE if i == worst else
                  NATURE_GREEN for i in range(len(vals))]

        y_pos = np.arange(len(vals))[::-1]  # Reverse so the best appears at the top.

        # ------- Line segments + dots -------
        for x, y, c in zip(vals[y_pos], y_pos, np.array(colors)[y_pos]):
            ax.hlines(y, 0, x, color=c, lw=3, alpha=0.9, zorder=1)  # Match the line color to the dot.
            ax.scatter(x, y, s=180, color=c, edgecolors="k", zorder=2)

        # Axes and labels.
        ax.set_yticks(y_pos)
        ax.set_yticklabels(np.array(names)[y_pos])
        ax.invert_yaxis()
        ax.set_xlabel(metric_label, fontsize=12)

        # -------- Value annotations --------
        # Ensure the x-axis upper bound is at least 1.25 * threshold_h.
        value_lim = max(1.25 * threshold_h, 1.8 * float(np.max(vals)))
        shift = 0.03 * value_lim
        for xv, yv in zip(vals, y_pos[::-1]):
            ax.text(xv + shift, yv, f"{xv:.2f}",
                    va="center", ha="left",
                    fontsize=12)

        # -------- Threshold shading --------
        zones: list[Artist]
        if threshold_l == 0:
            ax.axvspan(0, threshold_h, color=GRAY, alpha=0.2, zorder=0)
            zones = [Patch(facecolor=GRAY, alpha=0.2, label="Acceptable")]
        else:
            ax.axvspan(0, threshold_l, color=GRAY, alpha=0.2, zorder=0)
            ax.axvspan(threshold_l, threshold_h,
                       color=LIGHT_RED, alpha=0.3, zorder=0)
            ax.axvline(threshold_l, color=GRAY, ls="--", lw=1.8)
            ax.axvline(threshold_h, color=GRAY, ls="--", lw=1.8)
            zones = [
                Patch(facecolor=GRAY, alpha=0.2, label="Acceptable"),
                Patch(facecolor=LIGHT_RED, alpha=0.3, label="Overfitting Risk")
            ]

        # -------- Axis range and ticks --------
        ax.set_xlim(0, value_lim)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))

        # -------- Legend --------
        zones.extend([
            Line2D([], [], marker='o', color='w',
                   markerfacecolor=NATURE_RED, markersize=10, label="Best"),
            Line2D([], [], marker='o', color='w',
                   markerfacecolor=NATURE_BLUE, markersize=10, label="Worst"),
            Line2D([], [], marker='o', color='w',
                   markerfacecolor=NATURE_GREEN, markersize=10, label="Ordinary")
        ])
        ax.legend(handles=zones, loc="lower right", fontsize=9)

    # Left: MSE Ratio.
    lollipop(axes[0], model_names, msr_vals,
             "MSE Ratio (Val / Train)",
             threshold_h=10, threshold_l=5)

    # Right: R² diff.
    lollipop(axes[1], model_names, r2d_vals,
             "R$^2$ difference (Train - Val)",
             threshold_h=0.20, threshold_l=0.15)

    plt.tight_layout()
    plt.savefig(save_name, dpi=700)
    plt.close()
    print(f"[plot_overfitting_horizontal] → {save_name}")


class OnlyPositiveNoZeroLocator(ticker.Locator):
    """
    Generate nbins ticks only within [0, vmax].
    If vmax <= 0, generate no ticks.
    Skip 0 so it does not appear as a tick.
    """
    def __init__(self, nbins=6):
        self.nbins = nbins
    def __call__(self):
        assert self.axis is not None
        vmin, vmax = self.axis.get_data_interval()
        lower = max(0, min(vmin, vmax))
        upper = max(vmax, lower)
        if upper <= 0:
            return []
        ticks = np.linspace(lower, upper, self.nbins)
        # Filter out 0.
        filtered = [t for t in ticks if t>1e-9]
        return filtered
    def tick_values(self, vmin, vmax):
        return self.__call__()

class OnlyPositiveIntegerLocator(ticker.Locator):
    """
    Generate nbins integer ticks only within [1, floor(vmax)], skipping 0 and negatives.
    """
    def __init__(self, nbins=4):
        self.nbins = nbins
    def __call__(self):
        assert self.axis is not None
        vmin, vmax = self.axis.get_data_interval()
        lower = max(5, int(np.ceil(vmin)))
        upper = int(np.floor(vmax))
        if upper<1:
            return []
        ticks = np.linspace(lower, upper, self.nbins)
        ticks = [int(round(t)) for t in ticks]
        ticks = sorted(set(ticks))
        return ticks
    def tick_values(self, vmin, vmax):
        return self.__call__()

class NoSciNoOffsetFormatter(ticker.ScalarFormatter):
    """
    Disable scientific notation and offset; always keep two decimals with '%.2f'.
    """
    def __init__(self, decimals=2, useMathText=False):
        super().__init__(useMathText=useMathText)
        self.decimals = decimals
        self.set_scientific(False)
        self.set_useOffset(False)
    def _set_format(self):
        self.format = f'%.{self.decimals}f'

class TwoSigFigSciFormatter(ticker.ScalarFormatter):
    """Two significant digits with MathText scientific notation (×10^n)."""
    def __init__(self, **kwargs):
        super().__init__(useMathText=True, **kwargs)
        # Always use scientific notation and move the factor to the upper-left of the axis.
        self.set_scientific(True)
        self.set_powerlimits((0, 0))   # Enable offset for all ranges.

    def _set_format(self):
        # Matplotlib calls this inside set_locs() without arguments.
        self.format = "%1.2g"          # Two significant digits.

def plot_multi_model_residual_distribution_single_dim(
    residuals_dict,
    out_label="Output",
    bins=6,
    filename="multi_model_residual_dual_axis.jpg",
    rug_negative_space=0.15,
    show_zero_line_arrow=True
):
    """
    A simple approach:
      - left axis: show only positive integers (>= 1), nbins = 5, skip 0 and negatives
      - right axis: show only positive floats (> 0), keep two decimals, no scientific notation or offset
      - y < 0 is reserved for rug plots, with no negative tick labels
      - all other logic (histogram, KDE, rug, legend, etc.) stays the same
    """

    if not residuals_dict:
        print("[plot_multi_model_residual_distribution_single_dim] => empty dict, skip.")
        return
    residuals_dict = {m: np.asarray(residuals_dict[m]) for m in residuals_dict}

    # Collect data.
    all_data_list = []
    for m in residuals_dict:
        arr = np.asarray(residuals_dict[m])
        if arr.size==0:
            warnings.warn(f"Model {m} has empty residual array.")
        else:
            all_data_list.append(arr)
    if not all_data_list:
        print("[plot_multi_model_residual_distribution_single_dim] => no valid data, skip.")
        return

    all_data = np.concatenate(all_data_list)
    data_min, data_max = np.min(all_data), np.max(all_data)
    if data_min==data_max:
        data_min -= 1e-6
        data_max += 1e-6

    # Use a symmetric range => [-R, R].
    R = max(abs(data_min), abs(data_max))
    range_left, range_right = -R, R

    bin_width = (range_right - range_left)/bins
    base_edges = np.linspace(range_left, range_right, bins+1)
    edges = base_edges + bin_width/2

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.rcParams["font.size"] = 16  # Font setting.
    fig, ax = plt.subplots(figsize=(5, 4))  # Adjust the canvas size.
    ax2 = ax.twinx()

    model_names = list(residuals_dict.keys())
    n_models = len(model_names)
    color_palette = ["#24345C", "#279DE1", "#329845", "#8A233F", "#912C2C"]
    if n_models > len(color_palette):
        color_palette = color_palette * (n_models // len(color_palette) + 1)

    # ========== Left axis: histogram (count) ==========
    hist_counts_dict = {}
    max_count = 0
    for m in model_names:
        arr = np.asarray(residuals_dict[m])
        counts, _ = np.histogram(arr, bins=edges)
        hist_counts_dict[m] = counts
        cmax = int(np.max(counts))
        if cmax>max_count:
            max_count=cmax

    group_width = bin_width*0.9
    bar_width = group_width/n_models
    gap = bin_width-group_width

    for b_idx in range(len(edges)-1):
        x_left_bin = edges[b_idx]
        for i,m in enumerate(model_names):
            c = hist_counts_dict[m][b_idx]
            color_ = color_palette[i]
            rect_left = x_left_bin + 0.5*gap + i*bar_width
            ax.bar(
                rect_left, c,
                width=bar_width, bottom=0,
                color=color_, alpha=0.6,
                align="edge", edgecolor='none'
            )

    ax.set_xlim(range_left, range_right)
    ax.set_ylim(-rug_negative_space*max_count, max_count*1.1)
    ax.set_xlabel("Residual", fontsize=16)
    ax.set_ylabel("Count", fontsize=16)

    # # Left axis => positive integers only => OnlyPositiveIntegerLocator
    # ax.yaxis.set_major_locator(OnlyPositiveIntegerLocator(nbins=5))
    # # Hide offset text
    # ax.yaxis.get_offset_text().set_visible(False)
    # 1) Show only positive ticks (up to 5), excluding 0.
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='lower'))

    # 2) Use two significant digits with scientific notation.
    ax.yaxis.set_major_formatter(TwoSigFigSciFormatter())

    # 3) Optionally match the offset text size to the overall font size.
    ax.yaxis.get_offset_text().set_fontsize(14)

    # ========== Right axis: KDE + rug ==========
    max_density = 0
    for i,m in enumerate(model_names):
        arr = np.asarray(residuals_dict[m])
        color_ = color_palette[i]
        # KDE.
        kde_obj = sns.kdeplot(
            arr, ax=ax2, color=color_,
            fill=False, alpha=0.9,
            linewidth=2, bw_adjust=0.8
        )
        if kde_obj.lines:
            ydata = np.asarray(kde_obj.lines[-1].get_ydata())
            if ydata.size > 0:
                cur_max = float(np.max(ydata))
                if cur_max>max_density:
                    max_density=cur_max

        # Rug plot.
        sns.rugplot(
            arr, ax=ax2,
            height=0.1, color=color_,
            alpha=0.4, lw=1, clip_on=False
        )

    ax2.set_ylim(-rug_negative_space*max_density, max_density*1.2)
    ax2.set_ylabel("Density", fontsize=12)

    # Draw a line at y = 0 on the right axis.
    ax2.axhline(0, color='k', linewidth=1.5, zorder=2)
    ax2.spines["bottom"].set_visible(False)
    ax2.set_axisbelow(False)

    # Right axis => positive floats only => 5 ticks.
    ax2.yaxis.set_major_locator(OnlyPositiveNoZeroLocator(nbins=6))
    # Two decimals => custom NoSciNoOffsetFormatter.
    no_sci_fmt = NoSciNoOffsetFormatter(decimals=2, useMathText=False)
    ax2.yaxis.set_major_formatter(no_sci_fmt)

    # Vertical line at x = 0.
    ax.axvline(0, color='k', linestyle='--', linewidth=1)
    if show_zero_line_arrow and (range_left<0<range_right):
        arrow_y = max_density*0.6
        ax2.annotate(
            "Zero line",
            xy=(0, arrow_y),
            xytext=(0, max_density*1.05),
            ha="center",
            arrowprops=dict(arrowstyle="->", color='k')
        )

    # Legend (histogram style).
    legend_patches = []
    for i,m in enumerate(model_names):
        patch = mpatches.Patch(
            facecolor=color_palette[i],
            edgecolor='none',
            alpha=0.6,
            label=m
        )
        legend_patches.append(patch)
    ax.legend(handles=legend_patches, loc="upper right", fontsize=10)

    # ax.set_title(out_label, fontsize=13)

    # Show borders.
    for spine in ax.spines.values():
        spine.set_visible(True)
    for spine in ax2.spines.values():
        spine.set_visible(True)

    ax.grid(False)
    ax2.grid(False)

    plt.tight_layout()
    plt.savefig(filename, dpi=700)
    plt.close()
    print(f"[plot_multi_model_residual_distribution_single_dim] => {filename}")


def plot_optuna_tuning_curve(trials_df, out_path):
    """
    Draw the optimization history with matplotlib.
    The figure uses a black-and-white style, with only font size and weight adjusted.
    trials_df must contain at least a "value" column.
    If a "number" column exists, use it as the iteration index; otherwise use the DataFrame index.
    """

    # Extract X data (iteration index) and Y data (objective value).
    if "number" in trials_df.columns:
        x = trials_df["number"]
    else:
        x = trials_df.index
    if "value" in trials_df.columns:
        y = trials_df["value"]
    else:
        raise ValueError("trials_df must contain a 'value' column")

    plt.figure(figsize=(8, 6))
    plt.plot(x, y, marker='o', linewidth=3, markersize=8, color="#4DBBD5")
    plt.xlabel("Iteration", fontsize=14)
    plt.ylabel("Objective Value", fontsize=14)
    # plt.title("Optimization History", fontsize=18)
    plt.tight_layout()
    plt.savefig(out_path, dpi=700)
    plt.close()
    print(f"[INFO] Custom styled Optuna Optimization History saved => {out_path}")


def plot_optuna_summary_curve(trials_dict, out_path):
    """
    Draw a summary plot for all models in trials_dict.
    Each model's tuning history is shown, and the best result
    (minimum objective value) is highlighted with a star marker.

    Parameters:
      trials_dict: dict whose keys are model names and values are trials DataFrames
      out_path: output file path for the generated figure
    """
    colors = [
        "#1f77b4",  # Blue
        "#2ca02c",  # Green
        "#d62728",  # Red
        "#9467bd",  # Purple
        "#8c564b",  # Brown
        "#17becf"   # Cyan
    ]
    plt.figure(figsize=(8, 6))

    for i, (mtype, trials_df) in enumerate(trials_dict.items()):
        # Use the "number" column as iteration index when available; otherwise use the DataFrame index.
        if "number" in trials_df.columns:
            x = trials_df["number"]
        else:
            x = trials_df.index
        if "value" not in trials_df.columns:
            raise ValueError("trials_df must contain a 'value' column")
        y = trials_df["value"]
        color = colors[i % len(colors)]

        # Plot the tuning history with transparency.
        plt.plot(x, y, marker='o', linewidth=3, markersize=5,
                 color=color, alpha=0.15, label=f"{mtype} History")

        # Mark the best point (minimum objective value) with an opaque star.
        best_idx = y.idxmin()
        if "number" in trials_df.columns:
            best_x = trials_df.loc[best_idx, "number"]
        else:
            best_x = best_idx
        best_y = float(np.min(y))
        plt.plot(best_x, best_y, marker='*', markersize=14,
                 color=color, linestyle='None', label=f"{mtype} Best")

    plt.xlabel("Iteration", fontsize=14)
    plt.ylabel("Objective Value", fontsize=14)
    # plt.title("Optimization Summary for All Models", fontsize=18)

    # Place the legend above the plot in a single row.
    plt.legend(bbox_to_anchor=(0.5, 1.12), loc='upper center',
               ncol=len(trials_dict), fontsize=8, frameon=False)

    # Reserve space for the legend.
    plt.tight_layout(rect=(0, 0, 1, 0.13))
    plt.savefig(out_path, dpi=700)
    plt.close()
    print(f"[INFO] Custom styled Optuna Summary Curve saved => {out_path}")


def plot_optuna_slice(trials_df, params, out_path):
    """
    Draw parameter slice plots with matplotlib and seaborn.
    The style stays black-and-white, with only font size and weight adjusted.
    trials_df must contain a "value" column and every parameter listed in params.
    """
    import seaborn as sns

    n_params = len(params)
    fig, axes = plt.subplots(n_params, 1, figsize=(8, 4 * n_params), sharey=True)
    if n_params == 1:
        axes = [axes]

    for ax, param in zip(axes, params):
        if param not in trials_df.columns:
            ax.text(0.5, 0.5, f"Column '{param}' not found", ha='center', va='center', fontsize=14)
            continue
        x = trials_df[param]
        y = trials_df["value"]
        # Draw the scatter plot with seaborn (default black style).
        sns.scatterplot(x=x, y=y, ax=ax, color="black", s=50, edgecolor="black", alpha=0.7)
        # Use the actual parameter name as the x-axis label.
        ax.set_xlabel(param, fontsize=14, color="black")
        ax.set_ylabel("Objective Value", fontsize=14, color="black")
        ax.tick_params(labelsize=12, colors="black")
    # fig.suptitle("Parameter Slice Plot", fontsize=18, color="black")
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(out_path, dpi=700)
    plt.close()
    print(f"[INFO] Custom styled Optuna Slice Plot saved => {out_path}")


def plot_optuna_param_importances(trials_df, out_path):
    """
    Draw a horizontal bar chart of parameter importances with matplotlib.
    Importance is computed as the absolute correlation between each parameter column and "value".
    The chart uses a blue-only palette, with larger bold fonts and no other style changes.
    This assumes trials_df has already been renamed so parameter columns no longer have
    the "params_" prefix, therefore built-in fields are excluded explicitly.
    """

    # Built-in fields to exclude.
    exclude = {'number', 'value', 'datetime_start', 'datetime_complete', 'duration', 'state'}

    # Select all parameter columns not in the exclude set.
    param_cols = [col for col in trials_df.columns if col not in exclude]
    if not param_cols:
        print("[WARN] No parameter columns found!")
        return

    objective = trials_df["value"].to_numpy(dtype=float)
    importances = {}
    for col in param_cols:
        data = trials_df[col]
        try:
            # First try converting the data to float.
            data_numeric = data.to_numpy(dtype=float)
        except Exception:
            data_numeric = None

        corr = 0.0
        try:
            if data_numeric is not None:
                valid_mask = ~np.isnan(data_numeric) & ~np.isnan(objective)
                if valid_mask.sum() < 2:
                    corr = 0.0
                else:
                    corr = np.corrcoef(data_numeric[valid_mask], objective[valid_mask])[0, 1]
            else:
                raise ValueError
            if np.isnan(corr):
                corr = 0.0
        except Exception:
            try:
                data_cat = data.astype("category").cat.codes.to_numpy(dtype=float)
                valid_mask = ~np.isnan(data_cat) & ~np.isnan(objective)
                if valid_mask.sum() < 2:
                    corr = 0.0
                else:
                    corr = np.corrcoef(data_cat[valid_mask], objective[valid_mask])[0, 1]
                if np.isnan(corr):
                    corr = 0.0
            except Exception:
                corr = 0.0
        importances[col] = abs(corr)

    # Sort in descending order.
    param_names = list(importances.keys())
    imp_values = [importances[name] for name in param_names]
    sorted_idx = np.argsort(imp_values)[::-1]
    param_names_sorted = [param_names[i] for i in sorted_idx]
    imp_values_sorted = [imp_values[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(8, 6))
    y_pos = np.arange(len(param_names_sorted))
    # Use blue for all bars.
    ax.barh(y_pos, imp_values_sorted, color="blue", edgecolor="black", height=0.6)
    # Show the original column names directly.
    ax.set_yticks(y_pos)
    ax.set_yticklabels(param_names_sorted, fontsize=12, color="black")
    ax.set_xlabel("Importance Score", fontsize=14, color="black")
    # ax.set_title("Parameter Importances", fontsize=18, color="black")
    ax.invert_yaxis()  # Highest importance on top.
    plt.tight_layout()
    plt.savefig(out_path, dpi=700)
    plt.close()
    print(f"[INFO] Custom styled Optuna Parameter Importances Plot saved => {out_path}")

# inference
def _upsample_grid(grid_x, grid_y, Z, smooth=4, order=3):
    """
    Use scipy.ndimage.zoom to apply bicubic interpolation to a regular grid
    (grid_x, grid_y, Z) for a smoother surface. smooth=4 means both rows and
    columns are subdivided by a factor of 4.
    """
    grid_x = np.asarray(grid_x)
    grid_y = np.asarray(grid_y)
    Z = np.asarray(Z)
    if smooth <= 1:
        return grid_x, grid_y, Z                # No interpolation.

    # Original grid size.
    H, W = Z.shape
    zoom_factor = (smooth, smooth)              # (y, x) directions.

    # Interpolate Z.
    Z_fine = zoom(Z, zoom_factor, order=order)

    # Generate the matching evenly spaced grid coordinates.
    x_min, x_max = float(np.min(grid_x)), float(np.max(grid_x))
    y_min, y_max = float(np.min(grid_y)), float(np.max(grid_y))
    x_vals = np.linspace(x_min, x_max, W * smooth)
    y_vals = np.linspace(y_min, y_max, H * smooth)
    grid_x_fine, grid_y_fine = np.meshgrid(x_vals, y_vals)

    return grid_x_fine, grid_y_fine, Z_fine


# ===============================================================
# 1) 2-D heatmap (smoothed)
# ===============================================================
def plot_2d_heatmap_from_npy(grid_x, grid_y, heatmap_pred,
                             out_dir,
                             x_label="X-axis",
                             y_label="Y-axis",
                             y_col_names=None,
                             stats_dict=None,
                             colorbar_extend_ratio=0.25,
                             smooth=4):          # New parameter, default x4 smoothing.
    """
    smooth: when >= 2, use bicubic interpolation to densify the grid for a smoother look;
            use 1 to keep the original resolution.
    """
    os.makedirs(out_dir, exist_ok=True)
    grid_x = np.asarray(grid_x)
    grid_y = np.asarray(grid_y)
    heatmap_pred = np.asarray(heatmap_pred)
    _, _, out_dim = heatmap_pred.shape

    for odx in range(out_dim):
        # ---------- 1) Interpolation ----------
        gx_f, gy_f, z_f = _upsample_grid(grid_x,
                                         grid_y,
                                         heatmap_pred[:, :, odx],
                                         smooth=smooth,
                                         order=3)   # Bicubic interpolation.
        gx_f = np.asarray(gx_f)
        gy_f = np.asarray(gy_f)
        z_f = np.asarray(z_f)

        auto_min, auto_max = float(np.min(z_f)), float(np.max(z_f))
        if stats_dict and y_col_names and odx < len(y_col_names) \
           and y_col_names[odx] in stats_dict:
            real_min = stats_dict[y_col_names[odx]]["min"]
            real_max = stats_dict[y_col_names[odx]]["max"]
            vmin_ = max(0, real_min * (1 - colorbar_extend_ratio))
            vmax_ = real_max * (1 + colorbar_extend_ratio)
        else:
            vmin_, vmax_ = auto_min, auto_max

        norm_ = mcolors.Normalize(vmin=vmin_, vmax=vmax_)

        plt.figure(figsize=(6, 5))
        # shading='gouraud' plus a finer grid creates a smoother color transition.
        cm_ = plt.pcolormesh(gx_f, gy_f, z_f,
                             shading="gouraud",
                             cmap="GnBu",
                             norm=norm_)
        cb_ = plt.colorbar(cm_)
        cb_.set_label(y_col_names[odx] if y_col_names and odx < len(y_col_names)
                      else f"Output_{odx}",
                      fontsize=12)
        cb_.locator = MaxNLocator(nbins=5)
        cb_.update_ticks()

        ax = plt.gca()
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.3g"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3g"))

        out_jpg = os.path.join(out_dir, f"heatmap_output_{odx + 1}.jpg")
        plt.savefig(out_jpg, dpi=700, bbox_inches="tight")
        plt.close()
        print(f"[INFO] 2D Heatmap saved → {out_jpg}")


# ===============================================================
# 2) 3-D surface (smoothed)
# ===============================================================
def plot_3d_surface_from_heatmap(grid_x, grid_y, heatmap_pred,
                                 out_dir,
                                 x_label="X-axis",
                                 y_label="Y-axis",
                                 y_col_names=None,
                                 stats_dict=None,
                                 colorbar_extend_ratio=0.25,
                                 cmap_name="GnBu",
                                 smooth=4):          # New parameter.
    os.makedirs(out_dir, exist_ok=True)
    heatmap_pred = np.asarray(heatmap_pred)
    _, _, out_dim = heatmap_pred.shape

    for odx in range(out_dim):
        # ---------- 1) Interpolation ----------
        gx_f, gy_f, Z_f = _upsample_grid(grid_x,
                                         grid_y,
                                         heatmap_pred[:, :, odx],
                                         smooth=smooth,
                                         order=3)
        gx_f = np.asarray(gx_f)
        gy_f = np.asarray(gy_f)
        Z_f = np.asarray(Z_f)

        auto_min, auto_max = float(np.min(Z_f)), float(np.max(Z_f))
        if stats_dict and y_col_names and odx < len(y_col_names) \
           and y_col_names[odx] in stats_dict:
            real_min = stats_dict[y_col_names[odx]]["min"]
            real_max = stats_dict[y_col_names[odx]]["max"]
            vmin_ = max(0, real_min * (1 - colorbar_extend_ratio))
            vmax_ = real_max * (1 + colorbar_extend_ratio)
        else:
            vmin_, vmax_ = auto_min, auto_max

        norm_ = mcolors.Normalize(vmin=vmin_, vmax=vmax_)
        cmap_ = plt.get_cmap(cmap_name)
        colors_rgba = np.asarray(cmap_(norm_(np.ravel(Z_f)))).reshape(
            (Z_f.shape[0], Z_f.shape[1], 4)
        )

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        ax.plot_surface(gx_f, gy_f, Z_f,
                        facecolors=colors_rgba,
                        rstride=1, cstride=1,
                        linewidth=0,
                        antialiased=True,   # Anti-aliasing.
                        shade=False)

        sm = cm.ScalarMappable(norm=norm_, cmap=cmap_)
        sm.set_array([])
        cb = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.1, aspect=15)
        cb.set_label(y_col_names[odx] if y_col_names and odx < len(y_col_names)
                     else f"Output_{odx}",
                     fontsize=12)
        cb.locator = MaxNLocator(nbins=5)
        cb.update_ticks()

        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_zlabel("Value",  fontsize=12)

        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.zaxis.set_major_locator(MaxNLocator(nbins=5))

        ax.xaxis.set_major_formatter(FormatStrFormatter("%.3g"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3g"))
        ax.zaxis.set_major_formatter(FormatStrFormatter("%.3g"))
        ax.grid(False)

        out_jpg = os.path.join(out_dir, f"heatmap_3d_surface_output_{odx + 1}.jpg")
        plt.savefig(out_jpg, dpi=700, bbox_inches="tight")
        plt.close()
        print(f"[INFO] 3D Surface saved → {out_jpg}")


# ────────────────────────── Helper functions ────────────────────────── #
def _prep_labels(labels):            # Trim labels.
    return [short_label(l) for l in labels]

def _draw_grid(ax, n_rows, n_cols, cell):
    for rr in range(n_rows + 1):
        ax.axhline(rr * cell, color='black', linewidth=1)
    for cc in range(n_cols + 1):
        ax.axvline(cc * cell, color='black', linewidth=1)

def _set_axes(ax, n_rows, n_cols, cell, row_lbls, col_lbls,
              row_axis_name, col_axis_name):
    ax.set_xlim(0, n_cols * cell)
    ax.set_ylim(0, n_rows * cell)
    ax.invert_yaxis()

    ax.set_xticks([(j + 0.5) * cell for j in range(n_cols)])
    ax.set_yticks([(i + 0.5) * cell for i in range(n_rows)])
    ax.set_xticklabels(col_lbls, rotation=45, ha='right', fontsize=12)
    ax.set_yticklabels(row_lbls, fontsize=9)
    ax.set_xlabel(col_axis_name, fontsize=14)
    ax.set_ylabel(row_axis_name, fontsize=14)

def _save(fig, out_dir, fname):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=700, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Confusion saved ⇒ {path}")

# ───────────── Draw top horizontal colorbars ───────────── #
def _draw_top_colorbars(fig, cmaps, norms, dim_used, y_col_names,
                        left0=0.03, width=0.21, height=0.02, bottom=0.93):
    """
    Draw a row of horizontal colorbars at the top of the figure.

    - When dim_used == 1, center the colorbar horizontally with width = width
    - When dim_used > 1, use fixed spacing with left0 + k * width
    """
    for k in range(dim_used):
        sm = cm.ScalarMappable(norm=norms[k], cmap=cmaps[k])
        sm.set_array([])

        if dim_used == 1:
            # Center the only colorbar.
            left = 0.5 - width / 2          # Canvas x-axis range is 0-1.
        else:
            left = left0 + k * width        # Original layout.

        cax = fig.add_axes((left, bottom, width, height))
        cb  = fig.colorbar(sm, cax=cax, orientation='horizontal')

        label = (y_col_names[k] if (y_col_names and k < len(y_col_names))
                 else f"Out {k}")
        cb.set_label(label, fontsize=12, labelpad=2)

        cb.set_ticks([])                    # Leave only the label text.
        cb.ax.xaxis.set_label_position('bottom')
        cb.ax.xaxis.set_ticks_position('top')

# ──────────────────────────────────────────────── #

def plot_confusion_from_npy(confusion_pred,
                            row_labels, col_labels,
                            out_dir,
                            y_col_names=None,
                            stats_dict=None,
                            cell_scale=1/5,
                            row_axis_name="Row Axis",
                            col_axis_name="Col Axis"):
    """
    Support confusion-matrix visualization for out_dim == 1 (full cell)
    and out_dim == 2-4 (four-triangle layout).
    """
    confusion_pred = np.asarray(confusion_pred)
    n_rows, n_cols, out_dim = confusion_pred.shape
    # If there are more row labels than column labels, swap rows and columns
    # so the side with more elements becomes the horizontal axis.
    if len(row_labels) > len(col_labels):
        confusion_pred = confusion_pred.transpose(1, 0, 2)  # (rows, cols, out) >> (cols, rows, out)
        row_labels, col_labels = col_labels, row_labels
        n_rows, n_cols = n_cols, n_rows
        # If axis titles should also be swapped, keep the next line active.
        row_axis_name, col_axis_name = col_axis_name, row_axis_name

    row_labels = _prep_labels(row_labels)
    col_labels = _prep_labels(col_labels)
    if y_col_names:
        y_col_names = _prep_labels(y_col_names)

    # ───────────── Single output: full-cell fill ───────────── #
    if out_dim == 1:
        vmin, vmax = float(np.min(confusion_pred)), float(np.max(confusion_pred))
        cmap, norm = plt.get_cmap("Purples"), mcolors.Normalize(vmin, vmax)

        fig, ax = plt.subplots(figsize=(16, 10))
        ax.set_aspect("equal", "box")
        _draw_grid(ax, n_rows, n_cols, cell_scale)

        for i in range(n_rows):
            for j in range(n_cols):
                val = confusion_pred[i, j, 0]
                ax.add_patch(Rectangle((j * cell_scale, i * cell_scale),
                                       cell_scale, cell_scale,
                                       facecolor=cmap(norm(val)),
                                       edgecolor="black"))

        _set_axes(ax, n_rows, n_cols, cell_scale, row_labels, col_labels,
                  row_axis_name, col_axis_name)

        # ---- Top horizontal colorbar ----
        _draw_top_colorbars(fig,
                            cmaps=[cmap],                    # Only one colormap.
                            norms=[norm],
                            width=0.63,
                            height=0.03,
                            dim_used=1,
                            y_col_names=y_col_names)

        _save(fig, out_dir, "confusion_matrix_1d.jpg")
        return

    # ───────────── Multi-output: four triangles ───────────── #
    dim_used = min(4, out_dim)
    cmaps = [plt.get_cmap(c) for c in ["Purples", "Blues", "Greens", "Oranges"]]

    # Normalize to [0, 1], optionally constrained by stats_dict.
    norms = []
    for k in range(dim_used):
        vals = confusion_pred[:, :, k]
        vmin, vmax = float(np.min(vals)), float(np.max(vals))
        if (stats_dict and y_col_names and k < len(y_col_names)
                and y_col_names[k] in stats_dict):
            vmin = stats_dict[y_col_names[k]]["min"]
            vmax = stats_dict[y_col_names[k]]["max"]
        confusion_pred[:, :, k] = normalize_data(vals, vmin, vmax)
        norms.append(mcolors.Normalize(0, 1))

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_aspect("equal", "box")
    fig.subplots_adjust(left=0.08, right=0.92, top=0.88, bottom=0.1)
    _draw_grid(ax, n_rows, n_cols, cell_scale)

    # Triangle templates.
    tri_idx = [
        [(0, 1), (0.5, 0.5), (1, 1)],      # Top-left
        [(1, 1), (0.5, 0.5), (1, 0)],      # Top-right
        [(1, 0), (0.5, 0.5), (0, 0)],      # Bottom-right
        [(0, 0), (0.5, 0.5), (0, 1)],      # Bottom-left
    ]

    for i in range(n_rows):
        for j in range(n_cols):
            cx, cy = j * cell_scale, i * cell_scale
            for k in range(dim_used):
                poly = [(cx + dx*cell_scale, cy + dy*cell_scale)
                        for dx, dy in tri_idx[k]]
                ax.add_patch(
                    Polygon(poly,
                            facecolor=cmaps[k](norms[k](confusion_pred[i, j, k])),
                            alpha=0.9))

    _set_axes(ax, n_rows, n_cols, cell_scale, row_labels, col_labels,
              row_axis_name, col_axis_name)

    # Top horizontal colorbars.
    _draw_top_colorbars(fig, cmaps, norms, dim_used, y_col_names)

    _save(fig, out_dir, "confusion_matrix_mimo.jpg")


def plot_3d_bars_from_confusion(confusion_pred,
                                row_labels, col_labels,
                                out_dir,
                                y_col_names=None,
                                stats_dict=None,
                                colorbar_extend_ratio=0.02,
                                cmap_name="GnBu"):
    """
    Draw a 3D bar-chart "confusion-like" figure.
    - If stats_dict provides the range for a dimension, use its min/max.
      Otherwise use the min/max from that dimension's data.
    - Align x/y ticks with the bar centers and center the tick labels.
    - Export one 3D bar chart per dimension.
    """

    os.makedirs(out_dir, exist_ok=True)
    confusion_pred = np.asarray(confusion_pred)
    n_rows, n_cols, out_dim = confusion_pred.shape

    # Label cleanup.
    row_labels = [short_label(lbl) for lbl in row_labels]
    col_labels = [short_label(lbl) for lbl in col_labels]
    if y_col_names:
        y_col_names = [short_label(name) for name in y_col_names]

    # Draw one figure per output dimension without limiting the dimension count here.
    for odx in range(out_dim):
        all_vals_dim = confusion_pred[:, :, odx]
        auto_min, auto_max = float(np.min(all_vals_dim)), float(np.max(all_vals_dim))

        if (stats_dict is not None) and (y_col_names is not None) \
           and (odx < len(y_col_names)) and (y_col_names[odx] in stats_dict):
            real_min = stats_dict[y_col_names[odx]]["min"]
            real_max = stats_dict[y_col_names[odx]]["max"]
        else:
            real_min = auto_min
            real_max = auto_max

        # Normalize this dimension to [0, 1].
        Z = normalize_data(all_vals_dim, real_min, real_max)

        norm_ = mcolors.Normalize(vmin=0, vmax=1)
        cmap_ = plt.get_cmap(cmap_name)

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        dx = dy = 0.5

        x_vals, y_vals, z_vals = [], [], []
        dz_vals, facecolors = [], []

        for i in range(n_rows):
            for j in range(n_cols):
                val_ = Z[i, j]
                x_vals.append(j)
                y_vals.append(i)
                z_vals.append(0)
                dz_vals.append(val_)
                # Generate colors from normalized values.
                facecolors.append(cmap_(norm_(val_)))

        x_vals = np.array(x_vals)
        y_vals = np.array(y_vals)
        z_vals = np.array(z_vals)
        dz_vals = np.array(dz_vals)

        ax.bar3d(
            x_vals, y_vals, z_vals,
            dx, dy, dz_vals,
            color=facecolors, alpha=0.75, shade=True
        )

        ax.grid(False)

        # Center ticks on the bars.
        ax.set_xticks(np.arange(n_cols) + dx / 2)
        ax.set_yticks(np.arange(n_rows) + dy / 2)

        # X-axis labels: rotate 45 degrees and right-align.
        ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=10)
        # Y-axis labels: use a modest rotation.
        ax.set_yticklabels(row_labels, rotation=-15, ha='left', va='center', fontsize=10)

        # Keep only the Z-axis label.
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("Value", fontsize=12)

        # Colorbar.
        sm = cm.ScalarMappable(norm=norm_, cmap=cmap_)
        sm.set_array([])  # Not tied to a specific array; used only for color mapping.
        cb_ = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.1, aspect=15)

        # Title / label selection.
        if y_col_names and odx < len(y_col_names):
            var_name = y_col_names[odx]
            # ax.set_title(f"3D Bars Confusion - {var_name}", fontsize=14)
            cb_.set_label(var_name, fontsize=12)
        else:
            var_name = f"Output_{odx}"
            # ax.set_title(f"3D Bars Confusion - out {odx}", fontsize=14)
            cb_.set_label(var_name, fontsize=12)

        out_jpg = os.path.join(out_dir, f"3d_bars_confusion_output_{odx+1}.jpg")
        plt.savefig(out_jpg, dpi=700, bbox_inches='tight')
        plt.close()
        print(f"[INFO] 3D Bars Confusion saved => {out_jpg}")


def plot_3d_surface_from_3d_heatmap(
        grid_x, grid_y, grid_z, heatmap_pred,
        out_dir,
        axes_labels=("X", "Y", "Z"),
        y_col_names=None,
        out_idx=0,
        cmap_name="GnBu",
        alpha_mode="value",        # "value" or "inverse"
        alpha_gamma=1.8            # New: alpha gamma bias (gamma = 1 -> purely linear)
):
    """
    Transparent 3D slice surfaces, one surface per Z slice.

    Parameters
    ----------
    alpha_mode  : "value"   -> larger values get higher alpha
                  "inverse" -> larger values get lower alpha
    alpha_gamma : gamma bias for alpha; >1 emphasizes high values, <1 makes the surface more solid overall
    """
    os.makedirs(out_dir, exist_ok=True)
    grid_x = np.asarray(grid_x)
    grid_y = np.asarray(grid_y)
    grid_z = np.asarray(grid_z)
    heatmap_pred = np.asarray(heatmap_pred)

    Zval = heatmap_pred[..., out_idx]           # (H,W,D)
    vmin_, vmax_ = float(np.min(Zval)), float(np.max(Zval))
    norm_  = mcolors.Normalize(vmin_, vmax_)
    cmap_  = plt.get_cmap(cmap_name)

    rgba = np.asarray(cmap_(norm_(Zval)))       # (H,W,D,4)
    alpha_base = norm_(Zval) ** alpha_gamma     # 0-1 after gamma
    if alpha_mode == "value":                   # Large values -> more opaque
        rgba[..., -1] = alpha_base
    elif alpha_mode == "inverse":               # Large values -> more transparent
        rgba[..., -1] = 1.0 - alpha_base
    else:
        raise ValueError("alpha_mode must be 'value' or 'inverse'")

    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(111, projection="3d")

    nz = grid_z.shape[2]
    for k in range(nz):
        ax.plot_surface(grid_x[:, :, k], grid_y[:, :, k], grid_z[:, :, k],
                        facecolors=rgba[:, :, k, :],
                        rstride=1, cstride=1,
                        linewidth=0, antialiased=True, shade=False)

    # --- Colorbar ---
    sm = cm.ScalarMappable(norm=norm_, cmap=cmap_)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.1, aspect=15)
    cb.set_label(
        y_col_names[out_idx] if (y_col_names and out_idx < len(y_col_names))
        else f"Output_{out_idx}", fontsize=12)

    # --- Axis settings ---
    ax.set_xlabel(axes_labels[0])
    ax.set_ylabel(axes_labels[1])
    ax.set_zlabel(axes_labels[2])
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.3g"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3g"))
    ax.zaxis.set_major_formatter(FormatStrFormatter("%.3g"))
    ax.grid(False)

    out_jpg = os.path.join(out_dir, f"surface3d_output_{out_idx+1}.jpg")
    plt.savefig(out_jpg, dpi=700, bbox_inches="tight")
    plt.close()
    print(f"[INFO] 3D Color Surface saved → {out_jpg}")
