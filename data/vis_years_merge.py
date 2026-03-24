import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FixedLocator, FormatStrFormatter

# ---------- Shared constants ----------
FIGSIZE        = (12, 10)
HEIGHT_RATIOS  = [0.08, 0.87, 0.05]
WIDTH_RATIOS   = [1, 1.5]
CBAR_SHRINK    = 0.80
CBAR_ASPECT    = 25
CBAR_WIDTH_FRA = 0.65
LABEL_FS       = 14
TICK_FS        = 12

# ===== Load data =====
df = pd.read_csv("Test_0611.csv")
df_plot = df[[
    "Year", "Temperature (°C)",
    "Methanol selectivity (%)", "CO2 conversion efficiency (%)"
]].dropna()
df_plot.columns = ["year", "temperature",
                   "methanol_selectivity", "co2_conversion"]
df_plot["combo_score"] = (
    df_plot["methanol_selectivity"] + df_plot["co2_conversion"]
)

# Yearly max points for line plots
selectivity_max = df_plot.loc[df_plot.groupby("year")
                              ["methanol_selectivity"].idxmax()]
conversion_max  = df_plot.loc[df_plot.groupby("year")
                              ["co2_conversion"].idxmax()]

# Max combined-score point
max_idx = df_plot["combo_score"].idxmax()
x_max, y_max, z_max, t_max = df_plot.loc[
    max_idx, ["year", "methanol_selectivity",
              "co2_conversion", "temperature"]
]

# Bubble sizes
base = 15
size = (df_plot["methanol_selectivity"] + base)
size = (size / size.max()) * 100

# ===== Global style =====
plt.style.use("default")
mpl.rcParams.update({
    "font.size": TICK_FS,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 1.2,
    "xtick.labelsize": TICK_FS,
    "ytick.labelsize": TICK_FS
})

# ===== Figure and grid =====
fig = plt.figure(figsize=FIGSIZE)
gs  = GridSpec(3, 1, height_ratios=HEIGHT_RATIOS, figure=fig)
gs_mid = gs[1].subgridspec(1, 2, wspace=0.05,
                           width_ratios=WIDTH_RATIOS)

# --- Left: 2D scatter + lines + annotation ---
ax2d = fig.add_subplot(gs_mid[0])
sc2d = ax2d.scatter(df_plot["year"], df_plot["co2_conversion"],
                    c=df_plot["temperature"], s=size,
                    cmap="plasma", alpha=0.6, edgecolors="none")
line1, = ax2d.plot(selectivity_max["year"],
                   selectivity_max["co2_conversion"],
                   color="#1f77b4", linewidth=2)
line2, = ax2d.plot(conversion_max["year"],
                   conversion_max["co2_conversion"],
                   color="#d62728", linestyle="--", linewidth=2)
# Max-point marker
ax2d.plot(x_max, z_max, marker='+', markersize=12,
          linestyle='None', color='black', markeredgewidth=2)
# Annotation
ax2d.text(
    x_max - 10, z_max + 0.75,
    f"T = {t_max:.1f} °C\nMS = {y_max:.1f} %\nCCE = {z_max:.1f} %",
    fontsize=10, va='bottom', ha='left', color='blue'
)

ax2d.set_xlabel("Year", fontsize=LABEL_FS)
ax2d.set_ylabel("CO₂ Conversion Efficiency (%)",
                fontsize=LABEL_FS)
ax2d.tick_params(axis='x', rotation=30)
ax2d.grid(False)

# --- Right: 3D scatter + lines + annotation ---
ax3d = fig.add_subplot(gs_mid[1], projection="3d")
ax3d.scatter(df_plot["year"], df_plot["methanol_selectivity"],
             df_plot["co2_conversion"], c=df_plot["temperature"],
             s=size, cmap="plasma", alpha=0.7,
             edgecolors="k", linewidth=0.3)
ax3d.plot(selectivity_max["year"],
          selectivity_max["methanol_selectivity"],
          selectivity_max["co2_conversion"],
          color="#1f77b4", linewidth=2)
ax3d.plot(conversion_max["year"],
          conversion_max["methanol_selectivity"],
          conversion_max["co2_conversion"],
          color="#d62728", linestyle="--", linewidth=2)
# Max point
ax3d.scatter([x_max], [y_max], [z_max],
             marker='+', s=200, color='black', linewidths=2)
# Annotation
ax3d.text(x_max - 15, y_max, z_max,
          f"T = {t_max:.1f}°C\nMS+CCE = {y_max + z_max:.1f}",
          fontsize=10, color='blue')

# z-axis ticks: 5 steps, 2 significant digits
zmin, zmax_val = df_plot["co2_conversion"].min(), df_plot["co2_conversion"].max()
ticks = np.linspace(zmin, zmax_val, 5)
ax3d.set_zlim(zmin, zmax_val)
ax3d.zaxis.set_major_locator(FixedLocator(ticks))
ax3d.zaxis.set_major_formatter(FormatStrFormatter('%.2g'))

ax3d.set_xlabel("Year", fontsize=LABEL_FS)
ax3d.set_ylabel("Methanol Selectivity (%)", fontsize=LABEL_FS)
ax3d.set_zlabel("CO₂ Conversion Efficiency (%)",
                fontsize=LABEL_FS)
ax3d.tick_params(labelsize=TICK_FS)
ax3d.grid(False)

# --- Top legend ---
ax_legend = fig.add_subplot(gs[0])
ax_legend.axis("off")
ax_legend.legend(
    handles=[line1, line2,
             mpl.lines.Line2D([], [], marker='+', linestyle='None',
                              color='black', markersize=12, markeredgewidth=2)],
    labels=["Max Methanol Selectivity",
            "Max CO₂ Conversion",
            "Max Combined Score"],
    loc="center", ncol=3, frameon=False, fontsize=LABEL_FS
)

# --- Bottom colorbar ---
ax_cbar = fig.add_subplot(gs[2])
cbar = fig.colorbar(sc2d, cax=ax_cbar,
                    orientation="horizontal",
                    aspect=CBAR_ASPECT, shrink=CBAR_SHRINK)
cbar.set_label("Temperature (°C)",
               fontsize=LABEL_FS + 2)
cbar.ax.tick_params(labelsize=LABEL_FS + 2)

# Shrink and center the colorbar
pos = ax_cbar.get_position()
new_w = pos.width * CBAR_WIDTH_FRA
ax_cbar.set_position(
    [pos.x0 + (pos.width-new_w)/2, pos.y0, new_w, pos.height]
)

plt.subplots_adjust(left=0.08, right=0.95,
                    top=0.94, bottom=0.08)
plt.savefig("year_conversion_layout.jpg",
            dpi=700, bbox_inches="tight")
plt.show()
