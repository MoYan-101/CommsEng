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
df = pd.read_csv("Test_0611.csv")          # Change to your CSV path if needed
df_plot = df[[
    "Pressure (bar)", "Temperature (°C)",
    "GHSV (mL/g.h) (LN scale)",           # Updated column
    "STY_CH3OH (g/kg·h) (LN scale)"
]].dropna()
df_plot.columns = ["pressure", "temperature", "ghsv_ln", "sty_ch3oh"]

# ===== Bubble sizes (still based on "Methanol selectivity (%)") =====
base = 15
size = (df["Methanol selectivity (%)"] + base)      # Keep the original source column
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
gs_mid = gs[1].subgridspec(1, 2, wspace=0.05, width_ratios=WIDTH_RATIOS)

# --- Left: 2D scatter ---
ax2d = fig.add_subplot(gs_mid[0])
sc2d = ax2d.scatter(df_plot["pressure"], df_plot["sty_ch3oh"],
                    c=df_plot["temperature"], s=size,
                    cmap="plasma", alpha=0.7, edgecolors="none")
ax2d.set_xlabel("Pressure (bar)", fontsize=LABEL_FS)
ax2d.set_ylabel("STY_CH3OH (g·kg⁻¹·h⁻¹) (LN)", fontsize=LABEL_FS)
ax2d.grid(False)

# --- Right: 3D scatter ---
ax3d = fig.add_subplot(gs_mid[1], projection="3d")
ax3d.scatter(df_plot["pressure"], df_plot["ghsv_ln"],       # Use ghsv_ln
             df_plot["sty_ch3oh"], c=df_plot["temperature"], s=size,
             cmap="plasma", alpha=0.7, edgecolors="k", linewidth=0.3)

# z-axis ticks
zmin, zmax = df_plot["sty_ch3oh"].min(), df_plot["sty_ch3oh"].max()
ticks = np.linspace(zmin, zmax, 5)
ax3d.set_zlim(zmin, zmax)
ax3d.zaxis.set_major_locator(FixedLocator(ticks))
ax3d.zaxis.set_major_formatter(FormatStrFormatter('%.2g'))

ax3d.set_xlabel("Pressure (bar)", fontsize=LABEL_FS)
ax3d.set_ylabel("GHSV (mL g⁻¹ h⁻¹) (LN)", fontsize=LABEL_FS)   # Updated label
ax3d.set_zlabel("STY_CH3OH (g·kg⁻¹·h⁻¹) (LN)", fontsize=LABEL_FS)
ax3d.tick_params(labelsize=TICK_FS)
ax3d.grid(False)

# --- Top spacer (reserved for legend if needed) ---
fig.add_subplot(gs[0]).axis("off")

# --- Bottom colorbar ---
ax_cbar = fig.add_subplot(gs[2])
cbar = fig.colorbar(sc2d, cax=ax_cbar,
                    orientation="horizontal", aspect=CBAR_ASPECT, shrink=CBAR_SHRINK)
cbar.set_label("Temperature (°C)", fontsize=LABEL_FS + 2)
cbar.ax.tick_params(labelsize=LABEL_FS + 2)

# Shrink and center the colorbar
pos = ax_cbar.get_position()
new_w = pos.width * CBAR_WIDTH_FRA
ax_cbar.set_position([pos.x0 + (pos.width-new_w)/2, pos.y0, new_w, pos.height])

plt.subplots_adjust(left=0.08, right=0.95, top=0.94, bottom=0.08)
plt.savefig("pressure_sty_layout.jpg", dpi=700, bbox_inches="tight")
plt.show()
