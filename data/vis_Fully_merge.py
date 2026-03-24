import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FixedLocator, FormatStrFormatter, MaxNLocator
from matplotlib.lines import Line2D

# ================= Global style =================
plt.style.use("default")
mpl.rcParams.update({
    "font.size": 12,            # Base font size for ticks
    "axes.edgecolor": "#333333",
    "axes.linewidth": 1.2,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12
})

# ================= Load data =================
df = pd.read_csv("Test_0611.csv")          # Change to your CSV path if needed

# ---------- Dataset 1 (Pressure view) ----------
# Use "GHSV (mL/g.h) (LN scale)" instead of "Methanol selectivity (%)"
df1 = df[[
    "Pressure (bar)", "Temperature (°C)",
    "GHSV (mL/g.h) (LN scale)",
    "STY_CH3OH (g/kg·h) (LN scale)"
]].dropna().copy()
df1.columns = ["pressure", "temperature", "ghsv_ln", "sty_ch3oh"]

# ---------- Dataset 2 (Year view) ----------
df2 = df[[
    "Year", "Temperature (°C)",
    "Methanol selectivity (%)",
    "CO2 conversion efficiency (%)"
]].dropna().copy()
df2.columns = ["year", "temperature",
               "methanol_selectivity", "co2_conversion"]

# ---------- Year-series helpers ----------
selectivity_max = df2.loc[df2.groupby("year")["methanol_selectivity"].idxmax()]
conversion_max  = df2.loc[df2.groupby("year")["co2_conversion"].idxmax()]

df2["combo_score"] = df2["methanol_selectivity"] + df2["co2_conversion"]
max_idx = df2["combo_score"].idxmax()
x_max, y_max, z_max, t_max = df2.loc[max_idx, ["year",
                                               "methanol_selectivity",
                                               "co2_conversion",
                                               "temperature"]]

# ---------- Bubble sizes ----------
base = 15
# Keep size1 based on "Methanol selectivity (%)"
size1 = (df["Methanol selectivity (%)"] + base) / \
        (df["Methanol selectivity (%)"].max() + base) * 100
size2 = (df2["methanol_selectivity"] + base) / \
        (df2["methanol_selectivity"].max() + base) * 100

# ========== Figure and layout ==========
fig = plt.figure(figsize=(14, 8))
gs  = GridSpec(3, 1, height_ratios=[0.08, 0.87, 0.05], figure=fig)
gs_mid = gs[1].subgridspec(1, 2, wspace=0.05, width_ratios=[1.1, 1.1])

# ========== 3D plot 1 (Pressure - GHSV) ==========
ax1 = fig.add_subplot(gs_mid[0], projection="3d")
sc1 = ax1.scatter(
    df1["pressure"], df1["ghsv_ln"], df1["sty_ch3oh"],
    c=df1["temperature"], s=size1,
    cmap="plasma", alpha=0.7, edgecolors="k", linewidth=0.3, marker="o"
)

ax1.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
ax1.yaxis.set_major_formatter(FormatStrFormatter('%d'))

zmin, zmax = df1["sty_ch3oh"].min(), df1["sty_ch3oh"].max()
ticks = np.linspace(zmin, zmax, 5)
ax1.set_zlim(zmin, zmax)
ax1.zaxis.set_major_locator(FixedLocator(ticks))
ax1.zaxis.set_major_formatter(FormatStrFormatter('%.2g'))

ax1.set_xlabel("Pressure (bar)", fontsize=13)
ax1.set_ylabel("GHSV (mL g⁻¹ h⁻¹) (LN)", fontsize=13)
ax1.set_zlabel("STY_CH3OH (g kg⁻¹ h⁻¹) (LN)", fontsize=13)
ax1.tick_params(labelsize=12)
ax1.grid(False)

# ========== 3D plot 2 (Year view) ==========
ax2 = fig.add_subplot(gs_mid[1], projection="3d")
sc2 = ax2.scatter(
    df2["year"], df2["methanol_selectivity"], df2["co2_conversion"],
    c=df2["temperature"], s=size2,
    cmap="plasma", alpha=0.7, edgecolors="k", linewidth=0.3, marker="o"
)

line1 = ax2.plot(selectivity_max["year"], selectivity_max["methanol_selectivity"],
                 selectivity_max["co2_conversion"], color="#1f77b4", linewidth=2)[0]
line2 = ax2.plot(conversion_max["year"], conversion_max["methanol_selectivity"],
                 conversion_max["co2_conversion"], color="#d62728", linestyle="--", linewidth=2)[0]

ax2.scatter([x_max], [y_max], [z_max], marker='+', s=200, color='black', linewidths=2)
proxy_max = Line2D([], [], marker='+', color='black',
                   linestyle='None', markersize=12, markeredgewidth=2)

ax2.set_xlabel("Year", fontsize=13)
ax2.set_ylabel("Methanol selectivity (%)", fontsize=13)
ax2.set_zlabel("CO₂ Conversion Efficiency (%)", fontsize=13)
ax2.tick_params(labelsize=12)
ax2.grid(False)

# ========== Top legend ==========
ax_legend = fig.add_subplot(gs[0])
ax_legend.axis("off")
ax_legend.legend(
    handles=[line1, line2, proxy_max],
    labels=["Max Methanol Selectivity",
            "Max CO₂ Conversion",
            "Max Combined Score"],
    loc="center", ncol=3, frameon=False, fontsize=14
)

# ========== Bottom colorbar ==========
ax_cbar = fig.add_subplot(gs[2])
cbar = fig.colorbar(sc2, cax=ax_cbar,
                    orientation="horizontal", aspect=25, shrink=0.80)
cbar.set_label("Temperature (°C)", fontsize=16)
cbar.ax.tick_params(labelsize=16)

# Shrink and center the colorbar
pos = ax_cbar.get_position()
new_w = pos.width * 0.65
ax_cbar.set_position([pos.x0 + (pos.width-new_w)/2, pos.y0, new_w, pos.height])

# ========== Save ==========
plt.subplots_adjust(left=0.07, right=0.96, top=0.94, bottom=0.08)
plt.savefig("merged_3d_plots.jpg", dpi=700, bbox_inches="tight")
plt.show()
