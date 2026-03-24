import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# ===== Load data and select fields =====
df = pd.read_csv("DATA_VIS_YEAR.csv")
df_plot = df[[
    "year",
    "Temperature (°C)",
    "Methanol selectivity (%)",
    "CO2 conversion efficiency (%)"
]].dropna()
df_plot.columns = ["year", "temperature", "methanol_selectivity", "co2_conversion"]

# ===== Find the highest-CO2-conversion point among max-selectivity samples =====
max_selectivity = df_plot["methanol_selectivity"].max()
subset = df_plot[df_plot["methanol_selectivity"] == max_selectivity]
max_idx = subset["co2_conversion"].idxmax()

x_max = df_plot.loc[max_idx, "year"]
y_max = df_plot.loc[max_idx, "co2_conversion"]
t_max = df_plot.loc[max_idx, "temperature"]
s_max = df_plot.loc[max_idx, "methanol_selectivity"]

# ===== Scale point sizes with a minimum floor =====
base = 15  # Keep a visible minimum size
size_scaled = (df_plot["methanol_selectivity"] + base)
size_scaled = (size_scaled / size_scaled.max()) * 200


# ===== Set style =====
plt.style.use("seaborn-v0_8-whitegrid")
mpl.rcParams["axes.edgecolor"] = "#333333"
mpl.rcParams["axes.linewidth"] = 1.2

# ===== Prepare figure =====
fig, ax = plt.subplots(figsize=(12, 8))
ax.grid(False)

# ===== Scatter plot =====
sc = ax.scatter(
    df_plot["year"],
    df_plot["co2_conversion"],
    c=df_plot["temperature"],
    s=size_scaled,
    cmap="plasma",
    alpha=0.6,
    edgecolors="none",
    marker="o"
)

# ===== Build yearly max-selectivity and max-conversion lines =====
selectivity_max_points = df_plot.loc[df_plot.groupby("year")["methanol_selectivity"].idxmax()]
conversion_max_points = df_plot.loc[df_plot.groupby("year")["co2_conversion"].idxmax()]

# Draw direct lines
ax.plot(selectivity_max_points["year"], selectivity_max_points["co2_conversion"],
        color="#1f77b4", linestyle="-", linewidth=2, label="Max Methanol Selectivity")

ax.plot(conversion_max_points["year"], conversion_max_points["co2_conversion"],
        color="#d62728", linestyle="--", linewidth=2, label="Max CO₂ Conversion")


# ===== Add max-point marker and guide lines (no legend) =====
ax.plot(x_max, y_max, marker='+', markersize=12, color='black', markeredgewidth=2)
ax.axhline(y=y_max, color='gray', linestyle='--', linewidth=1, zorder=0)
ax.axvline(x=x_max, color='gray', linestyle='--', linewidth=1, zorder=0)

# ===== Add max-point annotation =====
label_text = f"T = {t_max:.1f} °C\nMS = {s_max:.1f} %\nCCE = {y_max:.1f} %"
ax.text(x_max + 0.2, y_max + 0.15, label_text, fontsize=8, va='bottom', ha='left', color='blue')

# ===== Legend, axes, and colorbar =====
ax.set_xlabel("Year", fontsize=14)
ax.set_ylabel("CO₂ Conversion Efficiency (%)", fontsize=14)

cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("Temperature (°C)", fontsize=12)

ax.legend(loc="upper left", fontsize=14, frameon=True)

# ===== Adaptive y-axis range =====
y_margin = (df_plot["co2_conversion"].max() - df_plot["co2_conversion"].min()) * 0.015
ax.set_ylim(df_plot["co2_conversion"].min() - y_margin, df_plot["co2_conversion"].max() + y_margin)

# ===== Show full frame =====
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(True)
    ax.spines[spine].set_color("#333333")
    ax.spines[spine].set_linewidth(1.2)

plt.tight_layout()
plt.savefig("vis_by_year.jpg", dpi=700)
plt.show()
