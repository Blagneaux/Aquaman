# Code for fish analysis: Statistics Plot
# G. Jodin, after C. Porcon, 2024

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import filtfilt, butter 
import csv

# Cambridge-friendly style settings (see fusionPython conventions)
CAMBRIDGE_RC = {
    "font.size": 24,
    "axes.titlesize": 24,
    "axes.labelsize": 24,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "legend.fontsize": 24,
    "text.usetex": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "lines.linewidth": 0.8,
    "lines.markersize": 3.5,
    "lines.markeredgewidth": 0.6,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
}


def apply_cambridge_style():
    plt.rcParams.update(CAMBRIDGE_RC)


def pVSs(speed):
    a = -1 / 2 / (225**2 - 100**2)
    b = a * (225**2 - 2 * 100**2)
    return a * speed**2 + b


def pVSd(distance):
    a = -6 / 4 / (1/20**4 - 1/37**4)
    b = a * (1/20**4 - 4/37**4) / 3
    return a/distance**4 + b


def eval_curve(curve_func, x_vals):
    try:
        y_vals = curve_func(x_vals)
    except Exception:
        y_vals = np.array([curve_func(x) for x in x_vals], dtype=float)
    return np.asarray(y_vals, dtype=float)


def fraction_hist_above_curve(hist2d, xedges, yedges, curve_func):
    xcenters = 0.5 * (xedges[:-1] + xedges[1:])
    ycenters = 0.5 * (yedges[:-1] + yedges[1:])
    curve_vals = eval_curve(curve_func, xcenters)

    Xc, Yc = np.meshgrid(xcenters, ycenters, indexing="ij")
    mask = Yc >= curve_vals[:, None]

    total = float(hist2d.sum())
    if total <= 0:
        return np.nan, 0.0, total

    above = float(hist2d[mask].sum())
    return above / total, above, total


# List of possible sensors
channels_to_read = ["S1", "S2", "S4", "S6", "S7"]
fAcqPressure = 500
data_path = "E:/saveStatPlotsV2_All.csv"

# Load data
print("Loading data..., can take a while...")
with open(data_path, 'r') as file:
    reader = csv.reader(file)

    # Split the rows into variables
    pressure_data_concat = np.array(next(reader), dtype=float)
    distance_to_sensors_all_concat = np.array(next(reader), dtype=float)
    X_to_sensors_all_concat = np.array(next(reader), dtype=float)
    speed_concat = np.array(next(reader), dtype=float)
    numFish_all_concat = np.array(next(reader), dtype=float)

# Filtering the data
subIndexesToRemove = numFish_all_concat < 0
tstampsToRemove = 0

for i in range(len(subIndexesToRemove)): 
    if tstampsToRemove > 0:
        tstampsToRemove -= 1
        subIndexesToRemove[i] = True
    else:
        if(abs(pressure_data_concat[i])) > 3: # not physically coherent
            tstampsToRemove = 1.2 * fAcqPressure

# Remove shocks
pressure_data_concat[subIndexesToRemove] = 0

# Subselection
varSpeed = np.insert(np.diff(speed_concat), 0, 0)
subIndexes = (abs(X_to_sensors_all_concat) < 25) & (pressure_data_concat < -.42) & (distance_to_sensors_all_concat < 40) & (speed_concat < 600)
dis = distance_to_sensors_all_concat[subIndexes]
pres = pressure_data_concat[subIndexes]
speed = speed_concat[subIndexes]
X2s = X_to_sensors_all_concat[subIndexes]
fishNum = numFish_all_concat[subIndexes]

# Plot 1
# histogram to draw limits
heatmap1, xedges1, yedges1 = np.histogram2d(speed, pres, bins=(30, 30))
heatmap2, xedges2, yedges2 = np.histogram2d(dis, pres, bins=(30, 30))

apply_cambridge_style()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(19, 8))

vmax_display = 300

# Pressure vs Speed
extent = [xedges1[0], xedges1[-1], yedges1[0], yedges1[-1]]
im1 = ax1.imshow(heatmap1.T, extent=extent, origin='lower', cmap="binary", vmax=vmax_display)
ax1.set_aspect('auto')
ax1.set_xlabel("Speed (mm/s)")
ax1.set_ylabel("Pressure at closest sensor (Pa)")
ax1.set_xlim(0, 600)
ax1.set_ylim(-3, -0.42)

# Pressure vs Distance
extent = [xedges2[0], xedges2[-1], yedges2[0], yedges2[-1]]
im2 = ax2.imshow(heatmap2.T, extent=extent, origin='lower', cmap="binary", vmax=vmax_display)
ax2.set_aspect('auto')
ax2.set_xlabel("Distance to closest sensor (mm)")
ax2.set_xlim(0, 40)
ax2.set_ylim(-3, -0.42)

speed_line = np.linspace(xedges1[0], xedges1[-1], 300)
dist_line = np.linspace(max(xedges2[0], 1e-3), xedges2[-1], 300)

try:
    pressure_line_speed = eval_curve(pVSs, speed_line)
    ax1.plot(
        speed_line,
        pressure_line_speed,
        color="red",
        linewidth=0.8,
        label="contour from theory",
    )
except NotImplementedError:
    print("pVSs not defined; skipping speed-law overlay.")

try:
    pressure_line_dist = eval_curve(pVSd, dist_line)
    ax2.plot(
        dist_line,
        pressure_line_dist,
        color="red",
        linewidth=0.8,
        label="contour from theory",
    )
except NotImplementedError:
    print("pVSd not defined; skipping distance-law overlay.")

# Shared colorbar for density
cbar = fig.colorbar(im1, ax=[ax1, ax2], pad=0.02)
cbar.set_label("Count of instances of fish")

try:
    frac1, above1, total1 = fraction_hist_above_curve(heatmap1, xedges1, yedges1, pVSs)
    print(f"Histogram above pVSs: {frac1 * 100:.2f}% ({above1:.0f}/{total1:.0f})")
    ax1.text(
        0.02,
        0.02,
        f"Inside contour: {frac1 * 100:.1f}%",
        transform=ax1.transAxes,
        va="bottom",
        ha="left",
    )
except NotImplementedError:
    print("pVSs not defined; skipping histogram coverage for speed plot.")

try:
    frac2, above2, total2 = fraction_hist_above_curve(heatmap2, xedges2, yedges2, pVSd)
    print(f"Histogram above pVSd: {frac2 * 100:.2f}% ({above2:.0f}/{total2:.0f})")
    ax2.text(
        0.02,
        0.02,
        f"Inside contour: {frac2 * 100:.1f}%",
        transform=ax2.transAxes,
        va="bottom",
        ha="left",
    )
except NotImplementedError:
    print("pVSd not defined; skipping histogram coverage for distance plot.")

plt.show()

# Save figure as SVG for vector editing/embedding
fig.savefig("fish_stats.svg", format="svg", bbox_inches="tight")
