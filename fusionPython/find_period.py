import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ----------------------------
# Reference-frame correlation computation
# ----------------------------
def reference_frame_correlation(data):
    # Take the last column as the reference state
    reference_state = data[:, -1]

    # Normalize reference once (saves time)
    ref = (reference_state - np.mean(reference_state)) / (np.std(reference_state) + 1e-10)

    correlations = []
    for i in range(data.shape[1]):
        col = data[:, i]
        col = (col - np.mean(col)) / (np.std(col) + 1e-10)

        # correlation at lag 0 (valid mode gives one value for equal-length arrays)
        correlation = np.correlate(col, ref, mode="valid")[0]
        correlations.append(correlation)

    return np.asarray(correlations)


# ----------------------------
# Publication-quality plotting
# ----------------------------
def plot_reference_frame_correlation(
    correlations,
    out_base="figure2_autocorrelation",
    downsample=5,               # plot every Nth point to reduce overplotting
    fig_width_in=6.8,           # ~single-column width; adjust if needed
    fig_height_in=4.2
):
    """
    Creates Cambridge-ready vector output (EPS) with editable labels and
    ensures minimum line widths are >= 0.5 pt.
    Also writes a high-res TIFF fallback.
    """

    # --- Style choices aimed at print/publisher requirements ---
    plt.rcParams.update({
        "font.size": 10,                 # readable at journal size
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,

        # IMPORTANT: keep text editable in vector outputs
        "text.usetex": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,              # TrueType (editable) if you also save PDF
        "ps.fonttype": 42,               # TrueType in EPS/PS

        # Ensure lines meet the minimum 0.5 pt requirement
        "lines.linewidth": 0.8,          # >= 0.5 pt (safe margin)
        "lines.markersize": 3.5,
        "lines.markeredgewidth": 0.6,    # >= 0.5 pt
        "axes.linewidth": 0.8,           # axes thickness
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    })

    y = np.asarray(correlations)
    x = np.arange(len(y))

    # Downsample to reduce file weight and visual clutter
    if downsample is not None and downsample > 1:
        x = x[::downsample]
        y = y[::downsample]

    fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in))

    # Use a line + small markers; avoid gigantic filled dots
    ax.plot(
        x/1000, y/1000,
        linestyle="-",
        marker="o",
        markerfacecolor="none",      # helps avoid solid “ink blobs”
        markeredgewidth=0.6,         # >= 0.5 pt
    )

    # Keep title shorter for journal aesthetics; move details to caption in manuscript
    ax.set_xlabel(r"Time steps ($10^3$)")
    ax.set_ylabel(r"Correlation with final snapshot ($10^3$)")

    ax.grid(True, linewidth=0.6)     # >= 0.5 pt
    fig.tight_layout()

    # --- PRODUCTION OUTPUTS ---
    # 1) EPS (preferred): vector + editable text
    fig.savefig(f"{out_base}.eps", format="eps", bbox_inches="tight")

    # 2) PDF for the manuscript and local checking
    fig.savefig(f"{out_base}.pdf", format="pdf", bbox_inches="tight")

    plt.show()


def load_data(path):
    """Load data from a CSV file"""
    data = pd.read_csv(path, header=None)
    return data.values


# ----------------------------
# Example usage
# ----------------------------
figure_dir = Path(__file__).resolve().parent / "article" / "figures"
data_candidates = [
    Path("D:/simuChina/cartesianBDD_FullVorticityMapFixed/Re10000_h6.0/FullMap.csv"),
    Path("D:/simuChina/cartesianBDD_FullVorticityMap_Fixed/Re10000_h6.0/FullMap.csv"),
    Path("E:/simuChina/cartesianBDD_FullVorticityMapFixed/Re10000_h6.0/FullMap.csv"),
    Path("E:/simuChina/cartesianBDD_FullVorticityMap_Fixed/Re10000_h6.0/FullMap.csv"),
]
full_data_path = next((path for path in data_candidates if path.exists()), None)
if full_data_path is None:
    raise FileNotFoundError("Could not find Re10000_h6.0/FullMap.csv in expected data folders.")

data = load_data(full_data_path)

correlation_results = reference_frame_correlation(data)

plot_reference_frame_correlation(
    correlation_results,
    out_base=str(figure_dir / "figure2_autocorrelation"),
    downsample=5  # increase to 10 if still too dense
)
