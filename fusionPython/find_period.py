import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Autocorrelation computation
# ----------------------------
def autocorrelation_with_reference(data):
    # Take the last column as the reference state
    reference_state = data[:, -1]

    # Normalize reference once (saves time)
    ref = (reference_state - np.mean(reference_state)) / (np.std(reference_state) + 1e-10)

    autocorrelations = []
    for i in range(data.shape[1]):
        col = data[:, i]
        col = (col - np.mean(col)) / (np.std(col) + 1e-10)

        # correlation at lag 0 (valid mode gives one value for equal-length arrays)
        correlation = np.correlate(col, ref, mode="valid")[0]
        autocorrelations.append(correlation)

    return np.asarray(autocorrelations)


# ----------------------------
# Publication-quality plotting
# ----------------------------
def plot_autocorrelation(
    autocorrelations,
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

    y = np.asarray(autocorrelations)
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
    ax.set_ylabel(r"Autocorrelation ($10^3$)")

    ax.grid(True, linewidth=0.6)     # >= 0.5 pt
    fig.tight_layout()

    # --- PRODUCTION OUTPUTS ---
    # 1) EPS (preferred): vector + editable text
    fig.savefig(f"{out_base}.eps", format="eps", bbox_inches="tight")

    # 2) TIFF fallback: line art should be 1200 dpi
    fig.savefig(f"{out_base}.tiff", format="tiff", dpi=1200, bbox_inches="tight")

    # Optional: also save PDF for your own checking
    fig.savefig(f"{out_base}.pdf", format="pdf", bbox_inches="tight")

    plt.show()


def load_data(path):
    """Load data from a CSV file"""
    data = pd.read_csv(path, header=None)
    return data.values


# ----------------------------
# Example usage
# ----------------------------
full_data_path = "D:/simuChina/cartesianBDD_FullVorticityMapFixed/Re10000_h6.0/FullMap.csv"
data = load_data(full_data_path)

autocorr_results = autocorrelation_with_reference(data)

plot_autocorrelation(
    autocorr_results,
    out_base="figure2_autocorrelation",
    downsample=5  # increase to 10 if still too dense
)
