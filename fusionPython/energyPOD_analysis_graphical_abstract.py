import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
        "pdf.fonttype": 42,              # TrueType (editable if you also save PDF
        "ps.fonttype": 42,               # TrueType in EPS/PS

        # Ensure lines meet the minimum 0.5 pt requirement
        "lines.linewidth": 0.8,          # >= 0.5 pt (safe margin)
        "lines.markersize": 3.5,
        "lines.markeredgewidth": 0.6,    # >= 0.5 pt
        "axes.linewidth": 0.8,           # axes thickness
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    })

def frame_reconstruction(X):
    fig = plt.figure(figsize=(6, 5))

    X_reshaped = X.reshape(512, 256)
    plt.imshow(X_reshaped[128:-160,:-64].T, 
               cmap="seismic", 
               vmin=-0.5, 
               vmax=0.5,
               alpha=1,
    )
    plt.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )
    plt.show()
    fig.savefig("graphical_abstract.eps", format="eps", bbox_inches="tight")
    fig.savefig("graphical_abstract.pdf", format="pdf", bbox_inches="tight")

data = pd.read_csv("D:/simuChina/cartesianBDD_FullVorticityMap_Fixed_HiRes/Re10000_h12.0/FullMap.csv", header=None).to_numpy()

snapshot = data[:, -1]
frame_reconstruction(snapshot)