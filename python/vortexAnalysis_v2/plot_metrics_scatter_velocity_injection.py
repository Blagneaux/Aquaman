"""Scatter plot of normalized RMSE vs mean vortex-core offset distance."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_metrics(path: Path):
    frames = []
    rmse_norm = []
    core_dist = []
    mae_norm = []
    max_abs_norm = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["frame"].strip().lower() == "mean":
                continue
            try:
                frames.append(int(row["frame"]))
                rmse_norm.append(float(row["rmse_norm"]))
                core_dist.append(float(row["mean_min_core_distance"]))
                mae_norm.append(float(row.get("mae_norm", "nan")))
                max_abs_norm.append(float(row.get("max_abs_norm", "nan")))
            except ValueError:
                continue
    return (
        np.array(frames),
        np.array(rmse_norm),
        np.array(core_dist),
        np.array(mae_norm),
        np.array(max_abs_norm),
    )


def main():
    parser = argparse.ArgumentParser(description="Plot normalized RMSE vs mean vortex-core distance.")
    parser.add_argument("--metrics", type=Path, default=Path(__file__).with_name("error_frames") / "error_metrics.csv", help="Path to error_metrics.csv")
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("error_frames") / "rmse_vs_core_distance.png", help="Output plot path")
    parser.add_argument("--dpi", type=int, default=150, help="Output DPI")
    args = parser.parse_args()

    frames, rmse_norm, core_dist, mae_norm, max_abs_norm = load_metrics(args.metrics)
    if frames.size == 0:
        print("No metric rows found to plot.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=False)

    sc = axes[0].scatter(core_dist, rmse_norm, c=frames, cmap="viridis", s=16, alpha=0.9)
    axes[0].set_xlabel("Mean min core distance (grid units)")
    axes[0].set_ylabel("Normalized RMSE (by ref_std)")
    axes[0].set_title("Vortex core alignment vs normalized RMSE")
    cbar = fig.colorbar(sc, ax=axes[0], fraction=0.046, pad=0.02)
    cbar.set_label("Frame index")

    axes[1].plot(frames, rmse_norm, label="RMSE norm", color="tab:blue")
    axes[1].plot(frames, mae_norm, label="MAE norm", color="tab:orange", alpha=0.7)
    axes[1].plot(frames, max_abs_norm, label="Max abs norm", color="tab:green", alpha=0.7)
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Normalized error")
    axes[1].set_title("Normalized errors vs frame")
    axes[1].legend()

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi)
    plt.close(fig)
    print(f"Saved scatter plot to {args.out}")


if __name__ == "__main__":
    main()
