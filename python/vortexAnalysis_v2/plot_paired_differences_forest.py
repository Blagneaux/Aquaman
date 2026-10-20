"""Alternative layout for the paired source-video difference figure.

Renders the 21 pressure-conditioned vortex descriptors as horizontal rows sharing
one vertical zero line, in the style of a SHAP summary plot, instead of the grid of
individual boxplot panels produced by analyze_pressure_conditioned_vortices.py.

Rows are coloured by direction: red where the cylinder-associated summary is larger,
blue where the fish-associated summary is larger. Rows whose 95% t-interval contains
zero are drawn muted.

The 21 descriptors carry incompatible units (grid cells, frames, dimensionless
ratios), so a shared horizontal axis needs a common scale. Two are provided:

  sd       each video difference divided by the standard deviation of that
           descriptor's own differences, so every row is in SD units
  percent  each video difference divided by that video's fish-associated mean,
           giving a fractional change

Physical mean and 95% confidence interval are annotated on the right of each row in
both cases, so no information is lost by the rescaling.

Existing outputs are not overwritten: files carry a _forest / _forest_pct suffix.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = (
    HERE
    / "vortex_outputs"
    / "revision_analysis"
    / "pressure_conditioned_vortices"
    / "vortex_paired_video_differences_boai_95_thr0.99.csv"
)
DEFAULT_OUT = HERE / "Royal_Society_Interface___LaTeX_Template" / "figures"

PREFIX = "cylinder_minus_fish_"

# Ordered as in analyze_pressure_conditioned_vortices.SEGMENT_METRICS.
METRICS = (
    "mean_area",
    "max_area",
    "mean_equivalent_radius",
    "max_equivalent_radius",
    "displacement",
    "path_length",
    "straightness",
    "duration_frames",
    "duration_ratio",
    "mean_speed",
    "max_speed",
    "mean_core_q",
    "max_core_q",
    "mean_body_distance",
    "mean_sensor_distance",
    "mean_core_x",
    "mean_core_y",
    "vortex_count",
    "total_vortex_frames",
    "mean_simultaneous_vortices",
    "max_simultaneous_vortices",
)

# Short names for the axis; units live in the right-hand annotation instead.
LABELS = {
    "mean_area": "Mean area",
    "max_area": "Maximum area",
    "mean_equivalent_radius": "Mean equivalent radius",
    "max_equivalent_radius": "Maximum equivalent radius",
    "displacement": "Displacement",
    "path_length": "Path length",
    "straightness": "Straightness",
    "duration_frames": "Duration",
    "duration_ratio": "Duration / segment length",
    "mean_speed": "Mean speed",
    "max_speed": "Maximum speed",
    "mean_core_q": "Mean core $Q$",
    "max_core_q": "Maximum core $Q$",
    "mean_body_distance": "Mean body distance",
    "mean_sensor_distance": "Mean sensor distance",
    "mean_core_x": "Mean core $x$",
    "mean_core_y": "Mean core $y$",
    "vortex_count": "Vortices per segment",
    "total_vortex_frames": "Total vortex-frame occupancy",
    "mean_simultaneous_vortices": "Mean simultaneous vortices",
    "max_simultaneous_vortices": "Maximum simultaneous vortices",
}

UNITS = {
    "mean_area": "cells",
    "max_area": "cells",
    "mean_equivalent_radius": "cells",
    "max_equivalent_radius": "cells",
    "displacement": "cells",
    "path_length": "cells",
    "straightness": "",
    "duration_frames": "frames",
    "duration_ratio": "",
    "mean_speed": "cells/frame",
    "max_speed": "cells/frame",
    "mean_core_q": "",
    "max_core_q": "",
    "mean_body_distance": "cells",
    "mean_sensor_distance": "cells",
    "mean_core_x": "cells",
    "mean_core_y": "cells",
    "vortex_count": "",
    "total_vortex_frames": "frames",
    "mean_simultaneous_vortices": "",
    "max_simultaneous_vortices": "",
}

# Matches MODEL_COLOURS in analyze_pressure_conditioned_vortices.py, so the class
# colours agree with the other figures in the article.
COLOUR_CYLINDER = "#ff7f0e"  # cylinder-associated summary larger
COLOUR_FISH = "#1f77b4"  # fish-associated summary larger

RC = {
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 7.5,
    "text.usetex": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
}


def summarise(frame: pd.DataFrame, metric: str, scale: str) -> dict | None:
    """Physical mean/CI plus the rescaled per-video values used for plotting."""
    raw = frame[PREFIX + metric].to_numpy(float)
    raw = raw[np.isfinite(raw)]
    if raw.size < 2:
        return None

    n = raw.size
    mean = float(raw.mean())
    sd = float(raw.std(ddof=1))
    half = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n))

    if sd == 0.0:
        return None

    if scale == "sd":
        # Dividing by the descriptor's own SD puts every row on a common scale, so
        # the distance of the cloud from zero reads directly as a standardised
        # effect size. The drawn interval is the physical CI under the same divisor.
        shown = raw / sd
        shown_lo, shown_hi = (mean - half) / sd, (mean + half) / sd
    else:
        base = frame["fish_" + metric].to_numpy(float)
        base = base[np.isfinite(frame[PREFIX + metric].to_numpy(float))]
        with np.errstate(divide="ignore", invalid="ignore"):
            shown = np.where(np.abs(base) > 0, raw / np.abs(base), np.nan)
        shown = shown[np.isfinite(shown)]
        if shown.size < 2:
            return None
        # The mean of per-video ratios is not the ratio of the means, so this row's
        # interval is computed on the ratios themselves.
        shown_half = float(
            stats.t.ppf(0.975, shown.size - 1) * shown.std(ddof=1) / np.sqrt(shown.size)
        )
        shown_lo, shown_hi = shown.mean() - shown_half, shown.mean() + shown_half

    return {
        "metric": metric,
        "values": shown,
        "centre": float(np.mean(shown)),
        "shown_lo": float(shown_lo),
        "shown_hi": float(shown_hi),
        # Colour, ordering and emphasis all follow the physical difference, so they
        # never contradict the annotation.
        "mean": mean,
        "lo": mean - half,
        "hi": mean + half,
        "dz": mean / sd,
        "excludes_zero": (mean - half) * (mean + half) > 0,
    }


def format_annotation(row: dict) -> str:
    unit = UNITS[row["metric"]]
    suffix = f" {unit}" if unit else ""
    return f"{row['mean']:+.3g} [{row['lo']:+.3g}, {row['hi']:+.3g}]{suffix}"


def build(frame: pd.DataFrame, scale: str, output_base: Path, seed: int = 20240917) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update(RC)

    rows = [summarise(frame, metric, scale) for metric in METRICS]
    rows = [row for row in rows if row is not None]
    # Strongest cylinder-ward effect at the top, strongest fish-ward at the bottom.
    # Ordering by the standardised physical difference keeps both scales in step.
    rows.sort(key=lambda row: row["dz"], reverse=True)

    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(8.6, 0.42 * len(rows) + 1.5))

    for index, row in enumerate(rows):
        y = len(rows) - 1 - index
        colour = COLOUR_CYLINDER if row["mean"] > 0 else COLOUR_FISH
        solid = row["excludes_zero"]

        ax.axhline(y, color="#000000", alpha=0.06, linewidth=0.6, zorder=0)

        values = row["values"]
        jitter = rng.uniform(-0.17, 0.17, size=values.size)
        ax.scatter(
            values,
            np.full(values.size, y) + jitter,
            s=17,
            color=colour,
            alpha=0.75 if solid else 0.28,
            linewidths=0,
            zorder=3,
        )

        ax.plot(
            [row["shown_lo"], row["shown_hi"]],
            [y, y],
            color=colour,
            linewidth=1.9 if solid else 1.1,
            alpha=0.95 if solid else 0.4,
            solid_capstyle="round",
            zorder=4,
        )
        ax.plot(
            [row["centre"]],
            [y],
            marker="D",
            markersize=4.2,
            color=colour,
            markeredgecolor="white",
            markeredgewidth=0.6,
            alpha=1.0 if solid else 0.45,
            zorder=5,
        )

        ax.annotate(
            format_annotation(row),
            xy=(1.015, y),
            xycoords=("axes fraction", "data"),
            va="center",
            ha="left",
            fontsize=7,
            color="#222222" if solid else "#999999",
        )

    ax.axvline(0.0, color="black", linestyle=(0, (3, 3)), linewidth=0.9, zorder=2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([LABELS[row["metric"]] for row in reversed(rows)])
    ax.set_ylim(-0.75, len(rows) - 0.25)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    # The direction of the axis is carried by the legend and the caption, so no
    # second annotation line is drawn here.
    if scale == "sd":
        ax.set_xlabel("Paired source-video difference (SD units)")
    else:
        ax.set_xlabel("Paired source-video difference (fraction of fish-associated mean)")
        ax.xaxis.set_major_formatter(lambda v, _pos: f"{v:+.0%}".replace("+0%", "0"))

    handles = [
        Line2D([], [], marker="o", linestyle="", color=COLOUR_CYLINDER, markersize=5,
               label="Cylinder-associated larger"),
        Line2D([], [], marker="o", linestyle="", color=COLOUR_FISH, markersize=5,
               label="Fish-associated larger"),
        Line2D([], [], marker="D", linestyle="-", color="#555555", markersize=4.2,
               linewidth=1.9, label="Mean and 95% $t$-based CI"),
        Line2D([], [], marker="o", linestyle="", color="#999999", markersize=5, alpha=0.4,
               label="Interval contains zero"),
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=4,
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.4,
    )

    output_base.parent.mkdir(parents=True, exist_ok=True)
    # The tag contains dots (thr0.99), so with_suffix would truncate it.
    for suffix in (".pdf", ".png", ".eps"):
        fig.savefig(Path(f"{output_base}{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_base}.pdf/.png/.eps  ({len(rows)} descriptors)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--tag", default="boai_95_thr0.99")
    parser.add_argument("--scale", choices=("sd", "percent", "both"), default="both")
    args = parser.parse_args()

    frame = pd.read_csv(args.csv)
    print(f"read {len(frame)} paired source videos from {args.csv.name}")

    if args.scale in ("sd", "both"):
        build(frame, "sd", args.output_dir / f"vortex_paired_video_differences_forest_{args.tag}")
    if args.scale in ("percent", "both"):
        build(frame, "percent", args.output_dir / f"vortex_paired_video_differences_forest_pct_{args.tag}")


if __name__ == "__main__":
    main()
