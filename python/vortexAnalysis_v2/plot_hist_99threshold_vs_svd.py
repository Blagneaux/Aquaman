from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CAMBRIDGE_RC = {
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
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


def apply_cambridge_style() -> None:
    plt.rcParams.update(CAMBRIDGE_RC)


def save_figure_multi(fig, save_path: Path) -> None:
    base = save_path
    if base.suffix:
        base = base.with_suffix("")
    fig.savefig(f"{base}.eps", format="eps", bbox_inches="tight")
    fig.savefig(f"{base}.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(f"{base}.png", format="png", dpi=150, bbox_inches="tight")
    if save_path.suffix:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")


labels = ["fish", "none", "cylinder"]
datasets = ["nadia", "thomas", "boai", "full"]
dataset_titles = {
    "nadia": "raw model",
    "thomas": "constrained model",
    "boai": "x deformation model",
    "full": "full model",
}

# Counts per dataset for each minimum distance-to-body-grid scenario.
# Order of datasets: nadia, thomas, boai, full.
pred3 = [[20, 403, 65], [31, 328, 52], [49, 299, 115], [30, 350, 52]]
pred6 = [[14, 393, 67], [10, 343, 38], [19, 285, 132], [8, 333, 56]]
pred12 = [[22, 326, 87], [18, 252, 81], [28, 214, 148], [17, 245, 88]]

distance_scenarios = [3, 6, 12]
pred_by_distance = {3: pred3, 6: pred6, 12: pred12}


def build_summary() -> dict[str, dict[int, tuple[int, int, int, int]]]:
    summary: dict[str, dict[int, tuple[int, int, int, int]]] = {}
    for dataset_idx, dataset in enumerate(datasets):
        for distance in distance_scenarios:
            counts = pred_by_distance[distance][dataset_idx]
            # Raw values had fish/cylinder inverted; fix ordering here.
            cyl_count, none_count, fish_count = counts
            total = fish_count + none_count + cyl_count
            summary.setdefault(dataset, {})[distance] = (fish_count, none_count, cyl_count, total)
    return summary


def plot_dataset_distribution(ax, dataset: str, dist_summary) -> None:
    threshold = 0.99
    colors = {
        "fish": "#4C78A8",
        "none": "#BAB0AC",
        "cylinder": "#F58518",
    }
    dist_stats = dist_summary.get(dataset, {})

    x = np.arange(len(distance_scenarios))
    x_labels = [str(d) for d in distance_scenarios]

    fish_vals: list[float] = []
    none_vals: list[float] = []
    cyl_vals: list[float] = []
    fish_counts: list[int] = []
    none_counts: list[int] = []
    cyl_counts: list[int] = []

    for distance in distance_scenarios:
        fish_count, none_count, cyl_count, total = dist_stats.get(distance, (0, 0, 0, 0))
        fish_counts.append(fish_count)
        none_counts.append(none_count)
        cyl_counts.append(cyl_count)
        if total <= 0:
            fish_vals.append(0.0)
            none_vals.append(0.0)
            cyl_vals.append(0.0)
        else:
            fish_vals.append(fish_count / total)
            none_vals.append(none_count / total)
            cyl_vals.append(cyl_count / total)

    bottom = np.zeros(len(distance_scenarios))
    ax.bar(x, fish_vals, label=f"fish > {threshold*100:.0f}%", color=colors["fish"], bottom=bottom)
    for i, frac in enumerate(fish_vals):
        if frac > 0:
            total = fish_counts[i] + none_counts[i] + cyl_counts[i]
            pct = (fish_counts[i] / total * 100.0) if total else 0.0
            ax.text(
                x[i],
                bottom[i] + frac / 2,
                f"{fish_counts[i]} ({pct:.1f}%)",
                ha="center",
                va="center",
                fontsize=8,
                color="white",
            )

    bottom += np.array(fish_vals)
    ax.bar(x, none_vals, label="none", color=colors["none"], bottom=bottom)
    for i, frac in enumerate(none_vals):
        if frac > 0:
            total = fish_counts[i] + none_counts[i] + cyl_counts[i]
            pct = (none_counts[i] / total * 100.0) if total else 0.0
            ax.text(
                x[i],
                bottom[i] + frac / 2,
                f"{none_counts[i]} ({pct:.1f}%)",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )

    bottom += np.array(none_vals)
    ax.bar(x, cyl_vals, label=f"cylinder > {threshold*100:.0f}%", color=colors["cylinder"], bottom=bottom)
    for i, frac in enumerate(cyl_vals):
        if frac > 0:
            total = fish_counts[i] + none_counts[i] + cyl_counts[i]
            pct = (cyl_counts[i] / total * 100.0) if total else 0.0
            ax.text(
                x[i],
                bottom[i] + frac / 2,
                f"{cyl_counts[i]} ({pct:.1f}%)",
                ha="center",
                va="center",
                fontsize=8,
                color="white",
            )

    ax.set_xticks(x, x_labels)
    ax.set_xlabel("min distance to body grid")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("fraction of samples")
    title = dataset_titles.get(dataset, dataset)
    ax.set_title(f"{title} (threshold={threshold:.2f})")
    ax.grid(axis="y", alpha=0.2)


def plot_distribution() -> None:
    apply_cambridge_style()
    dist_summary = build_summary()
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = repo_root / "python/vortexAnalysis_v2/vortex_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, dataset in zip(axes.flat, datasets):
        plot_dataset_distribution(ax, dataset, dist_summary)

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=3)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])
    save_figure_multi(fig, output_dir / "paper_ready_distribution_min_distance")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    plot_dataset_distribution(ax, "boai", dist_summary)
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=3)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])
    save_figure_multi(fig, output_dir / "paper_ready_distribution_min_distance_x_deformation")
    plt.close(fig)


if __name__ == "__main__":
    plot_distribution()
