import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SnapshotSeries:
    values: np.ndarray
    length: int


def load_matrix(path: Path) -> np.ndarray:
    matrix = np.genfromtxt(path, delimiter=",")
    if matrix.size == 0:
        raise ValueError(f"Matrix is empty: {path}")
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if np.isnan(matrix).any():
        matrix = np.nan_to_num(matrix, nan=np.nan)
    return matrix.astype(np.float64, copy=False)


def load_lengths(path: Path) -> list[int]:
    lengths: list[int] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = row.get("length", "")
            try:
                lengths.append(int(float(raw)))
            except ValueError:
                continue
    return lengths


def infer_length_from_column(column: np.ndarray) -> int:
    finite = np.isfinite(column)
    if not np.any(finite):
        return 0
    return int(np.max(np.where(finite))) + 1


def extract_series(
    matrix: np.ndarray,
    lengths: list[int],
    min_length: int,
    label: str,
) -> list[SnapshotSeries]:
    series_list: list[SnapshotSeries] = []
    max_cols = matrix.shape[1]
    if len(lengths) != max_cols:
        print(
            f"⚠️  {label} length metadata ({len(lengths)}) does not match columns ({max_cols}). "
            f"Using min."
        )
    usable = max_cols
    for col in range(usable):
        length = lengths[col] if col < len(lengths) else 0
        if length <= 0 or length > matrix.shape[0]:
            length = infer_length_from_column(matrix[:, col])
        if length < min_length:
            continue
        length = min(length, matrix.shape[0])
        if length <= 0:
            continue
        values = matrix[:length, col].astype(np.float64, copy=True)
        series_list.append(SnapshotSeries(values=values, length=length))
    return series_list


def fill_internal_nans_1d(values: np.ndarray) -> np.ndarray:
    if not np.isnan(values).any():
        return values
    idx = np.arange(values.size)
    good = np.isfinite(values)
    if not np.any(good):
        return np.zeros_like(values)
    filled = np.interp(idx, idx[good], values[good])
    return filled


def zero_mean_series(values: np.ndarray) -> np.ndarray:
    mean_val = float(np.nanmean(values)) if values.size else 0.0
    if not np.isfinite(mean_val):
        return values
    return values - mean_val


def normalize_series(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    max_abs = float(np.nanmax(np.abs(values)))
    if not np.isfinite(max_abs) or max_abs == 0.0:
        return values
    return values / max_abs


def resample_series(series: np.ndarray, target_length: int, method: str) -> np.ndarray:
    if series.size == target_length:
        return series
    if series.size <= 1:
        return np.full(target_length, series[0] if series.size == 1 else 0.0, dtype=np.float64)
    old_x = np.linspace(0.0, 1.0, series.size)
    new_x = np.linspace(0.0, 1.0, target_length)
    if method == "cubic":
        if series.size < 4:
            return np.interp(new_x, old_x, series)
        try:
            from scipy.interpolate import CubicSpline
        except Exception as exc:
            raise RuntimeError("Cubic interpolation requires scipy. Install scipy or use --interp linear.") from exc
        cs = CubicSpline(old_x, series, bc_type="natural")
        return cs(new_x)
    return np.interp(new_x, old_x, series)


def build_resampled_matrix(series_list: list[SnapshotSeries], method: str) -> tuple[np.ndarray, int]:
    if not series_list:
        return np.empty((0, 0), dtype=np.float64), 0
    max_len = max(s.length for s in series_list)
    matrix = np.zeros((max_len, len(series_list)), dtype=np.float64)
    for col, series in enumerate(series_list):
        values = fill_internal_nans_1d(series.values)
        resampled = resample_series(values, max_len, method)
        resampled = zero_mean_series(resampled)
        matrix[:, col] = resampled
    return matrix, max_len


def balance_series(
    fish_series: list[SnapshotSeries],
    cyl_series: list[SnapshotSeries],
) -> tuple[list[SnapshotSeries], list[SnapshotSeries]]:
    if not fish_series or not cyl_series:
        return fish_series, cyl_series
    target = min(len(fish_series), len(cyl_series))
    fish_sorted = sorted(fish_series, key=lambda s: s.length, reverse=True)
    cyl_sorted = sorted(cyl_series, key=lambda s: s.length, reverse=True)
    return fish_sorted[:target], cyl_sorted[:target]


def has_tail_spike(values: np.ndarray, tail_fraction: float, spike_ratio: float, min_tail: int) -> bool:
    n = int(values.size)
    if n < 2:
        return False
    tail_len = max(int(round(n * tail_fraction)), min_tail)
    tail_len = min(tail_len, n - 1)
    if tail_len <= 0:
        return False
    body = values[: n - tail_len]
    tail = values[n - tail_len :]
    if body.size == 0 or tail.size == 0:
        return False
    body_max = float(np.nanmax(np.abs(body)))
    tail_max = float(np.nanmax(np.abs(tail)))
    if not np.isfinite(body_max) or not np.isfinite(tail_max):
        return False
    baseline = max(body_max, 1e-9)
    return tail_max > spike_ratio * baseline


def filter_tail_spikes(
    series_list: list[SnapshotSeries],
    tail_fraction: float,
    spike_ratio: float,
    min_tail: int,
    label: str,
) -> tuple[list[SnapshotSeries], list[SnapshotSeries]]:
    if not series_list:
        return series_list, []
    kept: list[SnapshotSeries] = []
    removed: list[SnapshotSeries] = []
    for series in series_list:
        values = fill_internal_nans_1d(series.values)
        if has_tail_spike(values, tail_fraction, spike_ratio, min_tail):
            removed.append(series)
        else:
            kept.append(series)
    if removed:
        print(
            f"⚠️  Removed {len(removed)} {label} snapshots due to tail spikes "
            f"(ratio>{spike_ratio}, tail_frac={tail_fraction})."
        )
    return kept, removed


def normalize_snapshots(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if matrix.size == 0:
        return matrix, np.empty((0,), dtype=np.float64)
    scales = np.nanmax(np.abs(matrix), axis=0)
    invalid = ~np.isfinite(scales) | (scales == 0.0)
    if np.any(invalid):
        scales = scales.copy()
        scales[invalid] = 1.0
    return matrix / scales, scales


def compute_modes(matrix: np.ndarray, energy_target: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    if matrix.size == 0:
        return np.empty((0, 0)), np.empty((0,)), np.empty((0,)), 0
    u, s, _ = np.linalg.svd(matrix, full_matrices=False)
    energy = s * s
    total = float(np.sum(energy))
    if total == 0.0:
        return u[:, :1], s, np.zeros_like(s), 1
    cumulative = np.cumsum(energy) / total
    k = int(np.searchsorted(cumulative, energy_target) + 1)
    k = min(k, u.shape[1])
    return u[:, :k], s, cumulative, k


def plot_debug_resample(
    raw_series: np.ndarray,
    resampled_series: np.ndarray,
    dataset: str,
    method: str,
    raw_first: np.ndarray | None,
    resampled_first: np.ndarray | None,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    ax_main, ax_first = axes

    raw_x = np.linspace(0.0, 1.0, raw_series.size)
    resampled_x = np.linspace(0.0, 1.0, resampled_series.size)
    ax_main.plot(raw_x, raw_series, label=f"raw ({raw_series.size})", linewidth=1.4)
    ax_main.plot(resampled_x, resampled_series, label=f"resampled ({resampled_series.size})", linewidth=1.2)
    ax_main.set_title(f"{dataset} resampling ({method})")
    ax_main.set_xlabel("normalized time")
    ax_main.set_ylabel("pressure")
    ax_main.legend(loc="upper right")
    ax_main.grid(alpha=0.2)

    if raw_first is not None and resampled_first is not None:
        raw_first_x = np.linspace(0.0, 1.0, raw_first.size)
        resampled_first_x = np.linspace(0.0, 1.0, resampled_first.size)
        ax_first.plot(raw_first_x, raw_first, label=f"raw ({raw_first.size})", linewidth=1.4)
        ax_first.plot(
            resampled_first_x,
            resampled_first,
            label=f"resampled ({resampled_first.size})",
            linewidth=1.2,
        )
        ax_first.set_title("First column")
        ax_first.set_xlabel("normalized time")
        ax_first.legend(loc="upper right")
        ax_first.grid(alpha=0.2)
    else:
        ax_first.set_axis_off()

    plt.show()
    plt.close(fig)


def mean_resampled_series(
    series_list: list[SnapshotSeries],
    target_length: int,
    method: str,
) -> np.ndarray | None:
    if not series_list:
        return None
    matrix = np.zeros((target_length, len(series_list)), dtype=np.float64)
    for col, series in enumerate(series_list):
        values = fill_internal_nans_1d(series.values)
        resampled = resample_series(values, target_length, method)
        resampled = zero_mean_series(resampled)
        matrix[:, col] = resampled
    return np.mean(matrix, axis=1)


def plot_mean_snapshots(
    mean_fish: np.ndarray | None,
    mean_cyl: np.ndarray | None,
    dataset: str,
    method: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    if mean_fish is not None:
        x_fish = np.linspace(0.0, 1.0, mean_fish.size)
        ax.plot(x_fish, mean_fish, label="fish mean", linewidth=1.5)
    if mean_cyl is not None:
        x_cyl = np.linspace(0.0, 1.0, mean_cyl.size)
        ax.plot(x_cyl, mean_cyl, label="cylinder mean", linewidth=1.5)
    ax.set_title(f"{dataset} mean snapshots ({method})")
    ax.set_xlabel("normalized time")
    ax.set_ylabel("pressure")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.2)
    plt.show()
    plt.close(fig)


def plot_cylinder_snapshots(
    resampled_matrix: np.ndarray,
    n_fish: int,
    n_cyl: int,
    dataset: str,
) -> None:
    import matplotlib.pyplot as plt

    if n_cyl <= 0:
        return

    start = n_fish
    end = n_fish + n_cyl
    time_axis = np.linspace(0.0, 1.0, resampled_matrix.shape[0])

    for idx, col in enumerate(range(start, end)):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(time_axis, resampled_matrix[:, col], linewidth=1.2)
        ax.set_title(f"{dataset} cylinder snapshot {idx} (col {col})")
        ax.set_xlabel("normalized time")
        ax.set_ylabel("pressure (normalized)")
        ax.grid(alpha=0.2)
        plt.show()
    plt.close(fig)


def plot_tail_spike_snapshots(
    removed_series: list[SnapshotSeries],
    dataset: str,
    label: str,
) -> None:
    import matplotlib.pyplot as plt

    for idx, series in enumerate(removed_series):
        values = fill_internal_nans_1d(series.values)
        time_axis = np.linspace(0.0, 1.0, values.size)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(time_axis, values, linewidth=1.2)
        ax.set_title(f"{dataset} {label} tail-spike {idx} (len {values.size})")
        ax.set_xlabel("normalized time")
        ax.set_ylabel("pressure")
        ax.grid(alpha=0.2)
        plt.show()
    plt.close(fig)


def plot_energy_curve(
    cumulative_energy: np.ndarray,
    energy_target: float,
    k: int,
    dataset: str,
) -> None:
    import matplotlib.pyplot as plt

    if cumulative_energy.size == 0:
        return

    x = np.arange(1, cumulative_energy.size + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, cumulative_energy, linewidth=1.4, label="cumulative energy")
    ax.axhline(energy_target, color="red", linestyle="--", linewidth=1.0, label=f"{energy_target:.2f} target")
    ax.axvline(k, color="black", linestyle="--", linewidth=1.0, label=f"k={k}")
    if 1 <= k <= cumulative_energy.size:
        ax.scatter([k], [cumulative_energy[k - 1]], color="black", s=20, zorder=3)
    ax.set_title(f"{dataset} energy vs modes")
    ax.set_xlabel("number of modes")
    ax.set_ylabel("cumulative energy")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right")
    plt.show()
    plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Compute SVD pressure modes from vortex snapshots.")
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=repo_root / "python/vortexAnalysis_v2/vortex_outputs/pressure_snapshots",
        help="Directory containing pressure snapshot matrices and meta CSVs",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="nadia,thomas,boai,full",
        help="Comma-separated dataset list to process",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=100,
        help="Minimum snapshot length (frames) to keep",
    )
    parser.add_argument(
        "--energy",
        type=float,
        default=0.99,
        help="Energy threshold for selecting modes (0-1)",
    )
    parser.add_argument(
        "--interp",
        type=str,
        choices=["linear", "cubic"],
        default="cubic",
        help="Interpolation method used to resample each snapshot to the max length",
    )
    parser.add_argument(
        "--debug-plot",
        action="store_true",
        help="Plot the first raw snapshot vs its resampled version for a quick visual check",
    )
    parser.add_argument(
        "--plot-means",
        action="store_true",
        help="Plot mean snapshots (fish vs cylinder) after resampling",
    )
    parser.add_argument(
        "--plot-cylinder-all",
        action="store_true",
        help="Plot every cylinder snapshot after resampling/normalization",
    )
    parser.add_argument(
        "--plot-tail-spikes",
        action="store_true",
        help="Plot snapshots removed by the tail-spike filter",
    )
    parser.add_argument(
        "--plot-energy",
        action="store_true",
        help="Plot cumulative energy vs number of modes for each dataset",
    )
    parser.add_argument(
        "--spike-ratio",
        type=float,
        default=5.0,
        help="Tail spike ratio threshold (tail max / body max) to drop snapshots",
    )
    parser.add_argument(
        "--spike-tail-frac",
        type=float,
        default=0.15,
        help="Fraction of the series considered as tail for spike detection",
    )
    parser.add_argument(
        "--spike-min-tail",
        type=int,
        default=5,
        help="Minimum tail length (points) for spike detection",
    )
    parser.add_argument(
        "--no-tail-spike-filter",
        dest="tail_spike_filter",
        action="store_false",
        help="Disable tail spike filtering",
    )
    parser.set_defaults(tail_spike_filter=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=repo_root / "python/vortexAnalysis_v2/vortex_outputs/pressure_modes",
        help="Output directory for saved modes",
    )
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    snapshot_dir = args.snapshot_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_shown = False

    for dataset in datasets:
        fish_meta = snapshot_dir / f"pressure_snapshots_{dataset}_fish_meta.csv"
        fish_mat = snapshot_dir / f"pressure_snapshots_{dataset}_fish.csv"
        cyl_meta = snapshot_dir / f"pressure_snapshots_{dataset}_cylinder_meta.csv"
        cyl_mat = snapshot_dir / f"pressure_snapshots_{dataset}_cylinder.csv"

        if not fish_meta.exists() or not fish_mat.exists():
            print(f"⚠️  Missing fish files for {dataset}, skipping.")
            continue
        if not cyl_meta.exists() or not cyl_mat.exists():
            print(f"⚠️  Missing cylinder files for {dataset}, skipping.")
            continue

        fish_lengths = load_lengths(fish_meta)
        cyl_lengths = load_lengths(cyl_meta)

        fish_matrix = load_matrix(fish_mat)
        cyl_matrix = load_matrix(cyl_mat)

        fish_series = extract_series(fish_matrix, fish_lengths, args.min_length, f"{dataset} fish")
        cyl_series = extract_series(cyl_matrix, cyl_lengths, args.min_length, f"{dataset} cylinder")
        if args.tail_spike_filter:
            fish_series, fish_removed = filter_tail_spikes(
                fish_series,
                args.spike_tail_frac,
                args.spike_ratio,
                args.spike_min_tail,
                f"{dataset} fish",
            )
            cyl_series, cyl_removed = filter_tail_spikes(
                cyl_series,
                args.spike_tail_frac,
                args.spike_ratio,
                args.spike_min_tail,
                f"{dataset} cylinder",
            )
            if args.plot_tail_spikes:
                plot_tail_spike_snapshots(fish_removed, dataset, "fish")
                plot_tail_spike_snapshots(cyl_removed, dataset, "cylinder")
        fish_series, cyl_series = balance_series(fish_series, cyl_series)
        all_series = fish_series + cyl_series

        if not all_series:
            print(f"⚠️  No snapshots above length {args.min_length} for {dataset}.")
            continue

        if len(fish_series) != len(cyl_series):
            print(
                f"⚠️  Unbalanced data for {dataset}: fish={len(fish_series)} "
                f"cylinder={len(cyl_series)}"
            )

        resampled_matrix, target_len = build_resampled_matrix(all_series, args.interp)
        resampled_matrix, scales = normalize_snapshots(resampled_matrix)
        n_fish = len(fish_series)
        n_cyl = len(cyl_series)
        if args.plot_means:
            mean_fish = np.mean(resampled_matrix[:, :n_fish], axis=1) if n_fish > 0 else None
            mean_cyl = (
                np.mean(resampled_matrix[:, n_fish : n_fish + n_cyl], axis=1) if n_cyl > 0 else None
            )
            plot_mean_snapshots(mean_fish, mean_cyl, dataset, args.interp)
        if args.plot_cylinder_all:
            plot_cylinder_snapshots(resampled_matrix, len(fish_series), len(cyl_series), dataset)
        if args.debug_plot and not debug_shown and all_series:
            series_index = next(
                (i for i, s in enumerate(all_series) if s.length == args.min_length),
                1 if len(all_series) > 1 else 0,
            )
            selected_series = all_series[series_index]
            raw_values = fill_internal_nans_1d(selected_series.values)
            raw_values = zero_mean_series(raw_values)
            raw_plot = normalize_series(raw_values)
            resampled = resample_series(raw_values, target_len, args.interp)
            resampled = zero_mean_series(resampled)
            resampled = normalize_series(resampled)
            first_series = all_series[0] if all_series else None
            raw_first = fill_internal_nans_1d(first_series.values) if first_series else None
            if raw_first is not None:
                raw_first = zero_mean_series(raw_first)
                raw_first_plot = normalize_series(raw_first)
                resampled_first = resample_series(raw_first, target_len, args.interp)
                resampled_first = zero_mean_series(resampled_first)
                resampled_first = normalize_series(resampled_first)
            else:
                raw_first_plot = None
                resampled_first = None
            plot_debug_resample(raw_plot, resampled, dataset, args.interp, raw_first_plot, resampled_first)
            debug_shown = True
        modes, singular_values, cumulative_energy, k = compute_modes(resampled_matrix, args.energy)
        if args.plot_energy:
            plot_energy_curve(cumulative_energy, args.energy, k, dataset)

        out_path = out_dir / f"pressure_modes_{dataset}.npz"
        np.savez(
            out_path,
            modes=modes,
            singular_values=singular_values,
            cumulative_energy=cumulative_energy,
            energy_target=args.energy,
            k=k,
            scales=scales,
            normalization="per_snapshot_maxabs",
            target_length=target_len,
            n_series=resampled_matrix.shape[1],
            min_length=args.min_length,
        )

        print(
            f"✅ {dataset}: {resampled_matrix.shape[1]} snapshots -> "
            f"{k} modes ({args.energy:.2f} energy) saved to {out_path}"
        )


if __name__ == "__main__":
    main()
