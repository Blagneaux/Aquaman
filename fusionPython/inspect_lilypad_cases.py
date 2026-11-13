"""Stream diagnostics for the regenerated LilyPad pressure/vorticity cases."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np


DATA_ROOT = Path(r"E:\simuChina\newBDD")
OUTPUT_ROOT = Path(
    r"C:\Users\blagn771\OneDrive - ENS RENNES\Documents\Aquaman\Aquaman"
    r"\fusionPython\newbdd_edge_diagnostics"
)
CASES = {
    "Re1000_h6.0160003": 6.0160003,
    "Re1000_h32.0": 32.0,
}
NX = 256
NY = 128
CYLINDER_X = 4 * 128 / 7
CYLINDER_DIAMETER = 6.4


def read_time_file(path: Path) -> dict:
    times: list[float] = []
    pressure_min = np.inf
    pressure_max = -np.inf
    pressure_nan = 0
    pressure_count = 0
    row_widths: set[int] = set()

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line or line.startswith("%"):
                continue
            values = np.fromstring(line.rstrip(";\r\n"), sep=" ")
            if values.size == 0:
                continue
            times.append(float(values[0]))
            surface_pressure = values[1:]
            row_widths.add(int(surface_pressure.size))
            pressure_nan += int(np.isnan(surface_pressure).sum())
            finite = surface_pressure[np.isfinite(surface_pressure)]
            if finite.size:
                pressure_min = min(pressure_min, float(finite.min()))
                pressure_max = max(pressure_max, float(finite.max()))
                pressure_count += int(finite.size)

    time_values = np.asarray(times)
    dt = np.diff(time_values)
    return {
        "n_rows": int(time_values.size),
        "time_start": float(time_values[0]) if time_values.size else None,
        "time_end": float(time_values[-1]) if time_values.size else None,
        "dt_min": float(dt.min()) if dt.size else None,
        "dt_max": float(dt.max()) if dt.size else None,
        "dt_mean": float(dt.mean()) if dt.size else None,
        "strictly_increasing": bool(np.all(dt > 0)) if dt.size else True,
        "surface_pressure_widths": sorted(row_widths),
        "surface_pressure_min": float(pressure_min) if pressure_count else None,
        "surface_pressure_max": float(pressure_max) if pressure_count else None,
        "surface_pressure_nan": pressure_nan,
    }


def spatial_metrics(field: np.ndarray) -> dict:
    absolute = np.abs(field)
    maximum_index = np.unravel_index(np.nanargmax(absolute), field.shape)
    return {
        "min": float(np.nanmin(field)),
        "max": float(np.nanmax(field)),
        "mean": float(np.nanmean(field)),
        "rms": float(np.sqrt(np.nanmean(field**2))),
        "abs_max_location": [int(maximum_index[0]), int(maximum_index[1])],
        "inlet_abs_mean_x0_11": float(np.nanmean(absolute[:12, :])),
        "inlet_lower_corner_abs_mean": float(np.nanmean(absolute[:12, :8])),
        "inlet_upper_corner_abs_mean": float(np.nanmean(absolute[:12, -8:])),
        "bottom_four_rows_abs_mean": float(np.nanmean(absolute[:, :4])),
        "top_four_rows_abs_mean": float(np.nanmean(absolute[:, -4:])),
        "central_abs_mean": float(np.nanmean(absolute[12:244, 8:120])),
    }


def inspect_csv(path: Path, label: str) -> tuple[dict, np.ndarray, list[int]]:
    with path.open("r", encoding="utf-8") as handle:
        first = np.fromstring(handle.readline().rstrip("\r\n"), sep=",")
        n_snapshots = int(first.size)
        sample_indices = sorted(
            set(
                [
                    0,
                    n_snapshots // 10,
                    n_snapshots // 2,
                    9 * n_snapshots // 10,
                    n_snapshots - 1,
                ]
            )
        )
        samples = np.empty((len(sample_indices), NX * NY), dtype=np.float32)
        temporal_mean = np.empty(NX * NY, dtype=np.float32)
        temporal_std = np.empty(NX * NY, dtype=np.float32)
        snapshot_min = np.full(n_snapshots, np.inf, dtype=np.float64)
        snapshot_max = np.full(n_snapshots, -np.inf, dtype=np.float64)
        snapshot_sum_sq = np.zeros(n_snapshots, dtype=np.float64)

        global_min = np.inf
        global_max = -np.inf
        finite_count = 0
        nan_count = 0
        inf_count = 0
        bad_width_rows = 0

        def consume(row: int, values: np.ndarray) -> None:
            nonlocal global_min, global_max, finite_count, nan_count, inf_count
            nonlocal bad_width_rows
            if values.size != n_snapshots:
                bad_width_rows += 1
                return
            finite = np.isfinite(values)
            nan_count += int(np.isnan(values).sum())
            inf_count += int(np.isinf(values).sum())
            if finite.any():
                finite_values = values[finite]
                global_min = min(global_min, float(finite_values.min()))
                global_max = max(global_max, float(finite_values.max()))
                finite_count += int(finite_values.size)
            samples[:, row] = values[sample_indices]
            temporal_mean[row] = float(np.nanmean(values))
            temporal_std[row] = float(np.nanstd(values))
            snapshot_min[:] = np.minimum(snapshot_min, values)
            snapshot_max[:] = np.maximum(snapshot_max, values)
            snapshot_sum_sq[:] += values * values

        consume(0, first)
        for row, line in enumerate(handle, start=1):
            if row >= NX * NY:
                bad_width_rows += 1
                continue
            consume(row, np.fromstring(line.rstrip("\r\n"), sep=","))
            if row % 4096 == 0:
                print(f"{label}: {row}/{NX * NY} spatial rows", flush=True)

        n_spatial_rows = row + 1

    sample_maps = samples.reshape(len(sample_indices), NX, NY)
    temporal_mean_map = temporal_mean.reshape(NX, NY)
    temporal_std_map = temporal_std.reshape(NX, NY)
    snapshot_rms = np.sqrt(snapshot_sum_sq / (NX * NY))
    snapshot_abs_peak = np.maximum(np.abs(snapshot_min), np.abs(snapshot_max))
    largest_snapshot_indices = np.argsort(snapshot_abs_peak)[-10:][::-1]
    return (
        {
            "n_spatial_rows": int(n_spatial_rows),
            "n_snapshots": n_snapshots,
            "bad_width_rows": bad_width_rows,
            "nan_count": nan_count,
            "inf_count": inf_count,
            "finite_count": finite_count,
            "global_min": float(global_min),
            "global_max": float(global_max),
            "largest_abs_peak_snapshots": [
                {
                    "index": int(index),
                    "time": float((int(index) + 1) * 0.5),
                    "min": float(snapshot_min[index]),
                    "max": float(snapshot_max[index]),
                    "rms": float(snapshot_rms[index]),
                }
                for index in largest_snapshot_indices
            ],
            "temporal_std_metrics": spatial_metrics(temporal_std_map),
            "sample_metrics": {
                str(index): spatial_metrics(sample_maps[position])
                for position, index in enumerate(sample_indices)
            },
        },
        sample_maps,
        sample_indices,
        temporal_mean_map,
        temporal_std_map,
    )


def add_geometry(ax: plt.Axes, cylinder_y: float) -> None:
    ax.add_patch(
        Circle(
            (CYLINDER_X, cylinder_y),
            CYLINDER_DIAMETER / 2,
            facecolor="none",
            edgecolor="black",
            linewidth=0.8,
        )
    )
    ax.set_xlabel("x [grid cells]")
    ax.set_ylabel("y [saved fluid coordinates]")


def plot_samples(
    case_name: str,
    cylinder_y: float,
    field_name: str,
    maps: np.ndarray,
    indices: list[int],
) -> None:
    plot_maps = maps.astype(np.float64)
    if field_name == "pressure":
        plot_maps -= plot_maps.mean(axis=(1, 2), keepdims=True)
    magnitude = np.abs(plot_maps[np.isfinite(plot_maps)])
    limit = float(np.percentile(magnitude, 99.5)) if magnitude.size else 1.0
    limit = max(limit, 1e-8)

    fig, axes = plt.subplots(1, len(indices), figsize=(4 * len(indices), 4), constrained_layout=True)
    for ax, field, index in zip(axes, plot_maps, indices):
        image = ax.imshow(
            field.T,
            origin="lower",
            extent=(0, NX, 0, NY),
            aspect="auto",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
        )
        add_geometry(ax, cylinder_y)
        ax.set_title(f"snapshot {index}")
    fig.colorbar(image, ax=axes, shrink=0.78, label=field_name)
    fig.suptitle(f"{case_name}: {field_name} (99.5th-percentile scale)")
    fig.savefig(OUTPUT_ROOT / f"{case_name}_{field_name}_samples.png", dpi=160)
    plt.close(fig)


def plot_inlet_comparison(case_maps: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for row, (case_name, maps) in enumerate(case_maps.items()):
        field = maps[-1].astype(np.float64)
        limit = max(float(np.percentile(np.abs(field), 99.5)), 1e-8)
        for column, x_limit in enumerate((40, NX)):
            ax = axes[row, column]
            image = ax.imshow(
                field[:x_limit, :].T,
                origin="lower",
                extent=(0, x_limit, 0, NY),
                aspect="auto",
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
            )
            add_geometry(ax, CASES[case_name])
            ax.set_title(f"{case_name}: final, x < {x_limit}")
            fig.colorbar(image, ax=ax, shrink=0.75)
    fig.savefig(OUTPUT_ROOT / "final_vorticity_inlet_comparison.png", dpi=180)
    plt.close(fig)


def plot_temporal_std(
    case_name: str,
    cylinder_y: float,
    field_name: str,
    temporal_std: np.ndarray,
) -> None:
    limit = max(float(np.percentile(temporal_std, 99.5)), 1e-8)
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    image = ax.imshow(
        temporal_std.T,
        origin="lower",
        extent=(0, NX, 0, NY),
        aspect="auto",
        cmap="magma",
        vmin=0,
        vmax=limit,
    )
    add_geometry(ax, cylinder_y)
    ax.set_title(f"{case_name}: temporal standard deviation of {field_name}")
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.savefig(OUTPUT_ROOT / f"{case_name}_{field_name}_temporal_std.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}
    final_vorticity: dict[str, np.ndarray] = {}

    for case_name, cylinder_y in CASES.items():
        case_root = DATA_ROOT / case_name
        print(f"Inspecting {case_name}", flush=True)
        case_summary = {"time": read_time_file(case_root / "Time.txt")}

        for field_name, filename in (
            ("vorticity", "FullVorticityMap.csv"),
            ("pressure", "FullPressureMap.csv"),
        ):
            field_summary, maps, indices, temporal_mean, temporal_std = inspect_csv(
                case_root / filename, f"{case_name}/{field_name}"
            )
            case_summary[field_name] = field_summary
            plot_samples(case_name, cylinder_y, field_name, maps, indices)
            plot_temporal_std(case_name, cylinder_y, field_name, temporal_std)
            if field_name == "vorticity":
                final_vorticity[case_name] = maps

        summary[case_name] = case_summary

    plot_inlet_comparison(final_vorticity)
    (OUTPUT_ROOT / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Wrote diagnostics to {OUTPUT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
