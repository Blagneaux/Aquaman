import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class TrackRow:
    frame: int
    track_id: int
    core_x: float
    core_y: float
    sensor_id: int
    visible: bool


@dataclass
class SnapshotMeta:
    dataset: str
    model: str
    experiment: int
    snapshot: str
    sensor_id: int
    sensor_x: float
    sensor_y: float
    start_frame: int
    end_frame: int
    length: int
    track_path: Path
    pressure_path: Path


def reshape_frame(flat_frame: np.ndarray, num_cols: int, num_rows: int) -> np.ndarray:
    """Convert flattened (i-major) frame to a 2D (rows, cols) grid."""
    if flat_frame.size != num_cols * num_rows:
        raise ValueError(f"Frame size {flat_frame.size} does not match grid {num_cols}x{num_rows}.")
    return flat_frame.reshape(num_cols, num_rows).T


def parse_sensors(sensor_str: str) -> list[tuple[float, float]]:
    sensors: list[tuple[float, float]] = []
    if not sensor_str:
        return sensors
    for entry in sensor_str.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split(",")
        if len(parts) != 2:
            raise ValueError(f"Invalid sensor entry '{entry}'. Expected 'x,y'.")
        sensors.append((float(parts[0]), float(parts[1])))
    return sensors


def parse_experiments(value: str | None, default: list[int]) -> list[int]:
    if not value:
        return default
    experiments: list[int] = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_str, end_str = chunk.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            if end < start:
                start, end = end, start
            experiments.extend(range(start, end + 1))
        else:
            experiments.append(int(chunk))
    return experiments


def collect_snapshots(base_path: Path, experiment: int) -> list[str]:
    subpath = base_path / str(experiment)
    if not subpath.exists():
        return []
    return [p.name for p in subpath.iterdir() if p.is_dir()]


def load_tracks_csv(path: Path) -> list[TrackRow]:
    rows: list[TrackRow] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_raw = row.get("frame")
            if frame_raw is None or frame_raw == "":
                continue
            try:
                frame = int(float(frame_raw))
            except ValueError:
                continue

            track_raw = row.get("track_id", "")
            try:
                track_id = int(float(track_raw)) if track_raw not in ("", "nan", None) else -1
            except ValueError:
                track_id = -1

            core_x_raw = row.get("core_x", "nan")
            core_y_raw = row.get("core_y", "nan")
            try:
                core_x = float(core_x_raw)
                core_y = float(core_y_raw)
            except ValueError:
                core_x = float("nan")
                core_y = float("nan")

            sensor_raw = row.get("sensor_id", "")
            try:
                sensor_id = int(float(sensor_raw)) if sensor_raw not in ("", "nan", None) else -1
            except ValueError:
                sensor_id = -1

            visible_raw = row.get("visible", "1")
            try:
                visible = bool(int(float(visible_raw)))
            except ValueError:
                visible = True

            rows.append(
                TrackRow(
                    frame=frame,
                    track_id=track_id,
                    core_x=core_x,
                    core_y=core_y,
                    sensor_id=sensor_id,
                    visible=visible,
                )
            )
    return rows


def load_pressure_map(path: Path) -> np.ndarray:
    matrix = np.genfromtxt(path, delimiter=",")
    if matrix.size == 0:
        raise ValueError(f"Pressure map is empty: {path}")
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if np.isnan(matrix).any():
        matrix = np.nan_to_num(matrix, nan=0.0)
    return matrix


def sensor_indices(sensors: list[tuple[float, float]], num_rows: int, num_cols: int) -> list[int | None]:
    indices: list[int | None] = []
    for sx, sy in sensors:
        x = int(round(sx))
        y = int(round(sy))
        if x < 0 or x >= num_cols or y < 0 or y >= num_rows:
            indices.append(None)
        else:
            indices.append(x * num_rows + y)
    return indices


def sensor_label(sensor: tuple[float, float], idx: int) -> str:
    sx, sy = sensor
    xi = int(round(sx))
    yi = int(round(sy))
    if (xi, yi) == (198, 80):
        return "S1"
    if (xi, yi) == (146, 80):
        return "S2"
    if (xi, yi) == (200, 39):
        return "S4"
    return f"S{idx}"


def build_active_mask(
    rows: list[TrackRow],
    n_frames: int,
    sensor_id: int,
    require_visible: bool,
) -> np.ndarray:
    active = np.zeros(n_frames, dtype=bool)
    for row in rows:
        if row.sensor_id != sensor_id:
            continue
        if require_visible and not row.visible:
            continue
        if 0 <= row.frame < n_frames:
            active[row.frame] = True
    return active


def find_segments(active: np.ndarray) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start = None
    for idx, val in enumerate(active):
        if val and start is None:
            start = idx
        elif not val and start is not None:
            segments.append((start, idx - 1))
            start = None
    if start is not None:
        segments.append((start, len(active) - 1))
    return segments


def write_snapshot_matrix(
    out_path: Path,
    snapshots: list[np.ndarray],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not snapshots:
        np.savetxt(out_path, np.empty((0, 0)), delimiter=",")
        return
    max_len = max(len(s) for s in snapshots)
    matrix = np.full((max_len, len(snapshots)), np.nan, dtype=np.float64)
    for col, series in enumerate(snapshots):
        matrix[: len(series), col] = series
    np.savetxt(out_path, matrix, delimiter=",")


def write_metadata(out_path: Path, meta: list[SnapshotMeta]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "model",
        "experiment",
        "snapshot",
        "sensor_id",
        "sensor_x",
        "sensor_y",
        "start_frame",
        "end_frame",
        "length",
        "track_path",
        "pressure_path",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in meta:
            writer.writerow(
                {
                    "dataset": entry.dataset,
                    "model": entry.model,
                    "experiment": entry.experiment,
                    "snapshot": entry.snapshot,
                    "sensor_id": entry.sensor_id,
                    "sensor_x": entry.sensor_x,
                    "sensor_y": entry.sensor_y,
                    "start_frame": entry.start_frame,
                    "end_frame": entry.end_frame,
                    "length": entry.length,
                    "track_path": str(entry.track_path),
                    "pressure_path": str(entry.pressure_path),
                }
            )


def show_debug_snapshot(
    pressure_matrix: np.ndarray,
    tracks: list[TrackRow],
    sensors: list[tuple[float, float]],
    num_cols: int,
    num_rows: int,
    segment: tuple[int, int],
    fps: int,
    max_frames: int | None,
    require_visible: bool,
) -> None:
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    start, end = segment
    total_frames = pressure_matrix.shape[1]
    frames = list(range(total_frames))
    if max_frames is not None and len(frames) > max_frames:
        frames = frames[:max_frames]

    idxs = sensor_indices(sensors, num_rows, num_cols)
    sensor_series: list[np.ndarray] = []
    for idx in idxs:
        if idx is None:
            sensor_series.append(np.full(len(frames), np.nan))
        else:
            sensor_series.append(pressure_matrix[idx, frames])

    frame_set = set(frames)
    frame_to_idx = {frame: idx for idx, frame in enumerate(frames)}
    active_by_sensor = [np.zeros(len(frames), dtype=bool) for _ in sensors]
    tracks_by_id: dict[int, list[tuple[int, float, float]]] = {}
    for row in tracks:
        if row.frame not in frame_set:
            continue
        if require_visible and not row.visible:
            continue
        if 0 <= row.sensor_id < len(sensors):
            active_by_sensor[row.sensor_id][frame_to_idx[row.frame]] = True
        tracks_by_id.setdefault(row.track_id, []).append((row.frame, row.core_x, row.core_y))
    for track_id in tracks_by_id:
        tracks_by_id[track_id].sort(key=lambda v: v[0])

    fig, (ax_map, ax_ts) = plt.subplots(1, 2, figsize=(12, 5))
    ax_map.set_title("Pressure map + vortex tracks")
    ax_map.set_xlim(0, num_cols - 1)
    ax_map.set_ylim(0, num_rows - 1)
    ax_map.set_xlabel("x")
    ax_map.set_ylabel("y")

    frame0 = reshape_frame(pressure_matrix[:, frames[0]], num_cols, num_rows)
    im = ax_map.imshow(frame0, origin="lower", cmap="seismic", vmin=-1.0, vmax=1.0, alpha=0.9)

    if sensors:
        ax_map.scatter(
            [s[0] for s in sensors],
            [s[1] for s in sensors],
            c="lime",
            s=30,
            label="Sensors",
            edgecolors="black",
            linewidths=0.5,
        )
        for idx, sensor in enumerate(sensors):
            ax_map.text(
                sensor[0] + 1.0,
                sensor[1] + 1.0,
                sensor_label(sensor, idx),
                fontsize=8,
                color="black",
                ha="left",
                va="bottom",
            )

    track_lines: dict[int, plt.Line2D] = {}
    cmap = plt.cm.get_cmap("tab20", max(len(tracks_by_id), 1))
    for idx, track_id in enumerate(sorted(tracks_by_id.keys())):
        track_lines[track_id], = ax_map.plot([], [], color=cmap(idx), linewidth=1.2, alpha=0.8)

    if sensors or track_lines:
        ax_map.legend(loc="upper right", fontsize=8)

    ax_ts.set_title("Pressure at sensors")
    ax_ts.set_xlabel("frame")
    ax_ts.set_ylabel("pressure")
    ax_ts.set_ylim(-1.0, 1.0)
    time_axis = np.array(frames, dtype=int)

    sensor_lines: list[plt.Line2D] = []
    for idx, series in enumerate(sensor_series):
        line, = ax_ts.plot([], [], linewidth=1.4, label=sensor_label(sensors[idx], idx))
        sensor_lines.append(line)
    vline = ax_ts.axvline(time_axis[0], color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    if sensor_lines:
        ax_ts.legend(loc="upper right", fontsize=8)

    def update(frame_idx: int):
        frame = frames[frame_idx]
        im.set_data(reshape_frame(pressure_matrix[:, frame], num_cols, num_rows))

        for track_id, line in track_lines.items():
            track = tracks_by_id[track_id]
            xs = [x for f, x, _ in track if f <= frame]
            ys = [y for f, _, y in track if f <= frame]
            line.set_data(xs, ys)

        for idx, line in enumerate(sensor_lines):
            series = sensor_series[idx]
            line.set_data(time_axis[: frame_idx + 1], series[: frame_idx + 1])
            line.set_linestyle("-" if active_by_sensor[idx][frame_idx] else "--")

        xval = time_axis[frame_idx]
        vline.set_xdata([xval, xval])
        x_min = time_axis[0]
        x_max = time_axis[frame_idx]
        if x_max == x_min:
            x_max = x_min + 1
        ax_ts.set_xlim(x_min, x_max)
        ax_map.set_title(f"Pressure map + vortex tracks (frame {frame})")
        return (im, *track_lines.values(), *sensor_lines, vline)

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=max(1, int(1000 / max(fps, 1))),
        blit=False,
    )

    plt.show()
    plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Build pressure snapshots from vortex track CSVs.")
    parser.add_argument("--path-nadia", type=Path, default=Path("D:/crop_nadia"), help="Root path for Nadia dataset")
    parser.add_argument("--path-thomas", type=Path, default=Path("D:/thomas_files"), help="Root path for Thomas dataset")
    parser.add_argument("--path-boai", type=Path, default=Path("D:/boai_files"), help="Root path for Boai dataset")
    parser.add_argument("--path-full", type=Path, default=Path("D:/full_files"), help="Root path for Full dataset")
    parser.add_argument(
        "--datasets",
        type=str,
        default="nadia,thomas,boai,full",
        help="Comma-separated dataset list (nadia, thomas, boai, full).",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="fish,cylinder",
        help="Comma-separated model list (fish, cylinder).",
    )
    parser.add_argument("--tracks-fish", type=str, default="vortex_tracks.csv", help="Track filename for fish model")
    parser.add_argument(
        "--tracks-cylinder",
        type=str,
        default="circle_vortex_tracks.csv",
        help="Track filename for cylinder model",
    )
    parser.add_argument(
        "--pressure-fish",
        type=str,
        default="pressure_map.csv",
        help="Pressure map filename for fish model",
    )
    parser.add_argument(
        "--pressure-cylinder",
        type=str,
        default="circle_pressure_map.csv",
        help="Pressure map filename for cylinder model",
    )
    parser.add_argument("--num-cols", type=int, default=256, help="Grid columns (x-direction)")
    parser.add_argument("--num-rows", type=int, default=128, help="Grid rows (y-direction)")
    parser.add_argument(
        "--sensors",
        type=str,
        default="146,80;198,80;200,39",
        help="Sensors as 'x,y;x,y;...'",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        default=None,
        help="Comma-separated experiment list or ranges (e.g. '1,2,5-8'). Defaults to known sensor experiments.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=repo_root / "python/vortexAnalysis_v2/vortex_outputs/pressure_snapshots",
        help="Output directory for snapshot matrices",
    )
    parser.add_argument("--include-occluded", action="store_true", help="Include occluded vortices (ignore visible flag)")
    parser.add_argument("--split-by-sensor", action="store_true", help="Write separate matrices per sensor")
    parser.add_argument("--debug", action="store_true", help="Show debug animation for the first snapshot found")
    parser.add_argument("--debug-fps", type=int, default=24, help="Debug animation FPS")
    parser.add_argument(
        "--debug-max-frames",
        type=int,
        default=None,
        help="Limit debug animation length (frames)",
    )
    args = parser.parse_args()

    sensors = parse_sensors(args.sensors)
    if not sensors:
        raise ValueError("No sensors provided. Use --sensors to set sensor coordinates.")

    default_experiments = [
        1,
        2,
        3,
        6,
        7,
        8,
        9,
        10,
        11,
        14,
        15,
        16,
        17,
        18,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        35,
        40,
    ]
    experiments = parse_experiments(args.experiments, default_experiments)

    dataset_roots = {
        "nadia": args.path_nadia,
        "thomas": args.path_thomas,
        "boai": args.path_boai,
        "full": args.path_full,
    }
    dataset_list = [d.strip() for d in args.datasets.split(",") if d.strip()]
    model_list = [m.strip() for m in args.models.split(",") if m.strip()]

    models = {
        "fish": {"tracks": args.tracks_fish, "pressure": args.pressure_fish},
        "cylinder": {"tracks": args.tracks_cylinder, "pressure": args.pressure_cylinder},
    }

    require_visible = not args.include_occluded
    debug_shown = False

    for dataset in dataset_list:
        if dataset not in dataset_roots:
            print(f"⚠️  Unknown dataset '{dataset}', skipping.")
            continue
        root = dataset_roots[dataset]
        if not root.exists():
            print(f"⚠️  Dataset path not found: {root}")
            continue

        for model in model_list:
            if model not in models:
                print(f"⚠️  Unknown model '{model}', skipping.")
                continue

            track_filename = models[model]["tracks"]
            pressure_filename = models[model]["pressure"]

            snapshots: list[np.ndarray] = []
            metadata: list[SnapshotMeta] = []
            snapshots_by_sensor: dict[int, list[np.ndarray]] = {i: [] for i in range(len(sensors))}
            metadata_by_sensor: dict[int, list[SnapshotMeta]] = {i: [] for i in range(len(sensors))}

            for experiment in experiments:
                snapshot_names = collect_snapshots(root, experiment)
                if not snapshot_names:
                    continue

                for snapshot_name in snapshot_names:
                    snapshot_path = root / str(experiment) / snapshot_name
                    track_path = snapshot_path / track_filename
                    pressure_path = snapshot_path / pressure_filename

                    if not track_path.exists() or not pressure_path.exists():
                        print(
                            f"⏭️  {dataset}/{model} exp={experiment} snapshot={snapshot_name} "
                            f"(missing {track_filename} or {pressure_filename})"
                        )
                        continue

                    try:
                        tracks = load_tracks_csv(track_path)
                        pressure = load_pressure_map(pressure_path)
                    except Exception as exc:
                        print(f"⚠️  Failed to read data for {snapshot_path}: {exc}")
                        continue

                    n_cells, n_frames = pressure.shape
                    expected_cells = args.num_cols * args.num_rows
                    if n_cells != expected_cells:
                        print(
                            f"⚠️  Pressure map shape mismatch at {pressure_path} "
                            f"({n_cells} cells, expected {expected_cells}). Skipping."
                        )
                        continue

                    idxs = sensor_indices(sensors, args.num_rows, args.num_cols)
                    sensor_series_all: list[np.ndarray | None] = []
                    for idx in idxs:
                        if idx is None:
                            sensor_series_all.append(None)
                        else:
                            sensor_series_all.append(pressure[idx, :])

                    for sensor_id, series in enumerate(sensor_series_all):
                        if series is None:
                            continue
                        active = build_active_mask(tracks, n_frames, sensor_id, require_visible)
                        segments = find_segments(active)
                        if not segments:
                            continue

                        for start, end in segments:
                            snap_series = series[start : end + 1].astype(np.float64, copy=True)
                            meta = SnapshotMeta(
                                dataset=dataset,
                                model=model,
                                experiment=experiment,
                                snapshot=snapshot_name,
                                sensor_id=sensor_id,
                                sensor_x=sensors[sensor_id][0],
                                sensor_y=sensors[sensor_id][1],
                                start_frame=start,
                                end_frame=end,
                                length=len(snap_series),
                                track_path=track_path,
                                pressure_path=pressure_path,
                            )

                            snapshots.append(snap_series)
                            metadata.append(meta)
                            snapshots_by_sensor[sensor_id].append(snap_series)
                            metadata_by_sensor[sensor_id].append(meta)

                            if args.debug and not debug_shown:
                                try:
                                    show_debug_snapshot(
                                        pressure_matrix=pressure,
                                        tracks=tracks,
                                        sensors=sensors,
                                        num_cols=args.num_cols,
                                        num_rows=args.num_rows,
                                        segment=(start, end),
                                        fps=args.debug_fps,
                                        max_frames=args.debug_max_frames,
                                        require_visible=require_visible,
                                    )
                                finally:
                                    debug_shown = True
                    print(f"✅ Processed {dataset}/{model} exp={experiment} snapshot={snapshot_name}")

                print(f"✅ Completed {dataset}/{model} experiment {experiment}")

            out_dir = args.out_dir
            out_dir.mkdir(parents=True, exist_ok=True)

            if args.split_by_sensor:
                for sensor_id in range(len(sensors)):
                    out_matrix = out_dir / f"pressure_snapshots_{dataset}_{model}_sensor{sensor_id}.csv"
                    out_meta = out_dir / f"pressure_snapshots_{dataset}_{model}_sensor{sensor_id}_meta.csv"
                    write_snapshot_matrix(out_matrix, snapshots_by_sensor[sensor_id])
                    write_metadata(out_meta, metadata_by_sensor[sensor_id])
                    print(
                        f"✅ {dataset}/{model} sensor {sensor_id}: "
                        f"{len(snapshots_by_sensor[sensor_id])} snapshots -> {out_matrix}"
                    )
            else:
                out_matrix = out_dir / f"pressure_snapshots_{dataset}_{model}.csv"
                out_meta = out_dir / f"pressure_snapshots_{dataset}_{model}_meta.csv"
                write_snapshot_matrix(out_matrix, snapshots)
                write_metadata(out_meta, metadata)
                print(f"✅ {dataset}/{model}: {len(snapshots)} snapshots -> {out_matrix}")


if __name__ == "__main__":
    main()
