#!/usr/bin/env python3
"""Save measured pressure snapshots used for real-pressure inference.

This script deliberately stops before classifier inference. It reads the same
TDMS pressure traces, timestamp files and simulation event metadata used by
``test_classifier_real_pressure.py``, applies the same preprocessing, and writes
only the measured pressure snapshot matrix plus its metadata.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from nptdms import TdmsFile
except ImportError:
    TdmsFile = None

try:
    from scipy.signal import butter, filtfilt
except ImportError:
    butter = None
    filtfilt = None

from compute_pressure_modes import (
    fill_internal_nans_1d,
    find_modes_file,
    format_energy_tag,
    has_tail_spike,
    normalize_series,
    resample_series,
    zero_mean_series,
)


@dataclass
class SnapshotMeta:
    dataset: str
    source_model: str
    experiment: int
    snapshot: int
    sensor_id: int
    sensor_x: float
    sensor_y: float
    start_frame: int
    end_frame: int
    length: int
    track_path: str
    pressure_path: str


@dataclass
class RealSnapshotMeta:
    dataset: str
    experiment: int
    snapshot: int
    sensor: str
    source_model: str
    start_frame: int
    end_frame: int
    original_length_frames: int
    target_length: int


def time_to_seconds(time_str: str) -> float:
    h, m, s = map(int, time_str.split(":"))
    return (h * 3600 + m * 60 + s) / 4.0


def sensor_label(sensor_x: float, sensor_y: float, fallback_idx: int) -> str:
    xi = int(round(sensor_x))
    yi = int(round(sensor_y))
    if (xi, yi) == (198, 80):
        return "S1"
    if (xi, yi) == (146, 80):
        return "S2"
    if (xi, yi) == (200, 39):
        return "S4"
    return f"S{fallback_idx}"


def read_timestamps(path: Path) -> Any:
    if pd is None:
        raise RuntimeError("pandas is required to read timestamps. Install pandas.")
    df = pd.read_csv(path)
    if "start_time" in df.columns:
        df["start_time"] = df["start_time"].apply(time_to_seconds)
    if "end_time" in df.columns:
        df["end_time"] = df["end_time"].apply(time_to_seconds)
    return df


def load_tdms_channel(tdms_path: Path, sensor_name: str) -> np.ndarray:
    if TdmsFile is None:
        raise RuntimeError("nptdms is required to read TDMS. Install nptdms.")
    tdms_file = TdmsFile.read(tdms_path)
    for group in tdms_file.groups()[1:]:
        for channel in group.channels():
            if channel.name == sensor_name:
                return np.asarray(channel.data)
    raise ValueError(f"Sensor {sensor_name} not found in {tdms_path}")


def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 2) -> tuple[np.ndarray, np.ndarray]:
    if butter is None:
        raise RuntimeError("scipy is required for filtering. Install scipy.")
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    return butter(order, [low, high], btype="band")


def bandpass_filter(data: np.ndarray, lowcut: float = 0.3, highcut: float = 30.0, fs: float = 500.0, order: int = 2) -> np.ndarray:
    if filtfilt is None:
        raise RuntimeError("scipy is required for filtering. Install scipy.")
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, data)


def frame_to_sample(frame_idx: int, frame_rate: float, fs_raw: float) -> int:
    return int(round(frame_idx * fs_raw / frame_rate))


def parse_dataset_roots(value: str | None, datasets: list[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    if value:
        for chunk in value.split(";"):
            chunk = chunk.strip()
            if not chunk:
                continue
            if "=" in chunk:
                name, path = chunk.split("=", 1)
                roots[name.strip()] = Path(path.strip())
    for dataset in datasets:
        roots.setdefault(dataset, Path(f"D:/crop_{dataset}"))
    return roots


def load_snapshot_meta(path: Path) -> list[SnapshotMeta]:
    rows: list[SnapshotMeta] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                rows.append(
                    SnapshotMeta(
                        dataset=row.get("dataset", ""),
                        source_model=row.get("model", ""),
                        experiment=int(float(row.get("experiment", "0"))),
                        snapshot=int(float(row.get("snapshot", "0"))),
                        sensor_id=int(float(row.get("sensor_id", "-1"))),
                        sensor_x=float(row.get("sensor_x", "nan")),
                        sensor_y=float(row.get("sensor_y", "nan")),
                        start_frame=int(float(row.get("start_frame", "0"))),
                        end_frame=int(float(row.get("end_frame", "0"))),
                        length=int(float(row.get("length", "0"))),
                        track_path=row.get("track_path", ""),
                        pressure_path=row.get("pressure_path", ""),
                    )
                )
            except ValueError:
                continue
    return rows


def build_real_snapshots(
    meta_rows: list[SnapshotMeta],
    dataset_root: Path,
    timestamps_root: Path,
    tdms_root: Path,
    fs_raw: float,
    frame_rate: float,
    interp: str,
    target_length: int,
    min_length: int,
    tail_spike_filter: bool,
    tail_fraction: float,
    tail_ratio: float,
    tail_min: int,
    frame_offset: int,
    tdms_subdir: str,
    timestamps_subdir: str,
    timestamps_prefix: str,
) -> tuple[np.ndarray, list[RealSnapshotMeta]]:
    snapshots: list[np.ndarray] = []
    metadata: list[RealSnapshotMeta] = []
    tdms_cache: dict[tuple[Path, int, str], np.ndarray] = {}
    timestamps_cache: dict[tuple[Path, int], Any] = {}
    pressure_cache: dict[tuple[Path, int, int, str], tuple[np.ndarray, int]] = {}

    for meta in meta_rows:
        if meta.length < min_length:
            continue
        sensor_name = sensor_label(meta.sensor_x, meta.sensor_y, meta.sensor_id)

        timestamps_key = (timestamps_root, meta.experiment)
        if timestamps_key not in timestamps_cache:
            timestamps_path = timestamps_root / timestamps_subdir / f"{timestamps_prefix}{meta.experiment}.csv"
            if not timestamps_path.exists():
                print(f"WARN: Missing timestamps file: {timestamps_path}")
                continue
            timestamps_cache[timestamps_key] = read_timestamps(timestamps_path)

        timestamp_rows = timestamps_cache[timestamps_key]
        if meta.snapshot - 1 < 0 or meta.snapshot - 1 >= len(timestamp_rows):
            print(f"WARN: Snapshot index {meta.snapshot} out of range for experiment {meta.experiment}")
            continue
        timestamp_row = timestamp_rows.iloc[meta.snapshot - 1]
        try:
            start_time = float(timestamp_row["start_time"])
            end_time = float(timestamp_row["end_time"])
            start_index = int(float(timestamp_row.get("start_frame", 0)))
        except Exception:
            print(f"WARN: Missing timestamp columns in experiment {meta.experiment}")
            continue

        pressure_key = (dataset_root, meta.experiment, meta.snapshot, sensor_name)
        if pressure_key not in pressure_cache:
            tdms_key = (tdms_root, meta.experiment, sensor_name)
            if tdms_key not in tdms_cache:
                tdms_path = tdms_root / tdms_subdir / f"{meta.experiment}.tdms"
                if not tdms_path.exists():
                    print(f"WARN: Missing TDMS file: {tdms_path}")
                    continue
                tdms_cache[tdms_key] = load_tdms_channel(tdms_path, sensor_name)

            filtered = bandpass_filter(tdms_cache[tdms_key], highcut=9.0, fs=fs_raw)
            start_sample = int(start_time * fs_raw)
            end_sample = int(end_time * fs_raw)
            cropped_full = filtered[start_sample:end_sample]
            skip_samples = frame_to_sample(start_index, frame_rate, fs_raw)
            pressure_exp = cropped_full[skip_samples:] if skip_samples > 0 else cropped_full
            pressure_cache[pressure_key] = (pressure_exp, skip_samples)

        pressure_exp, _ = pressure_cache[pressure_key]
        if pressure_exp.size == 0:
            continue

        start_frame = max(meta.start_frame - frame_offset, 0)
        end_frame = meta.end_frame - frame_offset
        if end_frame < start_frame:
            continue
        start_sample = frame_to_sample(start_frame, frame_rate, fs_raw)
        end_sample = frame_to_sample(end_frame + 1, frame_rate, fs_raw)
        if start_sample >= pressure_exp.size:
            continue
        end_sample = min(end_sample, pressure_exp.size)

        segment = np.asarray(pressure_exp[start_sample:end_sample], dtype=np.float64)
        if segment.size == 0:
            continue
        if tail_spike_filter and has_tail_spike(segment, tail_fraction, tail_ratio, tail_min):
            continue

        segment = fill_internal_nans_1d(segment)
        resampled = resample_series(segment, target_length, interp)
        resampled = zero_mean_series(resampled)
        resampled = normalize_series(resampled)
        snapshots.append(resampled)
        metadata.append(
            RealSnapshotMeta(
                dataset=meta.dataset,
                experiment=meta.experiment,
                snapshot=meta.snapshot,
                sensor=sensor_name,
                source_model=meta.source_model,
                start_frame=meta.start_frame,
                end_frame=meta.end_frame,
                original_length_frames=meta.length,
                target_length=target_length,
            )
        )

    if not snapshots:
        return np.empty((target_length, 0), dtype=np.float64), []
    return np.vstack(snapshots).T, metadata


def save_outputs(matrix_path: Path, meta_path: Path, matrix: np.ndarray, rows: list[RealSnapshotMeta]) -> None:
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(matrix_path, matrix, delimiter=",")
    with meta_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "dataset",
                "experiment",
                "snapshot",
                "sensor",
                "source_model",
                "start_frame",
                "end_frame",
                "original_length_frames",
                "target_length",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.dataset,
                    row.experiment,
                    row.snapshot,
                    row.sensor,
                    row.source_model,
                    row.start_frame,
                    row.end_frame,
                    row.original_length_frames,
                    row.target_length,
                ]
            )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="boai")
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=repo_root / "python/vortexAnalysis_v2/vortex_outputs/pressure_snapshots",
        help="Directory containing simulated pressure snapshot metadata.",
    )
    parser.add_argument(
        "--modes-dir",
        type=Path,
        default=repo_root / "python/vortexAnalysis_v2/vortex_outputs/pressure_modes",
        help="Directory containing pressure mode files used to get target length.",
    )
    parser.add_argument("--modes-energy", type=float, default=0.95)
    parser.add_argument(
        "--dataset-roots",
        default=None,
        help="Override dataset roots, e.g. 'boai=/mnt/d/crop_boai;nadia=/mnt/d/crop_nadia'.",
    )
    parser.add_argument("--tdms-subdir", default="TDMS")
    parser.add_argument("--timestamps-subdir", default="timestamps")
    parser.add_argument("--timestamps-prefix", default="timestamps")
    parser.add_argument("--timestamps-dataset", default="nadia")
    parser.add_argument("--tdms-dataset", default="nadia")
    parser.add_argument("--min-length", type=int, default=100)
    parser.add_argument("--interp", choices=["linear", "cubic"], default="cubic")
    parser.add_argument("--frame-rate", type=float, default=100.0)
    parser.add_argument("--fs-raw", type=float, default=500.0)
    parser.add_argument("--tail-spike-filter", action="store_true")
    parser.add_argument("--tail-frac", type=float, default=0.15)
    parser.add_argument("--tail-ratio", type=float, default=5.0)
    parser.add_argument("--tail-min", type=int, default=5)
    parser.add_argument("--frame-offset", type=int, default=0)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=repo_root / "python/vortexAnalysis_v2/vortex_outputs/real_pressure_snapshots",
    )
    args = parser.parse_args()

    datasets = [dataset.strip() for dataset in args.datasets.split(",") if dataset.strip()]
    dataset_roots = parse_dataset_roots(args.dataset_roots, datasets)
    suffix = f"_{format_energy_tag(args.modes_energy)}" if args.modes_energy is not None else ""

    for dataset in datasets:
        fish_meta_path = args.snapshot_dir / f"pressure_snapshots_{dataset}_fish_meta.csv"
        cyl_meta_path = args.snapshot_dir / f"pressure_snapshots_{dataset}_cylinder_meta.csv"
        if not fish_meta_path.exists() or not cyl_meta_path.exists():
            print(f"WARN: Missing simulated snapshot metadata for {dataset}")
            continue

        modes_path = find_modes_file(args.modes_dir, dataset, args.modes_energy)
        if modes_path is None or not modes_path.exists():
            print(f"WARN: Missing pressure mode file for {dataset} in {args.modes_dir}")
            continue
        modes_data = np.load(modes_path)
        target_length = int(modes_data.get("target_length", modes_data["modes"].shape[0]))

        dataset_root = dataset_roots.get(dataset)
        timestamps_root = dataset_roots.get(args.timestamps_dataset, dataset_root)
        tdms_root = dataset_roots.get(args.tdms_dataset, dataset_root)
        if timestamps_root is None or not timestamps_root.exists():
            print(f"WARN: Missing timestamps root for {dataset}: {timestamps_root}")
            continue
        if tdms_root is None or not tdms_root.exists():
            print(f"WARN: Missing TDMS root for {dataset}: {tdms_root}")
            continue
        if dataset_root is None or not dataset_root.exists():
            dataset_root = tdms_root

        meta_rows = load_snapshot_meta(fish_meta_path) + load_snapshot_meta(cyl_meta_path)
        matrix, metadata = build_real_snapshots(
            meta_rows=meta_rows,
            dataset_root=dataset_root,
            timestamps_root=timestamps_root,
            tdms_root=tdms_root,
            fs_raw=args.fs_raw,
            frame_rate=args.frame_rate,
            interp=args.interp,
            target_length=target_length,
            min_length=args.min_length,
            tail_spike_filter=args.tail_spike_filter,
            tail_fraction=args.tail_frac,
            tail_ratio=args.tail_ratio,
            tail_min=args.tail_min,
            frame_offset=args.frame_offset,
            tdms_subdir=args.tdms_subdir,
            timestamps_subdir=args.timestamps_subdir,
            timestamps_prefix=args.timestamps_prefix,
        )
        if matrix.shape[1] == 0:
            print(f"WARN: No measured pressure snapshots saved for {dataset}")
            continue

        matrix_path = args.out_dir / f"real_pressure_snapshots_{dataset}{suffix}.csv"
        meta_path = args.out_dir / f"real_pressure_snapshots_{dataset}{suffix}_meta.csv"
        save_outputs(matrix_path, meta_path, matrix, metadata)
        print(f"OK: {dataset}: {matrix.shape[1]} measured snapshots saved to {matrix_path}")


if __name__ == "__main__":
    main()
