#!/usr/bin/env python3
"""Compare pressure-conditioned vortex descriptors without classifier training.

The script uses existing measured-pressure predictions, pressure-snapshot
metadata, and vortex-track CSV files. It does not rerun CFD, pressure inference,
or either classifier. Fish-like pressure segments are linked to fish-model
tracks and cylinder-like pressure segments are linked to cylinder-model tracks.

Outputs include unbalanced track-level descriptors, equally weighted segment
and video summaries, paired video differences, long-form distribution summaries,
an audit report, and optional box plots.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Iterable, Sequence

import numpy as np


MODEL_ORDER = ("fish", "cylinder")
MODEL_LABELS = {"fish": "Fish-like", "cylinder": "Cylinder-like"}
MODEL_COLOURS = {"fish": "#1f77b4", "cylinder": "#ff7f0e"}

TRACK_DESCRIPTORS = (
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
)

SEGMENT_EXTRA_METRICS = (
    "vortex_count",
    "total_vortex_frames",
    "mean_simultaneous_vortices",
    "max_simultaneous_vortices",
)

SEGMENT_METRICS = TRACK_DESCRIPTORS + SEGMENT_EXTRA_METRICS

METRIC_LABELS = {
    "mean_area": "Mean area (grid cells)",
    "max_area": "Maximum area (grid cells)",
    "mean_equivalent_radius": "Mean equivalent radius (grid cells)",
    "max_equivalent_radius": "Maximum equivalent radius (grid cells)",
    "displacement": "Displacement (grid cells)",
    "path_length": "Path length (grid cells)",
    "straightness": "Straightness",
    "duration_frames": "Duration (detected frames)",
    "duration_ratio": "Duration / segment length",
    "mean_speed": "Mean speed (grid cells frame$^{-1}$)",
    "max_speed": "Maximum speed (grid cells frame$^{-1}$)",
    "mean_core_q": "Mean core Q",
    "max_core_q": "Maximum core Q",
    "mean_body_distance": "Mean body distance (grid cells)",
    "mean_sensor_distance": "Mean sensor distance (grid cells)",
    "mean_core_x": "Mean core x (grid cells)",
    "mean_core_y": "Mean core y (grid cells)",
    "vortex_count": "Vortices per segment",
    "total_vortex_frames": "Total vortex-frame occupancy",
    "mean_simultaneous_vortices": "Mean simultaneous vortices",
    "max_simultaneous_vortices": "Maximum simultaneous vortices",
}

CAMBRIDGE_RC = {
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "text.usetex": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
}


@dataclass(frozen=True)
class PredictionRow:
    dataset: str
    architecture: str
    experiment: int
    snapshot: int
    sensor_label: str
    source_model: str
    start_frame: int
    end_frame: int
    probability: float


@dataclass(frozen=True)
class MetaRow:
    dataset: str
    model: str
    experiment: int
    snapshot: int
    sensor_id: int
    sensor_x: float
    sensor_y: float
    start_frame: int
    end_frame: int
    track_path_raw: str


def parse_int(value: str | None, default: int = -1) -> int:
    try:
        return int(float(value)) if value is not None else default
    except (TypeError, ValueError):
        return default


def parse_float(value: str | None, default: float = math.nan) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def format_energy_tag(energy: float) -> int:
    percentage = energy * 100.0 if energy <= 1.0 else energy
    return int(round(max(0.0, min(100.0, percentage))))


def sensor_label(sensor_x: float, sensor_y: float, fallback_idx: int) -> str:
    coordinates = (int(round(sensor_x)), int(round(sensor_y)))
    known = {(198, 80): "S1", (146, 80): "S2", (200, 39): "S4"}
    return known.get(coordinates, f"S{fallback_idx}")


def classify_prediction(probability: float, threshold: float) -> str:
    if not math.isfinite(probability):
        return "none"
    probability = min(1.0, max(0.0, probability))
    if probability >= threshold:
        return "cylinder"
    if probability <= 1.0 - threshold:
        return "fish"
    return "none"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_predictions(path: Path) -> list[PredictionRow]:
    predictions: list[PredictionRow] = []
    for row in read_csv(path):
        predictions.append(
            PredictionRow(
                dataset=row.get("dataset", ""),
                architecture=row.get("architecture", ""),
                experiment=parse_int(row.get("experiment")),
                snapshot=parse_int(row.get("snapshot")),
                sensor_label=row.get("sensor", "").strip(),
                source_model=row.get("model_true", "").strip(),
                start_frame=parse_int(row.get("start_frame")),
                end_frame=parse_int(row.get("end_frame")),
                probability=parse_float(row.get("prob")),
            )
        )
    return predictions


def load_metadata(path: Path) -> list[MetaRow]:
    metadata: list[MetaRow] = []
    for row in read_csv(path):
        metadata.append(
            MetaRow(
                dataset=row.get("dataset", ""),
                model=row.get("model", ""),
                experiment=parse_int(row.get("experiment")),
                snapshot=parse_int(row.get("snapshot")),
                sensor_id=parse_int(row.get("sensor_id")),
                sensor_x=parse_float(row.get("sensor_x")),
                sensor_y=parse_float(row.get("sensor_y")),
                start_frame=parse_int(row.get("start_frame")),
                end_frame=parse_int(row.get("end_frame")),
                track_path_raw=row.get("track_path", "").strip(),
            )
        )
    return metadata


MetaExactIndex = dict[tuple[int, int, str, int, int], MetaRow]
MetaFallbackIndex = dict[tuple[int, int, str], list[MetaRow]]


def build_metadata_index(rows: Iterable[MetaRow]) -> tuple[MetaExactIndex, MetaFallbackIndex]:
    exact: MetaExactIndex = {}
    fallback: MetaFallbackIndex = defaultdict(list)
    for row in rows:
        sensor = sensor_label(row.sensor_x, row.sensor_y, row.sensor_id)
        exact.setdefault(
            (row.experiment, row.snapshot, sensor, row.start_frame, row.end_frame),
            row,
        )
        fallback[(row.experiment, row.snapshot, sensor)].append(row)
    return exact, dict(fallback)


def select_metadata(
    exact: MetaExactIndex,
    fallback: MetaFallbackIndex,
    prediction: PredictionRow,
) -> tuple[MetaRow | None, bool]:
    exact_key = (
        prediction.experiment,
        prediction.snapshot,
        prediction.sensor_label,
        prediction.start_frame,
        prediction.end_frame,
    )
    if exact_key in exact:
        return exact[exact_key], True
    candidates = fallback.get(
        (prediction.experiment, prediction.snapshot, prediction.sensor_label),
        [],
    )
    if not candidates:
        return None, False
    selected = min(
        candidates,
        key=lambda row: abs(row.start_frame - prediction.start_frame)
        + abs(row.end_frame - prediction.end_frame),
    )
    return selected, False


def resolve_track_path(raw_path: str, dataset_root: Path | None) -> Path:
    """Resolve metadata paths on Windows or through a WSL /mnt/<drive> mount."""
    direct = Path(raw_path)
    if direct.exists():
        return direct

    windows_path = PureWindowsPath(raw_path)
    if dataset_root is not None:
        candidate = dataset_root / Path(*windows_path.parts[2:])
        if candidate.exists():
            return candidate
        candidate = dataset_root / windows_path.name
        if candidate.exists():
            return candidate

    drive_match = re.match(r"^([A-Za-z]):[\\/](.*)$", raw_path)
    if os.name != "nt" and drive_match:
        drive = drive_match.group(1).lower()
        relative = drive_match.group(2).replace("\\", "/")
        mounted = Path("/mnt") / drive / Path(relative)
        if mounted.exists():
            return mounted

    return direct


def finite_array(rows: Sequence[dict[str, str]], key: str) -> np.ndarray:
    values = np.asarray([parse_float(row.get(key)) for row in rows], dtype=np.float64)
    return values[np.isfinite(values)]


def finite_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else math.nan


def finite_max(values: np.ndarray) -> float:
    return float(np.max(values)) if values.size else math.nan


def unique_rows_by_frame(rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    by_frame: dict[int, dict[str, str]] = {}
    for row in rows:
        frame = parse_int(row.get("frame"))
        if frame >= 0:
            by_frame.setdefault(frame, row)
    return [by_frame[frame] for frame in sorted(by_frame)]


def longest_consecutive_frame_run(rows: Sequence[dict[str, str]]) -> int:
    frames: list[int] = []
    for row in rows:
        frame = parse_int(row.get("frame"))
        if frame >= 0:
            frames.append(frame)
    frames = sorted(set(frames))
    if not frames:
        return 0

    longest = 1
    current = 1
    for previous, frame in zip(frames, frames[1:]):
        if frame == previous + 1:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    return longest


def compute_track_descriptors(
    rows: Sequence[dict[str, str]],
    segment_length: int,
) -> tuple[dict[str, float], tuple[int, ...]]:
    ordered_rows = unique_rows_by_frame(rows)
    frames = np.asarray([parse_int(row.get("frame")) for row in ordered_rows], dtype=np.float64)
    core_x = np.asarray([parse_float(row.get("core_x")) for row in ordered_rows], dtype=np.float64)
    core_y = np.asarray([parse_float(row.get("core_y")) for row in ordered_rows], dtype=np.float64)

    valid_position = np.isfinite(core_x) & np.isfinite(core_y) & np.isfinite(frames)
    frames = frames[valid_position]
    core_x = core_x[valid_position]
    core_y = core_y[valid_position]

    displacement = 0.0
    path_length = 0.0
    mean_speed = math.nan
    max_speed = math.nan
    if frames.size >= 2:
        dx = np.diff(core_x)
        dy = np.diff(core_y)
        distances = np.hypot(dx, dy)
        frame_steps = np.diff(frames)
        valid_steps = frame_steps > 0
        path_length = float(np.sum(distances[valid_steps]))
        speeds = distances[valid_steps] / frame_steps[valid_steps]
        mean_speed = finite_mean(speeds)
        max_speed = finite_max(speeds)
        displacement = float(math.hypot(core_x[-1] - core_x[0], core_y[-1] - core_y[0]))

    straightness = displacement / path_length if path_length > 0.0 else math.nan
    area = finite_array(ordered_rows, "area")
    radius = finite_array(ordered_rows, "radius")
    core_q = finite_array(ordered_rows, "core_q")
    body_distance = finite_array(ordered_rows, "body_distance")
    sensor_distance = finite_array(ordered_rows, "sensor_distance")
    duration_frames = int(frames.size)

    descriptors = {
        "mean_area": finite_mean(area),
        "max_area": finite_max(area),
        "mean_equivalent_radius": finite_mean(radius),
        "max_equivalent_radius": finite_max(radius),
        "displacement": displacement,
        "path_length": path_length,
        "straightness": straightness,
        "duration_frames": float(duration_frames),
        "duration_ratio": duration_frames / segment_length if segment_length > 0 else math.nan,
        "mean_speed": mean_speed,
        "max_speed": max_speed,
        "mean_core_q": finite_mean(core_q),
        "max_core_q": finite_max(core_q),
        "mean_body_distance": finite_mean(body_distance),
        "mean_sensor_distance": finite_mean(sensor_distance),
        "mean_core_x": finite_mean(core_x),
        "mean_core_y": finite_mean(core_y),
    }
    return descriptors, tuple(int(frame) for frame in frames)


def load_track_rows(path: Path) -> list[dict[str, str]]:
    return read_csv(path)


def collect_pressure_conditioned_tracks(
    predictions_path: Path,
    fish_metadata_path: Path,
    cylinder_metadata_path: Path,
    confidence_threshold: float,
    min_segment_frames: int,
    min_consecutive_visible_frames: int,
    dataset_root: Path | None,
    allow_missing_tracks: bool,
) -> tuple[list[dict[str, object]], Counter[str]]:
    predictions = load_predictions(predictions_path)
    fish_exact, fish_fallback = build_metadata_index(load_metadata(fish_metadata_path))
    cylinder_exact, cylinder_fallback = build_metadata_index(load_metadata(cylinder_metadata_path))
    indices = {
        "fish": (fish_exact, fish_fallback),
        "cylinder": (cylinder_exact, cylinder_fallback),
    }

    audit: Counter[str] = Counter()
    track_cache: dict[Path, list[dict[str, str]]] = {}
    retained: list[dict[str, object]] = []
    association_keys: set[tuple[object, ...]] = set()
    missing_paths: set[str] = set()

    audit["prediction_rows"] = len(predictions)
    for prediction in predictions:
        pressure_class = classify_prediction(prediction.probability, confidence_threshold)
        audit[f"prediction_class_{pressure_class}"] += 1
        if pressure_class == "none":
            continue

        exact, fallback = indices[pressure_class]
        metadata, exact_match = select_metadata(exact, fallback, prediction)
        if metadata is None:
            audit["predictions_without_matching_metadata"] += 1
            continue
        audit["metadata_exact_matches" if exact_match else "metadata_fallback_matches"] += 1

        segment_length = metadata.end_frame - metadata.start_frame + 1
        if segment_length < min_segment_frames:
            audit["segments_below_minimum_length"] += 1
            continue

        track_path = resolve_track_path(metadata.track_path_raw, dataset_root)
        if not track_path.exists():
            audit["missing_track_paths"] += 1
            missing_paths.add(str(track_path))
            continue
        if track_path not in track_cache:
            track_cache[track_path] = load_track_rows(track_path)

        grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
        for row in track_cache[track_path]:
            frame = parse_int(row.get("frame"))
            if frame < metadata.start_frame or frame > metadata.end_frame:
                continue
            if parse_int(row.get("sensor_id")) != metadata.sensor_id:
                continue
            if parse_int(row.get("visible"), default=1) == 0:
                continue
            track_id = parse_int(row.get("track_id"))
            if track_id < 0:
                continue
            if not math.isfinite(parse_float(row.get("core_x"))) or not math.isfinite(
                parse_float(row.get("core_y"))
            ):
                continue
            grouped[track_id].append(row)

        audit["matched_pressure_segments"] += 1
        for track_id, track_rows in grouped.items():
            valid_rows = unique_rows_by_frame(track_rows)
            longest_run = longest_consecutive_frame_run(valid_rows)
            if longest_run < min_consecutive_visible_frames:
                audit["track_associations_below_minimum_consecutive_visible_frames"] += 1
                continue

            association_key = (
                pressure_class,
                metadata.experiment,
                metadata.snapshot,
                metadata.sensor_id,
                metadata.start_frame,
                metadata.end_frame,
                str(track_path),
                track_id,
            )
            if association_key in association_keys:
                audit["duplicate_track_associations"] += 1
                continue
            association_keys.add(association_key)

            descriptors, visible_frames = compute_track_descriptors(valid_rows, segment_length)
            segment_id = (
                f"{pressure_class}|{metadata.experiment}|{metadata.snapshot}|"
                f"{metadata.sensor_id}|{metadata.start_frame}|{metadata.end_frame}|{track_path}"
            )
            confidence = (
                min(1.0, max(0.0, prediction.probability))
                if pressure_class == "cylinder"
                else 1.0 - min(1.0, max(0.0, prediction.probability))
            )
            retained.append(
                {
                    "dataset": prediction.dataset,
                    "pressure_class": pressure_class,
                    "model": pressure_class,
                    "source_prediction_model": prediction.source_model,
                    "experiment": metadata.experiment,
                    "snapshot": metadata.snapshot,
                    "sensor": prediction.sensor_label,
                    "sensor_id": metadata.sensor_id,
                    "segment_start_frame": metadata.start_frame,
                    "segment_end_frame": metadata.end_frame,
                    "segment_length": segment_length,
                    "segment_id": segment_id,
                    "track_id": track_id,
                    "track_path": str(track_path),
                    "pressure_probability": prediction.probability,
                    "pressure_class_confidence": confidence,
                    "longest_consecutive_visible_frames": longest_run,
                    **descriptors,
                    "_visible_frames": visible_frames,
                }
            )
            audit[f"retained_track_associations_{pressure_class}"] += 1

    if missing_paths and not allow_missing_tracks:
        examples = "\n".join(sorted(missing_paths)[:10])
        raise FileNotFoundError(
            f"{len(missing_paths)} required vortex-track files were not found. "
            "Use --dataset-root to remap metadata paths or --allow-missing-tracks "
            f"to continue with incomplete data. Examples:\n{examples}"
        )

    physical_counts = Counter(
        (row["model"], row["track_path"], row["track_id"]) for row in retained
    )
    audit["physical_tracks_reused_across_associations"] = sum(
        count > 1 for count in physical_counts.values()
    )
    return retained, audit


def values_for(rows: Sequence[dict[str, object]], metric: str) -> np.ndarray:
    values = np.asarray([parse_float(str(row.get(metric, ""))) for row in rows], dtype=np.float64)
    return values[np.isfinite(values)]


def mean_metric(rows: Sequence[dict[str, object]], metric: str) -> float:
    return finite_mean(values_for(rows, metric))


def build_segment_summaries(track_rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in track_rows:
        groups[str(row["segment_id"])].append(row)

    summaries: list[dict[str, object]] = []
    for segment_id, rows in groups.items():
        first = rows[0]
        segment_length = int(first["segment_length"])
        concurrency: Counter[int] = Counter()
        for row in rows:
            for frame in set(row["_visible_frames"]):
                concurrency[int(frame)] += 1

        total_vortex_frames = sum(concurrency.values())
        complete_counts = [
            concurrency.get(frame, 0)
            for frame in range(
                int(first["segment_start_frame"]),
                int(first["segment_end_frame"]) + 1,
            )
        ]
        summary: dict[str, object] = {
            "dataset": first["dataset"],
            "pressure_class": first["pressure_class"],
            "model": first["model"],
            "experiment": first["experiment"],
            "snapshot": first["snapshot"],
            "sensor": first["sensor"],
            "sensor_id": first["sensor_id"],
            "segment_start_frame": first["segment_start_frame"],
            "segment_end_frame": first["segment_end_frame"],
            "segment_length": segment_length,
            "segment_id": segment_id,
            "vortex_count": len(rows),
            "total_vortex_frames": total_vortex_frames,
            "mean_simultaneous_vortices": (
                float(np.mean(complete_counts)) if complete_counts else math.nan
            ),
            "max_simultaneous_vortices": max(complete_counts, default=0),
        }
        for metric in TRACK_DESCRIPTORS:
            summary[metric] = mean_metric(rows, metric)
        summaries.append(summary)
    return summaries


def build_video_summaries(segment_rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in segment_rows:
        groups[(str(row["pressure_class"]), int(row["experiment"]))].append(row)

    summaries: list[dict[str, object]] = []
    for (pressure_class, experiment), rows in groups.items():
        summary: dict[str, object] = {
            "dataset": rows[0]["dataset"],
            "pressure_class": pressure_class,
            "model": pressure_class,
            "experiment": experiment,
            "segment_count": len(rows),
            "total_vortex_associations": int(sum(int(row["vortex_count"]) for row in rows)),
        }
        for metric in SEGMENT_METRICS:
            summary[metric] = mean_metric(rows, metric)
        summaries.append(summary)
    return summaries


def build_paired_video_differences(
    video_rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    by_video: dict[int, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in video_rows:
        by_video[int(row["experiment"])][str(row["pressure_class"])] = row

    differences: list[dict[str, object]] = []
    for experiment, classes in sorted(by_video.items()):
        if not all(model in classes for model in MODEL_ORDER):
            continue
        fish = classes["fish"]
        cylinder = classes["cylinder"]
        row: dict[str, object] = {
            "dataset": fish["dataset"],
            "experiment": experiment,
            "fish_segment_count": fish["segment_count"],
            "cylinder_segment_count": cylinder["segment_count"],
        }
        for metric in SEGMENT_METRICS:
            fish_value = parse_float(str(fish.get(metric, "")))
            cylinder_value = parse_float(str(cylinder.get(metric, "")))
            row[f"fish_{metric}"] = fish_value
            row[f"cylinder_{metric}"] = cylinder_value
            row[f"cylinder_minus_fish_{metric}"] = (
                cylinder_value - fish_value
                if math.isfinite(fish_value) and math.isfinite(cylinder_value)
                else math.nan
            )
        differences.append(row)
    return differences


def quantile(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(values, probability)) if values.size else math.nan


def describe(values: np.ndarray) -> dict[str, float | int]:
    return {
        "n": int(values.size),
        "mean": finite_mean(values),
        "median": quantile(values, 0.5),
        "q1": quantile(values, 0.25),
        "q3": quantile(values, 0.75),
        "iqr": quantile(values, 0.75) - quantile(values, 0.25) if values.size else math.nan,
    }


def distribution_summary(
    rows: Sequence[dict[str, object]],
    metrics: Sequence[str],
    level: str,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for pressure_class in MODEL_ORDER:
        class_rows = [row for row in rows if row.get("pressure_class") == pressure_class]
        for metric in metrics:
            output.append(
                {
                    "level": level,
                    "pressure_class": pressure_class,
                    "metric": metric,
                    **describe(values_for(class_rows, metric)),
                }
            )
    return output


def class_comparison_summary(
    rows: Sequence[dict[str, object]],
    metrics: Sequence[str],
    level: str,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for metric in metrics:
        fish = describe(
            values_for([row for row in rows if row.get("pressure_class") == "fish"], metric)
        )
        cylinder = describe(
            values_for(
                [row for row in rows if row.get("pressure_class") == "cylinder"], metric
            )
        )
        fish_median = float(fish["median"])
        cylinder_median = float(cylinder["median"])
        output.append(
            {
                "level": level,
                "metric": metric,
                "n_fish": fish["n"],
                "fish_median": fish_median,
                "fish_q1": fish["q1"],
                "fish_q3": fish["q3"],
                "n_cylinder": cylinder["n"],
                "cylinder_median": cylinder_median,
                "cylinder_q1": cylinder["q1"],
                "cylinder_q3": cylinder["q3"],
                "cylinder_minus_fish_median": (
                    cylinder_median - fish_median
                    if math.isfinite(fish_median) and math.isfinite(cylinder_median)
                    else math.nan
                ),
            }
        )
    return output


def paired_distribution_summary(
    paired_rows: Sequence[dict[str, object]],
    metrics: Sequence[str],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for metric in metrics:
        difference_metric = f"cylinder_minus_fish_{metric}"
        output.append(
            {
                "level": "paired_video",
                "metric": metric,
                **describe(values_for(paired_rows, difference_metric)),
            }
        )
    return output


def clean_rows_for_csv(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    return [{key: value for key, value in row.items() if not key.startswith("_")} for row in rows]


def save_figure(fig, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(f"{output_base}.pdf"), bbox_inches="tight")
    fig.savefig(Path(f"{output_base}.png"), dpi=180, bbox_inches="tight")


def load_noninteractive_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plots. Use --no-plots to skip them.") from exc
    return plt


def plot_class_distributions(
    rows: Sequence[dict[str, object]],
    metrics: Sequence[str],
    title: str,
    output_base: Path,
    seed: int,
) -> None:
    plt = load_noninteractive_pyplot()

    plt.rcParams.update(CAMBRIDGE_RC)
    n_columns = 4
    n_rows = math.ceil(len(metrics) / n_columns)
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(12.0, 2.65 * n_rows), squeeze=False)
    rng = np.random.default_rng(seed)

    for index, metric in enumerate(metrics):
        ax = axes.flat[index]
        values_by_class = [
            values_for([row for row in rows if row.get("pressure_class") == model], metric)
            for model in MODEL_ORDER
        ]
        box = ax.boxplot(
            values_by_class,
            positions=(1, 2),
            widths=0.55,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.1},
            whiskerprops={"linewidth": 0.8},
            capprops={"linewidth": 0.8},
        )
        for patch, model in zip(box["boxes"], MODEL_ORDER):
            patch.set_facecolor(MODEL_COLOURS[model])
            patch.set_alpha(0.35)
            patch.set_edgecolor(MODEL_COLOURS[model])

        for position, model, values in zip((1, 2), MODEL_ORDER, values_by_class):
            if values.size:
                jitter = rng.uniform(-0.13, 0.13, size=values.size)
                ax.scatter(
                    np.full(values.size, position) + jitter,
                    values,
                    s=8,
                    alpha=0.32,
                    color=MODEL_COLOURS[model],
                    linewidths=0,
                )
        ax.set_title(METRIC_LABELS.get(metric, metric))
        ax.set_xticks((1, 2), ("Fish", "Cylinder"))
        ax.grid(axis="y", alpha=0.18)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4), useOffset=False)

    for index in range(len(metrics), len(axes.flat)):
        axes.flat[index].axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.975))
    save_figure(fig, output_base)
    plt.close(fig)


def plot_paired_differences(
    paired_rows: Sequence[dict[str, object]],
    metrics: Sequence[str],
    title: str,
    output_base: Path,
    seed: int,
) -> None:
    plt = load_noninteractive_pyplot()

    plt.rcParams.update(CAMBRIDGE_RC)
    n_columns = 4
    n_rows = math.ceil(len(metrics) / n_columns)
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(12.0, 2.65 * n_rows), squeeze=False)
    rng = np.random.default_rng(seed)

    for index, metric in enumerate(metrics):
        ax = axes.flat[index]
        values = values_for(paired_rows, f"cylinder_minus_fish_{metric}")
        if values.size:
            box = ax.boxplot(
                [values],
                positions=(1,),
                widths=0.45,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "black", "linewidth": 1.1},
            )
            box["boxes"][0].set_facecolor("#777777")
            box["boxes"][0].set_alpha(0.25)
            jitter = rng.uniform(-0.12, 0.12, size=values.size)
            ax.scatter(np.ones(values.size) + jitter, values, s=11, color="#444444", alpha=0.6)
        ax.axhline(0.0, color="black", linestyle=":", linewidth=0.8)
        ax.set_title(METRIC_LABELS.get(metric, metric))
        ax.set_xticks((1,), ("Cyl. - fish",))
        ax.grid(axis="y", alpha=0.18)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4), useOffset=False)

    for index in range(len(metrics), len(axes.flat)):
        axes.flat[index].axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.975))
    save_figure(fig, output_base)
    plt.close(fig)


def format_value(value: object) -> str:
    parsed = parse_float(str(value))
    return f"{parsed:.6g}" if math.isfinite(parsed) else "NA"


def write_audit_report(
    path: Path,
    args: argparse.Namespace,
    audit: Counter[str],
    tracks: Sequence[dict[str, object]],
    segments: Sequence[dict[str, object]],
    videos: Sequence[dict[str, object]],
    paired: Sequence[dict[str, object]],
) -> None:
    lines = [
        "Pressure-conditioned vortex analysis audit",
        "",
        f"Dataset: {args.dataset}",
        f"Architecture: {args.architecture}",
        f"Pressure-mode energy: {args.modes_energy}",
        f"Pressure confidence threshold: {args.confidence_threshold}",
        f"Minimum pressure-segment length: {args.min_segment_frames} frames",
        "Minimum consecutive visible track detections: "
        f"{args.min_consecutive_visible_frames} frames",
        "Class balancing: disabled",
        "Classifier training: not performed",
        "SHAP calculation: not performed",
        "",
        "Selection audit:",
    ]
    lines.extend(f"- {key}: {audit[key]}" for key in sorted(audit))
    lines.extend(["", "Retained analysis units:"])
    for model in MODEL_ORDER:
        model_tracks = [row for row in tracks if row["pressure_class"] == model]
        model_segments = [row for row in segments if row["pressure_class"] == model]
        model_videos = [row for row in videos if row["pressure_class"] == model]
        lines.append(
            f"- {model}: {len(model_tracks)} vortex tracks, "
            f"{len(model_segments)} pressure segments, {len(model_videos)} source videos"
        )
    lines.append(f"- paired source videos: {len(paired)}")
    lines.extend(["", "Segment-level quantities:"])
    for metric in SEGMENT_EXTRA_METRICS:
        comparison = class_comparison_summary(segments, [metric], "segment")[0]
        lines.append(
            f"- {metric}: fish median {format_value(comparison['fish_median'])}, "
            f"cylinder median {format_value(comparison['cylinder_median'])}, "
            f"cylinder-fish {format_value(comparison['cylinder_minus_fish_median'])}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    default_output = (
        repo_root
        / "python/vortexAnalysis_v2/vortex_outputs/revision_analysis/pressure_conditioned_vortices"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Compare descriptors of existing pressure-conditioned vortex tracks "
            "without training a classifier."
        )
    )
    parser.add_argument("--dataset", default="boai", help="Simulation variant name.")
    parser.add_argument("--architecture", default="linear", help="Pressure classifier name.")
    parser.add_argument(
        "--modes-energy",
        type=float,
        default=0.95,
        help="Existing pressure-mode energy used for the predictions.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.99,
        help="Confidence threshold for fish-like and cylinder-like pressure labels.",
    )
    parser.add_argument(
        "--min-segment-frames",
        type=int,
        default=100,
        help="Minimum associated pressure-segment length.",
    )
    parser.add_argument(
        "--min-consecutive-visible-frames",
        "--min-visible-frames",
        dest="min_consecutive_visible_frames",
        type=int,
        default=10,
        help=(
            "Minimum consecutive valid visible detections required for a vortex track. "
            "--min-visible-frames is retained as a compatibility alias."
        ),
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=repo_root / "python/vortexAnalysis_v2/vortex_outputs/pressure_modes",
        help="Directory containing existing real-pressure prediction CSVs.",
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=repo_root / "python/vortexAnalysis_v2/vortex_outputs/pressure_snapshots",
        help="Directory containing fish and cylinder pressure-snapshot metadata.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Optional root used to remap track paths stored in metadata.",
    )
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument(
        "--allow-missing-tracks",
        action="store_true",
        help="Continue with incomplete data when required track files are missing.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Write tables without figures.")
    parser.add_argument("--seed", type=int, default=42, help="Seed used only for plot jitter.")
    args = parser.parse_args()

    if not 0.5 <= args.confidence_threshold <= 1.0:
        parser.error("--confidence-threshold must be between 0.5 and 1.0")
    if args.min_consecutive_visible_frames < 1 or args.min_segment_frames < 1:
        parser.error("minimum frame counts must be positive")

    energy_tag = format_energy_tag(args.modes_energy)
    run_tag = f"{args.dataset}_{energy_tag}_thr{args.confidence_threshold:.2f}"
    predictions_path = (
        args.predictions_dir
        / f"real_pressure_predictions_{args.architecture}_{args.dataset}_{energy_tag}.csv"
    )
    fish_metadata_path = args.snapshot_dir / f"pressure_snapshots_{args.dataset}_fish_meta.csv"
    cylinder_metadata_path = (
        args.snapshot_dir / f"pressure_snapshots_{args.dataset}_cylinder_meta.csv"
    )

    track_rows, audit = collect_pressure_conditioned_tracks(
        predictions_path=predictions_path,
        fish_metadata_path=fish_metadata_path,
        cylinder_metadata_path=cylinder_metadata_path,
        confidence_threshold=args.confidence_threshold,
        min_segment_frames=args.min_segment_frames,
        min_consecutive_visible_frames=args.min_consecutive_visible_frames,
        dataset_root=args.dataset_root,
        allow_missing_tracks=args.allow_missing_tracks,
    )
    if not track_rows:
        raise RuntimeError("No vortex tracks remained after pressure conditioning and filtering.")

    segment_rows = build_segment_summaries(track_rows)
    video_rows = build_video_summaries(segment_rows)
    paired_rows = build_paired_video_differences(video_rows)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    track_fields = (
        "dataset",
        "pressure_class",
        "model",
        "source_prediction_model",
        "experiment",
        "snapshot",
        "sensor",
        "sensor_id",
        "segment_start_frame",
        "segment_end_frame",
        "segment_length",
        "segment_id",
        "track_id",
        "track_path",
        "pressure_probability",
        "pressure_class_confidence",
        "longest_consecutive_visible_frames",
    ) + TRACK_DESCRIPTORS
    segment_fields = (
        "dataset",
        "pressure_class",
        "model",
        "experiment",
        "snapshot",
        "sensor",
        "sensor_id",
        "segment_start_frame",
        "segment_end_frame",
        "segment_length",
        "segment_id",
    ) + SEGMENT_METRICS
    video_fields = (
        "dataset",
        "pressure_class",
        "model",
        "experiment",
        "segment_count",
        "total_vortex_associations",
    ) + SEGMENT_METRICS
    paired_fields: list[str] = [
        "dataset",
        "experiment",
        "fish_segment_count",
        "cylinder_segment_count",
    ]
    for metric in SEGMENT_METRICS:
        paired_fields.extend(
            (f"fish_{metric}", f"cylinder_{metric}", f"cylinder_minus_fish_{metric}")
        )

    write_csv(
        output_dir / f"vortex_track_descriptors_{run_tag}.csv",
        clean_rows_for_csv(track_rows),
        track_fields,
    )
    write_csv(
        output_dir / f"vortex_segment_summaries_{run_tag}.csv",
        segment_rows,
        segment_fields,
    )
    write_csv(
        output_dir / f"vortex_video_summaries_{run_tag}.csv",
        video_rows,
        video_fields,
    )
    write_csv(
        output_dir / f"vortex_paired_video_differences_{run_tag}.csv",
        paired_rows,
        paired_fields,
    )

    distribution_rows = (
        distribution_summary(track_rows, TRACK_DESCRIPTORS, "pooled_vortex")
        + distribution_summary(segment_rows, SEGMENT_METRICS, "segment")
        + distribution_summary(video_rows, SEGMENT_METRICS, "video")
    )
    write_csv(
        output_dir / f"vortex_distribution_summary_{run_tag}.csv",
        distribution_rows,
        ("level", "pressure_class", "metric", "n", "mean", "median", "q1", "q3", "iqr"),
    )

    comparison_rows = (
        class_comparison_summary(track_rows, TRACK_DESCRIPTORS, "pooled_vortex")
        + class_comparison_summary(segment_rows, SEGMENT_METRICS, "segment")
        + class_comparison_summary(video_rows, SEGMENT_METRICS, "video")
    )
    write_csv(
        output_dir / f"vortex_class_comparisons_{run_tag}.csv",
        comparison_rows,
        (
            "level",
            "metric",
            "n_fish",
            "fish_median",
            "fish_q1",
            "fish_q3",
            "n_cylinder",
            "cylinder_median",
            "cylinder_q1",
            "cylinder_q3",
            "cylinder_minus_fish_median",
        ),
    )

    paired_summary_rows = paired_distribution_summary(paired_rows, SEGMENT_METRICS)
    write_csv(
        output_dir / f"vortex_paired_video_summary_{run_tag}.csv",
        paired_summary_rows,
        ("level", "metric", "n", "mean", "median", "q1", "q3", "iqr"),
    )

    write_audit_report(
        output_dir / f"vortex_analysis_audit_{run_tag}.txt",
        args,
        audit,
        track_rows,
        segment_rows,
        video_rows,
        paired_rows,
    )

    if not args.no_plots:
        plot_class_distributions(
            track_rows,
            TRACK_DESCRIPTORS,
            "Pooled pressure-conditioned vortex descriptors",
            output_dir / f"vortex_pooled_distributions_{run_tag}",
            args.seed,
        )
        plot_class_distributions(
            segment_rows,
            SEGMENT_METRICS,
            "Equal-weight pressure-segment summaries",
            output_dir / f"vortex_segment_distributions_{run_tag}",
            args.seed + 1,
        )
        plot_paired_differences(
            paired_rows,
            SEGMENT_METRICS,
            "Paired source-video differences",
            output_dir / f"vortex_paired_video_differences_{run_tag}",
            args.seed + 2,
        )

    counts = {
        model: (
            sum(row["pressure_class"] == model for row in track_rows),
            sum(row["pressure_class"] == model for row in segment_rows),
            sum(row["pressure_class"] == model for row in video_rows),
        )
        for model in MODEL_ORDER
    }
    print(f"Analysis complete: {output_dir}")
    for model in MODEL_ORDER:
        n_tracks, n_segments, n_videos = counts[model]
        print(f"  {model}: {n_tracks} tracks, {n_segments} segments, {n_videos} videos")
    print(f"  paired videos: {len(paired_rows)}")


if __name__ == "__main__":
    main()
