#!/usr/bin/env python3
"""Metric discovery, evaluation, and active-learning utilities for ASMR 2.0."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

import discover_asmr_metrics as discovery


BUNDLE_VERSION = 1
SENSOR_COLUMN_RE = re.compile(r"^ESP(\d+)_Sensor(\d+)$")


@dataclass(frozen=True)
class MetricBundle:
    path: Path
    models: dict[str, dict[str, Any]]
    extractor: dict[str, Any]
    domain: dict[str, tuple[float, float]]
    candidate_metrics_path: Path


@dataclass(frozen=True)
class SelectionResult:
    re_value: float
    h_value: float
    score: float
    strategy: str
    candidates_considered: int


@dataclass(frozen=True)
class ExperimentalTrace:
    pressure_pa: np.ndarray
    time_s: np.ndarray
    sensor_columns: tuple[str, ...]
    alignment_source: str
    motion_origin_s: float


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_optional_int(value: Any) -> int | None:
    if value in (None, "", 0):
        return None
    return int(value)


def _discovery_namespace(config: Mapping[str, Any]) -> argparse.Namespace:
    """Create the complete namespace expected by discover_asmr_metrics.run_discovery."""

    return argparse.Namespace(
        root=Path(config["root"]),
        pattern=str(config.get("pattern", "**/pressureMotion.txt")),
        metadata_csv=config.get("metadata_csv") or None,
        path_regex=str(config.get("path_regex", discovery.DEFAULT_PATH_REGEX)),
        parameter_columns=str(config.get("parameter_columns", "Re,h")),
        targets=str(config.get("targets", "log10_Re,log10_h")),
        input_format=str(config.get("input_format", "lilypad-wall")),
        output_dir=Path(config["output_dir"]),
        warmup_samples=int(config.get("warmup_samples", 4)),
        max_sensors=_as_optional_int(config.get("max_sensors")),
        min_cases=int(config.get("min_cases", 6)),
        cv_folds=int(config.get("cv_folds", 5)),
        top_k=int(config.get("top_k", 15)),
        degree=int(config.get("degree", 1)),
        max_base_features=int(config.get("max_base_features", 40)),
        max_interaction_sources=int(config.get("max_interaction_sources", 12)),
        max_library_features=int(config.get("max_library_features", 80)),
        max_terms=int(config.get("max_terms", 5)),
        max_classic_results=int(config.get("max_classic_results", 40)),
    )


def _domain_from_candidates(candidate_path: Path) -> dict[str, tuple[float, float]]:
    candidates = pd.read_csv(candidate_path)
    domain: dict[str, tuple[float, float]] = {}
    for parameter in ("Re", "h"):
        if parameter not in candidates.columns:
            raise ValueError(f"Candidate metric file is missing '{parameter}': {candidate_path}")
        values = pd.to_numeric(candidates[parameter], errors="coerce")
        values = values[np.isfinite(values)]
        if values.empty:
            raise ValueError(f"Candidate metric file has no finite '{parameter}' values.")
        domain[parameter] = (float(values.min()), float(values.max()))
    return domain


def _write_bundle_manifest(
    output_dir: Path,
    config: Mapping[str, Any],
    model_path: Path,
    candidate_path: Path,
) -> Path:
    models = json.loads(model_path.read_text(encoding="utf-8"))
    domain = _domain_from_candidates(candidate_path)
    manifest = {
        "bundle_version": BUNDLE_VERSION,
        "created_at": _utc_now(),
        "models": models,
        "extractor": {
            "input_format": str(config.get("input_format", "lilypad-wall")),
            "warmup_samples": int(config.get("warmup_samples", 4)),
            "max_sensors": _as_optional_int(config.get("max_sensors")),
        },
        "domain": {key: [limits[0], limits[1]] for key, limits in domain.items()},
        "candidate_metrics": candidate_path.name,
        "source": {
            "root": str(config["root"]),
            "pattern": str(config.get("pattern", "**/pressureMotion.txt")),
            "parameter_columns": str(config.get("parameter_columns", "Re,h")),
            "targets": str(config.get("targets", "log10_Re,log10_h")),
        },
    }
    bundle_path = output_dir / "metric_bundle.json"
    bundle_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return bundle_path


def ensure_metric_bundle(config: Mapping[str, Any], refresh: bool = False) -> MetricBundle:
    """Load a cached metric bundle, running discovery only when required."""

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "sparse_models.json"
    candidate_path = output_dir / "candidate_metric_values.csv"
    bundle_path = output_dir / "metric_bundle.json"

    if refresh or not model_path.exists() or not candidate_path.exists():
        discovery.run_discovery(_discovery_namespace(config))

    cache_sources_are_newer = bundle_path.exists() and (
        model_path.stat().st_mtime > bundle_path.stat().st_mtime
        or candidate_path.stat().st_mtime > bundle_path.stat().st_mtime
    )
    if refresh or not bundle_path.exists() or cache_sources_are_newer:
        if not model_path.exists() or not candidate_path.exists():
            raise FileNotFoundError(
                "Metric discovery did not create sparse_models.json and "
                "candidate_metric_values.csv."
            )
        bundle_path = _write_bundle_manifest(output_dir, config, model_path, candidate_path)

    raw = json.loads(bundle_path.read_text(encoding="utf-8"))
    if int(raw.get("bundle_version", -1)) != BUNDLE_VERSION:
        raise ValueError(
            f"Unsupported metric bundle version {raw.get('bundle_version')} in {bundle_path}."
        )

    candidate_metrics_path = output_dir / str(
        raw.get("candidate_metrics", "candidate_metric_values.csv")
    )
    domain = {
        name: (float(limits[0]), float(limits[1]))
        for name, limits in raw["domain"].items()
    }
    return MetricBundle(
        path=bundle_path,
        models=dict(raw["models"]),
        extractor=dict(raw["extractor"]),
        domain=domain,
        candidate_metrics_path=candidate_metrics_path,
    )


def _natural_sensor_key(name: str) -> tuple[int, int]:
    match = SENSOR_COLUMN_RE.fullmatch(name)
    if not match:
        return (math.inf, math.inf)
    return int(match.group(1)), int(match.group(2))


def load_experimental_trace(
    file_path: Path,
    delimiter: str = ";",
    timestamp_column: str = "timestamp",
    max_sensors: int | None = None,
) -> ExperimentalTrace:
    """Read Interface-style UDP CSV data and apply the legacy pressure scaling."""

    frame = pd.read_csv(file_path, sep=delimiter, decimal=",")
    if timestamp_column not in frame.columns:
        raise ValueError(f"Missing timestamp column '{timestamp_column}' in {file_path}")

    sensor_columns = sorted(
        [column for column in frame.columns if SENSOR_COLUMN_RE.fullmatch(str(column))],
        key=_natural_sensor_key,
    )
    if max_sensors is not None:
        sensor_columns = sensor_columns[:max_sensors]
    if not sensor_columns:
        raise ValueError(f"No ESP*_Sensor* columns found in {file_path}")

    timestamp = pd.to_numeric(frame[timestamp_column], errors="coerce").to_numpy(dtype=float)
    pressure_raw = frame[sensor_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    valid_rows = np.isfinite(timestamp) & np.isfinite(pressure_raw).any(axis=1)
    timestamp = timestamp[valid_rows]
    pressure_raw = pressure_raw[valid_rows]
    if timestamp.size < 8:
        raise ValueError(f"Not enough valid UDP samples in {file_path}")

    order = np.argsort(timestamp)
    timestamp = timestamp[order]
    pressure_raw = pressure_raw[order]
    time_s = (timestamp - timestamp[0]) / 1_000_000.0
    frame = frame.iloc[np.flatnonzero(valid_rows)[order]].reset_index(drop=True)

    # This is algebraically equivalent to the legacy ASMR conversion after
    # centering each signed-int16 channel:
    # (centered + 32768) * 1000 / 65536 - 500.
    channel_means = np.nanmean(pressure_raw, axis=0, keepdims=True)
    centered = pressure_raw - channel_means
    pressure_pa = centered * (1000.0 / 65536.0)

    alignment_source = "recording_start"
    motion_origin_s = 0.0
    if "CNC_Moving" in frame.columns:
        moving = pd.to_numeric(frame["CNC_Moving"], errors="coerce").fillna(0).to_numpy()
        moving_indices = np.flatnonzero(moving > 0)
        if moving_indices.size:
            motion_origin_s = float(time_s[moving_indices[0]])
            alignment_source = "CNC_Moving"
    if alignment_source == "recording_start" and "CNC_X" in frame.columns:
        cnc_x = pd.to_numeric(frame["CNC_X"], errors="coerce").to_numpy(dtype=float)
        finite = np.flatnonzero(np.isfinite(cnc_x))
        if finite.size >= 2:
            initial_x = cnc_x[finite[0]]
            moved = finite[np.abs(cnc_x[finite] - initial_x) > 1e-3]
            if moved.size:
                motion_origin_s = float(time_s[moved[0]])
                alignment_source = "CNC_X"

    return ExperimentalTrace(
        pressure_pa=pressure_pa,
        time_s=time_s,
        sensor_columns=tuple(str(column) for column in sensor_columns),
        alignment_source=alignment_source,
        motion_origin_s=motion_origin_s,
    )


def load_experimental_pressure(
    file_path: Path,
    delimiter: str = ";",
    timestamp_column: str = "timestamp",
    max_sensors: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    trace = load_experimental_trace(
        file_path=file_path,
        delimiter=delimiter,
        timestamp_column=timestamp_column,
        max_sensors=max_sensors,
    )
    return trace.pressure_pa, trace.time_s


def _interpolate_trace(
    source_time: np.ndarray,
    source_pressure: np.ndarray,
    target_time: np.ndarray,
) -> np.ndarray:
    interpolated = np.empty((target_time.size, source_pressure.shape[1]), dtype=float)
    for column in range(source_pressure.shape[1]):
        values = source_pressure[:, column]
        finite = np.isfinite(source_time) & np.isfinite(values)
        if np.sum(finite) < 2:
            raise ValueError(
                f"Sensor column {column + 1} has fewer than two finite samples."
            )
        unique_time, unique_indices = np.unique(source_time[finite], return_index=True)
        unique_values = values[finite][unique_indices]
        if unique_time.size < 2:
            raise ValueError(
                f"Sensor column {column + 1} has fewer than two unique timestamps."
            )
        interpolated[:, column] = np.interp(
            target_time, unique_time, unique_values
        )
    return interpolated


def subtract_experimental_recordings(
    measurement_path: Path,
    reference_path: Path,
    output_path: Path,
    delimiter: str = ";",
    max_sensors: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Align paired tows and subtract the Z-offset reference from the measurement."""

    measurement = load_experimental_trace(
        measurement_path, delimiter=delimiter, max_sensors=max_sensors
    )
    reference = load_experimental_trace(
        reference_path, delimiter=delimiter, max_sensors=max_sensors
    )
    if measurement.sensor_columns != reference.sensor_columns:
        raise ValueError(
            "Measurement and reference recordings do not contain the same ordered "
            "sensor columns."
        )

    measurement_time = measurement.time_s - measurement.motion_origin_s
    reference_time = reference.time_s - reference.motion_origin_s
    overlap_start = max(float(measurement_time[0]), float(reference_time[0]))
    overlap_end = min(float(measurement_time[-1]), float(reference_time[-1]))
    if overlap_end <= overlap_start:
        raise ValueError("Paired recordings have no overlapping aligned time interval.")

    measurement_mask = (
        (measurement_time >= overlap_start) & (measurement_time <= overlap_end)
    )
    aligned_time = measurement_time[measurement_mask]
    if aligned_time.size < 8:
        raise ValueError("Fewer than eight aligned samples remain after pairing.")

    measurement_aligned = _interpolate_trace(
        measurement_time,
        measurement.pressure_pa,
        aligned_time,
    )
    reference_aligned = _interpolate_trace(
        reference_time,
        reference.pressure_pa,
        aligned_time,
    )
    subtracted_pressure = measurement_aligned - reference_aligned
    relative_time = aligned_time - aligned_time[0]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = pd.DataFrame(
        subtracted_pressure, columns=list(measurement.sensor_columns)
    )
    output.insert(0, "timestamp", np.rint(relative_time * 1_000_000.0).astype(np.int64))
    output.to_csv(output_path, sep=delimiter, index=False)

    metadata = {
        "measurement_file": str(measurement_path),
        "reference_file": str(reference_path),
        "measurement_alignment": measurement.alignment_source,
        "reference_alignment": reference.alignment_source,
        "measurement_motion_origin_s": measurement.motion_origin_s,
        "reference_motion_origin_s": reference.motion_origin_s,
        "overlap_start_s": overlap_start,
        "overlap_end_s": overlap_end,
        "samples": int(relative_time.size),
        "sensors": list(measurement.sensor_columns),
        "operation": "measurement_minus_reference",
    }
    return subtracted_pressure, relative_time, metadata


def load_simulation_pressure(
    file_path: Path,
    re_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    return discovery.read_lilypad_wall_pressure(file_path, re_value=re_value)


def _evaluate_feature_term(name: str, features: Mapping[str, float]) -> float:
    name = name.strip()
    if "*" in name:
        value = 1.0
        for part in name.split("*"):
            value *= _evaluate_feature_term(part, features)
        return float(value)

    wrappers = (
        ("log10_abs(", lambda value: math.log10(max(abs(value), discovery.EPS))),
        ("signed_log10(", lambda value: math.copysign(math.log10(1.0 + abs(value)), value)),
        ("inv(", lambda value: 1.0 / value),
    )
    for prefix, transform in wrappers:
        if name.startswith(prefix) and name.endswith(")"):
            inner = name[len(prefix) : -1]
            value = _evaluate_feature_term(inner, features)
            if prefix == "inv(" and abs(value) <= discovery.EPS:
                raise ValueError(f"Cannot evaluate inverse feature '{name}' at zero.")
            return float(transform(value))

    if name not in features:
        raise KeyError(f"Discovered metric requires missing feature '{name}'.")
    value = float(features[name])
    if not np.isfinite(value):
        raise ValueError(f"Feature '{name}' is not finite.")
    return value


def evaluate_discovered_metrics(
    pressure: np.ndarray,
    time_s: np.ndarray,
    re_value: float,
    bundle: MetricBundle,
) -> tuple[dict[str, float], dict[str, float]]:
    features = discovery.extract_pressure_features(
        pressure=pressure,
        time=time_s,
        re_value=re_value,
        warmup_samples=int(bundle.extractor.get("warmup_samples", 4)),
        max_sensors=_as_optional_int(bundle.extractor.get("max_sensors")),
    )

    values: dict[str, float] = {}
    for target, model in bundle.models.items():
        prediction = float(model["intercept"])
        for term in model.get("terms", []):
            prediction += float(term["coefficient"]) * _evaluate_feature_term(
                str(term["feature"]), features
            )
        alias = discovery.target_to_metric_alias(target)
        values[alias] = float(prediction)

    for required in ("f_re", "f_h"):
        if required not in values:
            raise ValueError(f"Metric bundle does not define required metric '{required}'.")
    return values, features


def initialize_metric_tables(
    bundle: MetricBundle,
    simulation_metrics_path: Path,
    experiment_metrics_path: Path,
) -> None:
    simulation_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    experiment_metrics_path.parent.mkdir(parents=True, exist_ok=True)

    if not simulation_metrics_path.exists():
        source = pd.read_csv(bundle.candidate_metrics_path)
        required = ["Re", "h", "f_re", "f_h"]
        missing = [column for column in required if column not in source.columns]
        if missing:
            raise ValueError(
                f"Candidate metric file is missing columns {missing}: "
                f"{bundle.candidate_metrics_path}"
            )
        initial = source[required].copy()
        initial["source"] = "discovery"
        initial["run_id"] = ""
        initial["file"] = source["file"] if "file" in source.columns else ""
        initial.to_csv(simulation_metrics_path, index=False)

    if not experiment_metrics_path.exists():
        pd.DataFrame(
            columns=["Re", "h", "f_re", "f_h", "source", "run_id", "file"]
        ).to_csv(experiment_metrics_path, index=False)


def append_metric_row(
    metric_path: Path,
    re_value: float,
    h_value: float,
    metrics: Mapping[str, float],
    source: str,
    run_id: str,
    pressure_file: Path,
) -> None:
    row = pd.DataFrame(
        [
            {
                "Re": float(re_value),
                "h": float(h_value),
                "f_re": float(metrics["f_re"]),
                "f_h": float(metrics["f_h"]),
                "source": source,
                "run_id": run_id,
                "file": str(pressure_file),
            }
        ]
    )
    if metric_path.exists() and metric_path.stat().st_size:
        existing = pd.read_csv(metric_path)
        result = pd.concat([existing, row], ignore_index=True)
    else:
        result = row
    result.to_csv(metric_path, index=False)


def _finite_metric_rows(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["Re", "h", "f_re", "f_h"]
    if any(column not in frame.columns for column in required):
        return pd.DataFrame(columns=required)
    cleaned = frame.copy()
    for column in required:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    return cleaned.replace([np.inf, -np.inf], np.nan).dropna(subset=required)


def _fit_gp(
    x_values: np.ndarray,
    y_values: np.ndarray,
    x_center: np.ndarray,
    x_scale: np.ndarray,
    y_center: np.ndarray,
    y_scale: np.ndarray,
    random_state: int,
) -> list[GaussianProcessRegressor]:
    x_norm = (x_values - x_center) / x_scale
    models: list[GaussianProcessRegressor] = []
    for target_index in range(y_values.shape[1]):
        y_norm = (y_values[:, target_index] - y_center[target_index]) / y_scale[target_index]
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
            length_scale=np.ones(x_values.shape[1]),
            length_scale_bounds=(1e-2, 1e2),
        )
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-8,
            normalize_y=False,
            n_restarts_optimizer=2,
            random_state=random_state,
        )
        model.fit(x_norm, y_norm)
        models.append(model)
    return models


def _predict_gp(
    models: Iterable[GaussianProcessRegressor],
    candidates: np.ndarray,
    x_center: np.ndarray,
    x_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x_norm = (candidates - x_center) / x_scale
    means = []
    stds = []
    for model in models:
        mean, std = model.predict(x_norm, return_std=True)
        means.append(mean)
        stds.append(std)
    return np.column_stack(means), np.column_stack(stds)


def select_next_parameters(
    simulation_metrics_path: Path,
    experiment_metrics_path: Path,
    bundle: MetricBundle,
    search_config: Mapping[str, Any],
) -> SelectionResult:
    simulation = _finite_metric_rows(pd.read_csv(simulation_metrics_path))
    if simulation.empty:
        raise ValueError("No simulation metrics are available for parameter selection.")

    experiment = (
        _finite_metric_rows(pd.read_csv(experiment_metrics_path))
        if experiment_metrics_path.exists()
        else pd.DataFrame(columns=["Re", "h", "f_re", "f_h"])
    )
    if "source" in experiment.columns:
        # Do not mix legacy raw-pressure metrics with the paired Z-reference
        # subtraction introduced in ASMR 2.0.
        experiment = experiment[
            experiment["source"].astype(str) == "experiment_subtracted"
        ].copy()

    configured_re = (
        float(search_config.get("re_min", bundle.domain["Re"][0])),
        float(search_config.get("re_max", bundle.domain["Re"][1])),
    )
    configured_h = (
        float(search_config.get("h_min", bundle.domain["h"][0])),
        float(search_config.get("h_max", bundle.domain["h"][1])),
    )
    re_limits = (
        max(configured_re[0], bundle.domain["Re"][0]),
        min(configured_re[1], bundle.domain["Re"][1]),
    )
    h_limits = (
        max(configured_h[0], bundle.domain["h"][0]),
        min(configured_h[1], bundle.domain["h"][1]),
    )
    if re_limits[0] > re_limits[1] or h_limits[0] > h_limits[1]:
        raise ValueError("Configured search bounds do not overlap the metric training domain.")

    re_grid = np.linspace(re_limits[0], re_limits[1], int(search_config.get("re_points", 40)))
    h_grid = np.linspace(h_limits[0], h_limits[1], int(search_config.get("h_points", 100)))
    mesh_re, mesh_h = np.meshgrid(re_grid, h_grid)
    candidates = np.column_stack([mesh_re.ravel(), mesh_h.ravel()])

    duplicate_re_tol = float(search_config.get("duplicate_re_tolerance", 1e-6))
    duplicate_h_tol = float(search_config.get("duplicate_h_tolerance", 1e-6))
    sampled = pd.concat(
        [simulation[["Re", "h"]], experiment[["Re", "h"]]],
        ignore_index=True,
    ).drop_duplicates()
    available = np.ones(len(candidates), dtype=bool)
    for re_value, h_value in sampled.to_numpy(dtype=float):
        duplicate = (
            np.abs(candidates[:, 0] - re_value) <= duplicate_re_tol
        ) & (
            np.abs(candidates[:, 1] - h_value) <= duplicate_h_tol
        )
        available &= ~duplicate
    candidates = candidates[available]
    if candidates.size == 0:
        raise RuntimeError("No unsampled candidate remains in the configured search grid.")

    sim_x = simulation[["Re", "h"]].to_numpy(dtype=float)
    sim_y = simulation[["f_re", "f_h"]].to_numpy(dtype=float)
    x_center = np.array(
        [(re_limits[0] + re_limits[1]) / 2.0, (h_limits[0] + h_limits[1]) / 2.0]
    )
    x_scale = np.array(
        [max((re_limits[1] - re_limits[0]) / 2.0, 1.0), max((h_limits[1] - h_limits[0]) / 2.0, 1.0)]
    )

    combined_y = sim_y
    if not experiment.empty:
        combined_y = np.vstack(
            [combined_y, experiment[["f_re", "f_h"]].to_numpy(dtype=float)]
        )
    y_center = np.mean(combined_y, axis=0)
    y_scale = np.std(combined_y, axis=0)
    y_scale[y_scale <= discovery.EPS] = 1.0

    random_state = int(search_config.get("random_state", 42))
    sim_models = _fit_gp(
        sim_x, sim_y, x_center, x_scale, y_center, y_scale, random_state
    )
    sim_mean, sim_std = _predict_gp(sim_models, candidates, x_center, x_scale)

    if not experiment.empty:
        exp_x = experiment[["Re", "h"]].to_numpy(dtype=float)
        exp_y = experiment[["f_re", "f_h"]].to_numpy(dtype=float)
        exp_models = _fit_gp(
            exp_x, exp_y, x_center, x_scale, y_center, y_scale, random_state + 1
        )
        exp_mean, _ = _predict_gp(exp_models, candidates, x_center, x_scale)
        scores = np.linalg.norm(sim_mean - exp_mean, axis=1)
        strategy = "simulation_experiment_disagreement"
    else:
        scores = np.linalg.norm(sim_std, axis=1)
        strategy = "simulation_uncertainty"

    best_index = int(np.nanargmax(scores))
    return SelectionResult(
        re_value=float(candidates[best_index, 0]),
        h_value=float(candidates[best_index, 1]),
        score=float(scores[best_index]),
        strategy=strategy,
        candidates_considered=int(len(candidates)),
    )
