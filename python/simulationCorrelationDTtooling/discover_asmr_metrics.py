"""
Discover ASMR-style scalar metrics from brute-force pressure simulations.

The current ASMR.py workflow uses hand-written functions such as compute_Re and
compute_h to reduce a pressure trace into two scalar metrics. This script is a
standalone experiment for finding such metrics automatically from a set of
simulations with known parameters.

It intentionally does not import or modify ASMR.py. Instead it:

1. Finds Lilypad result files and reads Re/h from the path, or from a metadata
   CSV.
2. Extracts wall-pressure rows in the same spirit as chinaPressureExtraction.py.
   In pressureMotion.txt the cylinder rows are used only to recover time; the
   pressure descriptors are computed from wall rows.
3. Extracts many scalar descriptors from each wall-pressure time series.
4. Ranks descriptors by how strongly they relate to target parameters.
5. Fits sparse SINDy-like static expressions with sequential thresholded
   least squares.

This is "SINDy-like" rather than standard SINDy: the target here is a static
parameter/measurement relationship, not a time derivative dx/dt.

Example:

    python python/simulationCorrelationDTtooling/discover_asmr_metrics.py \
        --root E:/simuChina \
        --pattern "**/pressureMotion.txt" \
        --output-dir E:/simuChina/metric_discovery

If folders do not encode the parameters, provide a CSV containing a file/path
column plus Re and h columns:

    python python/simulationCorrelationDTtooling/discover_asmr_metrics.py \
        --root E:/simuChina \
        --metadata-csv E:/simuChina/brut_force_index.csv
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


EPS = 1e-12
DEFAULT_PATH_REGEX = (
    r"Re(?P<Re>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    r"[_-]?h(?P<h>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)

# Constants copied from ASMR.py's Lilypad pressure parsing. They are only used
# to convert Lilypad nondimensional time to the same physical-time convention.
N_GRID = 2**7
GRID_H = 1 / N_GRID
NU = 1 / 1_000_000
LENGTH_SCALE = N_GRID / 20


@dataclass(frozen=True)
class Case:
    pressure_file: Path
    parameters: dict[str, float]


@dataclass
class SparseModelResult:
    target: str
    expression: str
    intercept: float
    terms: list[dict[str, float | str]]
    in_sample_r2: float
    in_sample_mae: float
    cv_r2_mean: float | None
    cv_r2_std: float | None
    selected_features: list[str]
    predictions: list[float]


@dataclass
class ClassicRelationshipResult:
    parameter: str
    feature: str
    target_transform: str
    feature_transform: str
    degree: int
    expression: str
    in_sample_r2: float
    in_sample_mae: float
    cv_r2_mean: float | None
    cv_r2_std: float | None
    coefficients: list[float]


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def safe_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


def safe_log10(values: np.ndarray | float) -> np.ndarray | float:
    return np.log10(np.maximum(np.abs(values), EPS))


def signed_log10(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.log10(1.0 + np.abs(values))


def transformed_values(values: np.ndarray, transform: str) -> np.ndarray | None:
    values = np.asarray(values, dtype=float)
    if transform == "identity":
        return values
    if transform == "log10":
        if np.any(values <= 0):
            return None
        return np.log10(values)
    if transform == "log10_abs":
        return safe_log10(values)
    if transform == "signed_log10":
        return signed_log10(values)
    if transform == "sqrt":
        if np.any(values < 0):
            return None
        return np.sqrt(values)
    if transform == "sqrt_abs":
        return np.sqrt(np.abs(values))
    if transform == "inverse":
        if np.any(np.abs(values) <= EPS):
            return None
        return 1.0 / values
    if transform == "square":
        return values**2
    if transform == "cuberoot":
        return np.cbrt(values)
    raise ValueError(f"Unknown transform: {transform}")


def transform_label(name: str, transform: str) -> str:
    if transform == "identity":
        return name
    if transform == "log10":
        return f"log10({name})"
    if transform == "log10_abs":
        return f"log10_abs({name})"
    if transform == "signed_log10":
        return f"signed_log10({name})"
    if transform == "sqrt":
        return f"sqrt({name})"
    if transform == "sqrt_abs":
        return f"sqrt_abs({name})"
    if transform == "inverse":
        return f"1/({name})"
    if transform == "square":
        return f"({name})^2"
    if transform == "cuberoot":
        return f"cuberoot({name})"
    return f"{transform}({name})"


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if np.sum(valid) < 3:
        return np.nan
    x = x[valid]
    y = y[valid]
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    denominator = np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))
    if denominator <= EPS:
        return np.nan
    return float(np.sum(x_centered * y_centered) / denominator)


def rankdata_average(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        # Average rank for ties. Ranks are 1-based to match scipy.stats.rankdata.
        average_rank = 0.5 * (start + 1 + end)
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if np.sum(valid) < 3:
        return np.nan
    return pearson_corr(rankdata_average(x[valid]), rankdata_average(y[valid]))


def discretized_mutual_information(x: np.ndarray, y: np.ndarray, max_bins: int = 10) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if np.sum(valid) < 4:
        return np.nan
    x = x[valid]
    y = y[valid]
    if np.std(x) <= EPS or np.std(y) <= EPS:
        return np.nan
    bins = max(2, min(max_bins, int(np.sqrt(x.size))))
    hist, _, _ = np.histogram2d(x, y, bins=bins)
    total = np.sum(hist)
    if total <= 0:
        return np.nan
    pxy = hist / total
    px = np.sum(pxy, axis=1, keepdims=True)
    py = np.sum(pxy, axis=0, keepdims=True)
    expected = px @ py
    mask = (pxy > 0) & (expected > 0)
    return float(np.sum(pxy[mask] * np.log(pxy[mask] / expected[mask])))


def make_cv(n_samples: int, requested_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    # R2 is undefined for folds with one test sample, so keep test folds >= 2.
    n_splits = min(requested_folds, n_samples // 2)
    if n_splits < 2:
        return []
    indices = np.arange(n_samples)
    rng = np.random.default_rng(0)
    rng.shuffle(indices)
    folds = np.array_split(indices, n_splits)
    splits = []
    for test_index in folds:
        if test_index.size < 2:
            continue
        train_index = np.setdiff1d(indices, test_index, assume_unique=False)
        splits.append((train_index, test_index))
    return splits


def standardize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    mean = np.mean(X, axis=0)
    scale = np.std(X, axis=0)
    scale[scale <= EPS] = 1.0
    return (X - mean) / scale, mean, scale


def standardize_apply(X: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (np.asarray(X, dtype=float) - mean) / scale


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    if denominator <= EPS:
        return np.nan
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denominator)


def mean_absolute_error_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))))


def fit_linear_model(X: np.ndarray, y: np.ndarray, ridge_alpha: float = 1e-10) -> tuple[float, np.ndarray]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    design = np.column_stack([np.ones(X.shape[0]), X])
    penalty = np.eye(design.shape[1]) * ridge_alpha
    penalty[0, 0] = 0.0
    try:
        coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(coefficients[0]), coefficients[1:].astype(float)


def predict_linear_model(X: np.ndarray, intercept: float, coefficients: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return intercept + X @ coefficients


def fit_stlsq_scaled(
    X_scaled: np.ndarray,
    y: np.ndarray,
    ridge_alpha: float,
    threshold: float,
    max_iter: int = 20,
) -> tuple[float, np.ndarray]:
    """Sequential thresholded least squares on already standardized features."""

    X_scaled = np.asarray(X_scaled, dtype=float)
    y = np.asarray(y, dtype=float)
    n_features = X_scaled.shape[1]
    active = np.ones(n_features, dtype=bool)
    coefficients = np.zeros(n_features, dtype=float)
    intercept = float(np.mean(y))

    for _ in range(max_iter):
        active_indices = np.flatnonzero(active)
        if active_indices.size == 0:
            break

        intercept, active_coefficients = fit_linear_model(
            X_scaled[:, active_indices],
            y,
            ridge_alpha=ridge_alpha,
        )
        coefficients[:] = 0.0
        coefficients[active_indices] = active_coefficients

        if threshold <= 0:
            break

        small = np.abs(active_coefficients) < threshold
        if not np.any(small):
            break

        if np.all(small):
            keep_local = int(np.argmax(np.abs(active_coefficients)))
            small[keep_local] = False

        next_active = active.copy()
        next_active[active_indices[small]] = False
        if np.array_equal(next_active, active):
            break
        active = next_active

    return intercept, coefficients


def cv_stlsq_score(
    X: np.ndarray,
    y: np.ndarray,
    cv: list[tuple[np.ndarray, np.ndarray]],
    ridge_alpha: float,
    threshold: float,
) -> tuple[float, float]:
    scores = []
    for train_index, test_index in cv:
        X_train, mean, scale = standardize_fit(X[train_index])
        X_test = standardize_apply(X[test_index], mean, scale)
        intercept, coefficients = fit_stlsq_scaled(
            X_train,
            y[train_index],
            ridge_alpha=ridge_alpha,
            threshold=threshold,
        )
        prediction = predict_linear_model(X_test, intercept, coefficients)
        score = r2_score_np(y[test_index], prediction)
        if np.isfinite(score):
            scores.append(score)
    if not scores:
        return np.nan, np.nan
    return float(np.mean(scores)), float(np.std(scores))


def choose_stlsq_hyperparameters(
    X: np.ndarray,
    y: np.ndarray,
    cv: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[float, float, float | None, float | None]:
    alpha_grid = [0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]
    threshold_grid = [0.0, 1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]

    if not cv:
        return 1e-8, 1e-4, None, None

    best_alpha = alpha_grid[0]
    best_threshold = threshold_grid[0]
    best_mean = -np.inf
    best_std = np.nan
    for alpha in alpha_grid:
        for threshold in threshold_grid:
            score_mean, score_std = cv_stlsq_score(X, y, cv, alpha, threshold)
            if not np.isfinite(score_mean):
                continue
            if score_mean > best_mean:
                best_mean = score_mean
                best_std = score_std
                best_alpha = alpha
                best_threshold = threshold

    if best_mean == -np.inf:
        return 1e-8, 1e-4, None, None
    return best_alpha, best_threshold, float(best_mean), float(best_std)


def discover_cases(
    root: Path,
    pattern: str,
    path_regex: str,
    parameter_columns: list[str],
    metadata_csv: Path | None = None,
) -> list[Case]:
    if metadata_csv is not None:
        return discover_cases_from_metadata(root, metadata_csv, parameter_columns)

    regex = re.compile(path_regex, re.IGNORECASE)
    cases: list[Case] = []
    for pressure_file in sorted(root.glob(pattern)):
        if not pressure_file.is_file():
            continue
        match = regex.search(str(pressure_file))
        if match is None:
            continue
        parameters = {}
        for column in parameter_columns:
            value = safe_float(match.groupdict().get(column))
            if value is None:
                parameters = {}
                break
            parameters[column] = value
        if parameters:
            cases.append(Case(pressure_file=pressure_file, parameters=parameters))
    return cases


def discover_cases_from_metadata(
    root: Path,
    metadata_csv: Path,
    parameter_columns: list[str],
) -> list[Case]:
    df = pd.read_csv(metadata_csv)
    file_column_candidates = ["file", "path", "pressure_file", "pressureMotion"]
    file_column = next((col for col in file_column_candidates if col in df.columns), None)
    if file_column is None:
        raise ValueError(
            "Metadata CSV must contain one file path column named one of: "
            + ", ".join(file_column_candidates)
        )

    missing = [col for col in parameter_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Metadata CSV is missing parameter columns: {missing}")

    cases: list[Case] = []
    for _, row in df.iterrows():
        raw_path = Path(str(row[file_column]))
        pressure_file = raw_path if raw_path.is_absolute() else root / raw_path
        parameters = {}
        for column in parameter_columns:
            value = safe_float(row[column])
            if value is None:
                parameters = {}
                break
            parameters[column] = value
        if parameters and pressure_file.exists():
            cases.append(Case(pressure_file=pressure_file, parameters=parameters))
    return cases


def parse_numeric_line(line: str) -> list[float]:
    cleaned = line.replace(";", " ").strip()
    if not cleaned:
        return []
    values: list[float] = []
    for token in cleaned.split():
        value = safe_float(token)
        if value is not None:
            values.append(value)
    return values


def read_lilypad_wall_pressure(file_path: Path, re_value: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Read wall-pressure rows from Lilypad pressureMotion.txt.

    chinaPressureExtraction.py parses pressureMotion.txt as alternating rows:
    one cylinder/body row carrying the nondimensional time, then one wall row.
    This reader discards the cylinder pressure values and returns only
    (wall_pressure, time).
    """

    with file_path.open("r", encoding="utf-8", errors="ignore") as pressure_file:
        lines = [
            line
            for line in pressure_file
            if line.strip() and not line.lstrip().startswith("%")
        ]

    times: list[float] = []
    wall_rows: list[list[float]] = []
    for index, line in enumerate(lines):
        values = parse_numeric_line(line)
        if not values:
            continue
        if index % 2 == 0:
            raw_time = values[0]
            if re_value is not None and re_value > 0:
                local_nu = LENGTH_SCALE / re_value
                raw_time = raw_time * GRID_H * GRID_H * local_nu / NU
            times.append(raw_time)
        else:
            wall_rows.append(values)

    if not wall_rows or not times:
        raise ValueError(f"No pressure rows were parsed from {file_path}")

    n_rows = min(len(times), len(wall_rows))
    width = min(len(row) for row in wall_rows[:n_rows])
    if width == 0:
        raise ValueError(f"Pressure rows are empty in {file_path}")

    pressure = np.asarray([row[:width] for row in wall_rows[:n_rows]], dtype=float)
    time = np.asarray(times[:n_rows], dtype=float)
    return pressure, time


read_lilypad_pressure = read_lilypad_wall_pressure


def read_csv_pressure(file_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a generic CSV pressure file.

    If a time-like column exists, it is used as time. Otherwise the row index is
    used. All remaining numeric columns are treated as pressure sensors.
    """

    df = pd.read_csv(file_path)
    numeric = df.select_dtypes(include=[np.number]).copy()
    if numeric.empty:
        raise ValueError(f"No numeric columns found in {file_path}")

    time_column = None
    for candidate in ["time", "Time", "t", "T"]:
        if candidate in numeric.columns:
            time_column = candidate
            break

    if time_column is not None:
        time = numeric.pop(time_column).to_numpy(dtype=float)
    else:
        time = np.arange(len(numeric), dtype=float)

    pressure = numeric.to_numpy(dtype=float)
    return pressure, time


def add_stats(features: dict[str, float], prefix: str, values: Iterable[float]) -> None:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return

    features[f"{prefix}_mean"] = float(np.mean(arr))
    features[f"{prefix}_median"] = float(np.median(arr))
    features[f"{prefix}_std"] = float(np.std(arr))
    features[f"{prefix}_min"] = float(np.min(arr))
    features[f"{prefix}_max"] = float(np.max(arr))
    features[f"{prefix}_q25"] = float(np.quantile(arr, 0.25))
    features[f"{prefix}_q75"] = float(np.quantile(arr, 0.75))
    features[f"{prefix}_iqr"] = float(np.quantile(arr, 0.75) - np.quantile(arr, 0.25))


def add_time_series_stats(
    features: dict[str, float],
    prefix: str,
    values: np.ndarray,
    time: np.ndarray,
) -> None:
    arr = np.asarray(values, dtype=float)
    valid = np.isfinite(arr)
    if not np.any(valid):
        return

    arr_valid = arr[valid]
    time_valid = time[valid]
    features[f"{prefix}_mean"] = float(np.mean(arr_valid))
    features[f"{prefix}_std"] = float(np.std(arr_valid))
    features[f"{prefix}_rms"] = float(np.sqrt(np.mean(arr_valid**2)))
    features[f"{prefix}_absmax"] = float(np.max(np.abs(arr_valid)))
    features[f"{prefix}_range"] = float(np.max(arr_valid) - np.min(arr_valid))
    features[f"{prefix}_t_min"] = float(time_valid[np.argmin(arr_valid)])
    features[f"{prefix}_t_max"] = float(time_valid[np.argmax(arr_valid)])
    features[f"{prefix}_log10_range"] = float(safe_log10(features[f"{prefix}_range"]))


def finite_columns(pressure: np.ndarray) -> np.ndarray:
    has_data = np.isfinite(pressure).any(axis=0)
    if not np.any(has_data):
        raise ValueError("No pressure sensor column contains finite values.")
    return pressure[:, has_data]


def fft_high_pass_by_frequency(
    pressure: np.ndarray,
    time: np.ndarray,
    cutoff_freq: float,
) -> np.ndarray:
    pressure = np.asarray(pressure, dtype=float)
    time = np.asarray(time, dtype=float)
    dt = np.diff(time)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if pressure.shape[0] < 4 or dt.size == 0:
        return pressure - np.nanmean(pressure, axis=0, keepdims=True)

    mean_dt = float(np.mean(dt))
    if mean_dt <= 0 or cutoff_freq <= 0:
        return pressure - np.nanmean(pressure, axis=0, keepdims=True)

    clean = pressure - np.nanmean(pressure, axis=0, keepdims=True)
    clean = np.nan_to_num(clean, nan=0.0)
    freqs = np.fft.rfftfreq(clean.shape[0], d=mean_dt)
    spectrum = np.fft.rfft(clean, axis=0)
    spectrum[freqs < cutoff_freq, :] = 0.0
    return np.fft.irfft(spectrum, n=clean.shape[0], axis=0)


def add_china_wall_reference_features(
    features: dict[str, float],
    pressure: np.ndarray,
    time: np.ndarray,
    re_value: float | None,
    warmup_samples: int,
) -> None:
    """Add wall-pressure descriptors mirroring chinaPressureExtraction.py.

    Manual Re trend:
        log10(argmin(Cp_wall)) vs log10(Re), at fixed h.

    Manual h trend:
        log10(1 / (max(Cp_wall_highpass)-min(Cp_wall_highpass))^4)
        vs log10(h), at fixed Re.
    """

    if pressure.shape[0] <= warmup_samples + 2:
        return

    wall_after_warmup = pressure[warmup_samples:]
    time_after_warmup = time[warmup_samples:]
    safe_for_min = np.where(np.isfinite(wall_after_warmup), wall_after_warmup, np.inf)
    idx_min_rel = np.argmin(safe_for_min, axis=0)

    # This reproduces the indexing convention in chinaPressureExtraction.py:
    # time[np.argmin(data[pos][4:])]. The corrected variant below keeps the
    # warmup offset in the time index.
    exact_time = time[np.clip(idx_min_rel, 0, len(time) - 1)]
    corrected_time = time_after_warmup[idx_min_rel]
    add_stats(features, "china_wall_log10_t_min_exact", safe_log10(exact_time))
    add_stats(features, "china_wall_log10_t_min_corrected", safe_log10(corrected_time))
    features["china_wall_re_metric_exact"] = float(np.mean(safe_log10(exact_time)))
    features["china_wall_re_metric_corrected"] = float(np.mean(safe_log10(corrected_time)))

    raw_ranges = np.nanmax(wall_after_warmup, axis=0) - np.nanmin(wall_after_warmup, axis=0)
    add_stats(features, "china_wall_raw_log10_inv_range4", -4.0 * safe_log10(raw_ranges))
    features["china_wall_h_metric_raw"] = float(np.mean(-4.0 * safe_log10(raw_ranges)))

    if re_value is None or re_value <= 0:
        return

    cutoff_freq = 0.004 * re_value / 1000.0
    filtered = fft_high_pass_by_frequency(wall_after_warmup, time_after_warmup, cutoff_freq)
    filtered_ranges = np.nanmax(filtered, axis=0) - np.nanmin(filtered, axis=0)
    add_stats(features, "china_wall_highpass_range", filtered_ranges)
    add_stats(features, "china_wall_highpass_log10_inv_range4", -4.0 * safe_log10(filtered_ranges))
    features["china_wall_h_metric_highpass"] = float(
        np.mean(-4.0 * safe_log10(filtered_ranges))
    )


def extract_pressure_features(
    pressure: np.ndarray,
    time: np.ndarray,
    re_value: float | None = None,
    warmup_samples: int = 4,
    max_sensors: int | None = None,
) -> dict[str, float]:
    pressure = np.asarray(pressure, dtype=float)
    time = np.asarray(time, dtype=float)

    n_rows = min(pressure.shape[0], time.size)
    pressure = pressure[:n_rows]
    time = time[:n_rows]
    if max_sensors is not None:
        pressure = pressure[:, :max_sensors]

    pressure = finite_columns(pressure)
    order = np.argsort(time)
    time = time[order]
    pressure = pressure[order]

    if n_rows <= warmup_samples + 2:
        raise ValueError("Not enough samples after warmup removal.")

    features: dict[str, float] = {
        "wall_n_samples_raw": float(pressure.shape[0]),
        "wall_n_sensors_raw": float(pressure.shape[1]),
    }
    add_china_wall_reference_features(features, pressure, time, re_value, warmup_samples)

    pressure = pressure[warmup_samples:]
    time = time[warmup_samples:]
    pressure = finite_columns(pressure)

    features.update({
        "n_samples": float(pressure.shape[0]),
        "n_sensors": float(pressure.shape[1]),
        "time_start": float(time[0]),
        "time_end": float(time[-1]),
        "time_span": float(time[-1] - time[0]),
    })

    dt = np.diff(time)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size:
        features["dt_mean"] = float(np.mean(dt))
        features["dt_std"] = float(np.std(dt))
        features["sampling_frequency"] = float(1.0 / np.mean(dt))

    safe_pressure_for_min = np.where(np.isfinite(pressure), pressure, np.inf)
    safe_pressure_for_max = np.where(np.isfinite(pressure), pressure, -np.inf)
    idx_min = np.argmin(safe_pressure_for_min, axis=0)
    idx_max = np.argmax(safe_pressure_for_max, axis=0)

    mins = np.nanmin(pressure, axis=0)
    maxs = np.nanmax(pressure, axis=0)
    ranges = maxs - mins
    absmax = np.maximum(np.abs(mins), np.abs(maxs))
    stds = np.nanstd(pressure, axis=0)
    rms = np.sqrt(np.nanmean(pressure**2, axis=0))
    mean_abs = np.nanmean(np.abs(pressure), axis=0)
    t_min = time[idx_min]
    t_max = time[idx_max]

    add_stats(features, "pressure_min", mins)
    add_stats(features, "pressure_max", maxs)
    add_stats(features, "pressure_range", ranges)
    add_stats(features, "pressure_absmax", absmax)
    add_stats(features, "pressure_std", stds)
    add_stats(features, "pressure_rms", rms)
    add_stats(features, "pressure_mean_abs", mean_abs)
    add_stats(features, "t_min", t_min)
    add_stats(features, "t_max", t_max)
    add_stats(features, "log10_t_min", safe_log10(t_min))
    add_stats(features, "log10_t_max", safe_log10(t_max))
    add_stats(features, "log10_pressure_range", safe_log10(ranges))
    add_stats(features, "log10_pressure_absmax", safe_log10(absmax))
    add_stats(features, "log10_inv_pressure_range4", -4.0 * safe_log10(ranges))

    features["wall_re_metric_corrected"] = float(np.mean(safe_log10(t_min)))
    features["wall_h_metric_raw_range"] = float(np.mean(-4.0 * safe_log10(ranges)))
    features["t_min_span"] = float(np.max(t_min) - np.min(t_min))
    features["t_max_span"] = float(np.max(t_max) - np.min(t_max))

    if pressure.shape[1] > 1:
        sensor_index = np.arange(pressure.shape[1], dtype=float)
        intercept_min, slope_min_arr = fit_linear_model(sensor_index, t_min)
        intercept_max, slope_max_arr = fit_linear_model(sensor_index, t_max)
        slope_min = float(slope_min_arr[0])
        slope_max = float(slope_max_arr[0])
        min_prediction = predict_linear_model(sensor_index, intercept_min, slope_min_arr)
        max_prediction = predict_linear_model(sensor_index, intercept_max, slope_max_arr)
        features["t_min_sensor_slope"] = float(slope_min)
        features["t_min_sensor_intercept"] = float(intercept_min)
        features["t_min_sensor_r2"] = float(r2_score_np(t_min, min_prediction))
        features["t_max_sensor_slope"] = float(slope_max)
        features["t_max_sensor_intercept"] = float(intercept_max)
        features["t_max_sensor_r2"] = float(r2_score_np(t_max, max_prediction))

    spatial_mean = np.nanmean(pressure, axis=1)
    spatial_std = np.nanstd(pressure, axis=1)
    spatial_range = np.nanmax(pressure, axis=1) - np.nanmin(pressure, axis=1)
    add_time_series_stats(features, "spatial_mean_signal", spatial_mean, time)
    add_time_series_stats(features, "spatial_std_signal", spatial_std, time)
    add_time_series_stats(features, "spatial_range_signal", spatial_range, time)

    if pressure.shape[0] > 2 and dt.size:
        mean_dt = float(np.mean(dt))
        if mean_dt > 0:
            derivative = np.diff(pressure, axis=0) / mean_dt
            add_stats(features, "dpdt_std", np.nanstd(derivative, axis=0))
            add_stats(features, "dpdt_absmax", np.nanmax(np.abs(derivative), axis=0))
            add_stats(features, "dpdt_mean_abs", np.nanmean(np.abs(derivative), axis=0))

    if pressure.shape[1] > 1:
        gradient = np.diff(pressure, axis=1)
        add_stats(features, "spatial_gradient", gradient.ravel())
        add_stats(features, "spatial_gradient_abs", np.abs(gradient).ravel())
        add_time_series_stats(
            features,
            "spatial_gradient_rms_signal",
            np.sqrt(np.nanmean(gradient**2, axis=1)),
            time,
        )

        correlations = []
        for sensor_id in range(pressure.shape[1] - 1):
            left = pressure[:, sensor_id]
            right = pressure[:, sensor_id + 1]
            valid = np.isfinite(left) & np.isfinite(right)
            if np.sum(valid) < 4:
                continue
            if np.std(left[valid]) < EPS or np.std(right[valid]) < EPS:
                continue
            correlations.append(float(np.corrcoef(left[valid], right[valid])[0, 1]))
        add_stats(features, "adjacent_sensor_corr", correlations)

    add_svd_features(features, pressure)
    add_frequency_features(features, pressure, time)
    add_filter_features(features, pressure)
    return features


def add_svd_features(features: dict[str, float], pressure: np.ndarray) -> None:
    centered = pressure - np.nanmean(pressure, axis=0, keepdims=True)
    if np.isnan(centered).any():
        column_means = np.nanmean(centered, axis=0)
        inds = np.where(np.isnan(centered))
        centered[inds] = np.take(column_means, inds[1])

    if centered.shape[0] < 2 or centered.shape[1] < 2:
        return

    try:
        singular_values = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    except np.linalg.LinAlgError:
        return

    energy = singular_values**2
    total = float(np.sum(energy))
    if total <= EPS:
        return

    fractions = energy / total
    for index in range(min(5, fractions.size)):
        features[f"svd_energy_fraction_{index + 1}"] = float(fractions[index])

    entropy = -np.sum(fractions * np.log(np.maximum(fractions, EPS)))
    features["svd_entropy"] = float(entropy)
    features["svd_effective_rank"] = float(np.exp(entropy))
    features["svd_first_to_second_ratio"] = float(
        singular_values[0] / max(singular_values[1], EPS)
    )


def add_frequency_features(features: dict[str, float], pressure: np.ndarray, time: np.ndarray) -> None:
    if pressure.shape[0] < 8:
        return

    dt = np.diff(time)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return
    mean_dt = float(np.mean(dt))
    if mean_dt <= 0:
        return

    demeaned = pressure - np.nanmean(pressure, axis=0, keepdims=True)
    if np.isnan(demeaned).any():
        demeaned = np.nan_to_num(demeaned, nan=0.0)

    freqs = np.fft.rfftfreq(demeaned.shape[0], d=mean_dt)
    if freqs.size < 3:
        return

    spectrum = np.fft.rfft(demeaned, axis=0)
    power = np.abs(spectrum) ** 2
    nonzero_freqs = freqs[1:]
    nonzero_power = power[1:]
    power_sum = np.sum(nonzero_power, axis=0)
    valid = power_sum > EPS
    if not np.any(valid):
        return

    dominant_idx = np.argmax(nonzero_power[:, valid], axis=0)
    dominant_freq = nonzero_freqs[dominant_idx]
    centroid = np.sum(nonzero_freqs[:, None] * nonzero_power[:, valid], axis=0) / power_sum[valid]
    add_stats(features, "dominant_frequency", dominant_freq)
    add_stats(features, "spectral_centroid", centroid)

    max_freq = float(nonzero_freqs[-1])
    low_mask = nonzero_freqs <= max_freq / 3.0
    high_mask = nonzero_freqs >= 2.0 * max_freq / 3.0
    low_fraction = np.sum(nonzero_power[low_mask][:, valid], axis=0) / power_sum[valid]
    high_fraction = np.sum(nonzero_power[high_mask][:, valid], axis=0) / power_sum[valid]
    add_stats(features, "spectral_low_energy_fraction", low_fraction)
    add_stats(features, "spectral_high_energy_fraction", high_fraction)


def add_filter_features(features: dict[str, float], pressure: np.ndarray) -> None:
    if pressure.shape[0] < 24:
        return

    pressure_clean = pressure - np.nanmean(pressure, axis=0, keepdims=True)
    pressure_clean = np.nan_to_num(pressure_clean, nan=0.0)

    for cutoff_fraction in [0.01, 0.03, 0.10, 0.20]:
        if not 0 < cutoff_fraction < 1:
            continue
        spectrum = np.fft.rfft(pressure_clean, axis=0)
        cutoff_index = int(np.ceil(cutoff_fraction * (spectrum.shape[0] - 1)))
        cutoff_index = max(1, min(cutoff_index, spectrum.shape[0] - 1))
        spectrum[:cutoff_index, :] = 0.0
        filtered = np.fft.irfft(spectrum, n=pressure_clean.shape[0], axis=0)
        ranges = np.nanmax(filtered, axis=0) - np.nanmin(filtered, axis=0)
        label = f"highpass_{int(cutoff_fraction * 1000):03d}"
        add_stats(features, f"{label}_range", ranges)
        add_stats(features, f"{label}_log10_inv_range4", -4.0 * safe_log10(ranges))


def make_target_values(
    parameter_df: pd.DataFrame,
    target_specs: list[str],
) -> dict[str, np.ndarray]:
    targets: dict[str, np.ndarray] = {}
    for spec in target_specs:
        if spec in parameter_df.columns:
            targets[spec] = parameter_df[spec].to_numpy(dtype=float)
            continue

        if spec.startswith("log10_"):
            column = spec.removeprefix("log10_")
            if column not in parameter_df.columns:
                raise ValueError(f"Unknown target '{spec}': no parameter column '{column}'")
            values = parameter_df[column].to_numpy(dtype=float)
            if np.any(values <= 0):
                raise ValueError(f"Target '{spec}' requires positive '{column}' values")
            targets[spec] = np.log10(values)
            continue

        raise ValueError(f"Unknown target '{spec}'")
    return targets


def numeric_feature_columns(df: pd.DataFrame, excluded_columns: set[str]) -> list[str]:
    columns = []
    for column in df.columns:
        if column in excluded_columns:
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            values = df[column].to_numpy(dtype=float)
            if np.isfinite(values).sum() >= 3 and np.nanstd(values) > EPS:
                columns.append(column)
    return columns


def feature_columns_for_target(feature_columns: list[str], target_name: str) -> list[str]:
    metadata_exact = {"time_start", "time_end", "time_span", "sampling_frequency"}
    metadata_prefixes = ("dt_", "n_samples", "n_sensors", "wall_n_samples", "wall_n_sensors")
    columns = [
        column
        for column in feature_columns
        if column not in metadata_exact
        and not any(column.startswith(prefix) for prefix in metadata_prefixes)
    ]

    base_target = target_name.removeprefix("log10_").lower()
    if base_target != "re":
        return columns

    # The manual high-pass h descriptor uses Re to choose its cutoff
    # (0.004 * Re / 1000). It is useful as a candidate h metric, but it leaks
    # target information for Re discovery.
    blocked_fragments = ["china_wall_highpass", "china_wall_h_metric_highpass"]
    return [
        column
        for column in columns
        if not any(fragment in column for fragment in blocked_fragments)
    ]


def cleaned_matrix(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    cleaned = df[columns].replace([np.inf, -np.inf], np.nan).copy()
    for column in columns:
        median = cleaned[column].median()
        if not np.isfinite(median):
            median = 0.0
        cleaned[column] = cleaned[column].fillna(median)
    return cleaned


def score_feature_table(
    df: pd.DataFrame,
    feature_columns: list[str],
    target: np.ndarray,
    cv_folds: int,
) -> pd.DataFrame:
    X = cleaned_matrix(df, feature_columns)
    y = np.asarray(target, dtype=float)
    rows = []

    cv = make_cv(len(y), cv_folds)
    for column in feature_columns:
        x = X[column].to_numpy(dtype=float)
        if np.std(x) <= EPS or np.std(y) <= EPS:
            continue

        pearson = pearson_corr(x, y)
        spearman = spearman_corr(x, y)
        mutual_information = discretized_mutual_information(x, y)

        single_cv_r2 = np.nan
        if cv:
            scores = []
            x_2d = x.reshape(-1, 1)
            for train_index, test_index in cv:
                X_train, mean, scale = standardize_fit(x_2d[train_index])
                X_test = standardize_apply(x_2d[test_index], mean, scale)
                intercept, coefficients = fit_linear_model(X_train, y[train_index])
                prediction = predict_linear_model(X_test, intercept, coefficients)
                score = r2_score_np(y[test_index], prediction)
                if np.isfinite(score):
                    scores.append(score)
            if scores:
                single_cv_r2 = float(np.mean(scores))

        rows.append(
            {
                "feature": column,
                "pearson": pearson,
                "abs_pearson": abs(pearson) if np.isfinite(pearson) else np.nan,
                "spearman": spearman,
                "abs_spearman": abs(spearman) if np.isfinite(spearman) else np.nan,
                "mutual_information": mutual_information,
                "single_feature_cv_r2": single_cv_r2,
            }
        )

    score_df = pd.DataFrame(rows)
    if score_df.empty:
        return score_df
    score_df["rank_score"] = score_df[
        ["abs_spearman", "abs_pearson", "mutual_information"]
    ].max(axis=1)
    return score_df.sort_values(
        ["rank_score", "single_feature_cv_r2"],
        ascending=False,
        na_position="last",
    )


def transformed_library(
    feature_df: pd.DataFrame,
    feature_columns: list[str],
    target: np.ndarray,
    max_base_features: int,
    degree: int,
    max_interaction_sources: int,
) -> pd.DataFrame:
    """Build a static sparse-regression library from pressure descriptors."""

    base = cleaned_matrix(feature_df, feature_columns)
    base_scores = score_feature_table(feature_df, feature_columns, target, cv_folds=5)
    if not base_scores.empty:
        ordered = base_scores["feature"].tolist()
        base_columns = ordered[:max_base_features]
    else:
        base_columns = feature_columns[:max_base_features]

    library: dict[str, np.ndarray] = {}
    for column in base_columns:
        values = base[column].to_numpy(dtype=float)
        library[column] = values
        library[f"log10_abs({column})"] = safe_log10(values)
        library[f"signed_log10({column})"] = signed_log10(values)
        if np.nanmin(np.abs(values)) > EPS:
            library[f"inv({column})"] = 1.0 / values

    if degree >= 2:
        interaction_sources = list(library.keys())[:max_interaction_sources]
        for i, left_name in enumerate(interaction_sources):
            for right_name in interaction_sources[i:]:
                library[f"{left_name}*{right_name}"] = library[left_name] * library[right_name]

    result = pd.DataFrame(library, index=feature_df.index)
    keep = []
    for column in result.columns:
        values = result[column].to_numpy(dtype=float)
        finite = np.isfinite(values)
        if finite.sum() >= 3 and np.nanstd(values[finite]) > EPS:
            keep.append(column)
    return cleaned_matrix(result, keep)


def fit_sparse_model(
    library_df: pd.DataFrame,
    target: np.ndarray,
    target_name: str,
    cv_folds: int,
    max_library_features: int,
    max_terms: int,
) -> SparseModelResult:
    y = np.asarray(target, dtype=float)
    ranking = score_feature_table(library_df, list(library_df.columns), y, cv_folds)
    if ranking.empty:
        raise ValueError(f"No usable library features for target '{target_name}'")

    selected_columns = ranking["feature"].tolist()[:max_library_features]
    X = cleaned_matrix(library_df, selected_columns)
    X_values = X.to_numpy(dtype=float)
    cv = make_cv(len(y), cv_folds)
    alpha, threshold, _, _ = choose_stlsq_hyperparameters(X_values, y, cv)
    X_scaled, mean, scale = standardize_fit(X_values)
    scaled_intercept, scaled_coefficients = fit_stlsq_scaled(
        X_scaled,
        y,
        ridge_alpha=alpha,
        threshold=threshold,
    )
    original_coefficients = scaled_coefficients / scale

    nonzero = np.flatnonzero(np.abs(original_coefficients) > 1e-10)
    if nonzero.size == 0:
        nonzero = np.array([0])

    if nonzero.size > max_terms:
        strongest = np.argsort(np.abs(original_coefficients[nonzero]))[-max_terms:]
        nonzero = nonzero[strongest]

    final_columns = [selected_columns[index] for index in nonzero]
    final_X = X[final_columns].to_numpy(dtype=float)
    intercept, coefficients = fit_linear_model(final_X, y)
    predictions = predict_linear_model(final_X, intercept, coefficients)
    terms = [
        {"feature": column, "coefficient": float(coefficient)}
        for column, coefficient in zip(final_columns, coefficients)
        if abs(coefficient) > 1e-12
    ]
    expression = format_expression(intercept, terms)

    cv_r2_mean = None
    cv_r2_std = None
    if cv:
        scores = []
        for train_index, test_index in cv:
            fold_intercept, fold_coefficients = fit_linear_model(
                final_X[train_index],
                y[train_index],
            )
            fold_prediction = predict_linear_model(
                final_X[test_index],
                fold_intercept,
                fold_coefficients,
            )
            score = r2_score_np(y[test_index], fold_prediction)
            if np.isfinite(score):
                scores.append(score)
        if scores:
            cv_r2_mean = float(np.mean(scores))
            cv_r2_std = float(np.std(scores))

    return SparseModelResult(
        target=target_name,
        expression=expression,
        intercept=intercept,
        terms=terms,
        in_sample_r2=float(r2_score_np(y, predictions)),
        in_sample_mae=float(mean_absolute_error_np(y, predictions)),
        cv_r2_mean=cv_r2_mean,
        cv_r2_std=cv_r2_std,
        selected_features=final_columns,
        predictions=[float(value) for value in predictions],
    )


def make_polynomial_design(values: np.ndarray, degree: int) -> np.ndarray | None:
    values = np.asarray(values, dtype=float)
    if np.nanstd(values) <= EPS:
        return None
    columns = [values]
    if degree >= 2:
        squared = values**2
        if np.nanstd(squared) > EPS:
            columns.append(squared)
    return np.column_stack(columns)


def format_classic_expression(
    parameter: str,
    feature: str,
    target_transform: str,
    feature_transform: str,
    coefficients: np.ndarray,
) -> str:
    target_label = transform_label(parameter, target_transform)
    feature_label = transform_label(feature, feature_transform)
    expression = f"{target_label} ~= {coefficients[0]:.8g}"
    if len(coefficients) > 1:
        sign = "+" if coefficients[1] >= 0 else "-"
        expression += f" {sign} {abs(coefficients[1]):.8g}*{feature_label}"
    if len(coefficients) > 2:
        sign = "+" if coefficients[2] >= 0 else "-"
        expression += f" {sign} {abs(coefficients[2]):.8g}*({feature_label})^2"
    return expression


def evaluate_classic_relationship(
    parameter: str,
    feature: str,
    parameter_values: np.ndarray,
    feature_values: np.ndarray,
    target_transform: str,
    feature_transform: str,
    degree: int,
    cv_folds: int,
) -> ClassicRelationshipResult | None:
    y = transformed_values(parameter_values, target_transform)
    x = transformed_values(feature_values, feature_transform)
    if y is None or x is None:
        return None

    valid = np.isfinite(x) & np.isfinite(y)
    if np.sum(valid) < 6:
        return None
    x = x[valid]
    y = y[valid]
    design = make_polynomial_design(x, degree)
    if design is None:
        return None

    intercept, coefficients = fit_linear_model(design, y)
    all_coefficients = np.concatenate([[intercept], coefficients])
    predictions = predict_linear_model(design, intercept, coefficients)
    in_sample_r2 = r2_score_np(y, predictions)
    if not np.isfinite(in_sample_r2):
        return None

    cv = make_cv(len(y), cv_folds)
    cv_r2_mean = None
    cv_r2_std = None
    if cv:
        scores = []
        for train_index, test_index in cv:
            fold_intercept, fold_coefficients = fit_linear_model(
                design[train_index],
                y[train_index],
            )
            fold_predictions = predict_linear_model(
                design[test_index],
                fold_intercept,
                fold_coefficients,
            )
            score = r2_score_np(y[test_index], fold_predictions)
            if np.isfinite(score):
                scores.append(score)
        if scores:
            cv_r2_mean = float(np.mean(scores))
            cv_r2_std = float(np.std(scores))

    return ClassicRelationshipResult(
        parameter=parameter,
        feature=feature,
        target_transform=target_transform,
        feature_transform=feature_transform,
        degree=degree,
        expression=format_classic_expression(
            parameter,
            feature,
            target_transform,
            feature_transform,
            all_coefficients,
        ),
        in_sample_r2=float(in_sample_r2),
        in_sample_mae=mean_absolute_error_np(y, predictions),
        cv_r2_mean=cv_r2_mean,
        cv_r2_std=cv_r2_std,
        coefficients=[float(value) for value in all_coefficients],
    )


def find_classic_relationships(
    feature_df: pd.DataFrame,
    feature_columns: list[str],
    parameter: str,
    cv_folds: int,
    max_results: int,
) -> list[ClassicRelationshipResult]:
    parameter_values = feature_df[parameter].to_numpy(dtype=float)
    target_transforms = ["identity", "log10", "sqrt", "inverse", "cuberoot"]
    feature_transforms = [
        "identity",
        "log10",
        "log10_abs",
        "signed_log10",
        "sqrt",
        "sqrt_abs",
        "inverse",
        "square",
        "cuberoot",
    ]
    results = []

    clean_features = cleaned_matrix(feature_df, feature_columns)
    for feature in feature_columns:
        feature_values = clean_features[feature].to_numpy(dtype=float)
        for target_transform in target_transforms:
            for feature_transform in feature_transforms:
                for degree in [1, 2]:
                    result = evaluate_classic_relationship(
                        parameter=parameter,
                        feature=feature,
                        parameter_values=parameter_values,
                        feature_values=feature_values,
                        target_transform=target_transform,
                        feature_transform=feature_transform,
                        degree=degree,
                        cv_folds=cv_folds,
                    )
                    if result is not None:
                        results.append(result)

    def sort_key(result: ClassicRelationshipResult) -> tuple[float, float]:
        cv_score = result.cv_r2_mean if result.cv_r2_mean is not None else -np.inf
        return cv_score, result.in_sample_r2

    results.sort(key=sort_key, reverse=True)
    return results[:max_results]


def classic_results_to_dataframe(results: list[ClassicRelationshipResult]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "parameter": result.parameter,
                "feature": result.feature,
                "target_transform": result.target_transform,
                "feature_transform": result.feature_transform,
                "degree": result.degree,
                "expression": result.expression,
                "in_sample_r2": result.in_sample_r2,
                "in_sample_mae": result.in_sample_mae,
                "cv_r2_mean": result.cv_r2_mean,
                "cv_r2_std": result.cv_r2_std,
                "coefficients": json.dumps(result.coefficients),
            }
            for result in results
        ]
    )


def format_expression(intercept: float, terms: list[dict[str, float | str]]) -> str:
    expression = f"{intercept:.8g}"
    for term in terms:
        coefficient = float(term["coefficient"])
        feature = str(term["feature"])
        sign = "+" if coefficient >= 0 else "-"
        expression += f" {sign} {abs(coefficient):.8g}*{feature}"
    return expression


def target_to_metric_alias(target_name: str) -> str:
    if target_name.endswith("Re") or target_name.endswith("_Re") or target_name == "log10_Re":
        return "f_re"
    if target_name.endswith("h") or target_name.endswith("_h") or target_name == "log10_h":
        return "f_h"
    return f"metric_{target_name}"


def write_report(
    output_dir: Path,
    cases: list[Case],
    target_rankings: dict[str, pd.DataFrame],
    sparse_results: dict[str, SparseModelResult],
    classic_results: dict[str, list[ClassicRelationshipResult]],
    args: argparse.Namespace,
) -> None:
    lines = [
        "# ASMR metric discovery report",
        "",
        f"- Cases used: {len(cases)}",
        f"- Input root: `{args.root}`",
        f"- Pattern: `{args.pattern}`",
        f"- Warmup samples skipped: {args.warmup_samples}",
        f"- Static library degree: {args.degree}",
        "",
        "This report ranks pressure-derived scalar descriptors and fits sparse",
        "static expressions. It is not a standard dynamical SINDy fit because",
        "the target is a scenario parameter, not a time derivative.",
        "",
    ]

    for target_name, ranking in target_rankings.items():
        lines.extend([f"## Target `{target_name}`", ""])
        sparse = sparse_results[target_name]
        lines.extend(
            [
                f"Sparse expression: `{target_name} ~= {sparse.expression}`",
                "",
                f"- In-sample R2: {sparse.in_sample_r2:.4g}",
                f"- In-sample MAE: {sparse.in_sample_mae:.4g}",
            ]
        )
        if sparse.cv_r2_mean is not None:
            lines.append(f"- Cross-validated R2: {sparse.cv_r2_mean:.4g} +/- {sparse.cv_r2_std:.4g}")
        else:
            lines.append("- Cross-validated R2: not available with this sample size")
        lines.extend(["", "Best single descriptors:", ""])

        if ranking.empty:
            lines.append("No usable descriptors were found.")
            lines.append("")
            continue

        top = ranking.head(args.top_k)
        header = "| feature | Spearman | Pearson | MI | single-feature CV R2 |"
        separator = "|---|---:|---:|---:|---:|"
        lines.extend([header, separator])
        for _, row in top.iterrows():
            lines.append(
                "| {feature} | {spearman:.4g} | {pearson:.4g} | {mi:.4g} | {cv:.4g} |".format(
                    feature=row["feature"],
                    spearman=row["spearman"],
                    pearson=row["pearson"],
                    mi=row["mutual_information"],
                    cv=row["single_feature_cv_r2"],
                )
            )
        lines.append("")

    for parameter, results in classic_results.items():
        lines.extend([f"## Classic relationships for `{parameter}`", ""])
        if not results:
            lines.extend(["No usable classic relationships were found.", ""])
            continue

        lines.extend(
            [
                "These fits compare identity/log/square-root/inverse/signed-log",
                "transforms and optional quadratic terms. R2 is measured in the",
                "transformed target space shown in the expression.",
                "",
                "| expression | CV R2 | in-sample R2 | MAE |",
                "|---|---:|---:|---:|",
            ]
        )
        for result in results[: args.top_k]:
            cv_text = ""
            if result.cv_r2_mean is not None:
                cv_text = f"{result.cv_r2_mean:.4g} +/- {result.cv_r2_std:.3g}"
            else:
                cv_text = "n/a"
            lines.append(
                f"| `{result.expression}` | {cv_text} | "
                f"{result.in_sample_r2:.4g} | {result.in_sample_mae:.4g} |"
            )
        lines.append("")

    (output_dir / "discovery_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_discovery(args: argparse.Namespace) -> None:
    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parameter_columns = parse_csv_list(args.parameter_columns)
    target_specs = parse_csv_list(args.targets)
    cases = discover_cases(
        root=root,
        pattern=args.pattern,
        path_regex=args.path_regex,
        parameter_columns=parameter_columns,
        metadata_csv=Path(args.metadata_csv) if args.metadata_csv else None,
    )

    if len(cases) < args.min_cases:
        raise RuntimeError(
            f"Only found {len(cases)} usable cases. Need at least {args.min_cases}. "
            "Check --root, --pattern, --path-regex, or use --metadata-csv."
        )

    feature_rows: list[dict[str, float | str]] = []
    failures = []
    for case in cases:
        re_value = case.parameters.get("Re")
        try:
            if args.input_format in {"lilypad", "lilypad-wall"}:
                pressure, time = read_lilypad_wall_pressure(case.pressure_file, re_value=re_value)
            else:
                pressure, time = read_csv_pressure(case.pressure_file)
            features = extract_pressure_features(
                pressure,
                time,
                re_value=re_value,
                warmup_samples=args.warmup_samples,
                max_sensors=args.max_sensors,
            )
        except Exception as exc:  # noqa: BLE001 - keep batch discovery running.
            failures.append({"file": str(case.pressure_file), "error": str(exc)})
            continue

        row: dict[str, float | str] = {"file": str(case.pressure_file)}
        row.update(case.parameters)
        row.update(features)
        feature_rows.append(row)

    if failures:
        pd.DataFrame(failures).to_csv(output_dir / "failed_cases.csv", index=False)

    if len(feature_rows) < args.min_cases:
        raise RuntimeError(
            f"Only {len(feature_rows)} cases could be parsed. "
            f"See {output_dir / 'failed_cases.csv'} for failures."
        )

    feature_df = pd.DataFrame(feature_rows)
    feature_df.to_csv(output_dir / "case_features.csv", index=False)

    parameter_df = feature_df[parameter_columns].copy()
    targets = make_target_values(parameter_df, target_specs)

    excluded = set(["file", *parameter_columns])
    primitive_columns = numeric_feature_columns(feature_df, excluded)
    if not primitive_columns:
        raise RuntimeError("No usable pressure features were extracted.")

    target_rankings: dict[str, pd.DataFrame] = {}
    sparse_results: dict[str, SparseModelResult] = {}
    classic_results: dict[str, list[ClassicRelationshipResult]] = {}
    metric_output = feature_df[["file", *parameter_columns]].copy()

    for target_name, target_values in targets.items():
        target_feature_columns = feature_columns_for_target(primitive_columns, target_name)
        ranking = score_feature_table(
            feature_df,
            target_feature_columns,
            target_values,
            cv_folds=args.cv_folds,
        )
        target_rankings[target_name] = ranking
        ranking.to_csv(output_dir / f"feature_ranking_{target_name}.csv", index=False)

        library = transformed_library(
            feature_df=feature_df,
            feature_columns=target_feature_columns,
            target=target_values,
            max_base_features=args.max_base_features,
            degree=args.degree,
            max_interaction_sources=args.max_interaction_sources,
        )
        sparse = fit_sparse_model(
            library_df=library,
            target=target_values,
            target_name=target_name,
            cv_folds=args.cv_folds,
            max_library_features=args.max_library_features,
            max_terms=args.max_terms,
        )
        sparse_results[target_name] = sparse

        alias = target_to_metric_alias(target_name)
        metric_output[alias] = sparse.predictions
        metric_output[f"target_{target_name}"] = target_values

    for parameter in parameter_columns:
        target_feature_columns = feature_columns_for_target(primitive_columns, parameter)
        results = find_classic_relationships(
            feature_df=feature_df,
            feature_columns=target_feature_columns,
            parameter=parameter,
            cv_folds=args.cv_folds,
            max_results=args.max_classic_results,
        )
        classic_results[parameter] = results
        classic_results_to_dataframe(results).to_csv(
            output_dir / f"classic_relationships_{parameter}.csv",
            index=False,
        )

    metric_output.to_csv(output_dir / "candidate_metric_values.csv", index=False)

    serializable_results = {
        target: {
            "expression": result.expression,
            "intercept": result.intercept,
            "terms": result.terms,
            "in_sample_r2": result.in_sample_r2,
            "in_sample_mae": result.in_sample_mae,
            "cv_r2_mean": result.cv_r2_mean,
            "cv_r2_std": result.cv_r2_std,
            "selected_features": result.selected_features,
        }
        for target, result in sparse_results.items()
    }
    (output_dir / "sparse_models.json").write_text(
        json.dumps(serializable_results, indent=2),
        encoding="utf-8",
    )

    write_report(output_dir, cases, target_rankings, sparse_results, classic_results, args)

    print(f"Parsed cases: {len(feature_rows)}")
    if failures:
        print(f"Failed cases: {len(failures)} (see {output_dir / 'failed_cases.csv'})")
    print(f"Wrote features: {output_dir / 'case_features.csv'}")
    print(f"Wrote report: {output_dir / 'discovery_report.md'}")
    print("")
    for target_name, result in sparse_results.items():
        print(f"{target_name} ~= {result.expression}")
        if result.cv_r2_mean is not None:
            print(f"  CV R2: {result.cv_r2_mean:.4g} +/- {result.cv_r2_std:.4g}")
        print(f"  In-sample R2: {result.in_sample_r2:.4g}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Discover ASMR-style pressure metrics from brute-force simulations."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("E:/simuChina"),
        help="Root folder containing simulation result folders.",
    )
    parser.add_argument(
        "--pattern",
        default="**/pressureMotion.txt",
        help="Glob pattern under --root for pressure files.",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=None,
        help="Optional CSV with file/path plus parameter columns.",
    )
    parser.add_argument(
        "--path-regex",
        default=DEFAULT_PATH_REGEX,
        help="Regex with named groups matching parameter columns in the file path.",
    )
    parser.add_argument(
        "--parameter-columns",
        default="Re,h",
        help="Comma-separated parameter columns to read from path regex or metadata CSV.",
    )
    parser.add_argument(
        "--targets",
        default="log10_Re,log10_h",
        help="Comma-separated regression targets. Use raw names or log10_<name>.",
    )
    parser.add_argument(
        "--input-format",
        choices=["lilypad-wall", "lilypad", "csv"],
        default="lilypad-wall",
        help="Input format. lilypad-wall extracts wall rows from pressureMotion.txt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("asmr_metric_discovery"),
        help="Directory where reports and CSV files are written.",
    )
    parser.add_argument(
        "--warmup-samples",
        type=int,
        default=4,
        help="Initial samples to ignore before extracting descriptors.",
    )
    parser.add_argument(
        "--max-sensors",
        type=int,
        default=None,
        help="Optional limit on the number of sensor columns used.",
    )
    parser.add_argument(
        "--min-cases",
        type=int,
        default=6,
        help="Minimum number of usable cases required for discovery.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Maximum number of cross-validation folds.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=15,
        help="Number of top descriptors shown per target in the Markdown report.",
    )
    parser.add_argument(
        "--degree",
        type=int,
        choices=[1, 2],
        default=1,
        help="Static library degree. Degree 2 adds pairwise products among top transforms.",
    )
    parser.add_argument(
        "--max-base-features",
        type=int,
        default=40,
        help="Number of primitive descriptors used before transforms/interactions.",
    )
    parser.add_argument(
        "--max-interaction-sources",
        type=int,
        default=12,
        help="Maximum transformed features used to build degree-2 interactions.",
    )
    parser.add_argument(
        "--max-library-features",
        type=int,
        default=80,
        help="Maximum transformed library features passed to sparse regression.",
    )
    parser.add_argument(
        "--max-terms",
        type=int,
        default=5,
        help="Maximum terms retained in each final sparse expression.",
    )
    parser.add_argument(
        "--max-classic-results",
        type=int,
        default=40,
        help="Maximum classic one-feature relationships saved per parameter.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_discovery(args)


if __name__ == "__main__":
    main()
