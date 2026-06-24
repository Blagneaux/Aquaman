from __future__ import annotations

import argparse
import gc
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


WIDTH = 256
HEIGHT = 128
CYLINDER_DIAMETER_GRID = HEIGHT / 20.0
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
SPLIT_SEED = 20260612
POD_ENERGY_THRESHOLD = 0.99

RE_LIST = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
H_LIST = [6.0, 8.64, 11.34, 13.98, 16.68, 19.32, 22.02, 24.66, 27.36, 30.0]

GAPPY_RANK_CANDIDATES = [1, 2, 3, 4, 5, 6, 7, 8]
RIDGE_FACTORS = [0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1.0]

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
    "lines.linewidth": 0.9,
    "lines.markersize": 4.0,
    "lines.markeredgewidth": 0.7,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
}


def pressure_root_candidates() -> list[Path]:
    env_root = os.environ.get("AQUAMAN_PRESSURE_ROOT")
    if env_root:
        return [Path(env_root)]
    if os.name == "nt":
        return [
            Path(f"{drive}:/simuChina/cartesianBDD_FullPressureMap")
            for drive in ("E", "D", "C")
        ]
    return [
        Path(f"/mnt/{drive}/simuChina/cartesianBDD_FullPressureMap")
        for drive in ("e", "d", "c")
    ]


def default_pressure_root() -> Path:
    candidates = pressure_root_candidates()
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def project_root() -> Path:
    return Path(__file__).resolve().parent


def figures_dir() -> Path:
    return project_root() / "article" / "figures"


def cache_dir() -> Path:
    return Path(
        os.environ.get(
            "AQUAMAN_PRESSURE_CACHE",
            str(Path(tempfile.gettempdir()) / "aquaman_sparse_pressure_cache"),
        )
    )


@dataclass(frozen=True)
class Case:
    re: int
    h: float
    gap: float
    regime: str
    path: Path
    n_snapshots: int


@dataclass
class PODModel:
    name: str
    regime: str
    mean: np.ndarray
    modes: np.ndarray
    eigenvalues: np.ndarray
    sensor_indices: np.ndarray
    full_rank: int | None = None
    gappy_rank: int | None = None
    ridge_factor: float | None = None
    ridge_value: float | None = None
    condition_number: float | None = None


@dataclass
class SplitBlock:
    case: Case
    data: np.ndarray
    denominator: float


@dataclass
class ProjectedBlock:
    case: Case
    norm_y: float
    denominator: float
    full_coefficients: np.ndarray
    sensor_values: np.ndarray
    sensor_denominator: float


def gap_from_h(h: float) -> float:
    return round((h - CYLINDER_DIAMETER_GRID / 2.0) / CYLINDER_DIAMETER_GRID, 2)


def classify_gap(gap: float) -> str:
    if gap <= 0.44 + 1e-9:
        return "small"
    if gap <= 1.68 + 1e-9:
        return "intermediate"
    return "large"


def parse_case_dir(path: Path) -> tuple[int, float] | None:
    name = path.name
    if not name.startswith("Re") or "_h" not in name:
        return None
    left, right = name.split("_h", maxsplit=1)
    try:
        return int(left.replace("Re", "")), float(right)
    except ValueError:
        return None


def count_csv_columns(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return len(f.readline().rstrip("\n").split(","))


def discover_cases(root: Path) -> list[Case]:
    cases: list[Case] = []
    for folder in sorted(root.iterdir(), key=lambda p: p.name):
        parsed = parse_case_dir(folder)
        if parsed is None:
            continue
        re, h = parsed
        if re not in RE_LIST or h not in H_LIST:
            continue
        csv_path = folder / "FullMap.csv"
        if not csv_path.exists():
            continue
        gap = gap_from_h(h)
        cases.append(
            Case(
                re=re,
                h=h,
                gap=gap,
                regime=classify_gap(gap),
                path=csv_path,
                n_snapshots=count_csv_columns(csv_path),
            )
        )
    cases.sort(key=lambda c: (c.re, c.h))
    return cases


def case_split_seed(case: Case) -> int:
    return SPLIT_SEED + case.re * 100_000 + int(round(case.h * 100.0))


def split_indices(n_snapshots: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_train = int(round(TRAIN_FRACTION * n_snapshots))
    n_validation = int(round(VALIDATION_FRACTION * n_snapshots))
    n_train = min(max(n_train, 1), n_snapshots - 2)
    n_validation = min(max(n_validation, 1), n_snapshots - n_train - 1)

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(n_snapshots)
    train_idx = np.sort(shuffled[:n_train])
    val_idx = np.sort(shuffled[n_train : n_train + n_validation])
    test_idx = np.sort(shuffled[n_train + n_validation :])
    return train_idx.astype(int), val_idx.astype(int), test_idx.astype(int)


def load_columns(path: Path, columns: np.ndarray | list[int] | None = None) -> np.ndarray:
    data = load_full_map(path)
    if columns is not None:
        columns = np.asarray(columns, dtype=int)
        data = data[:, columns]
    if data.ndim == 1:
        data = data[:, np.newaxis]
    return np.asarray(data, dtype=np.float32)


def load_full_map(path: Path) -> np.ndarray:
    cache = cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    cache_path = cache / f"{path.parent.name}.npy"
    if cache_path.exists() and cache_path.stat().st_mtime >= path.stat().st_mtime:
        return np.load(cache_path, mmap_mode="r")
    print(f"Caching pressure map {path.parent.name}...", flush=True)
    data = np.loadtxt(str(path), delimiter=",", dtype=np.float32)
    np.save(cache_path, data)
    return np.load(cache_path, mmap_mode="r")


def wall_sensor_indices(sensor_count: int = 8, *, y_index: int = 0) -> np.ndarray:
    # Place sensors over the central half of the wall, matching the region swept by the body.
    x_positions = np.rint(np.linspace(WIDTH / 4.0, 3.0 * WIDTH / 4.0, sensor_count)).astype(int)
    return x_positions * HEIGHT + y_index


def collect_basis_matrix(cases: list[Case], basis_stride: int) -> np.ndarray:
    blocks = []
    for i, case in enumerate(cases, start=1):
        if i == 1 or i % 10 == 0 or i == len(cases):
            print(f"  loading basis samples {i}/{len(cases)}", flush=True)
        train_idx = selected_split_indices(case, "train")
        sampled = train_idx[::basis_stride]
        blocks.append(load_columns(case.path, sampled))
    matrix = np.concatenate(blocks, axis=1)
    for block in blocks:
        del block
    gc.collect()
    return matrix


def fit_pod_model(
    name: str,
    regime: str,
    cases: list[Case],
    sensor_indices: np.ndarray,
    *,
    basis_stride: int,
) -> PODModel:
    print(f"Fitting {name} POD basis from {len(cases)} cases...", flush=True)
    matrix = collect_basis_matrix(cases, basis_stride=basis_stride)
    mean = matrix.mean(axis=1).astype(np.float32)
    centered = matrix - mean[:, np.newaxis]
    n_snapshots = centered.shape[1]

    covariance = (centered.T @ centered) / max(n_snapshots - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(covariance.astype(np.float64, copy=False))
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    eigvecs = eigvecs[:, order]

    positive = eigvals > 1e-12
    eigvals = eigvals[positive]
    eigvecs = eigvecs[:, positive]
    rank_limit = retained_energy_rank(eigvals, POD_ENERGY_THRESHOLD)
    modal_eigvals = eigvals[:rank_limit]
    eigvecs = eigvecs[:, :rank_limit]

    singular_values = np.sqrt(modal_eigvals * max(n_snapshots - 1, 1))
    modes = centered @ (eigvecs / singular_values[np.newaxis, :])
    norms = np.linalg.norm(modes, axis=0)
    modes = (modes / norms[np.newaxis, :]).astype(np.float32)

    del matrix, centered, covariance, eigvecs
    gc.collect()
    return PODModel(
        name=name,
        regime=regime,
        mean=mean,
        modes=modes,
        eigenvalues=eigvals.astype(np.float64),
        sensor_indices=sensor_indices,
        full_rank=rank_limit,
    )


def retained_energy_rank(eigenvalues: np.ndarray, threshold: float) -> int:
    if eigenvalues.size == 0:
        return 1
    total = float(np.sum(eigenvalues))
    if total <= 0.0:
        return 1
    cumulative = np.cumsum(eigenvalues) / total
    return int(np.searchsorted(cumulative, threshold, side="left") + 1)


def ridge_value_for(A: np.ndarray, ridge_factor: float) -> float:
    if ridge_factor == 0.0:
        return 0.0
    scale = float(np.trace(A.T @ A) / max(A.shape[1], 1))
    if scale <= 0.0:
        scale = 1.0
    return ridge_factor * scale


def gappy_coefficients(
    modes_sensor: np.ndarray,
    sensor_values: np.ndarray,
    ridge_factor: float,
) -> tuple[np.ndarray, float, float]:
    ridge_value = ridge_value_for(modes_sensor, ridge_factor)
    gram = modes_sensor.T @ modes_sensor
    if ridge_value > 0.0:
        gram = gram + ridge_value * np.eye(gram.shape[0], dtype=gram.dtype)
    rhs = modes_sensor.T @ sensor_values
    try:
        coeffs = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        coeffs = np.linalg.pinv(gram) @ rhs
    condition = float(np.linalg.cond(modes_sensor))
    return coeffs, ridge_value, condition


def centered_error_sse(
    norm_y_squared: float,
    full_coefficients: np.ndarray,
    candidate_coefficients: np.ndarray,
) -> float:
    cross = float(np.sum(candidate_coefficients * full_coefficients))
    energy = float(np.sum(candidate_coefficients * candidate_coefficients))
    return max(norm_y_squared - 2.0 * cross + energy, 0.0)


def denominator_sse(data: np.ndarray) -> float:
    local_mean = data.mean(axis=1, keepdims=True)
    fluctuation = data - local_mean
    value = float(np.sum(fluctuation * fluctuation))
    return max(value, 1e-20)


def evaluate_full_rank(
    model: PODModel,
    cases: list[Case],
    rank: int,
    *,
    split: str,
    stride: int,
) -> tuple[float, float]:
    rank = min(rank, model.modes.shape[1])
    total_sse = 0.0
    total_den = 0.0
    for case in cases:
        indices = selected_split_indices(case, split)[::stride]
        data = load_columns(case.path, indices)
        y = data - model.mean[:, np.newaxis]
        coeffs = model.modes[:, :rank].T @ y
        norm_y = float(np.sum(y * y))
        sse = max(norm_y - float(np.sum(coeffs * coeffs)), 0.0)
        total_sse += sse
        total_den += denominator_sse(data)
        del data, y, coeffs
    gc.collect()
    return math.sqrt(total_sse / total_den), total_sse


def evaluate_gappy_params(
    model: PODModel,
    cases: list[Case],
    rank: int,
    ridge_factor: float,
    *,
    split: str,
    stride: int,
) -> tuple[float, float, float, float]:
    rank = min(rank, model.modes.shape[1])
    modes = model.modes[:, :rank]
    modes_sensor = modes[model.sensor_indices, :]
    total_sse = 0.0
    total_den = 0.0
    total_sensor_sse = 0.0
    total_sensor_den = 0.0
    ridge_value = ridge_value_for(modes_sensor, ridge_factor)
    condition = float(np.linalg.cond(modes_sensor))

    for case in cases:
        indices = selected_split_indices(case, split)[::stride]
        data = load_columns(case.path, indices)
        y = data - model.mean[:, np.newaxis]
        y_sensor = y[model.sensor_indices, :]
        full_coefficients = modes.T @ y
        coeffs, _, _ = gappy_coefficients(modes_sensor, y_sensor, ridge_factor)

        norm_y = float(np.sum(y * y))
        total_sse += centered_error_sse(norm_y, full_coefficients, coeffs)
        total_den += denominator_sse(data)

        sensor_error = y_sensor - modes_sensor @ coeffs
        total_sensor_sse += float(np.sum(sensor_error * sensor_error))
        sensor_den_data = data[model.sensor_indices, :]
        total_sensor_den += denominator_sse(sensor_den_data)

        del data, y, y_sensor, full_coefficients, coeffs, sensor_error, sensor_den_data
    gc.collect()
    return (
        math.sqrt(total_sse / total_den),
        math.sqrt(total_sensor_sse / max(total_sensor_den, 1e-20)),
        ridge_value,
        condition,
    )


def selected_split_indices(case: Case, split: str) -> np.ndarray:
    train_idx, val_idx, test_idx = split_indices(case.n_snapshots, case_split_seed(case))
    if split == "train":
        return train_idx
    if split == "validation":
        return val_idx
    if split == "test":
        return test_idx
    raise ValueError(f"Unknown split: {split}")


def load_split_blocks(cases: list[Case], split: str, stride: int) -> list[SplitBlock]:
    blocks = []
    for case in cases:
        indices = selected_split_indices(case, split)[::stride]
        data = load_columns(case.path, indices)
        blocks.append(SplitBlock(case=case, data=data, denominator=denominator_sse(data)))
    return blocks


def release_blocks(blocks: list[SplitBlock]) -> None:
    for block in blocks:
        del block.data
    blocks.clear()
    gc.collect()


def project_blocks(model: PODModel, blocks: list[SplitBlock]) -> list[ProjectedBlock]:
    projected = []
    modes = model.modes
    for block in blocks:
        y = block.data - model.mean[:, np.newaxis]
        sensor_data = block.data[model.sensor_indices, :]
        projected.append(
            ProjectedBlock(
                case=block.case,
                norm_y=float(np.sum(y * y)),
                denominator=block.denominator,
                full_coefficients=modes.T @ y,
                sensor_values=y[model.sensor_indices, :],
                sensor_denominator=denominator_sse(sensor_data),
            )
        )
        del y, sensor_data
    gc.collect()
    return projected


def release_projected_blocks(blocks: list[ProjectedBlock]) -> None:
    for block in blocks:
        del block.full_coefficients
        del block.sensor_values
    blocks.clear()
    gc.collect()


def score_full_blocks(model: PODModel, blocks: list[SplitBlock], rank: int) -> tuple[float, float]:
    rank = min(rank, model.modes.shape[1])
    modes = model.modes[:, :rank]
    total_sse = 0.0
    total_den = 0.0
    for block in blocks:
        y = block.data - model.mean[:, np.newaxis]
        coeffs = modes.T @ y
        norm_y = float(np.sum(y * y))
        total_sse += max(norm_y - float(np.sum(coeffs * coeffs)), 0.0)
        total_den += block.denominator
        del y, coeffs
    return math.sqrt(total_sse / total_den), total_sse


def score_full_projected(blocks: list[ProjectedBlock], rank: int) -> tuple[float, float]:
    total_sse = 0.0
    total_den = 0.0
    for block in blocks:
        coeffs = block.full_coefficients[:rank, :]
        total_sse += max(block.norm_y - float(np.sum(coeffs * coeffs)), 0.0)
        total_den += block.denominator
    return math.sqrt(total_sse / total_den), total_sse


def score_gappy_blocks(
    model: PODModel,
    blocks: list[SplitBlock],
    rank: int,
    ridge_factor: float,
) -> tuple[float, float, float, float]:
    rank = min(rank, model.modes.shape[1])
    modes = model.modes[:, :rank]
    modes_sensor = modes[model.sensor_indices, :]
    total_sse = 0.0
    total_den = 0.0
    total_sensor_sse = 0.0
    total_sensor_den = 0.0
    ridge_value = ridge_value_for(modes_sensor, ridge_factor)
    condition = float(np.linalg.cond(modes_sensor))

    for block in blocks:
        data = block.data
        y = data - model.mean[:, np.newaxis]
        y_sensor = y[model.sensor_indices, :]
        full_coefficients = modes.T @ y
        coeffs, _, _ = gappy_coefficients(modes_sensor, y_sensor, ridge_factor)
        norm_y = float(np.sum(y * y))
        total_sse += centered_error_sse(norm_y, full_coefficients, coeffs)
        total_den += block.denominator

        sensor_error = y_sensor - modes_sensor @ coeffs
        total_sensor_sse += float(np.sum(sensor_error * sensor_error))
        sensor_data = data[model.sensor_indices, :]
        total_sensor_den += denominator_sse(sensor_data)

        del y, y_sensor, full_coefficients, coeffs, sensor_error, sensor_data
    return (
        math.sqrt(total_sse / total_den),
        math.sqrt(total_sensor_sse / max(total_sensor_den, 1e-20)),
        ridge_value,
        condition,
    )


def score_gappy_projected(
    model: PODModel,
    blocks: list[ProjectedBlock],
    rank: int,
    ridge_factor: float,
) -> tuple[float, float, float, float]:
    rank = min(rank, model.modes.shape[1])
    modes_sensor = model.modes[model.sensor_indices, :rank]
    total_sse = 0.0
    total_den = 0.0
    total_sensor_sse = 0.0
    total_sensor_den = 0.0
    ridge_value = ridge_value_for(modes_sensor, ridge_factor)
    condition = float(np.linalg.cond(modes_sensor))

    for block in blocks:
        coeffs, _, _ = gappy_coefficients(modes_sensor, block.sensor_values, ridge_factor)
        full_coefficients = block.full_coefficients[:rank, :]
        total_sse += centered_error_sse(block.norm_y, full_coefficients, coeffs)
        total_den += block.denominator

        sensor_error = block.sensor_values - modes_sensor @ coeffs
        total_sensor_sse += float(np.sum(sensor_error * sensor_error))
        total_sensor_den += block.sensor_denominator

        del coeffs, full_coefficients, sensor_error
    return (
        math.sqrt(total_sse / total_den),
        math.sqrt(total_sensor_sse / max(total_sensor_den, 1e-20)),
        ridge_value,
        condition,
    )


def score_mean_data(mean: np.ndarray, data: np.ndarray, denominator: float) -> float:
    error = data - mean[:, np.newaxis]
    nrmse = math.sqrt(float(np.sum(error * error)) / denominator)
    del error
    return nrmse


def select_model_parameters(
    model: PODModel,
    cases: list[Case],
    validation_stride: int,
) -> tuple[int, int, float, float, float]:
    full_rank = int(model.full_rank or model.modes.shape[1])
    gappy_candidates = [
        rank
        for rank in GAPPY_RANK_CANDIDATES
        if rank <= len(model.sensor_indices) and rank <= model.modes.shape[1]
    ]
    if not gappy_candidates:
        gappy_candidates = [min(len(model.sensor_indices), model.modes.shape[1])]
    blocks = load_split_blocks(cases, "validation", validation_stride)
    projected = project_blocks(model, blocks)
    try:
        gappy_scores = []
        for gappy_rank in gappy_candidates:
            for ridge_factor in RIDGE_FACTORS:
                nrmse, _, ridge_value, condition = score_gappy_projected(
                    model,
                    projected,
                    gappy_rank,
                    ridge_factor,
                )
                gappy_scores.append((nrmse, gappy_rank, ridge_factor, ridge_value, condition))
        gappy_scores.sort(key=lambda row: row[0])
    finally:
        release_projected_blocks(projected)
        release_blocks(blocks)

    _, gappy_rank, ridge_factor, ridge_value, condition = gappy_scores[0]
    return full_rank, gappy_rank, ridge_factor, ridge_value, condition


def evaluate_mean_baseline(mean: np.ndarray, case: Case) -> tuple[float, float]:
    data = load_columns(case.path, selected_split_indices(case, "test"))
    error = data - mean[:, np.newaxis]
    nrmse = math.sqrt(float(np.sum(error * error)) / denominator_sse(data))
    del data, error
    return nrmse, float("nan")


def evaluate_case_method(
    model: PODModel,
    case: Case,
    method: str,
) -> tuple[float, float, int | None, float | None, float | None, float | None]:
    if method == "full_field_pod":
        rank = int(model.full_rank or 1)
        nrmse, _ = evaluate_full_rank(model, [case], rank, split="test", stride=1)
        return nrmse, float("nan"), rank, None, None, None
    if method == "gappy_pod":
        rank = int(model.gappy_rank or 1)
        ridge_factor = float(model.ridge_factor or 0.0)
        nrmse, sensor_nrmse, ridge_value, condition = evaluate_gappy_params(
            model,
            [case],
            rank,
            ridge_factor,
            split="test",
            stride=1,
        )
        return nrmse, sensor_nrmse, rank, ridge_factor, ridge_value, condition
    raise ValueError(method)


def evaluate_test_method_on_block(
    model: PODModel,
    block: SplitBlock,
    method: str,
) -> tuple[float, float, int | None, float | None, float | None, float | None]:
    if method == "full_field_pod":
        rank = int(model.full_rank or 1)
        nrmse, _ = score_full_blocks(model, [block], rank)
        return nrmse, float("nan"), rank, None, None, None
    if method == "gappy_pod":
        rank = int(model.gappy_rank or 1)
        ridge_factor = float(model.ridge_factor or 0.0)
        nrmse, sensor_nrmse, ridge_value, condition = score_gappy_blocks(
            model,
            [block],
            rank,
            ridge_factor,
        )
        return nrmse, sensor_nrmse, rank, ridge_factor, ridge_value, condition
    raise ValueError(method)


def configure_models(
    cases: list[Case],
    sensor_count: int,
    basis_stride: int,
    validation_stride: int,
) -> tuple[PODModel, dict[str, PODModel]]:
    sensors = wall_sensor_indices(sensor_count)
    global_model = fit_pod_model(
        "global",
        "all",
        cases,
        sensors,
        basis_stride=basis_stride,
    )
    (
        global_model.full_rank,
        global_model.gappy_rank,
        global_model.ridge_factor,
        global_model.ridge_value,
        global_model.condition_number,
    ) = select_model_parameters(global_model, cases, validation_stride)
    print(
        "Global model:",
        f"full_rank={global_model.full_rank}",
        f"gappy_rank={global_model.gappy_rank}",
        f"ridge_factor={global_model.ridge_factor}",
        flush=True,
    )

    local_models: dict[str, PODModel] = {}
    for regime in ["small", "intermediate", "large"]:
        regime_cases = [case for case in cases if case.regime == regime]
        model = fit_pod_model(
            f"local_{regime}",
            regime,
            regime_cases,
            sensors,
            basis_stride=basis_stride,
        )
        (
            model.full_rank,
            model.gappy_rank,
            model.ridge_factor,
            model.ridge_value,
            model.condition_number,
        ) = select_model_parameters(model, regime_cases, validation_stride)
        print(
            f"Local {regime} model:",
            f"full_rank={model.full_rank}",
            f"gappy_rank={model.gappy_rank}",
            f"ridge_factor={model.ridge_factor}",
            flush=True,
        )
        local_models[regime] = model
    return global_model, local_models


def run_benchmark(
    cases: list[Case],
    *,
    sensor_count: int,
    basis_stride: int,
    validation_stride: int,
) -> pd.DataFrame:
    global_model, local_models = configure_models(
        cases,
        sensor_count=sensor_count,
        basis_stride=basis_stride,
        validation_stride=validation_stride,
    )
    rows = []
    for i, case in enumerate(cases, start=1):
        print(
            f"Evaluating case {i}/{len(cases)}: Re={case.re}, h={case.h}, gap={case.gap}",
            flush=True,
        )
        test_data = load_columns(case.path, selected_split_indices(case, "test"))
        test_block = SplitBlock(case=case, data=test_data, denominator=denominator_sse(test_data))

        mean_nrmse = score_mean_data(global_model.mean, test_data, test_block.denominator)
        rows.append(result_row(case, "mean_field", "global", mean_nrmse))

        nrmse, sensor_nrmse, rank, ridge_factor, ridge_value, condition = evaluate_test_method_on_block(
            global_model,
            test_block,
            "full_field_pod",
        )
        rows.append(
            result_row(
                case,
                "full_field_pod",
                "global",
                nrmse,
                sensor_nrmse=sensor_nrmse,
                rank=rank,
                ridge_factor=ridge_factor,
                ridge_value=ridge_value,
                condition_number=condition,
            )
        )

        nrmse, sensor_nrmse, rank, ridge_factor, ridge_value, condition = evaluate_test_method_on_block(
            global_model,
            test_block,
            "gappy_pod",
        )
        rows.append(
            result_row(
                case,
                "global_wall_gappy_pod",
                "global",
                nrmse,
                sensor_nrmse=sensor_nrmse,
                rank=rank,
                ridge_factor=ridge_factor,
                ridge_value=ridge_value,
                condition_number=condition,
            )
        )

        local_model = local_models[case.regime]
        nrmse, sensor_nrmse, rank, ridge_factor, ridge_value, condition = evaluate_test_method_on_block(
            local_model,
            test_block,
            "gappy_pod",
        )
        rows.append(
            result_row(
                case,
                "regime_local_wall_gappy_pod",
                local_model.name,
                nrmse,
                sensor_nrmse=sensor_nrmse,
                rank=rank,
                ridge_factor=ridge_factor,
                ridge_value=ridge_value,
                condition_number=condition,
            )
        )

        nrmse, sensor_nrmse, rank, ridge_factor, ridge_value, condition = evaluate_test_method_on_block(
            local_model,
            test_block,
            "full_field_pod",
        )
        rows.append(
            result_row(
                case,
                "regime_local_full_field_pod",
                local_model.name,
                nrmse,
                sensor_nrmse=sensor_nrmse,
                rank=rank,
                ridge_factor=ridge_factor,
                ridge_value=ridge_value,
                condition_number=condition,
            )
        )
        del test_data, test_block
        gc.collect()
    return pd.DataFrame(rows)


def result_row(
    case: Case,
    method: str,
    basis: str,
    nrmse: float,
    *,
    sensor_nrmse: float = float("nan"),
    rank: int | None = None,
    ridge_factor: float | None = None,
    ridge_value: float | None = None,
    condition_number: float | None = None,
) -> dict[str, object]:
    return {
        "Re": case.re,
        "h": case.h,
        "gap": case.gap,
        "regime": case.regime,
        "method": method,
        "basis": basis,
        "nrmse": nrmse,
        "sensor_nrmse": sensor_nrmse,
        "rank": rank,
        "ridge_factor": ridge_factor,
        "ridge_value": ridge_value,
        "condition_number": condition_number,
        "n_snapshots": case.n_snapshots,
    }


def apply_style() -> None:
    plt.rcParams.update(CAMBRIDGE_RC)


def save_figure(
    fig: plt.Figure,
    out_base: Path,
    *,
    save_pdf: bool = True,
    save_tiff: bool = False,
) -> None:
    fig.savefig(f"{out_base}.eps", format="eps", bbox_inches="tight")
    if save_pdf:
        fig.savefig(f"{out_base}.pdf", format="pdf", bbox_inches="tight")
    if save_tiff:
        fig.savefig(f"{out_base}.tiff", format="tiff", dpi=1200, bbox_inches="tight")


METHOD_LABELS = {
    "mean_field": "Mean field",
    "full_field_pod": "Full-field POD",
    "regime_local_full_field_pod": "Regime-local full-field POD",
    "global_wall_gappy_pod": "Global wall gappy POD",
    "regime_local_wall_gappy_pod": "Regime-local wall gappy POD",
}

METHOD_ORDER = [
    "mean_field",
    "full_field_pod",
    "regime_local_full_field_pod",
    "global_wall_gappy_pod",
    "regime_local_wall_gappy_pod",
]


def aggregate(df: pd.DataFrame, by: str) -> pd.DataFrame:
    plot_df = df[df["method"].isin(METHOD_ORDER)]
    return (
        plot_df.groupby(["method", by], as_index=False)
        .agg(mean_nrmse=("nrmse", "mean"), std_nrmse=("nrmse", "std"))
        .sort_values([by, "method"])
    )


def plot_methods(df: pd.DataFrame, by: str, out_base: Path, xlabel: str) -> None:
    apply_style()
    agg = aggregate(df, by)
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for method in METHOD_ORDER:
        sub = agg[agg["method"] == method].sort_values(by)
        ax.errorbar(
            sub[by],
            sub["mean_nrmse"],
            yerr=sub["std_nrmse"].fillna(0.0),
            marker="o",
            markerfacecolor="none",
            capsize=2,
            label=METHOD_LABELS[method],
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Test pressure nRMSE")
    ax.grid(True, linewidth=0.6)
    if by == "gap":
        ax.set_xticks(sorted(df["gap"].unique()))
        ax.tick_params(axis="x", labelrotation=45)
        for label in ax.get_xticklabels():
            label.set_ha("right")
    ax.legend(frameon=True, framealpha=1.0)
    fig.tight_layout()
    save_figure(fig, out_base)
    plt.close(fig)


def plot_condition_numbers(df: pd.DataFrame, out_base: Path) -> None:
    apply_style()
    methods = ["global_wall_gappy_pod", "regime_local_wall_gappy_pod"]
    sub = df[df["method"].isin(methods)].copy()
    agg = (
        sub.groupby(["method", "gap"], as_index=False)
        .agg(mean_condition=("condition_number", "mean"))
        .sort_values("gap")
    )
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for method in methods:
        method_df = agg[agg["method"] == method]
        ax.plot(
            method_df["gap"],
            method_df["mean_condition"],
            marker="o",
            markerfacecolor="none",
            label=METHOD_LABELS[method],
        )
    ax.set_xlabel("Gap / cylinder diameter")
    ax.set_ylabel(r"Condition number of $H\Phi_r$")
    ax.set_yscale("log")
    ax.grid(True, linewidth=0.6)
    ax.set_xticks(sorted(df["gap"].unique()))
    ax.tick_params(axis="x", labelrotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.legend(frameon=True, framealpha=1.0)
    fig.tight_layout()
    save_figure(fig, out_base)
    plt.close(fig)


def write_latex_tables(df: pd.DataFrame, out_dir: Path) -> None:
    dataset_rows = (
        df.drop_duplicates(["Re", "h"])[["Re", "h", "gap", "regime", "n_snapshots"]]
        .sort_values(["Re", "h"])
        .groupby("regime", as_index=False)
        .agg(
            cases=("Re", "count"),
            gaps=("gap", lambda s: f"{s.min():.2f}-{s.max():.2f}"),
            reynolds=("Re", lambda s: f"{s.min()}-{s.max()}"),
            snapshots=("n_snapshots", lambda s: f"{s.min()}-{s.max()}"),
        )
    )
    method_rows = (
        df[df["method"].isin(METHOD_ORDER)]
        .groupby("method", as_index=False)
        .agg(
            mean_nrmse=("nrmse", "mean"),
            max_nrmse=("nrmse", "max"),
            mean_sensor_nrmse=("sensor_nrmse", "mean"),
            median_condition=("condition_number", "median"),
        )
    )
    method_rows["method"] = method_rows["method"].map(METHOD_LABELS)
    method_rows["_order"] = method_rows["method"].map(
        {METHOD_LABELS[method]: i for i, method in enumerate(METHOD_ORDER)}
    )
    method_rows = method_rows.sort_values("_order").drop(columns="_order")
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_rows.to_csv(out_dir / "sparse_pressure_dataset_summary.csv", index=False)
    method_rows.to_csv(out_dir / "sparse_pressure_method_summary.csv", index=False)
    with (out_dir / "sparse_pressure_dataset_summary.tex").open("w", encoding="utf-8") as f:
        f.write(simple_latex_table(dataset_rows))
    with (out_dir / "sparse_pressure_method_summary.tex").open("w", encoding="utf-8") as f:
        f.write(simple_latex_table(method_rows))


def simple_latex_table(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    lines = [
        "\\begin{tabular}{" + "l" * len(columns) + "}",
        "\\hline",
        " & ".join(escape_latex(str(col)) for col in columns) + r" \\",
        "\\hline",
    ]
    for _, row in df.iterrows():
        values = []
        for value in row:
            if isinstance(value, float):
                if math.isnan(value):
                    values.append("--")
                elif abs(value) >= 100 or abs(value) < 0.01:
                    values.append(f"{value:.3g}")
                else:
                    values.append(f"{value:.3f}")
            else:
                values.append(escape_latex(str(value)))
        lines.append(" & ".join(values) + r" \\")
    lines.extend(["\\hline", "\\end{tabular}", ""])
    return "\n".join(lines)


def escape_latex(value: str) -> str:
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
    )


def run_subset(cases: list[Case]) -> list[Case]:
    keep_re = {1000, 10000}
    keep_h = {6.0, 13.98, 30.0}
    return [case for case in cases if case.re in keep_re and case.h in keep_h]


def main() -> None:
    parser = argparse.ArgumentParser(description="Sparse wall-pressure POD benchmark")
    parser.add_argument("--pressure-root", type=Path, default=default_pressure_root())
    parser.add_argument("--results", type=Path)
    parser.add_argument("--sensor-count", type=int, default=8)
    parser.add_argument("--basis-stride", type=int, default=6)
    parser.add_argument("--validation-stride", type=int, default=4)
    parser.add_argument("--subset", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    out_dir = figures_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_subset" if args.subset else ""

    if args.plot_only:
        results_path = args.results or out_dir / f"sparse_pressure_benchmark_results{suffix}.csv"
        df = pd.read_csv(results_path)
        print(f"Regenerating figures and tables from {results_path}", flush=True)
    else:
        cases = discover_cases(args.pressure_root)
        if args.subset:
            cases = run_subset(cases)
        if not cases:
            raise RuntimeError(f"No pressure cases found in {args.pressure_root}")

        print(f"Running sparse pressure benchmark on {len(cases)} cases", flush=True)
        print(f"Pressure root: {args.pressure_root}", flush=True)
        print(f"Sensor indices: {wall_sensor_indices(args.sensor_count).tolist()}", flush=True)
        df = run_benchmark(
            cases,
            sensor_count=args.sensor_count,
            basis_stride=args.basis_stride,
            validation_stride=args.validation_stride,
        )
        results_path = out_dir / f"sparse_pressure_benchmark_results{suffix}.csv"
        df.to_csv(results_path, index=False)

    write_latex_tables(df, out_dir)
    plot_methods(
        df,
        "gap",
        out_dir / f"sparse_pressure_methods_vs_gap{suffix}",
        "Gap / cylinder diameter",
    )
    plot_methods(
        df,
        "Re",
        out_dir / f"sparse_pressure_methods_vs_re{suffix}",
        "Reynolds number",
    )
    plot_condition_numbers(
        df,
        out_dir / f"sparse_pressure_condition_vs_gap{suffix}",
    )
    print(f"Wrote results to {results_path}", flush=True)


if __name__ == "__main__":
    main()
