#!/usr/bin/env python3
"""Compute video-identity ICC from saved measured pressure snapshots."""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class IccResult:
    mode: int
    eigenvalue: float
    variance_fraction: float
    cumulative_variance: float
    icc1: float
    sigma2_video: float
    sigma2_extract: float


def format_energy_tag(energy: float) -> int:
    return int(round(energy * 100 if energy <= 1.0 else energy))


def read_meta(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def quantile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    left = int(math.floor(pos))
    right = int(math.ceil(pos))
    if left == right:
        return sorted_values[left]
    alpha = pos - left
    return sorted_values[left] * (1.0 - alpha) + sorted_values[right] * alpha


def load_snapshot_matrix(path: Path) -> np.ndarray:
    matrix = np.genfromtxt(path, delimiter=",")
    if matrix.size == 0:
        raise ValueError(f"Measured pressure snapshot matrix is empty: {path}")
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    return np.asarray(matrix, dtype=np.float64)


def filter_groups(
    matrix: np.ndarray,
    metadata: list[dict[str, str]],
    group_column: str,
    min_samples_per_group: int,
) -> tuple[np.ndarray, list[str], list[dict[str, str]]]:
    if matrix.shape[1] != len(metadata):
        raise ValueError(
            f"Matrix column count ({matrix.shape[1]}) does not match metadata rows ({len(metadata)})."
        )

    groups = [row.get(group_column, "") for row in metadata]
    counts: dict[str, int] = {}
    for group in groups:
        if group:
            counts[group] = counts.get(group, 0) + 1
    keep = [
        idx
        for idx, group in enumerate(groups)
        if group and counts.get(group, 0) >= min_samples_per_group
    ]
    if not keep:
        raise ValueError(f"No groups with at least {min_samples_per_group} measured pressure segments.")
    return matrix[:, keep], [groups[idx] for idx in keep], [metadata[idx] for idx in keep]


def measured_pod_coefficients(
    matrix: np.ndarray,
    energy_target: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    modes, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    eigenvalues = singular_values * singular_values
    total = float(np.sum(eigenvalues))
    if total <= 0.0:
        raise ValueError("Measured pressure matrix has zero POD energy.")
    cumulative = np.cumsum(eigenvalues) / total
    n_modes = int(np.searchsorted(cumulative, energy_target) + 1)
    n_modes = min(n_modes, modes.shape[1])
    coeffs = (modes[:, :n_modes].T @ matrix).T
    return coeffs, eigenvalues, cumulative, n_modes


def icc_by_mode(coeffs: np.ndarray, groups: list[str], eigenvalues: np.ndarray, cumulative: np.ndarray) -> list[IccResult]:
    by_group: dict[str, list[int]] = {}
    for idx, group in enumerate(groups):
        by_group.setdefault(group, []).append(idx)

    results: list[IccResult] = []
    total_energy = float(np.sum(eigenvalues))
    for mode_idx in range(coeffs.shape[1]):
        values = coeffs[:, mode_idx]
        overall = float(np.mean(values))
        ss_between = 0.0
        ss_within = 0.0
        for indices in by_group.values():
            group_values = values[indices]
            group_mean = float(np.mean(group_values))
            ss_between += len(indices) * (group_mean - overall) ** 2
            ss_within += float(np.sum((group_values - group_mean) ** 2))

        n_samples = len(values)
        n_groups = len(by_group)
        if n_groups > 1 and n_samples > n_groups:
            ms_between = ss_between / (n_groups - 1)
            ms_within = ss_within / (n_samples - n_groups)
            group_sizes = [len(indices) for indices in by_group.values()]
            n_effective = (n_samples - sum(size * size for size in group_sizes) / n_samples) / (n_groups - 1)
            sigma2_extract = max(ms_within, 0.0)
            sigma2_video = max((ms_between - ms_within) / n_effective, 0.0) if n_effective > 0 else 0.0
            denom = sigma2_video + sigma2_extract
            icc1 = sigma2_video / denom if denom > 0 else 0.0
        else:
            sigma2_extract = 0.0
            sigma2_video = 0.0
            icc1 = 0.0

        eigenvalue = float(eigenvalues[mode_idx])
        results.append(
            IccResult(
                mode=mode_idx + 1,
                eigenvalue=eigenvalue,
                variance_fraction=eigenvalue / total_energy if total_energy > 0 else 0.0,
                cumulative_variance=float(cumulative[mode_idx]),
                icc1=icc1,
                sigma2_video=sigma2_video,
                sigma2_extract=sigma2_extract,
            )
        )
    return results


def standardize_coefficients(coeffs: np.ndarray) -> np.ndarray:
    means = np.mean(coeffs, axis=0)
    stds = np.std(coeffs, axis=0, ddof=1)
    stds[~np.isfinite(stds) | (stds == 0.0)] = 1.0
    return (coeffs - means) / stds


def pairwise_distances(coeffs: np.ndarray) -> list[tuple[int, int, float]]:
    pairs: list[tuple[int, int, float]] = []
    for idx in range(coeffs.shape[0]):
        diffs = coeffs[idx + 1 :] - coeffs[idx]
        distances = np.sqrt(np.sum(diffs * diffs, axis=1))
        for offset, distance in enumerate(distances, start=idx + 1):
            pairs.append((idx, offset, float(distance)))
    return pairs


def median_distance_contrast(pairs: list[tuple[int, int, float]], groups: list[str]) -> tuple[float, float, float, float]:
    same: list[float] = []
    different: list[float] = []
    for left, right, distance in pairs:
        if groups[left] == groups[right]:
            same.append(distance)
        else:
            different.append(distance)
    if not same or not different:
        return math.nan, math.nan, math.nan, math.nan
    same_median = statistics.median(same)
    different_median = statistics.median(different)
    difference = different_median - same_median
    ratio = same_median / different_median if different_median > 0.0 else math.nan
    return same_median, different_median, difference, ratio


def distance_lookup(pairs: list[tuple[int, int, float]], n_samples: int) -> np.ndarray:
    lookup = np.zeros((n_samples, n_samples), dtype=np.float64)
    for left, right, distance in pairs:
        lookup[left, right] = distance
        lookup[right, left] = distance
    return lookup


def bootstrap_distance_ci(
    distance_matrix: np.ndarray,
    groups: list[str],
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float]:
    rng = random.Random(seed)
    by_group: dict[str, list[int]] = {}
    for idx, group in enumerate(groups):
        by_group.setdefault(group, []).append(idx)

    group_names = list(by_group)
    deltas: list[float] = []
    for _ in range(n_bootstrap):
        sampled_indices: list[int] = []
        sampled_groups: list[str] = []
        for draw_idx in range(len(group_names)):
            group = rng.choice(group_names)
            members = by_group[group]
            sampled_indices.extend(members)
            sampled_groups.extend([f"{group}__draw{draw_idx}"] * len(members))

        pairs: list[tuple[int, int, float]] = []
        for local_left in range(len(sampled_indices)):
            source_left = sampled_indices[local_left]
            for local_right in range(local_left + 1, len(sampled_indices)):
                source_right = sampled_indices[local_right]
                pairs.append((local_left, local_right, float(distance_matrix[source_left, source_right])))
        _, _, delta, _ = median_distance_contrast(pairs, sampled_groups)
        if math.isfinite(delta):
            deltas.append(delta)
    return quantile(deltas, 0.025), quantile(deltas, 0.975)


def permutation_p_value(
    pairs: list[tuple[int, int, float]],
    groups: list[str],
    observed_delta: float,
    n_permutations: int,
    seed: int,
) -> float:
    rng = random.Random(seed)
    labels = groups[:]
    count_ge = 0
    valid = 0
    for _ in range(n_permutations):
        rng.shuffle(labels)
        _, _, delta, _ = median_distance_contrast(pairs, labels)
        if not math.isfinite(delta):
            continue
        valid += 1
        if delta >= observed_delta:
            count_ge += 1
    return (count_ge + 1) / (valid + 1) if valid else math.nan


def write_per_mode(path: Path, dataset: str, results: list[IccResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "dataset",
                "mode",
                "eigenvalue",
                "variance_fraction",
                "cumulative_variance",
                "icc1",
                "sigma2_video",
                "sigma2_extract",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    dataset,
                    result.mode,
                    f"{result.eigenvalue:.10g}",
                    f"{result.variance_fraction:.10g}",
                    f"{result.cumulative_variance:.10g}",
                    f"{result.icc1:.10g}",
                    f"{result.sigma2_video:.10g}",
                    f"{result.sigma2_extract:.10g}",
                ]
            )


def write_summary(
    csv_path: Path,
    text_path: Path,
    dataset: str,
    n_samples: int,
    n_groups: int,
    n_modes: int,
    variance_explained: float,
    results: list[IccResult],
    median_same_video_distance: float,
    median_different_video_distance: float,
    median_distance_difference: float,
    median_distance_ratio: float,
    bootstrap_ci_low: float,
    bootstrap_ci_high: float,
    permutation_p_value_value: float,
    seed: int,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    weights = [result.eigenvalue for result in results]
    total_weight = sum(weights)
    iccs = [result.icc1 for result in results]
    weighted_icc = sum(weight * icc for weight, icc in zip(weights, iccs)) / total_weight if total_weight else 0.0
    row = {
        "dataset": dataset,
        "n_samples": n_samples,
        "n_groups": n_groups,
        "n_modes": n_modes,
        "variance_explained": variance_explained,
        "weighted_icc1": weighted_icc,
        "median_icc1": statistics.median(iccs),
        "min_icc1": min(iccs),
        "q1_icc1": quantile(iccs, 0.25),
        "q3_icc1": quantile(iccs, 0.75),
        "max_icc1": max(iccs),
        "median_same_video_distance": median_same_video_distance,
        "median_different_video_distance": median_different_video_distance,
        "median_distance_difference": median_distance_difference,
        "median_distance_ratio": median_distance_ratio,
        "bootstrap_ci_low": bootstrap_ci_low,
        "bootstrap_ci_high": bootstrap_ci_high,
        "permutation_p_value": permutation_p_value_value,
        "seed": seed,
    }
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    lines = [
        "Measured-pressure video-identity ICC",
        "====================================",
        "",
        (
            f"{dataset}: {n_samples} measured pressure segments from {n_groups} videos were projected onto "
            f"{n_modes} measured-pressure POD modes explaining {variance_explained:.3f} variance."
        ),
        (
            f"The variance-weighted ICC(1) was {weighted_icc:.3f}; the median per-mode ICC(1) was "
            f"{statistics.median(iccs):.3f} (range {min(iccs):.3f}-{max(iccs):.3f}, "
            f"IQR {quantile(iccs, 0.25):.3f}-{quantile(iccs, 0.75):.3f})."
        ),
        (
            f"Median same-video and different-video distances were {median_same_video_distance:.3f} and "
            f"{median_different_video_distance:.3f}, respectively (difference {median_distance_difference:.3f}, "
            f"95% bootstrap CI {bootstrap_ci_low:.3f}-{bootstrap_ci_high:.3f}; "
            f"permutation p={permutation_p_value_value:.3f}; seed={seed})."
        ),
    ]
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    repo_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="boai")
    parser.add_argument("--input-energy", type=float, default=0.95, help="Energy tag used when saving measured snapshots.")
    parser.add_argument("--pod-energy", type=float, default=0.99, help="Measured POD variance target.")
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=repo_dir / "vortex_outputs/real_pressure_snapshots",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_dir / "vortex_outputs/revision_analysis",
    )
    parser.add_argument("--group-column", default="experiment")
    parser.add_argument("--min-samples-per-group", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260529)
    args = parser.parse_args()

    input_tag = format_energy_tag(args.input_energy)
    matrix_path = args.snapshot_dir / f"real_pressure_snapshots_{args.dataset}_{input_tag}.csv"
    meta_path = args.snapshot_dir / f"real_pressure_snapshots_{args.dataset}_{input_tag}_meta.csv"
    if not matrix_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Missing measured snapshots for {args.dataset}_{input_tag}. "
            "Run save_real_pressure_snapshots.py first."
        )

    matrix = load_snapshot_matrix(matrix_path)
    metadata = read_meta(meta_path)
    matrix, groups, _ = filter_groups(matrix, metadata, args.group_column, args.min_samples_per_group)
    coeffs, eigenvalues, cumulative, n_modes = measured_pod_coefficients(matrix, args.pod_energy)
    coeffs = coeffs[:, :n_modes]
    retained_eigenvalues = eigenvalues[:n_modes]
    retained_cumulative = cumulative[:n_modes]
    results = icc_by_mode(coeffs, groups, retained_eigenvalues, retained_cumulative)
    standardized_coeffs = standardize_coefficients(coeffs)
    pairs = pairwise_distances(standardized_coeffs)
    same_distance, different_distance, distance_difference, distance_ratio = median_distance_contrast(pairs, groups)
    distance_matrix = distance_lookup(pairs, len(groups))
    ci_low, ci_high = bootstrap_distance_ci(distance_matrix, groups, args.bootstrap, args.seed + 17)
    p_value = permutation_p_value(pairs, groups, distance_difference, args.permutations, args.seed + 31)

    suffix = f"{args.dataset}_{input_tag}_pod{format_energy_tag(args.pod_energy)}"
    write_per_mode(args.output_dir / f"measured_pressure_identity_per_mode_{suffix}.csv", args.dataset, results)
    write_summary(
        csv_path=args.output_dir / f"measured_pressure_identity_summary_{suffix}.csv",
        text_path=args.output_dir / f"measured_pressure_identity_summary_{suffix}.txt",
        dataset=args.dataset,
        n_samples=len(groups),
        n_groups=len(set(groups)),
        n_modes=n_modes,
        variance_explained=float(cumulative[n_modes - 1]),
        results=results,
        median_same_video_distance=same_distance,
        median_different_video_distance=different_distance,
        median_distance_difference=distance_difference,
        median_distance_ratio=distance_ratio,
        bootstrap_ci_low=ci_low,
        bootstrap_ci_high=ci_high,
        permutation_p_value_value=p_value,
        seed=args.seed,
    )
    print(
        f"OK: {args.dataset}: n={len(groups)}, videos={len(set(groups))}, "
        f"modes={n_modes}, variance={float(cumulative[n_modes - 1]):.3f}, "
        f"distance_delta={distance_difference:.3f}, p={p_value:.3f}"
    )


if __name__ == "__main__":
    main()
