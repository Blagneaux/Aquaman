#!/usr/bin/env python3
"""Assess pressure-segment dependence on video/fish identity.

This script intentionally uses only the Python standard library so it can run in
minimal review environments. It reuses the existing 99% pressure-mode bases from
``vortex_outputs/pressure_modes`` and projects retained fish-model pressure
segments into that POD space. By default, fish identity is approximated by the
video/experiment identifier, matching the current metadata available in this
repository.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class Segment:
    sample_id: str
    group: str
    values: list[float]


@dataclass
class IdentitySummary:
    dataset: str
    n_samples: int
    n_groups: int
    n_modes: int
    weighted_identity_fraction: float
    weighted_icc1: float
    median_icc1: float
    min_icc1: float
    q1_icc1: float
    q3_icc1: float
    max_icc1: float
    median_identity_fraction: float
    min_identity_fraction: float
    q1_identity_fraction: float
    q3_identity_fraction: float
    max_identity_fraction: float
    mean_icc1: float
    median_same_group_distance: float
    median_different_group_distance: float
    median_distance_difference: float
    median_distance_ratio: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float
    permutation_p_value: float
    seed: int


def parse_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_float(raw: str) -> float:
    try:
        return float(raw)
    except ValueError:
        return math.nan


def read_npy_from_npz(npz_path: Path, member: str) -> tuple[tuple[int, ...], list[float | int | str]]:
    """Read a simple C-order .npy array from an .npz without NumPy."""

    with zipfile.ZipFile(npz_path, "r") as archive:
        payload = archive.read(member)

    if not payload.startswith(b"\x93NUMPY"):
        raise ValueError(f"{member} in {npz_path} is not a .npy payload")

    major = payload[6]
    if major == 1:
        header_len = int.from_bytes(payload[8:10], "little")
        offset = 10
    elif major in (2, 3):
        header_len = int.from_bytes(payload[8:12], "little")
        offset = 12
    else:
        raise ValueError(f"Unsupported .npy version {major}")

    header = payload[offset : offset + header_len].decode("latin1")
    data = payload[offset + header_len :]

    descr = header.split("'descr':", 1)[1].split(",", 1)[0].strip().strip("'\"")
    shape_text = header.split("'shape':", 1)[1].split(")", 1)[0] + ")"
    shape_values = [v.strip() for v in shape_text.strip("() ").split(",") if v.strip()]
    shape = tuple(int(v) for v in shape_values)
    fortran_order = "True" in header.split("'fortran_order':", 1)[1].split(",", 1)[0]
    if fortran_order:
        raise ValueError(f"Fortran-order arrays are not supported: {npz_path}:{member}")

    import struct

    count = 1
    for dim in shape:
        count *= dim

    if descr in ("<f8", "|f8"):
        fmt = "<" + "d" * count
        values = list(struct.unpack(fmt, data[: 8 * count]))
    elif descr in ("<f4", "|f4"):
        fmt = "<" + "f" * count
        values = [float(v) for v in struct.unpack(fmt, data[: 4 * count])]
    elif descr in ("<i8", "|i8"):
        fmt = "<" + "q" * count
        values = list(struct.unpack(fmt, data[: 8 * count]))
    elif descr in ("<i4", "|i4"):
        fmt = "<" + "i" * count
        values = list(struct.unpack(fmt, data[: 4 * count]))
    elif descr.startswith("|S"):
        width = int(descr[2:])
        values = [data[i * width : (i + 1) * width].rstrip(b"\x00").decode("utf-8") for i in range(count)]
    else:
        raise ValueError(f"Unsupported dtype {descr} in {npz_path}:{member}")

    return shape, values


def load_modes(npz_path: Path) -> tuple[list[list[float]], int]:
    shape, values = read_npy_from_npz(npz_path, "modes.npy")
    if len(shape) != 2:
        raise ValueError(f"Expected 2D modes in {npz_path}, got {shape}")
    rows, cols = shape
    modes: list[list[float]] = []
    for row_idx in range(rows):
        start = row_idx * cols
        modes.append([float(v) for v in values[start : start + cols]])
    return modes, cols


def load_scalar_int(npz_path: Path, member: str, fallback: int) -> int:
    try:
        _, values = read_npy_from_npz(npz_path, member)
    except Exception:
        return fallback
    if not values:
        return fallback
    return int(values[0])


def read_metadata(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_matrix_columns(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            rows.append([parse_float(value) for value in row])
    if not rows:
        return []
    n_cols = max(len(row) for row in rows)
    cols: list[list[float]] = [[] for _ in range(n_cols)]
    for row in rows:
        for idx in range(n_cols):
            value = row[idx] if idx < len(row) else math.nan
            cols[idx].append(value)
    return cols


def fill_internal_nans(values: list[float]) -> list[float]:
    if not values:
        return []
    good = [idx for idx, value in enumerate(values) if math.isfinite(value)]
    if not good:
        return [0.0 for _ in values]
    out = values[:]
    first = good[0]
    for idx in range(0, first):
        out[idx] = out[first]
    last = good[-1]
    for idx in range(last + 1, len(out)):
        out[idx] = out[last]
    for left, right in zip(good, good[1:]):
        if right == left + 1:
            continue
        span = right - left
        for idx in range(left + 1, right):
            alpha = (idx - left) / span
            out[idx] = out[left] * (1.0 - alpha) + out[right] * alpha
    return out


def has_tail_spike(values: list[float], tail_fraction: float, spike_ratio: float, min_tail: int) -> bool:
    n = len(values)
    if n < 2:
        return False
    tail_len = max(int(round(n * tail_fraction)), min_tail)
    tail_len = min(tail_len, n - 1)
    body = [abs(v) for v in values[: n - tail_len] if math.isfinite(v)]
    tail = [abs(v) for v in values[n - tail_len :] if math.isfinite(v)]
    if not body or not tail:
        return False
    return max(tail) > spike_ratio * max(max(body), 1e-9)


def resample(values: list[float], target_length: int) -> list[float]:
    if len(values) == target_length:
        return values[:]
    if not values:
        return [0.0] * target_length
    if len(values) == 1:
        return [values[0]] * target_length
    out: list[float] = []
    old_last = len(values) - 1
    new_last = max(target_length - 1, 1)
    for idx in range(target_length):
        pos = idx * old_last / new_last
        left = int(math.floor(pos))
        right = min(left + 1, old_last)
        alpha = pos - left
        out.append(values[left] * (1.0 - alpha) + values[right] * alpha)
    return out


def preprocess(values: list[float], target_length: int) -> list[float]:
    filled = fill_internal_nans(values)
    sampled = resample(filled, target_length)
    mean = statistics.fmean(sampled) if sampled else 0.0
    centred = [value - mean for value in sampled]
    scale = max((abs(value) for value in centred), default=0.0)
    if scale > 0.0 and math.isfinite(scale):
        centred = [value / scale for value in centred]
    return centred


def load_segments(
    matrix_path: Path,
    meta_path: Path,
    target_length: int,
    group_column: str,
    min_length: int,
    min_samples_per_group: int,
    tail_spike_filter: bool,
    tail_fraction: float,
    tail_ratio: float,
    tail_min: int,
) -> list[Segment]:
    metadata = read_metadata(meta_path)
    columns = read_matrix_columns(matrix_path)
    segments: list[Segment] = []
    for idx, values in enumerate(columns):
        if idx >= len(metadata):
            break
        meta = metadata[idx]
        try:
            length = int(float(meta.get("length", "0")))
        except ValueError:
            length = 0
        if length < min_length:
            continue
        raw = values[: min(length, len(values))]
        if tail_spike_filter and has_tail_spike(raw, tail_fraction, tail_ratio, tail_min):
            continue
        group = meta.get(group_column, "")
        if not group:
            continue
        sample_id = f"{meta.get('dataset', '')}:{meta.get('model', '')}:{meta.get('experiment', '')}:{meta.get('snapshot', '')}:{meta.get('sensor_id', '')}:{meta.get('start_frame', '')}-{meta.get('end_frame', '')}"
        segments.append(Segment(sample_id=sample_id, group=group, values=preprocess(raw, target_length)))

    group_counts: dict[str, int] = {}
    for segment in segments:
        group_counts[segment.group] = group_counts.get(segment.group, 0) + 1
    allowed = {group for group, count in group_counts.items() if count >= min_samples_per_group}
    return [segment for segment in segments if segment.group in allowed]


def project_segments(segments: list[Segment], modes: list[list[float]]) -> list[list[float]]:
    if not segments:
        return []
    n_modes = len(modes[0])
    coeffs: list[list[float]] = []
    for segment in segments:
        row = [0.0] * n_modes
        for t_idx, value in enumerate(segment.values):
            if t_idx >= len(modes):
                break
            mode_row = modes[t_idx]
            for mode_idx in range(n_modes):
                row[mode_idx] += mode_row[mode_idx] * value
        coeffs.append(row)
    return coeffs


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else math.nan


def quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return math.nan
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    left = int(math.floor(pos))
    right = int(math.ceil(pos))
    if left == right:
        return sorted_values[left]
    alpha = pos - left
    return sorted_values[left] * (1.0 - alpha) + sorted_values[right] * alpha


def variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = statistics.fmean(values)
    return sum((value - mu) ** 2 for value in values) / (len(values) - 1)


def identity_variance_fractions(
    coeffs: list[list[float]], groups: list[str]
) -> tuple[list[float], list[float], list[float], list[float], list[float]]:
    n_modes = len(coeffs[0]) if coeffs else 0
    by_group: dict[str, list[int]] = {}
    for idx, group in enumerate(groups):
        by_group.setdefault(group, []).append(idx)

    fractions: list[float] = []
    iccs: list[float] = []
    video_variances: list[float] = []
    extract_variances: list[float] = []
    mode_variances: list[float] = []
    for mode_idx in range(n_modes):
        values = [row[mode_idx] for row in coeffs]
        overall = statistics.fmean(values)
        ss_total = sum((value - overall) ** 2 for value in values)
        ss_between = 0.0
        ss_within = 0.0
        for indices in by_group.values():
            group_values = [values[idx] for idx in indices]
            group_mean = statistics.fmean(group_values)
            ss_between += len(indices) * (group_mean - overall) ** 2
            ss_within += sum((value - group_mean) ** 2 for value in group_values)
        fraction = ss_between / ss_total if ss_total > 0 else 0.0
        fractions.append(fraction)
        mode_variances.append(variance(values))

        n = len(values)
        k = len(by_group)
        if k > 1 and n > k:
            ms_between = ss_between / (k - 1)
            ms_within = ss_within / (n - k)
            group_sizes = [len(indices) for indices in by_group.values()]
            n_effective = (n - sum(size * size for size in group_sizes) / n) / (k - 1)
            sigma_extract = max(ms_within, 0.0)
            sigma_video = max((ms_between - ms_within) / n_effective, 0.0) if n_effective > 0 else 0.0
            denom = sigma_video + sigma_extract
            iccs.append(sigma_video / denom if denom > 0 else 0.0)
            video_variances.append(sigma_video)
            extract_variances.append(sigma_extract)
        else:
            iccs.append(0.0)
            video_variances.append(0.0)
            extract_variances.append(0.0)
    return fractions, iccs, video_variances, extract_variances, mode_variances


def standardize_coeffs(coeffs: list[list[float]]) -> list[list[float]]:
    if not coeffs:
        return []
    n_modes = len(coeffs[0])
    means = [statistics.fmean(row[idx] for row in coeffs) for idx in range(n_modes)]
    stds = [math.sqrt(variance([row[idx] for row in coeffs])) for idx in range(n_modes)]
    stds = [std if std > 0 and math.isfinite(std) else 1.0 for std in stds]
    return [[(row[idx] - means[idx]) / stds[idx] for idx in range(n_modes)] for row in coeffs]


def pairwise_distances(coeffs: list[list[float]]) -> list[tuple[int, int, float]]:
    pairs: list[tuple[int, int, float]] = []
    for i in range(len(coeffs)):
        xi = coeffs[i]
        for j in range(i + 1, len(coeffs)):
            xj = coeffs[j]
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(xi, xj)))
            pairs.append((i, j, dist))
    return pairs


def distance_lookup(pairs: list[tuple[int, int, float]], n: int) -> list[list[float]]:
    lookup = [[0.0] * n for _ in range(n)]
    for i, j, dist in pairs:
        lookup[i][j] = dist
        lookup[j][i] = dist
    return lookup


def median_contrast(pairs: list[tuple[int, int, float]], groups: list[str]) -> tuple[float, float, float, float]:
    same: list[float] = []
    different: list[float] = []
    for i, j, dist in pairs:
        if groups[i] == groups[j]:
            same.append(dist)
        else:
            different.append(dist)
    if not same or not different:
        return math.nan, math.nan, math.nan, math.nan
    same_med = statistics.median(same)
    diff_med = statistics.median(different)
    delta = diff_med - same_med
    ratio = same_med / diff_med if diff_med > 0 else math.nan
    return same_med, diff_med, delta, ratio


def bootstrap_distance_ci(
    distance_matrix: list[list[float]],
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
        for local_i in range(len(sampled_indices)):
            source_i = sampled_indices[local_i]
            for local_j in range(local_i + 1, len(sampled_indices)):
                source_j = sampled_indices[local_j]
                pairs.append((local_i, local_j, distance_matrix[source_i][source_j]))
        _, _, delta, _ = median_contrast(pairs, sampled_groups)
        if math.isfinite(delta):
            deltas.append(delta)
    deltas.sort()
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
        _, _, delta, _ = median_contrast(pairs, labels)
        if not math.isfinite(delta):
            continue
        valid += 1
        if delta >= observed_delta:
            count_ge += 1
    return (count_ge + 1) / (valid + 1) if valid else math.nan


def analyze_dataset(args: argparse.Namespace, dataset: str) -> tuple[IdentitySummary, list[dict[str, str]]]:
    mode_tag = int(round(args.energy * 100 if args.energy <= 1.0 else args.energy))
    modes_path = args.modes_dir / f"pressure_modes_{dataset}_{mode_tag}.npz"
    matrix_path = args.snapshot_dir / f"pressure_snapshots_{dataset}_{args.model}.csv"
    meta_path = args.snapshot_dir / f"pressure_snapshots_{dataset}_{args.model}_meta.csv"
    if not modes_path.exists():
        raise FileNotFoundError(modes_path)
    if not matrix_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing pressure snapshots for {dataset} {args.model}")

    modes, n_modes = load_modes(modes_path)
    target_length = load_scalar_int(modes_path, "target_length.npy", len(modes))
    segments = load_segments(
        matrix_path=matrix_path,
        meta_path=meta_path,
        target_length=target_length,
        group_column=args.group_column,
        min_length=args.min_length,
        min_samples_per_group=args.min_samples_per_group,
        tail_spike_filter=not args.no_tail_spike_filter,
        tail_fraction=args.tail_frac,
        tail_ratio=args.tail_ratio,
        tail_min=args.tail_min,
    )
    if len(segments) < 3:
        raise RuntimeError(f"Not enough retained segments for {dataset}: {len(segments)}")

    groups = [segment.group for segment in segments]
    coeffs = project_segments(segments, modes)
    fractions, iccs, video_variances, extract_variances, mode_variances = identity_variance_fractions(coeffs, groups)
    total_variance = sum(mode_variances)
    weighted_fraction = (
        sum(weight * frac for weight, frac in zip(mode_variances, fractions)) / total_variance
        if total_variance > 0
        else 0.0
    )
    weighted_icc = (
        sum(weight * icc for weight, icc in zip(mode_variances, iccs)) / total_variance
        if total_variance > 0
        else 0.0
    )
    sorted_fractions = sorted(fractions)
    sorted_iccs = sorted(iccs)

    scaled_coeffs = standardize_coeffs(coeffs)
    pairs = pairwise_distances(scaled_coeffs)
    distance_matrix = distance_lookup(pairs, len(scaled_coeffs))
    same_med, diff_med, delta, ratio = median_contrast(pairs, groups)
    ci_low, ci_high = bootstrap_distance_ci(distance_matrix, groups, args.bootstrap, args.seed + 17)
    p_value = permutation_p_value(pairs, groups, delta, args.permutations, args.seed + 31)

    group_set = set(groups)
    summary = IdentitySummary(
        dataset=dataset,
        n_samples=len(segments),
        n_groups=len(group_set),
        n_modes=n_modes,
        weighted_identity_fraction=weighted_fraction,
        weighted_icc1=weighted_icc,
        median_icc1=statistics.median(iccs),
        min_icc1=min(iccs),
        q1_icc1=quantile(sorted_iccs, 0.25),
        q3_icc1=quantile(sorted_iccs, 0.75),
        max_icc1=max(iccs),
        median_identity_fraction=statistics.median(fractions),
        min_identity_fraction=min(fractions),
        q1_identity_fraction=quantile(sorted_fractions, 0.25),
        q3_identity_fraction=quantile(sorted_fractions, 0.75),
        max_identity_fraction=max(fractions),
        mean_icc1=statistics.fmean(iccs),
        median_same_group_distance=same_med,
        median_different_group_distance=diff_med,
        median_distance_difference=delta,
        median_distance_ratio=ratio,
        bootstrap_ci_low=ci_low,
        bootstrap_ci_high=ci_high,
        permutation_p_value=p_value,
        seed=args.seed,
    )
    per_mode = [
        {
            "dataset": dataset,
            "mode": str(idx + 1),
            "identity_variance_fraction": f"{fractions[idx]:.10g}",
            "icc1": f"{iccs[idx]:.10g}",
            "sigma2_video": f"{video_variances[idx]:.10g}",
            "sigma2_extract": f"{extract_variances[idx]:.10g}",
            "coefficient_variance": f"{mode_variances[idx]:.10g}",
        }
        for idx in range(len(fractions))
    ]
    return summary, per_mode


def write_summary(path: Path, rows: list[IdentitySummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(IdentitySummary.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: getattr(row, field) for field in fields})


def write_per_mode(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "mode", "identity_variance_fraction", "icc1", "sigma2_video", "sigma2_extract", "coefficient_variance"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_text_summary(path: Path, rows: list[IdentitySummary]) -> None:
    lines = [
        "Fish-identity dependence analysis",
        "=================================",
        "",
        "Fish identity was approximated by video/experiment identifier because no separate fish_id column is available in the current metadata.",
        "",
    ]
    for row in rows:
        lines.append(
            f"{row.dataset}: {row.n_samples} retained pressure segments from {row.n_groups} videos were projected onto {row.n_modes} 99% POD modes. "
            f"The variance-weighted ICC(1) was {row.weighted_icc1:.3f}; "
            f"the median per-mode ICC(1) was {row.median_icc1:.3f} "
            f"(range {row.min_icc1:.3f}-{row.max_icc1:.3f}, IQR {row.q1_icc1:.3f}-{row.q3_icc1:.3f}). "
            f"Median same-video and different-video distances were {row.median_same_group_distance:.3f} and {row.median_different_group_distance:.3f}, "
            f"respectively (difference {row.median_distance_difference:.3f}, 95% bootstrap CI {row.bootstrap_ci_low:.3f}-{row.bootstrap_ci_high:.3f}; "
            f"permutation p={row.permutation_p_value:.3f}; seed={row.seed})."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    repo_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="boai", help="Comma-separated dataset names.")
    parser.add_argument("--model", default="fish", choices=["fish", "cylinder"])
    parser.add_argument("--energy", type=float, default=0.99)
    parser.add_argument("--snapshot-dir", type=Path, default=repo_dir / "vortex_outputs/pressure_snapshots")
    parser.add_argument("--modes-dir", type=Path, default=repo_dir / "vortex_outputs/pressure_modes")
    parser.add_argument("--output-dir", type=Path, default=repo_dir / "vortex_outputs/revision_analysis")
    parser.add_argument("--group-column", default="experiment")
    parser.add_argument("--min-length", type=int, default=100)
    parser.add_argument("--min-samples-per-group", type=int, default=5)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--tail-frac", type=float, default=0.15)
    parser.add_argument("--tail-ratio", type=float, default=5.0)
    parser.add_argument("--tail-min", type=int, default=5)
    parser.add_argument("--no-tail-spike-filter", action="store_true")
    args = parser.parse_args()

    summaries: list[IdentitySummary] = []
    per_mode_rows: list[dict[str, str]] = []
    for dataset in parse_list(args.datasets):
        summary, per_mode = analyze_dataset(args, dataset)
        summaries.append(summary)
        per_mode_rows.extend(per_mode)
        print(
            f"{dataset}: n={summary.n_samples}, videos={summary.n_groups}, modes={summary.n_modes}, "
            f"weighted_icc1={summary.weighted_icc1:.4f}, "
            f"distance_delta={summary.median_distance_difference:.4f}, p={summary.permutation_p_value:.4f}"
        )

    suffix = f"_{args.model}_{int(round(args.energy * 100 if args.energy <= 1.0 else args.energy))}"
    write_summary(args.output_dir / f"fish_identity_dependence_summary{suffix}.csv", summaries)
    write_per_mode(args.output_dir / f"fish_identity_dependence_per_mode{suffix}.csv", per_mode_rows)
    write_text_summary(args.output_dir / f"fish_identity_dependence_summary{suffix}.txt", summaries)


if __name__ == "__main__":
    main()
