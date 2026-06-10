#!/usr/bin/env python3
"""Grouped-by-video validation helper for pressure-signal features.

The default behaviour is conservative: inspect the fish-identity dependence
summary and skip grouped validation unless the predefined trigger is met. When
triggered, the script trains no neural model. It projects existing pressure
segments into the existing POD basis and performs a deterministic grouped
nearest-centroid validation by video/experiment.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from pathlib import Path

from analyze_fish_identity_dependence import (
    load_modes,
    load_scalar_int,
    load_segments,
    parse_list,
    project_segments,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return math.nan, math.nan
    phat = successes / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def should_run(identity_summary: Path, dataset: str, fraction_threshold: float, p_threshold: float) -> tuple[bool, str]:
    rows = [row for row in read_csv(identity_summary) if row.get("dataset") == dataset]
    if not rows:
        return False, f"No identity summary found for {dataset}; grouped validation not run."
    row = rows[0]
    weighted = safe_float(row.get("weighted_icc1", row.get("weighted_identity_fraction", "nan")))
    p_value = safe_float(row.get("permutation_p_value", "nan"))
    if math.isfinite(weighted) and weighted >= fraction_threshold:
        return True, f"Triggered because weighted ICC(1) {weighted:.3f} >= {fraction_threshold:.3f}."
    if math.isfinite(p_value) and p_value < p_threshold:
        return True, f"Triggered because permutation p-value {p_value:.3f} < {p_threshold:.3f}."
    return False, (
        f"Not triggered: weighted ICC(1) {weighted:.3f}, permutation p-value {p_value:.3f}; "
        f"thresholds are {fraction_threshold:.3f} and {p_threshold:.3f}."
    )


def vector_mean(rows: list[list[float]]) -> list[float]:
    if not rows:
        return []
    n_features = len(rows[0])
    return [statistics.fmean(row[idx] for row in rows) for idx in range(n_features)]


def distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def grouped_nearest_centroid(
    features: list[list[float]],
    labels: list[int],
    groups: list[str],
    seed: int,
    repeats: int,
    test_group_fraction: float,
) -> list[dict[str, object]]:
    rng = random.Random(seed)
    unique_groups = sorted(set(groups))
    n_test_groups = max(1, round(len(unique_groups) * test_group_fraction))
    rows: list[dict[str, object]] = []
    for repeat in range(1, repeats + 1):
        test_groups = set(rng.sample(unique_groups, n_test_groups))
        train_idx = [idx for idx, group in enumerate(groups) if group not in test_groups]
        test_idx = [idx for idx, group in enumerate(groups) if group in test_groups]
        if not train_idx or not test_idx:
            continue
        train_fish = [features[idx] for idx in train_idx if labels[idx] == 0]
        train_cylinder = [features[idx] for idx in train_idx if labels[idx] == 1]
        if not train_fish or not train_cylinder:
            continue
        fish_centroid = vector_mean(train_fish)
        cylinder_centroid = vector_mean(train_cylinder)
        correct = 0
        for idx in test_idx:
            pred = 0 if distance(features[idx], fish_centroid) <= distance(features[idx], cylinder_centroid) else 1
            correct += int(pred == labels[idx])
        low, high = wilson_interval(correct, len(test_idx))
        rows.append(
            {
                "repeat": repeat,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "n_test_groups": len(test_groups),
                "accuracy": correct / len(test_idx),
                "correct": correct,
                "ci95_low": low,
                "ci95_high": high,
                "test_groups": ";".join(sorted(test_groups)),
            }
        )
    return rows


def build_dataset_features(args: argparse.Namespace, dataset: str) -> tuple[list[list[float]], list[int], list[str]]:
    mode_tag = int(round(args.energy * 100 if args.energy <= 1.0 else args.energy))
    modes_path = args.modes_dir / f"pressure_modes_{dataset}_{mode_tag}.npz"
    modes, _ = load_modes(modes_path)
    target_length = load_scalar_int(modes_path, "target_length.npy", len(modes))

    by_class: list[tuple[list[list[float]], list[int], list[str]]] = []
    for model, label in (("fish", 0), ("cylinder", 1)):
        segments = load_segments(
            matrix_path=args.snapshot_dir / f"pressure_snapshots_{dataset}_{model}.csv",
            meta_path=args.snapshot_dir / f"pressure_snapshots_{dataset}_{model}_meta.csv",
            target_length=target_length,
            group_column=args.group_column,
            min_length=args.min_length,
            min_samples_per_group=1,
            tail_spike_filter=not args.no_tail_spike_filter,
            tail_fraction=args.tail_frac,
            tail_ratio=args.tail_ratio,
            tail_min=args.tail_min,
        )
        projected = project_segments(segments, modes)
        labels = [label] * len(projected)
        groups = [segment.group for segment in segments]
        by_class.append((projected, labels, groups))

    if args.balance_classes and len(by_class) == 2:
        target = min(len(by_class[0][0]), len(by_class[1][0]))
        by_class = [(features[:target], labels[:target], groups[:target]) for features, labels, groups in by_class]

    features: list[list[float]] = []
    labels: list[int] = []
    groups: list[str] = []
    for class_features, class_labels, class_groups in by_class:
        features.extend(class_features)
        labels.extend(class_labels)
        groups.extend(class_groups)
    return features, labels, groups


def write_skip_note(path: Path, note: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(note + "\n", encoding="utf-8")


def main() -> None:
    repo_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="boai")
    parser.add_argument("--energy", type=float, default=0.99)
    parser.add_argument("--snapshot-dir", type=Path, default=repo_dir / "vortex_outputs/pressure_snapshots")
    parser.add_argument("--modes-dir", type=Path, default=repo_dir / "vortex_outputs/pressure_modes")
    parser.add_argument("--output-dir", type=Path, default=repo_dir / "vortex_outputs/revision_analysis")
    parser.add_argument(
        "--identity-summary",
        type=Path,
        default=repo_dir / "vortex_outputs/revision_analysis/fish_identity_dependence_summary_fish_99.csv",
    )
    parser.add_argument("--group-column", default="experiment")
    parser.add_argument("--dependency-fraction-threshold", type=float, default=0.10)
    parser.add_argument("--dependency-p-threshold", type=float, default=0.05)
    parser.add_argument("--force", action="store_true", help="Run even if identity-dependence trigger is not met.")
    parser.add_argument("--min-length", type=int, default=100)
    parser.add_argument("--no-balance-classes", dest="balance_classes", action="store_false")
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--test-group-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--tail-frac", type=float, default=0.15)
    parser.add_argument("--tail-ratio", type=float, default=5.0)
    parser.add_argument("--tail-min", type=int, default=5)
    parser.add_argument("--no-tail-spike-filter", action="store_true")
    parser.set_defaults(balance_classes=True)
    args = parser.parse_args()

    all_rows: list[dict[str, object]] = []
    notes: list[str] = []
    for dataset in parse_list(args.datasets):
        triggered, reason = should_run(
            args.identity_summary,
            dataset,
            args.dependency_fraction_threshold,
            args.dependency_p_threshold,
        )
        if not triggered and not args.force:
            note = f"{dataset}: grouped validation skipped. {reason}"
            notes.append(note)
            print(note)
            continue
        print(f"{dataset}: grouped validation running. {reason}")
        features, labels, groups = build_dataset_features(args, dataset)
        rows = grouped_nearest_centroid(
            features,
            labels,
            groups,
            seed=args.seed,
            repeats=args.repeats,
            test_group_fraction=args.test_group_fraction,
        )
        for row in rows:
            row["dataset"] = dataset
        all_rows.extend(rows)

    if all_rows:
        write_csv(
            args.output_dir / "grouped_by_video_validation.csv",
            all_rows,
            [
                "dataset",
                "repeat",
                "n_train",
                "n_test",
                "n_test_groups",
                "accuracy",
                "correct",
                "ci95_low",
                "ci95_high",
                "test_groups",
            ],
        )
        by_dataset: dict[str, list[float]] = {}
        for row in all_rows:
            by_dataset.setdefault(str(row["dataset"]), []).append(float(row["accuracy"]))
        balance_text = "class-balanced grouped" if args.balance_classes else "grouped"
        notes.extend(
            f"{dataset}: {balance_text} nearest-centroid validation mean accuracy {statistics.fmean(values):.3f} across {len(values)} repeats."
            for dataset, values in sorted(by_dataset.items())
        )
    write_skip_note(args.output_dir / "grouped_by_video_validation_note.txt", "\n".join(notes))


if __name__ == "__main__":
    main()
