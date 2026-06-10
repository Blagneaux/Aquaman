#!/usr/bin/env python3
"""Create reviewer-facing uncertainty summaries from existing CSV outputs.

No training or inference is rerun. The script reads the already-generated
classifier summaries and real-pressure prediction files and writes Wilson-score
confidence intervals for accuracy and classification proportions.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def parse_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def safe_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def safe_int(value: str) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return math.nan, math.nan
    phat = successes / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


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
        for row in rows:
            writer.writerow(row)


def split_sizes(n: int, train_frac: float = 0.7, val_frac: float = 0.15) -> tuple[int, int, int]:
    if n <= 2:
        return n, 0, 0
    train = round(n * train_frac)
    val = round(n * val_frac)
    train = max(1, min(train, n - 2))
    val = max(1, min(val, n - train - 1))
    test = n - train - val
    return train, val, test


def pipeline_count_lookup(path: Path) -> dict[str, int]:
    rows = read_csv(path)
    by_dataset: dict[str, int] = {}
    for row in rows:
        dataset = row.get("dataset", "")
        balanced = safe_int(row.get("snapshots_balanced", "0"))
        if dataset:
            by_dataset[dataset] = by_dataset.get(dataset, 0) + balanced
    return by_dataset


def summarize_pressure_classifier(modes_dir: Path, energies: list[str]) -> list[dict[str, object]]:
    rows_out: list[dict[str, object]] = []
    for energy in energies:
        counts = pipeline_count_lookup(modes_dir / f"pressure_pipeline_counts_{energy}.csv")
        seen_datasets: set[str] = set()
        for row in read_csv(modes_dir / f"classifier_results_summary_{energy}.csv"):
            dataset = row.get("dataset", "")
            if not dataset or dataset in seen_datasets:
                continue
            seen_datasets.add(dataset)
            n_total = counts.get(dataset, 0)
            train_n, val_n, test_n = split_sizes(n_total)
            for split, denom in (("train", train_n), ("validation", val_n), ("test", test_n)):
                acc = safe_float(row.get(f"{'val' if split == 'validation' else split}_acc", "nan"))
                successes = round(acc * denom) if math.isfinite(acc) else 0
                low, high = wilson_interval(successes, denom)
                rows_out.append(
                    {
                        "classifier": "pressure_signal",
                        "dataset": dataset,
                        "energy": energy,
                        "split": split,
                        "n": denom,
                        "estimate": f"{acc:.6g}" if math.isfinite(acc) else "",
                        "successes": successes,
                        "ci95_low": f"{low:.6g}" if math.isfinite(low) else "",
                        "ci95_high": f"{high:.6g}" if math.isfinite(high) else "",
                    }
                )
    return rows_out


def summarize_real_predictions(
    modes_dir: Path,
    energies: list[str],
    datasets: list[str],
    thresholds: list[float],
) -> list[dict[str, object]]:
    rows_out: list[dict[str, object]] = []
    for energy in energies:
        for dataset in datasets:
            path = modes_dir / f"real_pressure_predictions_linear_{dataset}_{energy}.csv"
            rows = read_csv(path)
            probs = [safe_float(row.get("prob", "")) for row in rows]
            probs = [min(1.0, max(0.0, prob)) for prob in probs if math.isfinite(prob)]
            n = len(probs)
            for threshold in thresholds:
                counts = {
                    "fish_like": sum(prob <= 1.0 - threshold for prob in probs),
                    "cylinder_like": sum(prob >= threshold for prob in probs),
                }
                counts["undetermined"] = n - counts["fish_like"] - counts["cylinder_like"]
                for label, count in counts.items():
                    estimate = count / n if n else math.nan
                    low, high = wilson_interval(count, n)
                    rows_out.append(
                        {
                            "dataset": dataset,
                            "energy": energy,
                            "threshold": f"{threshold:.2f}",
                            "class": label,
                            "n": n,
                            "count": count,
                            "estimate": f"{estimate:.6g}" if math.isfinite(estimate) else "",
                            "ci95_low": f"{low:.6g}" if math.isfinite(low) else "",
                            "ci95_high": f"{high:.6g}" if math.isfinite(high) else "",
                        }
                    )
    return rows_out


def summarize_vortex_classifier(results_dir: Path, summary_name: str) -> list[dict[str, object]]:
    rows_out: list[dict[str, object]] = []
    for row in read_csv(results_dir / summary_name):
        dataset = row.get("dataset", "")
        n_total = safe_int(row.get("n_samples", "0"))
        train_n, val_n, test_n = split_sizes(n_total)
        for metric, split, denom in (
            ("train_acc", "train", train_n),
            ("val_acc", "validation", val_n),
            ("test_acc", "test", test_n),
            ("test_f1", "test_f1", test_n),
        ):
            estimate = safe_float(row.get(metric, "nan"))
            successes = round(estimate * denom) if math.isfinite(estimate) else 0
            low, high = wilson_interval(successes, denom)
            rows_out.append(
                {
                    "classifier": "vortex_feature",
                    "dataset": dataset,
                    "metric": metric,
                    "split": split,
                    "n": denom,
                    "estimate": f"{estimate:.6g}" if math.isfinite(estimate) else "",
                    "successes": successes,
                    "ci95_low": f"{low:.6g}" if math.isfinite(low) else "",
                    "ci95_high": f"{high:.6g}" if math.isfinite(high) else "",
                }
            )
    return rows_out


def write_text_summary(path: Path, pressure_rows: list[dict[str, object]], real_rows: list[dict[str, object]]) -> None:
    lines = [
        "Revision uncertainty summary",
        "============================",
        "",
        "Confidence intervals are Wilson 95% intervals computed from existing CSV outputs; no training or inference was rerun.",
        "",
        "Pressure classifier test accuracy, 95% retained variance:",
    ]
    for row in pressure_rows:
        if row["energy"] == "95" and row["split"] == "test":
            lines.append(
                f"- {row['dataset']}: {float(row['estimate']):.3f} "
                f"(95% CI {float(row['ci95_low']):.3f}-{float(row['ci95_high']):.3f}, n={row['n']})."
            )
    lines.append("")
    lines.append("Real-pressure classification proportions, 95% retained variance, threshold 0.99:")
    for row in real_rows:
        if row["energy"] == "95" and row["threshold"] == "0.99" and row["class"] in {"fish_like", "cylinder_like"}:
            lines.append(
                f"- {row['dataset']} {row['class']}: {float(row['estimate']):.3f} "
                f"(95% CI {float(row['ci95_low']):.3f}-{float(row['ci95_high']):.3f}, count={row['count']}/{row['n']})."
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    repo_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes-dir", type=Path, default=repo_dir / "vortex_outputs/pressure_modes")
    parser.add_argument("--vortex-dir", type=Path, default=repo_dir / "vortex_outputs/vortex_classifier")
    parser.add_argument("--output-dir", type=Path, default=repo_dir / "vortex_outputs/revision_analysis")
    parser.add_argument("--datasets", default="nadia,thomas,boai,full")
    parser.add_argument("--energies", default="75,95,99")
    parser.add_argument("--thresholds", default="0.95,0.99")
    parser.add_argument("--vortex-summary", default="vortex_classifier_results_summary_95_thr0.99.csv")
    args = parser.parse_args()

    energies = parse_list(args.energies)
    datasets = parse_list(args.datasets)
    thresholds = [float(value) for value in parse_list(args.thresholds)]

    pressure_rows = summarize_pressure_classifier(args.modes_dir, energies)
    real_rows = summarize_real_predictions(args.modes_dir, energies, datasets, thresholds)
    vortex_rows = summarize_vortex_classifier(args.vortex_dir, args.vortex_summary)

    write_csv(
        args.output_dir / "classifier_accuracy_wilson_ci.csv",
        pressure_rows,
        ["classifier", "dataset", "energy", "split", "n", "estimate", "successes", "ci95_low", "ci95_high"],
    )
    write_csv(
        args.output_dir / "real_pressure_classification_proportion_ci.csv",
        real_rows,
        ["dataset", "energy", "threshold", "class", "n", "count", "estimate", "ci95_low", "ci95_high"],
    )
    write_csv(
        args.output_dir / "vortex_classifier_wilson_ci.csv",
        vortex_rows,
        ["classifier", "dataset", "metric", "split", "n", "estimate", "successes", "ci95_low", "ci95_high"],
    )
    write_text_summary(args.output_dir / "revision_uncertainty_summary.txt", pressure_rows, real_rows)
    print(f"Wrote uncertainty summaries to {args.output_dir}")


if __name__ == "__main__":
    main()
