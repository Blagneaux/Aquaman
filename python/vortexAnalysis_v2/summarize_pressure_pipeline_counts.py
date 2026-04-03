import argparse
import csv
from pathlib import Path
from typing import Dict, Tuple
import sys
import numpy as np

from compute_pressure_modes import (
    balance_series,
    extract_series,
    filter_tail_spikes,
    find_modes_file,
    format_energy_tag,
    load_lengths,
    load_matrix,
)


def parse_list(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def count_simulations(meta_path: Path) -> int:
    sims: set[Tuple[str, str]] = set()
    with meta_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            exp = str(row.get("experiment", "")).strip()
            snap = str(row.get("snapshot", "")).strip()
            sims.add((exp, snap))
    return len(sims)


def load_modes_info(modes_dir: Path, dataset: str, energy: float | None) -> Tuple[str, str]:
    if energy is None:
        return "", ""
    modes_path = find_modes_file(modes_dir, dataset, energy)
    if modes_path is None or not modes_path.exists():
        return "", ""
    data = np.load(modes_path)
    modes = data.get("modes")
    if modes is None:
        return "", ""
    return str(modes.shape[0]), str(modes.shape[1])


def count_labeled_vortices(path: Path) -> Dict[str, int]:
    if not path.exists():
        return {}
    per_model: Dict[str, set[Tuple[str, ...]]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = str(row.get("model", "")).strip()
            if not model:
                continue
            track_id = row.get("track_id", row.get("track", row.get("id", "")))
            key = (
                str(row.get("dataset", "")),
                str(row.get("experiment", "")),
                str(row.get("snapshot", "")),
                str(row.get("sensor_id", "")),
                str(track_id),
                str(row.get("start_frame", "")),
                str(row.get("end_frame", "")),
                str(row.get("track_path", "")),
            )
            per_model.setdefault(model, set()).add(key)
    return {model: len(keys) for model, keys in per_model.items()}


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Summarize snapshot counts at each stage of the pressure-classifier pipeline."
        )
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=repo_root / "python/vortexAnalysis_v2/vortex_outputs/pressure_snapshots",
        help="Directory with pressure snapshot matrices and meta CSVs",
    )
    parser.add_argument(
        "--modes-dir",
        type=Path,
        default=repo_root / "python/vortexAnalysis_v2/vortex_outputs/pressure_modes",
        help="Directory with pressure mode NPZ files",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="nadia,thomas,boai,full",
        help="Comma-separated dataset list",
    )
    parser.add_argument(
        "--modes-energy",
        type=float,
        default=None,
        help="Energy tag for modes (e.g. 0.95 or 95).",
    )
    parser.add_argument(
        "--matching-dir",
        type=Path,
        default=repo_root / "python/vortexAnalysis_v2/vortex_outputs/matching_vortices",
        help="Directory containing matching_vortices CSVs.",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="linear",
        help="Architecture tag used in matching_vortices filenames.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.95,
        help="Confidence threshold used for matching_vortices filenames.",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=100,
        help="Minimum snapshot length to keep (matches training default).",
    )
    parser.add_argument(
        "--spike-ratio",
        type=float,
        default=5.0,
        help="Tail spike ratio threshold (matches training default).",
    )
    parser.add_argument(
        "--spike-tail-frac",
        type=float,
        default=0.15,
        help="Tail fraction for spike detection (matches training default).",
    )
    parser.add_argument(
        "--spike-min-tail",
        type=int,
        default=5,
        help="Minimum tail length for spike detection (matches training default).",
    )
    parser.add_argument(
        "--no-tail-spike-filter",
        dest="tail_spike_filter",
        action="store_false",
        help="Disable tail spike filtering.",
    )
    parser.set_defaults(tail_spike_filter=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV path to write results.",
    )
    args = parser.parse_args()

    datasets = parse_list(args.datasets)
    models = ["fish", "cylinder"]
    tag = f"_{format_energy_tag(args.modes_energy)}" if args.modes_energy is not None else ""
    threshold_tag = f"thr{args.confidence_threshold:.2f}"

    rows: list[Dict[str, str]] = []
    for dataset in datasets:
        mode_length, num_modes = load_modes_info(args.modes_dir, dataset, args.modes_energy)
        matching_path = (
            args.matching_dir
            / f"matching_vortices_{args.architecture}_{dataset}{tag}_{threshold_tag}.csv"
        )
        labeled_counts = count_labeled_vortices(matching_path)
        stage_data: Dict[str, Dict[str, object]] = {}

        for model in models:
            meta_path = args.snapshot_dir / f"pressure_snapshots_{dataset}_{model}_meta.csv"
            matrix_path = args.snapshot_dir / f"pressure_snapshots_{dataset}_{model}.csv"
            if not meta_path.exists() or not matrix_path.exists():
                continue

            sim_count = count_simulations(meta_path)
            lengths = load_lengths(meta_path)
            matrix = load_matrix(matrix_path)
            total_snapshots = int(matrix.shape[1])
            series = extract_series(matrix, lengths, args.min_length, f"{dataset} {model}")
            after_min = len(series)
            if args.tail_spike_filter:
                series, _ = filter_tail_spikes(
                    series,
                    args.spike_tail_frac,
                    args.spike_ratio,
                    args.spike_min_tail,
                    f"{dataset} {model}",
                )
            after_tail = len(series)
            stage_data[model] = {
                "simulations": sim_count,
                "snapshots_total": total_snapshots,
                "snapshots_min_length": after_min,
                "snapshots_tail_filtered": after_tail,
                "series": series,
            }

        if "fish" in stage_data and "cylinder" in stage_data:
            fish_bal, cyl_bal = balance_series(
                stage_data["fish"]["series"],
                stage_data["cylinder"]["series"],
            )
            balanced_counts = {"fish": len(fish_bal), "cylinder": len(cyl_bal)}
        else:
            balanced_counts = {"fish": "", "cylinder": ""}

        for model in models:
            if model not in stage_data:
                continue
            row = {
                "dataset": dataset,
                "model": model,
                "simulations": str(stage_data[model]["simulations"]),
                "snapshots_total": str(stage_data[model]["snapshots_total"]),
                "snapshots_min_length": str(stage_data[model]["snapshots_min_length"]),
                "snapshots_tail_filtered": str(stage_data[model]["snapshots_tail_filtered"]),
                "snapshots_balanced": str(balanced_counts.get(model, "")),
                "labeled_vortices": str(labeled_counts.get(model, "")),
                "mode_length": mode_length,
                "num_modes": num_modes,
            }
            rows.append(row)

    fieldnames = [
        "dataset",
        "model",
        "simulations",
        "snapshots_total",
        "snapshots_min_length",
        "snapshots_tail_filtered",
        "snapshots_balanced",
        "labeled_vortices",
        "mode_length",
        "num_modes",
    ]

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
