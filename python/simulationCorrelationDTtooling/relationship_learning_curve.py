"""
Learning curve for automatic wall-pressure relationship discovery.

This script uses the feature table produced by discover_asmr_metrics.py and
asks: how many simulation cases are needed before the discovered classic
relationship has good cross-validated predictive value?

By default, it first discovers a library of good classic one-feature
relationships on the full feature table. For each sample count n, it repeatedly
subsamples n cases, refits those candidate relationship forms, and records the
best cross-validated R2. This keeps the learning-curve question focused on
"how many samples are needed for a relationship form to become predictive"
without paying the cost of a full transform search for every subsample.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_discovery_module():
    module_path = Path(__file__).with_name("discover_asmr_metrics.py")
    spec = importlib.util.spec_from_file_location("discover_asmr_metrics_local", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def numeric_feature_columns(discovery, df: pd.DataFrame, parameters: list[str]) -> list[str]:
    excluded = set(["file", *parameters])
    return discovery.numeric_feature_columns(df, excluded)


def select_candidate_features(
    discovery,
    subset: pd.DataFrame,
    all_feature_columns: list[str],
    parameter: str,
    max_features: int,
    cv_folds: int,
) -> list[str]:
    target_columns = discovery.feature_columns_for_target(all_feature_columns, parameter)
    if len(target_columns) <= max_features:
        return target_columns

    ranking = discovery.score_feature_table(
        subset,
        target_columns,
        subset[parameter].to_numpy(dtype=float),
        cv_folds=cv_folds,
    )
    if ranking.empty:
        return target_columns[:max_features]
    return ranking["feature"].head(max_features).tolist()


def discover_candidate_relationships(
    discovery,
    features: pd.DataFrame,
    all_feature_columns: list[str],
    parameters: list[str],
    args: argparse.Namespace,
) -> dict[str, list]:
    candidates = {}
    for parameter in parameters:
        candidate_features = select_candidate_features(
            discovery,
            features,
            all_feature_columns,
            parameter,
            max_features=args.max_classic_features,
            cv_folds=args.cv_folds,
        )
        results = discovery.find_classic_relationships(
            feature_df=features,
            feature_columns=candidate_features,
            parameter=parameter,
            cv_folds=args.cv_folds,
            max_results=args.candidate_relationships,
        )
        candidates[parameter] = results
        print(
            f"{parameter}: using {len(results)} fixed candidate relationships "
            f"from {len(candidate_features)} prefiltered features"
        )
    return candidates


def evaluate_candidate_relationships(
    discovery,
    subset: pd.DataFrame,
    parameter: str,
    candidates: list,
    cv_folds: int,
):
    if not candidates:
        return None

    feature_names = sorted({candidate.feature for candidate in candidates})
    clean_features = discovery.cleaned_matrix(subset, feature_names)
    parameter_values = subset[parameter].to_numpy(dtype=float)
    evaluated = []
    for candidate in candidates:
        result = discovery.evaluate_classic_relationship(
            parameter=parameter,
            feature=candidate.feature,
            parameter_values=parameter_values,
            feature_values=clean_features[candidate.feature].to_numpy(dtype=float),
            target_transform=candidate.target_transform,
            feature_transform=candidate.feature_transform,
            degree=candidate.degree,
            cv_folds=cv_folds,
        )
        if result is not None:
            evaluated.append(result)

    if not evaluated:
        return None

    def sort_key(result) -> tuple[float, float]:
        cv_score = result.cv_r2_mean if result.cv_r2_mean is not None else -np.inf
        return cv_score, result.in_sample_r2

    evaluated.sort(key=sort_key, reverse=True)
    return evaluated[0]


def run_learning_curve(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    discovery = load_discovery_module()
    features = pd.read_csv(args.features_csv)
    parameters = [item.strip() for item in args.parameters.split(",") if item.strip()]
    missing = [parameter for parameter in parameters if parameter not in features.columns]
    if missing:
        raise ValueError(f"Missing parameter columns in feature table: {missing}")

    all_feature_columns = numeric_feature_columns(discovery, features, parameters)
    rng = np.random.default_rng(args.seed)
    max_samples = min(args.max_samples or len(features), len(features))
    fixed_candidates = None
    if args.search_mode == "fixed-candidates":
        fixed_candidates = discover_candidate_relationships(
            discovery,
            features,
            all_feature_columns,
            parameters,
            args,
        )

    detail_rows = []
    for n_samples in range(1, max_samples + 1):
        if args.progress_every and (
            n_samples == 1
            or n_samples == max_samples
            or n_samples % args.progress_every == 0
        ):
            print(f"Evaluating {n_samples}/{max_samples} samples")
        n_repeats = 1 if n_samples == len(features) else args.repeats
        for repeat in range(n_repeats):
            if n_samples < args.min_cv_samples:
                for parameter in parameters:
                    detail_rows.append(
                        {
                            "sample_count": n_samples,
                            "repeat": repeat,
                            "parameter": parameter,
                            "cv_r2": np.nan,
                            "in_sample_r2": np.nan,
                            "expression": "",
                            "feature": "",
                            "target_transform": "",
                            "feature_transform": "",
                            "degree": np.nan,
                            "search_mode": args.search_mode,
                        }
                    )
                continue

            indices = rng.choice(len(features), size=n_samples, replace=False)
            subset = features.iloc[indices].reset_index(drop=True)
            for parameter in parameters:
                if fixed_candidates is None:
                    candidate_features = select_candidate_features(
                        discovery,
                        subset,
                        all_feature_columns,
                        parameter,
                        max_features=args.max_classic_features,
                        cv_folds=args.cv_folds,
                    )
                    results = discovery.find_classic_relationships(
                        feature_df=subset,
                        feature_columns=candidate_features,
                        parameter=parameter,
                        cv_folds=args.cv_folds,
                        max_results=1,
                    )
                    best = results[0] if results else None
                    library_size = len(candidate_features)
                else:
                    best = evaluate_candidate_relationships(
                        discovery,
                        subset,
                        parameter,
                        fixed_candidates[parameter],
                        args.cv_folds,
                    )
                    library_size = len(fixed_candidates[parameter])

                if best is not None:
                    cv_r2 = best.cv_r2_mean
                    cv_r2_std = best.cv_r2_std
                    detail_rows.append(
                        {
                            "sample_count": n_samples,
                            "repeat": repeat,
                            "parameter": parameter,
                            "cv_r2": cv_r2,
                            "cv_r2_fold_std": cv_r2_std,
                            "in_sample_r2": best.in_sample_r2,
                            "expression": best.expression,
                            "feature": best.feature,
                            "target_transform": best.target_transform,
                            "feature_transform": best.feature_transform,
                            "degree": best.degree,
                            "search_mode": args.search_mode,
                            "candidate_library_size": library_size,
                        }
                    )
                else:
                    detail_rows.append(
                        {
                            "sample_count": n_samples,
                            "repeat": repeat,
                            "parameter": parameter,
                            "cv_r2": np.nan,
                            "cv_r2_fold_std": np.nan,
                            "in_sample_r2": np.nan,
                            "expression": "",
                            "feature": "",
                            "target_transform": "",
                            "feature_transform": "",
                            "degree": np.nan,
                            "search_mode": args.search_mode,
                            "candidate_library_size": library_size,
                        }
                    )

    detail = pd.DataFrame(detail_rows)
    summary = (
        detail.groupby(["parameter", "sample_count"], as_index=False)
        .agg(
            cv_r2_mean=("cv_r2", "mean"),
            cv_r2_std=("cv_r2", "std"),
            cv_r2_median=("cv_r2", "median"),
            cv_r2_q25=("cv_r2", lambda x: x.quantile(0.25)),
            cv_r2_q75=("cv_r2", lambda x: x.quantile(0.75)),
            valid_repeats=("cv_r2", lambda x: int(x.notna().sum())),
        )
        .sort_values(["parameter", "sample_count"])
    )
    return detail, summary


def first_threshold_sample(summary: pd.DataFrame, parameter: str, threshold: float) -> int | None:
    sub = summary[
        (summary["parameter"] == parameter)
        & (summary["cv_r2_mean"].notna())
        & (summary["cv_r2_mean"] >= threshold)
    ]
    if sub.empty:
        return None
    return int(sub["sample_count"].iloc[0])


def plot_learning_curve(summary: pd.DataFrame, output_path: Path, thresholds: list[float]) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    colors = {"Re": "tab:blue", "h": "tab:orange"}
    plotted_values = []

    for parameter, group in summary.groupby("parameter"):
        group = group.sort_values("sample_count")
        x = group["sample_count"].to_numpy(dtype=float)
        mean = group["cv_r2_mean"].to_numpy(dtype=float)
        q25 = group["cv_r2_q25"].to_numpy(dtype=float)
        q75 = group["cv_r2_q75"].to_numpy(dtype=float)
        plotted_values.extend(mean[np.isfinite(mean)])
        plotted_values.extend(q25[np.isfinite(q25)])
        plotted_values.extend(q75[np.isfinite(q75)])
        color = colors.get(parameter, None)
        ax.plot(x, mean, label=parameter, color=color, linewidth=2.0)
        ax.fill_between(x, q25, q75, color=color, alpha=0.18, linewidth=0)

    for threshold in thresholds:
        ax.axhline(threshold, color="0.35", linestyle="--", linewidth=0.9)
        ax.text(
            1.0,
            threshold + 0.002,
            f"R2={threshold:g}",
            color="0.25",
            fontsize=9,
            va="bottom",
        )

    ax.set_xlabel("Number of simulation cases used")
    ax.set_ylabel("Best classic relationship CV R2")
    ax.set_title("Wall-pressure relationship discovery learning curve")
    ax.set_xlim(left=1)
    finite_values = np.asarray([*plotted_values, *thresholds], dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size:
        ax.set_ylim(
            max(-0.1, float(np.min(finite_values)) - 0.02),
            min(1.05, max(1.0, float(np.max(finite_values)) + 0.02)),
        )
    else:
        ax.set_ylim(-0.1, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(title="Parameter")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_threshold_report(
    summary: pd.DataFrame,
    output_path: Path,
    parameters: list[str],
    thresholds: list[float],
) -> None:
    lines = ["# Relationship Discovery Learning Curve", ""]
    for parameter in parameters:
        lines.append(f"## {parameter}")
        for threshold in thresholds:
            sample_count = first_threshold_sample(summary, parameter, threshold)
            if sample_count is None:
                lines.append(f"- Mean CV R2 never reaches {threshold:g}.")
            else:
                lines.append(f"- First mean CV R2 >= {threshold:g}: {sample_count} samples.")
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build learning curves for automatic relationship discovery."
    )
    parser.add_argument(
        "--features-csv",
        type=Path,
        default=Path("asmr_metric_discovery_china/case_features.csv"),
        help="case_features.csv produced by discover_asmr_metrics.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("asmr_metric_discovery_china/learning_curve"),
        help="Directory for learning-curve CSV/PNG outputs.",
    )
    parser.add_argument(
        "--parameters",
        default="Re,h",
        help="Comma-separated parameter columns to evaluate.",
    )
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--min-cv-samples", type=int, default=6)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--max-classic-features",
        type=int,
        default=35,
        help="Feature prefilter count before the classic transform sweep.",
    )
    parser.add_argument(
        "--search-mode",
        choices=["fixed-candidates", "exhaustive"],
        default="fixed-candidates",
        help=(
            "fixed-candidates discovers good relationship forms once on the "
            "full table, then refits them on each subsample. exhaustive reruns "
            "the full transform search on every subsample."
        ),
    )
    parser.add_argument(
        "--candidate-relationships",
        type=int,
        default=60,
        help="Number of full-table classic relationships kept per parameter in fixed-candidates mode.",
    )
    parser.add_argument(
        "--thresholds",
        default="0.8,0.9,0.95,0.99",
        help="Comma-separated CV R2 thresholds reported in Markdown.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N sample counts. Use 0 to disable.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parameters = [item.strip() for item in args.parameters.split(",") if item.strip()]
    thresholds = [float(item) for item in args.thresholds.split(",") if item.strip()]

    detail, summary = run_learning_curve(args)
    detail.to_csv(args.output_dir / "relationship_learning_curve_detail.csv", index=False)
    summary.to_csv(args.output_dir / "relationship_learning_curve_summary.csv", index=False)
    plot_learning_curve(summary, args.output_dir / "relationship_learning_curve.png", thresholds)
    write_threshold_report(
        summary,
        args.output_dir / "relationship_learning_curve_report.md",
        parameters,
        thresholds,
    )

    print(f"Wrote {args.output_dir / 'relationship_learning_curve_summary.csv'}")
    print(f"Wrote {args.output_dir / 'relationship_learning_curve.png'}")
    for parameter in parameters:
        threshold_text = []
        for threshold in thresholds:
            sample_count = first_threshold_sample(summary, parameter, threshold)
            threshold_text.append(f"{threshold:g}: {sample_count}")
        print(f"{parameter} thresholds -> " + ", ".join(threshold_text))


if __name__ == "__main__":
    main()
