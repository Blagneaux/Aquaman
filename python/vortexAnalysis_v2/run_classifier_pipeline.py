import argparse
import subprocess
import sys
from pathlib import Path

from compute_pressure_modes import format_energy_tag


def parse_list(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def run_command(cmd: list[str], dry_run: bool) -> None:
    print("$", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Run modes -> train -> real-pressure test for multiple energy targets."
    )
    parser.add_argument(
        "--energies",
        type=str,
        default="0.99,0.95",
        help="Comma-separated energy targets (e.g. 0.99,0.95 or 99,95)",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="nadia,thomas,boai,full",
        help="Comma-separated dataset list",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default="0.75",
        help="Comma-separated confidence thresholds for real-pressure test",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=30,
        help="Optuna trials for training",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Optuna timeout in seconds",
    )
    parser.add_argument(
        "--loss",
        type=str,
        choices=["bce", "mse"],
        default="mse",
        help="Loss function for training/testing",
    )
    parser.add_argument(
        "--skip-optuna",
        action="store_true",
        help="Skip Optuna and use fixed params in training script",
    )
    parser.add_argument(
        "--fixed-params",
        type=str,
        default=None,
        help="Fixed params for training if --skip-optuna",
    )
    parser.add_argument(
        "--plot-distribution",
        action="store_true",
        help="Plot distribution for each energy/threshold run",
    )
    parser.add_argument(
        "--plot-save-dir",
        type=Path,
        default=repo_root / "python/vortexAnalysis_v2/vortex_outputs/pressure_modes",
        help="Directory to save distribution plots",
    )
    parser.add_argument(
        "--plot-from-files",
        action="store_true",
        help="Only generate plots from existing prediction CSVs",
    )
    parser.add_argument(
        "--plot-threshold-curves",
        action="store_true",
        help="Plot threshold curves (single figure per energy).",
    )
    parser.add_argument(
        "--dataset-roots",
        type=str,
        default=None,
        help="Override dataset roots: 'nadia=D:/crop_nadia;thomas=D:/crop_thomas;...'",
    )
    parser.add_argument(
        "--timestamps-dataset",
        type=str,
        default="nadia",
        help="Dataset name to use for timestamps",
    )
    parser.add_argument(
        "--tdms-dataset",
        type=str,
        default="nadia",
        help="Dataset name to use for TDMS",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them",
    )
    args = parser.parse_args()

    energies = [float(e) for e in parse_list(args.energies)]
    thresholds = [float(t) for t in parse_list(args.thresholds)]
    datasets = ",".join(parse_list(args.datasets))

    python_exec = sys.executable
    compute_script = repo_root / "python/vortexAnalysis_v2/compute_pressure_modes.py"
    train_script = repo_root / "python/vortexAnalysis_v2/train_classifier_pressure.py"
    test_script = repo_root / "python/vortexAnalysis_v2/test_classifier_real_pressure.py"

    for energy in energies:
        if not args.plot_from_files:
            compute_cmd = [
                python_exec,
                str(compute_script),
                "--energy",
                str(energy),
                "--datasets",
                datasets,
            ]
            run_command(compute_cmd, args.dry_run)

            train_cmd = [
                python_exec,
                str(train_script),
                "--datasets",
                datasets,
                "--modes-energy",
                str(energy),
                "--device",
                args.device,
                "--loss",
                args.loss,
                "--n-trials",
                str(args.n_trials),
            ]
            if args.timeout is not None:
                train_cmd += ["--timeout", str(args.timeout)]
            if args.skip_optuna:
                train_cmd.append("--skip-optuna")
                if args.fixed_params:
                    train_cmd += ["--fixed-params", args.fixed_params]
            run_command(train_cmd, args.dry_run)

        if args.plot_threshold_curves:
            tag = format_energy_tag(energy)
            plot_name = args.plot_save_dir / f"real_pressure_threshold_curves_{tag}.png"
            thresholds_arg = ",".join(f"{t:.4g}" for t in thresholds)
            test_cmd = [
                python_exec,
                str(test_script),
                "--datasets",
                datasets,
                "--modes-energy",
                str(energy),
                "--loss",
                args.loss,
                "--timestamps-dataset",
                args.timestamps_dataset,
                "--tdms-dataset",
                args.tdms_dataset,
                "--plot-from-files",
                "--plot-threshold-curves",
                "--confidence-thresholds",
                thresholds_arg,
                "--plot-save",
                str(plot_name),
            ]
            if args.dataset_roots:
                test_cmd += ["--dataset-roots", args.dataset_roots]
            run_command(test_cmd, args.dry_run)
        else:
            for threshold in thresholds:
                plot_name = None
                if args.plot_distribution:
                    tag = format_energy_tag(energy)
                    plot_name = args.plot_save_dir / f"real_pressure_distribution_{tag}_thr{threshold:.2f}.png"

                test_cmd = [
                    python_exec,
                    str(test_script),
                    "--datasets",
                    datasets,
                    "--modes-energy",
                    str(energy),
                    "--confidence-threshold",
                    str(threshold),
                    "--loss",
                    args.loss,
                    "--timestamps-dataset",
                    args.timestamps_dataset,
                    "--tdms-dataset",
                    args.tdms_dataset,
                ]
                if args.dataset_roots:
                    test_cmd += ["--dataset-roots", args.dataset_roots]
                if args.plot_distribution:
                    test_cmd.append("--plot-distribution")
                if plot_name is not None:
                    test_cmd += ["--plot-save", str(plot_name)]
                if args.plot_from_files:
                    test_cmd.append("--plot-from-files")
                run_command(test_cmd, args.dry_run)


if __name__ == "__main__":
    main()
