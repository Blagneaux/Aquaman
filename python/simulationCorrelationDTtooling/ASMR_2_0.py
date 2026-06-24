#!/usr/bin/env python3
"""ASMR 2.0: automated simulation/experiment campaign orchestration."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import tomllib
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from asmr_acquisition import AcquisitionSession, AcquisitionStats
from asmr_cnc import (
    SerialCNCController,
    TowProgram,
    build_paired_tow_programs,
    write_program_files,
)
from asmr_metrics import (
    MetricBundle,
    append_metric_row,
    ensure_metric_bundle,
    evaluate_discovered_metrics,
    initialize_metric_tables,
    load_experimental_pressure,
    load_simulation_pressure,
    select_next_parameters,
    subtract_experimental_recordings,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "asmr_2_0.toml"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def compact_number(value: float) -> str:
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:.8g}"


def scenario_context(
    re_value: float,
    h_value: float,
    parameter_file: Path,
    run_directory: Path,
) -> dict[str, str]:
    return {
        "re": str(float(re_value)),
        "re_int": str(int(round(re_value))),
        "re_compact": compact_number(re_value),
        "h": str(float(h_value)),
        "h_compact": compact_number(h_value),
        "parameter_file": str(parameter_file),
        "run_dir": str(run_directory),
    }


def format_template(template: str, context: Mapping[str, str]) -> str:
    try:
        return template.format_map(context)
    except KeyError as exc:
        raise ValueError(f"Unknown placeholder {exc} in template '{template}'.") from exc


def load_config(path: Path) -> dict[str, Any]:
    with Path(path).open("rb") as config_file:
        config = tomllib.load(config_file)

    required_sections = [
        "paths",
        "metric_discovery",
        "simulation",
        "sensor",
        "telemetry",
        "fluid",
        "tank",
        "safety",
        "cnc",
        "search",
        "timing",
        "campaign",
    ]
    missing = [section for section in required_sections if section not in config]
    if missing:
        raise ValueError(f"Missing configuration sections: {', '.join(missing)}")
    return config


class CampaignLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, message: str) -> None:
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as output:
            output.write(line + "\n")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_parameter_csv(
    output_path: Path,
    re_value: float,
    h_value: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"Re": float(re_value), "h": float(h_value)}]).to_csv(
        output_path, index=False
    )


class SimulationJob:
    """Start LilyPad early, then wait/retry and verify its explicit output file."""

    def __init__(
        self,
        config: Mapping[str, Any],
        context: Mapping[str, str],
        run_directory: Path,
        log: CampaignLogger,
    ):
        self.config = config
        self.context = context
        self.run_directory = run_directory
        self.log = log
        self.process: subprocess.Popen | None = None
        self.log_file = None
        self.attempt = 0
        self.started_wall_time = 0.0
        self.started_monotonic = 0.0
        self.expected_output = Path(
            format_template(str(config["expected_output"]), context)
        )

    def _command(self) -> list[str]:
        raw_command = self.config.get("command")
        if not isinstance(raw_command, Sequence) or isinstance(raw_command, (str, bytes)):
            raise ValueError("simulation.command must be a TOML array of command arguments.")
        return [format_template(str(argument), self.context) for argument in raw_command]

    def _start_attempt(self) -> None:
        self.attempt += 1
        command = self._command()
        if not command:
            raise ValueError("simulation.command cannot be empty.")
        simulation_log = self.run_directory / "simulation.log"
        self.log_file = simulation_log.open("a", encoding="utf-8")
        self.log_file.write(
            f"\n--- attempt {self.attempt} at {datetime.now().isoformat()} ---\n"
        )
        self.log_file.flush()
        self.log(f"[Simulation] Starting attempt {self.attempt}: {' '.join(command)}")
        creation_flags = 0
        if os.name == "nt" and hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
            creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP
        self.started_wall_time = time.time()
        self.started_monotonic = time.monotonic()
        try:
            self.process = subprocess.Popen(
                command,
                cwd=self.config.get("working_directory") or None,
                stdout=self.log_file,
                stderr=subprocess.STDOUT,
                creationflags=creation_flags,
            )
        except BaseException:
            self._close_attempt_log()
            raise

    def start(self) -> None:
        if bool(self.config.get("enabled", True)):
            self._start_attempt()
        else:
            self.log(
                f"[Simulation] Launch disabled; expecting existing output "
                f"{self.expected_output}"
            )

    def _close_attempt_log(self) -> None:
        if self.log_file:
            self.log_file.flush()
            self.log_file.close()
            self.log_file = None

    def _wait_process(self) -> bool:
        if self.process is None:
            return True
        timeout_s = float(self.config.get("timeout_s", 300.0))
        elapsed = time.monotonic() - self.started_monotonic
        remaining_timeout = max(0.0, timeout_s - elapsed)
        try:
            return_code = self.process.wait(timeout=remaining_timeout)
        except subprocess.TimeoutExpired:
            self.log(f"[Simulation] Attempt {self.attempt} timed out after {timeout_s}s")
            self.process.kill()
            self.process.wait()
            return False
        finally:
            self._close_attempt_log()
        if return_code != 0:
            self.log(
                f"[Simulation] Attempt {self.attempt} exited with code {return_code}"
            )
            return False
        return True

    def _output_is_valid(self) -> bool:
        if not self.expected_output.exists() or self.expected_output.stat().st_size == 0:
            return False
        if not bool(self.config.get("require_fresh_output", True)):
            return True
        if not bool(self.config.get("enabled", True)):
            return True
        return self.expected_output.stat().st_mtime >= self.started_wall_time - 1.0

    def wait(self) -> Path:
        retries = max(0, int(self.config.get("retries", 2)))
        while True:
            process_ok = self._wait_process()
            output_wait_s = float(self.config.get("output_wait_s", 10.0))
            deadline = time.monotonic() + output_wait_s
            while process_ok and time.monotonic() < deadline:
                if self._output_is_valid():
                    self.log(f"[Simulation] Output ready: {self.expected_output}")
                    return self.expected_output
                time.sleep(0.25)

            if self.attempt > retries or not bool(self.config.get("enabled", True)):
                break
            self.log("[Simulation] Retrying after failed process or missing output")
            self._start_attempt()

        raise RuntimeError(
            f"Simulation failed after {self.attempt} attempt(s), or output was not "
            f"created at {self.expected_output}."
        )

    def abort(self) -> None:
        if self.process and self.process.poll() is None:
            self.log("[Simulation] Terminating active process")
            self.process.terminate()
            try:
                self.process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self._close_attempt_log()


def create_run_directory(
    campaign_root: Path,
    iteration: int,
    re_value: float,
    h_value: float,
) -> tuple[str, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = (
        f"{timestamp}_i{iteration:04d}_Re{compact_number(re_value)}"
        f"_h{compact_number(h_value)}"
    )
    run_directory = campaign_root / "runs" / run_id
    suffix = 1
    while run_directory.exists():
        run_directory = campaign_root / "runs" / f"{run_id}_{suffix}"
        suffix += 1
    run_directory.mkdir(parents=True)
    return run_directory.name, run_directory


def metric_summary(
    bundle: MetricBundle,
    source: str,
    pressure_path: Path,
    re_value: float,
    delimiter: str = ";",
) -> tuple[dict[str, float], dict[str, float]]:
    if source == "simulation":
        pressure, time_s = load_simulation_pressure(pressure_path, re_value)
    elif source == "experiment":
        pressure, time_s = load_experimental_pressure(
            pressure_path,
            delimiter=delimiter,
            max_sensors=bundle.extractor.get("max_sensors"),
        )
    else:
        raise ValueError(f"Unknown metric source '{source}'.")
    return evaluate_discovered_metrics(pressure, time_s, re_value, bundle)


def acquisition_stats_payload(stats: AcquisitionStats) -> dict[str, int]:
    return {key: int(value) for key, value in asdict(stats).items()}


def record_tow(
    controller: SerialCNCController,
    program: TowProgram,
    output_path: Path,
    config: Mapping[str, Any],
    log: CampaignLogger,
    settling_after_preposition_s: float,
) -> AcquisitionStats:
    controller.preposition(program)
    if settling_after_preposition_s > 0:
        log(
            f"[ASMR] Waiting {settling_after_preposition_s:.1f}s after "
            f"{program.label} pre-positioning"
        )
        time.sleep(settling_after_preposition_s)
    acquisition = AcquisitionSession(
        sensor_config=config["sensor"],
        telemetry_config=config["telemetry"],
        output_path=output_path,
        log=log,
    )
    acquisition.start(
        timeout_s=float(config["sensor"].get("startup_timeout_s", 5.0))
    )
    try:
        time.sleep(float(config["timing"].get("pre_roll_s", 1.0)))
        controller.tow(program)
        time.sleep(float(config["timing"].get("post_roll_s", 1.0)))
    except BaseException:
        try:
            acquisition.stop()
        except Exception as stop_error:  # noqa: BLE001 - retain the motion failure.
            log(f"[ASMR] Acquisition shutdown error: {stop_error}")
        raise
    return acquisition.stop()


def run_campaign(
    config: dict[str, Any],
    iterations: int,
    refresh_metrics: bool,
    dry_run: bool,
) -> None:
    paths = config["paths"]
    campaign_root = Path(paths["campaign_root"])
    campaign_root.mkdir(parents=True, exist_ok=True)
    log = CampaignLogger(campaign_root / "campaign.log")

    log("[ASMR] Loading metric bundle")
    bundle = ensure_metric_bundle(config["metric_discovery"], refresh=refresh_metrics)
    log(f"[ASMR] Metric bundle: {bundle.path}")
    log(
        f"[ASMR] Metric domain: Re={bundle.domain['Re']}, h={bundle.domain['h']}"
    )

    simulation_metrics_path = campaign_root / "simulation_metrics.csv"
    experiment_metrics_path = campaign_root / "experiment_metrics.csv"
    initialize_metric_tables(
        bundle, simulation_metrics_path, experiment_metrics_path
    )

    parameter_file = Path(paths["parameter_csv"])
    cnc_enabled = bool(config["cnc"].get("enabled", True))
    controller: SerialCNCController | None = None

    try:
        if not dry_run and cnc_enabled:
            controller = SerialCNCController(config["cnc"], log=log)
            controller.connect()
            controller.home_once()
        elif not dry_run:
            raise RuntimeError(
                "cnc.enabled is false. Use --dry-run for a hardware-free plan, "
                "or enable CNC control in the configuration."
            )

        for iteration in range(1, iterations + 1):
            selection = select_next_parameters(
                simulation_metrics_path,
                experiment_metrics_path,
                bundle,
                config["search"],
            )
            re_value = selection.re_value
            h_value = selection.h_value
            run_id, run_directory = create_run_directory(
                campaign_root, iteration, re_value, h_value
            )
            run_log = CampaignLogger(run_directory / "run.log")
            run_log(
                f"[ASMR] Selected Re={re_value:.8g}, h={h_value:.8g}; "
                f"strategy={selection.strategy}, score={selection.score:.6g}"
            )

            run_parameter_file = run_directory / "parameters.csv"
            write_parameter_csv(run_parameter_file, re_value, h_value)
            write_parameter_csv(parameter_file, re_value, h_value)

            measurement_program, reference_program = build_paired_tow_programs(
                re_value,
                h_value,
                config["tank"],
                config["fluid"],
                config["safety"],
            )
            measurement_preposition_path, measurement_tow_path = write_program_files(
                measurement_program, run_directory / "measurement"
            )
            reference_preposition_path, reference_tow_path = write_program_files(
                reference_program, run_directory / "reference"
            )
            measurement_sensor_path = run_directory / "measurement" / "sensors.csv"
            reference_sensor_path = run_directory / "reference" / "sensors.csv"
            difference_sensor_path = run_directory / "pressure_difference.csv"

            context = scenario_context(
                re_value, h_value, parameter_file, run_directory
            )
            simulation_job = SimulationJob(
                config["simulation"], context, run_directory, run_log
            )
            manifest_path = run_directory / "manifest.json"
            manifest: dict[str, Any] = {
                "run_id": run_id,
                "iteration": iteration,
                "status": "prepared",
                "created_at": utc_now(),
                "selection": asdict(selection),
                "parameters": {"Re": re_value, "h": h_value},
                "files": {
                    "parameters": str(run_parameter_file),
                    "measurement_preposition_gcode": str(
                        measurement_preposition_path
                    ),
                    "measurement_tow_gcode": str(measurement_tow_path),
                    "measurement_sensors": str(measurement_sensor_path),
                    "reference_preposition_gcode": str(reference_preposition_path),
                    "reference_tow_gcode": str(reference_tow_path),
                    "reference_sensors": str(reference_sensor_path),
                    "pressure_difference": str(difference_sensor_path),
                    "expected_simulation": str(simulation_job.expected_output),
                },
                "motion_pair": {
                    "x_start_mm": measurement_program.x_start_mm,
                    "x_end_mm": measurement_program.x_end_mm,
                    "y_mm": measurement_program.y_mm,
                    "measurement_z_mm": measurement_program.z_mm,
                    "reference_z_mm": reference_program.z_mm,
                    "reference_z_offset_mm": (
                        reference_program.z_mm - measurement_program.z_mm
                    ),
                    "fourth_mm": measurement_program.fourth_mm,
                    "subtraction": "measurement_minus_reference",
                },
            }
            write_json(manifest_path, manifest)

            if dry_run:
                manifest["status"] = "dry_run"
                manifest["finished_at"] = utc_now()
                write_json(manifest_path, manifest)
                run_log("[ASMR] Dry run complete; no process, network, or serial I/O used")
                if iterations > 1:
                    log("[ASMR] Dry-run mode stops after one generated scenario")
                break

            try:
                manifest["status"] = "running"
                manifest["started_at"] = utc_now()
                write_json(manifest_path, manifest)

                simulation_job.start()
                assert controller is not None
                measurement_stats = record_tow(
                    controller,
                    measurement_program,
                    measurement_sensor_path,
                    config,
                    run_log,
                    settling_after_preposition_s=float(
                        config["timing"].get("preposition_settling_s", 300.0)
                    ),
                )
                manifest["acquisition"] = {
                    "measurement": acquisition_stats_payload(measurement_stats)
                }
                write_json(manifest_path, manifest)

                reference_stats = record_tow(
                    controller,
                    reference_program,
                    reference_sensor_path,
                    config,
                    run_log,
                    settling_after_preposition_s=float(
                        config["timing"].get("between_pair_settling_s", 300.0)
                    ),
                )
                manifest["acquisition"]["reference"] = acquisition_stats_payload(
                    reference_stats
                )
                write_json(manifest_path, manifest)

                simulation_path = simulation_job.wait()
                simulation_metrics, simulation_features = metric_summary(
                    bundle, "simulation", simulation_path, re_value
                )
                difference_pressure, difference_time, pairing_metadata = (
                    subtract_experimental_recordings(
                        measurement_path=measurement_sensor_path,
                        reference_path=reference_sensor_path,
                        output_path=difference_sensor_path,
                        delimiter=str(config["sensor"].get("csv_delimiter", ";")),
                        max_sensors=bundle.extractor.get("max_sensors"),
                    )
                )
                experiment_metrics, experiment_features = (
                    evaluate_discovered_metrics(
                        difference_pressure,
                        difference_time,
                        re_value,
                        bundle,
                    )
                )
                manifest["pressure_subtraction"] = pairing_metadata
                write_json(manifest_path, manifest)

                append_metric_row(
                    simulation_metrics_path,
                    re_value,
                    h_value,
                    simulation_metrics,
                    "simulation",
                    run_id,
                    simulation_path,
                )
                append_metric_row(
                    experiment_metrics_path,
                    re_value,
                    h_value,
                    experiment_metrics,
                    "experiment_subtracted",
                    run_id,
                    difference_sensor_path,
                )
                write_json(
                    run_directory / "metrics.json",
                    {
                        "simulation": simulation_metrics,
                        "experiment": experiment_metrics,
                        "pressure_subtraction": pairing_metadata,
                        "selected_features": {
                            "simulation": {
                                name: simulation_features[name]
                                for model in bundle.models.values()
                                for name in model.get("selected_features", [])
                                if name in simulation_features
                            },
                            "experiment": {
                                name: experiment_features[name]
                                for model in bundle.models.values()
                                for name in model.get("selected_features", [])
                                if name in experiment_features
                            },
                        },
                    },
                )

                manifest["status"] = "complete"
                manifest["finished_at"] = utc_now()
                manifest["simulation_output"] = str(simulation_path)
                manifest["metrics"] = {
                    "simulation": simulation_metrics,
                    "experiment": experiment_metrics,
                }
                write_json(manifest_path, manifest)
                run_log("[ASMR] Run completed")
            except BaseException as exc:
                simulation_job.abort()
                if controller:
                    controller.feed_hold()
                manifest["status"] = "failed"
                manifest["finished_at"] = utc_now()
                manifest["error"] = f"{type(exc).__name__}: {exc}"
                write_json(manifest_path, manifest)
                run_log(f"[ASMR] Run failed: {exc}")
                raise

            if iteration < iterations:
                settling_time = float(config["timing"].get("settling_s", 300.0))
                if settling_time > 0:
                    log(f"[ASMR] Waiting {settling_time:.1f}s for the water to settle")
                    time.sleep(settling_time)
    finally:
        if controller:
            controller.close()


def command_discover(config: dict[str, Any], refresh: bool) -> None:
    bundle = ensure_metric_bundle(config["metric_discovery"], refresh=refresh)
    print(f"Metric bundle: {bundle.path}")
    print(f"Candidate metrics: {bundle.candidate_metrics_path}")
    print(f"Domain: {bundle.domain}")


def command_cnc_debug(
    config: dict[str, Any],
    re_value: float,
    h_value: float,
    output_directory: Path,
    execute: bool,
) -> None:
    measurement_program, reference_program = build_paired_tow_programs(
        re_value,
        h_value,
        config["tank"],
        config["fluid"],
        config["safety"],
    )
    measurement_paths = write_program_files(
        measurement_program, output_directory / "measurement"
    )
    reference_paths = write_program_files(
        reference_program, output_directory / "reference"
    )
    print(f"Measurement pre-position G-code: {measurement_paths[0]}")
    print(f"Measurement tow G-code: {measurement_paths[1]}")
    print(f"Reference pre-position G-code: {reference_paths[0]}")
    print(f"Reference tow G-code: {reference_paths[1]}")
    print(
        f"Path: X {measurement_program.x_start_mm:.3f} -> "
        f"{measurement_program.x_end_mm:.3f} mm, "
        f"Y {measurement_program.y_mm:.3f} mm, "
        f"measurement Z {measurement_program.z_mm:.3f} mm, "
        f"reference Z {reference_program.z_mm:.3f} mm, "
        f"{len(measurement_program.points)} segments per tow"
    )
    if not execute:
        print("Dry run only. Pass --execute to connect, home, pre-position, and tow.")
        return

    controller = SerialCNCController(config["cnc"])
    try:
        controller.connect()
        controller.home_once()
        controller.preposition(measurement_program)
        controller.tow(measurement_program)
        controller.preposition(reference_program)
        controller.tow(reference_program)
    except BaseException:
        controller.feed_hold()
        raise
    finally:
        controller.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ASMR 2.0 automated simulation/experiment campaign."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"TOML configuration file (default: {DEFAULT_CONFIG_PATH}).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run or resume an ASMR campaign.")
    run_parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Number of new iterations; defaults to campaign.iterations.",
    )
    run_parser.add_argument(
        "--refresh-metrics",
        action="store_true",
        help="Re-run automatic metric discovery before the campaign.",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare one scenario without launching simulation, UDP, or CNC I/O.",
    )

    discovery_parser = subparsers.add_parser(
        "discover-metrics", help="Load or rebuild the automatic metric bundle."
    )
    discovery_parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force discovery even when cached artifacts exist.",
    )

    cnc_parser = subparsers.add_parser(
        "cnc-debug", help="Generate and optionally execute the paired CNC tows."
    )
    cnc_parser.add_argument("--re", type=float, required=True)
    cnc_parser.add_argument("--h", type=float, required=True)
    cnc_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("python/simulationCorrelationDTtooling/cncjs_tests/asmr_2_0"),
    )
    cnc_parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually connect and execute both tows; omitted means dry-run.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "discover-metrics":
        command_discover(config, refresh=args.refresh)
        return 0
    if args.command == "cnc-debug":
        command_cnc_debug(
            config,
            re_value=args.re,
            h_value=args.h,
            output_directory=args.output_dir,
            execute=args.execute,
        )
        return 0
    if args.command == "run":
        iterations = (
            args.iterations
            if args.iterations is not None
            else int(config["campaign"].get("iterations", 1))
        )
        if iterations <= 0:
            parser.error("--iterations must be positive.")
        run_campaign(
            config,
            iterations=iterations,
            refresh_metrics=args.refresh_metrics,
            dry_run=args.dry_run,
        )
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
