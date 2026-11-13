#!/usr/bin/env python3
"""Run the no-slip LilyPad parameter grid with isolated parallel workers.

The defaults reproduce the article's 10 x 10 experiment-scenario grid and run
seven Processing sketches concurrently.  A case is only moved into its final
directory after Processing exits successfully and its three output files pass
basic structural checks.  Completed cases are skipped when the command is run
again, so an interrupted batch can be resumed safely.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence


DEFAULT_RE_VALUES = tuple(range(1000, 10001, 1000))
DEFAULT_H_VALUES = (6.0, 8.64, 11.34, 13.98, 16.68, 19.32, 22.02, 24.66, 27.36, 30.0)
FIELD_FILES = ("FullVorticityMap.csv", "FullPressureMap.csv")
TIME_FILE = "Time.txt"
COMPLETE_FILE = "batch_complete.json"
BASE_NX = 256
BASE_NY = 128
STARTUP_FAILURE_MARKER = "Could not run the sketch."
MAX_STARTUP_ATTEMPTS = 3


@dataclass(frozen=True)
class Case:
    re: int
    h: float

    @property
    def name(self) -> str:
        return f"Re{self.re}_h{format_number(self.h)}"

    @property
    def gap_ratio(self) -> float:
        return self.h / 6.4 - 0.5


@dataclass
class Result:
    case: str
    status: str
    message: str = ""
    maximum_saved_pressure: float | None = None


def format_number(value: float) -> str:
    text = f"{value:.10f}".rstrip("0").rstrip(".")
    return text if "." in text else text + ".0"


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run the LilyPad no-slip Cartesian parameter grid in parallel."
    )
    parser.add_argument(
        "--processing-java",
        type=Path,
        help="Path to processing-java.exe (auto-detected when possible).",
    )
    parser.add_argument(
        "--sketch",
        type=Path,
        default=repo_root / "lily-pad-master" / "LilyPad",
        help="Source LilyPad sketch directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("E:/simuChina/newBDD"),
        help="Directory receiving the completed case folders.",
    )
    parser.add_argument("--jobs", type=int, default=7, help="Number of parallel simulations (default: 7).")
    parser.add_argument("--re-values", nargs="+", type=int, default=DEFAULT_RE_VALUES)
    parser.add_argument("--h-values", nargs="+", type=float, default=DEFAULT_H_VALUES)
    parser.add_argument("--end-time", type=float, default=500.0)
    parser.add_argument("--save-interval", type=float, default=0.5)
    parser.add_argument(
        "--fields",
        choices=("both", "vorticity"),
        default="both",
        help="Save pressure and vorticity, or only vorticity (default: both).",
    )
    parser.add_argument(
        "--resolution-scale",
        choices=(1, 2),
        type=int,
        default=1,
        help="Spatial grid scale: 1 gives 256x128; 2 gives 512x256 (default: 1).",
    )
    parser.add_argument(
        "--motion",
        choices=("stationary", "translate"),
        default="stationary",
        help="Keep the cylinder fixed or translate it 1 m left through still fluid (default: stationary).",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        help="Worker-sketch directory (default: OUTPUT_ROOT/_batch_workers).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rerun completed cases and archive their existing directories before replacement.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned cases without launching Processing.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Run only the first N cases (useful for a production smoke test).",
    )
    return parser.parse_args()


def find_processing_java(explicit: Path | None) -> Path:
    if explicit is not None:
        candidate = explicit.expanduser().resolve()
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"processing-java was not found at {candidate}")

    on_path = shutil.which("processing-java") or shutil.which("processing-java.exe")
    if on_path:
        return Path(on_path).resolve()

    home = Path.home()
    candidates = [
        home / "Desktop" / "software" / "processing-4.2" / "processing-java.exe",
        home / "Documents" / "software" / "processing-4.2" / "processing-java.exe",
        Path("C:/Program Files/Processing/processing-java.exe"),
    ]
    # On machines with a French/OneDrive-redirected Desktop, the directory is
    # commonly named "Bureau" and lives below a "OneDrive ..." folder.
    for one_drive in home.glob("OneDrive*"):
        candidates.extend(
            (
                one_drive / "Bureau" / "software" / "processing-4.2" / "processing-java.exe",
                one_drive / "Desktop" / "software" / "processing-4.2" / "processing-java.exe",
            )
        )
        candidates.extend(
            sorted(
                one_drive.glob("Bureau/software/processing-*/processing-java.exe"),
                reverse=True,
            )
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not find processing-java. Pass its full path with --processing-java."
    )


def effective_end_time(options: argparse.Namespace) -> float:
    # The article translation travels one metre: 128 baseline cells at unit
    # numerical speed. LilyPad scales this time internally for a finer grid.
    return float(BASE_NY) if options.motion == "translate" else options.end_time


def validate_configuration(options: argparse.Namespace) -> None:
    if options.jobs < 1:
        raise ValueError("--jobs must be at least 1")
    if options.end_time <= 0 or options.save_interval <= 0:
        raise ValueError("--end-time and --save-interval must be positive")
    snapshots = effective_end_time(options) / options.save_interval
    if not math.isclose(snapshots, round(snapshots), abs_tol=1e-9):
        raise ValueError("--end-time must be an integer multiple of --save-interval")
    if any(value <= 0 for value in options.re_values):
        raise ValueError("all Reynolds numbers must be positive")
    if any(value <= 0 for value in options.h_values):
        raise ValueError("all cylinder-center heights must be positive")
    if not options.sketch.is_dir() or not (options.sketch / "LilyPad.pde").is_file():
        raise FileNotFoundError(f"Not a LilyPad sketch directory: {options.sketch}")


def field_shape_is_plausible(
    path: Path, expected_columns: int, expected_grid_points: int
) -> tuple[bool, str]:
    if not path.is_file():
        return False, f"missing {path.name}"
    # Even one-character values plus delimiters require roughly this many bytes.
    minimum_size = expected_grid_points * expected_columns * 2
    if path.stat().st_size < minimum_size:
        return False, f"{path.name} is unexpectedly small ({path.stat().st_size} bytes)"
    with path.open("r", encoding="utf-8", errors="strict", newline="") as stream:
        first_row = stream.readline()
    columns = first_row.count(",") + 1 if first_row else 0
    if columns != expected_columns:
        return False, f"{path.name} has {columns} columns; expected {expected_columns}"
    return True, ""


def validate_case(
    directory: Path,
    expected_snapshots: int,
    fields: str,
    resolution_scale: int,
) -> tuple[bool, str]:
    expected_fields = FIELD_FILES if fields == "both" else ("FullVorticityMap.csv",)
    expected_grid_points = BASE_NX * BASE_NY * resolution_scale**2
    for filename in expected_fields:
        valid, reason = field_shape_is_plausible(
            directory / filename, expected_snapshots, expected_grid_points
        )
        if not valid:
            return False, reason

    if fields == "both":
        time_path = directory / TIME_FILE
        if not time_path.is_file():
            return False, f"missing {TIME_FILE}"
        with time_path.open("r", encoding="utf-8", errors="strict") as stream:
            data_lines = sum(1 for line in stream if line and not line.startswith("%"))
        if data_lines != expected_snapshots:
            return False, f"{TIME_FILE} has {data_lines} snapshots; expected {expected_snapshots}"
    return True, ""


def prepare_worker(source: Path, work_root: Path, worker_id: int) -> Path:
    worker_parent = work_root / f"worker_{worker_id:02d}"
    worker_sketch = worker_parent / "LilyPad"
    if worker_sketch.exists():
        shutil.rmtree(worker_sketch)
    worker_parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, worker_sketch)
    return worker_sketch


def archive_existing(directory: Path, archive_root: Path) -> None:
    if not directory.exists():
        return
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    destination = archive_root / f"{directory.name}-{timestamp}"
    suffix = 1
    while destination.exists():
        destination = archive_root / f"{directory.name}-{timestamp}-{suffix}"
        suffix += 1
    archive_root.mkdir(parents=True, exist_ok=True)
    directory.replace(destination)


def extract_max_pressure(log_path: Path) -> float | None:
    pattern = re.compile(r"Maximum saved \|p\|:\s*([+\-0-9.eE]+)")
    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                match = pattern.search(line)
                if match:
                    return float(match.group(1))
    except OSError:
        pass
    return None


def log_contains_startup_failure(log_path: Path) -> bool:
    try:
        with log_path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - 16384))
            tail = stream.read().decode("utf-8", errors="replace")
        return STARTUP_FAILURE_MARKER in tail
    except OSError:
        return False


def terminate_process_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()


def wait_for_processing(
    process: subprocess.Popen[str], log_path: Path
) -> tuple[int, bool]:
    while True:
        try:
            return process.wait(timeout=2), False
        except subprocess.TimeoutExpired:
            if log_contains_startup_failure(log_path):
                terminate_process_tree(process)
                return process.returncode if process.returncode is not None else -1, True


def run_case(
    case: Case,
    worker_sketch: Path,
    processing_java: Path,
    options: argparse.Namespace,
    output_root: Path,
    in_progress_root: Path,
    log_root: Path,
    active_processes: dict[int, subprocess.Popen[str]],
    process_lock: threading.Lock,
) -> Result:
    final_directory = output_root / case.name
    case_end_time = effective_end_time(options)
    expected_snapshots = round(case_end_time / options.save_interval)

    if final_directory.exists() and not options.overwrite:
        valid, reason = validate_case(
            final_directory,
            expected_snapshots,
            options.fields,
            options.resolution_scale,
        )
        if valid:
            return Result(case.name, "skipped", "existing output passed validation")
        return Result(
            case.name,
            "failed",
            f"existing directory is incomplete ({reason}); use --overwrite to replace it",
        )

    temporary_directory = in_progress_root / case.name
    if temporary_directory.exists():
        shutil.rmtree(temporary_directory)
    temporary_directory.mkdir(parents=True)
    log_path = log_root / f"{case.name}.log"

    command = [
        str(processing_java),
        f"--sketch={worker_sketch}",
        "--run",
        "--args",
        str(case.re),
        format_number(case.h),
        str(temporary_directory.resolve()),
        format_number(case_end_time),
        format_number(options.save_interval),
        options.fields,
        str(options.resolution_scale),
        options.motion,
    ]
    started_at = datetime.now().astimezone()
    return_code = -1
    for attempt in range(1, MAX_STARTUP_ATTEMPTS + 1):
        # Start each attempt with a clean log so a previous fatal marker cannot
        # be mistaken for a failure of the new Processing process.
        with log_path.open("w", encoding="utf-8") as log:
            if attempt > 1:
                log.write(f"\n--- Processing startup retry {attempt}/{MAX_STARTUP_ATTEMPTS} ---\n")
            log.write("Command: " + subprocess.list2cmdline(command) + "\n\n")
            log.flush()
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            with process_lock:
                active_processes[threading.get_ident()] = process
            try:
                return_code, startup_failure = wait_for_processing(process, log_path)
            finally:
                with process_lock:
                    active_processes.pop(threading.get_ident(), None)
        startup_failure = startup_failure or log_contains_startup_failure(log_path)
        if return_code == 0:
            break
        if not startup_failure or attempt == MAX_STARTUP_ATTEMPTS:
            break
    if return_code != 0:
        startup_note = " after startup retries" if startup_failure else ""
        return Result(
            case.name,
            "failed",
            f"Processing exited with code {return_code}{startup_note}; see {log_path}",
        )

    valid, reason = validate_case(
        temporary_directory,
        expected_snapshots,
        options.fields,
        options.resolution_scale,
    )
    if not valid:
        return Result(case.name, "failed", f"output validation failed: {reason}; see {log_path}")

    maximum_pressure = extract_max_pressure(log_path)
    metadata = {
        "case": asdict(case),
        "gap_ratio": case.gap_ratio,
        "end_time": case_end_time,
        "save_interval": options.save_interval,
        "fields": options.fields,
        "resolution_scale": options.resolution_scale,
        "motion": options.motion,
        "grid_shape": [
            BASE_NX * options.resolution_scale,
            BASE_NY * options.resolution_scale,
        ],
        "solver_end_time": case_end_time * options.resolution_scale,
        "solver_save_interval": options.save_interval * options.resolution_scale,
        "snapshots": expected_snapshots,
        "maximum_saved_pressure": maximum_pressure,
        "started_at": started_at.isoformat(),
        "completed_at": datetime.now().astimezone().isoformat(),
        "log": str(log_path),
    }
    (temporary_directory / COMPLETE_FILE).write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    if final_directory.exists():
        archive_existing(final_directory, output_root / "_replaced_cases")
    temporary_directory.replace(final_directory)
    return Result(case.name, "completed", maximum_saved_pressure=maximum_pressure)


def write_manifest(path: Path, cases: Sequence[Case]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("case", "Re", "h", "gap_over_D"))
        for case in cases:
            writer.writerow((case.name, case.re, format_number(case.h), f"{case.gap_ratio:.10g}"))


def main() -> int:
    options = parse_args()
    try:
        validate_configuration(options)
    except (FileNotFoundError, ValueError) as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2

    cases = [Case(re_value, h_value) for re_value in options.re_values for h_value in options.h_values]
    if options.limit is not None:
        if options.limit < 1:
            print("Configuration error: --limit must be at least 1", file=sys.stderr)
            return 2
        cases = cases[: options.limit]

    print(f"Planned cases: {len(cases)}; parallel workers: {options.jobs}")
    print(f"Output root: {options.output_root.resolve()}")
    print(
        f"Configuration: motion={options.motion}, fields={options.fields}, "
        f"grid={BASE_NX * options.resolution_scale}x{BASE_NY * options.resolution_scale}, "
        f"snapshots={round(effective_end_time(options) / options.save_interval)}"
    )
    if options.dry_run:
        for case in cases:
            print(f"  {case.name} (gap/D={case.gap_ratio:.4f})")
        return 0

    try:
        processing_java = find_processing_java(options.processing_java)
    except FileNotFoundError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2

    output_root = options.output_root.resolve()
    work_base = (options.work_root or output_root / "_batch_workers").resolve()
    run_cache_name = datetime.now().strftime("run_%Y%m%d-%H%M%S_") + str(os.getpid())
    work_root = work_base / run_cache_name
    in_progress_root = output_root / "_in_progress"
    log_root = output_root / "_logs"
    for directory in (output_root, work_root, in_progress_root, log_root):
        directory.mkdir(parents=True, exist_ok=True)
    write_manifest(output_root / "batch_manifest.csv", cases)

    lock_path = output_root / ".lilypad_batch.lock"
    try:
        lock_descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print(f"Another batch may be running: lock file exists at {lock_path}", file=sys.stderr)
        return 2
    os.write(lock_descriptor, f"pid={os.getpid()}\n".encode())
    os.close(lock_descriptor)

    case_queue: queue.Queue[Case] = queue.Queue()
    for case in cases:
        case_queue.put(case)
    results: list[Result] = []
    result_lock = threading.Lock()
    print_lock = threading.Lock()
    process_lock = threading.Lock()
    active_processes: dict[int, subprocess.Popen[str]] = {}
    stop_event = threading.Event()

    def worker(worker_id: int) -> None:
        try:
            worker_sketch = prepare_worker(options.sketch.resolve(), work_root, worker_id)
        except Exception as error:  # report setup failure without hiding the traceback context
            with result_lock:
                results.append(Result(f"worker_{worker_id:02d}", "failed", f"worker setup: {error}"))
            return
        while not stop_event.is_set():
            try:
                case = case_queue.get_nowait()
            except queue.Empty:
                return
            with print_lock:
                print(f"[worker {worker_id:02d}] starting {case.name}", flush=True)
            try:
                result = run_case(
                    case,
                    worker_sketch,
                    processing_java,
                    options,
                    output_root,
                    in_progress_root,
                    log_root,
                    active_processes,
                    process_lock,
                )
            except Exception as error:
                result = Result(case.name, "failed", f"unexpected error: {error}")
            with result_lock:
                results.append(result)
            with print_lock:
                detail = f" - {result.message}" if result.message else ""
                pressure = (
                    f" (max |p|={result.maximum_saved_pressure:g})"
                    if result.maximum_saved_pressure is not None
                    else ""
                )
                print(f"[worker {worker_id:02d}] {result.status}: {case.name}{pressure}{detail}", flush=True)
            case_queue.task_done()

    threads = [
        threading.Thread(target=worker, args=(worker_id,), name=f"lilypad-{worker_id:02d}")
        for worker_id in range(min(options.jobs, len(cases)))
    ]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        stop_event.set()
        print("Interrupted; terminating active Processing simulations.", file=sys.stderr)
        with process_lock:
            processes_to_stop = list(active_processes.values())
        for process in processes_to_stop:
            terminate_process_tree(process)
        for thread in threads:
            thread.join()
        return 130
    finally:
        lock_path.unlink(missing_ok=True)

    counts = {status: sum(result.status == status for result in results) for status in ("completed", "skipped", "failed")}
    print(
        "Batch finished: "
        + ", ".join(f"{count} {status}" for status, count in counts.items())
    )
    failures = [result for result in results if result.status == "failed"]
    if failures:
        print("Failures:", file=sys.stderr)
        for result in failures:
            print(f"  {result.case}: {result.message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
