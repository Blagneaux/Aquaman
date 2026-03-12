import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SpineSeries:
    experiment: str
    sub_experiment: str
    path: Path
    length: int
    x: np.ndarray
    y: np.ndarray
    q: np.ndarray
    qt: np.ndarray
    v: np.ndarray
    a: np.ndarray


@dataclass
class AutomationEntry:
    experiment: str
    sub_experiment: str
    skip_rows: int


def parse_range_list(value: str | None) -> set[str] | None:
    if not value:
        return None
    items: set[str] = set()
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_s, end_s = chunk.split("-", 1)
            try:
                start = int(start_s)
                end = int(end_s)
            except ValueError:
                items.add(chunk)
                continue
            if end < start:
                start, end = end, start
            for val in range(start, end + 1):
                items.add(str(val))
        else:
            items.add(chunk)
    return items


def iter_spine_files(root: Path, exp_filter: set[str] | None, sub_filter: set[str] | None) -> list[Path]:
    files: list[Path] = []
    if not root.exists():
        return files
    for exp_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        exp_name = exp_dir.name
        if exp_filter is not None and exp_name not in exp_filter:
            continue
        for sub_dir in sorted(p for p in exp_dir.iterdir() if p.is_dir()):
            sub_name = sub_dir.name
            if sub_filter is not None and sub_name not in sub_filter:
                continue
            path = sub_dir / "final" / "spines_interpolated.csv"
            if path.exists():
                files.append(path)
    return files


def load_automation_csv(path: Path) -> list[AutomationEntry]:
    entries: list[AutomationEntry] = []
    if not path.exists():
        return entries
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if row[0].strip().lower().startswith("experiment"):
                continue
            if len(row) < 3:
                continue
            exp = row[0].strip()
            sub = row[1].strip()
            try:
                skip = int(float(row[2]))
            except ValueError:
                skip = 0
            entries.append(AutomationEntry(exp, sub, max(0, skip)))
    return entries


def build_spine_jobs(
    root: Path,
    automation: list[AutomationEntry],
    exp_filter: set[str] | None,
    sub_filter: set[str] | None,
) -> list[tuple[Path, str, str, int]]:
    jobs: list[tuple[Path, str, str, int]] = []
    if automation:
        for entry in automation:
            if exp_filter is not None and entry.experiment not in exp_filter:
                continue
            if sub_filter is not None and entry.sub_experiment not in sub_filter:
                continue
            path = root / entry.experiment / entry.sub_experiment / "final" / "spines_interpolated.csv"
            jobs.append((path, entry.experiment, entry.sub_experiment, entry.skip_rows))
        return jobs

    for path in iter_spine_files(root, exp_filter, sub_filter):
        parts = path.parts
        experiment = parts[-3] if len(parts) >= 3 else "unknown"
        sub_experiment = parts[-4] if len(parts) >= 4 else "unknown"
        jobs.append((path, experiment, sub_experiment, 0))
    return jobs


def read_spine_series(
    path: Path,
    rescale: bool,
    skip_rows: int,
    experiment: str | None,
    sub_experiment: str | None,
) -> SpineSeries:
    xs: list[float] = []
    ys: list[float] = []
    qs: list[float] = []
    qts: list[float] = []

    scale_x = 1.0 / 2000.0
    scale_y = 1.0 / 1200.0

    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            if idx < skip_rows:
                continue
            if not row:
                continue
            try:
                values = [float(v) for v in row]
            except ValueError:
                continue
            if len(values) < 4:
                continue
            if len(values) <= 70:
                raise ValueError(f"{path} has too few columns for spine index 69/70.")

            com_x = values[69]
            com_y = values[70]
            nose_x = values[-2]
            nose_y = values[-1]

            tail_x1 = values[1]
            tail_y1 = values[2]
            tail_x2 = values[3]
            tail_y2 = values[4]
            tail_x3 = values[5]
            tail_y3 = values[6]

            if rescale:
                com_x *= scale_x
                com_y *= scale_y
                nose_x *= scale_x
                nose_y *= scale_y
                tail_x1 *= scale_x
                tail_y1 *= scale_y
                tail_x2 *= scale_x
                tail_y2 *= scale_y
                tail_x3 *= scale_x
                tail_y3 *= scale_y

            xs.append(com_x)
            ys.append(com_y)

            dx = nose_x - com_x
            dy = nose_y - com_y
            angle = float(np.arctan2(dy, dx))
            if rescale:
                angle = (angle % (2.0 * np.pi)) / (2.0 * np.pi)
            qs.append(angle)

            tail_dx = tail_x3 - tail_x1
            tail_dy = tail_y3 - tail_y1
            orient_norm = np.hypot(dx, dy)
            tail_norm = np.hypot(tail_dx, tail_dy)
            if orient_norm > 0.0 and tail_norm > 0.0:
                cross = dx * tail_dy - dy * tail_dx
                dot = dx * tail_dx + dy * tail_dy
                tail_angle = float(np.arctan2(cross, dot))
                tail_angle = tail_angle % (2.0 * np.pi)
            else:
                tail_angle = 0.0
            if rescale:
                tail_angle = tail_angle / (2.0 * np.pi)
            qts.append(tail_angle)

    x_arr = np.asarray(xs, dtype=np.float64)
    y_arr = np.asarray(ys, dtype=np.float64)
    q_arr = np.asarray(qs, dtype=np.float64)
    qt_arr = np.asarray(qts, dtype=np.float64)

    if x_arr.size <= 1:
        v_arr = np.zeros_like(x_arr)
        a_arr = np.zeros_like(x_arr)
    else:
        dx = np.diff(x_arr)
        dy = np.diff(y_arr)
        speed = np.sqrt(dx * dx + dy * dy)
        v_arr = np.concatenate([speed, speed[-1:]])

        accel = np.diff(v_arr)
        a_arr = np.concatenate([accel, accel[-1:]])

    if experiment is None or sub_experiment is None:
        parts = path.parts
        experiment = parts[-3] if len(parts) >= 3 else "unknown"
        sub_experiment = parts[-4] if len(parts) >= 4 else "unknown"
    return SpineSeries(
        experiment=experiment,
        sub_experiment=sub_experiment,
        path=path,
        length=int(x_arr.size),
        x=x_arr,
        y=y_arr,
        q=q_arr,
        qt=qt_arr,
        v=v_arr,
        a=a_arr,
    )


def build_matrix(series_list: list[SpineSeries], attr: str) -> np.ndarray:
    if not series_list:
        return np.empty((0, 0), dtype=np.float64)
    max_len = max(s.length for s in series_list)
    matrix = np.full((max_len, len(series_list)), np.nan, dtype=np.float64)
    for idx, series in enumerate(series_list):
        values = getattr(series, attr)
        matrix[: values.size, idx] = values
    return matrix


def save_matrix(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, matrix, delimiter=",", fmt="%.6f")


def save_series(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, values.reshape(-1, 1), delimiter=",", fmt="%.6f")


def save_meta(path: Path, series_list: list[SpineSeries]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["experiment", "sub_experiment", "length", "path"],
        )
        writer.writeheader()
        for series in series_list:
            writer.writerow(
                {
                    "experiment": series.experiment,
                    "sub_experiment": series.sub_experiment,
                    "length": series.length,
                    "path": str(series.path),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess spine files to extract COM position, orientation, speed, and acceleration."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("D:/thomas_files"),
        help="Root directory containing experiment/sub_experiment folders.",
    )
    parser.add_argument(
        "--automation-csv",
        type=Path,
        default=Path("D:/crop_nadia/list_for_automation.csv"),
        help="CSV with experiment/sub_experiment/skip rows (shared across datasets).",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        default=None,
        help="Filter experiments (e.g. '1,2,5-8').",
    )
    parser.add_argument(
        "--sub-experiments",
        type=str,
        default=None,
        help="Filter sub-experiments (e.g. '1,2,5-8').",
    )
    parser.add_argument(
        "--no-rescale",
        action="store_true",
        help="Disable rescaling of positions to 0..1 (keeps raw coordinates).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for aggregated matrices (omit to save next to each spine file).",
    )
    args = parser.parse_args()

    exp_filter = parse_range_list(args.experiments)
    sub_filter = parse_range_list(args.sub_experiments)

    automation = load_automation_csv(args.automation_csv)
    jobs = build_spine_jobs(args.root, automation, exp_filter, sub_filter)
    if not jobs:
        print(f"No spine files found under {args.root}")
        return

    series_list: list[SpineSeries] = []
    for path, exp_name, sub_name, skip_rows in jobs:
        try:
            series_list.append(
                read_spine_series(
                    path,
                    rescale=not args.no_rescale,
                    skip_rows=skip_rows,
                    experiment=exp_name,
                    sub_experiment=sub_name,
                )
            )
        except Exception as exc:
            print(f"WARN: Failed to read {path}: {exc}")

    if not series_list:
        print("No valid spine series found.")
        return

    if args.output_dir is None:
        for series in series_list:
            out_dir = series.path.parent
            save_series(out_dir / "spine_X.csv", series.x)
            save_series(out_dir / "spine_Y.csv", series.y)
            save_series(out_dir / "spine_Q.csv", series.q)
            save_series(out_dir / "spine_Qt.csv", series.qt)
            save_series(out_dir / "spine_V.csv", series.v)
            save_series(out_dir / "spine_A.csv", series.a)
        print(f"OK: processed {len(series_list)} spine files -> per-file outputs")
    else:
        output_dir = args.output_dir
        save_matrix(output_dir / "spine_X.csv", build_matrix(series_list, "x"))
        save_matrix(output_dir / "spine_Y.csv", build_matrix(series_list, "y"))
        save_matrix(output_dir / "spine_Q.csv", build_matrix(series_list, "q"))
        save_matrix(output_dir / "spine_Qt.csv", build_matrix(series_list, "qt"))
        save_matrix(output_dir / "spine_V.csv", build_matrix(series_list, "v"))
        save_matrix(output_dir / "spine_A.csv", build_matrix(series_list, "a"))
        save_meta(output_dir / "spine_meta.csv", series_list)

        print(f"OK: processed {len(series_list)} spine files -> {output_dir}")


if __name__ == "__main__":
    main()
