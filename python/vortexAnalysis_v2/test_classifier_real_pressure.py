import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from nptdms import TdmsFile
except ImportError:
    TdmsFile = None

try:
    from scipy.signal import butter, filtfilt
except ImportError:
    butter = None
    filtfilt = None

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

from compute_pressure_modes import (
    fill_internal_nans_1d,
    find_modes_file,
    format_energy_tag,
    normalize_series,
    resample_series,
    zero_mean_series,
    has_tail_spike,
)

CAMBRIDGE_RC = {
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "text.usetex": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "lines.linewidth": 0.8,
    "lines.markersize": 3.5,
    "lines.markeredgewidth": 0.6,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
}


def apply_cambridge_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(CAMBRIDGE_RC)


def save_figure_multi(fig, save_path: Path) -> None:
    base = save_path
    if base.suffix:
        base = base.with_suffix("")
    fig.savefig(f"{base}.eps", format="eps", bbox_inches="tight")
    fig.savefig(f"{base}.pdf", format="pdf", bbox_inches="tight")
    if save_path.suffix:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")


@dataclass
class SnapshotMeta:
    dataset: str
    model: str
    experiment: int
    snapshot: int
    sensor_id: int
    sensor_x: float
    sensor_y: float
    start_frame: int
    end_frame: int
    length: int
    track_path: str
    pressure_path: str


@dataclass
class PredictionRow:
    dataset: str
    architecture: str
    experiment: int
    snapshot: int
    sensor: str
    model_true: str
    start_frame: int
    end_frame: int
    length: int
    score: float
    prob: float
    pred_label: str


def time_to_seconds(time_str: str) -> float:
    h, m, s = map(int, time_str.split(':'))
    return (h * 3600 + m * 60 + s) / 4.0


def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 2) -> tuple[np.ndarray, np.ndarray]:
    if butter is None:
        raise RuntimeError("scipy is required for filtering. Install scipy.")
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return b, a


def bandpass_filter(data: np.ndarray, lowcut: float = 0.3, highcut: float = 30.0, fs: float = 500.0, order: int = 2) -> np.ndarray:
    if filtfilt is None:
        raise RuntimeError("scipy is required for filtering. Install scipy.")
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, data)


def sensor_label(sensor_x: float, sensor_y: float, fallback_idx: int) -> str:
    xi = int(round(sensor_x))
    yi = int(round(sensor_y))
    if (xi, yi) == (198, 80):
        return "S1"
    if (xi, yi) == (146, 80):
        return "S2"
    if (xi, yi) == (200, 39):
        return "S4"
    return f"S{fallback_idx}"


def load_snapshot_meta(path: Path) -> list[SnapshotMeta]:
    rows: list[SnapshotMeta] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append(
                    SnapshotMeta(
                        dataset=row.get("dataset", ""),
                        model=row.get("model", ""),
                        experiment=int(float(row.get("experiment", "0"))),
                        snapshot=int(float(row.get("snapshot", "0"))),
                        sensor_id=int(float(row.get("sensor_id", "-1"))),
                        sensor_x=float(row.get("sensor_x", "nan")),
                        sensor_y=float(row.get("sensor_y", "nan")),
                        start_frame=int(float(row.get("start_frame", "0"))),
                        end_frame=int(float(row.get("end_frame", "0"))),
                        length=int(float(row.get("length", "0"))),
                        track_path=row.get("track_path", ""),
                        pressure_path=row.get("pressure_path", ""),
                    )
                )
            except ValueError:
                continue
    return rows


def find_meta_index(
    meta_path: Path,
    experiment: int,
    snapshot: int,
    sensor_id: int,
    start_frame: int,
    end_frame: int,
) -> tuple[int, int, int] | None:
    match_fallback: tuple[int, int, int] | None = None
    with meta_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            try:
                exp = int(float(row.get("experiment", "0")))
                snap = int(float(row.get("snapshot", "0")))
                sensor = int(float(row.get("sensor_id", "-1")))
                s_frame = int(float(row.get("start_frame", "0")))
                e_frame = int(float(row.get("end_frame", "0")))
                length = int(float(row.get("length", "0")))
            except ValueError:
                continue
            if exp == experiment and snap == snapshot and sensor == sensor_id:
                if s_frame == start_frame and e_frame == end_frame:
                    return idx, length, s_frame
                if match_fallback is None:
                    match_fallback = (idx, length, s_frame)
    return match_fallback


def load_simulation_snapshot(
    snapshot_dir: Path,
    dataset: str,
    model: str,
    experiment: int,
    snapshot: int,
    sensor_id: int,
    start_frame: int,
    end_frame: int,
) -> tuple[np.ndarray, int] | None:
    meta_path = snapshot_dir / f"pressure_snapshots_{dataset}_{model}_meta.csv"
    matrix_path = snapshot_dir / f"pressure_snapshots_{dataset}_{model}.csv"
    if not meta_path.exists() or not matrix_path.exists():
        return None
    meta_info = find_meta_index(meta_path, experiment, snapshot, sensor_id, start_frame, end_frame)
    if meta_info is None:
        return None
    col_idx, length, sim_start_frame = meta_info
    matrix = np.genfromtxt(matrix_path, delimiter=",")
    if matrix.size == 0:
        return None
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if col_idx >= matrix.shape[1]:
        return None
    series = matrix[:length, col_idx].astype(np.float64, copy=False)
    return fill_internal_nans_1d(series), sim_start_frame


def infer_dataset_root(meta: SnapshotMeta) -> Path | None:
    if not meta.track_path:
        return None
    path = Path(meta.track_path)
    if len(path.parents) < 3:
        return None
    return path.parents[3]


def parse_dataset_roots(value: str | None, datasets: list[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    if value:
        for chunk in value.split(";"):
            chunk = chunk.strip()
            if not chunk:
                continue
            if "=" in chunk:
                name, path = chunk.split("=", 1)
                roots[name.strip()] = Path(path.strip())
    for dataset in datasets:
        if dataset not in roots:
            roots[dataset] = Path(f"D:/crop_{dataset}")
    return roots


def read_timestamps(path: Path) -> Any:
    if pd is None:
        raise RuntimeError("pandas is required to read timestamps. Install pandas.")
    df = pd.read_csv(path)
    if "start_time" in df.columns:
        df["start_time"] = df["start_time"].apply(time_to_seconds)
    if "end_time" in df.columns:
        df["end_time"] = df["end_time"].apply(time_to_seconds)
    return df


def load_tdms_channel(tdms_path: Path, sensor_name: str) -> np.ndarray:
    if TdmsFile is None:
        raise RuntimeError("nptdms is required to read TDMS. Install nptdms.")
    tdms_file = TdmsFile.read(tdms_path)
    for group in tdms_file.groups()[1:]:
        for channel in group.channels():
            if channel.name == sensor_name:
                return np.asarray(channel.data)
    raise ValueError(f"Sensor {sensor_name} not found in {tdms_path}")


def frame_to_sample(frame_idx: int, frame_rate: float, fs_raw: float) -> int:
    return int(round(frame_idx * fs_raw / frame_rate))


def plot_debug_steps(
    dataset: str,
    experiment: int,
    snapshot: int,
    sensor_name: str,
    start_time: float,
    fs_raw: float,
    frame_rate: float,
    skipped: np.ndarray,
    segment: np.ndarray,
    segment_start_sample: int,
    skip_samples: int,
    fish_sim: np.ndarray | None,
    fish_start_frame: int | None,
    cyl_sim: np.ndarray | None,
    cyl_start_frame: int | None,
) -> None:
    import matplotlib.pyplot as plt

    apply_cambridge_style()
    t_skip = start_time + skip_samples / fs_raw + np.arange(skipped.size) / fs_raw
    seg_start_time = start_time + (skip_samples + segment_start_sample) / fs_raw
    t_segment = seg_start_time + np.arange(segment.size) / fs_raw

    processed = normalize_series(zero_mean_series(segment.copy()))

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=False)

    axes[0].plot(t_skip, skipped, linewidth=1.0)
    axes[0].axvline(seg_start_time, color="red", linestyle="--", linewidth=1.0)
    axes[0].axvline(seg_start_time + segment.size / fs_raw, color="red", linestyle="--", linewidth=1.0)
    axes[0].set_title("After startIndex skip (red = tracking window)")
    axes[0].set_ylabel("pressure")

    axes[1].plot(t_segment, segment, linewidth=1.0)
    axes[1].set_title("Tracking window (raw)")
    axes[1].set_ylabel("pressure")

    axes[2].plot(t_segment, processed, linewidth=1.0, label="real (normalized)")
    if fish_sim is not None and fish_sim.size and fish_start_frame is not None:
        fish_time = start_time + skip_samples / fs_raw + (fish_start_frame + np.arange(fish_sim.size)) / frame_rate
        fish_norm = normalize_series(zero_mean_series(fish_sim))
        axes[2].plot(fish_time, fish_norm, linestyle="--", linewidth=1.0, label="fish sim")
    if cyl_sim is not None and cyl_sim.size and cyl_start_frame is not None:
        cyl_time = start_time + skip_samples / fs_raw + (cyl_start_frame + np.arange(cyl_sim.size)) / frame_rate
        cyl_norm = normalize_series(zero_mean_series(cyl_sim))
        axes[2].plot(cyl_time, cyl_norm, linestyle="--", linewidth=1.0, label="cylinder sim")
    axes[2].set_title("Tracking window (normalized, no resample)")
    axes[2].set_xlabel("time (s)")
    axes[2].set_ylabel("pressure")
    axes[2].legend(loc="upper right")

    fig.suptitle(f"{dataset} exp={experiment} snap={snapshot} sensor={sensor_name}", fontsize=12)
    plt.tight_layout()
    plt.show()
    plt.close(fig)


class LinearClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, hidden2_size: int, activation: str) -> None:
        super().__init__()
        if activation == "tanh":
            act = nn.Tanh()
        elif activation == "leaky_relu":
            act = nn.LeakyReLU(0.1)
        else:
            act = nn.ReLU()
        if hidden2_size > 0:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                act,
                nn.Linear(hidden_size, hidden2_size),
                act,
                nn.Linear(hidden2_size, 1),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                act,
                nn.Linear(hidden_size, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


class SimpleConvClassifier(nn.Module):
    def __init__(self, input_len: int, conv_channels: int, kernel_size: int) -> None:
        super().__init__()
        kernel_size = min(kernel_size, input_len)
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(1, conv_channels, kernel_size=kernel_size)
        out_len = input_len - kernel_size + 1
        self.fc = nn.Linear(conv_channels * out_len, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.flatten(1)
        return self.fc(x).squeeze(1)


def run_model(
    model: nn.Module,
    X: np.ndarray,
    loss_name: str,
    device: str,
    is_conv: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if torch is None:
        raise RuntimeError("PyTorch is required. Install torch.")
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(X).float()
        if is_conv:
            tensor = tensor.unsqueeze(1)
        tensor = tensor.to(device)
        logits = model(tensor).cpu().numpy()
    if loss_name == "bce":
        probs = torch.sigmoid(torch.from_numpy(logits)).numpy()
    else:
        probs = logits
    preds = (probs >= 0.5).astype(np.int64)
    return logits, probs, preds


def remap_state_dict_prefix(state_dict: dict, prefix: str) -> dict:
    if not state_dict:
        return state_dict
    if all(k.startswith(prefix) for k in state_dict.keys()):
        return state_dict
    if all(k.split(".")[0].isdigit() for k in state_dict.keys()):
        return {f"{prefix}{k}": v for k, v in state_dict.items()}
    return state_dict


def infer_hidden_size(state_dict: dict) -> int:
    for key in ("0.weight", "net.0.weight"):
        if key in state_dict:
            return int(state_dict[key].shape[0])
    raise ValueError("Unable to infer hidden size from state_dict.")


def build_snapshots(
    meta_rows: list[SnapshotMeta],
    dataset_root: Path,
    snapshot_dir: Path,
    timestamps_root: Path,
    tdms_root: Path,
    fs_raw: float,
    frame_rate: float,
    interp: str,
    target_length: int,
    min_length: int,
    tail_spike_filter: bool,
    tail_fraction: float,
    tail_ratio: float,
    tail_min: int,
    frame_offset: int,
    tdms_subdir: str,
    timestamps_subdir: str,
    timestamps_prefix: str,
    debug_plot_first: bool,
) -> tuple[np.ndarray, np.ndarray, list[PredictionRow]]:
    snapshots: list[np.ndarray] = []
    labels: list[int] = []
    meta_out: list[PredictionRow] = []
    debug_done = False

    tdms_cache: dict[tuple[Path, int, str], np.ndarray] = {}
    timestamps_cache: dict[tuple[Path, int], Any] = {}
    pressure_cache: dict[tuple[Path, int, int, str], tuple[np.ndarray, np.ndarray, int]] = {}

    for meta in meta_rows:
        if meta.length < min_length:
            continue
        sensor_name = sensor_label(meta.sensor_x, meta.sensor_y, meta.sensor_id)
        root = dataset_root

        if root is None:
            continue

        timestamps_key = (timestamps_root, meta.experiment)
        if timestamps_key not in timestamps_cache:
            timestamps_path = timestamps_root / timestamps_subdir / f"{timestamps_prefix}{meta.experiment}.csv"
            if not timestamps_path.exists():
                print(f"WARN: Missing timestamps file: {timestamps_path}")
                continue
            timestamps_cache[timestamps_key] = read_timestamps(timestamps_path)

        df = timestamps_cache[timestamps_key]
        if meta.snapshot - 1 < 0 or meta.snapshot - 1 >= len(df):
            print(f"WARN: Snapshot index {meta.snapshot} out of range for {root} experiment {meta.experiment}")
            continue
        row = df.iloc[meta.snapshot - 1]
        try:
            start_time = float(row["start_time"])
            end_time = float(row["end_time"])
            start_index = int(float(row.get("start_frame", 0)))
        except Exception:
            print(f"WARN: Missing timestamp columns in {root} experiment {meta.experiment}")
            continue

        pressure_key = (root, meta.experiment, meta.snapshot, sensor_name)
        if pressure_key not in pressure_cache:
            tdms_key = (tdms_root, meta.experiment, sensor_name)
            if tdms_key not in tdms_cache:
                tdms_path = tdms_root / tdms_subdir / f"{meta.experiment}.tdms"
                if not tdms_path.exists():
                    print(f"WARN: Missing TDMS file: {tdms_path}")
                    continue
                raw_channel = load_tdms_channel(tdms_path, sensor_name)
                tdms_cache[tdms_key] = raw_channel
            else:
                raw_channel = tdms_cache[tdms_key]

            filtered = bandpass_filter(raw_channel, highcut=9.0, fs=fs_raw)
            start_sample = int(start_time * fs_raw)
            end_sample = int(end_time * fs_raw)
            cropped_full = filtered[start_sample:end_sample]
            skip_samples = frame_to_sample(start_index, frame_rate, fs_raw)
            if skip_samples > 0:
                cropped_skip = cropped_full[skip_samples:]
            else:
                cropped_skip = cropped_full
            pressure_cache[pressure_key] = (cropped_full, cropped_skip, skip_samples)

        cropped_full, pressure_exp, skip_samples = pressure_cache[pressure_key]
        if pressure_exp.size == 0:
            continue

        start_frame = meta.start_frame - frame_offset
        end_frame = meta.end_frame - frame_offset
        if end_frame < start_frame:
            continue
        if start_frame < 0:
            start_frame = 0
        start_sample = frame_to_sample(start_frame, frame_rate, fs_raw)
        end_sample = frame_to_sample(end_frame + 1, frame_rate, fs_raw)
        if start_sample >= pressure_exp.size:
            continue
        if end_sample > pressure_exp.size:
            end_sample = pressure_exp.size

        segment = np.asarray(pressure_exp[start_sample:end_sample], dtype=np.float64)
        if segment.size == 0:
            continue

        if debug_plot_first and not debug_done:
            fish_sim_info = load_simulation_snapshot(
                snapshot_dir,
                meta.dataset,
                "fish",
                meta.experiment,
                meta.snapshot,
                meta.sensor_id,
                meta.start_frame,
                meta.end_frame,
            )
            cyl_sim_info = load_simulation_snapshot(
                snapshot_dir,
                meta.dataset,
                "cylinder",
                meta.experiment,
                meta.snapshot,
                meta.sensor_id,
                meta.start_frame,
                meta.end_frame,
            )
            fish_sim = fish_sim_info[0] if fish_sim_info is not None else None
            fish_start = fish_sim_info[1] if fish_sim_info is not None else None
            cyl_sim = cyl_sim_info[0] if cyl_sim_info is not None else None
            cyl_start = cyl_sim_info[1] if cyl_sim_info is not None else None
            debug_done = True
            plot_debug_steps(
                dataset=meta.dataset,
                experiment=meta.experiment,
                snapshot=meta.snapshot,
                sensor_name=sensor_name,
                start_time=start_time,
                fs_raw=fs_raw,
                frame_rate=frame_rate,
                skipped=pressure_exp,
                segment=segment,
                segment_start_sample=start_sample,
                skip_samples=skip_samples,
                fish_sim=fish_sim,
                fish_start_frame=fish_start,
                cyl_sim=cyl_sim,
                cyl_start_frame=cyl_start,
            )

        if tail_spike_filter:
            if has_tail_spike(segment, tail_fraction, tail_ratio, tail_min):
                continue

        segment = fill_internal_nans_1d(segment)
        resampled = resample_series(segment, target_length, interp)
        resampled = zero_mean_series(resampled)
        resampled = normalize_series(resampled)

        snapshots.append(resampled)
        labels.append(0 if meta.model == "fish" else 1)
        meta_out.append(
            PredictionRow(
                dataset=meta.dataset,
                architecture="",
                experiment=meta.experiment,
                snapshot=meta.snapshot,
                sensor=sensor_name,
                model_true=meta.model,
                start_frame=meta.start_frame,
                end_frame=meta.end_frame,
                length=meta.length,
                score=0.0,
                prob=0.0,
                pred_label="",
            )
        )

    if not snapshots:
        return np.empty((0, target_length)), np.empty((0,), dtype=np.int64), []

    return np.vstack(snapshots), np.asarray(labels, dtype=np.int64), meta_out


def save_predictions(path: Path, rows: list[PredictionRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "dataset",
                "architecture",
                "experiment",
                "snapshot",
                "sensor",
                "model_true",
                "start_frame",
                "end_frame",
                "length",
                "score",
                "prob",
                "pred_label",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.dataset,
                    row.architecture,
                    row.experiment,
                    row.snapshot,
                    row.sensor,
                    row.model_true,
                    row.start_frame,
                    row.end_frame,
                    row.length,
                    row.score,
                    row.prob,
                    row.pred_label,
                ]
            )


def load_prediction_probs(path: Path) -> list[float]:
    probs: list[float] = []
    if not path.exists():
        return probs
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = row.get("prob", "")
            try:
                probs.append(float(raw))
            except ValueError:
                continue
    return probs


def plot_prediction_distribution(
    dist_summary: dict[str, dict[str, tuple[int, int, int, int]]],
    datasets: list[str],
    architectures: list[str],
    threshold: float,
    save_path: Path | None,
) -> None:
    import matplotlib.pyplot as plt

    apply_cambridge_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = {
        "fish": "#4C78A8",
        "none": "#BAB0AC",
        "cylinder": "#F58518",
    }

    for idx, dataset in enumerate(datasets[:4]):
        ax = axes[idx // 2, idx % 2]
        arch_stats = dist_summary.get(dataset, {})

        x_labels = [a for a in architectures if a in arch_stats]
        x = np.arange(len(x_labels))
        if len(x_labels) == 0:
            ax.set_title(f"{dataset} (no data)")
            ax.axis("off")
            continue

        fish_vals = []
        none_vals = []
        cyl_vals = []
        fish_counts = []
        none_counts = []
        cyl_counts = []
        totals = []
        for arch in x_labels:
            fish_count, none_count, cyl_count, total = arch_stats[arch]
            fish_counts.append(fish_count)
            none_counts.append(none_count)
            cyl_counts.append(cyl_count)
            totals.append(total)
            if total <= 0:
                fish_vals.append(0.0)
                none_vals.append(0.0)
                cyl_vals.append(0.0)
            else:
                fish_vals.append(fish_count / total)
                none_vals.append(none_count / total)
                cyl_vals.append(cyl_count / total)

        bottom = np.zeros(len(x_labels))
        ax.bar(x, fish_vals, label=f"fish > {threshold*100}%", color=colors["fish"], bottom=bottom)
        for i, frac in enumerate(fish_vals):
            if frac > 0:
                ax.text(
                    x[i],
                    bottom[i] + frac / 2,
                    f"{fish_counts[i]}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                )
        bottom += np.array(fish_vals)
        ax.bar(x, none_vals, label="none", color=colors["none"], bottom=bottom)
        for i, frac in enumerate(none_vals):
            if frac > 0:
                ax.text(
                    x[i],
                    bottom[i] + frac / 2,
                    f"{none_counts[i]}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                )
        bottom += np.array(none_vals)
        ax.bar(x, cyl_vals, label=f"cylinder > {threshold*100}%", color=colors["cylinder"], bottom=bottom)
        for i, frac in enumerate(cyl_vals):
            if frac > 0:
                ax.text(
                    x[i],
                    bottom[i] + frac / 2,
                    f"{cyl_counts[i]}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                )

        ax.set_xticks(x, x_labels, rotation=20)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("fraction of samples")
        ax.set_title(f"{dataset} (threshold={threshold:.2f})")
        ax.grid(axis="y", alpha=0.2)

    for idx in range(len(datasets), 4):
        axes[idx // 2, idx % 2].axis("off")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_figure_multi(fig, save_path)
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


def parse_threshold_list(value: str) -> list[float]:
    thresholds: list[float] = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        thresholds.append(float(chunk))
    return thresholds


def plot_threshold_curves(
    series: dict[str, dict[str, dict[str, list[float]]]],
    datasets: list[str],
    thresholds: list[float],
    save_path: Path | None,
) -> None:
    import matplotlib.pyplot as plt

    apply_cambridge_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    class_colors = {"fish": "#4C78A8", "none": "#BAB0AC", "cylinder": "#F58518"}
    linestyles = ["-", "--", ":"]

    for idx, dataset in enumerate(datasets[:4]):
        ax = axes[idx // 2, idx % 2]
        dataset_series = series.get(dataset, {})
        if not dataset_series:
            ax.set_title(f"{dataset} (no data)")
            ax.axis("off")
            continue

        for arch_idx, (arch, class_series) in enumerate(dataset_series.items()):
            ls = linestyles[arch_idx % len(linestyles)]
            for label in ("fish", "none", "cylinder"):
                values = class_series.get(label, [])
                if not values:
                    continue
                ax.plot(
                    thresholds,
                    values,
                    linestyle=ls,
                    color=class_colors[label],
                    label=f"{arch} {label}" if arch_idx == 0 else None,
                )

        ax.set_title(dataset)
        ax.set_xlabel("confidence threshold")
        ax.set_ylabel("fraction of samples")
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.2)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_figure_multi(fig, save_path)
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Test trained classifiers on real pressure signals.")
    parser.add_argument(
        "--datasets",
        type=str,
        default="nadia,thomas,boai,full",
        help="Comma-separated dataset list to process",
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=repo_root / "python/vortexAnalysis_v2/vortex_outputs/pressure_snapshots",
        help="Directory with snapshot meta CSVs",
    )
    parser.add_argument(
        "--modes-dir",
        type=Path,
        default=repo_root / "python/vortexAnalysis_v2/vortex_outputs/pressure_modes",
        help="Directory with modes and weights",
    )
    parser.add_argument(
        "--modes-energy",
        type=float,
        default=None,
        help="Energy tag for modes/weights (e.g. 0.99 or 99).",
    )
    parser.add_argument(
        "--dataset-roots",
        type=str,
        default=None,
        help="Override dataset roots: 'nadia=D:/crop_nadia;thomas=D:/crop_thomas;...'",
    )
    parser.add_argument(
        "--tdms-subdir",
        type=str,
        default="TDMS",
        help="TDMS subdirectory under each dataset root",
    )
    parser.add_argument(
        "--timestamps-subdir",
        type=str,
        default="timestamps",
        help="Timestamps subdirectory under each dataset root",
    )
    parser.add_argument(
        "--timestamps-prefix",
        type=str,
        default="timestamps",
        help="Prefix for timestamp files (e.g. timestamps{experiment}.csv)",
    )
    parser.add_argument(
        "--timestamps-dataset",
        type=str,
        default="nadia",
        help="Dataset name to use for timestamps (shared across datasets).",
    )
    parser.add_argument(
        "--tdms-dataset",
        type=str,
        default="nadia",
        help="Dataset name to use for TDMS (shared across datasets).",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=100,
        help="Minimum snapshot length in frames to keep",
    )
    parser.add_argument(
        "--interp",
        type=str,
        choices=["linear", "cubic"],
        default="cubic",
        help="Interpolation method for resampling",
    )
    parser.add_argument(
        "--frame-rate",
        type=float,
        default=100.0,
        help="Frame rate of simulation/tracking (Hz)",
    )
    parser.add_argument(
        "--fs-raw",
        type=float,
        default=500.0,
        help="Sampling rate of raw pressure (Hz)",
    )
    parser.add_argument(
        "--tail-spike-filter",
        action="store_true",
        help="Enable tail-spike filtering on raw snapshots",
    )
    parser.add_argument(
        "--tail-frac",
        type=float,
        default=0.15,
        help="Tail fraction for spike detection",
    )
    parser.add_argument(
        "--tail-ratio",
        type=float,
        default=5.0,
        help="Spike ratio threshold",
    )
    parser.add_argument(
        "--tail-min",
        type=int,
        default=5,
        help="Minimum tail length for spike detection",
    )
    parser.add_argument(
        "--frame-offset",
        type=int,
        default=0,
        help="Optional offset to subtract from snapshot frames before slicing raw pressure",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device",
    )
    parser.add_argument(
        "--loss",
        type=str,
        choices=["bce", "mse"],
        default="mse",
        help="Loss type used during training (affects probability calculation)",
    )
    parser.add_argument(
        "--plot-distribution",
        action="store_true",
        help="Plot 2x2 distribution of confident predictions per dataset/architecture.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.75,
        help="Threshold for confident predictions (prob >= t => cylinder, prob <= 1-t => fish).",
    )
    parser.add_argument(
        "--confidence-thresholds",
        type=str,
        default=None,
        help="Comma-separated thresholds for curve plot (e.g. 0.5,0.6,0.75).",
    )
    parser.add_argument(
        "--plot-save",
        type=Path,
        default=None,
        help="Optional path to save the distribution plot instead of showing it.",
    )
    parser.add_argument(
        "--plot-from-files",
        action="store_true",
        help="Build distribution plot from saved prediction CSVs (skip inference).",
    )
    parser.add_argument(
        "--plot-threshold-curves",
        action="store_true",
        help="Plot fraction vs threshold curves from saved prediction CSVs.",
    )
    parser.add_argument(
        "--debug-plot-first",
        action="store_true",
        help="Plot raw pressure cropping steps for the first processed snapshot (no resample in plot).",
    )
    args = parser.parse_args()

    if torch is None:
        raise RuntimeError("PyTorch is required. Install torch.")
    if nn is None:
        raise RuntimeError("PyTorch is required. Install torch.")

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    dataset_roots = parse_dataset_roots(args.dataset_roots, datasets)
    suffix = f"_{format_energy_tag(args.modes_energy)}" if args.modes_energy is not None else ""

    dist_summary: dict[str, dict[str, tuple[int, int, int, int]]] = {}
    architectures_order = ["linear"]

    if args.plot_from_files:
        if args.plot_threshold_curves:
            if not args.confidence_thresholds:
                raise RuntimeError("--confidence-thresholds is required for threshold curves.")
            thresholds = parse_threshold_list(args.confidence_thresholds)
            curve_series: dict[str, dict[str, dict[str, list[float]]]] = {}
            for dataset in datasets:
                for arch_name in architectures_order:
                    pred_path = args.modes_dir / f"real_pressure_predictions_{arch_name}_{dataset}{suffix}.csv"
                    probs = load_prediction_probs(pred_path)
                    if not probs:
                        continue
                    probs_arr = np.clip(np.array(probs, dtype=np.float64), 0.0, 1.0)
                    fish_vals: list[float] = []
                    none_vals: list[float] = []
                    cyl_vals: list[float] = []
                    for thr in thresholds:
                        fish_conf = int(np.sum(probs_arr <= (1.0 - thr)))
                        cyl_conf = int(np.sum(probs_arr >= thr))
                        none_conf = int(len(probs_arr) - fish_conf - cyl_conf)
                        total = len(probs_arr)
                        fish_vals.append(fish_conf / total if total else 0.0)
                        none_vals.append(none_conf / total if total else 0.0)
                        cyl_vals.append(cyl_conf / total if total else 0.0)
                    curve_series.setdefault(dataset, {})[arch_name] = {
                        "fish": fish_vals,
                        "none": none_vals,
                        "cylinder": cyl_vals,
                    }

            plot_threshold_curves(curve_series, datasets, thresholds, args.plot_save)
            return

        for dataset in datasets:
            for arch_name in architectures_order:
                pred_path = args.modes_dir / f"real_pressure_predictions_{arch_name}_{dataset}{suffix}.csv"
                probs = load_prediction_probs(pred_path)
                if not probs:
                    continue
                probs_arr = np.clip(np.array(probs, dtype=np.float64), 0.0, 1.0)
                fish_conf = int(np.sum(probs_arr <= (1.0 - args.confidence_threshold)))
                cyl_conf = int(np.sum(probs_arr >= args.confidence_threshold))
                none_conf = int(len(probs_arr) - fish_conf - cyl_conf)
                dist_summary.setdefault(dataset, {})[arch_name] = (
                    fish_conf,
                    none_conf,
                    cyl_conf,
                    len(probs_arr),
                )

        if args.plot_distribution:
            plot_prediction_distribution(
                dist_summary,
                datasets,
                architectures_order,
                args.confidence_threshold,
                args.plot_save,
            )
        return

    for dataset in datasets:
        fish_meta_path = args.snapshot_dir / f"pressure_snapshots_{dataset}_fish_meta.csv"
        cyl_meta_path = args.snapshot_dir / f"pressure_snapshots_{dataset}_cylinder_meta.csv"
        if not fish_meta_path.exists() or not cyl_meta_path.exists():
            print(f"WARN: Missing meta files for {dataset}")
            continue

        meta_rows = load_snapshot_meta(fish_meta_path) + load_snapshot_meta(cyl_meta_path)
        if not meta_rows:
            print(f"WARN: No meta rows for {dataset}")
            continue

        root = dataset_roots.get(dataset)
        if root is None or not root.exists():
            inferred = infer_dataset_root(meta_rows[0])
            if inferred is not None:
                root = inferred
        if root is None:
            print(f"WARN: Missing dataset root for {dataset}")
            continue

        timestamps_root = dataset_roots.get(args.timestamps_dataset, None)
        if timestamps_root is None or not timestamps_root.exists():
            timestamps_root = root
        tdms_root = dataset_roots.get(args.tdms_dataset, None)
        if tdms_root is None or not tdms_root.exists():
            tdms_root = root

        modes_path = find_modes_file(args.modes_dir, dataset, args.modes_energy)
        if modes_path is None or not modes_path.exists():
            print(f"WARN: Missing modes for {dataset} in {args.modes_dir}")
            continue
        data = np.load(modes_path)
        modes = data["modes"]
        target_length = int(data.get("target_length", modes.shape[0]))

        X_full, y, meta_out = build_snapshots(
            meta_rows=meta_rows,
            dataset_root=root,
            snapshot_dir=args.snapshot_dir,
            timestamps_root=timestamps_root,
            tdms_root=tdms_root,
            fs_raw=args.fs_raw,
            frame_rate=args.frame_rate,
            interp=args.interp,
            target_length=target_length,
            min_length=args.min_length,
            tail_spike_filter=args.tail_spike_filter,
            tail_fraction=args.tail_frac,
            tail_ratio=args.tail_ratio,
            tail_min=args.tail_min,
            frame_offset=args.frame_offset,
            tdms_subdir=args.tdms_subdir,
            timestamps_subdir=args.timestamps_subdir,
            timestamps_prefix=args.timestamps_prefix,
            debug_plot_first=args.debug_plot_first,
        )

        if X_full.size == 0:
            print(f"WARN: No snapshots built for {dataset}")
            continue

        coeffs = modes.T @ X_full.T
        X_modes = coeffs.T

        architectures = []

        # Linear on modes
        linear_weights = args.modes_dir / f"classifier_best_weights_{dataset}{suffix}.pt"
        if linear_weights.exists():
            state = torch.load(linear_weights, map_location=args.device)
            if "hidden_size" in state:
                hidden = int(state["hidden_size"])
            else:
                hidden = infer_hidden_size(state["state_dict"])
            model = LinearClassifier(X_modes.shape[1], hidden)
            linear_state = remap_state_dict_prefix(state["state_dict"], "net.")
            model.load_state_dict(linear_state)
            model.to(args.device)
            architectures.append(("linear", model, X_modes, False))
        else:
            print(f"WARN: Missing linear weights for {dataset}: {linear_weights}")

        for arch_name, model, X_input, is_conv in architectures:
            logits, probs, preds = run_model(model, X_input, args.loss, args.device, is_conv=is_conv)
            pred_labels = np.where(preds == 0, "fish", "cylinder")

            probs_clamped = np.clip(probs, 0.0, 1.0)
            fish_conf = int(np.sum(probs_clamped <= (1.0 - args.confidence_threshold)))
            cyl_conf = int(np.sum(probs_clamped >= args.confidence_threshold))
            none_conf = int(len(probs_clamped) - fish_conf - cyl_conf)
            dist_summary.setdefault(dataset, {})[arch_name] = (
                fish_conf,
                none_conf,
                cyl_conf,
                len(probs_clamped),
            )

            rows: list[PredictionRow] = []
            for base_row, score, prob, pred_label in zip(meta_out, logits, probs, pred_labels):
                rows.append(
                    PredictionRow(
                        dataset=base_row.dataset,
                        architecture=arch_name,
                        experiment=base_row.experiment,
                        snapshot=base_row.snapshot,
                        sensor=base_row.sensor,
                        model_true=base_row.model_true,
                        start_frame=base_row.start_frame,
                        end_frame=base_row.end_frame,
                        length=base_row.length,
                        score=float(score),
                        prob=float(prob),
                        pred_label=str(pred_label),
                    )
                )

            out_path = args.modes_dir / f"real_pressure_predictions_{arch_name}_{dataset}{suffix}.csv"
            save_predictions(out_path, rows)
            print(f"OK: {dataset} {arch_name}: samples={len(y)} -> {out_path}")

    if args.plot_distribution:
        plot_prediction_distribution(
            dist_summary,
            datasets,
            architectures_order,
            args.confidence_threshold,
            args.plot_save,
        )


if __name__ == "__main__":
    main()
