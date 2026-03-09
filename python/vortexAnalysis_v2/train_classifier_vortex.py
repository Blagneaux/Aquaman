import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import optuna
except ImportError:
    optuna = None

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    torch = None
    nn = None

try:
    import shap
except ImportError:
    shap = None

from compute_pressure_modes import format_energy_tag

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


def compute_shap_values_linear(
    state_dict: dict,
    mean: np.ndarray,
    std: np.ndarray,
    X: np.ndarray,
) -> np.ndarray:
    weight = state_dict.get("weight")
    bias = state_dict.get("bias")
    if weight is None or bias is None:
        raise ValueError("State dict missing weight/bias for linear model.")
    w = weight.detach().cpu().numpy().reshape(-1)
    std_safe = np.where(std == 0, 1.0, std)
    w_x = w / std_safe
    shap_values = (X - mean) * w_x
    return shap_values


def plot_shap_summary(
    shap_values: np.ndarray,
    features: np.ndarray,
    feature_names: list[str],
    dataset: str,
    save_path: Path,
) -> None:
    if shap is None:
        raise RuntimeError("shap is required for SHAP plots. Install shap.")

    import matplotlib.pyplot as plt

    apply_cambridge_style()
    fig, ax = plt.subplots(1, 1, figsize=(12.5, 5.6))

    plt.sca(ax)
    shap.summary_plot(shap_values, features, feature_names=feature_names, show=False)
    ax.set_title(f"{dataset} SHAP summary")
    ax.set_xlabel("SHAP value (negative \u2190 fish | cylinder \u2192 positive)")
    ax.axvline(0.0, color="black", linestyle=":", linewidth=0.8, alpha=0.6)

    # Give more room for feature labels on the left.
    fig.subplots_adjust(left=0.34, right=0.98)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure_multi(fig, save_path)
    plt.close(fig)


@dataclass
class PredictionRow:
    dataset: str
    architecture: str
    experiment: int
    snapshot: int
    sensor_label: str
    source_model: str
    start_frame: int
    end_frame: int
    prob: float


@dataclass
class MetaRow:
    dataset: str
    model: str
    experiment: int
    snapshot: int
    sensor_id: int
    sensor_x: float
    sensor_y: float
    start_frame: int
    end_frame: int
    track_path: Path


def parse_list(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def parse_int(value: str | None, default: int = -1) -> int:
    try:
        return int(float(value)) if value is not None else default
    except ValueError:
        return default


def parse_float(value: str | None, default: float = np.nan) -> float:
    try:
        return float(value) if value is not None else default
    except ValueError:
        return default


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


def load_predictions(path: Path) -> list[PredictionRow]:
    rows: list[PredictionRow] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append(
                    PredictionRow(
                        dataset=row.get("dataset", ""),
                        architecture=row.get("architecture", ""),
                        experiment=parse_int(row.get("experiment")),
                        snapshot=parse_int(row.get("snapshot")),
                        sensor_label=row.get("sensor", ""),
                        source_model=row.get("model_true", "").strip(),
                        start_frame=parse_int(row.get("start_frame")),
                        end_frame=parse_int(row.get("end_frame")),
                        prob=parse_float(row.get("prob")),
                    )
                )
            except Exception:
                continue
    return rows


def load_meta(path: Path) -> list[MetaRow]:
    rows: list[MetaRow] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                MetaRow(
                    dataset=row.get("dataset", ""),
                    model=row.get("model", ""),
                    experiment=parse_int(row.get("experiment")),
                    snapshot=parse_int(row.get("snapshot")),
                    sensor_id=parse_int(row.get("sensor_id")),
                    sensor_x=parse_float(row.get("sensor_x")),
                    sensor_y=parse_float(row.get("sensor_y")),
                    start_frame=parse_int(row.get("start_frame")),
                    end_frame=parse_int(row.get("end_frame")),
                    track_path=Path(row.get("track_path", "")),
                )
            )
    return rows


def build_meta_index(rows: list[MetaRow]) -> tuple[dict[tuple[int, int, str, int, int], MetaRow], dict[tuple[int, int, str], list[MetaRow]]]:
    index: dict[tuple[int, int, str, int, int], MetaRow] = {}
    fallback: dict[tuple[int, int, str], list[MetaRow]] = {}
    for row in rows:
        label = sensor_label(row.sensor_x, row.sensor_y, row.sensor_id)
        key = (row.experiment, row.snapshot, label, row.start_frame, row.end_frame)
        if key not in index:
            index[key] = row
        fallback.setdefault((row.experiment, row.snapshot, label), []).append(row)
    return index, fallback


def select_meta(
    index: dict[tuple[int, int, str, int, int], MetaRow],
    fallback: dict[tuple[int, int, str], list[MetaRow]],
    pred: PredictionRow,
) -> MetaRow | None:
    key = (pred.experiment, pred.snapshot, pred.sensor_label, pred.start_frame, pred.end_frame)
    if key in index:
        return index[key]
    candidates = fallback.get((pred.experiment, pred.snapshot, pred.sensor_label))
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda row: abs(row.start_frame - pred.start_frame) + abs(row.end_frame - pred.end_frame),
    )


def classify_prediction(prob: float, threshold: float) -> str:
    prob_clamped = max(0.0, min(1.0, prob))
    if prob_clamped >= threshold:
        return "cylinder"
    if prob_clamped <= 1.0 - threshold:
        return "fish"
    return "none"


def load_track_rows(path: Path) -> tuple[list[dict[str, float | int | str]], list[str]]:
    rows: list[dict[str, float | int | str]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for row in reader:
            parsed: dict[str, float | int | str] = {}
            for key in fieldnames:
                val = row.get(key, "")
                if key in ("frame", "track_id", "sensor_id", "visible"):
                    parsed[key] = parse_int(val)
                else:
                    parsed[key] = parse_float(val, default=np.nan)
            rows.append(parsed)
    return rows, fieldnames


def load_matching_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def get_float_array(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.array([parse_float(row.get(key), np.nan) for row in rows], dtype=np.float64)


def get_int_array(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.array([parse_int(row.get(key), -1) for row in rows], dtype=np.int64)


def safe_nanmean(values: np.ndarray) -> float:
    return float(np.nanmean(values)) if values.size else 0.0


def safe_nanmax(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    return float(np.nanmax(finite))


def compute_vortex_features(
    rows: list[dict[str, str]],
    snapshot_length: int,
    track_duration: int,
    duration_ratio: float,
) -> dict[str, float]:
    frames = get_int_array(rows, "frame").astype(np.float64)
    core_x = get_float_array(rows, "core_x")
    core_y = get_float_array(rows, "core_y")
    core_q = get_float_array(rows, "core_q")
    radius = get_float_array(rows, "radius")
    area = get_float_array(rows, "area")
    body_distance = get_float_array(rows, "body_distance")
    sensor_distance = get_float_array(rows, "sensor_distance")

    valid = np.isfinite(core_x) & np.isfinite(core_y)
    displacement = 0.0
    path_length = 0.0
    mean_speed = 0.0
    max_speed = 0.0
    mean_core_x = safe_nanmean(core_x)
    mean_core_y = safe_nanmean(core_y)

    if np.any(valid):
        frames_valid = frames[valid]
        x_valid = core_x[valid]
        y_valid = core_y[valid]
        order = np.argsort(frames_valid)
        frames_sorted = frames_valid[order]
        x_sorted = x_valid[order]
        y_sorted = y_valid[order]
        if x_sorted.size >= 2:
            dx = np.diff(x_sorted)
            dy = np.diff(y_sorted)
            dist = np.sqrt(dx * dx + dy * dy)
            frame_dt = np.diff(frames_sorted)
            frame_dt[frame_dt <= 0] = 1.0
            speeds = dist / frame_dt
            path_length = float(np.sum(dist))
            mean_speed = float(np.mean(speeds)) if speeds.size else 0.0
            max_speed = float(np.max(speeds)) if speeds.size else 0.0
            displacement = float(np.sqrt((x_sorted[-1] - x_sorted[0]) ** 2 + (y_sorted[-1] - y_sorted[0]) ** 2))

    features = {
        "duration_frames": float(track_duration),
        "duration_ratio": float(duration_ratio),
        "displacement": displacement,
        "path_length": path_length,
        "mean_speed": mean_speed,
        "max_speed": max_speed,
        "mean_core_q": safe_nanmean(core_q),
        "max_core_q": safe_nanmax(core_q),
        "mean_abs_core_q": safe_nanmean(np.abs(core_q)),
        "mean_radius": safe_nanmean(radius),
        "max_radius": safe_nanmax(radius),
        "mean_area": safe_nanmean(area),
        "max_area": safe_nanmax(area),
        "mean_body_distance": safe_nanmean(body_distance),
        "mean_sensor_distance": safe_nanmean(sensor_distance),
        "mean_core_x": mean_core_x,
        "mean_core_y": mean_core_y,
    }
    return features


def build_feature_matrix(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rows = load_matching_rows(path)
    if not rows:
        return np.empty((0, 0)), np.empty((0,)), []

    required_cols = {"model", "track_id"}
    if not required_cols.issubset(rows[0].keys()):
        raise ValueError(f"{path} is missing required columns {required_cols}.")

    groups: dict[tuple[str, int, int, int, int, int, int, str], list[dict[str, str]]] = {}
    meta_cache: dict[tuple[str, int, int, int, int, int, int, str], dict[str, float]] = {}

    for row in rows:
        model = row.get("model", "")
        if model not in ("fish", "cylinder"):
            continue
        experiment = parse_int(row.get("experiment"), -1)
        snapshot = parse_int(row.get("snapshot"), -1)
        sensor_id = parse_int(row.get("sensor_id"), -1)
        track_id = parse_int(row.get("track_id"), -1)
        start_frame = parse_int(row.get("start_frame"), -1)
        end_frame = parse_int(row.get("end_frame"), -1)
        track_path = row.get("track_path", "")
        if track_id < 0:
            continue
        key = (model, experiment, snapshot, sensor_id, track_id, start_frame, end_frame, track_path)
        groups.setdefault(key, []).append(row)
        if key not in meta_cache:
            snapshot_length = parse_int(row.get("snapshot_length"), 0)
            track_duration = parse_int(row.get("track_duration"), 0)
            duration_ratio = parse_float(row.get("duration_ratio"), 0.0)
            meta_cache[key] = {
                "snapshot_length": float(snapshot_length),
                "track_duration": float(track_duration),
                "duration_ratio": float(duration_ratio),
            }

    feature_rows: list[list[float]] = []
    labels: list[int] = []
    feature_names: list[str] = []

    for key, group_rows in groups.items():
        meta = meta_cache.get(key, {})
        snapshot_length = int(meta.get("snapshot_length", 0))
        track_duration = int(meta.get("track_duration", 0))
        duration_ratio = float(meta.get("duration_ratio", 0.0))
        if snapshot_length <= 0 and group_rows:
            start_frame = parse_int(group_rows[0].get("start_frame"), 0)
            end_frame = parse_int(group_rows[0].get("end_frame"), 0)
            if end_frame >= start_frame:
                snapshot_length = end_frame - start_frame + 1
        if track_duration <= 0:
            frames = get_int_array(group_rows, "frame")
            valid_frames = frames[frames >= 0]
            track_duration = int(np.unique(valid_frames).size)
        if duration_ratio <= 0.0 and snapshot_length > 0:
            duration_ratio = track_duration / snapshot_length

        features = compute_vortex_features(
            group_rows,
            snapshot_length,
            track_duration,
            duration_ratio,
        )
        if not feature_names:
            feature_names = list(features.keys())
        feature_rows.append([features[name] for name in feature_names])
        labels.append(0 if key[0] == "fish" else 1)

    X = np.asarray(feature_rows, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    return X, y, feature_names


def standardize_features(
    X: np.ndarray,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_out = X.astype(np.float64, copy=True)
    if mean is None:
        mean = np.nanmean(X_out, axis=0)
    if std is None:
        std = np.nanstd(X_out, axis=0)
    std = np.where(std == 0, 1.0, std)
    X_out = np.where(np.isfinite(X_out), X_out, mean)
    X_out = (X_out - mean) / std
    return X_out, mean, std


def split_data_three(
    X: np.ndarray,
    y: np.ndarray,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    order = rng.permutation(n)
    train_size = int(round(n * train_frac))
    val_size = int(round(n * val_frac))
    train_size = max(1, min(train_size, n - 2))
    val_size = max(1, min(val_size, n - train_size - 1))
    train_idx = order[:train_size]
    val_idx = order[train_size : train_size + val_size]
    test_idx = order[train_size + val_size :]
    return (
        X[train_idx],
        y[train_idx],
        X[val_idx],
        y[val_idx],
        X[test_idx],
        y[test_idx],
    )


def predict_logits(model: nn.Module, X: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(X).float().to(device)
        logits = model(tensor).cpu().numpy()
    return logits.reshape(-1)


def compute_accuracy(logits: np.ndarray, y: np.ndarray, loss_name: str) -> float:
    probs = logits_to_probs(logits, loss_name)
    preds = (probs >= 0.5).astype(np.int64)
    return float(np.mean(preds == y)) if y.size else 0.0


def logits_to_probs(logits: np.ndarray, loss_name: str) -> np.ndarray:
    if loss_name == "bce":
        probs = 1.0 / (1.0 + np.exp(-logits))
    else:
        probs = logits
    return np.clip(probs, 0.0, 1.0)


def compute_f1(logits: np.ndarray, y: np.ndarray, loss_name: str) -> float:
    probs = logits_to_probs(logits, loss_name)
    preds = (probs >= 0.5).astype(np.int64)
    tp = int(np.sum((preds == 1) & (y == 1)))
    fp = int(np.sum((preds == 1) & (y == 0)))
    fn = int(np.sum((preds == 0) & (y == 1)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def train_linear_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    lr: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
    seed: int,
    device: str,
    loss_name: str,
    patience: int,
    min_delta: float,
    return_history: bool = False,
) -> tuple[dict, dict, int, dict | None]:
    if torch is None or nn is None:
        raise RuntimeError("PyTorch is required for training. Install torch first.")

    torch.manual_seed(seed)
    model = nn.Linear(X_train.shape[1], 1).to(device)

    if loss_name == "mse":
        loss_fn = nn.MSELoss()
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).float()
    X_val_t = torch.from_numpy(X_val).float().to(device)
    y_val_t = torch.from_numpy(y_val).float().to(device)
    X_test_t = torch.from_numpy(X_test).float().to(device)
    y_test_t = torch.from_numpy(y_test).float().to(device)

    batch_size = max(1, min(batch_size, len(X_train_t)))
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)

    history = {"train_loss": [], "val_loss": [], "test_loss": []} if return_history else None
    best_loss = float("inf")
    best_state = None
    best_epoch = 1
    patience_left = patience

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        seen = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb).squeeze(1)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            batch_size = int(xb.shape[0])
            epoch_loss += loss.item() * batch_size
            seen += batch_size

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t).squeeze(1)
            val_loss = loss_fn(val_logits, y_val_t).item()
            test_logits = model(X_test_t).squeeze(1)
            test_loss = loss_fn(test_logits, y_test_t).item()

        if history is not None:
            history["train_loss"].append(epoch_loss / max(seen, 1))
            history["val_loss"].append(float(val_loss))
            history["test_loss"].append(float(test_loss))

        if val_loss + min_delta < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_logits = predict_logits(model, X_train, device)
    val_logits = predict_logits(model, X_val, device)
    test_logits = predict_logits(model, X_test, device)

    metrics = {
        "train_acc": compute_accuracy(train_logits, y_train, loss_name),
        "val_acc": compute_accuracy(val_logits, y_val, loss_name),
        "test_acc": compute_accuracy(test_logits, y_test, loss_name),
        "best_val_loss": float(best_loss),
        "test_f1": compute_f1(test_logits, y_test, loss_name),
    }

    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return metrics, {"state_dict": state}, best_epoch, history


def train_with_optuna(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: str,
    loss_name: str,
    seed: int,
    n_trials: int,
    timeout: int | None,
    patience: int,
    min_delta: float,
    lr_min: float,
    lr_max: float,
    wd_min: float,
    wd_max: float,
    epoch_min: int,
    epoch_max: int,
    batch_choices: list[int],
) -> tuple[dict, dict, int, dict]:
    if optuna is None:
        raise RuntimeError("Optuna is required for hyperparameter search. Install optuna.")

    best_outcome: dict | None = None

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", lr_min, lr_max, log=True)
        wd = trial.suggest_float("weight_decay", wd_min, wd_max, log=True)
        epochs = trial.suggest_int("epochs", epoch_min, epoch_max)
        batch_size = trial.suggest_categorical("batch_size", batch_choices)
        metrics, state, best_epoch, _ = train_linear_model(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            lr=lr,
            weight_decay=wd,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
            device=device,
            loss_name=loss_name,
            patience=patience,
            min_delta=min_delta,
        )
        trial.set_user_attr("metrics", metrics)
        trial.set_user_attr("state", state)
        trial.set_user_attr("best_epoch", best_epoch)
        return metrics["best_val_loss"]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    best_params = study.best_trial.params
    metrics = study.best_trial.user_attrs.get("metrics", {})
    state = study.best_trial.user_attrs.get("state", {})
    best_epoch = int(study.best_trial.user_attrs.get("best_epoch", 1))
    best_outcome = {
        "params": best_params,
        "metrics": metrics,
        "state": state,
        "best_epoch": best_epoch,
    }
    return (
        best_outcome["params"],
        best_outcome["state"],
        best_outcome["best_epoch"],
        best_outcome["metrics"],
    )


def parse_fixed_params(value: str | None) -> dict[str, float | int] | None:
    if not value:
        return None
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if len(parts) != 4:
        raise ValueError("fixed params must be 'lr,weight_decay,epochs,batch_size'")
    return {
        "lr": float(parts[0]),
        "weight_decay": float(parts[1]),
        "epochs": int(float(parts[2])),
        "batch_size": int(float(parts[3])),
    }


def save_summary(path: Path, row: dict[str, float | int | str]) -> None:
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def load_torch_state(path: Path, device: str) -> dict:
    if torch is None:
        raise RuntimeError("PyTorch is required to load weights. Install torch first.")
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def plot_loss_histories(
    histories: dict[str, dict],
    best_epochs: dict[str, int],
    f1_scores: dict[str, float],
    datasets: list[str],
    save_path: Path | None,
) -> None:
    import matplotlib.pyplot as plt

    apply_cambridge_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for idx, dataset in enumerate(datasets[:4]):
        ax = axes[idx // 2, idx % 2]
        history = histories.get(dataset)
        if not history:
            ax.set_title(f"{dataset} (no history)")
            ax.axis("off")
            continue
        train_loss = history.get("train_loss", [])
        test_loss = history.get("test_loss", [])
        epochs = np.arange(1, len(train_loss) + 1)
        ax.plot(epochs, train_loss, label="train")
        ax.plot(epochs, test_loss, label="test")
        best_epoch = best_epochs.get(dataset, 1)
        ax.axvline(best_epoch, color="black", linestyle=":", linewidth=1.0)
        f1 = f1_scores.get(dataset, 0.0)
        ax.set_title(f"{dataset} (F1={f1:.3f})")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.grid(alpha=0.2)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_figure_multi(fig, save_path)
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


def train_from_matching_file(
    matching_path: Path,
    dataset: str,
    tag: str,
    threshold_tag: str,
    args: argparse.Namespace,
) -> tuple[dict | None, int | None, float | None]:
    X, y, feature_names = build_feature_matrix(matching_path)
    if X.size == 0:
        print(f"WARN: No feature data for {dataset} ({matching_path})")
        return None, None, None
    if np.unique(y).size < 2:
        print(f"WARN: Only one class present for {dataset}; skipping training.")
        return None, None, None

    X_train_raw, y_train, X_val_raw, y_val, X_test_raw, y_test = split_data_three(
        X, y, train_frac=args.train_frac, val_frac=args.val_frac, seed=args.seed
    )

    weights_path = args.results_dir / f"vortex_classifier_best_weights_{dataset}{tag}_{threshold_tag}.pt"
    summary_path = args.results_dir / f"vortex_classifier_results_summary{tag}_{threshold_tag}.csv"

    if weights_path.exists() and not args.force_train:
        if torch is None or nn is None:
            raise RuntimeError("PyTorch is required to evaluate existing weights. Install torch.")
        state = load_torch_state(weights_path, args.device)
        saved_features = state.get("feature_names", feature_names)
        if saved_features != feature_names:
            print(f"WARN: Feature list mismatch for {dataset}; using current feature order.")
        saved_mean = state.get("scaler_mean")
        saved_std = state.get("scaler_std")
        if saved_mean is None or saved_std is None:
            X_train, mean, std = standardize_features(X_train_raw)
            X_val, _, _ = standardize_features(X_val_raw, mean, std)
            X_test, _, _ = standardize_features(X_test_raw, mean, std)
        else:
            mean = np.asarray(saved_mean)
            std = np.asarray(saved_std)
            X_train, _, _ = standardize_features(X_train_raw, mean, std)
            X_val, _, _ = standardize_features(X_val_raw, mean, std)
            X_test, _, _ = standardize_features(X_test_raw, mean, std)
        model = nn.Linear(X_train.shape[1], 1).to(args.device)
        model.load_state_dict(state["state_dict"])
        train_logits = predict_logits(model, X_train, args.device)
        val_logits = predict_logits(model, X_val, args.device)
        test_logits = predict_logits(model, X_test, args.device)
        metrics = {
            "train_acc": compute_accuracy(train_logits, y_train, args.loss),
            "val_acc": compute_accuracy(val_logits, y_val, args.loss),
            "test_acc": compute_accuracy(test_logits, y_test, args.loss),
            "best_val_loss": float(state.get("best_val_loss", np.nan)),
            "test_f1": compute_f1(test_logits, y_test, args.loss),
        }
        print(
            f"OK: {dataset} reuse weights -> train={metrics['train_acc']:.3f} "
            f"val={metrics['val_acc']:.3f} test={metrics['test_acc']:.3f}"
        )
        save_summary(
            summary_path,
            {
                "dataset": dataset,
                "n_samples": int(X.shape[0]),
                "n_features": int(X.shape[1]),
                "train_acc": metrics["train_acc"],
                "val_acc": metrics["val_acc"],
                "test_acc": metrics["test_acc"],
                "test_f1": metrics["test_f1"],
                "best_val_loss": metrics["best_val_loss"],
                "lr": state.get("lr", ""),
                "weight_decay": state.get("weight_decay", ""),
                "epochs": state.get("epochs", ""),
                "batch_size": state.get("batch_size", ""),
                "loss": args.loss,
                "reuse": 1,
            },
        )

        history = None
        best_epoch = int(state.get("best_epoch", 1))
        if args.plot_loss:
            params = {
                "lr": float(state.get("lr", 1e-3)),
                "weight_decay": float(state.get("weight_decay", 1e-4)),
                "epochs": int(state.get("epochs", 100)),
                "batch_size": int(state.get("batch_size", 32)),
            }
            metrics, _, best_epoch, history = train_linear_model(
                X_train,
                y_train,
                X_val,
                y_val,
                X_test,
                y_test,
                lr=float(params["lr"]),
                weight_decay=float(params["weight_decay"]),
                epochs=int(params["epochs"]),
                batch_size=int(params["batch_size"]),
                seed=args.seed,
                device=args.device,
                loss_name=args.loss,
                patience=args.patience,
                min_delta=args.min_delta,
                return_history=True,
            )

        if args.plot_shap:
            shap_dir = args.plot_shap_dir or args.results_dir
            shap_path = shap_dir / f"vortex_shap_{dataset}{tag}_{threshold_tag}.png"
            shap_values = compute_shap_values_linear(state["state_dict"], mean, std, X_test_raw)
            shap_dir.mkdir(parents=True, exist_ok=True)
            plot_shap_summary(shap_values, X_test_raw, feature_names, dataset, shap_path)
        return history, best_epoch, metrics.get("test_f1")

    X_train, mean, std = standardize_features(X_train_raw)
    X_val, _, _ = standardize_features(X_val_raw, mean, std)
    X_test, _, _ = standardize_features(X_test_raw, mean, std)

    fixed_params = parse_fixed_params(args.fixed_params)

    if args.skip_optuna:
        params = fixed_params or {"lr": 1e-3, "weight_decay": 1e-4, "epochs": 100, "batch_size": 32}
        metrics, state, best_epoch, history = train_linear_model(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            lr=float(params["lr"]),
            weight_decay=float(params["weight_decay"]),
            epochs=int(params["epochs"]),
            batch_size=int(params["batch_size"]),
            seed=args.seed,
            device=args.device,
            loss_name=args.loss,
            patience=args.patience,
            min_delta=args.min_delta,
            return_history=args.plot_loss,
        )
    else:
        params, state, best_epoch, metrics = train_with_optuna(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            device=args.device,
            loss_name=args.loss,
            seed=args.seed,
            n_trials=args.n_trials,
            timeout=args.timeout,
            patience=args.patience,
            min_delta=args.min_delta,
            lr_min=args.lr_min,
            lr_max=args.lr_max,
            wd_min=args.wd_min,
            wd_max=args.wd_max,
            epoch_min=args.epoch_min,
            epoch_max=args.epoch_max,
            batch_choices=[int(v.strip()) for v in args.batch_sizes.split(",") if v.strip()],
        )
        history = None
        if args.plot_loss:
            metrics, state, best_epoch, history = train_linear_model(
                X_train,
                y_train,
                X_val,
                y_val,
                X_test,
                y_test,
                lr=float(params["lr"]),
                weight_decay=float(params["weight_decay"]),
                epochs=int(params["epochs"]),
                batch_size=int(params["batch_size"]),
                seed=args.seed,
                device=args.device,
                loss_name=args.loss,
                patience=args.patience,
                min_delta=args.min_delta,
                return_history=True,
            )

    weights = {
        "state_dict": state["state_dict"],
        "feature_names": feature_names,
        "scaler_mean": mean,
        "scaler_std": std,
        "lr": float(params["lr"]),
        "weight_decay": float(params["weight_decay"]),
        "epochs": int(params["epochs"]),
        "batch_size": int(params["batch_size"]),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(metrics["best_val_loss"]),
        "test_f1": float(metrics["test_f1"]),
        "loss": args.loss,
    }
    torch.save(weights, weights_path)

    print(
        f"OK: {dataset} trained -> train={metrics['train_acc']:.3f} "
        f"val={metrics['val_acc']:.3f} test={metrics['test_acc']:.3f} "
        f"saved={weights_path}"
    )

    save_summary(
        summary_path,
        {
            "dataset": dataset,
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "train_acc": metrics["train_acc"],
            "val_acc": metrics["val_acc"],
            "test_acc": metrics["test_acc"],
            "test_f1": metrics["test_f1"],
            "best_val_loss": metrics["best_val_loss"],
            "lr": float(params["lr"]),
            "weight_decay": float(params["weight_decay"]),
            "epochs": int(params["epochs"]),
            "batch_size": int(params["batch_size"]),
            "loss": args.loss,
            "reuse": 0,
        },
    )

    if args.plot_shap:
        shap_dir = args.plot_shap_dir or args.results_dir
        shap_path = shap_dir / f"vortex_shap_{dataset}{tag}_{threshold_tag}.png"
        shap_values = compute_shap_values_linear(state["state_dict"], mean, std, X_test_raw)
        shap_dir.mkdir(parents=True, exist_ok=True)
        plot_shap_summary(shap_values, X_test_raw, feature_names, dataset, shap_path)

    return history, best_epoch, metrics.get("test_f1")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Collect vortices from simulations matching real-pressure classifier predictions."
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
        default=0.95,
        help="Energy tag for prediction files (e.g. 0.99 or 99).",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="linear",
        help="Architecture name in prediction CSV filenames.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.95,
        help="Confidence threshold for fish/cylinder classification.",
    )
    parser.add_argument(
        "--min-vortex-fraction",
        type=float,
        default=0.10,
        help="Minimum fraction of snapshot length a vortex must persist to be kept.",
    )
    parser.add_argument(
        "--no-balance",
        action="store_false",
        dest="balance_models",
        help="Disable balancing vortex counts between fish/cylinder.",
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=repo_root / "python/vortexAnalysis_v2/vortex_outputs/pressure_modes",
        help="Directory containing real_pressure_predictions CSVs.",
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=repo_root / "python/vortexAnalysis_v2/vortex_outputs/pressure_snapshots",
        help="Directory with pressure snapshot meta CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "python/vortexAnalysis_v2/vortex_outputs/matching_vortices",
        help="Directory to save matching vortex matrices.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing outputs.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Only build matching vortex matrix (skip training).",
    )
    parser.add_argument(
        "--force-train",
        action="store_true",
        help="Retrain even if weights already exist.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=repo_root / "python/vortexAnalysis_v2/vortex_outputs/vortex_classifier",
        help="Directory to save vortex classifier weights/results.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device for training.",
    )
    parser.add_argument(
        "--loss",
        type=str,
        choices=["bce", "mse"],
        default="mse",
        help="Loss function for training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val/test split.",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.7,
        help="Training fraction.",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.15,
        help="Validation fraction.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=30,
        help="Optuna trials for hyperparameter search.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Optuna timeout in seconds.",
    )
    parser.add_argument(
        "--skip-optuna",
        action="store_true",
        help="Skip Optuna and use fixed params (or defaults).",
    )
    parser.add_argument(
        "--fixed-params",
        type=str,
        default=None,
        help="Fixed params 'lr,weight_decay,epochs,batch_size' if skipping Optuna.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience.",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.0,
        help="Minimum validation loss improvement for early stopping.",
    )
    parser.add_argument(
        "--lr-min",
        type=float,
        default=1e-4,
        help="Minimum learning rate for Optuna (log scale).",
    )
    parser.add_argument(
        "--lr-max",
        type=float,
        default=1e-2,
        help="Maximum learning rate for Optuna (log scale).",
    )
    parser.add_argument(
        "--wd-min",
        type=float,
        default=1e-6,
        help="Minimum weight decay for Optuna (log scale).",
    )
    parser.add_argument(
        "--wd-max",
        type=float,
        default=1e-2,
        help="Maximum weight decay for Optuna (log scale).",
    )
    parser.add_argument(
        "--epoch-min",
        type=int,
        default=50,
        help="Minimum epochs for Optuna.",
    )
    parser.add_argument(
        "--epoch-max",
        type=int,
        default=200,
        help="Maximum epochs for Optuna.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="8,16,32,64",
        help="Comma-separated batch sizes for Optuna.",
    )
    parser.add_argument(
        "--plot-loss",
        action="store_true",
        help="Plot train/test loss histories (2x2) with best epoch and F1.",
    )
    parser.add_argument(
        "--plot-loss-save",
        type=Path,
        default=None,
        help="Optional path to save the loss plot instead of showing it.",
    )
    parser.add_argument(
        "--plot-shap",
        action="store_true",
        help="Plot SHAP summary + bar for each dataset.",
    )
    parser.add_argument(
        "--plot-shap-dir",
        type=Path,
        default=None,
        help="Directory to save SHAP plots (defaults to results dir).",
    )
    args = parser.parse_args()

    datasets = parse_list(args.datasets)
    tag = f"_{format_energy_tag(args.modes_energy)}" if args.modes_energy is not None else ""
    threshold_tag = f"thr{args.confidence_threshold:.2f}"

    args.output_dir.mkdir(parents=True, exist_ok=True)

    track_cache: dict[Path, tuple[list[dict[str, float | int | str]], list[str]]] = {}
    args.results_dir.mkdir(parents=True, exist_ok=True)

    loss_histories: dict[str, dict] = {}
    best_epochs: dict[str, int] = {}
    f1_scores: dict[str, float] = {}

    for dataset in datasets:
        out_path = args.output_dir / f"matching_vortices_{args.architecture}_{dataset}{tag}_{threshold_tag}.csv"
        if out_path.exists() and not args.force:
            print(f"OK: {out_path} already exists. Use --force to overwrite.")
            if not args.skip_train:
                history, best_epoch, f1_score = train_from_matching_file(out_path, dataset, tag, threshold_tag, args)
                if history is not None:
                    loss_histories[dataset] = history
                    if best_epoch is not None:
                        best_epochs[dataset] = best_epoch
                    if f1_score is not None:
                        f1_scores[dataset] = f1_score
            continue

        pred_path = args.predictions_dir / f"real_pressure_predictions_{args.architecture}_{dataset}{tag}.csv"
        if not pred_path.exists():
            print(f"WARN: Missing predictions file: {pred_path}")
            continue

        preds = load_predictions(pred_path)
        if not preds:
            print(f"WARN: No predictions in {pred_path}")
            continue

        fish_meta_path = args.snapshot_dir / f"pressure_snapshots_{dataset}_fish_meta.csv"
        cyl_meta_path = args.snapshot_dir / f"pressure_snapshots_{dataset}_cylinder_meta.csv"
        if not fish_meta_path.exists() or not cyl_meta_path.exists():
            print(f"WARN: Missing snapshot meta files for {dataset}")
            continue

        fish_meta = load_meta(fish_meta_path)
        cyl_meta = load_meta(cyl_meta_path)
        fish_index, fish_fallback = build_meta_index(fish_meta)
        cyl_index, cyl_fallback = build_meta_index(cyl_meta)

        vortex_entries: dict[
            tuple[str, int, int, int, int, int, int, str],
            dict[str, object],
        ] = {}
        track_fields_all: list[str] = []

        total = 0
        matched = 0
        skipped_none = 0
        skipped_short = 0

        for pred in preds:
            total += 1
            label = classify_prediction(pred.prob, args.confidence_threshold)
            if label == "none":
                skipped_none += 1
                continue

            meta_index = fish_index if label == "fish" else cyl_index
            meta_fallback = fish_fallback if label == "fish" else cyl_fallback
            meta_row = select_meta(meta_index, meta_fallback, pred)
            if meta_row is None:
                continue

            track_path = meta_row.track_path
            if not track_path.exists():
                print(f"WARN: Missing track file: {track_path}")
                continue

            if track_path not in track_cache:
                track_rows, track_fields = load_track_rows(track_path)
                track_cache[track_path] = (track_rows, track_fields)
            else:
                track_rows, track_fields = track_cache[track_path]

            if not track_fields_all:
                track_fields_all = list(track_fields)
            else:
                for field in track_fields:
                    if field not in track_fields_all:
                        track_fields_all.append(field)

            snapshot_len = max(0, meta_row.end_frame - meta_row.start_frame + 1)
            if snapshot_len <= 0:
                continue

            track_groups: dict[int, list[dict[str, float | int | str]]] = {}
            track_frames: dict[int, set[int]] = {}

            for row in track_rows:
                frame = parse_int(str(row.get("frame", "")), default=-1)
                if frame < meta_row.start_frame or frame > meta_row.end_frame:
                    continue
                sensor_id = parse_int(str(row.get("sensor_id", "")), default=-1)
                if sensor_id >= 0 and sensor_id != meta_row.sensor_id:
                    continue
                visible = parse_int(str(row.get("visible", "1")), default=1)
                if visible == 0:
                    continue
                track_id = parse_int(str(row.get("track_id", "")), default=-1)
                if track_id < 0:
                    continue
                track_groups.setdefault(track_id, []).append(row)
                track_frames.setdefault(track_id, set()).add(frame)

            for track_id, rows in track_groups.items():
                duration = len(track_frames.get(track_id, set()))
                ratio = duration / snapshot_len if snapshot_len else 0.0
                if ratio < args.min_vortex_fraction:
                    skipped_short += 1
                    continue

                entry_key = (
                    label,
                    pred.experiment,
                    pred.snapshot,
                    meta_row.sensor_id,
                    track_id,
                    meta_row.start_frame,
                    meta_row.end_frame,
                    str(track_path),
                )
                if entry_key in vortex_entries:
                    continue

                vortex_entries[entry_key] = {
                    "model": label,
                    "rows": rows,
                    "duration": duration,
                    "ratio": ratio,
                    "snapshot_length": snapshot_len,
                    "meta": {
                        "dataset": pred.dataset,
                        "source_model": pred.source_model,
                        "architecture": pred.architecture,
                        "experiment": pred.experiment,
                        "snapshot": pred.snapshot,
                        "sensor": pred.sensor_label,
                        "sensor_id": meta_row.sensor_id,
                        "start_frame": meta_row.start_frame,
                        "end_frame": meta_row.end_frame,
                        "pred_prob": pred.prob,
                        "confidence_threshold": args.confidence_threshold,
                        "track_path": str(track_path),
                    },
                }

            matched += 1

        if not vortex_entries:
            print(f"WARN: No matching vortices for {dataset}")
            continue

        if args.balance_models:
            model_keys: dict[str, list[tuple[str, int, int, int, int, int, int, str]]] = {"fish": [], "cylinder": []}
            for key, entry in vortex_entries.items():
                model_keys[entry["model"]].append(key)

            fish_count = len(model_keys["fish"])
            cyl_count = len(model_keys["cylinder"])
            target = min(fish_count, cyl_count)
            if target == 0:
                print(f"WARN: Cannot balance {dataset}; fish={fish_count}, cylinder={cyl_count}")
            else:
                def sort_key(k: tuple[str, int, int, int, int, int, int, str]) -> tuple[float, int, str]:
                    entry = vortex_entries[k]
                    return (-float(entry["ratio"]), -int(entry["duration"]), str(k))

                fish_keep = sorted(model_keys["fish"], key=sort_key)[:target]
                cyl_keep = sorted(model_keys["cylinder"], key=sort_key)[:target]
                keep = set(fish_keep + cyl_keep)
                vortex_entries = {k: v for k, v in vortex_entries.items() if k in keep}

        fieldnames = [
            "dataset",
            "model",
            "source_model",
            "architecture",
            "experiment",
            "snapshot",
            "sensor",
            "sensor_id",
            "start_frame",
            "end_frame",
            "snapshot_length",
            "track_duration",
            "duration_ratio",
            "pred_prob",
            "confidence_threshold",
            "track_path",
        ] + track_fields_all

        kept_vortices = len(vortex_entries)

        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            row_count = 0
            for entry in vortex_entries.values():
                meta = entry["meta"]
                for row in entry["rows"]:
                    out = {
                        "dataset": meta["dataset"],
                        "model": entry["model"],
                        "source_model": meta["source_model"],
                        "architecture": meta["architecture"],
                        "experiment": meta["experiment"],
                        "snapshot": meta["snapshot"],
                        "sensor": meta["sensor"],
                        "sensor_id": meta["sensor_id"],
                        "start_frame": meta["start_frame"],
                        "end_frame": meta["end_frame"],
                        "snapshot_length": entry["snapshot_length"],
                        "track_duration": entry["duration"],
                        "duration_ratio": entry["ratio"],
                        "pred_prob": meta["pred_prob"],
                        "confidence_threshold": meta["confidence_threshold"],
                        "track_path": meta["track_path"],
                    }
                    for key in track_fields_all:
                        out[key] = row.get(key, np.nan)
                    writer.writerow(out)
                    row_count += 1

        print(
            f"OK: {dataset} -> {out_path} | total={total} matched={matched} "
            f"none={skipped_none} short={skipped_short} kept={kept_vortices} rows={row_count}"
        )

        if not args.skip_train:
            history, best_epoch, f1_score = train_from_matching_file(out_path, dataset, tag, threshold_tag, args)
            if history is not None:
                loss_histories[dataset] = history
                if best_epoch is not None:
                    best_epochs[dataset] = best_epoch
                if f1_score is not None:
                    f1_scores[dataset] = f1_score

    if args.plot_loss and loss_histories:
        plot_loss_histories(loss_histories, best_epochs, f1_scores, datasets, args.plot_loss_save)


if __name__ == "__main__":
    main()
