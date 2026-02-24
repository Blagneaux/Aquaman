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

from compute_pressure_modes import (
    balance_series,
    extract_series,
    filter_tail_spikes,
    fill_internal_nans_1d,
    load_lengths,
    load_matrix,
    normalize_series,
    resample_series,
    zero_mean_series,
)


@dataclass
class TrainResult:
    conv_channels: int
    kernel_size: int
    lr: float
    weight_decay: float
    epochs: int
    batch_size: int
    train_acc: float
    val_acc: float
    test_acc: float
    best_val_loss: float
    best_epoch: int


@dataclass
class TrainOutcome:
    result: TrainResult
    history: dict | None
    best_state: dict | None
    best_epoch: int


def parse_int_list(value: str, default: list[int]) -> list[int]:
    if not value:
        return default
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def parse_fixed_params(value: str) -> dict[str, float | int]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if len(parts) != 6:
        raise ValueError("fixed params must be 'conv_channels,kernel_size,lr,weight_decay,epochs,batch_size'")
    conv = int(float(parts[0]))
    kernel = int(float(parts[1]))
    lr = float(parts[2])
    wd = float(parts[3])
    epochs = int(float(parts[4]))
    batch = int(float(parts[5]))
    return {
        "conv_channels": conv,
        "kernel_size": kernel,
        "lr": lr,
        "weight_decay": wd,
        "epochs": epochs,
        "batch_size": batch,
    }


def preprocess_series(series_list: list, target_length: int, interp: str) -> np.ndarray:
    if not series_list:
        return np.empty((target_length, 0), dtype=np.float64)
    matrix = np.zeros((target_length, len(series_list)), dtype=np.float64)
    for col, series in enumerate(series_list):
        values = fill_internal_nans_1d(series.values)
        resampled = resample_series(values, target_length, interp)
        resampled = zero_mean_series(resampled)
        resampled = normalize_series(resampled)
        matrix[:, col] = resampled
    return matrix


def build_features(
    fish_series: list,
    cyl_series: list,
    target_length: int,
    interp: str,
) -> tuple[np.ndarray, np.ndarray]:
    fish_matrix = preprocess_series(fish_series, target_length, interp)
    cyl_matrix = preprocess_series(cyl_series, target_length, interp)

    if fish_matrix.size == 0 or cyl_matrix.size == 0:
        return np.empty((0, target_length)), np.empty((0,))

    data_matrix = np.concatenate([fish_matrix, cyl_matrix], axis=1)  # (L, N)
    X = data_matrix.T  # (N, L)
    y = np.concatenate(
        [np.zeros(fish_matrix.shape[1], dtype=np.int64), np.ones(cyl_matrix.shape[1], dtype=np.int64)]
    )
    return X, y


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


def compute_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def confusion_matrix_binary(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    tn = int(np.sum((y_true == 0) & (y_true == y_pred)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return np.array([[tn, fp], [fn, tp]], dtype=np.int64)


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


def train_torch_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    input_len: int,
    conv_channels: int,
    kernel_size: int,
    lr: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
    seed: int,
    device: str,
    return_history: bool = False,
    patience: int = 20,
    min_delta: float = 0.0,
    loss_name: str = "mse",
) -> TrainOutcome:
    if torch is None:
        raise RuntimeError("PyTorch is required for training. Install torch first.")

    torch.manual_seed(seed)

    model = SimpleConvClassifier(input_len, conv_channels, kernel_size).to(device)
    loss_fn = nn.MSELoss() if loss_name == "mse" else nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    def to_tensor(arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(arr).float().unsqueeze(1)

    X_train_t = to_tensor(X_train)
    y_train_t = torch.from_numpy(y_train).float()
    X_val_t = to_tensor(X_val).to(device)
    y_val_t = torch.from_numpy(y_val).float().to(device)
    X_test_t = to_tensor(X_test).to(device)
    y_test_t = torch.from_numpy(y_test).float().to(device)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)

    history = {"train_loss": [], "val_loss": []} if return_history else None
    best_loss = float("inf")
    best_epoch = 1
    best_state = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            train_logits = model(X_train_t.to(device))
            val_logits = model(X_val_t)
            train_loss = float(loss_fn(train_logits, y_train_t.to(device)).item())
            val_loss = float(loss_fn(val_logits, y_val_t).item())

        if return_history:
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

        if val_loss < best_loss - min_delta:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        raw_train = model(X_train_t.to(device)).cpu().numpy()
        raw_val = model(X_val_t).cpu().numpy()
        raw_test = model(X_test_t).cpu().numpy()

    if loss_name == "mse":
        train_scores = raw_train
        val_scores = raw_val
        test_scores = raw_test
    else:
        train_scores = torch.sigmoid(torch.from_numpy(raw_train)).numpy()
        val_scores = torch.sigmoid(torch.from_numpy(raw_val)).numpy()
        test_scores = torch.sigmoid(torch.from_numpy(raw_test)).numpy()

    train_pred = (train_scores >= 0.5).astype(np.int64)
    val_pred = (val_scores >= 0.5).astype(np.int64)
    test_pred = (test_scores >= 0.5).astype(np.int64)

    train_acc = float(np.mean(train_pred == y_train))
    val_acc = float(np.mean(val_pred == y_val))
    test_acc = float(np.mean(test_pred == y_test))

    result = TrainResult(
        conv_channels=conv_channels,
        kernel_size=kernel_size,
        lr=lr,
        weight_decay=weight_decay,
        epochs=epochs,
        batch_size=batch_size,
        train_acc=train_acc,
        val_acc=val_acc,
        test_acc=test_acc,
        best_val_loss=best_loss,
        best_epoch=best_epoch,
    )

    return TrainOutcome(result=result, history=history, best_state=best_state, best_epoch=best_epoch)


def plot_learning_and_confusion(
    history: dict,
    cm: np.ndarray,
    f1_train: float,
    f1_test: float,
    dataset: str,
    result: TrainResult,
    best_epoch: int,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax_loss, ax_cm = axes

    epochs = np.arange(1, len(history["train_loss"]) + 1)
    ax_loss.plot(epochs, history["train_loss"], label="train loss", linewidth=1.4)
    ax_loss.plot(epochs, history["val_loss"], label="val loss", linewidth=1.2)
    ax_loss.axvline(best_epoch, color="black", linestyle="--", linewidth=1.0, label="best epoch")
    if 1 <= best_epoch <= len(history["val_loss"]):
        ax_loss.scatter([best_epoch], [history["val_loss"][best_epoch - 1]], color="black", s=25, zorder=3)
    ax_loss.set_title(
        f"{dataset} learning curve (c={result.conv_channels}, k={result.kernel_size}, lr={result.lr})"
    )
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.legend(loc="upper right")
    ax_loss.grid(alpha=0.2)

    im = ax_cm.imshow(cm, cmap="Blues")
    ax_cm.set_title(f"Confusion matrix\nF1 train={f1_train:.3f} test={f1_test:.3f}")
    ax_cm.set_xlabel("predicted")
    ax_cm.set_ylabel("true")
    ax_cm.set_xticks([0, 1], labels=["fish", "cylinder"])
    ax_cm.set_yticks([0, 1], labels=["fish", "cylinder"])
    for (i, j), val in np.ndenumerate(cm):
        ax_cm.text(j, i, str(val), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()
    plt.close(fig)


def save_results(path: Path, results: list[TrainResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "conv_channels",
                "kernel_size",
                "lr",
                "weight_decay",
                "epochs",
                "batch_size",
                "train_acc",
                "val_acc",
                "test_acc",
                "best_val_loss",
                "best_epoch",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.conv_channels,
                    r.kernel_size,
                    r.lr,
                    r.weight_decay,
                    r.epochs,
                    r.batch_size,
                    r.train_acc,
                    r.val_acc,
                    r.test_acc,
                    r.best_val_loss,
                    r.best_epoch,
                ]
            )


def append_summary(path: Path, dataset: str, result: TrainResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                [
                    "dataset",
                    "conv_channels",
                    "kernel_size",
                    "lr",
                    "weight_decay",
                    "epochs",
                    "batch_size",
                    "train_acc",
                    "val_acc",
                    "test_acc",
                    "best_val_loss",
                    "best_epoch",
                ]
            )
        writer.writerow(
            [
                dataset,
                result.conv_channels,
                result.kernel_size,
                result.lr,
                result.weight_decay,
                result.epochs,
                result.batch_size,
                result.train_acc,
                result.val_acc,
                result.test_acc,
                result.best_val_loss,
                result.best_epoch,
            ]
        )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Train a full-signal CNN classifier on pressure snapshots.")
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=repo_root / "python/vortexAnalysis_v2/vortex_outputs/pressure_snapshots",
        help="Directory with snapshot matrices and meta CSVs",
    )
    parser.add_argument(
        "--modes-dir",
        type=Path,
        default=repo_root / "python/vortexAnalysis_v2/vortex_outputs/pressure_modes",
        help="Directory containing computed mode npz files (used for target length)",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="nadia,thomas,boai,full",
        help="Comma-separated dataset list to process",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=100,
        help="Minimum snapshot length to keep",
    )
    parser.add_argument(
        "--interp",
        type=str,
        choices=["linear", "cubic"],
        default="cubic",
        help="Interpolation method used to resample each snapshot",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.7,
        help="Training split fraction",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.1,
        help="Validation split fraction",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--conv-channels",
        type=str,
        default="1,2,4",
        help="Comma-separated conv channels to try",
    )
    parser.add_argument(
        "--kernel-sizes",
        type=str,
        default="1,3,5",
        help="Comma-separated kernel sizes to try",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="16,32,64",
        help="Comma-separated batch sizes to try",
    )
    parser.add_argument(
        "--lr-min",
        type=float,
        default=1e-5,
        help="Minimum learning rate for Optuna search",
    )
    parser.add_argument(
        "--lr-max",
        type=float,
        default=1e-3,
        help="Maximum learning rate for Optuna search",
    )
    parser.add_argument(
        "--wd-min",
        type=float,
        default=0.0,
        help="Minimum weight decay for Optuna search",
    )
    parser.add_argument(
        "--wd-max",
        type=float,
        default=1e-3,
        help="Maximum weight decay for Optuna search",
    )
    parser.add_argument(
        "--epochs-list",
        type=str,
        default="50,100,200",
        help="Comma-separated epoch counts to try",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience (epochs without val improvement)",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.0,
        help="Minimum val loss improvement to reset patience",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=30,
        help="Optuna trials per dataset",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Optuna timeout in seconds (optional)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device (cpu or cuda)",
    )
    parser.add_argument(
        "--plot-best-metrics",
        action="store_true",
        help="Plot learning curve, F1, and confusion matrix for the best hyperparameters",
    )
    parser.add_argument(
        "--loss",
        type=str,
        choices=["bce", "mse"],
        default="mse",
        help="Loss function to use (bce or mse)",
    )
    parser.add_argument(
        "--skip-optuna",
        action="store_true",
        help="Skip Optuna and use fixed hyperparameters",
    )
    parser.add_argument(
        "--fixed-params",
        type=str,
        default="1,3,0.0005,0.0001,100,16",
        help="Fixed hyperparameters as 'conv_channels,kernel_size,lr,weight_decay,epochs,batch_size'",
    )
    parser.add_argument(
        "--spike-ratio",
        type=float,
        default=5.0,
        help="Tail spike ratio threshold",
    )
    parser.add_argument(
        "--spike-tail-frac",
        type=float,
        default=0.15,
        help="Tail fraction for spike detection",
    )
    parser.add_argument(
        "--spike-min-tail",
        type=int,
        default=5,
        help="Minimum tail length for spike detection",
    )
    parser.add_argument(
        "--no-tail-spike-filter",
        dest="tail_spike_filter",
        action="store_false",
        help="Disable tail spike filtering",
    )
    parser.set_defaults(tail_spike_filter=True)
    args = parser.parse_args()

    if optuna is None:
        raise RuntimeError("Optuna is required. Install optuna before running this script.")
    if torch is None:
        raise RuntimeError("PyTorch is required. Install torch before running this script.")

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    conv_channels_list = parse_int_list(args.conv_channels, [1, 2, 4])
    kernel_sizes = parse_int_list(args.kernel_sizes, [1, 3, 5])
    batch_sizes = parse_int_list(args.batch_sizes, [16, 32, 64])
    epochs_list = parse_int_list(args.epochs_list, [50, 100, 200])

    for dataset in datasets:
        modes_path = args.modes_dir / f"pressure_modes_{dataset}.npz"
        if not modes_path.exists():
            print(f"⚠️  Missing modes for {dataset}: {modes_path}")
            continue

        data = np.load(modes_path)
        if "target_length" in data:
            target_length = int(data["target_length"])
        else:
            target_length = int(data["modes"].shape[0])

        fish_meta = args.snapshot_dir / f"pressure_snapshots_{dataset}_fish_meta.csv"
        fish_mat = args.snapshot_dir / f"pressure_snapshots_{dataset}_fish.csv"
        cyl_meta = args.snapshot_dir / f"pressure_snapshots_{dataset}_cylinder_meta.csv"
        cyl_mat = args.snapshot_dir / f"pressure_snapshots_{dataset}_cylinder.csv"

        if not fish_meta.exists() or not fish_mat.exists():
            print(f"⚠️  Missing fish snapshots for {dataset}")
            continue
        if not cyl_meta.exists() or not cyl_mat.exists():
            print(f"⚠️  Missing cylinder snapshots for {dataset}")
            continue

        fish_lengths = load_lengths(fish_meta)
        cyl_lengths = load_lengths(cyl_meta)
        fish_matrix = load_matrix(fish_mat)
        cyl_matrix = load_matrix(cyl_mat)

        fish_series = extract_series(fish_matrix, fish_lengths, args.min_length, f"{dataset} fish")
        cyl_series = extract_series(cyl_matrix, cyl_lengths, args.min_length, f"{dataset} cylinder")

        if args.tail_spike_filter:
            fish_series, _ = filter_tail_spikes(
                fish_series,
                args.spike_tail_frac,
                args.spike_ratio,
                args.spike_min_tail,
                f"{dataset} fish",
            )
            cyl_series, _ = filter_tail_spikes(
                cyl_series,
                args.spike_tail_frac,
                args.spike_ratio,
                args.spike_min_tail,
                f"{dataset} cylinder",
            )

        fish_series, cyl_series = balance_series(fish_series, cyl_series)
        if not fish_series or not cyl_series:
            print(f"⚠️  Not enough data after filtering for {dataset}")
            continue

        X, y = build_features(fish_series, cyl_series, target_length, args.interp)
        if X.size == 0:
            print(f"⚠️  Empty feature set for {dataset}")
            continue

        X_train, y_train, X_val, y_val, X_test, y_test = split_data_three(
            X, y, args.train_frac, args.val_frac, args.seed
        )

        input_len = X.shape[1]
        kernel_sizes_valid = [k for k in kernel_sizes if k <= input_len]
        if not kernel_sizes_valid:
            kernel_sizes_valid = [1]

        results: list[TrainResult] = []
        if args.skip_optuna:
            best_params = parse_fixed_params(args.fixed_params)
        else:
            def objective(trial: optuna.Trial) -> float:
                conv_channels = trial.suggest_categorical("conv_channels", conv_channels_list)
                kernel_size = trial.suggest_categorical("kernel_size", kernel_sizes_valid)
                batch_size = trial.suggest_categorical("batch_size", batch_sizes)
                epochs = trial.suggest_categorical("epochs", epochs_list)
                lr = trial.suggest_float("lr", args.lr_min, args.lr_max, log=True)
                if args.wd_min <= 0.0 or args.wd_max <= 0.0:
                    wd = trial.suggest_float("weight_decay", args.wd_min, args.wd_max)
                else:
                    wd = trial.suggest_float("weight_decay", args.wd_min, args.wd_max, log=True)

                outcome = train_torch_model(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    X_test,
                    y_test,
                    input_len=input_len,
                    conv_channels=conv_channels,
                    kernel_size=kernel_size,
                    lr=lr,
                    weight_decay=wd,
                    epochs=epochs,
                    batch_size=batch_size,
                    seed=args.seed,
                    device=args.device,
                    return_history=False,
                    patience=args.patience,
                    min_delta=args.min_delta,
                    loss_name=args.loss,
                )
                trial.set_user_attr("train_acc", outcome.result.train_acc)
                trial.set_user_attr("val_acc", outcome.result.val_acc)
                trial.set_user_attr("test_acc", outcome.result.test_acc)
                trial.set_user_attr("best_val_loss", outcome.result.best_val_loss)
                trial.set_user_attr("best_epoch", outcome.result.best_epoch)
                return outcome.result.best_val_loss

            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=args.seed),
            )
            study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)

            for trial in study.trials:
                if trial.values is None:
                    continue
                params = trial.params
                results.append(
                    TrainResult(
                        conv_channels=int(params["conv_channels"]),
                        kernel_size=int(params["kernel_size"]),
                        lr=float(params["lr"]),
                        weight_decay=float(params["weight_decay"]),
                        epochs=int(params["epochs"]),
                        batch_size=int(params["batch_size"]),
                        train_acc=float(trial.user_attrs.get("train_acc", 0.0)),
                        val_acc=float(trial.user_attrs.get("val_acc", 0.0)),
                        test_acc=float(trial.user_attrs.get("test_acc", 0.0)),
                        best_val_loss=float(trial.user_attrs.get("best_val_loss", 0.0)),
                        best_epoch=int(trial.user_attrs.get("best_epoch", 0)),
                    )
                )

            best_params = study.best_params

        best_outcome = train_torch_model(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            input_len=input_len,
            conv_channels=int(best_params["conv_channels"]),
            kernel_size=int(best_params["kernel_size"]),
            lr=float(best_params["lr"]),
            weight_decay=float(best_params["weight_decay"]),
            epochs=int(best_params["epochs"]),
            batch_size=int(best_params["batch_size"]),
            seed=args.seed,
            device=args.device,
            return_history=True,
            patience=args.patience,
            min_delta=args.min_delta,
            loss_name=args.loss,
        )

        print(
            f"✅ {dataset}: samples={len(y)} | best c={best_outcome.result.conv_channels}, "
            f"k={best_outcome.result.kernel_size}, lr={best_outcome.result.lr:.4g}, "
            f"wd={best_outcome.result.weight_decay:.4g}, "
            f"train={best_outcome.result.train_acc:.3f} "
            f"val={best_outcome.result.val_acc:.3f} "
            f"test={best_outcome.result.test_acc:.3f} "
            f"best_epoch={best_outcome.result.best_epoch}"
        )

        out_path = args.modes_dir / f"classifier_fullcnn_results_{dataset}.csv"
        if results:
            save_results(out_path, results)
        summary_path = args.modes_dir / "classifier_fullcnn_results_summary.csv"
        append_summary(summary_path, dataset, best_outcome.result)

        if args.plot_best_metrics and best_outcome.history is not None:
            best_state = best_outcome.best_state
            if best_state is not None:
                model = SimpleConvClassifier(input_len, best_outcome.result.conv_channels, best_outcome.result.kernel_size)
                model.load_state_dict(best_state)
                model.eval()
                with torch.no_grad():
                    raw_train = model(torch.from_numpy(X_train).float().unsqueeze(1)).numpy()
                    raw_test = model(torch.from_numpy(X_test).float().unsqueeze(1)).numpy()
                if args.loss == "mse":
                    train_scores = raw_train
                    test_scores = raw_test
                else:
                    train_scores = torch.sigmoid(torch.from_numpy(raw_train)).numpy()
                    test_scores = torch.sigmoid(torch.from_numpy(raw_test)).numpy()
                train_pred = (train_scores >= 0.5).astype(np.int64)
                test_pred = (test_scores >= 0.5).astype(np.int64)
                f1_train = compute_f1(y_train, train_pred)
                f1_test = compute_f1(y_test, test_pred)
                cm = confusion_matrix_binary(y_test, test_pred)
                plot_learning_and_confusion(
                    best_outcome.history,
                    cm,
                    f1_train,
                    f1_test,
                    dataset,
                    best_outcome.result,
                    best_outcome.best_epoch,
                )

                weights_path = args.modes_dir / f"classifier_fullcnn_best_weights_{dataset}.pt"
                torch.save(
                    {
                        "state_dict": best_state,
                        "best_epoch": best_outcome.best_epoch,
                        "conv_channels": best_outcome.result.conv_channels,
                        "kernel_size": best_outcome.result.kernel_size,
                        "lr": best_outcome.result.lr,
                        "weight_decay": best_outcome.result.weight_decay,
                        "train_acc": best_outcome.result.train_acc,
                        "val_acc": best_outcome.result.val_acc,
                        "test_acc": best_outcome.result.test_acc,
                        "seed": args.seed,
                    },
                    weights_path,
                )
                print(f"✅ Saved best weights to {weights_path}")


if __name__ == "__main__":
    main()
