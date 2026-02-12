import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

NUM_COLS = 256  # x-direction
NUM_ROWS = 128  # y-direction

Q_REFERENCE_PATH = "Q-criterion.csv"
Q_RECOMPUTED_PATH = "Q_criterion_test_map.csv"

Q_THRESHOLD_REFERENCE = 0.002
Q_THRESHOLD_RECOMPUTED = 0.002
FRAME_OFFSET_RECOMPUTED = 0  # use recomputed frame (t + offset) against reference frame t
PAUSE_SECONDS = 0.05
Q_VMIN = -0.01
Q_VMAX = 0.01


def reshape_frame(flat_frame, num_cols=NUM_COLS, num_rows=NUM_ROWS):
    """Convert flattened i-major frame to a 2D (rows, cols) grid."""
    if flat_frame.size != num_cols * num_rows:
        raise ValueError(f"Frame size {flat_frame.size} does not match grid {num_cols}x{num_rows}.")
    return flat_frame.reshape(num_cols, num_rows).T


def contour_vortices(Q_map, threshold):
    """Return a binary mask of candidate vortex regions where Q > threshold."""
    Q_clean = np.nan_to_num(Q_map, nan=0.0)
    return Q_clean > threshold


def safe_remove_contour(contour_set):
    """Remove an existing contour set in a matplotlib-version-safe way."""
    if contour_set is None:
        return None
    try:
        contour_set.remove()
    except Exception:
        if hasattr(contour_set, "collections"):
            for collection in contour_set.collections:
                collection.remove()
    return None


def contour_count(contour_set):
    if contour_set is None or not getattr(contour_set, "allsegs", None):
        return 0
    return len(contour_set.allsegs[0])


def fmt_metric(value):
    if np.isnan(value):
        return "nan"
    return f"{value:.3f}"


def frame_range(n_ref, n_recomputed, offset):
    """Compute valid reference frame range for an offset comparison."""
    start = max(0, -offset)
    end = min(n_ref, n_recomputed - offset)  # exclusive
    if start >= end:
        raise ValueError(
            f"No overlapping frames with offset={offset}: ref={n_ref} frames, recomputed={n_recomputed} frames."
        )
    return range(start, end)


Q_reference = pd.read_csv(Q_REFERENCE_PATH, header=None).to_numpy()
Q_recomputed = pd.read_csv(Q_RECOMPUTED_PATH, header=None).to_numpy()

if Q_reference.shape[0] != NUM_COLS * NUM_ROWS:
    raise ValueError(
        f"Reference Q row count {Q_reference.shape[0]} does not match grid {NUM_COLS}x{NUM_ROWS}."
    )
if Q_recomputed.shape[0] != NUM_COLS * NUM_ROWS:
    raise ValueError(
        f"Recomputed Q row count {Q_recomputed.shape[0]} does not match grid {NUM_COLS}x{NUM_ROWS}."
    )

if Q_reference.shape[1] != Q_recomputed.shape[1]:
    print(
        f"Warning: frame count mismatch: ref={Q_reference.shape[1]}, recomputed={Q_recomputed.shape[1]}. "
        "Using overlapping frames only."
    )

compare_frames = frame_range(Q_reference.shape[1], Q_recomputed.shape[1], FRAME_OFFSET_RECOMPUTED)

fig, (ax_ref, ax_recomp, ax_cmp) = plt.subplots(1, 3, figsize=(14, 5))
plt.ion()
plt.show(block=False)

ref_img = ax_ref.imshow(np.zeros((NUM_ROWS, NUM_COLS)), cmap="seismic", origin="lower", vmin=Q_VMIN, vmax=Q_VMAX)
recomp_img = ax_recomp.imshow(np.zeros((NUM_ROWS, NUM_COLS)), cmap="seismic", origin="lower", vmin=Q_VMIN, vmax=Q_VMAX)

cmp_cmap = ListedColormap(["#202020", "#2ca02c", "#ff7f0e", "#f5f5f5"])
cmp_img = ax_cmp.imshow(
    np.zeros((NUM_ROWS, NUM_COLS)),
    cmap=cmp_cmap,
    origin="lower",
    vmin=0,
    vmax=3,
    interpolation="nearest",
)
cbar = fig.colorbar(cmp_img, ax=ax_cmp, fraction=0.046, pad=0.04, ticks=[0, 1, 2, 3])
cbar.ax.set_yticklabels(["none", "ref only", "recomputed only", "both"])

for ax in (ax_ref, ax_recomp, ax_cmp):
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")

ref_contour = None
recomp_contour = None

for frame_ref in compare_frames:
    frame_recomp = frame_ref + FRAME_OFFSET_RECOMPUTED

    Q_ref_map = reshape_frame(Q_reference[:, frame_ref])
    Q_recomp_map = reshape_frame(Q_recomputed[:, frame_recomp])

    ref_mask = contour_vortices(Q_ref_map, threshold=Q_THRESHOLD_REFERENCE)
    recomp_mask = contour_vortices(Q_recomp_map, threshold=Q_THRESHOLD_RECOMPUTED)

    ref_img.set_data(Q_ref_map)
    recomp_img.set_data(Q_recomp_map)

    ref_contour = safe_remove_contour(ref_contour)
    recomp_contour = safe_remove_contour(recomp_contour)

    if np.any(ref_mask) and not np.all(ref_mask):
        ref_contour = ax_ref.contour(
            ref_mask,
            levels=[0.5],
            colors="lime",
            linewidths=1.2,
            origin="lower",
        )

    if np.any(recomp_mask) and not np.all(recomp_mask):
        recomp_contour = ax_recomp.contour(
            recomp_mask,
            levels=[0.5],
            colors="orange",
            linewidths=1.2,
            origin="lower",
        )

    overlap_map = ref_mask.astype(np.uint8) + 2 * recomp_mask.astype(np.uint8)
    cmp_img.set_data(overlap_map)

    intersection = np.count_nonzero(ref_mask & recomp_mask)
    union = np.count_nonzero(ref_mask | recomp_mask)
    ref_area = np.count_nonzero(ref_mask)
    recomp_area = np.count_nonzero(recomp_mask)
    iou = intersection / union if union > 0 else np.nan
    precision = intersection / recomp_area if recomp_area > 0 else np.nan
    recall = intersection / ref_area if ref_area > 0 else np.nan

    n_ref = contour_count(ref_contour)
    n_recomp = contour_count(recomp_contour)

    ax_ref.set_title(
        f"Reference Q (t={frame_ref}) | vortices={n_ref} | area={ref_area / ref_mask.size:.2%}"
    )
    ax_recomp.set_title(
        f"Recomputed Q (t={frame_recomp}) | vortices={n_recomp} | area={recomp_area / recomp_mask.size:.2%}"
    )
    ax_cmp.set_title(
        f"Detection overlap | IoU={fmt_metric(iou)} | Precision={fmt_metric(precision)} | Recall={fmt_metric(recall)}"
    )

    fig.tight_layout()
    fig.canvas.draw_idle()
    plt.pause(PAUSE_SECONDS)

plt.ioff()
plt.show()
