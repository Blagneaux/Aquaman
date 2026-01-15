"""Compare generated pressure maps against a reference and export per-frame error plots."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath
from scipy.spatial import cKDTree


def reshape_frame(flat_frame: np.ndarray, num_cols: int, num_rows: int) -> np.ndarray:
    """Convert a flattened (i-major) frame back to a 2D grid (rows, cols)."""
    if flat_frame.size != num_cols * num_rows:
        raise ValueError(f"Frame size {flat_frame.size} does not match grid {num_cols}x{num_rows}.")
    return flat_frame.reshape(num_cols, num_rows).T  # j is inner loop in the writer


def load_maps(path_ref: Path, path_test: Path) -> tuple[np.ndarray, np.ndarray]:
    ref = np.loadtxt(path_ref, delimiter=",")
    test = np.loadtxt(path_test, delimiter=",")
    if ref.shape != test.shape:
        raise ValueError(f"Shape mismatch: ref {ref.shape} vs test {test.shape}")
    return ref, test


def main():
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Compare pressure map CSVs frame by frame.")
    parser.add_argument("--ref", type=Path, default=repo_root / "pressure_map.csv", help="Reference pressure CSV")
    parser.add_argument(
        "--test",
        type=Path,
        default=repo_root / "pressure_from_velocity_map.csv",
        help="Pressure derived from velocity CSV",
    )
    parser.add_argument("--out-dir", type=Path, default=repo_root / "python/vortexAnalysis_v2/error_frames", help="Where to save error plots")
    parser.add_argument("--start", type=int, default=0, help="First frame (0-based, inclusive)")
    parser.add_argument("--end", type=int, default=None, help="Last frame (0-based, inclusive). Default: last available")
    parser.add_argument("--num-cols", type=int, default=256, help="Grid columns (x-direction)")
    parser.add_argument("--num-rows", type=int, default=128, help="Grid rows (y-direction)")
    parser.add_argument("--dpi", type=int, default=120, help="Figure DPI for saved images")
    parser.add_argument("--body-x", type=Path, default=repo_root / "x.csv", help="Body x-coordinate CSV (header row expected)")
    parser.add_argument("--body-y", type=Path, default=repo_root / "y.csv", help="Body y-coordinate CSV (header row expected)")
    parser.add_argument("--no-mask-body", dest="mask_body", action="store_false", help="Disable masking of the body interior")
    parser.add_argument("--no-round-body", dest="round_body", action="store_false", help="Keep body coords as-is (default: round to nearest grid index)")
    parser.add_argument("--body-clear-radius", type=float, default=3.0, help="Set error to zero within this distance (grid cells) from the body contour")
    parser.add_argument("--no-mask-left-of-body", dest="mask_left_of_body", action="store_false", help="Do not zero out the region x <= max(body_x)")
    parser.add_argument("--no-mask-outside-walls", dest="mask_outside_walls", action="store_false", help="Do not zero out regions above top wall or below bottom wall")
    parser.add_argument("--wall-top", type=float, default=36.5, help="Top wall y-coordinate (grid units) for wake-only masking")
    parser.add_argument("--wall-bottom", type=float, default=80.5, help="Bottom wall y-coordinate (grid units) for wake-only masking")
    parser.set_defaults(mask_body=True, round_body=True, mask_left_of_body=True, mask_outside_walls=True)
    args = parser.parse_args()

    ref, test = load_maps(args.ref, args.test)
    n_cells, n_frames = ref.shape
    expected_cells = args.num_cols * args.num_rows
    if n_cells != expected_cells:
        raise ValueError(f"CSV rows ({n_cells}) do not match num_cols*num_rows ({expected_cells}).")

    start = max(args.start, 0)
    end = n_frames - 1 if args.end is None else min(args.end, n_frames - 1)
    if start > end:
        raise ValueError(f"Invalid frame range: start {start} > end {end}.")

    # Load body coordinates if masking is enabled
    body_x = body_y = None
    if args.mask_body:
        body_x = np.genfromtxt(args.body_x, delimiter=",", skip_header=1)
        body_y = np.genfromtxt(args.body_y, delimiter=",", skip_header=1)
        # Ensure shape consistency
        if body_x.shape != body_y.shape:
            raise ValueError(f"Body x/y shape mismatch: {body_x.shape} vs {body_y.shape}")
        # Align frame counts
        max_frames_body = body_x.shape[1]
        if max_frames_body < end + 1:
            end = max_frames_body - 1
            print(f"Truncating end frame to {end} to match body coordinate length.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    diff_all = test - ref
    global_span = np.max(np.abs(diff_all))
    metrics_path = args.out_dir / "error_metrics.csv"
    # Precompute grid points for masking
    xs = np.arange(args.num_cols)
    ys = np.arange(args.num_rows)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    eps = 1e-8
    with metrics_path.open("w", encoding="utf-8") as f:
        f.write("frame,rmse,mae,max_abs,ref_min,ref_max,ref_mean,ref_std\n")
        for frame in range(start, end + 1):
            diff = diff_all[:, frame]
            ref_frame = ref[:, frame]
            if args.mask_body and body_x is not None:
                poly_x = body_x[:, frame]
                poly_y = body_y[:, frame]
                if getattr(args, "round_body", True):
                    poly_x = np.round(poly_x)
                    poly_y = np.round(poly_y)
                polygon = np.column_stack([poly_x, poly_y])
                if polygon.shape[0] >= 3:
                    # Build mask over the grid
                    mask_flat = MplPath(polygon).contains_points(points)
                    diff = diff.copy()
                    ref_frame = ref_frame.copy()
                    diff[mask_flat] = 0.0  # interior counts as zero error
                    ref_frame[mask_flat] = 0.0
                    if args.body_clear_radius > 0:
                        tree = cKDTree(polygon)
                        distances, _ = tree.query(points)
                        near_body = distances < args.body_clear_radius
                        diff[near_body] = 0.0
                        ref_frame[near_body] = 0.0
                    if args.mask_left_of_body:
                        x_max = np.max(poly_x)
                        xs_flat = points[:, 0]
                        left_mask = xs_flat <= x_max
                        diff[left_mask] = 0.0
                        ref_frame[left_mask] = 0.0
            # Mask exterior above top wall or below bottom wall to focus on wake
            if args.mask_outside_walls:
                if args.wall_top is not None:
                    y_top_mask = points[:, 1] <= args.wall_top
                    diff = diff.copy()
                    ref_frame = ref_frame.copy()
                    diff[y_top_mask] = 0.0
                    ref_frame[y_top_mask] = 0.0
                if args.wall_bottom is not None:
                    y_bottom_mask = points[:, 1] >= args.wall_bottom
                    diff = diff.copy()
                    ref_frame = ref_frame.copy()
                    diff[y_bottom_mask] = 0.0
                    ref_frame[y_bottom_mask] = 0.0

            rmse = np.sqrt(np.mean(diff**2))
            mae = np.mean(np.abs(diff))
            max_abs = np.max(np.abs(diff))
            ref_min = float(np.min(ref_frame))
            ref_max = float(np.max(ref_frame))
            ref_mean = float(np.mean(ref_frame))
            ref_std = float(np.std(ref_frame))
            f.write(f"{frame},{rmse},{mae},{max_abs},{ref_min},{ref_max},{ref_mean},{ref_std}\n")

            frame_grid = reshape_frame(diff_all[:, frame], args.num_cols, args.num_rows)
            poly_x = poly_y = None
            polygon = None
            if args.mask_body and body_x is not None and body_x.shape[0] >= 3:
                poly_x = body_x[:, frame]
                poly_y = body_y[:, frame]
                if getattr(args, "round_body", True):
                    poly_x = np.round(poly_x)
                    poly_y = np.round(poly_y)
                polygon = np.column_stack([poly_x, poly_y])
                if polygon.shape[0] >= 3:
                    mask_flat = MplPath(polygon).contains_points(points)
                    mask_grid = mask_flat.reshape(args.num_cols, args.num_rows).T
                    frame_grid = np.where(mask_grid, 0.0, frame_grid)
                    if args.body_clear_radius > 0:
                        tree = cKDTree(polygon)
                        distances, _ = tree.query(points)
                        near_body = distances < args.body_clear_radius
                        near_mask = near_body.reshape(args.num_cols, args.num_rows).T
                        frame_grid = np.where(near_mask, 0.0, frame_grid)
                    if args.mask_left_of_body:
                        x_max = np.max(poly_x)
                        xs_grid = points[:, 0].reshape(args.num_cols, args.num_rows).T
                        left_mask = xs_grid <= x_max
                        frame_grid = np.where(left_mask, 0.0, frame_grid)
            # Apply top/bottom wall masks to the visualization as well
            if args.mask_outside_walls:
                y_grid = points[:, 1].reshape(args.num_cols, args.num_rows).T
                if args.wall_top is not None:
                    frame_grid = np.where(y_grid <= args.wall_top, 0.0, frame_grid)
                if args.wall_bottom is not None:
                    frame_grid = np.where(y_grid >= args.wall_bottom, 0.0, frame_grid)
            fig, ax = plt.subplots(figsize=(8, 4))
            span = global_span if global_span > 11 else 1.0
            im = ax.imshow(
                frame_grid,
                cmap="coolwarm",
                origin="lower",
                vmin=-span,
                vmax=span,
                aspect="auto",
            )
            if polygon is not None and polygon.shape[0] >= 3:
                ax.fill(poly_x, poly_y, facecolor="gray", alpha=0.15, edgecolor="none")
                ax.plot(poly_x, poly_y, color="black", linewidth=0.7)
            ax.set_title(f"Error map (frame {frame})")
            ax.set_xlabel("x index")
            ax.set_ylabel("y index")
            fig.colorbar(im, ax=ax, label="pressure error")
            fig.tight_layout()
            out_path = args.out_dir / f"error_frame_{frame:04d}.png"
            fig.savefig(out_path, dpi=args.dpi)
            plt.close(fig)

    print(f"Saved metrics to {metrics_path}")
    print(f"Saved frames {start}..{end} to {args.out_dir}")


if __name__ == "__main__":
    main()
