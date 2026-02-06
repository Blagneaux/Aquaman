"""Compare generated pressure maps against a reference and export per-frame error plots."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath
from scipy.spatial import cKDTree
from scipy.ndimage import label


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


def compute_q_criterion(u: np.ndarray, v: np.ndarray, dx: float = 1.0, dy: float = 1.0) -> np.ndarray:
    """Classic Q = 0.5*(||Omega||^2 - ||S||^2) on a 2D (y,x) grid."""
    dudx = np.gradient(u, dx, axis=1)
    dudy = np.gradient(u, dy, axis=0)
    dvdx = np.gradient(v, dx, axis=1)
    dvdy = np.gradient(v, dy, axis=0)

    sxx = dudx
    syy = dvdy
    sxy = 0.5 * (dudy + dvdx)
    omega_xy = 0.5 * (dudy - dvdx)

    omega_sq = 2.0 * omega_xy**2  # Omega_xy and Omega_yx have equal magnitude in 2D
    s_sq = sxx**2 + syy**2 + 2.0 * sxy**2
    return 0.5 * (omega_sq - s_sq)


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
    parser.add_argument("--vx", type=Path, default=repo_root / "velocity_x_map.csv", help="Velocity x CSV (matches pressure grid)")
    parser.add_argument("--vy", type=Path, default=repo_root / "velocity_y_map.csv", help="Velocity y CSV (matches pressure grid)")
    parser.add_argument("--q-threshold", type=float, default=0.02, help="Q-criterion threshold; keep points where Q > threshold")
    print("There is an offset between reference and test frames. \n")
    parser.add_argument("--frame-offset", type=int, default=-4, help="Use test frame (t+offset) against ref frame t")
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
    vx = np.loadtxt(args.vx, delimiter=",")
    vy = np.loadtxt(args.vy, delimiter=",")
    n_cells, n_frames = ref.shape
    expected_cells = args.num_cols * args.num_rows
    if n_cells != expected_cells:
        raise ValueError(f"CSV rows ({n_cells}) do not match num_cols*num_rows ({expected_cells}).")
    if vx.shape != vy.shape:
        raise ValueError(f"Velocity shape mismatch: vx {vx.shape} vs vy {vy.shape}")
    if vx.shape[0] != expected_cells:
        raise ValueError(f"Velocity CSV rows ({vx.shape[0]}) do not match grid ({expected_cells}).")
    if vx.shape[1] != n_frames:
        raise ValueError(f"Velocity frames ({vx.shape[1]}) do not match pressure frames ({n_frames}).")

    # Adjust frame range to respect offset bounds
    offset = args.frame_offset
    start = max(args.start, 0, -offset)
    max_end = n_frames - 1 - max(offset, 0)
    min_end = n_frames - 1 + min(offset, 0)
    effective_end = min(max_end, min_end)
    end = effective_end if args.end is None else min(args.end, effective_end)
    if start > end:
        raise ValueError(f"Invalid frame range: start {start} > end {end}.")

    # Load body coordinates if masking is enabled
    body_x = body_y = None
    max_frames_body = None
    if args.mask_body:
        body_x = np.genfromtxt(args.body_x, delimiter=",", skip_header=1)
        body_y = np.genfromtxt(args.body_y, delimiter=",", skip_header=1)
        # Ensure shape consistency
        if body_x.shape != body_y.shape:
            raise ValueError(f"Body x/y shape mismatch: {body_x.shape} vs {body_y.shape}")
        max_frames_body = body_x.shape[1]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = args.out_dir / "error_metrics.csv"
    # Precompute grid points for masking
    xs = np.arange(args.num_cols)
    ys = np.arange(args.num_rows)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    metrics = []
    with metrics_path.open("w", encoding="utf-8") as f:
        f.write("frame,rmse_norm,mae_norm,max_abs_norm,ref_std,mean_min_core_distance,ref_mean,test_mean\n")
        for frame in range(start, end + 1):
            ref_frame = ref[:, frame]
            test_idx = frame + offset
            test_frame = test[:, test_idx]
            vx_frame = vx[:, test_idx]
            vy_frame = vy[:, test_idx]
            diff = test_frame - ref_frame

            # Build Q mask from velocity (classic definition) and add other masks on top
            vx_grid = reshape_frame(vx_frame, args.num_cols, args.num_rows)
            vy_grid = reshape_frame(vy_frame, args.num_cols, args.num_rows)
            q_grid = compute_q_criterion(vx_grid, vy_grid)
            q_mask_grid = q_grid > args.q_threshold
            mask_flat = q_mask_grid.T.reshape(-1)

            poly_x = poly_y = None
            polygon = None
            if args.mask_body and body_x is not None:
                col = frame if max_frames_body is None else min(frame, max_frames_body - 1)
                poly_x = body_x[:, col]
                poly_y = body_y[:, col]
                if getattr(args, "round_body", True):
                    poly_x = np.round(poly_x)
                    poly_y = np.round(poly_y)
                polygon = np.column_stack([poly_x, poly_y])
                if polygon.shape[0] >= 3:
                    body_mask_flat = MplPath(polygon).contains_points(points)
                    mask_flat = mask_flat & (~body_mask_flat)
                    if args.body_clear_radius > 0:
                        tree = cKDTree(polygon)
                        distances, _ = tree.query(points)
                        near_body = distances < args.body_clear_radius
                        mask_flat = mask_flat & (~near_body)
                    if args.mask_left_of_body:
                        x_max = np.max(poly_x)
                        xs_flat = points[:, 0]
                        left_mask = xs_flat <= x_max
                        mask_flat = mask_flat & (~left_mask)
            # Mask exterior above top wall or below bottom wall to focus on wake
            if args.mask_outside_walls:
                if args.wall_top is not None:
                    y_top_mask = points[:, 1] <= args.wall_top
                    mask_flat = mask_flat & (~y_top_mask)
                if args.wall_bottom is not None:
                    y_bottom_mask = points[:, 1] >= args.wall_bottom
                    mask_flat = mask_flat & (~y_bottom_mask)

            if not np.any(mask_flat):
                rmse_norm = mae_norm = max_abs_norm = ref_std_val = mean_min_dist = ref_mean_val = test_mean_val = np.nan
                metrics.append((rmse_norm, mae_norm, max_abs_norm, ref_std_val, mean_min_dist, ref_mean_val, test_mean_val))
                f.write(f"{frame},{rmse_norm},{mae_norm},{max_abs_norm},{ref_std_val},{mean_min_dist},{ref_mean_val},{test_mean_val}\n")
                continue

            diff_masked = diff[mask_flat]
            ref_masked = ref_frame[mask_flat]
            test_masked = test_frame[mask_flat]

            ref_mean_val = float(np.mean(ref_masked))
            test_mean_val = float(np.mean(test_masked))
            ref_std_val = float(np.std(ref_masked))
            if ref_std_val == 0:
                rmse_norm = mae_norm = max_abs_norm = np.nan
            else:
                rmse_norm = np.sqrt(np.mean(diff_masked**2)) / ref_std_val
                mae_norm = np.mean(np.abs(diff_masked)) / ref_std_val
                max_abs_norm = np.max(np.abs(diff_masked)) / ref_std_val

            # Mean distance between paired per-vortex minima (reference vs reconstruction)
            ref_min_points = []
            test_min_points = []
            labels, n_labels = label(mask_flat.reshape(args.num_cols, args.num_rows).T)
            ref_grid = reshape_frame(ref_frame, args.num_cols, args.num_rows)
            test_grid = reshape_frame(test[:, frame], args.num_cols, args.num_rows)
            for lbl in range(1, n_labels + 1):
                region = labels == lbl
                if not np.any(region):
                    continue
                ref_vals = ref_grid[region]
                test_vals = test_grid[region]
                ref_pos = np.argwhere(region)[np.argmin(ref_vals)]
                test_pos = np.argwhere(region)[np.argmin(test_vals)]
                ref_min_points.append((ref_pos[1], ref_pos[0]))   # (x, y)
                test_min_points.append((test_pos[1], test_pos[0]))  # (x, y)
            if ref_min_points and test_min_points:
                dists = [np.hypot(rx - tx, ry - ty) for (rx, ry), (tx, ty) in zip(ref_min_points, test_min_points)]
                mean_min_dist = float(np.mean(dists))
            else:
                mean_min_dist = np.nan

            metrics.append((rmse_norm, mae_norm, max_abs_norm, ref_std_val, mean_min_dist, ref_mean_val, test_mean_val))
            f.write(f"{frame},{rmse_norm},{mae_norm},{max_abs_norm},{ref_std_val},{mean_min_dist},{ref_mean_val},{test_mean_val}\n")

        
            frame_grid = reshape_frame(diff, args.num_cols, args.num_rows)
            ref_grid = reshape_frame(ref[:, frame], args.num_cols, args.num_rows)
            test_grid = reshape_frame(test_frame, args.num_cols, args.num_rows)
            mask_grid = mask_flat.reshape(args.num_cols, args.num_rows).T
            frame_grid_masked = np.where(mask_grid, frame_grid, 0.0)

            # Locate per-vortex minima inside each connected Q-region
            ref_min_points = []
            test_min_points = []
            if np.any(mask_grid):
                labels, n_labels = label(mask_grid)
                for lbl in range(1, n_labels + 1):
                    region = labels == lbl
                    if not np.any(region):
                        continue
                    ref_vals = ref_grid[region]
                    test_vals = test_grid[region]
                    ref_pos = np.argwhere(region)[np.argmin(ref_vals)]
                    test_pos = np.argwhere(region)[np.argmin(test_vals)]
                    ref_min_points.append((ref_pos[1], ref_pos[0]))   # (x, y)
                    test_min_points.append((test_pos[1], test_pos[0]))  # (x, y)

            fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
            # Use a common symmetric span based on the max absolute value across ref/test
            ref_span = 1
            err_span = ref_span
            cmap_main = "viridis"
            im_ref = axes[0].imshow(
                ref_grid,
                cmap=cmap_main,
                origin="lower",
                vmin=-ref_span,
                vmax=ref_span,
                aspect="auto",
            )
            axes[0].set_title(f"Reference (frame {frame})")
            im_test = axes[1].imshow(
                test_grid,
                cmap=cmap_main,
                origin="lower",
                vmin=-ref_span,
                vmax=ref_span,
                aspect="auto",
            )
            axes[1].set_title("Reconstruction")
            im_err = axes[2].imshow(
                frame_grid_masked,
                cmap=cmap_main,
                origin="lower",
                vmin=-err_span,
                vmax=err_span,
                aspect="auto",
            )
            axes[2].set_title("Error (recon - ref)")
            axes[2].contour(q_mask_grid, levels=[0.5], colors="white", linewidths=0.7, origin="lower")
            # Plot per-vortex core candidates on the error map
            if ref_min_points:
                xs_ref, ys_ref = zip(*ref_min_points)
                axes[2].scatter(xs_ref, ys_ref, color="black", marker="x", s=24, label="Ref min")
            if test_min_points:
                xs_test, ys_test = zip(*test_min_points)
                axes[2].scatter(xs_test, ys_test, color="red", marker="x", s=24, label="Recon min")
            if polygon is not None and polygon.shape[0] >= 3:
                for ax in axes:
                    ax.fill(poly_x, poly_y, facecolor="gray", alpha=0.15, edgecolor="none")
                    ax.plot(poly_x, poly_y, color="black", linewidth=0.7)
            if ref_min_points or test_min_points:
                axes[2].legend(loc="upper right", fontsize=8)
            axes[-1].set_xlabel("x index")
            for ax in axes:
                ax.set_ylabel("y index")
            # Add colorbars
            fig.colorbar(im_ref, ax=axes[0], fraction=0.046, pad=0.01, label="pressure (ref)")
            fig.colorbar(im_test, ax=axes[1], fraction=0.046, pad=0.01, label="pressure (recon)")
            fig.colorbar(im_err, ax=axes[2], fraction=0.046, pad=0.01, label="pressure error")
            fig.tight_layout()
            out_path = args.out_dir / f"error_frame_{frame:04d}.png"
            fig.savefig(out_path, dpi=args.dpi)
            plt.close(fig)

        if metrics:
            arr = np.array(metrics)
            mean_vals = np.nanmean(arr, axis=0)
            f.write(f"mean,{mean_vals[0]},{mean_vals[1]},{mean_vals[2]},{mean_vals[3]},{mean_vals[4]},{mean_vals[5]},{mean_vals[6]}\n")


    print(f"Saved metrics to {metrics_path}")
    print(f"Saved frames {start}..{end} to {args.out_dir}")


if __name__ == "__main__":
    main()
