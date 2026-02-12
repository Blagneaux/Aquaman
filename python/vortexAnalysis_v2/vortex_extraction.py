import argparse
import csv
import os
from dataclasses import dataclass
from math import hypot, pi, sqrt
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_erosion, gaussian_filter, label
from scipy.spatial import cKDTree


@dataclass
class Detection:
    frame: int
    core_x: float
    core_y: float
    core_q: float
    area: int
    radius: float
    boundary: np.ndarray  # (n, 2) as [x, y]
    body_distance: float
    sensor_id: int
    sensor_x: float
    sensor_y: float
    sensor_distance: float
    edge_x: float
    edge_y: float
    wall_top_x1: float
    wall_top_y1: float
    wall_top_x2: float
    wall_top_y2: float
    wall_bottom_x1: float
    wall_bottom_y1: float
    wall_bottom_x2: float
    wall_bottom_y2: float
    visible: bool = True
    track_id: int | None = None
    track_distance: float | None = None


def reshape_frame(flat_frame: np.ndarray, num_cols: int, num_rows: int) -> np.ndarray:
    """Convert flattened (i-major) frame to a 2D (rows, cols) grid."""
    if flat_frame.size != num_cols * num_rows:
        raise ValueError(f"Frame size {flat_frame.size} does not match grid {num_cols}x{num_rows}.")
    return flat_frame.reshape(num_cols, num_rows).T


def parse_sensors(sensor_str: str) -> list[tuple[float, float]]:
    sensors: list[tuple[float, float]] = []
    if not sensor_str:
        return sensors
    for entry in sensor_str.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split(",")
        if len(parts) != 2:
            raise ValueError(f"Invalid sensor entry '{entry}'. Expected 'x,y'.")
        sensors.append((float(parts[0]), float(parts[1])))
    return sensors


def load_body_contour(path_x: Path, path_y: Path, skip_header: bool, round_body: bool) -> tuple[np.ndarray, np.ndarray]:
    skip = 1 if skip_header else 0
    body_x = np.genfromtxt(path_x, delimiter=",", skip_header=skip)
    body_y = np.genfromtxt(path_y, delimiter=",", skip_header=skip)
    if body_x.ndim == 1:
        body_x = body_x[:, None]
    if body_y.ndim == 1:
        body_y = body_y[:, None]
    if body_x.shape != body_y.shape:
        raise ValueError(f"Body x/y shape mismatch: {body_x.shape} vs {body_y.shape}.")
    if round_body:
        body_x = np.round(body_x)
        body_y = np.round(body_y)
    return body_x, body_y


def body_tree_for_frame(body_x: np.ndarray, body_y: np.ndarray, frame: int) -> cKDTree:
    frame_idx = min(frame, body_x.shape[1] - 1)
    poly = np.column_stack([body_x[:, frame_idx], body_y[:, frame_idx]])
    return cKDTree(poly)


def compute_boundary(region_mask: np.ndarray) -> np.ndarray:
    if not np.any(region_mask):
        return np.empty((0, 2), dtype=np.float64)
    eroded = binary_erosion(region_mask, border_value=0)
    boundary = region_mask ^ eroded
    coords = np.argwhere(boundary)  # (y, x)
    if coords.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    return np.column_stack([coords[:, 1], coords[:, 0]]).astype(np.float64)


def closest_sensor(core_x: float, core_y: float, sensors: list[tuple[float, float]]) -> tuple[int, float]:
    if not sensors:
        return -1, float("inf")
    distances = [hypot(core_x - sx, core_y - sy) for sx, sy in sensors]
    idx = int(np.argmin(distances))
    return idx, float(distances[idx])


def closest_edge_point(boundary: np.ndarray, sensor: tuple[float, float]) -> tuple[float, float]:
    if boundary.size == 0:
        return float("nan"), float("nan")
    sx, sy = sensor
    dx = boundary[:, 0] - sx
    dy = boundary[:, 1] - sy
    idx = int(np.argmin(dx * dx + dy * dy))
    return float(boundary[idx, 0]), float(boundary[idx, 1])


def wall_intersections(boundary: np.ndarray, wall_y: float | None, tol: float) -> tuple[float, float, float, float]:
    if wall_y is None or boundary.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    idx = np.where(np.abs(boundary[:, 1] - wall_y) <= tol)[0]
    if idx.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    xs = boundary[idx, 0]
    ys = boundary[idx, 1]
    if idx.size == 1:
        return float(xs[0]), float(ys[0]), float("nan"), float("nan")
    left = int(np.argmin(xs))
    right = int(np.argmax(xs))
    return float(xs[left]), float(ys[left]), float(xs[right]), float(ys[right])


def boundary_segments(boundary: np.ndarray) -> np.ndarray:
    if boundary.size == 0:
        return np.empty((0, 4), dtype=np.float64)
    pts = boundary.astype(np.int64)
    point_set = {(int(x), int(y)) for x, y in pts}
    offsets = [
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    ]
    segments: list[tuple[int, int, int, int]] = []
    for x, y in point_set:
        for dx, dy in offsets:
            nx, ny = x + dx, y + dy
            if (nx, ny) in point_set and (x, y) < (nx, ny):
                segments.append((x, y, nx, ny))
    if not segments:
        return np.empty((0, 4), dtype=np.float64)
    return np.array(segments, dtype=np.float64)


def _orient(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> float:
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def _on_segment(ax: float, ay: float, bx: float, by: float, cx: float, cy: float, eps: float) -> bool:
    return (
        min(ax, bx) - eps <= cx <= max(ax, bx) + eps
        and min(ay, by) - eps <= cy <= max(ay, by) + eps
    )


def segments_intersect_strict(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    cx: float,
    cy: float,
    dx: float,
    dy: float,
    eps: float = 1e-9,
) -> bool:
    o1 = _orient(ax, ay, bx, by, cx, cy)
    o2 = _orient(ax, ay, bx, by, dx, dy)
    o3 = _orient(cx, cy, dx, dy, ax, ay)
    o4 = _orient(cx, cy, dx, dy, bx, by)

    if o1 * o2 < 0 and o3 * o4 < 0:
        return True
    if abs(o1) <= eps and _on_segment(ax, ay, bx, by, cx, cy, eps):
        return True
    if abs(o2) <= eps and _on_segment(ax, ay, bx, by, dx, dy, eps):
        return True
    if abs(o3) <= eps and _on_segment(cx, cy, dx, dy, ax, ay, eps):
        return True
    if abs(o4) <= eps and _on_segment(cx, cy, dx, dy, bx, by, eps):
        return True
    return False


def point_segment_distance(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    vx = bx - ax
    vy = by - ay
    wx = px - ax
    wy = py - ay
    c1 = vx * wx + vy * wy
    if c1 <= 0:
        return hypot(px - ax, py - ay)
    c2 = vx * vx + vy * vy
    if c2 <= c1:
        return hypot(px - bx, py - by)
    t = c1 / c2
    proj_x = ax + t * vx
    proj_y = ay + t * vy
    return hypot(px - proj_x, py - proj_y)


def segments_intersect_or_close(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    cx: float,
    cy: float,
    dx: float,
    dy: float,
    eps: float,
) -> bool:
    if segments_intersect_strict(ax, ay, bx, by, cx, cy, dx, dy):
        return True
    if eps <= 0:
        return False
    d1 = point_segment_distance(ax, ay, cx, cy, dx, dy)
    d2 = point_segment_distance(bx, by, cx, cy, dx, dy)
    d3 = point_segment_distance(cx, cy, ax, ay, bx, by)
    d4 = point_segment_distance(dx, dy, ax, ay, bx, by)
    return min(d1, d2, d3, d4) <= eps


def apply_occlusion_filter(
    detections: list[Detection],
    sensors: list[tuple[float, float]],
    occlusion_eps: float,
    enabled: bool,
) -> list[Detection]:
    if not sensors or not enabled or len(detections) < 2:
        for det in detections:
            det.visible = True
        return detections
    visible_indices: set[int] = set()
    segments_by_det = [boundary_segments(det.boundary) for det in detections]

    for sx, sy in sensors:
        dists = [hypot(det.core_x - sx, det.core_y - sy) for det in detections]
        order = np.argsort(dists)
        for pos, det_idx in enumerate(order):
            if not np.isfinite(dists[det_idx]):
                continue
            ax, ay = sx, sy
            bx, by = detections[det_idx].core_x, detections[det_idx].core_y
            occluded = False
            for closer_idx in order[:pos]:
                segs = segments_by_det[closer_idx]
                if segs.size == 0:
                    continue
                for x0, y0, x1, y1 in segs:
                    if segments_intersect_or_close(ax, ay, bx, by, x0, y0, x1, y1, occlusion_eps):
                        occluded = True
                        break
                if occluded:
                    break
            if not occluded:
                visible_indices.add(det_idx)

    for i, det in enumerate(detections):
        det.visible = i in visible_indices
    return detections


def detect_vortices_in_frame(
    q_grid: np.ndarray,
    frame: int,
    q_threshold: float,
    min_area: int,
    smooth_sigma: float,
    body_tree: cKDTree | None,
    body_distance_min: float,
    sensors: list[tuple[float, float]],
    sensor_max_distance: float,
    wall_top: float | None,
    wall_bottom: float | None,
    wall_tol: float,
) -> list[Detection]:
    q_clean = np.nan_to_num(q_grid, nan=0.0)
    if smooth_sigma > 0:
        q_clean = gaussian_filter(q_clean, sigma=smooth_sigma)

    mask = q_clean > q_threshold
    labels, n_labels = label(mask)
    detections: list[Detection] = []

    for lbl in range(1, n_labels + 1):
        region = labels == lbl
        area = int(np.count_nonzero(region))
        if area < min_area:
            continue

        coords = np.argwhere(region)
        core_idx = int(np.argmax(q_clean[region]))
        core_y, core_x = coords[core_idx]
        core_q = float(q_clean[core_y, core_x])

        body_distance = float("inf")
        if body_tree is not None:
            body_distance, _ = body_tree.query([core_x, core_y])
            body_distance = float(body_distance)
        if body_distance_min > 0 and body_distance < body_distance_min:
            continue

        sensor_id, sensor_distance = closest_sensor(core_x, core_y, sensors)
        if sensors and sensor_max_distance > 0 and sensor_distance > sensor_max_distance:
            continue

        boundary = compute_boundary(region)
        if sensor_id >= 0 and sensors:
            edge_x, edge_y = closest_edge_point(boundary, sensors[sensor_id])
            sensor_x, sensor_y = sensors[sensor_id]
        else:
            edge_x = edge_y = float("nan")
            sensor_x = sensor_y = float("nan")

        wall_top_x1, wall_top_y1, wall_top_x2, wall_top_y2 = wall_intersections(boundary, wall_top, wall_tol)
        wall_bottom_x1, wall_bottom_y1, wall_bottom_x2, wall_bottom_y2 = wall_intersections(boundary, wall_bottom, wall_tol)

        radius = sqrt(area / pi) if area > 0 else float("nan")

        detections.append(
            Detection(
                frame=frame,
                core_x=float(core_x),
                core_y=float(core_y),
                core_q=core_q,
                area=area,
                radius=radius,
                boundary=boundary,
                body_distance=body_distance,
                sensor_id=int(sensor_id),
                sensor_x=float(sensor_x),
                sensor_y=float(sensor_y),
                sensor_distance=float(sensor_distance),
                edge_x=edge_x,
                edge_y=edge_y,
                wall_top_x1=wall_top_x1,
                wall_top_y1=wall_top_y1,
                wall_top_x2=wall_top_x2,
                wall_top_y2=wall_top_y2,
                wall_bottom_x1=wall_bottom_x1,
                wall_bottom_y1=wall_bottom_y1,
                wall_bottom_x2=wall_bottom_x2,
                wall_bottom_y2=wall_bottom_y2,
            )
        )

    return detections


def assign_tracks(
    detections_by_frame: list[list[Detection]],
    frame_numbers: list[int],
    max_track_distance: float,
    max_gap: int,
) -> list[Detection]:
    next_track_id = 0
    active: dict[int, Detection] = {}

    for frame_num, detections in zip(frame_numbers, detections_by_frame):
        if detections:
            positions = np.array([[det.core_x, det.core_y] for det in detections], dtype=np.float64)
            tree = cKDTree(positions)
            candidates: list[tuple[float, int, int]] = []

            for track_id, last_det in active.items():
                if frame_num - last_det.frame > max_gap:
                    continue
                dist, det_idx = tree.query([last_det.core_x, last_det.core_y], distance_upper_bound=max_track_distance)
                if dist != np.inf:
                    candidates.append((float(dist), track_id, int(det_idx)))

            candidates.sort(key=lambda v: v[0])
            assigned_tracks: set[int] = set()
            assigned_dets: set[int] = set()
            for dist, track_id, det_idx in candidates:
                if track_id in assigned_tracks or det_idx in assigned_dets:
                    continue
                assigned_tracks.add(track_id)
                assigned_dets.add(det_idx)
                detections[det_idx].track_id = track_id
                detections[det_idx].track_distance = dist

            for i, det in enumerate(detections):
                if det.track_id is None:
                    det.track_id = next_track_id
                    det.track_distance = float("nan")
                    next_track_id += 1

            for det in detections:
                active[det.track_id] = det

        # Drop tracks that are too old even if no detections this frame
        stale_tracks = [track_id for track_id, det in active.items() if frame_num - det.frame > max_gap]
        for track_id in stale_tracks:
            del active[track_id]

    all_detections = [det for frame_dets in detections_by_frame for det in frame_dets]
    return all_detections


def write_tracks_csv(detections: list[Detection], out_path: Path, dt: float | None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    detections_sorted = sorted(detections, key=lambda d: (d.track_id, d.frame))
    header = [
        "track_id",
        "frame",
        "time",
        "core_x",
        "core_y",
        "core_q",
        "radius",
        "area",
        "body_distance",
        "visible",
        "sensor_id",
        "sensor_x",
        "sensor_y",
        "sensor_distance",
        "edge_x",
        "edge_y",
        "wall_top_x1",
        "wall_top_y1",
        "wall_top_x2",
        "wall_top_y2",
        "wall_bottom_x1",
        "wall_bottom_y1",
        "wall_bottom_x2",
        "wall_bottom_y2",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for det in detections_sorted:
            time_val = det.frame * dt if dt is not None else det.frame
            writer.writerow(
                [
                    det.track_id,
                    det.frame,
                    time_val,
                    det.core_x,
                    det.core_y,
                    det.core_q,
                    det.radius,
                    det.area,
                    det.body_distance,
                    int(det.visible),
                    det.sensor_id,
                    det.sensor_x,
                    det.sensor_y,
                    det.sensor_distance,
                    det.edge_x,
                    det.edge_y,
                    det.wall_top_x1,
                    det.wall_top_y1,
                    det.wall_top_x2,
                    det.wall_top_y2,
                    det.wall_bottom_x1,
                    det.wall_bottom_y1,
                    det.wall_bottom_x2,
                    det.wall_bottom_y2,
                ]
            )


def parse_experiments(value: str | None, default: list[int]) -> list[int]:
    if not value:
        return default
    experiments: list[int] = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_str, end_str = chunk.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            if end < start:
                start, end = end, start
            experiments.extend(range(start, end + 1))
        else:
            experiments.append(int(chunk))
    return experiments


def collect_snapshots(base_path: Path, experiment: int) -> list[str]:
    subpath = base_path / str(experiment)
    subfolders: list[str] = []
    for _, dirnames, _ in os.walk(subpath):
        subfolders.extend(dirnames)
    return subfolders


def extract_vortices_from_q(
    q_path: Path,
    out_csv: Path,
    args: argparse.Namespace,
    body_x_path: Path | None,
    body_y_path: Path | None,
    allow_animation: bool,
) -> bool:
    if not q_path.exists():
        print(f"⚠️  Q-criterion map not found, skipping: {q_path}")
        return False

    q_matrix = np.loadtxt(q_path, delimiter=",")
    n_cells, n_frames = q_matrix.shape
    expected_cells = args.num_cols * args.num_rows
    if n_cells != expected_cells:
        raise ValueError(f"Q-criterion rows ({n_cells}) do not match grid ({expected_cells}).")

    start = max(args.start, 0)
    end = n_frames - 1 if args.end is None else min(args.end, n_frames - 1)
    if start > end:
        raise ValueError(f"Invalid frame range: start {start} > end {end}.")

    sensors = parse_sensors(args.sensors)

    use_body = args.use_body and body_x_path is not None and body_y_path is not None
    if use_body:
        if not body_x_path.exists() or not body_y_path.exists():
            print(f"⚠️  Body contour files not found for {q_path}. Disabling body filter for this snapshot.")
            use_body = False

    body_x = body_y = None
    if use_body:
        body_x, body_y = load_body_contour(body_x_path, body_y_path, args.body_skip_header, args.round_body)

    detections_by_frame: list[list[Detection]] = []
    frame_numbers = list(range(start, end + 1))

    for frame in frame_numbers:
        q_grid = reshape_frame(q_matrix[:, frame], args.num_cols, args.num_rows)
        body_tree = None
        if use_body and body_x is not None and body_y is not None:
            body_tree = body_tree_for_frame(body_x, body_y, frame)

        detections = detect_vortices_in_frame(
            q_grid=q_grid,
            frame=frame,
            q_threshold=args.q_threshold,
            min_area=args.min_area,
            smooth_sigma=args.smooth_sigma,
            body_tree=body_tree,
            body_distance_min=args.body_distance_min if use_body else 0.0,
            sensors=sensors,
            sensor_max_distance=args.sensor_max_distance,
            wall_top=args.wall_top,
            wall_bottom=args.wall_bottom,
            wall_tol=args.wall_tol,
        )

        detections = apply_occlusion_filter(detections, sensors, args.occlusion_eps, args.occlusion)
        detections_by_frame.append(detections)

    tracked = assign_tracks(detections_by_frame, frame_numbers, args.max_track_distance, args.max_gap)
    tracks = build_track_history(tracked)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_tracks_csv(tracked, out_csv, args.dt)
    print(f"✅ Saved vortex time series to {out_csv}")

    if allow_animation and (args.show or args.save_anim):
        anim_path = args.anim_path
        if args.save_anim and anim_path is None:
            anim_path = out_csv.parent / "vortex_tracking.gif"
        animate_tracking(
            q_matrix=q_matrix[:, start : end + 1],
            detections_by_frame=detections_by_frame,
            frame_numbers=frame_numbers,
            tracks=tracks,
            sensors=sensors,
            body_x=body_x,
            body_y=body_y,
            num_cols=args.num_cols,
            num_rows=args.num_rows,
            out_path=anim_path if args.save_anim else None,
            show=args.show,
            fps=args.anim_fps,
            dpi=args.anim_dpi,
        )

    return True


def build_track_history(detections: list[Detection]) -> dict[int, list[Detection]]:
    tracks: dict[int, list[Detection]] = {}
    for det in detections:
        tracks.setdefault(det.track_id, []).append(det)
    for track_id in tracks:
        tracks[track_id].sort(key=lambda d: d.frame)
    return tracks


def animate_tracking(
    q_matrix: np.ndarray,
    detections_by_frame: list[list[Detection]],
    frame_numbers: list[int],
    tracks: dict[int, list[Detection]],
    sensors: list[tuple[float, float]],
    body_x: np.ndarray | None,
    body_y: np.ndarray | None,
    num_cols: int,
    num_rows: int,
    out_path: Path | None,
    show: bool,
    fps: int,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.set_xlim(0, num_cols - 1)
    ax.set_ylim(0, num_rows - 1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Vortex tracking (Q-criterion)")

    frame0 = reshape_frame(q_matrix[:, 0], num_cols, num_rows)
    vmin = np.nanmin(frame0)
    vmax = np.nanmax(frame0)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = -1.0, 1.0
    im = ax.imshow(frame0, origin="lower", cmap="seismic", vmin=vmin, vmax=vmax, alpha=0.9)

    sensor_scatter = None
    if sensors:
        sensor_scatter = ax.scatter([s[0] for s in sensors], [s[1] for s in sensors], c="lime", s=24, label="Sensors")

    body_line = None
    if body_x is not None and body_y is not None:
        body_line, = ax.plot([], [], color="black", linewidth=1.0, alpha=0.7, label="Body")

    core_scatter = ax.scatter([], [], c="black", s=18, label="Core")
    core_scatter_hidden = ax.scatter([], [], c="gray", s=18, label="Core (occluded)")
    edge_scatter = ax.scatter([], [], c="orange", s=14, marker="x", label="Edge->Sensor")
    edge_scatter_hidden = ax.scatter([], [], c="dimgray", s=14, marker="x", label="Edge (occluded)")

    track_lines: dict[int, plt.Line2D] = {}
    cmap = plt.cm.get_cmap("tab20", max(len(tracks), 1))
    for idx, track_id in enumerate(sorted(tracks.keys())):
        track_lines[track_id], = ax.plot([], [], color=cmap(idx), linewidth=1.2, alpha=0.8)

    if sensors or body_line:
        ax.legend(loc="upper right", fontsize=8)

    def update(frame_idx: int):
        frame_num = frame_numbers[frame_idx]
        q_grid = reshape_frame(q_matrix[:, frame_idx], num_cols, num_rows)
        im.set_data(q_grid)

        if body_line is not None:
            body_col = min(frame_num, body_x.shape[1] - 1)
            body_line.set_data(body_x[:, body_col], body_y[:, body_col])

        dets = detections_by_frame[frame_idx] if frame_idx < len(detections_by_frame) else []
        if dets:
            visible = [d for d in dets if d.visible]
            hidden = [d for d in dets if not d.visible]
            if visible:
                core_scatter.set_offsets(np.array([[d.core_x, d.core_y] for d in visible]))
                edge_scatter.set_offsets(np.array([[d.edge_x, d.edge_y] for d in visible]))
            else:
                core_scatter.set_offsets(np.empty((0, 2)))
                edge_scatter.set_offsets(np.empty((0, 2)))
            if hidden:
                core_scatter_hidden.set_offsets(np.array([[d.core_x, d.core_y] for d in hidden]))
                edge_scatter_hidden.set_offsets(np.array([[d.edge_x, d.edge_y] for d in hidden]))
            else:
                core_scatter_hidden.set_offsets(np.empty((0, 2)))
                edge_scatter_hidden.set_offsets(np.empty((0, 2)))
        else:
            core_scatter.set_offsets(np.empty((0, 2)))
            core_scatter_hidden.set_offsets(np.empty((0, 2)))
            edge_scatter.set_offsets(np.empty((0, 2)))
            edge_scatter_hidden.set_offsets(np.empty((0, 2)))

        for track_id, line in track_lines.items():
            track = tracks[track_id]
            xs = [d.core_x for d in track if d.frame <= frame_num]
            ys = [d.core_y for d in track if d.frame <= frame_num]
            line.set_data(xs, ys)

        ax.set_title(f"Vortex tracking (frame {frame_num})")
        return (im, core_scatter, core_scatter_hidden, edge_scatter, edge_scatter_hidden, *track_lines.values())

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(detections_by_frame),
        interval=max(1, int(1000 / max(fps, 1))),
        blit=False,
    )

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = animation.PillowWriter(fps=fps)
        anim.save(out_path, writer=writer, dpi=dpi)
        print(f"Saved animation to {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Detect and track vortices from Q-criterion maps.")
    parser.add_argument("--batch", action="store_true", help="Process all experiments and snapshots")
    parser.add_argument("--q-path", type=Path, default=repo_root / "Q-criterion.csv", help="Q-criterion CSV path")
    parser.add_argument("--num-cols", type=int, default=256, help="Grid columns (x-direction)")
    parser.add_argument("--num-rows", type=int, default=128, help="Grid rows (y-direction)")
    parser.add_argument("--start", type=int, default=0, help="First frame (0-based)")
    parser.add_argument("--end", type=int, default=None, help="Last frame (0-based, inclusive)")
    parser.add_argument("--q-threshold", type=float, default=0.002, help="Q threshold for vortex regions")
    parser.add_argument("--smooth-sigma", type=float, default=1.0, help="Gaussian smoothing sigma on Q map")
    parser.add_argument("--min-area", type=int, default=6, help="Minimum area (pixels) for vortex region")

    parser.add_argument("--body-x", type=Path, default=repo_root / "x.csv", help="Body x-coordinate CSV")
    parser.add_argument("--body-y", type=Path, default=repo_root / "y.csv", help="Body y-coordinate CSV")
    parser.add_argument("--body-skip-header", action="store_true", help="Skip first row of body CSVs")
    parser.add_argument("--no-round-body", dest="round_body", action="store_false", help="Keep body coords as-is")
    parser.add_argument("--body-distance-min", type=float, default=6.0, help="Minimum distance from body to keep vortex")
    parser.add_argument("--no-body", dest="use_body", action="store_false", help="Disable body proximity filtering")
    parser.set_defaults(round_body=True, use_body=True)

    parser.add_argument(
        "--sensors",
        type=str,
        default="146,80;198,80;200,39",
        help="Sensors as 'x,y;x,y;...'",
    )
    parser.add_argument("--sensor-max-distance", type=float, default=32.0, help="Max distance to closest sensor")
    parser.add_argument("--occlusion-eps", type=float, default=0.5, help="Ray/boundary intersection tolerance (grid units)")
    parser.add_argument("--no-occlusion", dest="occlusion", action="store_false", help="Disable occlusion filtering")
    parser.set_defaults(occlusion=True)

    parser.add_argument("--wall-top", type=float, default=36.5, help="Top wall y-coordinate (grid units)")
    parser.add_argument("--wall-bottom", type=float, default=80.5, help="Bottom wall y-coordinate (grid units)")
    parser.add_argument("--wall-tol", type=float, default=0.5, help="Distance tolerance for wall intersection")

    parser.add_argument("--max-track-distance", type=float, default=3.0, help="Max distance to link tracks across frames")
    parser.add_argument("--max-gap", type=int, default=8, help="Max frame gap to reconnect tracks")
    parser.add_argument("--dt", type=float, default=None, help="Time step per frame (optional)")

    parser.add_argument("--out-dir", type=Path, default=repo_root / "python/vortexAnalysis_v2/vortex_outputs", help="Output directory")
    parser.add_argument("--out-csv", type=Path, default=None, help="CSV path for time series (optional)")
    parser.add_argument("--out-name", type=str, default="vortex_tracks.csv", help="Output filename for batch mode")

    parser.add_argument("--path-nadia", type=Path, default=Path("D:/crop_nadia"), help="Root path for Nadia dataset")
    parser.add_argument("--path-thomas", type=Path, default=Path("D:/thomas_files"), help="Root path for Thomas dataset")
    parser.add_argument("--path-boai", type=Path, default=Path("D:/boai_files"), help="Root path for Boai dataset")
    parser.add_argument("--path-full", type=Path, default=Path("D:/full_files"), help="Root path for Full dataset")
    parser.add_argument("--q-filename", type=str, default="Q_criterion_map.csv", help="Q-criterion filename inside snapshot folder")
    parser.add_argument(
        "--experiments",
        type=str,
        default=None,
        help="Comma-separated experiment list or ranges (e.g. '1,2,5-8'). Defaults to known sensor experiments.",
    )

    parser.add_argument("--show", action="store_true", help="Show tracking animation")
    parser.add_argument("--save-anim", action="store_true", help="Save tracking animation")
    parser.add_argument("--anim-path", type=Path, default=None, help="Animation output path (gif)")
    parser.add_argument("--anim-fps", type=int, default=24, help="Animation FPS")
    parser.add_argument("--anim-dpi", type=int, default=120, help="Animation DPI")
    args = parser.parse_args()

    if args.batch:
        default_experiments = [1, 2, 3, 6, 7, 8, 9, 10, 11, 14, 15, 16, 17, 18, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 40]
        experiments = parse_experiments(args.experiments, default_experiments)

        dataset_roots = [
            ("nadia", args.path_nadia),
            ("thomas", args.path_thomas),
            ("boai", args.path_boai),
            ("full", args.path_full),
        ]

        if args.show or args.save_anim:
            print("ℹ️  Animation is disabled in batch mode.")

        for experiment in experiments:
            snapshots = collect_snapshots(args.path_boai, experiment)
            for snapshot in snapshots:
                for dataset_name, root in dataset_roots:
                    snapshot_path = root / str(experiment) / snapshot
                    q_path = snapshot_path / args.q_filename
                    out_csv = snapshot_path / args.out_name

                    if dataset_name in ("nadia", "boai"):
                        body_root = args.path_nadia
                        body_x_name = f"rawYolo{snapshot}_x.csv"
                        body_y_name = f"rawYolo{snapshot}_y.csv"
                    else:
                        body_root = args.path_thomas
                        body_x_name = "final/x.csv"
                        body_y_name = "final/y.csv"

                    body_x_path = body_root / str(experiment) / snapshot / body_x_name
                    body_y_path = body_root / str(experiment) / snapshot / body_y_name

                    extracted = extract_vortices_from_q(
                        q_path=q_path,
                        out_csv=out_csv,
                        args=args,
                        body_x_path=body_x_path,
                        body_y_path=body_y_path,
                        allow_animation=False,
                    )
                    if extracted:
                        print(f"Processed {dataset_name} exp={experiment} snapshot={snapshot}")

            print(f"Completed vortex extraction for experiment {experiment}.")
    else:
        out_dir = args.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = args.out_csv or (out_dir / "vortex_tracks.csv")
        extract_vortices_from_q(
            q_path=args.q_path,
            out_csv=out_csv,
            args=args,
            body_x_path=args.body_x,
            body_y_path=args.body_y,
            allow_animation=True,
        )


if __name__ == "__main__":
    main()
