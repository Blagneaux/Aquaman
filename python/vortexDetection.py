from scipy.spatial import cKDTree
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter, label
import pandas as pd
import matplotlib.animation as animation
from PIL import Image
import io


def detect_vortices_filtered(matrix, grid_shape, time_step_index,
                              contour_x, contour_y, distance_threshold=5.0,
                              threshold_abs=0.1, smoothing_sigma=1.0, neighborhood_size=15, source="vorticity"):
    """
    Detects vortex centers as vorticity peaks (both positive and negative).
    
    Parameters:
    - matrix: np.ndarray, shape (num_points, num_timesteps)
    - grid_shape: tuple, (ny, nx)
    - time_step_index: int, column index in vorticity_matrix
    - contour_x: np.array, shape (num_points, num_timesteps)
    - contour_y: np.array, shape (num_points, num_timesteps)
    - threshold_abs: float, minimum |vorticity| for detection
    - smoothing_sigma: float, std for Gaussian filter
    - neighborhood_size: int, neighborhood for local extrema detection
    - source: string, type of physical field
    
    Returns:
    - List of vortex centers [(y, x, vorticity_value), ...]
    - 2D vorticity field (for visualization)
    """

    flat = matrix[:, time_step_index]
    field = flat.reshape(grid_shape)

    smooth = gaussian_filter(field, sigma=smoothing_sigma)

    maxima = None
    minima = None

    if source == "vorticity":
        local_max = maximum_filter(smooth, size=neighborhood_size) == smooth
        local_min = minimum_filter(smooth, size=neighborhood_size) == smooth

        maxima = np.where((local_max) & (smooth > threshold_abs))
        minima = np.where((local_min) & (smooth < -threshold_abs))

    if source == "pressure":
        local_min = minimum_filter(smooth, size=neighborhood_size) == smooth

        minima = np.where((local_min) & (smooth < -threshold_abs))

    # Distance map to contour
    Y, X = np.meshgrid(np.arange(grid_shape[0]), np.arange(grid_shape[1]), indexing='ij')
    coords = np.vstack((Y.ravel(), X.ravel())).T
    tree = cKDTree(np.column_stack((contour_y, contour_x)))
    distances, _ = tree.query(coords)
    distance_map = distances.reshape(grid_shape)

    vortex_centers = []
    if maxima is not None:
        for y, x in zip(*maxima):
            if distance_map[y, x] >= distance_threshold:
                vortex_centers.append((y, x, smooth[y, x]))
    if minima is not None:
        for y, x in zip(*minima):
            if distance_map[y, x] >= distance_threshold:
                vortex_centers.append((y, x, smooth[y, x]))
    return vortex_centers, smooth


def track_vortices_over_time_with_fish(vorticity_matrix, grid_shape, 
                                       contour_x_all, contour_y_all,
                                       threshold_abs=0.1, smoothing_sigma=1.0,
                                       distance_threshold=5.0, neighborhood_size=15,
                                       max_tracking_distance=5.0, source="vorticity"):
    """
    Track vortices over time using proximity-based linking.
    
    Returns:
    - List of vortex tracks. Each track is a list of (time, y, x, vorticity_value).
    """

    n_timesteps = vorticity_matrix.shape[1]
    tracks = []

    for t in range(n_timesteps):
        contour_x = contour_x_all[:, t]
        contour_y = contour_y_all[:, t]
        vortex_centers, _ = detect_vortices_filtered(
            vorticity_matrix, grid_shape, t, contour_x, contour_y,
            distance_threshold, threshold_abs, smoothing_sigma, neighborhood_size, source)

        current_points = np.array([[y, x] for y, x, _ in vortex_centers])

        if t == 0:
            for (y, x, val) in vortex_centers:
                tracks.append([(t, y, x, val)])
            continue

        # Match with last frame
        previous_points = np.array([[trk[-1][1], trk[-1][2]] for trk in tracks])
        tree = cKDTree(current_points) if len(current_points) > 0 else None
        assigned = set()

        for idx, trk in enumerate(tracks):
            if tree is None: continue
            dist, nearest = tree.query(previous_points[idx], distance_upper_bound=max_tracking_distance)
            if dist != np.inf and nearest not in assigned:
                y, x, val = vortex_centers[nearest]
                trk.append((t, y, x, val))
                assigned.add(nearest)

        for i, (y, x, val) in enumerate(vortex_centers):
            if i not in assigned:
                tracks.append([(t, y, x, val)])

    return tracks


def filter_tracks_near_sensors_closest(tracks, sensors, x_threshold=5.0, y_start_threshold=0.0):
    """
    Filters tracks to retain only vortices that:
    1. Pass through the vertical plane of a sensor (x alignment within threshold).
    2. Are the closest vortex to that sensor at the given timestep.
    
    Parameters:
    - tracks: list of vortex tracks [(t, y, x, value), ...]
    - sensors: list of (x, y) sensor positions
    - x_threshold: horizontal proximity to sensor's vertical plane
    
    Returns:
    - Filtered list of tracks
    """

    sensor_ys = [sy for _, sy in sensors]

    # Remove tracks whose first point is too close to any sensor's y-coordinate
    clean_tracks = []
    for track in tracks:
        t0, y0, x0, _ = track[0]
        if any(abs(x0 - sy) <= y_start_threshold for sy in sensor_ys):
            continue  # Skip this track entirely
        clean_tracks.append(track)

    # Organize tracks into a structure to check all vortices per timestep
    timestep_dict = dict()
    for track_id, track in enumerate(clean_tracks):
        for t, y, x, val in track:
            if t not in timestep_dict:
                timestep_dict[t] = []
            timestep_dict[t].append((track_id, y, x, val))

    # Store track IDs to keep
    selected_track_ids = set()

    for t, vortices in timestep_dict.items():
        for sensor_x, sensor_y in sensors:
            closest_dist = float('inf')
            best_track_id = None

            for track_id, y, x, val in vortices:
                if abs(y - sensor_x) <= x_threshold:
                    dist = np.sqrt((y - sensor_x)**2 + (x - sensor_y)**2)
                    if dist < closest_dist:
                        closest_dist = dist
                        best_track_id = track_id

            if best_track_id is not None:
                selected_track_ids.add(best_track_id)

    # Keep only selected tracks
    filtered_tracks = [track for i, track in enumerate(clean_tracks) if i in selected_track_ids]

    return filtered_tracks


def animate_vortex_tracking_with_fish_mask(vorticity_matrix, grid_shape, tracks, tracks_all,
                                           contour_x_all, contour_y_all,
                                           filename="vortex_tracking.gif", dpi=100):
    """
    Animated GIF showing vortex tracking with fish contour and exclusion mask.
    
    Parameters:
    - vorticity_matrix: shape (ny*nx, nt)
    - grid_shape: (ny, nx)
    - tracks: list of vortex tracks
    - contour_x_all, contour_y_all: fish contour over time, shape (n_pts, nt)
    """
    n_timesteps = vorticity_matrix.shape[1]
    ny, nx = grid_shape
    frames = []

    for t in range(n_timesteps):
        fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)

        # Get vorticity and contour at this timestep
        vort_frame = vorticity_matrix[:, t].reshape(grid_shape).T  # Transpose to (x, y)
        vmax = np.max(np.abs(vort_frame))
        im = ax.imshow(vort_frame, cmap='bwr', origin='lower', vmin=-vmax, vmax=vmax)

        # Fish contour (transposed)
        contour_x = contour_x_all[:, t]
        contour_y = contour_y_all[:, t]
        ax.plot(contour_x, contour_y, 'go', markersize=1.5, label="Fish Contour")

        # Sensors
        ax.plot([146, 198, 200],[80, 80, 39],'go', markersize=3)
        for sx, _ in sensors:
            ax.axvline(x=sx-x_plane_threshold, color='g', linestyle='--', alpha=0.3)
            ax.axvline(x=sx+x_plane_threshold, color='g', linestyle='--', alpha=0.3)
            # ax.axvline(x=sx, color='g', linestyle='--', alpha=0.3)

        # Distance exclusion zone
        Y, X = np.meshgrid(np.arange(grid_shape[0]), np.arange(grid_shape[1]), indexing='ij')
        coords = np.vstack((Y.ravel(), X.ravel())).T

        # Overlay vortex tracks up to this frame
        for i, track in enumerate(tracks):
            coords = [(tt, y, x) for (tt, y, x, _) in track if tt <= t]
            if len(coords) >= 2:
                _, y_list, x_list = zip(*coords)
                ax.plot(y_list, x_list, '-', lw=1.2, alpha=0.8)

        for i, track_all in enumerate(tracks_all):
            coords = [(tt, y, x) for (tt, y, x, _) in track_all if tt <= t]
            if len(coords) >= 2:
                _, y_list, x_list = zip(*coords)
                ax.plot(y_list[-1], x_list[-1], 'ko', markersize=3)

        ax.set_title(f"Time Step {t}", fontsize=10)
        ax.set_xlim(0, ny)
        ax.set_ylim(0, nx)
        ax.set_axis_off()

        # Save figure to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        frames.append(img)

    # Save GIF
    frames[0].save(filename, save_all=True, append_images=frames[1:], duration=200, loop=0)
    print(f"✅ Saved animated GIF with fish mask to: {filename}")


def plot_theorical_pressure(tracks, sensor_pos, matrix, grid_shape):
    """
    Plot the allure of the theorical pressure coefficient generated by
    the vortex tracked, following a 1/R² dissipation law, and superpose it
    to the complete simulated pressure signal of the full simulation.

    Parameters:
    - tracks
    - a single sensor location coordinates
    - the pressure map
    """
    time0 = [snapshot[0] for snapshot in tracks[0]] 
    full_time = np.linspace(time0[0], time0[-1], time0[-1]-time0[0]+1,dtype=np.int16)

    pressure_allures = []
    times = []
    vorticities = []
    for track in tracks:
        time = [snapshot[0] for snapshot in track] 
        distance = [np.sqrt((snapshot[1] - sensor_pos[0])**2 + (snapshot[2] - sensor_pos[1])**2) for snapshot in track]
        pressure_allure = [-1/(r**2) for r in distance]
        vorticity = [snapshot[-1] for snapshot in track]
        pressure_allures.append(pressure_allure)
        times.append(time)
        vorticities.append(vorticity)

    full_pressure = []
    for t in full_time:
        flat = matrix[:, t]
        field = flat.reshape(grid_shape)
        full_pressure.append(field[sensor_pos[0], sensor_pos[1]])

    plt.figure()
    for time, allure in zip(times, pressure_allures):
        plt.plot(time, allure, 'o', markersize=1)
    plt.plot(full_time, full_pressure)

    plt.figure()
    for time, vorticity in zip(times, vorticities):
        plt.plot(time, vorticity, 'o', markersize=1)
    plt.show()

n, m = 256, 128
vorticity_matrix = pd.read_csv("D:/crop_nadia/14/9/vorticity_map.csv", header=None)
X = pd.read_csv("D:/crop_nadia/14/9/rawYolo9_x.csv", header=None)
Y = pd.read_csv("D:/crop_nadia/14/9/rawYolo9_y.csv", header=None)
pressure_matrix = pd.read_csv("D:/crop_nadia/14/9/pressure_map.csv", header=None)
vorticity_matrix = np.array(vorticity_matrix)
X = np.array(X)
Y = np.array(Y)
pressure_matrix = np.array(pressure_matrix)

# Define sensor positions (x, y) and x-threshold for vertical plane check
sensors = [(146, 80), (198, 80), (200, 39)]
x_plane_threshold = 5.0  # in pixels, adjust as needed

# Track all vortices
tracks_all = track_vortices_over_time_with_fish(
    vorticity_matrix, (n, m), Y[:, 20:], X[:, 20:], distance_threshold=2
)

# Filter by passing through sensor's x-plane
tracks = filter_tracks_near_sensors_closest(tracks_all, sensors, x_threshold=x_plane_threshold)

# Animate filtered tracks
# animate_vortex_tracking_with_fish_mask(vorticity_matrix, (n, m), tracks, tracks_all, X[:, 20:], Y[:, 20:])

# Theorically compute the pressure generated by the vortex on the sensor
plot_theorical_pressure(tracks[:6], sensors[0], pressure_matrix, (n,m))