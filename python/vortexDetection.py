from scipy.spatial import cKDTree
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter, label
import pandas as pd
import matplotlib.animation as animation
from PIL import Image
import io
from scipy.signal import butter, filtfilt
from nptdms import TdmsFile
from matplotlib.widgets import Button, RadioButtons, CheckButtons
import os
import itertools
import random


# Fonction pour créer un filtre passe-bande de second ordre
def butter_bandpass(lowcut, highcut, fs, order=2):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Appliquer le filtre passe-bande aux données
def bandpass_filter(data, lowcut=0.3, highcut=9, fs=500, order=2):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, data)

# Applique une normalisation pour comparer les deux signaux avec un meme ordre de grandeur
def normalisation(data):
    abs_max_data = np.max(np.abs(data))
    normalized_data = data / abs_max_data
    return normalized_data

# Function to convert hh:mm:ss to seconds
def time_to_seconds(time_str):
    h, m, s = map(int, time_str.split(':'))
    return (h * 3600 + m * 60 + s) / 4  # Time video converted to time camera


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
    pressure_allures = []
    times = []
    vorticities = []
    for track in tracks:
        time = [start_time + snapshot[0]/100 for snapshot in track] 
        distance = [np.sqrt((snapshot[1] - sensor_pos[0])**2 + (snapshot[2] - sensor_pos[1])**2) for snapshot in track]
        vorticity = [snapshot[-1] for snapshot in track]
        pressure_allure = [-(np.pi*1.5*1.5)*w/(r**2) for w,r in zip(vorticity, distance)]
        pressure_allures.append(pressure_allure)
        times.append(time)
        vorticities.append(vorticity)

    full_pressure = [matrix[i][146*128 + 80] for i in matrix.columns]

    plt.figure()
    for time, allure in zip(times, pressure_allures):
        plt.plot(time, allure, 'o', markersize=1)
    pressure_data = bandpass_filter(full_pressure, highcut=9, fs=100)
    pressure_data = normalisation(pressure_data)
    full_time = np.linspace(start_time, end_time, len(full_pressure))
    plt.plot(full_time, pressure_data, "--", color="0")
    full_time_exp = np.linspace(start_time, end_time, int(end_time*500)-int(start_time*500))
    plt.plot(full_time_exp, pressure_exp, color="0")
    plt.show()


def launch_labeling_ui(pressure_exp, haato, haachama, sensor_pos, tracks, fish):
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(left=0.35, bottom=0.05)

    if sensor_pos[0] == 146:
        sensor_name = "S2"
    elif sensor_pos[0] == 198:
        sensor_name = "S1"
    elif sensor_pos[0] == 200:
        sensor_name = "S4"

    pressure_allures = []
    times = []
    vorticities = []
    for track in tracks:
        time = [start_time + snapshot[0]/100 for snapshot in track] 
        distance = [np.sqrt((snapshot[1] - sensor_pos[0])**2 + (snapshot[2] - sensor_pos[1])**2) for snapshot in track]
        vorticity = [snapshot[-1] for snapshot in track]
        pressure_allure = [-(np.pi*1.5*1.5)*w/(r**2) for w,r in zip(vorticity, distance)]
        pressure_allures.append(pressure_allure)
        times.append(time)
        vorticities.append(vorticity)

    n_vortices = len(pressure_allures)
    lines = []
    for i, (time, allure) in enumerate(zip(times, pressure_allures)):
        line, = ax.plot(time, allure, 'o', markersize=1)
        lines.append(line)
    full_time_exp = np.linspace(start_time, end_time, int(end_time*500)-int(start_time*500))
    exp_line, = ax.plot(full_time_exp, pressure_exp, 'k-', label='Experimental')
    ax.legend(loc='upper right')
    ax.set_title("During the period where the vortex pressure (color signals) is not null, \n is its general allure also fund in the experimental pressure (black signal) at approximately the same time? \n Label each vortex: Match / No Match / Not Relevant")

    # CheckButtons for visibility toggles
    check_ax = plt.axes([0.02, 0.01, 0.10, 0.05 * (n_vortices+1)])
    labels = [f'Vortex {i}' for i in range(n_vortices)]
    visibility = [True] * n_vortices
    check = CheckButtons(check_ax, labels, visibility)

    def toggle_visibility(label):
        index = labels.index(label)
        lines[index].set_visible(not lines[index].get_visible())
        plt.draw()

    check.on_clicked(toggle_visibility)

    # RadioButtons for each vortex
    radio_axes = []
    radio_buttons = []
    label_options = ["match", "no match", "not relevant", "semi match"]
    selected_labels = ["not relevant"] * n_vortices

    for i in range(n_vortices):
        ax_radio = plt.axes([0.15, 0.01 + i*0.055, 0.15, 0.05])
        rb = RadioButtons(ax_radio, label_options, active=2)
        radio_axes.append(ax_radio)
        radio_buttons.append(rb)

        def make_handler(index):
            def handler(label):
                selected_labels[index] = label
            return handler

        rb.on_clicked(make_handler(i))

    # Confirm button
    confirm_ax = plt.axes([0.7, 0.01, 0.15, 0.05])
    confirm_button = Button(confirm_ax, 'Validate Selection')

    def on_confirm(event):
        output_file = "vortex_labels_test.csv"
        rows = []
        for i, label in enumerate(selected_labels):
            vortex_track = tracks[i]
            distances = [np.sqrt((snapshot[1] - sensor_pos[0])**2 + (snapshot[2] - sensor_pos[1])**2) for snapshot in vortex_track]
            vorticities = [abs(snapshot[3]) for snapshot in vortex_track]
            min_distance = np.min(distances)
            max_vorticity = np.max(vorticities)
            row = {
                "haato": haato,
                "haachama": haachama,
                "sensor": sensor_name,
                "fish": fish,
                "vortex_id": i,
                "min_distance": min_distance,
                "max_vorticity": max_vorticity,
                "label": label
            }
            rows.append(row)
        df = pd.DataFrame(rows)
        if os.path.exists(output_file):
            df.to_csv(output_file, mode='a', header=False, index=False)
        else:
            df.to_csv(output_file, mode='w', header=True, index=False)
        print(f"✅ Logged {len(rows)} vortex labels to {output_file}")
        plt.close()

    confirm_button.on_clicked(on_confirm)

    plt.show()



n, m = 256, 128

# Parameter selection to pick the experiment to analyze
# haato = 14
# haachama = 9
# sensor = "S2"
# fish = True

sample_possibility = pd.read_csv("D:/crop_nadia/list_for_automation.csv")
sample_possibility = list(sample_possibility.iloc[:,0:2].itertuples(index=False, name=None))
sensor_possibility = ["S1", "S2", "S4"]
fish_possibility = [True, False]

all_combinations = list(itertools.product(sample_possibility, sensor_possibility, fish_possibility))

# Load the already labeled combinations
output_labels = "vortex_labels_test.csv"

while True:
    if os.path.exists(output_labels) and os.path.getsize(output_labels) > 0:
        labeled_df = pd.read_csv(output_labels)
        used_combinations = set(
            (row['haato'], row['haachama'], row['sensor'], row['fish']) 
            for _, row in labeled_df.iterrows()
        )
    else:
        used_combinations = set()

    # Find an unused combination
    unused_combinations = [((haato, haachama), sensor, fish) 
                        for ((haato, haachama), sensor, fish) in all_combinations
                        if (haato, haachama, sensor, fish) not in used_combinations
                        ]
    
    if not unused_combinations:
        print("✅ All combinations have been labeled.")
        break
    
    selected = random.choice(unused_combinations)
    (haato, haachama), sensor, fish = selected
    print(unused_combinations)
    print("🎯 Remaining labels: ", len(unused_combinations), "/", len(all_combinations))


    # Load the corresponding data
    if fish:
        vorticity_matrix = pd.read_csv(f"D:/crop_nadia/{haato}/{haachama}/vorticity_map.csv", header=None)
        pressure_matrix = pd.read_csv(f"D:/crop_nadia/{haato}/{haachama}/pressure_map.csv", header=None)
    else:
        vorticity_matrix = pd.read_csv(f"D:/crop_nadia/{haato}/{haachama}/circle_vorticity_map.csv", header=None)
        # pressure_matrix = pd.read_csv(f"D:/crop_nadia/{haato}/{haachama}/circle_pressure_map.csv", header=None)
    X = pd.read_csv(f"D:/crop_nadia/{haato}/{haachama}/rawYolo{haachama}_x.csv", header=None)
    Y = pd.read_csv(f"D:/crop_nadia/{haato}/{haachama}/rawYolo{haachama}_y.csv", header=None)
    vorticity_matrix = np.array(vorticity_matrix)
    X = np.array(X)
    Y = np.array(Y)

    tdms_file = TdmsFile.read(f"D:/crop_nadia/TDMS/{haato}.tdms")
    digital_twin_time = pd.read_csv(f"D:/crop_nadia/timestamps/timestamps{haato}.csv")
    digital_twin_time["start_time"] = digital_twin_time["start_time"].apply(time_to_seconds)
    digital_twin_time["end_time"] = digital_twin_time["end_time"].apply(time_to_seconds)
    start_time = digital_twin_time["start_time"][haachama-1]
    end_time = digital_twin_time["end_time"][haachama-1]
    startIndex = digital_twin_time["start_frame"][haachama-1]

    # Extracte the experimental pressure corresponding to the chosen sensor
    for groupe in tdms_file.groups()[1:]:
        for canal in groupe.channels():
            if canal.name == sensor:
                pressure_exp = bandpass_filter(canal.data, highcut=3)
                pressure_exp = pressure_exp[int(start_time*500): int(end_time*500)]
                pressure_exp = normalisation(pressure_exp)

    # Define sensor positions (x, y) and x-threshold for vertical plane check
    sensors = [(146, 80), (198, 80), (200, 39)]
    x_plane_threshold = 5.0  # in pixels, adjust as needed

    # Track all vortices
    tracks_all = track_vortices_over_time_with_fish(
        vorticity_matrix, (n, m), Y[:, startIndex:], X[:, startIndex:], distance_threshold=2
    )

    # Filter by passing through sensor's x-plane
    tracks = filter_tracks_near_sensors_closest(tracks_all, sensors, x_threshold=x_plane_threshold)

    # Animate filtered tracks
    # animate_vortex_tracking_with_fish_mask(vorticity_matrix, (n, m), tracks, tracks_all, X[:, 20:], Y[:, 20:])

    # Theorically compute the pressure generated by the vortex on the sensor
    # plot_theorical_pressure(tracks, sensors[0], pressure_matrix, (n,m))
    if sensor == "S1":
        launch_labeling_ui(pressure_exp, haato, haachama, sensors[1], tracks, fish)
    elif sensor == "S2":
        launch_labeling_ui(pressure_exp, haato, haachama, sensors[0], tracks, fish)
    elif sensor == "S4":
        launch_labeling_ui(pressure_exp, haato, haachama, sensors[2], tracks, fish)