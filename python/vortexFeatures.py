import pandas as pd
import os
import numpy as np

# Load existing labels
output_labels = "vortex_labels.csv"
if os.path.exists(output_labels):
    labeled_df = pd.read_csv(output_labels)
else:
    raise FileNotFoundError("❌ vortex_labels.csv not found. Labeling data is required.")

# Load previously processed features to avoid duplication
feature_file = "vortex_features.csv"
if os.path.exists(feature_file):
    existing_features = pd.read_csv(feature_file)
    existing_keys = set(
        zip(
            existing_features['haato'],
            existing_features['haachama'],
            existing_features['sensor'],
            existing_features['fish'],
            existing_features['vortex_id']
        )
    )
else:
    existing_keys = set()

# Group labeled data
grouped_labels = labeled_df.groupby(['haato', 'haachama', 'sensor', 'fish'])
base_path = "D:/crop_nadia"
all_features = []

for (haato, haachama, sensor, fish), group in grouped_labels:
    if all(label.strip().lower() == 'not relevant' for label in group['label']):
        continue
    print(f"Processing: haato={haato}, haachama={haachama}, sensor={sensor}, fish={fish}")

    if fish:
        vorticity_matrix = pd.read_csv(f"{base_path}/{haato}/{haachama}/vorticity_map.csv", header=None)
    else:
        vorticity_matrix = pd.read_csv(f"{base_path}/{haato}/{haachama}/circle_vorticity_map.csv", header=None)

    X = pd.read_csv(f"{base_path}/{haato}/{haachama}/rawYolo{haachama}_x.csv", header=None)
    Y = pd.read_csv(f"{base_path}/{haato}/{haachama}/rawYolo{haachama}_y.csv", header=None)
    vorticity_matrix = np.array(vorticity_matrix)
    X = np.array(X)
    Y = np.array(Y)

    from scipy.spatial import cKDTree
    from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter

    def detect_vortices(matrix, grid_shape, time_step_index, contour_x, contour_y, distance_threshold=2.0):
        flat = matrix[:, time_step_index]
        field = flat.reshape(grid_shape)
        smooth = gaussian_filter(field, sigma=1.0)
        local_max = maximum_filter(smooth, size=15) == smooth
        local_min = minimum_filter(smooth, size=15) == smooth
        maxima = np.where((local_max) & (smooth > 0.1))
        minima = np.where((local_min) & (smooth < -0.1))
        Yg, Xg = np.meshgrid(np.arange(grid_shape[0]), np.arange(grid_shape[1]), indexing='ij')
        coords = np.vstack((Yg.ravel(), Xg.ravel())).T
        tree = cKDTree(np.column_stack((contour_y, contour_x)))
        distances, _ = tree.query(coords)
        distance_map = distances.reshape(grid_shape)
        vortex_centers = []
        for y, x in zip(*maxima):
            if distance_map[y, x] >= distance_threshold:
                vortex_centers.append((y, x, smooth[y, x]))
        for y, x in zip(*minima):
            if distance_map[y, x] >= distance_threshold:
                vortex_centers.append((y, x, smooth[y, x]))
        return vortex_centers

    def track_vortices(vorticity_matrix, grid_shape, contour_x_all, contour_y_all, max_tracking_distance=5.0):
        n_timesteps = vorticity_matrix.shape[1]
        tracks = []
        for t in range(n_timesteps):
            contour_x = contour_x_all[:, t]
            contour_y = contour_y_all[:, t]
            vortex_centers = detect_vortices(vorticity_matrix, grid_shape, t, contour_x, contour_y)
            current_points = np.array([[y, x] for y, x, _ in vortex_centers])
            if t == 0:
                for (y, x, val) in vortex_centers:
                    tracks.append([(t, y, x, val)])
                continue
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

    tracks = track_vortices(vorticity_matrix, (256, 128), Y[:, group.iloc[0]['vortex_id']:], X[:, group.iloc[0]['vortex_id']:])

    for _, row in group.iterrows():
        if row['label'].strip().lower() == 'not relevant':
            continue
        key = (row['haato'], row['haachama'], row['sensor'], row['fish'], row['vortex_id'])
        if key in existing_keys:
            continue  # Skip already processed

        vortex_id = row['vortex_id']
        track = tracks[vortex_id] if vortex_id < len(tracks) else []
        if not track:
            continue

        time_steps = [snap[0] for snap in track]
        ys = [snap[1] for snap in track]
        xs = [snap[2] for snap in track]
        vorts = [snap[3] for snap in track]
        duration = time_steps[-1] - time_steps[0] + 1
        displacement = np.sqrt((ys[-1] - ys[0])**2 + (xs[-1] - xs[0])**2)
        avg_speed = displacement / duration if duration > 0 else 0
        max_speed = max(
            np.sqrt((ys[i+1] - ys[i])**2 + (xs[i+1] - xs[i])**2)
            for i in range(len(track)-1)
        ) if len(track) > 1 else 0
        mean_vorticity = np.mean(np.abs(vorts))
        std_vorticity = np.std(vorts)

        all_features.append({
            **row,
            "duration": duration,
            "displacement": displacement,
            "avg_speed": avg_speed,
            "max_speed": max_speed,
            "mean_vorticity": mean_vorticity,
            "std_vorticity": std_vorticity
        })

# Save full feature set
if all_features:
    feature_df = pd.DataFrame(all_features)
    feature_df.to_csv("vortex_features.csv", mode='a', header=not os.path.exists("vortex_features.csv"), index=False)
    print("✅ New extended features appended to vortex_features.csv")
else:
    print("✅ No new features to add. All data was already processed.")
