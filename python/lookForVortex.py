import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import glob

from matplotlib.colors import LinearSegmentedColormap

# ----------------------------------------------------------------------------




# DEPRECATED





# ----------------------------------------------------------------------------

colors = ["blue", "white", "red"]  # Blue at the lowest, white in the middle, red at the highest
n_bins = 25  # More bins will make the transition smoother
cmap_name = 'custom'

# Create the colormap
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

# Setup paths and load sensor data
path = "D:/crop_nadia/matchingData_2s_window"
isCircle = False
files = glob.glob(path+"/*.csv")
relative_paths = np.int8([file.split('\\')[-1][:-4] for file in files])
relative_paths.sort()
pressure_path = "D:/crop_nadia"
if not isCircle:
    pressure_file = "/pressure_map.csv"
else:
    pressure_file = "/circle_pressure_map.csv"
print(relative_paths)
for video_number in relative_paths:
    info = pd.read_csv(path + f"/{video_number}.csv")

    # Iterate through each sensor, validity, and sample
    for index, row in info.iterrows():
        sensor, validity, sample, start = row['Channel'], row['Validity'], row['Sample'], row["Starting_offset"]

        # Only process valid entries
        if validity:
            x_file = f"/rawYolo{sample}_x.csv"
            y_file = f"/rawYolo{sample}_y.csv"
            pressure_data = pd.read_csv(pressure_path + f"/{video_number}/{sample}{pressure_file}", header=None)
            X = pd.read_csv(pressure_path + f"/{video_number}/{sample}{x_file}", header=None)
            Y = pd.read_csv(pressure_path + f"/{video_number}/{sample}{y_file}", header=None)

            # Process data
            pressure_array = np.array(pressure_data)
            x_array = np.array(X)[:, start:]
            y_array = np.array(Y)[:, start:]
            original_height, original_width = 128, 256
            new_height, new_width = 540, 960
            fps = 100
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            if not isCircle:
                video = cv2.VideoWriter(f"D:/crop_nadia/videos/{video_number}_{sample}_{sensor}.mp4", fourcc, fps, (new_width, new_height))
            else:
                video = cv2.VideoWriter(f"D:/crop_nadia/videos_circle/{video_number}_{sample}_{sensor}.mp4", fourcc, fps, (new_width, new_height))

            # Coordinate scaling
            if sensor == "S1":
                ref_x, ref_y = 198, 80
            elif sensor == "S2":
                ref_x, ref_y = 146, 80
            elif sensor == "S4":
                ref_x, ref_y = 200, 39

            # Scale coordinates
            x = int(ref_x * (new_width / original_width))
            y = int(ref_y * (new_height / original_height))

            for i in range(pressure_array.shape[1]):
                frame = pressure_array[:, i].reshape(original_width, original_height)
                frame = np.clip(frame, -0.25, 0.25)  # Clipping data to -0.5 to 0.5
                frame = (frame + 0.25) / 0.5  # Normalize to 0-1
                frame = (plt.get_cmap(cm)(frame)[:, :, :3] * 255).astype(np.uint8)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                frame = cv2.flip(frame, 0)
                frame_resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

                # Resize the polyline points accordingly
                scale_x = new_width / original_width
                scale_y = new_height / original_height
                resized_points = np.array([[int(x * scale_x), int(y * scale_y)] for x, y in zip(x_array[:,i], y_array[:,i])], dtype=np.int32).reshape((-1, 1, 2))
                
                cv2.polylines(frame_resized, [resized_points], True, (0, 255, 0), 2)

                # Draw a green point at the scaled coordinate
                cv2.circle(frame_resized, (x, y), radius=2, color=(0, 255, 0), thickness=-1)

                video.write(frame_resized)
                # cv2.imshow('Frame', frame_resized)
                # if cv2.waitKey(10) & 0xFF == ord('q'):
                #     break

            video.release()
            cv2.destroyAllWindows()
            print(f"Video {video_number} sample {sample} for sensor {sensor} completed.")