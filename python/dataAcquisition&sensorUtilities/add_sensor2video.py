import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import pandas as pd
from nptdms import TdmsFile
from scipy.signal import butter, filtfilt

# Initialize the video reader and writer
file_number = 9
sample = 12
sensor = "S2"
circleOn = False
isCircle = True
offset = 0
# input_video_path = f'C:/Users/blagn771/Desktop/{file_number}_{sample}_0.5.mp4'
if not isCircle:
    input_video_path = f'D:/crop_nadia/videos/{file_number}_{sample}_{sensor}.mp4'
else:
    input_video_path = f'D:/crop_nadia/videos_circle/{file_number}_{sample}_{sensor}.mp4'
# input_video_path = f'C:/Users/blagn771/Desktop/circle_{file_number}_{sample}.mp4'
output_video_path = 'C:/Users/blagn771/Desktop/output_video.mp4'

# Create a VideoCapture object to read the video
cap = cv2.VideoCapture(input_video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Create a VideoWriter object to write the video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Coordinate scaling
if sensor == "S1":
    ref_x, ref_y = 198 + offset, 80
elif sensor == "S2":
    ref_x, ref_y = 146 + offset, 80
elif sensor == "S4":
    ref_x, ref_y = 200 + offset, 39

ref_width, ref_height = 256, 128
video_width, video_height = frame_width, frame_height

# Scale coordinates
x = int(ref_x * (video_width / ref_width))
y = int(ref_y * (video_height / ref_height))

# Process the video
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Draw a green point at the scaled coordinate
    cv2.circle(frame, (x, y), radius=5, color=(0, 255, 0), thickness=-1)

    # Write the frame to the output video
    out.write(frame)

# Release the video capture and writer
cap.release()
out.release()

print("Video processing complete. The sensor has been added:", output_video_path)

# Function to convert hh:mm:ss to seconds
def time_to_seconds(time_str):
    h, m, s = map(int, time_str.split(':'))
    return (h * 3600 + m * 60 + s) / 4  # Time video converted to time camera

# Fonction pour créer un filtre passe-bande de second ordre
def butter_bandpass(lowcut, highcut, fs, order=2):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Appliquer le filtre passe-bande aux données
def bandpass_filter(data, lowcut=0.3, highcut=35, fs=500, order=2):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, data)

# Applique une normalisation pour comparer les deux signaux avec un meme ordre de grandeur
def normalisation(data):
    abs_max_data = np.max(np.abs(data))
    normalized_data = data / abs_max_data
    return normalized_data

digital_twin_time = pd.read_csv("D:/crop_nadia/timestamps/timestamps"+str(file_number)+".csv")
tdms_file = TdmsFile.read("D:/crop_nadia/TDMS/"+str(file_number)+".tdms")
digital_twin_time["start_time"] = digital_twin_time["start_time"].apply(time_to_seconds)
digital_twin_time["end_time"] = digital_twin_time["end_time"].apply(time_to_seconds)

start_time = digital_twin_time["start_time"][sample-1]
end_time = digital_twin_time["end_time"][sample-1]
end_time_cyl = end_time - 0.43
print(end_time)

if not isCircle:
    digital_twin_pressure = pd.read_csv("D:/crop_nadia/"+str(file_number)+"/"+str(sample)+"/pressure_map.csv")
else:
    digital_twin_pressure = pd.read_csv("D:/crop_nadia/"+str(file_number)+"/"+str(sample)+"/circle_pressure_map.csv")
if circleOn:
    circle_pressure = pd.read_csv("D:/crop_nadia/test_vortex/circle_"+str(file_number)+"_"+str(sample)+"_pressure_map.csv")
posX = None
if sensor == "S1":
    posX = (198 + offset)*128 + 37+43
elif sensor == "S2":
    posX = (146 + offset)*128 + 37+43
elif sensor == "S4":
    posX = (200 + offset)*128 + 39

dt_pressure_data = [digital_twin_pressure[i][posX] for i in digital_twin_pressure.columns]
if circleOn:
    circle_data = [circle_pressure[i][posX] for i in circle_pressure.columns]
X_dt = np.linspace(start_time, end_time, len(dt_pressure_data))
if circleOn:
    X_dt_cyl = np.linspace(start_time, end_time_cyl, len(circle_data))

dt_pressure_data = bandpass_filter(dt_pressure_data, highcut=1, fs=100)
dt_pressure_data = normalisation(dt_pressure_data)
if circleOn:
    circle_data = bandpass_filter(circle_data, highcut=9, fs=100)
    circle_data = normalisation(circle_data)
X_exp = np.linspace(start_time, end_time, int(end_time*500)-int(start_time*500))

for groupe in tdms_file.groups()[1:]:
    for canal in groupe.channels():
        if canal.name == sensor:
            pressure_data = bandpass_filter(canal.data, highcut=1)
            pressure_data = pressure_data[int(start_time*500): int(end_time*500)]
            pressure_data = normalisation(pressure_data)

# Set up the video paths and capture
input_video_path = 'C:/Users/blagn771/Desktop/output_video.mp4'
if not isCircle:
    output_video_path = f'D:/crop_nadia/videos/{file_number}_{sample}_{sensor}_comparison.mp4'
else:
    output_video_path = f'D:/crop_nadia/videos_circle/{file_number}_{sample}_{sensor}_comparison.mp4'
cap = cv2.VideoCapture(input_video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Create a VideoWriter object to write the video with the plot
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Function to get the plot as an image
def get_plot_as_image(frame_idx, frame_idx_exp, frame_idx_cyl):
    # Create a figure for plotting
    fig, ax = plt.subplots(figsize=(frame_width/100, frame_height/200), dpi=100)
    canvas = FigureCanvas(fig)
    ax.plot(X_dt[:frame_idx], dt_pressure_data[:frame_idx], label=f"Simulated Pressure (offset: {offset})")
    if circleOn:
        ax.plot(X_dt_cyl[:frame_idx_cyl], circle_data[:frame_idx_cyl], label=f"Simulated Circle Pressure (offset: {offset})")
    ax.plot(X_exp[:frame_idx_exp], pressure_data[:frame_idx_exp], label = "Experimental Pressure")
    ax.plot(X_exp, [0 for elmt in X_exp], "black")
    sec = X_exp[0::500]
    ax.plot(sec, [0 for elmt in sec], "black", marker=2)
    plt.legend()
    plt.title(f"Experiment {file_number} sample {sample} sensor {sensor}")
    ax.set_xlim(np.min(X_dt), np.max(X_dt))
    ax.set_ylim(-1, 1)
    ax.axis("off")  # Turn off axes

    # Convert plot to image
    canvas.draw()
    plot_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    plot_image = plot_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return plot_image

# Read and process each frame
for i in range(frame_count):
    ret, frame = cap.read()
    if not ret:
        break

    # Calculate index of the signal to display up to the current frame
    signal_idx = int((len(dt_pressure_data) * i) / frame_count)
    if circleOn:
        signal_idx_cyl = int((len(circle_data) * i) / frame_count)
    else:
        signal_idx_cyl = None
    signal_idx_exp = int((len(pressure_data) * i) / frame_count)

    # Get the plot image and resize it
    plot_image = get_plot_as_image(signal_idx, signal_idx_exp, signal_idx_cyl)
    plot_image_resized = cv2.resize(plot_image, (frame_width, frame_height//2))

    # Overlay the plot on the bottom quarter of the video frame
    frame[:(frame_height//2), :, :] = frame[(frame_height//4):(3*frame_height//4), :, :]
    frame[(frame_height//2):, :, :] = plot_image_resized

    # Write the updated frame to the output video
    out.write(frame)

# Release the video capture and writer objects
cap.release()
out.release()

print("Video processing complete. The output video with the plot is saved as:", output_video_path)
