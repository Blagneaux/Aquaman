import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Read the file, skipping the first four lines
file_path = "C:/Users/blagn771/Documents/Aquaman/Aquaman/lily-pad-master/LilyPad/saved/pressure.txt"
with open(file_path, 'r') as file:
    lines = file.readlines()[4:]

# Extract time and pressure values
time = []
pressure = []
for line in lines:
    try:
        t, p = map(float, line.strip(" ;\n").split())
        time.append(t)
        pressure.append(p*0.5*1000*0.13*0.13)
    except ValueError:
        continue  # skip lines that don't contain valid data

# Apply low-pass Butterworth filter
def lowpass_filter(data, cutoff_freq, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered = filtfilt(b, a, data)
    return filtered

# Estimate sampling frequency from time data
if len(time) > 1:
    dt = time[1] - time[0]
    fs = 1.0 / dt
else:
    fs = 1.0  # fallback

cutoff_frequency = 0.1  # Hz (adjust depending on how much smoothing you want)
pressure_filtered = lowpass_filter(pressure, cutoff_frequency, fs)

# Plotting
plt.figure(figsize=(12, 6))
# plt.plot(time, pressure, label='Original', alpha=0.4)
plt.plot(time, pressure_filtered, label='Filtered', linewidth=2)
plt.title("Pressure vs Time (with Low-Pass Filter)")
plt.xlabel("Time")
plt.ylabel("Pressure")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()