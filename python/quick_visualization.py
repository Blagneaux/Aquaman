import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from nptdms import TdmsFile

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

file_number = 1
sample = 15
sensor = "S4"
isOkForCrop = True

digital_twin_time = pd.read_csv("D:/crop_nadia/timestamps/timestamps"+str(file_number)+".csv")
tdms_file = TdmsFile.read("D:/crop_nadia/TDMS/"+str(file_number)+".tdms")
passage_time = pd.read_csv("D:/crop_nadia/passage_time/"+str(file_number)+".csv")

if isOkForCrop:
    digital_twin_pressure = pd.read_csv("D:/crop_nadia/"+str(file_number)+"/"+str(sample)+"/pressure_map.csv")
else:
    digital_twin_pressure = pd.read_csv("D:/crop_nadia/test_vortex/"+str(file_number)+"_"+str(sample)+"_pressure_map.csv")

digital_twin_time["start_time"] = digital_twin_time["start_time"].apply(time_to_seconds)
digital_twin_time["end_time"] = digital_twin_time["end_time"].apply(time_to_seconds)

start_time = digital_twin_time["start_time"][sample-1]
end_time = digital_twin_time["end_time"][sample-1]
if not isOkForCrop:
    end_time = start_time + 2*(end_time-start_time)
print("Starting time (in s):", start_time)
print("Ending time (in s):", end_time)
print("Number of steps:", len(digital_twin_pressure.columns))

posX = None
if sensor == "S1":
    posX = 198*128 + 37+43
elif sensor == "S2":
    posX = 146*128 + 37+43
elif sensor == "S4":
    posX = 200*128 + 39

dt_pressure_data = [digital_twin_pressure[i][posX] for i in digital_twin_pressure.columns]
X_dt = np.linspace(start_time, end_time, len(dt_pressure_data))

for groupe in tdms_file.groups()[1:]:
    for canal in groupe.channels():
        if canal.name == sensor:
            pressure_data = bandpass_filter(canal.data, highcut=9)
            pressure_data = pressure_data[int(start_time*500): int(end_time*500)]
            pressure_data = normalisation(pressure_data)

X_exp = np.linspace(start_time, end_time, int(end_time*500)-int(start_time*500))

dt_pressure_data = bandpass_filter(dt_pressure_data, highcut=9, fs=100)
dt_pressure_data = normalisation(dt_pressure_data)

# Plot the comparison between the Digital Twin and the experiments, and the area around in the simulation
fig, axs = plt.subplots(2, sharex=True)
fig.suptitle(f"Can we find a vortex in the {file_number, sample} DT?")
axs[0].set_title("Comparison of DT and Exp")
axs[0].plot(X_dt, dt_pressure_data, label="Digital Twin")
axs[0].plot(X_exp, pressure_data, label="Experiment")
axs[0].legend()
axs[1].set_title("Extraction of the signal with around the sensor")
for offset in np.linspace(-10, 10, 11):
    if offset == 0:
        axs[1].plot(X_dt, bandpass_filter([digital_twin_pressure[i][posX + offset*128] for i in digital_twin_pressure.columns], highcut=9, fs=100), 'k-o', label=f"Offset: {offset}")
    else:
        axs[1].plot(X_dt, bandpass_filter([digital_twin_pressure[i][posX + offset*128] for i in digital_twin_pressure.columns], highcut=9, fs=100), label=f"Offset: {offset}")
plt.legend()

# plt.figure()
# plt.plot(X_dt, normalisation(bandpass_filter([digital_twin_pressure[i][posX -6*128] for i in digital_twin_pressure.columns], highcut=9, fs=100)), label="Offset")
# plt.plot(X_exp, pressure_data, label="Experiment")
plt.show()