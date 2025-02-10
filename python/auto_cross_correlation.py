import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os
import re
from scipy.signal import butter, filtfilt, correlate, correlation_lags
from nptdms import TdmsFile

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

# Normaliser les signaux pour empecher qu'un signal plus intense n'ecrase toutes les autres correlation
def normalisation(data):
    abs_max_data = np.max(np.abs(data))
    normalized_data = data / abs_max_data
    return normalized_data

Correlations = []

path = "E:/crop_nadia/matchingData"
path_data = "E:/crop_nadia/"
folder = sorted(os.listdir(path), key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else float('inf'))
for file in folder:
    if file.endswith(".csv"):
        correlation = []

        print("Simulation file", file)
        file_name = os.path.join(path, file)
        data = pd.read_csv(file_name)
        validities = data["Validity"]
        times = data["Time"]
        samples = data["Sample"]
        channels = data["Channel"]

        tdms_file = TdmsFile.read(path_data+"TDMS/"+file[:-4]+".tdms")
        for groupe in tdms_file.groups()[1:]:
            for time, validity, sample, channel in zip(times, validities, samples, channels):
                if validity and file[-6:-4] != 26:
                    time_sensor = int(time * 500)

                    for canal in groupe.channels():
                        if canal.name == channel:
                            pressure_data = bandpass_filter(canal.data, highcut=9)
                            pressure_data = pressure_data[time_sensor - 1000: time_sensor + 1 + 1000]
                            pressure_data = normalisation(pressure_data)
                            X = np.linspace(-2, 2, 2001)

                            for ref_file in folder:
                                if ref_file.endswith(".csv"):
                                    print(ref_file)
                                    ref_file_name = os.path.join(path, ref_file)
                                    ref_data = pd.read_csv(ref_file_name)
                                    ref_validities = ref_data["Validity"]
                                    ref_times = ref_data["Time"]
                                    ref_samples = ref_data["Sample"]
                                    ref_channels = ref_data["Channel"]

                                    ref_tdms_file = TdmsFile.read(path_data+"TDMS/"+ref_file[:-4]+".tdms")
                                    for ref_groupe in ref_tdms_file.groups()[1:]:
                                        for ref_time, ref_validity, ref_sample, ref_channel in zip(ref_times, ref_validities, ref_samples, ref_channels):
                                            if ref_validity and ref_file[-6:-4] != 26:
                                                ref_time_sensor = int(ref_time * 500)

                                                for ref_canal in ref_groupe.channels():
                                                    if ref_canal.name == ref_channel:
                                                        ref_pressure_data = bandpass_filter(ref_canal.data, highcut=9)
                                                        ref_pressure_data = ref_pressure_data[ref_time_sensor - 1000: ref_time_sensor + 1 + 1000]
                                                        ref_pressure_data = normalisation(ref_pressure_data)
                                                        ref_X = np.linspace(-2, 2, 2001)

                                                        corr = correlate(ref_pressure_data, pressure_data)
                                                        lags = correlation_lags(len(ref_pressure_data), len(pressure_data))

                                                        correlation.append(np.max(corr))
                            
                            # correlation /= np.max(correlation)
                            Correlations.append(correlation)
                            correlation = []

fig = plt.figure()
correlation_array = np.array(Correlations)
upper_triangle = np.triu(correlation_array)
symetric_array = upper_triangle + upper_triangle.T - np.diag(np.diag(upper_triangle))
plt.imshow(symetric_array, cmap='viridis')
plt.colorbar()
plt.title("Cross-correlation Matrix between all the experiments and themselves")
plt.xlabel("Experimental sample")
plt.ylabel("Experimental sample")

plt.grid(False)
plt.show()

print("Mean maximum cross-correlation between experiments: ", np.mean(symetric_array))