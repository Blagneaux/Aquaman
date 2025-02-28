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

import matplotlib.animation as animation
def flow_reconstruction(signals):
    fig = plt.figure()  # Initialize the figure outside the loop

    img = []

    extract_coor = 128*127

    for i in signals.columns:
        frame = signals[i]
        frame = np.array(frame)
        frame[extract_coor] = 10
        frame = frame.reshape(256, 128)
        frame_rotated = np.rot90(frame)

        img.append([plt.imshow(frame_rotated, cmap="seismic", aspect='auto', vmin=-1, vmax=1)])

    ani = animation.ArtistAnimation(fig, img, interval=50, blit=True, repeat_delay=100)

    plt.show()

Correlations = []

path = 'E:/simuChina/cartesianBDD_FullPressureMap'
file = "FullMap.csv"
folder = sorted(os.listdir(path), key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else float('inf'))
plot_reconstruction = False

for sub_folder in folder:
    file_name = path + "/" + sub_folder + "/" + file
    correlation = []

    print("Main simulation file", file_name)
    signal = pd.read_csv(file_name, header=None)
    pressure_data = [signal[i][127*128] for i in signal.columns]
    pressure_data = normalisation(pressure_data)

    if plot_reconstruction:
        flow_reconstruction(signal)
        plt.figure()
        plt.plot(pressure_data)
        plt.show()

    for ref_sub_folder in folder:
        ref_file_name = path + "/" + ref_sub_folder + "/" + file

        print("Secondary simulation file", ref_file_name)
        ref_signal = pd.read_csv(ref_file_name, header=None)
        ref_pressure_data = [ref_signal[i][127*128] for i in ref_signal.columns]
        ref_pressure_data = normalisation(ref_pressure_data)
        
        # corr = correlate(ref_pressure_data, pressure_data)
        corr = correlate(ref_pressure_data - np.mean(ref_pressure_data), pressure_data - np.mean(pressure_data))
        corr /= np.sqrt(np.sum((ref_pressure_data - np.mean(ref_pressure_data))**2)*np.sum((pressure_data - np.mean(pressure_data))**2))
        lags = correlation_lags(len(ref_pressure_data), len(pressure_data))

        correlation.append(np.max(corr))
                            
    # correlation /= np.max(correlation)
    Correlations.append(correlation)
    correlation = []

fig = plt.figure()
correlation_array = np.array(Correlations)
# upper_triangle = np.triu(correlation_array)
# symetric_array = upper_triangle + upper_triangle.T - np.diag(np.diag(upper_triangle))
plt.imshow(correlation_array, cmap='viridis')
plt.colorbar()
plt.title("Cross-correlation Matrix between all the fish experimental data and cylinder simulations")

plt.grid(False)
plt.show()

print("Mean maximum cross-correlation between cylinder simulations: ", np.mean(correlation_array))
np.save("E:/crop_nadia/auto_cross-correlation_cylinder.npy", correlation_array)