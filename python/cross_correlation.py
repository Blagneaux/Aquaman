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
        file_name = os.path.join(path, file)
        data = pd.read_csv(file_name)
        print("Simulation file", file)
        print(data)
        channels = data["Channel"]
        validities = data["Validity"]
        samples = data["Sample"]
        times = data["Time"]
        speeds = data["Speed"]
        starting_times = data["Starting_time"]
        ending_times = data["Ending_time"]
        starting_frames = data["Starting_offset"]
        ending_frames = data["Ending_offset"]
        dtws = data["DTW_score"]
        frechets = data["Frechet_dist"]

        for time, channel, speed, sample, starting_frame, starting_time, ending_frame, ending_time, validity, dtw, frechet in zip(times, channels, speeds, samples, starting_frames, starting_times, ending_frames, ending_times, validities, dtws, frechets):
            if validity and file[-6:-4] != 26: # 26 gives unrealistic values, probably due to the tail eating combined with the important size of the fish
                
                speed = speed * 0.5145 / 0.4 * 4 # conversion from speed in px/s@25fps to m/s

                # Initialize the comparison window based on the previous similarity analysis
                comparison_window_start = 2
                comparison_window_end = 2

                # Adapt the window if it is too large and extract the simulated map
                if starting_frame != 0 and ending_frame != 0:
                    ending_frame = ending_frame + starting_frame

                time_sensor = int(time*500)
                digital_twin_pressure = pd.read_csv(path_data+file[:-4]+"/"+str(sample)+"/pressure_map.csv")

                if ending_frame == 0:
                    ending_frame = len(digital_twin_pressure.columns) + starting_frame

                simu_crop_start = int(starting_frame)
                simu_crop_end = int(ending_frame)

                if time - (starting_time + starting_frame * 0.01) < comparison_window_start:
                    comparison_window_start = time - (starting_time + starting_frame * 0.01)
                else:
                    simu_crop_start = int((-comparison_window_start + time - starting_time)/ 0.01)

                if (starting_time + ending_frame * 0.01) - time < comparison_window_end:
                    comparison_window_end = (starting_time + ending_frame * 0.01) - time
                else:
                    simu_crop_end = int((comparison_window_end + time - starting_time)/ 0.01)

                # Extract the simulated pressure and build the time vector
                if channel == "S1":
                    dt_pressure_data = [1025 * speed * speed * digital_twin_pressure[i][(198)*128+37+43] for i in digital_twin_pressure.columns[simu_crop_start:simu_crop_end+1]]
                elif channel == "S2":
                    dt_pressure_data = [1025 * 0.0275 * 0.0275 * digital_twin_pressure[i][(146)*126+37+43] for i in digital_twin_pressure.columns[simu_crop_start:simu_crop_end+1]]
                elif channel == "S4":
                    dt_pressure_data = [1025 * 0.0275 * 0.0275 * digital_twin_pressure[i][(200)*128+39] for i in digital_twin_pressure.columns[simu_crop_start:simu_crop_end+1]]

                dt_pressure_data = bandpass_filter(dt_pressure_data, highcut=9, fs=100)
                X_dt = np.linspace(-comparison_window_start, comparison_window_end, len(dt_pressure_data))

                # Interpolate the simulated pressure so it has the same acquisition frequency as the experimental signal
                f = interp1d(X_dt, dt_pressure_data)
                X = np.linspace(-comparison_window_start, comparison_window_end, int(comparison_window_end+comparison_window_start) * 500)
                dt_pressure_data_interp = f(X)
                dt_pressure_data_interp = normalisation(dt_pressure_data_interp)

                # Loop through all the experimental pressure
                for ref_file in folder:
                    if ref_file.endswith(".csv"):
                        print(ref_file)
                        file_name_available = os.path.join(path, ref_file)
                        data_available = pd.read_csv(file_name_available)
                        validities_available = data_available["Validity"]
                        times_available = data_available["Time"]
                        samples_available = data_available["Sample"]
                        channels_available = data_available["Channel"]
                        dtws_available = data_available["DTW_score"]
                        frechets_available = data_available["Frechet_dist"]
                        
                        tdms_file = TdmsFile.read(path_data+"TDMS/"+ref_file[:-4]+".tdms")
                        for groupe in tdms_file.groups()[1:]:
                            for time_available, validity_available, sample_available, channel_available, dtw_available, frechet_available in zip(times_available, validities_available, samples_available, channels_available, dtws_available, frechets_available):
                                if validity_available and ref_file[-6:-4] != 26:
                                    time_sensor_available = int(time_available*500)

                                    for canal in groupe.channels():
                                        if canal.name == channel_available:
                                            pressure_data = bandpass_filter(canal.data, highcut=9)
                                            pressure_data = pressure_data[time_sensor_available - 1000: time_sensor_available + 1 + 1000]
                                            pressure_data = normalisation(pressure_data)
                                            X_available = np.linspace(-2, 2, 2001)

                                            corr = correlate(pressure_data, dt_pressure_data_interp)
                                            lags = correlation_lags(len(pressure_data), len(dt_pressure_data_interp))
                                            # corr /= max(np.max(corr), np.abs(np.min(corr)))

                                            # fig, (ax, ax_corr) = plt.subplots(2,1,figsize=(5,5))
                                            # ax.plot(X_available, pressure_data)
                                            # ax.set_title(f"Experimental Pressure for sample {sample_available} at sensor {channel_available}")
                                            # ax.set_xlabel("Time")

                                            # ax.plot(X, dt_pressure_data_interp)

                                            # ax_corr.plot(lags, corr)
                                            # ax_corr.set_title("Cross-correlated signal")
                                            # ax_corr.set_xlabel("Lag")

                                            # fig.tight_layout()
                                            # plt.show()
                                            correlation.append(np.max(corr))
                correlation /= np.max(correlation)
                Correlations.append(correlation)
                correlation = []

fig = plt.figure()
correlation_array = np.array(Correlations)
upper_triangle = np.triu(correlation_array)
symetric_array = upper_triangle + upper_triangle.T - np.diag(np.diag(upper_triangle))
plt.imshow(symetric_array, cmap='viridis')
plt.colorbar()
plt.title("Cross-correlation Matrix between all the experiments and the simulations")
plt.xlabel("Experimental sample")
plt.ylabel("Simulation sample")

plt.grid(False)
plt.show()

print("Mean maximum cross-correlation between simulation and experiments: ",np.mean(symetric_array))