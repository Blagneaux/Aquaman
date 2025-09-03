import numpy as np
import pandas as pd
from nptdms import TdmsFile
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, correlate, correlation_lags
from dtw import *
import similaritymeasures
from scipy.interpolate import interp1d


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

def normalisation(data):
    abs_max_data = np.max(np.abs(data))
    normalized_data = data / abs_max_data
    return normalized_data


channels = ["S1", "S2", "S4"]
show = True
isCirle = False
# file_numbers = [1,2,3,6,7,8,9,10,11,14,15,16,17,18,22,23,24,25,26,27,28,29,30,31,32,33,35,40]
file_numbers = [40]
for file_number in file_numbers:
    print("______________________________________________________________________________")
    print(file_number)
    print("______________________________________________________________________________")
    # file_number = 1 # 28-2 with a DTW of 60.3 and a Frechet of 0.20 show a good second order fit, 14-1 with DTW of 21.3 and Frechet of 0.18 is nice to show a good first order fit and 17-4 with DTW of 156.8 and Frechet of 0.90 is nice for about the mean
    passage_time = pd.read_csv("D:/crop_nadia/passage_time/"+str(file_number)+".csv")
    tdms_file = TdmsFile.read("D:/crop_nadia/TDMS/"+str(file_number)+".tdms")
    digital_twin_time = pd.read_csv("D:/crop_nadia/timestamps/timestamps"+str(file_number)+".csv")

    digital_twin_time["start_time"] = digital_twin_time["start_time"].apply(time_to_seconds)
    digital_twin_time["end_time"] = digital_twin_time["end_time"].apply(time_to_seconds)

    clean_data_df = None

    print(digital_twin_time)

    for groupe in tdms_file.groups()[1:]:
        print("Groupe:", groupe)
        for channel in channels:
            print("Testing channel:", channel)
            if channel in passage_time.columns:
                print("ok for channel", channel)
                times = np.array(passage_time[channel])
                speeds = np.array(passage_time["v"+channel])
                times = times[~np.isnan(times)]
                speeds = speeds[~np.isnan(speeds)]
                times = times[times != 0.]
                speeds = speeds[speeds != 0.]

                lilypad_sample_indexes = []
                start_times = []
                start_frame = []
                end_times = []
                end_frame = []
                validitys = []
                for t in times:
                    for row_index in range(len(digital_twin_time)):
                        if t >= digital_twin_time["start_time"][row_index] and t <= digital_twin_time["end_time"][row_index]:
                            lilypad_sample_indexes.append(row_index + 1)
                            start_times.append(digital_twin_time["start_time"][row_index])
                            start_frame.append(digital_twin_time["start_frame"][row_index])
                            end_times.append(digital_twin_time["end_time"][row_index])
                            end_frame.append(digital_twin_time["end_frame"][row_index])
                            validitys.append(digital_twin_time["validity"][row_index])
                print("Passage times:", times)
                print("Passage speed:", speeds)
                print("Sample of the video corresponding to the passages:", lilypad_sample_indexes)
                print("Starting time of the video used for the DT (in s):", start_times)
                print("Offset frame for the begining of the video used for the DT (in frames):", start_frame)
                print("Ending time of the video used for the DT (in s):", end_times)
                print("Offset frame for the end (from the begining) of the video used for the DT (in frames):", end_frame)
                print("Validity of the DT simulation:", validitys)
                print("\n")

                DTW_score = []
                Frechet_distance = []
                Correlations = []

                for time, speed, sample, starting_time, starting_frame, ending_time, ending_frame, validity in zip(times, speeds, lilypad_sample_indexes, start_times, start_frame, end_times, end_frame, validitys):
                    if validity:

                        speed = speed * 0.5145 / 0.4 * 4 # conversion from speed in px/s@25fps to m/s

                        comparison_window_start = 1
                        comparison_window_end = 1

                        if starting_frame != 0 and ending_frame != 0:
                            ending_frame = ending_frame + starting_frame

                        time_sensor = int(time*500)
                        digital_twin_pressure = pd.read_csv("D:/crop_nadia/"+str(file_number)+"/"+str(sample)+"/pressure_map.csv")
                        if isCirle:
                            digital_circle_pressure = pd.read_csv("D://crop_nadia/"+str(file_number)+"/"+str(sample)+"/circle_pressure_map.csv")

                        print("Number of frames in the simulation:", len(digital_twin_pressure.columns))
                        if isCirle:
                            print("Number of frames in the circle simulation:", len(digital_circle_pressure.columns))
                        if ending_frame == 0:
                            ending_frame = len(digital_twin_pressure.columns) + starting_frame
                        
                        if isCirle:
                            ending_frame_cirle = len(digital_circle_pressure.columns) + starting_frame
                            if ending_frame_cirle < ending_frame:
                                ending_frame = ending_frame_cirle

                        simu_crop_start = starting_frame
                        simu_crop_end = ending_frame

                        if time - (starting_time + starting_frame * 0.01) < comparison_window_start:
                            comparison_window_start = time - (starting_time + starting_frame * 0.01)
                        else:
                            simu_crop_start = int((-comparison_window_start + time - starting_time)/ 0.01)

                        if (starting_time + ending_frame * 0.01) - time < comparison_window_end:
                            comparison_window_end = (starting_time + ending_frame * 0.01) - time
                        else:
                            simu_crop_end = int((comparison_window_end + time - starting_time)/ 0.01)
                        
                        print("The size of the comparison time window before the fish passing in front of the sensor is", comparison_window_start, "(in s)")
                        print("The size of the comparison time window after the fish passing in front of the sensor is", comparison_window_end, "(in s)")
                        print("Simulation starting offset:", simu_crop_start)
                        print("Simulation ending offset:", simu_crop_end)

                        for canal in groupe.channels():
                            if canal.name == channel:
                                pressure_data = bandpass_filter(canal.data, highcut=9)
                                pressure_data = pressure_data[time_sensor - int(comparison_window_start*500): time_sensor + 1 + int(comparison_window_end*500)] # 2s before and 2s after the fish passing in front of the sensor
                                pressure_data = normalisation(pressure_data)
                                print("Pressure data for channel", channel, ":", pressure_data)

                                X = np.linspace(-comparison_window_start,comparison_window_end,len(pressure_data))

                                if channel == "S1":
                                    if not isCirle:
                                        dt_pressure_data = [1025 * speed * speed * digital_twin_pressure[i][(198)*128+37+43] for i in digital_twin_pressure.columns[simu_crop_start:simu_crop_end+1]]
                                        X_dt = np.linspace(-comparison_window_start,comparison_window_end,len(dt_pressure_data))
                                    else:
                                        dt_pressure_data = [1025 * speed * speed * digital_circle_pressure[i][(198)*128+37+43] for i in digital_circle_pressure.columns[simu_crop_start:simu_crop_end+1]]
                                        X_dt = np.linspace(-comparison_window_start,comparison_window_end,len(dt_pressure_data))
                                    
                                    # f = interp1d(X_dt, dt_pressure_data)
                                    # dt_pressure_data_interp = f(X)
                                    
                                    dt_pressure_data = bandpass_filter(dt_pressure_data, highcut=9, fs=100)
                                    dt_pressure_data = normalisation(dt_pressure_data)
                                    if np.isnan(dt_pressure_data[0]):
                                        dtw_score = np.nan
                                        frechet_distance = np.nan
                                        correlation = np.nan
                                    else:
                                        dtw_score = dtw(pressure_data, dt_pressure_data, keep_internals=True).distance
                                        pressure_curve = np.column_stack((X, pressure_data))
                                        dt_curve = np.column_stack((X_dt, dt_pressure_data))
                                        frechet_distance = similaritymeasures.frechet_dist(pressure_curve, dt_curve)
                                        correlation = correlate(pressure_data - np.mean(pressure_data), dt_pressure_data - np.mean(dt_pressure_data))
                                        correlation /= np.sqrt(np.sum((pressure_data - np.mean(pressure_data))**2)*np.sum((dt_pressure_data - np.mean(dt_pressure_data))**2))
                                    DTW_score.append(dtw_score)
                                    Frechet_distance.append(frechet_distance)
                                    Correlations.append(np.max(correlation))
                                    if show:
                                        plt.figure()
                                        print("DTW score:", dtw_score)
                                        print("Frechet distance:", frechet_distance)
                                        print("Correlation:", np.max(correlation))
                                        plt.plot(X, pressure_data, label="Experimental pressure")
                                        plt.plot(X_dt, dt_pressure_data, label="Simulated pressure ")
                                        plt.xlabel("Time (in s) centered on the passage of the fish in front of the sensor")
                                        plt.ylabel("Pressure (normalized)")
                                        plt.legend()
                                        plt.title("Comparison of the evolution of the pressure extracted from the simulation \n with the evolution of the pressure measured by the sensors")
                                        # dtw(pressure_data, dt_pressure_data, keep_internals=True,
                                        #     step_pattern=rabinerJuangStepPattern(6, "c"))\
                                        #     .plot(type="twoway",offset=-2)
                                        plt.show()
                                elif channel == "S2":
                                    if not isCirle:
                                        dt_pressure_data = [1025 * speed * speed * digital_twin_pressure[i][(146)*128+37+43] for i in digital_twin_pressure.columns[simu_crop_start:simu_crop_end+1]]
                                        X_dt = np.linspace(-comparison_window_start,comparison_window_end,len(dt_pressure_data))
                                    else:
                                        dt_pressure_data = [1025 * speed * speed * digital_circle_pressure[i][(146)*128+37+43] for i in digital_circle_pressure.columns[simu_crop_start:simu_crop_end+1]]
                                        X_dt = np.linspace(-comparison_window_start,comparison_window_end,len(dt_pressure_data))
                                    
                                    # f = interp1d(X_dt, dt_pressure_data)
                                    # dt_pressure_data_interp = f(X)

                                    dt_pressure_data = bandpass_filter(dt_pressure_data, highcut=9, fs=100)
                                    dt_pressure_data = normalisation(dt_pressure_data)
                                    if np.isnan(dt_pressure_data[0]):
                                        dtw_score = np.nan
                                        frechet_distance = np.nan
                                        correlation = np.nan
                                    else:
                                        dtw_score = dtw(pressure_data, dt_pressure_data, keep_internals=True).distance
                                        pressure_curve = np.column_stack((X, pressure_data))
                                        dt_curve = np.column_stack((X_dt, dt_pressure_data))
                                        frechet_distance = similaritymeasures.frechet_dist(pressure_curve, dt_curve)
                                        correlation = correlate(pressure_data - np.mean(pressure_data), dt_pressure_data - np.mean(dt_pressure_data))
                                        correlation /= np.sqrt(np.sum((pressure_data - np.mean(pressure_data))**2)*np.sum((dt_pressure_data - np.mean(dt_pressure_data))**2))
                                    DTW_score.append(dtw_score)
                                    Frechet_distance.append(frechet_distance)
                                    Correlations.append(np.max(correlation))
                                    if show:
                                        plt.figure()
                                        print("DTW score:", dtw_score)
                                        print("Frechet distance:", frechet_distance)
                                        plt.plot(X, pressure_data, label="Experimental pressure")
                                        plt.plot(X_dt, dt_pressure_data, label="Simulated pressure ")
                                        plt.xlabel("Time (in s) centered on the passage of the fish in front of the sensor")
                                        plt.ylabel("Pressure (normalized)")
                                        plt.legend()
                                        plt.title("Comparison of the evolution of the pressure extracted from the simulation \n with the evolution of the pressure measured by the sensors")
                                        # dtw(pressure_data, dt_pressure_data, keep_internals=True,
                                        #     step_pattern=rabinerJuangStepPattern(6, "c"))\
                                        #     .plot(type="twoway",offset=-2)
                                        plt.show()
                                elif channel == "S4":
                                    if not isCirle:
                                        dt_pressure_data = [1025 * speed * speed * digital_twin_pressure[i][(200)*128+39] for i in digital_twin_pressure.columns[simu_crop_start:simu_crop_end+1]]
                                        X_dt = np.linspace(-comparison_window_start,comparison_window_end,len(dt_pressure_data))
                                    else:
                                        dt_pressure_data = [1025 * speed * speed * digital_circle_pressure[i][(200)*128+39] for i in digital_circle_pressure.columns[simu_crop_start:simu_crop_end+1]]
                                        X_dt = np.linspace(-comparison_window_start,comparison_window_end,len(dt_pressure_data))
                                    
                                    # f = interp1d(X_dt, dt_pressure_data)
                                    # dt_pressure_data_interp = f(X)

                                    dt_pressure_data = bandpass_filter(dt_pressure_data, highcut=9, fs=100)
                                    dt_pressure_data = normalisation(dt_pressure_data)
                                    if np.isnan(dt_pressure_data[0]):
                                        dtw_score = np.nan
                                        frechet_distance = np.nan
                                        correlation = np.nan
                                    else:
                                        dtw_score = dtw(pressure_data, dt_pressure_data, keep_internals=True).distance
                                        pressure_curve = np.column_stack((X, pressure_data))
                                        dt_curve = np.column_stack((X_dt, dt_pressure_data))
                                        frechet_distance = similaritymeasures.frechet_dist(pressure_curve, dt_curve)
                                        correlation = correlate(pressure_data - np.mean(pressure_data), dt_pressure_data - np.mean(dt_pressure_data))
                                        correlation /= np.sqrt(np.sum((pressure_data - np.mean(pressure_data))**2)*np.sum((dt_pressure_data - np.mean(dt_pressure_data))**2))
                                    DTW_score.append(dtw_score)
                                    Frechet_distance.append(frechet_distance)
                                    Correlations.append(np.max(correlation))
                                    if show:
                                        plt.figure()
                                        print("DTW score:", dtw_score)
                                        print("Frechet distance:", frechet_distance)
                                        plt.plot(X, pressure_data, label="Experimental pressure")
                                        plt.plot(X_dt, dt_pressure_data, label="Simulated pressure ")
                                        plt.xlabel("Time (in s) centered on the passage of the fish in front of the sensor")
                                        plt.ylabel("Pressure (normalized)")
                                        plt.legend()
                                        plt.title("Comparison of the evolution of the pressure extracted from the simulation \n with the evolution of the pressure measured by the sensors")
                                        # dtw(pressure_data, dt_pressure_data, keep_internals=True,
                                        #     step_pattern=rabinerJuangStepPattern(6, "c"))\
                                        #     .plot(type="twoway",offset=-2)
                                        plt.show()
                    else:
                        DTW_score.append(np.nan)
                        Frechet_distance.append(np.nan)
                        Correlations.append(np.nan)

                if len(lilypad_sample_indexes) > 0:
                    if clean_data_df is not None:
                        new_clean_data = pd.DataFrame({
                            'Channel': [channel for t in times],
                            'Time': times,
                            'Speed': speeds,
                            'Sample': lilypad_sample_indexes,
                            'Starting_time': start_times,
                            'Starting_offset': start_frame,
                            'Ending_time': end_times,
                            'Ending_offset': end_frame,
                            'Validity': validitys,
                            'DTW_score': DTW_score,
                            'Frechet_dist': Frechet_distance,
                            'Correlation': Correlations
                            })
                        clean_data_df = pd.concat([clean_data_df, new_clean_data], ignore_index=True)
                    else:
                        clean_data = {
                            'Channel': [channel for t in times],
                            'Time': times,
                            'Speed': speeds,
                            'Sample': lilypad_sample_indexes,
                            'Starting_time': start_times,
                            'Starting_offset': start_frame,
                            'Ending_time': end_times,
                            'Ending_offset': end_frame,
                            'Validity': validitys,
                            'DTW_score': DTW_score,
                            'Frechet_dist': Frechet_distance,
                            'Correlation': Correlations
                        }
                        clean_data_df = pd.DataFrame(clean_data)
                else:
                    print("No matching data")

                print("\n")


    print("Clean data:", clean_data_df)
    if clean_data_df is not None:
        if not isCirle:
            clean_data_df.to_csv("D:/crop_nadia/matchingData_0.5s_window/"+str(file_number)+".csv")
        else:
            clean_data_df.to_csv("D:/crop_nadia/matchingData_circle_0.5s_window/circle_"+str(file_number)+".csv")
    else:
        print("No matching data")