import numpy as np
import pandas as pd 
from scipy.signal import filtfilt, butter

# Fonction pour créer un filtre passe-bande de second ordre
def butter_bandpass(lowcut, highcut, fs, order=2):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Appliquer le filtre passe-bande aux données
def bandpass_filter(data_path, lowcut=0.3, highcut=30, fs=200, order=2):
    data = pd.read_csv(data_path)
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    meta = {"b":b,
            "a":a,
            "order":order,
            "type":"butter"}
    timestamp = data["timestamp"]
    camera = data["Etat_Camera"]
    timeframe = data["Time_Frame"]
    filtered_data = timestamp
    for col in data.columns:
        if col[:3] == "ESP":
            filterd_col = filtfilt(b, a, data[col])
            filtered_data = pd.concat([filtered_data, pd.DataFrame(filterd_col)], axis=1)
    filtered_data = pd.concat([filtered_data, camera], axis=1)
    filtered_data = pd.concat([filtered_data, timeframe], axis=1)
    filtered_data.columns = data.columns

    filtered_data = pd.concat([filtered_data, pd.DataFrame(meta)], axis=1)

    filtered_data.to_csv(data_path[:-4]+"filtered.csv", index=False)

if __name__== "__main__":
    bandpass_filter("C:/Users/blagn771/Downloads/T1.csv")