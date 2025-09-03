import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Parameters
file_path = "C:/Users/blagn771/Downloads/pressureMotion.xlsx"  # Replace with your actual file
fs = 500  # Sampling frequency in Hz (change this as needed)
cutoff = 3  # Cutoff frequency in Hz
order = 2  # Butterworth filter order

# Read data
df = pd.read_excel(file_path, usecols="A:H", header=None)

# Subtract mean
df_demeaned = df - df.mean()

# Design Butterworth filter
b, a = butter(order, cutoff / (0.5 * fs), btype='low')

# Apply filter
df_filtered = df_demeaned.apply(lambda col: filtfilt(b, a, col), axis=0)

# Plot results
plt.figure(figsize=(12, 8))
for i, column in enumerate(df_filtered.columns):
    plt.plot(df_filtered.index, df_filtered[column], label=f'Sensor {column}')
plt.xlabel('Sample Index')
plt.ylabel('Filtered Pressure')
plt.title('Filtered Pressure Signals from Sensors A-H')
plt.ylim(-250,250)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
