from numpy import genfromtxt
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal

fc = 5
fs = 10000
w = fc / (fs / 2)
b,a = signal.butter(2,w,'low')

my_data = genfromtxt("C:/Users/blagn771/Desktop/FullPressure.csv", delimiter=",")
sensor_file = pd.read_excel("C:/Users/blagn771/Desktop/DATA_10Khz.xlsx", index_col=False)
sensor_file.columns = ["pressure"]

sensor_data = sensor_file["pressure"].to_numpy()
sensor_data = signal.filtfilt(b,a,sensor_data)
sensor_data_mv = sensor_data[:49001]

def extractData(data, coord):
    dimx, dimy = 642, 258
    cx, cy = coord #in mm from the start of the motion and on the first side hit by the positive pressure
    px, py = dimx-cx, dimy-(cy+29)
    Cp2P = 0.5*1000*0.0915*0.0915

    pos = py*dimx+px
    print("line ",pos)
    pressure_t = data[pos]*Cp2P

    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(pressure_t[:92], color="C0")
    ax.set_xlabel("Scaled time", color="C0")
    ax.set_ylabel("Pressure during the simulation at [302mm,20mm]", color="C0")
    ax.tick_params(axis='x', color="C0")
    ax.tick_params(axis='y', color="C0")
    ax.set_ylim([-2.5,2.5])

    ax2.plot(sensor_data_mv, color="C1")
    ax2.set_ylabel("Sensed pressure at [302mm, 20mm]", color="C1")
    ax2.set_xlabel("Scaled time", color="C1")
    ax2.xaxis.set_label_position('top')
    ax2.yaxis.set_label_position('right')
    ax2.xaxis.tick_top()
    ax2.yaxis.tick_right()
    ax2.tick_params(axis='x', color="C1")
    ax2.tick_params(axis='y', color="C1")
    ax2.set_ylim([-2.5,2.5])

    plt.show()

    return pressure_t

pressure = extractData(my_data,[290,30])