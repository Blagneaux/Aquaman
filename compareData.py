from numpy import genfromtxt
import matplotlib.pyplot as plt
import pandas as pd

my_data = genfromtxt("C:/Users/blagn771/Desktop/FullPressure.csv", delimiter=",")
sensor_file = pd.read_excel("C:/Users/blagn771/Downloads/data_2.xlsx", index_col=False)
sensor_file.columns = ["time", "pressure"]

sensor_data = sensor_file["pressure"].to_numpy()
sensor_data_mv = sensor_data[64000:-1]-0.5

def extractData(data, coord):
    dimx, dimy = 642, 258
    cx, cy = coord #in mm from the start of the motion and on the first side hit by the positive pressure
    px, py = dimx-cx-42, cy+29
    Cp2P = 0.5*1000*0.089*0.089

    pos = py*dimx+px
    print("line ",pos)
    pressure_t = data[pos]*Cp2P

    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(pressure_t, color="C0")
    ax.set_xlabel("Scaled time", color="C0")
    ax.set_ylabel("Pressure during the simulation at [302mm,20mm]", color="C0")
    ax.tick_params(axis='x', color="C0")
    ax.tick_params(axis='y', color="C0")
    ax.set_ylim([-1.75,1.75])

    ax2.plot(sensor_data_mv, color="C1")
    ax2.set_ylabel("Sensed pressure at [302mm, 20mm]", color="C1")
    ax2.set_xlabel("Scaled time", color="C1")
    ax2.xaxis.set_label_position('top')
    ax2.yaxis.set_label_position('right')
    ax2.xaxis.tick_top()
    ax2.yaxis.tick_right()
    ax2.tick_params(axis='x', color="C1")
    ax2.tick_params(axis='y', color="C1")
    ax2.set_ylim([-1.75,1.75])

    plt.show()

    return pressure_t

pressure = extractData(my_data,[302,20])