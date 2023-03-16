from numpy import genfromtxt
import matplotlib.pyplot as plt
import pandas as pd

my_data = genfromtxt("C:/Users/blagn771/Desktop/FullPressure.csv", delimiter=",")
sensor_file = pd.read_excel("C:/Users/blagn771/Desktop/DATA.xlsx", index_col=False)
sensor_file.columns = ["time", "pressure"]

sensor_data = sensor_file["pressure"].to_numpy()
sensor_data_mv = sensor_data[219:692]+3

def extractData(data, coord):
    dimx, dimy = 642, 258
    cx, cy = coord #in mm from the start of the motion and on the first side hit by the positive pressure
    px, py = dimx-cx-42, dimy-cy-29
    Cp2P = 0.5*1000*0.089*0.089

    pos = py*dimx+px
    print("line ",pos)
    pressure_t = data[pos]*Cp2P

    plt.plot(pressure_t[:96])
    plt.plot(sensor_data_mv[::5])
    plt.ylabel("Pressure coefficient during the simulation at the position of the sensor")
    plt.xlabel("Scaled time")
    plt.show()

    return pressure_t

pressure = extractData(my_data,[302,20])