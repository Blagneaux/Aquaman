import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
from scipy import signal

file_path = 'C:/Users/blagn771/Documents/Aquaman/Aquaman/lily-pad-master/LilyPad/testDataSave/pressure_map_test.csv'
ref_path = 'python/data_08092023_x242y56_exp1.xlsx'

recreateVideo = True
plotSignal = False
plotComp = False

# Use the read_csv function to read the CSV file into a Pandas DataFrame
df = pd.read_csv(file_path, header=None)
df_ref = pd.read_excel(ref_path, header=None)
num_columns = df.shape[1]
min, max = min(df.min()[1:]), max(df.max()[1:])
chord = 32
len = 74.2
posx = 242
posy = 100 - 56
print(min, max)

def reformat(x):
    return 255 * (x - min)  / (max - min)

def extractCoord(i,j):
    return df[i][j]

def data_gen():
    i, j = int(16*chord - posx*chord/len), int(2*chord - posy*chord/len)
    for k in range(T.size):
        yield T[k], extractCoord(k, (i-1)*4*chord+j) * 0.5 * 1000 * 0.089 * 0.089 # P = Cp * rho * U^2 / 2

def init():
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlim(0, 181)
    del xdata[:]
    del ydata[:]
    line.set_data(xdata, ydata)
    return line

def run(data):
    # update the data
    x, y = data
    xdata.append(x)
    ydata.append(y)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    if x >= xmax:
        ax.set_xlim(xmin, 2*xmax)
        ax.figure.canvas.draw()
    if y >= ymax or y <= ymin:
        ax.set_ylim(2*ymin, 2*ymax)
        ax.figure.canvas.draw()
    line.set_data(xdata, ydata)

    return line, 

if recreateVideo:
    # Set the values to be shades of gray
    df1 = df.apply(reformat)
    df1.clip(0, 255)

    # Create the video
    size = 2**6, 2**6
    duration = num_columns
    fps = 25
    out = cv2.VideoWriter('dataMap.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), False)

    for col_name in df.columns:
        data = df1[col_name].to_numpy(np.uint8)
        # # Place the sensor in the video
        # i, j = int(16*chord - posx*chord/len), int(2*chord - posy*chord/len)
        # data[(i-1)*4*chord+j] = 255
        out.write(np.array(data.reshape(size, order='F')))
    out.release()

if plotSignal:
    T = np.linspace(0,180, df.columns.size)
    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2)
    ax.grid()
    xdata, ydata = [], []

    ani = animation.FuncAnimation(fig, run, data_gen, init_func=init, interval=40, save_count=df.columns.size)

    # plt.show()

    writervideo = animation.FFMpegWriter(fps=25) 
    mpl.rcParams['animation.ffmpeg_path'] = r'C:/Users/blagn771\Downloads/ffmpeg-master-latest-win64-gpl/bin/ffmpeg.exe'
    ani.save('signal.mp4', writer=writervideo)

if plotComp:
    i, j = int(16*chord - posx*chord/len), int(2*chord - posy*chord/len)
    dim = 0.5 * 1000 * 0.089 * 0.089
    simP = []
    simT = np.linspace(0, 4.9, 538)
    for k in range(df.columns.size):
        simP.append(extractCoord(k, (i-1)*4*chord+j) * dim)

    sensorMean = np.mean(df_ref[1])

    # Filter sensor data
    N = 2
    fc1 = 20
    fs = 1000
    Wn = fc1/(fs/2)
    btype = 'low'
    b, a = signal.butter(N, Wn, btype=btype)
    filtered_sensor = signal.filtfilt(b, a, df_ref[1] - sensorMean)
    trimmed_sensor = filtered_sensor[df_ref[2] != 0]
    trimmed_sensor = trimmed_sensor[:4901]
    refT = np.linspace(0,4.9,trimmed_sensor.size)

    plt.plot(simT, simP[:538], label="Simulation data")
    plt.plot(refT, -trimmed_sensor, label="Sensor data")

    plt.xlabel('Time')
    plt.ylabel('Pressure')
    plt.legend()

    plt.show()
