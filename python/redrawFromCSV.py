import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
import cv2

file_path = 'C:/Users/blagn771/Documents/Aquaman/Aquaman/lily-pad-master/LilyPad/testDataSave/pressure_data.csv'

recreateVideo = True
plotSignal = True

# Use the read_csv function to read the CSV file into a Pandas DataFrame
df = pd.read_csv(file_path, header=None)
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
        print(k, T[k], extractCoord(k, (i-1)*4*chord+j))
        yield T[k], extractCoord(k, (i-1)*4*chord+j)

def init():
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlim(0, 181)
    del xdata[:]
    del ydata[:]
    line.set_data(xdata, ydata)
    return line

def run(data):
    # update the data
    t, y = data
    xdata.append(t)
    ydata.append(y)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    if t >= xmax:
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
    size = 4*chord, 16*chord
    duration = num_columns
    fps = 25
    out = cv2.VideoWriter('dataMap.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), False)

    for col_name in df.columns:
        data = df1[col_name].to_numpy(np.uint8)
        i, j = int(16*chord - posx*chord/len), int(2*chord - posy*chord/len)
        data[(i-1)*4*chord+j] = 255
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
