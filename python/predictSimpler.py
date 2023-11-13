from ultralytics import YOLO
import cv2
import numpy as np
import csv
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy import signal

model = YOLO("yolov8n-seg-customNaca-mid.pt")
cap = cv2.VideoCapture("dataNaca_ref.mp4")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

size = (frame_width, frame_height)

a = [0, 0.2, -0.1]
c = 2**6/3
xc = 2**6/4 - 0.25*c
k = 2*np.pi / c
omega = 1.2*k
T = 2*np.pi/omega
s = 0
for ai in a:
    s+= ai
if s==0:
    s=1
    a[0] = 1
for i in range(len(a)):
    a[i] *= 0.25*T/s

def Ax(x):
    amp = a[0]
    for i in range(1, len(a)):
        amp += a[i]*((x-xc)/c)**i
    return amp

def deformation(x):
    return Ax(x)*np.sin(k*(x-xc))

# Init list of all coordinates
XY = []
XY_interpolated = []

# # Loop through the video frames
# while cap.isOpened():
#     ret, frame = cap.read()

#     if ret:
#         # run inference on a frame
#         results = model(frame)

#         # appends the results
#         for r in results:
#             if r.masks == None:
#                 break
#             mask = r.masks.xy
#             xys = mask[0]
#             XY.append(np.int32(xys))

#         # breal the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

#     else:
#         break

# cap.release()
# cv2.destroyAllWindows()

# Set the number of points the segmentation needs to be
desired_points_count = 512
screenX, screenY = 640, 640
resX, resY = 2**6, 2**6

# # for xy in XY:
# xy = XY[0]
# interpolated_f = np.zeros([desired_points_count, 2])

# min_init_x = np.min(xy[:,0])
# rotation_init_index = np.where(xy[:, 0] == min_init_x)[0]
# rotation_init_index = rotation_init_index.astype(np.int64)

# x_interp = np.roll(xy[:, 0], shift=-rotation_init_index, axis=0)
# y_interp = np.roll(xy[:, 1], shift=-rotation_init_index, axis=0)

# def remove_duplicates(array1, array2):
#     combined_array = list(zip(array1, array2))
#     seen = {}
#     unique1 = []
#     unique2 = []

#     for item in combined_array:
#         key = tuple(item)
#         if key not in seen:
#             seen[key] = True
#             unique1.append(item[0])
#             unique2.append(item[1])

#     return unique1, unique2

# x_interp, y_interp = remove_duplicates(x_interp, y_interp)
# # Append the starting x,y coordinates
# x_interp = np.r_[x_interp, x_interp[0]]
# y_interp = np.r_[y_interp, y_interp[0]]

# # Do the first interpolation, with a parametric one to get the whole edge
# tck, u = interpolate.splprep([x_interp, y_interp], s=len(x_interp)//2, per=True)

# # Evaluate the spline 
# xi, yi = interpolate.splev(np.linspace(0, 1, 2*desired_points_count), tck)
# min_x = np.min(xi)
# rotation_index = np.where(xi == min_x)[0]
# rotation_index = rotation_index.astype(np.int64)
# xi = np.roll(xi, shift=-rotation_index, axis=0)
# yi = np.roll(yi, shift=-rotation_index, axis=0)

# # Do the second interpolation, with half the points this time, to have a non parametric f
# # This f function can then be extended to the whole window, so we can compute the 
# # time derivatives for every pixel
# xi, yi = xi[15:desired_points_count-5], yi[15:desired_points_count-5]

# to_delete = []
# for i in range(len(xi)-1):
#     if xi[i] >= xi[i+1]:
#         to_delete.append(i)
# xi = np.delete(xi, to_delete)
# yi = np.delete(yi, to_delete)

# cs = interpolate.CubicSpline(xi, yi)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# # ax.plot(xi, yi, 'o', label="YOLO")
# ax.plot(np.linspace(min(xi), max(xi), 128), cs(np.linspace(min(xi), max(xi), 128)), label="interp")
# # ax.plot(np.linspace(0,screenX,100), cs(np.linspace(0,screenX,100)), 'r', label="extended")

# ax2 = ax.twiny()
# ax2.plot(np.linspace(0,resX,100), deformation(np.linspace(0,resX,100))+yi[0],'g', label="real")
# plt.show()

# Finalement, est ce que ce ne serait pas l'occasion de mettre un gros coup de SINDy?




x0, y0 = [], []
X_data = pd.read_csv("x.csv", header=None)
Y_data = pd.read_csv("y.csv", header=None)

real = []

for i in range(len(X_data.columns)):
    x0.append(X_data[i][20])
    y0.append(Y_data[i][20])
    real.append(-Ax(X_data[0][20])*omega*np.cos(k*(X_data[0][20] - xc)-omega*i/2))

# Filter sensor data
N = 2
fc1 = 25
fs = 1000
Wn = fc1/(fs/2)
btype = 'low'
b, a = signal.butter(N, Wn, btype=btype)
filtered_sensor = signal.filtfilt(b, a, y0 - np.mean(y0))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(y0)
ax.plot(real+np.mean(y0), 'g')
ax.plot(filtered_sensor+np.mean(y0),'r')
ax2 = ax.twiny()
ax2.plot( [y0[i] for i in range(0, len(y0), 10)], 'y')
plt.show()

# fig, ax = plt.subplots()
# for x,y in zip(x0,y0):
#     ax.plot(x,y,'ro')
#     plt.pause(0.01)