from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from numpy.polynomial import polynomial as pl
from scipy import interpolate

model = YOLO("yolov8n-seg-customNaca-mid.pt")
cap = cv2.VideoCapture("dataNaca_ref.mp4")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
   
size = (frame_width, frame_height)

# Init list of all the coordinates
XY = []
XY_interpolated = []

# loop through the video frames
while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        # run inference on a frame
        results = model(frame)

        # view results
        for r in results:
            if r.masks == None:
                break
            mask = r.masks.xy
            xys = mask[0]
            XY.append(np.int32(xys))
            cv2.polylines(frame, np.int32([xys]), True, (0, 255, 255), 2)

        cv2.imshow("img", frame)

        #break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    else:
        break

cap.release()
cv2.destroyAllWindows()

# XY = XY[20:]

# Set the number of points the segmentation needs to be
desired_points_count = 128          # power of 2
screenX, screenY = 640, 640
resX, resY = 2**6, 2**6

for xy in XY:
    interpolated_f = np.zeros([desired_points_count,2])

    min_init_x = np.min(xy[:,0])
    rotation_init_index = np.where(xy[:,0] == min_init_x)[0]
    rotation_init_index = rotation_init_index.astype(np.int64)

    x_interp = np.roll(xy[:,0], shift=-rotation_init_index, axis=0)
    y_interp = np.roll(xy[:,1], shift=-rotation_init_index, axis=0)

    # append the starting x,y coordinates
    x_interp = np.r_[x_interp, x_interp[0]]
    y_interp = np.r_[y_interp, y_interp[0]]

    def remove_duplicates(array1, array2):
        combined_array = list(zip(array1, array2))
        seen = {}
        unique1 = []
        unique2 = []

        for item in combined_array:
            key = tuple(item)
            if key not in seen:
                seen[key] = True
                unique1.append(item[0])
                unique2.append(item[1])

        return unique1, unique2
    
    x_interp, y_interp = remove_duplicates(x_interp, y_interp)
    x_interp = np.r_[x_interp, x_interp[0]]
    y_interp = np.r_[y_interp, y_interp[0]]
    # fit splines to x=f(u) and y=g(u), treating both as periodic. also note that s=0
    # is needed in order to force the spline fit to pass through all the input points.
    tck, u = interpolate.splprep([x_interp, y_interp], s=len(x_interp)//2, per=True)

    # evaluate the spline fits for 100 evenly spaced distance values
    xi, yi = interpolate.splev(np.linspace(0,1,2*desired_points_count), tck)

    min_x = np.min(xi)
    rotation_index = np.where(xi == min_x)[0]
    rotation_index = rotation_index.astype(np.int64)
    xi = np.roll(xi, shift=-rotation_index, axis=0)
    yi = np.roll(yi, shift=-rotation_index, axis=0)

    xi0, yi0 = remove_duplicates(xi, yi)
    xi0 = np.r_[xi0, xi0[0]]
    yi0 = np.r_[yi0, yi0[0]]
    tck, _ = interpolate.splprep([xi0, yi0], s=len(xi0) // 2, per=True)
    xi0, yi0 = interpolate.splev(np.linspace(0, 1, desired_points_count), tck)

    interpolated_f[:,0] = xi0*resX/screenX
    interpolated_f[:,1] = yi0*resY/screenY

    # interpolated_f = np.roll(interpolated_f, shift=-rotation_index, axis=0)

    # Add the interpolated frame to the list
    XY_interpolated.append(interpolated_f)

# Define the names for the output CSV files
x_file = 'x.csv'
y_file = 'y.csv'

# Open the CSV files for writing
with open(x_file, 'w', newline='') as file1, open(y_file, 'w', newline='') as file2:
    writer1 = csv.writer(file1)
    writer2 = csv.writer(file2)

    # Iterate through the main list
    for i in range(desired_points_count):
    # for i in range(len(XY[0])):
        # Extract the x and the y from each tuple
        columnX = [round(item[i][0],2) for item in XY_interpolated]
        columnY = [round(item[i][1],2) for item in XY_interpolated]
        # columnX = []
        # columnY = []
        # j = 0
        # for item in XY:
        #     if len(item)-1 < i:
        #         j = len(item)-1
        #     else:
        #         j = i
        #     columnX.append(item[j,0])
        #     columnY.append(item[j,1])

        # Write the columns to the respective CSV files
        writer1.writerow(columnX)
        writer2.writerow(columnY)

print(f"CSV files {x_file} and {y_file} have been created.")

# X_data = pd.read_csv("x.csv", header=None)
# Y_data = pd.read_csv("y.csv", header=None)

# fig, ax = plt.subplots()
# for i in range(len(X_data.columns)):
#     ax.clear()
#     ax.invert_yaxis()
#     x = X_data[i]
#     y = Y_data[i]
#     ax.plot(x,y,"-o")
#     # ax.plot(x[0],y[0], "ro")
#     # ax.plot(x[50],y[50], "ro")
#     plt.pause(0.1)