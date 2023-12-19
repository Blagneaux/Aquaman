from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from numpy.polynomial import polynomial as pl
from scipy import interpolate, signal
import pysindy as ps

model = YOLO("yolov8n-seg-customNaca-mid.pt")
cap = cv2.VideoCapture("dataNaca_ref.mp4")

def predict(model=model, cap=cap):

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

    # Set the number of points the segmentation needs to be
    desired_points_count = 512          # power of 2
    screenX, screenY = 640, 640
    resX, resY = 2**6, 2**6

    count = 0
    del XY[290]
    del XY[393]
    for xy in XY:
        interpolated_f = np.zeros([desired_points_count,2])

        min_init_x = np.min(xy[:,0])
        rotation_init_index = np.where(xy[:,0] == min_init_x)[0]
        rotation_init_index = rotation_init_index.astype(np.int64)

        x_interp = np.roll(xy[:,0], shift=-rotation_init_index, axis=0)
        y_interp = np.roll(xy[:,1], shift=-rotation_init_index, axis=0)

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
        # xi0, yi0 = np.array(xi), np.array(yi)
        tck, _ = interpolate.splprep([xi0, yi0], s=len(xi0) // 2, per=True)
        xi0, yi0 = interpolate.splev(np.linspace(0, 1, desired_points_count), tck)
        print(count)
        count += 1

        interpolated_f[:,0] = xi0*resX/screenX
        interpolated_f[:,1] = yi0*resY/screenY

        # Add the interpolated frame to the list
        XY_interpolated.append(interpolated_f)

    # Define the names for the output CSV files
    x_file = 'x.csv'
    y_file = 'y.csv'
    y_dot_file = 'y_dot.csv'

    # Open the CSV files for writing
    with open(x_file, 'w', newline='') as file1, open(y_file, 'w', newline='') as file2:
        writer1 = csv.writer(file1)
        writer2 = csv.writer(file2)

        # Iterate through the main list
        for i in range(desired_points_count):
            # Extract the x and the y from each tuple
            columnX = [round(item[i][0],2) for item in XY_interpolated]
            columnY = [round(item[i][1],2) for item in XY_interpolated]

            # Write the columns to the respective CSV files
            writer1.writerow(columnX)
            writer2.writerow(columnY)

    print(f"CSV files {x_file} and {y_file} have been created.")

    # Create a filtered Y to smooth the evolution in time and have a better derivative
    X_data = pd.read_csv("x.csv", header=None)
    Y_data = pd.read_csv("y.csv", header=None)
    y_dot = pd.DataFrame(index=range(len(Y_data[0])), columns=range(len(Y_data.columns)))

    sfd = ps.SmoothedFiniteDifference(smoother_kws={'window_length': 11}) # pySINDy smooth finite differenciation
    T0 = np.array([0.5 + j/2 for j in range(len(Y_data.columns))]) # time vector for the derivation
    for i in range(len(Y_data[0])):
        y0 = [Y_data[j][i] for j in range(len(Y_data.columns))]
        y_dot0 = sfd._differentiate(y0, T0)
        for j in range(len(Y_data.columns)):
            y_dot[j][i] = y_dot0[j]

    # Open the CSV files for writing
    with open(y_dot_file, 'w', newline='') as file:
        writer = csv.writer(file)

        # Iterate through the main list
        for i in range(desired_points_count):
            columnFilteredY = []
            for j in range(len(Y_data.columns)):
                # Extract the filtered y for each tuple
                columnFilteredY.append(y_dot[j][i])

            # Write the columns to the respective CSV files
            writer.writerow(columnFilteredY)

    print(f"CSV files {y_dot_file} has been created.")


def debug():

    fig, ax = plt.subplots()
    X_data = pd.read_csv("x.csv", header=None)
    Y_data = pd.read_csv("y.csv", header=None)

    for i in range(len(X_data.columns)):
        ax.clear()
        ax.invert_yaxis()
        x = X_data[i]
        y = Y_data[i]
        ax.plot(x,y,"-o")
        ax.plot(x[0],y[0], "ro")
        ax.plot(x[50],y[50], "ro")
        plt.pause(0.1)


if __name__ == "__main__":
    model = YOLO("yolov8n-seg-customNaca-big.pt")
    cap = cv2.VideoCapture("dataNaca_ref.mp4")
    predict(model, cap)
    # debug()