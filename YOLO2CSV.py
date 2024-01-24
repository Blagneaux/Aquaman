from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from numpy.polynomial import polynomial as pl
from scipy import interpolate, signal
import pysindy as ps
import os

model = YOLO("C:/Users/blagn771/Desktop/FishDataset/segment/train1280_32_291/weights/best.pt")
# cap = cv2.VideoCapture("C:/Users/blagn771/Desktop/testDetection.mp4")

# model = YOLO("C:/Users/blagn771/Desktop/FishDataset/segment/train1280_32_291/weights/best.pt")
cap = cv2.VideoCapture("C:/Users/blagn771/Desktop/FishDataset/videoNadia/T3_Fish3_C2_270923 - Trim2.mp4")

def calculate_angle(pt1, pt2, pt3):

    vector1 = np.array(pt1) - np.array(pt2)
    vector2 = np.array(pt3) - np.array(pt2)

    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    cosine_angle = dot_product / (magnitude1 * magnitude2)
    angle = np.degrees(np.arccos(cosine_angle))

    return angle

def find_most_acute_vertex(points):
    num_points = len(points)

    min_angle = float('inf')
    most_acute_vertex = None

    for i in range(num_points):
        pt1 = points[i - 1]
        pt2 = points[i]
        pt3 = points[(i + 1) % num_points]

        angle = calculate_angle(pt1, pt2, pt3)

        if angle < min_angle:
            min_angle = angle
            most_acute_vertex = i

    return most_acute_vertex

def calculate_distance(pt1, pt2):
    return np.linalg.norm(np.array(pt2) - np.array(pt1))

def calculate_shape_perimeter(points):
    num_points = len(points)
    perimeter = 0.0

    for i in range(num_points):
        pt1 = points[i]
        pt2 = points[(i + 1) % num_points]
        perimeter += calculate_distance(pt1, pt2)

    return perimeter

def find_opposite_end_point(points, most_acute_index, perimeter):
    num_points = len(points)
    cumulative_distance = 0.0

    for i in range(num_points):
        pt1 = points[(i + most_acute_index) % num_points]
        pt2 = points[(i + 1 + most_acute_index) % num_points]

        cumulative_distance += calculate_distance(pt1, pt2)

        if cumulative_distance >= perimeter / 2:
            opposite_end_index = (i + 1 + most_acute_index) % num_points
            break

    return opposite_end_index

def predict(model=model, cap=cap):

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    
    size = (frame_width, frame_height)

    # Init list of all the coordinates
    XY = []
    XY_interpolated = []

    # # loop through the video frames
    # while cap.isOpened():
    #     ret, frame = cap.read()

    #     # frame = cv2.flip(frame, 1)

    #     if ret:
    #         # run inference on a frame
    #         results = model(frame)

    #         # view results
    #         for r in results:
    #             if r.masks == None:
    #                 break
    #             mask = r.masks.xy
    #             xys = mask[0]
    #             XY.append(np.int32(xys))
    #             cv2.polylines(frame, np.int32([xys]), True, (0, 255, 255), 2)

    #         cv2.imshow("img", frame)

    #         #break the loop if 'q' is pressed
    #         if cv2.waitKey(1) & 0xFF == ord("q"):
    #             break
        
    #     else:
    #         break

    # cap.release()
    # cv2.destroyAllWindows()

    # Test to loop through the labels to see how it works
    labels_folder = "C:/Users/blagn771/Desktop/fish3-fully-labelled/labels"
    img_folder = "C:/Users/blagn771/Desktop/FishDataset/Fish3/images"
    for file in os.listdir(labels_folder):
        file_path = os.path.join(labels_folder, file)
        crop_path = os.path.join(img_folder, file[:-3]+'png')
        uncrop_path = os.path.join(img_folder, file[:-6]+'.png')
        uncrop = cv2.imread(uncrop_path)
        crop = cv2.imread(crop_path)
        result = cv2.matchTemplate(uncrop, crop, cv2.TM_CCOEFF_NORMED)
        _, _, _, top_left = cv2.minMaxLoc(result)
        df_label = pd.read_csv(file_path, sep=' ', header=None)
        xys = []
        for i in range(1, len(df_label.columns), 2):
            x = int(df_label[i][0] * 640 + top_left[0])
            y = int(df_label[i+1][0] * 640 + top_left[1])
            xys.append([x,y])
        XY.append(np.int32(xys))

    # Set the number of points the segmentation needs to be
    desired_points_count = 256          # power of 2
    screenX, screenY = 2000, 1200
    resX, resY = 2**8, 2**7

    count = 0
    # del XY[39]
    # del XY[393]
    print("number of points in the mask ", len(XY[0]))
    for xy in XY:
        interpolated_f = np.zeros([desired_points_count,2])

        # min_init_x = np.min(xy[:,0])
        # rotation_init_index = np.where(xy[:,0] == min_init_x)[0]
        # rotation_init_index = rotation_init_index.astype(np.int64)

        perimeter = calculate_shape_perimeter(xy)
        tail = find_most_acute_vertex(xy)
        head = find_opposite_end_point(xy, tail, perimeter)
        rotation_init_index = head

        x_interp = np.roll(xy[:,0], shift=-rotation_init_index, axis=0)
        y_interp = np.roll(xy[:,1], shift=-rotation_init_index, axis=0)
        # x_interp = xy[:,0]
        # y_interp = xy[:,1]

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
        tck, u = interpolate.splprep([x_interp, y_interp], s=len(x_interp)//4, per=True, k=1)

        # evaluate the spline fits for 100 evenly spaced distance values
        xi, yi = interpolate.splev(np.linspace(0,1,2*desired_points_count), tck)

        # # Probably useless ----------------------------------------------------------
        # min_x = np.min(xi)
        # rotation_index = np.where(xi == min_x)[0]
        # rotation_index = rotation_index.astype(np.int64)
        # xi = np.roll(xi, shift=-rotation_index, axis=0)
        # yi = np.roll(yi, shift=-rotation_index, axis=0)

        xi0, yi0 = remove_duplicates(xi, yi)
        xi0 = np.r_[xi0, xi0[0]]
        yi0 = np.r_[yi0, yi0[0]]
        # xi0, yi0 = np.array(xi), np.array(yi)
        tck, _ = interpolate.splprep([xi0, yi0], s=len(xi0) // 4, per=True)
        xi0, yi0 = interpolate.splev(np.linspace(0, 1, desired_points_count), tck)
        print(count)
        count += 1
        # xi0, yi0 = xi0[:-1], yi0[:-1] 

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
    predict(model, cap)
    # debug()