from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from numpy.polynomial import polynomial as pl
from scipy import interpolate, signal, fft
import pysindy as ps
import os

haato = '37'
haachama = '8'

# Best model at the moment: train640_32_500_manuel
model2 = YOLO("C:/Users/blagn771/Documents/Aquaman/Aquaman/runs/segment/train640_32_500_manuel/weights/best.pt")
model = YOLO("C:/Users/blagn771/Documents/Aquaman/Aquaman/runs/segment/bestProjet1a.pt")
# cap = cv2.VideoCapture("C:/Users/blagn771/Desktop/testDetection.mp4")
# cap = cv2.VideoCapture("E:/crop_nadia/"+haato+"/"+haachama+"/"+haachama+".mp4")
cap = cv2.VideoCapture("E:/data_HAACHAMA/FishDataset/videoNadia/T3_Fish3_C2_270923 - Trim2.mp4")

def calculate_angle(pt1, pt2, pt3):

    vector1 = np.array(pt1) - np.array(pt2)
    vector2 = np.array(pt3) - np.array(pt2)

    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    cosine_angle = dot_product / (magnitude1 * magnitude2)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))

    return angle

def find_most_acute_vertex(points):
    num_points = len(points)

    min_angle = float('inf')
    most_acute_vertex = None

    for i in range(num_points):
        pt1 = points[i - 5]
        pt2 = points[i]
        pt3 = points[(i + 5) % num_points]

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

def crop_and_resize_image(input_image, target_size=(640, 640)):
    # Read the input image
    img = input_image

    # Get the dimensions of the input image
    img_height, img_width = img.shape[:2]

    # Get the bounding box of the detected shape
    results = model2(img)
    r = results[0]
    boxes = r.boxes.xyxy.tolist()
    if boxes == []:
        return([[["Nothing to detect"]]])

    xmin, ymin, xmax, ymax = boxes[0]
    xmin = int(xmin)
    xmax = int(xmax)
    ymin = int(ymin)
    ymax = int(ymax)

    # Calculate the center of the bounding box
    center_x = (xmin + xmax) // 2
    center_y = (ymin + ymax) // 2

    # Calculate the cropping coordinates
    crop_xmin = max(0, center_x - target_size[0] // 2)
    crop_ymin = max(0, center_y - target_size[1] // 2)
    crop_xmax = min(img_width, center_x + target_size[0] // 2)
    crop_ymax = min(img_height, center_y + target_size[1] // 2)

    # Check if the target region exceeds image boundaries
    if (crop_xmax - crop_xmin) < target_size[0]:
        if center_x - (target_size[0] // 2) < 0:
            crop_xmax = crop_xmin + target_size[0]
        elif center_x + (target_size[0] // 2) > img_width:
            crop_xmin = crop_xmax - target_size[0]

    if (crop_ymax - crop_ymin) < target_size[1]:
        if center_y - (target_size[1] // 2) < 0:
            crop_ymax = crop_ymin + target_size[1]
        elif center_y + (target_size[1] // 2) > img_height:
            crop_ymin = crop_ymax - target_size[1]

    # Crop and resize the image
    cropped_resized_img = img[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
    return cropped_resized_img

def generalizeLabel(cropped_img, cropped_label, img):

    cropped_h, cropped_w = cropped_img.shape[:2]
    img_h, img_w = img.shape[:2]

    original_label_list = []
    label_list = cropped_label

    # If one of the images is the cropped version of the other
    if img_h > cropped_h:

        # Perform template matching
        result = cv2.matchTemplate(img, cropped_img, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        top_left_coordinates = max_loc

        x_top, y_top = top_left_coordinates

        # Create new label for the original image
        for i in label_list:
            original_label_list.append([i[0] + x_top, i[1] + y_top])

        return original_label_list

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
        if not ret:
            break
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        # frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # Apply contrast adjustment
        alpha = 1  # Contrast control (1.0 for no change)
        beta = 0     # Brightness control (0 for no change)

        if ret:
            # run inference on a frame
            frame_cropped = crop_and_resize_image(frame, (640,640))
            if frame_cropped[0][0][0] != "Nothing to detect":
                frame_cropped_contrasted = cv2.convertScaleAbs(frame_cropped, alpha=alpha, beta=beta)
                results = model(frame_cropped_contrasted)

                # view results
                for r in results:
                    if r.masks == None:
                        break
                    mask = r.masks.xy
                    xys = mask[0]
                    uncropped_xys = generalizeLabel(frame_cropped, xys, frame)
                    if uncropped_xys is not None:
                        XY.append(np.int32(uncropped_xys))
                        cv2.polylines(frame, np.int32([uncropped_xys]), True, (0, 0, 255), 2)

            cv2.imshow("img", frame)

            #break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

    # # Test to loop through the labels to see how it works
    # labels_folder = "C:/Users/blagn771/Desktop/FishDataset/Fish3/labelsByHandNormalSize"
    # img_folder = "C:/Users/blagn771/Desktop/FishDataset/Fish3/images"
    # for file in os.listdir(labels_folder):
    #     if file.endswith('.txt'):
    #         file_path = os.path.join(labels_folder, file)
    #         crop_path = os.path.join(img_folder, file[:-3]+'png')
    #         uncrop_path = os.path.join(img_folder, file[:-6]+'.png')
    #         uncrop = cv2.imread(uncrop_path)
    #         crop = cv2.imread(crop_path)
    #         result = cv2.matchTemplate(uncrop, crop, cv2.TM_CCOEFF_NORMED)
    #         _, _, _, top_left = cv2.minMaxLoc(result)
    #         df_label = pd.read_csv(file_path, sep=' ', header=None)
    #         xys = []
    #         for i in range(1, len(df_label.columns), 2):
    #             x = int(df_label[i][0] * 640 + top_left[0])
    #             y = int(df_label[i+1][0] * 640 + top_left[1])
    #             xys.append([x,y])
    #         XY.append(np.int32(xys))

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
    # x_file = 'E:/crop_nadia/'+haato+'/rawYolo'+haachama+'_x.csv'
    # y_file = 'E:/crop_nadia/'+haato+'/rawYolo'+haachama+'_y.csv'
    # y_dot_file = 'E:/crop_nadia/'+haato+'/rawYolo'+haachama+'_y_dot.csv'
    x_file = "x.csv"
    y_file = "y.csv"
    y_dot_file = "y_dot.csv"

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
    X_data = pd.read_csv(x_file, header=None)
    Y_data = pd.read_csv(y_file, header=None)
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
    
    print("Desired number of points for the edge:", desired_points_count)
    print("Screen size:", screenX, screenY)
    print("Resolution of the simulation:", resX, resY)

    print("Uncomment ligne 170 to 172 for vertical and RGB videos")


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
        ax.plot(x[50],y[50], "go")
        plt.pause(1)

def fish_scan(h = 0.004, dt = 0.69, nu = 0.00095, f_ac = 100):
    # h, dt and nu are the non-dimensioning parameters from Lylipad
    # h in m/grid
    # dt is the time step
    # nu is the dimensionless cinematic viscosity
    # f_ac is the fps of the camera

    fig = plt.figure()
    ax = fig.add_subplot(1,2,1)
    X_data = pd.read_csv("x.csv", header=None)
    N = len(X_data.columns)
    Y_data = pd.read_csv("y.csv", header=None)
    T = np.linspace(0, N/f_ac,N)
    tail_pos, head_pos = [], []

    for i in range(N):
        tail_pos.append(Y_data[i][len(Y_data[i]) // 2])
        head_pos.append(Y_data[i][0])

    head_fft = fft.fft(np.array(head_pos)-np.mean(head_pos))
    xf = fft.fftfreq(N, 1/f_ac)[:N//2]
    head_fft_plot = 2.0/N * np.abs(head_fft[0:N//2])
    max_head_fft = np.max(head_fft_plot)
    fc_indice = np.where( head_fft_plot[0:N//2] > max_head_fft / 10)

    fc = xf[np.max(fc_indice)]

    # Créer un filtre passe-bas
    ordre = 2  # Ordre du filtre
    nyquist = 0.5 * f_ac
    frequence_normale = fc / nyquist
    b, a = signal.butter(ordre, frequence_normale, btype='low', analog=False)

    dist = 0
    fish_length = 0,
    dist_time = 0
    tail_freq = 0

    dist = abs(X_data[N-1][0] - X_data[0][0])*h
    fish_length = np.mean([np.sqrt((X_data[i][0] - X_data[i][len(X_data[0]) // 2])**2 + (Y_data[i][0] - Y_data[i][len(X_data[0]) // 2])**2)*h for i in X_data.columns])
    dist_time = N/f_ac

    tail_filt = signal.filtfilt(b, a, tail_pos)
    head_filt = signal.filtfilt(b, a, head_pos)

    ax = plt.plot(T, tail_pos - tail_filt)
    ax = plt.plot(T, head_pos-np.mean(head_pos))
    # ax = plt.plot(T, tail_filt)

    ax2 = fig.add_subplot(1,2,2)
    tail_fft = fft.fft(np.array(tail_pos - tail_filt)-np.mean(tail_pos - tail_filt))
    ax2 = plt.plot(xf, 2.0/N * np.abs(tail_fft[0:N//2]))

    tail_fft_plot = 2.0/N * np.abs(tail_fft[0:N//2])
    max_tail_fft = np.max(tail_fft_plot)
    fc_indice = np.where( tail_fft_plot[0:N//2] == np.max(tail_fft_plot))

    tail_freq = xf[fc_indice]

    print("distance of the trajectory in m: ", np.round(dist, 3))
    print("length of the fish in m: ", np.round(fish_length, 3))
    print("duration of the trajectory in s: ", dist_time)
    print("tailbeat frequency in Hz: ", np.round(tail_freq[0], 3))
    print("distance of the trajectory in fish length: ", np.round(dist/fish_length, 3))
    print("duration of the trajectory in tailbeat: ", dist_time*tail_freq[0])

    print("Strouhal number: ", tail_freq[0]*fish_length/(dist/dist_time))
    print("Reynolds number: ", fish_length*fish_length/(dist/dist_time)*1000000)

    # plt.show()

    return dist, fish_length, dist_time


if __name__ == "__main__":
    predict(model, cap)
    # debug()
    # fish_scan()