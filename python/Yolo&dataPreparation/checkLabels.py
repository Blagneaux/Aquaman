import pandas as pd
import numpy as np
import cv2
import os

# --------------------------------------------------------------------------------------
# Hover the clean label to the openCV label to highlight the difference
# --------------------------------------------------------------------------------------

# file_path = "C:/Users/blagn771/Desktop/fish3-fully-labelled/labels/Fish3_frame-0113-1.txt"
# img_path = "C:/Users/blagn771/Desktop/FishDataset/Fish3/masks/Fish3_frame-0113.png"
# crop_path = "C:/Users/blagn771/Desktop/FishDataset/Fish3/images/Fish3_frame-0113-1.png"
# uncrop_path = "C:/Users/blagn771/Desktop/FishDataset/Fish3/images/Fish3_frame-0113.png"

# df = pd.read_csv(file_path, sep=' ', header=None)
# points = []
# uncrop = cv2.imread(uncrop_path)
# crop = cv2.imread(crop_path)
# result = cv2.matchTemplate(uncrop, crop, cv2.TM_CCOEFF_NORMED)
# _, _, _, top_left = cv2.minMaxLoc(result)
# for i in range(1, len(df.columns), 2):
#         x = int(df[i][0] * 640 + top_left[0])
#         y = int(df[i+1][0] * 640 + top_left[1])
#         points.append([x,y])
# img = cv2.imread(img_path)
# cv2.polylines(img, [np.array([points], dtype=np.int32)], isClosed=True, color=(0, 0, 255), thickness=2)
# cv2.imshow('diff of label', img)
# cv2.imwrite("C:/Users/blagn771/Desktop/diffLabels.png", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# --------------------------------------------------------------------------------------

def checkLabels(img_path, label_path):

    df = pd.read_csv(label_path, sep=" ", header=None)
    points = []
    img = cv2.imread(img_path)
    img_h, img_w = img.shape[:2]
    for i in range(1, len(df.columns), 2):
        x = int(df[i][0] * img_w)
        y = int(df[i+1][0] * img_h)
        points.append([x,y])

    # Convert points to a NumPy array
    points_array = np.array([points], dtype=np.int32)

    # Draw the polyline
    perimeter = calculate_shape_perimeter(points)
    tail = find_most_acute_vertex(points)
    # head = find_opposite_end_point(points, tail, perimeter)
    cv2.polylines(img, [points_array], isClosed=True, color=(0, 0, 255), thickness=2)
    # cv2.circle(img, points_array[0][tail], radius=5, color=(0, 255, 0), thickness=-1)

    cv2.imshow('img', img)
    # Add a delay (100 milliseconds in this example)
    print(img_path)
    cv2.waitKey(10)

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


img_fodler = "C:/Users/blagn771/Desktop/FishDataset/Fish1/images"
labels_folder = "C:/Users/blagn771/Desktop/FishDataset/Fish1/labelsByHand"

# # ---------------------------------------------------------
# # If the files have been renamded by the labeling software:
# # ---------------------------------------------------------
# # Iterate through all files in the folder
# for filename in os.listdir(labels_folder):
#     # Construct the current file's full path
#     current_path_label = os.path.join(labels_folder, filename)

#     # Remove the first 9 characters from the filename
#     new_filename = filename[9:]

#     # Construct the new file's full path
#     new_path_label = os.path.join(labels_folder, new_filename)

#     # Rename the file
#     os.rename(current_path_label, new_path_label)

for file in os.listdir(img_fodler):
    file_path = os.path.join(img_fodler, file)
    if file_path[-6] != '-':
    # if file_path[-5] == '4' and file_path[-6] == '-':
        label_path = os.path.join(labels_folder, file[:-3]+"txt")
        checkLabels(file_path, label_path)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cv2.destroyAllWindows()