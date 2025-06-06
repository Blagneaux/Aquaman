import cv2
import os
import numpy as np

# -------------------------------------------------------------------------------------------

# Takes a cropped picture and its label, finds the corresponding zone in the original picture
# and creates a new label for this image.

# -------------------------------------------------------------------------------------------

def generalizeLabel(cropped_img_path, cropped_label_path, img_path):

    # Read the images
    cropped_img = cv2.imread(cropped_img_path)
    img = cv2.imread(img_path)

    cropped_h, cropped_w = cropped_img.shape[:2]
    img_h, img_w = img.shape[:2]

    original_label_list = []

    # Open the label file and read it
    with open(cropped_label_path, 'r') as file:
        label_text = file.read()

    # Split the text into a list
    label_list_str = label_text.split()
    label_list = [float(x) for x in label_list_str[1:]]

    # Unnormalize the label
    for i in range(len(label_list)):
        if i%2 == 0:
            label_list[i] = label_list[i] * cropped_h
        else:
            label_list[i] = label_list[i] * cropped_w

    # If one of the images is the cropped version of the other
    if img_h > cropped_h:

        # Perform template matching
        result = cv2.matchTemplate(img, cropped_img, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        top_left_coordinates = max_loc

        x_top, y_top = top_left_coordinates

        # Create new label for the original image
        print("resize")
        for i in range(len(label_list)):
            if i%2 == 0: # WARNING: index 0 is now the first x instead of the class
                original_label_list.append((label_list[i] + x_top) / img_w)
            else:
                original_label_list.append((label_list[i] + y_top) / img_h)

    # If one of the images is the translated version of the other
    elif img_h == cropped_h:

        # Initialize the SIFT detector
        sift = cv2.SIFT_create()

        # Find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(cropped_img, None)
        kp2, des2 = sift.detectAndCompute(img, None)

        # Use the Brute-Force Matcher to find the best matches
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        # Extract the coordinates of the matching keypoints in both images
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Find the homography matrix
        M, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

        # Get the corners of the template image
        h, w = img.shape[:2]
        template_corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)

        # Transform the corners to the main image
        main_corners = cv2.perspectiveTransform(template_corners, M)

        top_left_coordinates = main_corners[0]
        print(top_left_coordinates[0], main_corners[2][0])

        x_top, y_top = np.int32(top_left_coordinates[0])

        print("translate")
        for i in range(len(label_list)):
            if i%2 == 0:
                new_x = label_list[i] - x_top
                new_y = label_list[i+1] - y_top
                if new_x >= 0 and new_x < img_h and new_y >= 0 and new_y < img_w:
                    original_label_list.append(new_x / img_h)
            else:
                new_x = label_list[i-1] - x_top
                new_y = label_list[i] - y_top
                if new_x >= 0 and new_x < img_h and new_y >= 0 and new_y < img_w:
                    original_label_list.append(new_y / img_w)


    # rectangeTest(label_list, cropped_img)
    # rectangeTest(original_label_list, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    strYolo = "0"
    for elmt in original_label_list:
        strYolo += " "+str(elmt)
    with open(img_path[:-3] + 'txt', 'w') as writeFile:
        writeFile.write(strYolo)


# For debug porpose
def rectangeTest(list, img):
    points = []
    for i in range(0, len(list) - 1, 2):
        points.append((int(list[i]), int(list[i+1])))
    print(points)
    cv2.rectangle(img, points[0], points[2], (0, 0, 255))
    cv2.imshow('test', img)


# # Test code
# original_path = "C:/Users/blagn771/Desktop/FishDataset/Fish1/images/Fish1_frame-0001.png"
# cropped_path = "C:/Users/blagn771/Desktop/FishDataset/Fish1/images/Fish1_frame-0001-1.png"
# cropped_label_path = "C:/Users/blagn771/Desktop/FishDataset/Fish1/images/Fish1_frame-0001-1.txt"
# generalizeLabel(cropped_path, cropped_label_path, original_path)

input_folder = "C:/Users/blagn771/Desktop/FishDataset/Fish3/images"
label_folder = "C:/Users/blagn771/Desktop/FishDataset/Fish3/labelsByHand"
for file in os.listdir(input_folder):
    file_path = os.path.join(input_folder, file)
    if file_path[-6] != '-':
        label_path = os.path.join(label_folder, file)
        original_path = file_path
        cropped_path = file_path[:-4]+'-1.png'
        cropped_label_path = label_path[:-4]+'-1.txt'
        generalizeLabel(cropped_path, cropped_label_path, original_path)
    if file_path[-5] == '4' and file_path[-6] == '-':
        print(file_path)
        label_path = os.path.join(label_folder, file)
        original_path = file_path
        cropped_path = file_path[:-5]+'1.png'
        cropped_label_path = label_path[:-6]+'-1.txt'
        generalizeLabel(cropped_path, cropped_label_path, original_path)
    # if file_path[-5] == '3' and file_path[-6] == '-':
    #     print(file_path)
    #     label_path = os.path.join(label_folder, file)
    #     original_path = file_path
    #     cropped_path = file_path[:-5]+'1.png'
    #     cropped_label_path = label_path[:-6]+'-1.txt'
    #     generalizeLabel(cropped_path, cropped_label_path, original_path)
    # if file_path[-5] == '2' and file_path[-6] == '-':
    #     print(file_path)
    #     label_path = os.path.join(label_folder, file)
    #     original_path = file_path
    #     cropped_path = file_path[:-5]+'1.png'
    #     cropped_label_path = label_path[:-6]+'-1.txt'
    #     generalizeLabel(cropped_path, cropped_label_path, original_path)