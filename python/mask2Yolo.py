import cv2
import os
import numpy as np

def mask2txt(file_path, new_file_path):

    min_area = 2000
    image = cv2.imread(file_path)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    contours, _ = cv2.findContours(img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a black background image
    result_img = np.zeros_like(img_gray)

    # Filter contours by area
    valid_contours = [contour for contour in contours if cv2.contourArea(contour) >= min_area]

    if valid_contours:
        # Draw the largest valid contour on the result image
        cv2.drawContours(result_img, [max(valid_contours, key=cv2.contourArea)], -1, 255, thickness=cv2.FILLED)

        # Find the coordinates of the contour
        contour_coordinates = valid_contours[0].reshape(-1, 2)  # Flatten the array

        strYolo = "0"
        for point in contour_coordinates:
            strYolo += ' '+str(point[0])
            strYolo += ' '+str(point[1])
        with open(new_file_path[:-3]+'txt', 'w') as writeFile:
                writeFile.write(strYolo)


input_folder = "C:/Users/blagn771/Desktop/Fish1C1/masks"
output_folder = "C:/Users/blagn771/Desktop/Fish1C1/labels"
for file in os.listdir(input_folder):
    file_path = os.path.join(input_folder, file)
    new_file_path = os.path.join(output_folder, file)
    mask2txt(file_path, new_file_path)
