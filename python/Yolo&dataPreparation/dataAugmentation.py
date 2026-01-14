import cv2
import numpy as np
from ultralytics import YOLO
import os

# -------------------------------------------------------------------------------------------

# Takes full size pictures, detect the fish inside of these picture, and crop around the fish

# -------------------------------------------------------------------------------------------

def crop_and_resize_image_from_mask(input_image_path, file, target_size=(640, 640)):
    # Find the corresponding mask
    print(file)
    input_mask_path = input_image_path[:-(len(file)+1+6)]+"masks/"+file

    # Read the input image
    img = cv2.imread(input_image_path)
    mask = cv2.imread(input_mask_path)

    # Get the dimensions of the input image
    img_height, img_width = img.shape[:2]

    # Get the contours in the image and select the biggest object
    min_area = 1500
    masks_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    contours, _ =cv2.findContours(masks_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [contour for contour in contours if cv2.contourArea(contour) >= min_area]
    if valid_contours:
        contour_coordinates = valid_contours[0]
        # Calculate the center of the contour
        x_mean = np.int16(np.mean([elmt[0][0] for elmt in contour_coordinates]))
        y_mean = np.int16(np.mean([elmt[0][1] for elmt in contour_coordinates]))

        print(x_mean, y_mean)

        # Calculate the cropping coordinates
        crop_xmin = max(0, x_mean - target_size[0] // 2)
        crop_ymin = max(0, y_mean - target_size[1] // 2)
        crop_xmax = min(img_width, x_mean + target_size[0] // 2)
        crop_ymax = min(img_height, y_mean + target_size[1] // 2)

        # Check if the target region exceeds image boundaries
        if (crop_xmax - crop_xmin) < target_size[0]:
            if x_mean - (target_size[0] // 2) < 0:
                crop_xmax = crop_xmin + target_size[0]
            elif x_mean + (target_size[0] // 2) > img_width:
                crop_xmin = crop_xmax - target_size[0]

        if (crop_ymax - crop_ymin) < target_size[1]:
            if y_mean - (target_size[1] // 2) < 0:
                crop_ymax = crop_ymin + target_size[1]
            elif y_mean + (target_size[1] // 2) > img_height:
                crop_ymin = crop_ymax - target_size[1]

        # Crop and resize the image
        cropped_resized_img = img[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
        cv2.imwrite(input_image_path[:-4]+"-1.png", cropped_resized_img)


def crop_and_resize_image(input_image_path, target_size=(640, 640)):
    # Read the input image
    img = cv2.imread(input_image_path)

    # Get the dimensions of the input image
    img_height, img_width = img.shape[:2]

    # Get the bounding box of the detected shape
    results = model(img)
    r = results[0]
    boxes = r.boxes.xyxy.tolist()
    if boxes == []:
        return(file_path)

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

    # Save the result
    if target_size[0] == 640:
        cv2.imwrite(input_image_path[:-4]+"-1.png", cropped_resized_img)
    elif target_size[0] == 1200:
        cv2.imwrite(input_image_path[:-4]+'-4.png', cropped_resized_img)


input_folder = "C:/Users/blagn771/Desktop/simingImages/20240116_f68_crop1/images"
fromMask = True
for file in os.listdir(input_folder):
    file_path = os.path.join(input_folder, file)
    if file_path[-6] != '-':
        if fromMask:
            crop_and_resize_image_from_mask(file_path, file, (320, 320))
        else:
            model = YOLO("C:/Users/blagn771/Desktop/FishDataset/segment/train1280_32_291/weights/best.pt")
            crop_and_resize_image(file_path, (640, 640))
            # crop_and_resize_image(file_path, (1200, 1200))
