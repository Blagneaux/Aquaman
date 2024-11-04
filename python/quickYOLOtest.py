from ultralytics import YOLO
import cv2
import numpy as np

# ---------------------------------------------------------

# Takes a video, apply a YOLO model, and display the result

# ---------------------------------------------------------

model2 = YOLO("C:/Users/blagn771/Documents/Aquaman/Aquaman/runs/segment/train640_32_500_manuel/weights/best.pt")
model = YOLO("C:/Users/blagn771/Documents/Aquaman/Aquaman/runs/segment/bestProjet1a.pt")
cap = cv2.VideoCapture("C:/Users/blagn771/Downloads/fish13_crop1.mp4")
# cap = cv2.VideoCapture("C:/Users/blagn771/Desktop/zoomed.mp4")
# cap = cv2.VideoCapture("E:/data_Bastien/datasetFish/video (2160p).mp4")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

size = (frame_width, frame_height)
print(size)

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

# Init list of all the coordinates
XY = []
XY_interpolated = []
count = 1

# loop through the video frames
while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
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