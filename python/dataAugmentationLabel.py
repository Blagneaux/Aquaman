import cv2
import os

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

    print(label_list)

    # Perform template matching
    result = cv2.matchTemplate(img, cropped_img, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    top_left_coordinates = max_loc

    x_top, y_top = top_left_coordinates

    # Create new label for the original image
    original_label_list = []
    for i in range(len(label_list)):
        if i%2 == 0: # WARNING: index 0 is now the first x instead of the class
            original_label_list.append((label_list[i] + x_top) / img_w)
        else:
            original_label_list.append((label_list[i] + y_top) / img_h)

    print(original_label_list)

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

input_folder = "C:/Users/blagn771/Desktop/FishDataset/Fish1/images"
for file in os.listdir(input_folder):
    file_path = os.path.join(input_folder, file)
    if file_path[-6] != '-':
        original_path = file_path
        cropped_path = file_path[:-4]+'-1.png'
        cropped_label_path = cropped_path[:-3]+'txt'
        generalizeLabel(cropped_path, cropped_label_path, original_path)