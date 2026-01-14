from ultralytics import YOLO
import cv2
import numpy as np
from scipy import interpolate

# ---------------------------------------------------------

# Takes a video, apply a YOLO model, and display the result

# ---------------------------------------------------------

model2 = YOLO("C:/Users/blagn771/Documents/Aquaman/Aquaman/runs/segment/train640_32_500_manuel/weights/best.pt")
model = YOLO("C:/Users/blagn771/Documents/Aquaman/Aquaman/runs/segment/bestProjet1a.pt")
# model = YOLO("C:/Users/blagn771/Desktop/best_combined.pt")
# cap = cv2.VideoCapture("C:/Users/blagn771/Downloads/fish13_crop1.mp4")
cap = cv2.VideoCapture("D:/crop_nadia/28/7/7.mp4")
# cap = cv2.VideoCapture("E:/data_Bastien/datasetFish/video (2160p).mp4")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

size = (frame_width, frame_height)
print(size)

# Target screen size for display
OUTPUT_WIDTH = 1280
OUTPUT_HEIGHT = 720

# Set interpolation parameters
DESIRED_POINTS_COUNT = 256  # Should be power of 2
SCREEN_X, SCREEN_Y = 2000, 1200
RES_X, RES_Y = 2**8, 2**7

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

def interpolate_segmentation(xy, frame_shape):
    """
    Interpolates a given segmentation mask using spline fitting.
    """
    if not isinstance(xy, np.ndarray):
        xy = np.array(xy, dtype=np.float32)  # Convert to NumPy array

    if len(xy) < 4:  # Avoid interpolation errors with too few points
        return xy

    frame_h, frame_w = frame_shape[:2]  # Get the original frame dimensions

    interpolated_f = np.zeros([DESIRED_POINTS_COUNT, 2])

    # Sort points and find rotation index
    perimeter = calculate_shape_perimeter(xy)
    tail = find_most_acute_vertex(xy)
    head = find_opposite_end_point(xy, tail, perimeter)
    rotation_init_index = head

    # Ensure array is properly rolled
    x_interp = np.roll(xy[:, 0], shift=-rotation_init_index, axis=0)
    y_interp = np.roll(xy[:, 1], shift=-rotation_init_index, axis=0)

    def remove_duplicates(arr1, arr2):
        combined = list(zip(arr1, arr2))
        seen = set()
        unique_x, unique_y = [], []

        for item in combined:
            if item not in seen:
                seen.add(item)
                unique_x.append(item[0])
                unique_y.append(item[1])

        return unique_x, unique_y

    x_interp, y_interp = remove_duplicates(x_interp, y_interp)
    x_interp = np.r_[x_interp, x_interp[0]]
    y_interp = np.r_[y_interp, y_interp[0]]

    # Fit splines
    try:
        tck, _ = interpolate.splprep([x_interp, y_interp], s=len(x_interp) // 4, per=True, k=1)
        xi, yi = interpolate.splev(np.linspace(0, 1, 2 * DESIRED_POINTS_COUNT), tck)

        xi0, yi0 = remove_duplicates(xi, yi)
        xi0 = np.r_[xi0, xi0[0]]
        yi0 = np.r_[yi0, yi0[0]]

        tck, _ = interpolate.splprep([xi0, yi0], s=len(xi0) // 4, per=True)
        xi0, yi0 = interpolate.splev(np.linspace(0, 1, DESIRED_POINTS_COUNT), tck)

        # Fix scaling issue by mapping back to the original frame size
        interpolated_f[:, 0] = xi0 * (frame_w / 2000)  # Scale X coordinates
        interpolated_f[:, 1] = yi0 * (frame_h / 1200)  # Scale Y coordinates

    except Exception as e:
        print(f"Interpolation error: {e}")
        return xy.astype(np.int32)  # Return original points if interpolation fails

    return interpolated_f.astype(np.int32)

def crop_and_resize_image(input_image, target_size=(640, 640)):
    """
    Crop and resize the input image based on YOLO bounding box detection.
    """
    img = input_image
    img_height, img_width = img.shape[:2]

    # Detect objects
    results = model2(img)
    r = results[0]
    boxes = r.boxes.xyxy.tolist()
    
    if not boxes:
        return None

    xmin, ymin, xmax, ymax = map(int, boxes[0])

    # Calculate cropping region centered around the detection
    center_x = (xmin + xmax) // 2
    center_y = (ymin + ymax) // 2

    crop_xmin = max(0, center_x - target_size[0] // 2)
    crop_ymin = max(0, center_y - target_size[1] // 2)
    crop_xmax = min(img_width, center_x + target_size[0] // 2)
    crop_ymax = min(img_height, center_y + target_size[1] // 2)

    cropped_resized_img = img[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
    return cropped_resized_img, (crop_xmin, crop_ymin)

def generalizeLabel(cropped_img, cropped_label, img, crop_origin):
    """
    Map the segmented label from the cropped image back to the original image.
    """
    crop_xmin, crop_ymin = crop_origin
    original_label_list = [[x + crop_xmin, y + crop_ymin] for x, y in cropped_label]
    return original_label_list

# Init list of all the coordinates
XY = []
count = 1

# loop through the video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale and rotate
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    # frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Detect objects and crop
    cropped_data = crop_and_resize_image(frame, (640, 640))
    if cropped_data is not None:
        frame_cropped, crop_origin = cropped_data

        # Apply contrast enhancement
        alpha = 1  # Contrast control (1.0 for no change)
        beta = 0   # Brightness control (0 for no change)
        frame_cropped_contrasted = cv2.convertScaleAbs(frame_cropped, alpha=alpha, beta=beta)

        # Apply YOLO segmentation
        results = model(frame_cropped_contrasted)

        for r in results:
            if r.masks is None:
                continue

            # Process segmentation mask
            mask = r.masks.xy
            xys = mask[0]
            uncropped_xys = generalizeLabel(frame_cropped, xys, frame, crop_origin)
            
            if uncropped_xys:
                XY.append(np.int32(uncropped_xys))
                cv2.polylines(frame, [np.array(uncropped_xys, np.int32)], isClosed=True, color=(0, 0, 255), thickness=3)

                # Interpolate segmentation mask
                interpolated_xy = interpolate_segmentation(uncropped_xys, frame.shape)

                # Draw interpolated segmentation
                cv2.polylines(frame, [interpolated_xy], isClosed=True, color=(255, 0, 0), thickness=2)

    # Resize frame for display
    frame_resized = cv2.resize(frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT))

    # Display video with raw & interpolated segmentation
    cv2.imshow("YOLO Segmentation & Interpolation", frame_resized)

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()