from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("C:/Users/blagn771/Desktop/FishDataset/segment/train1280_32_291/weights/best.pt")
cap = cv2.VideoCapture("C:/Users/blagn771/Desktop/FishDataset/T1_Fish1_C1_270923 - Trim1.mp4")

frame_width = int(cap.get(3))//2
frame_height = int(cap.get(4))//2

size = (frame_width, frame_height)

# Init list of all the coordinates
XY = []
XY_interpolated = []

# loop through the video frames
while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        # run inference on a frame
        frame = cv2.resize(frame, (frame_width, frame_height))
        results = model(frame)

        # view results
        for r in results:
            if r.masks == None:
                break
            mask = r.masks.xy
            xys = mask[0]
            XY.append(np.int32(xys))
            cv2.polylines(frame, np.int32([xys]), True, (0, 0, 255), 2)

        cv2.imshow("img", frame)

        #break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    else:
        break

cap.release()
cv2.destroyAllWindows()