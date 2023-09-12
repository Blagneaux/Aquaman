import cv2
from numpy import *

cap = cv2.VideoCapture('C:/Users/blagn771/Documents/Aquaman/RES15/Enregistrement_2023-04-21_160532.mp4')

if (cap.isOpened() == False):
    print("Error opening video stream or file")

count = 0

while cap.isOpened():
    _, frame = cap.read()
    blurred = cv2.GaussianBlur(frame, (5,5), 0)
    gray = cv2.cvtColor(blurred,cv2.COLOR_BGR2GRAY)

    trigger = 75
    mask = cv2.inRange(gray,0,trigger)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(frame,contours, -1, (0,255,0), 3)

    cv2.imshow("Frame",frame)
    # cv2.imshow("Mask", mask)

    cv2.imwrite("C:/Users/blagn771/Documents/Aquaman/Aquaman/outputContour/frame000%d.jpg" % count, frame)
    count += 1

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()