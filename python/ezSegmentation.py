import cv2
import numpy as np

# videoSeg part take a video in which there is no fish, and computes the median image
# The goal is to get delete the flickering of the light

# videoProcess part take a video in which there IS a fish, substract the median that has been computed
# and create a mask and an image for every frame of the video

frameSeg = False
videoSeg = False
videoProcess = True

if frameSeg:
    img = cv2.imread("python/frame-0001.png", cv2.IMREAD_ANYCOLOR)
    h,w = img.shape[:2]

    for i in range(h):
        for j in range(w):
            if (img[i][j][0] != 51) or (img[i][j][1] != 51) or (img[i][j][2] != 153):
                img[i,j] = [0,0,0]
            else:
                img[i,j] = [255,255,255]

    img = cv2.GaussianBlur(img, (11,11), 0)
    canny = cv2.Canny(img, 100, 150)
    cv2.imwrite("frame-0001-label.png", img)

    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if videoSeg:
    print("videoSeg")
    cap = cv2.VideoCapture("C:/Users/blagn771/Desktop/T3_Fish3_C1_270923 - Trim0.mp4")
    frames = []
    while cap.isOpened() and len(frames) < 200:
        ret, frame = cap.read()

        if ret:
            frames.append(frame)

    medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)

    cv2.imwrite("medianT3_Fish3_C1_270923.png", medianFrame)
    cv2.imshow('medianPicture', medianFrame)
    cv2.waitKey(0)
    
    cap.release()
    cv2.destroyAllWindows()

if videoProcess:
    print("videoProcess")
    grayMedian = cv2.imread("C:/Users/blagn771/Documents/Aquaman/Aquaman/medianT3_Fish3_C1_270923.png")
    grayMedian = cv2.cvtColor(grayMedian, cv2.COLOR_BGR2GRAY)

    cap = cv2.VideoCapture("C:/Users/blagn771/Desktop/T3_Fish3_C1_270923 - Trim1.mp4")
    count = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gframe = cv2.GaussianBlur(frame, (5,5), cv2.BORDER_DEFAULT)
            dframe = cv2.absdiff(gframe, grayMedian)
            th, dframe = cv2.threshold(dframe, 30, 255, cv2.THRESH_BINARY)

            cv2.imwrite("C:/Users/blagn771/Desktop/Fish3C1/images"+"/Fish1C1_frame-"+str(10000+count)[1:]+".png", frame)
            cv2.imwrite("C:/Users/blagn771/Desktop/Fish3C1/masks"+"/Fish1C1_frame-"+str(10000+count)[1:]+".png", dframe)
            count+=1
            cv2.imshow('frame', dframe)
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

        else:
            break

    cap.release()
    cv2.destroyAllWindows()