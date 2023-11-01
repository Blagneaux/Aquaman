import cv2
import numpy as np

img = cv2.imread("python/frame-0001.png", cv2.IMREAD_ANYCOLOR)
h,w = img.shape[:2]

for i in range(h):
    for j in range(w):
        if (img[i][j][0] != 51) or (img[i][j][1] != 51) or (img[i][j][2] != 153):
            img[i,j] = [255,255,255]
        else:
            print([i,j])

img = cv2.GaussianBlur(img, (11,11), 0)
canny = cv2.Canny(img, 100, 150)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()