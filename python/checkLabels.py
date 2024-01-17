import pandas as pd
import numpy as np
import cv2

file_path = "C:/Users/blagn771/Desktop/FishDataset/Fish1C1/labels/Fish1C1_frame-0001.txt"
df = pd.read_csv(file_path, sep=" ", header=None)
img = np.zeros((640,640))
print(df)
for i in range(1, len(df.columns), 2):
    x = int(df[i][0] * 640)
    y = int(df[i+1][0] * 640)
    print(x, y)
    img[y,x] = 1

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()