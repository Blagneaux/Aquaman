import pandas as pd
import cv2
import numpy as np

# 
# 
# For testing purpose only
# 
# 

file_ref = "C:/Users/blagn771/Documents/Aquaman/Aquaman/lily-pad-master/LilyPad/testDataSave/pressure_data.csv"
file_yolo = 'C:/Users/blagn771/Documents/Aquaman/Aquaman/lily-pad-master/LilyPad/testDataSave/pressure_data_yolo_noUpdate.csv'

df_ref = pd.read_csv(file_ref, header=None)
df_yolo = pd.read_csv(file_yolo, header=None)

num_columns = min(df_ref.shape[1], df_yolo.shape[1])
min, max = min(df_ref.min()[1:]), max(df_ref.max()[1:])
chord = 2**7 / 8

def reformat(x):
    return 255 * (x - min) / (max - min)

# Set the values to be shades of gray
df_ref1 = df_ref.apply(reformat)
df_yolo1 = df_yolo.apply(reformat)
df = abs(df_ref - df_yolo)

print(df)

min_comp, max_comp = min(df.min()[1:]), max(df.max()[1:])

def reformat_comp(x):
    return 255 * (x - min_comp) / (max_comp - min_comp)

df = df.apply(reformat_comp)
df.clip(0,255)

size = 2**7, 2**7
duration = num_columns
fps = 25
out_ref = cv2.VideoWriter('dataMapRef.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), False)
out_yolo = cv2.VideoWriter('dataMapYolo.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), False)
out_comp = cv2.VideoWriter('dataMapComp.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), False)

for col_name in df_ref.columns:
    data_ref = df_ref1[col_name].to_numpy(np.uint8)
    data_yolo = df_yolo1[col_name].to_numpy(np.uint8)
    data_comp = df[col_name].to_numpy(np.uint8)
    out_ref.write(np.array(data_ref.reshape(size, order='F')))
    out_yolo.write(np.array(data_yolo.reshape(size, order='F')))
    out_comp.write(np.array(data_comp.reshape(size, order='F')))
out_ref.release()
out_yolo.release()
out_comp.release()