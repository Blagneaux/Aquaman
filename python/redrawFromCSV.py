import numpy as np
import cv2
import pandas as pd

file_path = 'C:/Users/blagn771/Documents/Aquaman/Aquaman/lily-pad-master/LilyPad/testDataSave/pressure_data.csv'

# Use the read_csv function to read the CSV file into a Pandas DataFrame
df = pd.read_csv(file_path, header=None)
num_columns = df.shape[1]
min, max = min(df.min()), max(df.max())

# Set the values to be shades of gray
def reformat(x):
    return 255 * (x - min)  / (max - min)

df1 = df.apply(reformat)

# Create the video
size = 2**7, 2**7
duration = num_columns
fps = 25
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), False)

for col_name in df.columns:
    data = df1[col_name].to_numpy(np.uint8)
    out.write(np.array(data.reshape(size, order='F')))
out.release()