import numpy as np
import pandas as pd
from pydmd import DMD, BOPDMD
from pydmd.plotter import plot_modes_2D, plot_summary
import matplotlib.pyplot as plt
import scipy
from pydmd.preprocessing import hankel_preprocessing
import cv2
import os


raw_csv_path = 'C:/Users/blagn771/Documents/Aquaman/Aquaman/lily-pad-master/LilyPad/chinaBenchmark/FullMap.csv'

raw_df = pd.read_csv(raw_csv_path, header=None)
raw_array = np.array(raw_df)
raw_list = [np.array(raw_df[i]).reshape(2**8, 2**7) for i in range(len(raw_df.columns))]
raw_listT = [np.array(raw_df[i]).reshape(2**7, 2**8) for i in range(len(raw_df.columns))]
t = np.linspace(0, len(raw_df.columns)*0.4, len(raw_df.columns))

save_folder = "C:/Users/blagn771/Desktop/niqueFFMPEG"
os.chdir(save_folder)

for j in range(len(raw_df.columns)):
    img = raw_list[j]
    for i in range(len(raw_list[0])):
        for k in range(len(raw_list[0][0])):
            if img[i][k] < -0.5:
                img[i][k] = -0.5*255+175.5
            elif img[i][k] > 0.5:
                img[i][k] = 0.5*255+175.5
            else:
                img[i][k] = img[i][k]*255+175.5
    cv2.imshow('img', img) 
    cv2.imwrite("frame-"+str(10000+j)[1:]+".png", img)

    #break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()

# apply dmd to the 2D data
dmd = BOPDMD(svd_rank=10)
dmd.fit(raw_listT, t)
plot_modes_2D(dmd, figsize=(12,5))

fig = plt.plot(
    scipy.linalg.svdvals(
        np.array([snapshot.flatten() for snapshot in raw_list]).T
    ),
    "o",
)
plt.show()
print(
    f"Frequencies (imaginary component): {np.round(dmd.eigs / 2 / np.pi, decimals=3)}"
)

save_folder2 = "C:/Users/blagn771/Desktop/niqueFFMPEG2"
os.chdir(save_folder2)

dmd_states = [state.reshape(2**8,2**7).real for state in dmd.reconstructed_data.T]
for j in range(len(dmd_states)):
    img = dmd_states[j]
    for i in range(len(dmd_states[0])):
        for k in range(len(dmd_states[0][0])):
            if img[i][k] < -0.5:
                img[i][k] = -0.5*255+175.5
            elif img[i][k] > 0.5:
                img[i][k] = 0.5*255+175.5
            else:
                img[i][k] = img[i][k]*255+175.5
    cv2.imshow('img', img) 
    cv2.imwrite("frame-"+str(10000+j)[1:]+".png", img)

    #break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()

# # apply dmd on 1D data
d = 2 # number of delays
x = np.linspace(1, 2**15, 2**15)
# dt = 0.4
# dmd = DMD(svd_rank=8)
# delay_dmd = hankel_preprocessing(dmd, d=d)
# delay_dmd.fit(raw_array)
# plot_summary(delay_dmd, x=x, t=dt, d=d)
# print(
#     f"Frequencies (imaginary component): {np.round(np.log(delay_dmd.eigs) / dt / 2 / np.pi, decimals=12)}"
# )

# # apply  optimized dmd on 1D data
# optdmd = BOPDMD(svd_rank=4, num_trials=0)
# delay_optdmd = hankel_preprocessing(optdmd, d=d)
# delay_t = t[:-d+1]
# delay_optdmd.fit(raw_array, t=delay_t)
# plot_summary(delay_optdmd, x=x, d=d)
# print(
#     f"Frequencies (imaginary component): {np.round(delay_optdmd.eigs / 2 / np.pi, decimals=3)}"
# )