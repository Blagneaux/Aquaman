import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re

DTW = []
DTW1 = []
DTW2 = []
DTW4 = []
Frechet = []
index = 0

path = "E:/crop_nadia/matchingData"
folder = sorted(os.listdir(path), key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else float('inf'))
for file in folder:
    if file.endswith(".csv"):
        file_name = os.path.join(path, file)
        data = pd.read_csv(file_name)
        validity = data["Validity"]
        dtw_score = data["DTW_score"]
        frechet_dist = data["Frechet_dist"]
        channel = data["Channel"]

        for i in range(len(validity)):
            if validity[i] :#and file_name[-6:-4] != "31" and file_name[-6:-4] != "26": # 26 gives unrealistic values, probably due to the tail eating combined with the important size of the fish
                DTW.append(dtw_score[i])
                Frechet.append(frechet_dist[i])
                if channel[i] == "S1":
                    DTW1.append(dtw_score[i])
                elif channel[i] == "S2":
                    DTW2.append(dtw_score[i])
                elif channel[i] == "S4":
                    DTW4.append(dtw_score[i])
                # if index == 53:
                #     print(file_name, data["Channel"], data["Sample"])
                # index += 1

DTW_gliding_mean = [np.sum(DTW[:i+1])/(i+1) for i in range(len(DTW))]
DTW_gliding_mean.append(np.sum(DTW)/len(DTW))
Frechet_gliding_mean = [np.sum(Frechet[:i+1])/(i+1) for i in range(len(Frechet))]
Frechet_gliding_mean.append(np.sum(Frechet)/len(Frechet))

# Set global font size
plt.rcParams.update({'font.size': 12})  # Adjust the number as needed for your preference

# Create a figure with two subplots (two rows, one column)
fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # You can adjust the figure size as needed

# First subplot for DTW
axs[0].plot(DTW, label="DTW distance for a given sample")
axs[0].plot(DTW_gliding_mean, label="Gliding mean DTW distance")
axs[0].set_xlabel("Matching samples between the simulation and the sensors", fontsize=14)
axs[0].set_ylabel("DTW distance", fontsize=14)
axs[0].legend(fontsize=12)
axs[0].set_title("DTW distance for each sample and its gliding mean on a\n window  starting from the first sample to each sample", fontsize=16)

# Second subplot for Fréchet
axs[1].plot(Frechet, label="Fréchet distance for a given sample")
axs[1].plot(Frechet_gliding_mean, label="Gliding mean Fréchet distance")
axs[1].set_xlabel("Matching samples between the simulation and the sensors", fontsize=14)
axs[1].set_ylabel("Fréchet distance (in Pa)", fontsize=14)
axs[1].legend(fontsize=12)
axs[1].set_title("Fréchet distance for each sample and its gliding mean on a\n window  starting from the first sample to each sample", fontsize=16)

# Adjust layout to prevent overlap and ensure readability
plt.tight_layout()
plt.show()

print("Mean DTW distance:", np.mean(DTW))
print("Mean DTW distance for sensor 1:", np.mean(DTW1))
print("Mean DTW distance for sensor 2:", np.mean(DTW2))
print("Mean DTW distance for sensor 4:", np.mean(DTW4))
print("Mean Frechet distance:", np.mean(Frechet))
print("Number of samples:", len(DTW))

# plt.figure()
# plt.plot(DTW2)
# plt.show()