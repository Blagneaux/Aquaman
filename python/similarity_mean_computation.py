import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re

# ------------------------------------------------

# Based on the similarity distance metrics computed in similarity_computation.py,
# this code computes the mean value for each metric and the cumulative mean

# ------------------------------------------------


path_mid = "D:/crop_nadia/matchingData_2s_window"
path_left = "D:/crop_nadia/matchingData_minus3grid"
path_right = "D:/crop_nadia/matchingData_plus3grid"
path_circle = "D:/crop_nadia/matchingData_circle_2s_window"
paths = [path_mid, path_circle]#, path_left, path_right]
forbiden = [[1,7], [1,11], [2,3], [6,10], [9,5], [9,6], [9,16], [11,3], [17,15], [17,16], [18,7], [18,17], [18,20], [25,0], [25,2], [31,1], [31,3], [31,5]]
# forbiden = []

for path in paths:
    DTW = []
    Frechet = []
    Correlation = []
    index = 0
    folder = sorted(os.listdir(path), key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else float('inf'))
    
    for file in folder:
        if file.endswith(".csv"):
            file_name = os.path.join(path, file)
            data = pd.read_csv(file_name)
            validity = data["Validity"]
            dtw_score = data["DTW_score"]
            frechet_dist = data["Frechet_dist"]
            correlation = data["Correlation"]
            channel = data["Channel"]

            for i in range(len(validity)):
                if validity[i] :#and file_name[-6:-4] != "31" and file_name[-6:-4] != "26": # 26 gives unrealistic values, probably due to the tail eating combined with the important size of the fish
                    if path == path_mid:
                        if not [int(file[:-4]), i] in forbiden:
                            DTW.append(dtw_score[i])
                            Frechet.append(frechet_dist[i])
                            Correlation.append(correlation[i])
                    elif path == path_circle:
                        if not [int(file_name.split("_")[-1][:-4]), i] in forbiden:
                            DTW.append(dtw_score[i])
                            Frechet.append(frechet_dist[i])
                            Correlation.append(correlation[i])

    DTW_gliding_mean = [np.sum(DTW[:i+1])/(i+1) for i in range(len(DTW))]
    DTW_gliding_mean.append(np.sum(DTW)/len(DTW))
    Frechet_gliding_mean = [np.sum(Frechet[:i+1])/(i+1) for i in range(len(Frechet))]
    Frechet_gliding_mean.append(np.sum(Frechet)/len(Frechet))
    Correlation_gliding_mean = [np.sum(Correlation[:i+1])/(i+1) for i in range(len(Correlation))]
    Correlation_gliding_mean.append(np.sum(Correlation)/len(Correlation))

    # Set global font size
    plt.rcParams.update({'font.size': 12})  # Adjust the number as needed for your preference

    # Create a figure with two subplots (two rows, one column)
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # You can adjust the figure size as needed

    # First subplot for DTW
    axs[0].plot(DTW, 'o', label="DTW distance for a given sample")
    axs[0].plot(DTW_gliding_mean, label="Cumulative mean DTW distance")
    axs[0].plot(len(DTW_gliding_mean), DTW_gliding_mean[-1], 'rx', label="Mean DTW distance")
    axs[0].set_xlabel("Matching samples between the simulation and the sensors", fontsize=14)
    axs[0].set_ylabel("DTW distance", fontsize=14)
    axs[0].legend(fontsize=12)
    axs[0].set_title("DTW distance for each sample and its cumulative mean on a\n window  starting from the first sample to each sample", fontsize=16)

    # Second subplot for Fréchet
    axs[1].plot(Frechet, 'o', label="Fréchet distance for a given sample")
    axs[1].plot(Frechet_gliding_mean, label="Cumulative mean Fréchet distance")
    axs[1].plot(len(Frechet_gliding_mean), Frechet_gliding_mean[-1], 'rx', label="Mean Fréchet distance")
    axs[1].set_xlabel("Matching samples between the simulation and the sensors", fontsize=14)
    axs[1].set_ylabel("Fréchet distance (normalized pressure)", fontsize=14)
    axs[1].legend(fontsize=12)
    axs[1].set_title("Fréchet distance for each sample and its cumulative mean on\n a window  starting from the first sample to each sample", fontsize=16)

    # Adjust layout to prevent overlap and ensure readability
    plt.tight_layout()
    plt.show()

    print("Path:", path)

    print("Mean DTW distance:", np.mean(DTW))
    print("Mean Frechet distance:", np.mean(Frechet))
    print("Mean Correlation:", np.mean(Correlation))
    print("Number of samples:", len(DTW))

    # plt.figure()
    # plt.plot(DTW2)
    # plt.show()