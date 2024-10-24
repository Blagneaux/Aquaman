import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import os
import imageio
import io

def load_data(path):
    """Load data from a CSV file"""
    data = pd.read_csv(path, header=None)
    snapshots = data.values  # Convert dataframe into numpy array
    return snapshots

def find_subfolders_with_parameters(main_folder_path, re_value=None, h_value=None):
    matching_subfolders = []
    for root, dirs, files in os.walk(main_folder_path):
        for dir in dirs:
            # Extract the Re and h values from the folder name
            re_val, h_val = None, None
            if "Re" in dir:
                re_val = int(dir.split('_')[0].replace("Re", ""))
            if "h" in dir:
                h_val = float(dir.split('_')[1].replace("h", ""))
            # Check if the extracted values match the desired parameter values
            if (re_value is None or re_val == re_value) and (h_value is None or h_val == h_value):
                matching_subfolders.append(os.path.join(root, dir))
    matching_subfolders = sorted(matching_subfolders)
    if h_value is None:
        matching_subfolders = matching_subfolders[-2:] + matching_subfolders[:-2]
    if re_value is None:
        matching_subfolders = matching_subfolders[1:] + [matching_subfolders[0]]
    return matching_subfolders

plt.rcParams.update({'font.size' : 18, 'font.family' : 'Times New Roman', "text.usetex": False})

Path = 'C:/Users/blagn771/Desktop/DATA/FLUIDS/'
save_path = 'E:/Blog_Posts/OpenFOAM/ROM_Series/Post22/'
Files = os.listdir(Path)

full_path = "E:/benchmark_SINDy/FullPressureMapRe100_h60.csv"
### Data Matrix
X = load_data(full_path)

### Mean-removed Matrix
X_mean = np.mean(X, axis = 1)
Y = X - X_mean[:,np.newaxis]

### Covariance Matrix
C = np.dot(Y.T, Y)/(Y.shape[1]-1)

### SVD of Covariance Matrix
U, S, V = np.linalg.svd(C)

### POD modes
Phi = np.dot(Y, U)

### Temporal coefficients
a = np.dot(Phi.T, Y)

Energy = np.zeros((len(S),1))
for i in np.arange(0,len(S)):
    Energy[i] = S[i]/np.sum(S)

X_Axis = np.arange(Energy.shape[0])
heights = Energy[:,0]

# fig, axes = plt.subplots(1, 2, figsize = (12,4))
# ax = axes[0]
# ax.bar(X_Axis, heights, width=0.5)
# ax.set_xlim(-0.25, 25)
# ax.set_xlabel('Modes')
# ax.set_ylabel('Energy Content')

# ax = axes[1]
cumulative = np.cumsum(S)/np.sum(S)
index99 = 0
index95 = 0
while cumulative[index99]<0.99:
    index99 += 1
    if cumulative[index95]<0.95:
        index95 += 1

print("Reference number of mode to keep 99'%' of the total energy:", index99)
ref99 = index99
print("Reference number of mode to keep 95'%' of the total energy:", index95)
ref95 = index95

folder = "E:/simuChina/cartesianBDD_FullPressureMap"
subfolders = os.listdir(folder)
file_name = "/FullMap.csv"
index99_list = []
index95_list = []
Re_list = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
h_list = [6.0, 8.64, 11.34, 13.98, 16.68, 19.32, 22.02, 24.66, 27.36, 30.0]
for re in Re_list:
    re_list = find_subfolders_with_parameters(folder, re_value=re, h_value=None)
    re99_list = []
    re95_list = []
    for subfolder in re_list:
        full_path = subfolder + file_name

        ### Data Matrix
        X = load_data(full_path)

        ### Mean-removed Matrix
        X_mean = np.mean(X, axis = 1)
        Y = X - X_mean[:,np.newaxis]

        ### Covariance Matrix
        C = np.dot(Y.T, Y)/(Y.shape[1]-1)

        ### SVD of Covariance Matrix
        U, S, V = np.linalg.svd(C)

        ### POD modes
        Phi = np.dot(Y, U)

        ### Temporal coefficients
        a = np.dot(Phi.T, Y)

        Energy = np.zeros((len(S),1))
        for i in np.arange(0,len(S)):
            Energy[i] = S[i]/np.sum(S)

        X_Axis = np.arange(Energy.shape[0])
        heights = Energy[:,0]

        # fig, axes = plt.subplots(1, 2, figsize = (12,4))
        # ax = axes[0]
        # ax.bar(X_Axis, heights, width=0.5)
        # ax.set_xlim(-0.25, 25)
        # ax.set_xlabel('Modes')
        # ax.set_ylabel('Energy Content')

        # ax = axes[1]
        cumulative = np.cumsum(S)/np.sum(S)
        index99 = 0
        index95 = 0
        while cumulative[index99]<0.99:
            index99 += 1
            if cumulative[index95]<0.95:
                index95 += 1

        print("Number of mode to keep 99'%' of the total energy:", index99)
        print("Number of mode to keep 95'%' of the total energy:", index95)
        re99_list.append(index99)
        re95_list.append(index95)

        # ax.plot(cumulative, marker = 'o', markerfacecolor = 'none', markeredgecolor = 'k', ls='-', color = 'k')
        # ax.set_xlabel('Modes')
        # ax.set_ylabel('Cumulative Energy')
        # ax.set_xlim(0, 25)

        # plt.show()
    index99_list.append(re99_list)
    index95_list.append(re95_list)

# plt.figure()
# plt.title("Number of modes to keep at least 99% of the energy with respect to \n the Reynolds number, at a fixed distance between the center of the cylinder and the wall")
# for i, elmt in enumerate(index99_list):
#     plt.plot(Re_list, elmt, marker='o', label=f"h={h_list[i]}")
# plt.plot(Re_list, [ref99]*10, label="Reference")
# plt.xlabel("Reynolds number")
# plt.ylabel("Number of modes")
# plt.legend()
# plt.show()

# plt.figure()
# plt.title("Number of modes to keep at least 95% of the energy with respect to \n the Reynolds number, at a fixed distance between the center of the cylinder and the wall")
# for i, elmt in enumerate(index95_list):
#     plt.plot(Re_list, elmt, marker='o', label=f"h={h_list[i]}")
# plt.plot(Re_list, [ref95]*10, label="Reference")
# plt.xlabel("Reynolds number")
# plt.ylabel("Number of modes")
# plt.legend()
# plt.show()

plt.figure()
plt.title("Number of modes to keep at least 99% of the energy with respect to \n the distance between the center of the cylinder and the wall, at a fixed Reynolds number")
for i, elmt in enumerate(index99_list):
    plt.plot(h_list, elmt, marker='o', label=f"Re={Re_list[i]}")
plt.plot(h_list, [ref99]*10, label="Reference")
plt.legend()
plt.xlabel("Distance between the center of the cylinder and the wall")
plt.ylabel("Number of modes")
plt.show()

plt.figure()
plt.title("Number of modes to keep at least 95% of the energy with respect to \n the distance between the center of the cylinder and the wall, at a fixed Reynolds number")
for i, elmt in enumerate(index95_list):
    plt.plot(h_list, elmt, label=f"Re={Re_list[i]}")
plt.plot(h_list, [ref95]*10, marker='o', label="Reference")
plt.xlabel("Distance between the center of the cylinder and the wall")
plt.ylabel("Number of modes")
plt.legend()
plt.show()