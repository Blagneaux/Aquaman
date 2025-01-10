import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import os
import imageio
import io
import statistics
import matplotlib.animation as animation


font = {'family': 'Arial',
        'weight': 'normal',
        'size': 22}
matplotlib.rc('font', **font)

def reconstruction_error(data, vectors, signals, window_width=256, window_height=128, x=0, middle_y=64):
    steps = signals.shape[1]
    error = 0

    # if (window_height != 128 and window_width != 256):
    #     data = crop_video(data, window_width, window_height, x, middle_y)

    for i in range(steps):
        frame = vectors @ signals[:, i]
        # if (window_height != 128 and window_width != 256):
        #     frame = crop_image(frame.reshape(-1, 1), window_width, window_height, x, middle_y)
        error += np.sqrt(((data[:,i] - frame)**2).mean())
    
    return error/steps

def crop_image(flattened_image, window_width, window_height, x, middle_y):
    # Constants for image dimensions
    height = 128
    width = 256

    # Reshape the flattened image back into 2D
    image_2d = flattened_image.reshape(width, height)

    # Calculate the y starting point to center the crop around middle_y
    start_y = int(max(0, middle_y - window_height // 2))
    end_y = int(min(height, start_y + window_height))

    # Adjust start_y if the window exceeds image bounds
    if end_y > height:
        end_y = height
        start_y = max(0, end_y - window_height)

    # Crop the image
    crop = image_2d[x:x+window_width, start_y:end_y]
    return crop.flatten()

def crop_video(data, window_width, window_height, x, middle_y):
    # Prepare the output array with appropriate size
    num_images = data.shape[1]
    crop_size = crop_image(data[:, 0], window_width, window_height, x, middle_y).shape[0]
    cropped_images = np.empty((int(crop_size), num_images))

    # Prepare each image
    for i in range(num_images):
        cropped_images[:, i] = crop_image(data[:, i], window_width, window_height, x, middle_y)

    return cropped_images

def flow_reconstruction(X):
    steps = X.shape[1]
    print(X.shape)
    fig = plt.figure()  # Initialize the figure outside the loop

    img = []

    for i in range(steps):
        if crop:
            frame = X[:, i].reshape(window_width, -1)
        else:
            frame = X[:, i].reshape(256, 128)

        frame_rotated = np.rot90(frame)

        img.append([plt.imshow(frame_rotated, cmap="seismic", aspect='auto')])

    ani = animation.ArtistAnimation(fig, img, interval=50, blit=True, repeat_delay=100)
    plt.show()

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
                if re_val >= 1000:
                    matching_subfolders.append(os.path.join(root, dir))
    matching_subfolders = sorted(matching_subfolders)
    if h_value is None:
        matching_subfolders = matching_subfolders[-3:] + matching_subfolders[:-3]
        matching_subfolders = [matching_subfolders[-1]] + matching_subfolders[2:-1] + matching_subfolders[0:2]
    if re_value is None:
        matching_subfolders = matching_subfolders[1:] + [matching_subfolders[0]]
    return matching_subfolders

def find_subfolders_with_parameters_init(main_folder_path, re_value=None, h_value=None):
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
        matching_subfolders = matching_subfolders[-3:] + matching_subfolders[:-3]
        matching_subfolders = [matching_subfolders[-1]] + matching_subfolders[2:-1] + matching_subfolders[0:2]
    if re_value is None:
        matching_subfolders = matching_subfolders[0::2] + matching_subfolders[1::2]
        matching_subfolders = matching_subfolders[1:] + [matching_subfolders[0]]
    return matching_subfolders

def mean_of_elements(list_of_lists):
    # Transpose the list of lists using zip and unpacking (*)
    transposed = zip(*list_of_lists)
    # Compute the mean of each tuple (originally a column in list_of_lists)
    return [statistics.mean(sublist) for sublist in transposed]

plt.rcParams.update({"text.usetex": False})

Path = 'C:/Users/blagn771/Desktop/DATA/FLUIDS/'
save_path = 'E:/Blog_Posts/OpenFOAM/ROM_Series/Post22/'
Files = os.listdir(Path)

# -------------------------------------------------------------------------------------------------------
# Cropping parameters
# -------------------------------------------------------------------------------------------------------
n = 2**7
# window_width = int(n/2)
window_width = int(n)
window_height = int(window_width/2)
# x_start = int(3*n/7)
x_start = int(2*n/7)
middle_y = None
crop = True

full_path = "E:/simuChina/cartesianBDD_FullVorticityMapFixed/Re100_h64.0/FullMap.csv"
### Data Matrix
X = load_data(full_path)

### Crop to a narrower window
if crop:
    X = crop_video(X, window_width=window_width, window_height=window_height, x=x_start, middle_y=64)

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
print("Reference energy of the most energetic mode:", heights[0])
ref1 = heights[0]
print("Reference reconstruction rmse for 99'%' of the energy:", reconstruction_error(Y, Phi[:, :index99], a[:index99, :], window_width, window_height, x_start, 64))
ref_error = reconstruction_error(X,Phi[:, :index99],a[:index99, :], window_width, window_height, x_start, 64)

# --------------------------------------------------------------------------------------
# Analysis of the gradation of Re from what is usually done to what we need here
# --------------------------------------------------------------------------------------

# folder = "E:/simuChina/cartesianBDD_FullVorticityMapFixed"
# subfolders = os.listdir(folder)
# file_name = "/FullMap.csv"
# re_init = 64.0
# re_list = find_subfolders_with_parameters_init(folder, re_value=None, h_value=re_init)
# re99_list = []
# re95_list = []
# nrj_first_mode = []
# error_reconstruction = []
# for subfolder in re_list:
#     print(subfolder)
#     full_path = subfolder + file_name

#     ### Data Matrix
#     X = load_data(full_path)

#     h_val = float(subfolder.split('_')[2].replace("h", ""))

#     ### Crop to a narrower window
#     if crop:
#         X = crop_video(X, window_width=window_width, window_height=window_height, x=x_start, middle_y=h_val)

#     if (h_val == 6.0) and (debug == 0):
#         flow_reconstruction(X)
#         debug = 1

#     ### Mean-removed Matrix
#     X_mean = np.mean(X, axis = 1)
#     Y = X - X_mean[:,np.newaxis]

#     ### Covariance Matrix
#     C = np.dot(Y.T, Y)/(Y.shape[1]-1)

#     ### SVD of Covariance Matrix
#     U, S, V = np.linalg.svd(C)

#     ### POD modes
#     Phi = np.dot(Y, U)

#     ### Temporal coefficients
#     a = np.dot(Phi.T, Y)

#     Energy = np.zeros((len(S),1))
#     for i in np.arange(0,len(S)):
#         Energy[i] = S[i]/np.sum(S)

#     X_Axis = np.arange(Energy.shape[0])
#     heights = Energy[:,0]
#     cumulative = np.cumsum(S)/np.sum(S)
#     index99 = 0
#     index95 = 0
#     nrj_first = heights[0]
#     while cumulative[index99]<0.99:
#         index99 += 1
#         if cumulative[index95]<0.95:
#             index95 += 1

#     error = reconstruction_error(Y, Phi[:, :index99], a[:index99, :])

#     re99_list.append(index99)
#     re95_list.append(index95)
#     nrj_first_mode.append(nrj_first)
#     error_reconstruction.append(error)

# Re_list = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

# plt.figure()
# plt.title("Number of modes to keep at least 99% of the energy with respect to \n the Reynolds number, at a the center of the tank.")
# plt.plot(Re_list, re99_list, marker='o', linestyle='dashed')
# plt.plot(Re_list, [ref99]*len(Re_list), label="Reference")
# plt.xlabel("Reynolds number")
# plt.ylabel("Number of modes")
# plt.legend()

# plt.figure()
# plt.title("Number of modes to keep at least 95% of the energy with respect to \n the Reynolds number, at a the center of the tank.")
# plt.plot(Re_list, re95_list, marker='o', linestyle='dashed')
# plt.plot(Re_list, [ref95]*len(Re_list), label="Reference")
# plt.xlabel("Reynolds number")
# plt.ylabel("Number of modes")
# plt.legend()

# plt.figure()
# plt.title("Energy of the most energetic mode with respect to the Reynolds number \n at a the center of the tank.")
# plt.plot(Re_list, nrj_first_mode, marker='o', linestyle='dashed')
# plt.plot(Re_list, [ref1]*len(Re_list), label="Reference")
# plt.xlabel("Reynolds number")
# plt.ylabel("Energy of the most energetic mode")
# plt.legend()

# plt.figure()
# plt.title("RMSE of the reconstruction of the flow with 99% of the energy with respect to the \n Reynolds number, at a the center of the tank.")
# plt.plot(Re_list, error_reconstruction, marker='o', linestyle='dashed')
# plt.plot(Re_list, [ref_error]*len(Re_list), label="Reference")
# plt.xlabel("Reynolds number")
# plt.ylabel("RMSE of reconstruction with 99% of the energy")
# plt.legend()
# plt.show()

# ----------------------------------------------------------------------------------------
# Analysis of sensibility of POD for SUBARU
# ----------------------------------------------------------------------------------------

folder = "E:/simuChina/cartesianBDD_FullVorticityMapFixed"
# folder = "E:/simuChina/cartesianBDD_FullVorticityMap"
subfolders = os.listdir(folder)
file_name = "/FullMap.csv"
index99_list = []
index95_list = []
nrj_first_mode_list = []
error_reconstruction_list = []
Re_list = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
h_list = [6.0, 8.64, 11.34, 13.98, 16.68, 19.32, 22.02, 24.66, 27.36, 30.0, 60.0, 64.0]
debug = 0
# ------------------------------------------------------------------------------------------------------------
# Change the two next line for fixed Re or fixed h
# ------------------------------------------------------------------------------------------------------------
for re in h_list:
    re_list = find_subfolders_with_parameters(folder, re_value=None, h_value=re)
    re99_list = []
    re95_list = []
    nrj_first_mode = []
    error_reconstruction = []
    for subfolder in re_list:
        print(subfolder)
        full_path = subfolder + file_name

        ### Data Matrix
        X = load_data(full_path)
        # X = X[:, :X.shape[1]//4]  Crop to keep an equivalent length as to the moving scenarios

        h_val = float(subfolder.split('_')[2].replace("h", ""))

        ### Crop to a narrower window
        if crop:
            X = crop_video(X, window_width=window_width, window_height=window_height, x=x_start, middle_y=h_val)

        if (h_val == 6.0) and (debug == 0):
            flow_reconstruction(X)
            debug = 1

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
        cumulative = np.cumsum(S)/np.sum(S)
        index99 = 0
        index95 = 0
        nrj_first = heights[0]
        while cumulative[index99]<0.99:
            index99 += 1
            if cumulative[index95]<0.95:
                index95 += 1

        error = reconstruction_error(Y, Phi[:, :index99], a[:index99, :], window_width, window_height, x_start, h_val)

        re99_list.append(index99)
        re95_list.append(index95)
        nrj_first_mode.append(nrj_first)
        error_reconstruction.append(error)

    index99_list.append(re99_list)
    index95_list.append(re95_list)
    nrj_first_mode_list.append(nrj_first_mode)
    error_reconstruction_list.append(error_reconstruction)

index99_mean = mean_of_elements(index99_list)
index95_mean = mean_of_elements(index95_list)
nrj_first_mode_mean = mean_of_elements(nrj_first_mode_list)
erro_mean = mean_of_elements(error_reconstruction_list)
print(index99_mean)
print(index95_mean)
print(nrj_first_mode_mean)
print(erro_mean)

# _____________________________________________________________________________________________
# Don't forget to recalculate the gap for the higher resolution
# _____________________________________________________________________________________________
gap_list = [round((h-128/20/2)/(128/20),2) for h in h_list]

plt.figure()
plt.title("Number of modes to keep at least 99% of the energy with respect to \n the Reynolds number, at a fixed normalized gap between the center of the cylinder and the wall")
for i, elmt in enumerate(index99_list):
    plt.plot(Re_list, elmt, marker='o', label=f"h={gap_list[i]}", linestyle='dashed')
plt.plot(Re_list, [ref99]*len(Re_list), label="Reference")
plt.xlabel("Reynolds number")
plt.ylabel("Number of modes")
plt.legend()

plt.figure()
plt.title("Number of modes to keep at least 95% of the energy with respect to \n the Reynolds number, at a fixed normalized gap between the center of the cylinder and the wall")
for i, elmt in enumerate(index95_list):
    plt.plot(Re_list, elmt, marker='o', label=f"h={gap_list[i]}", linestyle='dashed')
plt.plot(Re_list, [ref95]*len(Re_list), label="Reference")
plt.xlabel("Reynolds number")
plt.ylabel("Number of modes")
plt.legend()

plt.figure()
plt.title("Energy of the most energetic mode with respect to the Reynolds number \n at a fixed normalized gap between the center of the cylinder and the wall")
for i, elmt in enumerate(nrj_first_mode_list):
    plt.plot(Re_list, elmt, label=f"h={gap_list[i]}", marker='o', linestyle='dashed')
plt.plot(Re_list, [ref1]*len(Re_list), label="Reference")
plt.xlabel("Reynolds number")
plt.ylabel("Energy of the most energetic mode")
plt.legend()

plt.figure()
plt.title("RMSE of the reconstruction of the flow with 99% of the energy with respect to the \n Reynolds number, at a fixed normalized gap between the center of the cylinder and the wall")
for i, elmt in enumerate(error_reconstruction_list):
    plt.plot(Re_list, elmt, label=f"h={gap_list[i]}", marker='o', linestyle='dashed')
plt.plot(Re_list, [ref_error]*len(Re_list), label="Reference")
plt.xlabel("Reynolds number")
plt.ylabel("RMSE of reconstruction with 99% of the energy")
plt.legend()
plt.show()

# ---------------------------------------------------------------------------------

# plt.figure()
# plt.title("Number of modes to keep at least 99% of the energy with respect to \n the normalized gap between the center of the cylinder and the wall, at a fixed Reynolds number")
# for i, elmt in enumerate(index99_list):
#     plt.plot(gap_list, elmt, marker='o', label=f"Re={Re_list[i]}", linestyle='dashed')
# plt.plot(gap_list, [ref99]*len(gap_list), label="Reference")
# plt.legend()
# plt.xlabel("Distance between the center of the cylinder and the wall over the diameter of the cylinder")
# plt.xticks(gap_list)
# plt.ylabel("Number of modes")

# plt.figure()
# plt.title("Number of modes to keep at least 95% of the energy with respect to \n the normalized gap between the center of the cylinder and the wall, at a fixed Reynolds number")
# for i, elmt in enumerate(index95_list):
#     plt.plot(gap_list, elmt, label=f"Re={Re_list[i]}", marker='o', linestyle='dashed')
# plt.plot(gap_list, [ref95]*len(gap_list), label="Reference")
# plt.xlabel("Distance between the center of the cylinder and the wall over the diameter of the cylinder")
# plt.xticks(gap_list)
# plt.ylabel("Number of modes")
# plt.legend()

# plt.figure()
# plt.title("Energy of the most energetic mode with respect to the distance between \n the normalized gap of the cylinder and the wall, at a fixed Reynolds number")
# for i, elmt in enumerate(nrj_first_mode_list):
#     plt.plot(gap_list, elmt, label=f"Re={Re_list[i]}", marker='o', linestyle='dashed')
# plt.plot(gap_list, [ref1]*len(gap_list), label="Reference")
# plt.xlabel("Distance between the center of the cylinder and the wall over the diameter of the cylinder")
# plt.xticks(gap_list)
# plt.ylabel("Energy of the most energetic mode")
# plt.legend()

# plt.figure()
# plt.title("RMSE of the reconstruction of the flow with 99% of the energy with respect to the \n  normalized gap between the center of the cylinder and the wall, at a fixed Reynolds number")
# for i, elmt in enumerate(error_reconstruction_list):
#     plt.plot(gap_list, elmt, label=f"Re={Re_list[i]}", marker='o', linestyle='dashed')
# plt.plot(gap_list, [ref_error]*len(gap_list), label="Reference")
# plt.xlabel("Distance between the center of the cylinder and the wall over the diameter of the cylinder")
# plt.xticks(gap_list)
# plt.ylabel("RMSE of reconstruction with 99% of the energy")
# plt.legend()
# plt.show()