import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import os
import re


def find_subfolders_with_parameters(main_folder_path, re_value=None, h_value=None):
    matching_subfolders = []
    for root, dirs, files in os.walk(main_folder_path):
        for dir in dirs:
            # Extract the Re and h values from the folder name
            re_val, h_val = None, None
            if "Re" in dir:
                re_val = int(dir.split('_')[0].replace("Re", ""))
            if "h" in dir:
                h_val = int(dir.split('_')[1].replace("h", ""))
            # Check if the extracted values match the desired parameter values
            if (re_value is None or re_val == re_value) and (h_value is None or h_val == h_value):
                matching_subfolders.append(os.path.join(root, dir))
    return matching_subfolders

def find_folders_for_parameter_combinations(main_folder_path, parameter_combinations):
    matching_folders = []
    for root, dirs, files in os.walk(main_folder_path):
        for dir in dirs:
            # Extract the Re and h values from the folder name
            re_val, h_val = None, None
            if "Re" in dir:
                re_val = int(dir.split('_')[0].replace("Re", ""))
            if "h" in dir:
                h_val = int(dir.split('_')[1].replace("h", ""))
            # Check if the extracted values match any of the specified parameter combinations
            if (re_val, h_val) in parameter_combinations:
                matching_folders.append(os.path.join(root, dir))
    return matching_folders

def extract_re_from_file_path(file_path):
    # Define a regular expression pattern to match the Re value
    pattern = r"Re(\d+)_h\d+"

    # Search for the pattern in the file path
    match = re.search(pattern, file_path)

    # If a match is found, extract and return the Re value
    if match:
        return int(match.group(1))
    else:
        return None  # Return None if no match is found

# # Example usage:
# main_folder_path = "D:/simuChina"
# parameter_combinations = [(1000, 6), (2000, 8)]  # Example list of (Re, h) combinations
# matching_folders = find_folders_for_parameter_combinations(main_folder_path, parameter_combinations)
# print("Matching folders:")
# for folder_path in matching_folders:
#     print(folder_path)

h = 0.004
nu = 1/1000000
L = 2**7 / 20

def read_cp_data(file_path):
    # input_folder = "C:/Users/blagn771/Documents/Aquaman/Aquaman/lily-pad-master/LilyPad/chinaBenchmark/pressureMotion.txt"
    output = []

    Re = extract_re_from_file_path(file_path)
    _nu = L / Re

    with open(file_path, 'r') as pressure:
        for line in pressure:
            output.append(line)

    X = output[1][12:-3].split()
    for i in range(len(X)):
        X[i] = float(X[i])
    X = np.array(X)
    index_min = np.where(X == X.min())

    X_ordered = np.roll(X, shift=index_min[0], axis=0)
    sensor = np.round(np.linspace(0,99,17)).astype(np.int16)
    sensor = sensor[:-1]
    X_sensor = X_ordered[sensor]

    data = output[4:]
    wall, cylinder, time = [], [], []
    for i in range(len(data)):
        to_add = data[i].split()
        if to_add[-1] == ';':
            to_add.pop()
        to_add = np.array(to_add)
        to_add = to_add.astype(np.float32)
        if i % 2 == 0:
            time.append(to_add[0] * h * h * _nu / nu)
            to_add_ordered = np.roll(to_add[1:], shift=index_min[0], axis=0)
            to_add_sensor = to_add_ordered[sensor]
            cylinder.append(to_add_sensor)
        else:
            wall.append(to_add)

    cylinder_dataframe = pd.DataFrame(cylinder)
    wall_dataframe = pd.DataFrame(wall)

    return cylinder_dataframe, wall_dataframe, time

# Function to plot Cp on the cylinder or wall at different positions
def plot_cp(data_list, time_list, positions, title, ylim):
    fig, axs = plt.subplots(4, 4, figsize=(12, 12))
    axs = axs.flatten()
    for i, pos in enumerate(positions):
        for j in range(len(data_list)):
            axs[i].semilogx(time_list[j][4:], data_list[j][pos][4:], label=f"h={h_values[j]}")
        axs[i].grid()
        axs[i].set_title(title.format(pos))
        axs[i].set_ylim(ylim)
    axs[i].legend()
    plt.suptitle("Evolution of Cp with time at different positions")
    plt.tight_layout()

def plot_mean_cp(theta, meanCp_list):
    plt.figure()
    for i, meanCp in enumerate(meanCp_list):
        plt.plot(theta[:-1], meanCp, label=f"h={h_values[i]}")
    plt.grid()
    plt.title("Mean Cp variation around the cylinder")
    plt.xlabel("Theta (in deg)")
    plt.ylabel("Mean Cp")
    plt.ylim(-0.85, 0.6)
    plt.legend()
    plt.show()

# Function to plot Cp on the wall at different positions
def plot_wall_cp(data_list, time_list, positions, title, ylim):
    fig, axs = plt.subplots(4, 6, figsize=(18, 12))
    axs = axs.flatten()
    for i, pos in enumerate(positions):
        for j in range(len(data_list)):
            axs[i].semilogx(time_list[j][4:], data_list[j][pos][4:], label=f"{j+1}")
        axs[i].grid()
        axs[i].set_title(title.format(pos))
        axs[i].set_ylim(ylim)
    axs[i].legend()
    plt.suptitle("Evolution of Cp with time at different positions on the wall")
    plt.tight_layout()

re_values = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
h_values = np.round(np.linspace(1,5,10),2)
main_folder_path = "D:/simuChina"
re_value = 1000  # Example Re value
h_value = 30  # Example h value
re_1000_subfolders = find_subfolders_with_parameters(main_folder_path, re_value, None)
h_6_subfolders = find_subfolders_with_parameters(main_folder_path, None, h_value)

# Function to plot minimum pressure values for each experiment
def plot_min_pressure(data_list, time_list, positions, title, x):
    fig, axs = plt.subplots(4, 6, figsize=(18, 12))
    axs = axs.flatten()
    for i, pos in enumerate(positions):
        min_pressures = []
        for j in range(len(data_list)):
            min_pressure = np.min(data_list[j][pos][4:])
            min_pressures.append(min_pressure)
        axs[i].plot(x, np.log10(np.abs(min_pressures)), marker='o', linestyle='-')
        axs[i].grid()
        axs[i].set_title(title.format(pos))
    plt.xlabel('Experiment')
    plt.ylabel('Minimum Pressure')

# Function to plot time instances when minimum pressure is reached for each experiment
def plot_time_min_pressure(data_list, time_list, positions, title, x):
    fig, axs = plt.subplots(4, 6, figsize=(18, 12))
    axs = axs.flatten()
    for i, pos in enumerate(positions):
        min_pressure_times = []
        for j in range(len(data_list)):
            min_pressure_time = time_list[j][np.argmin(data_list[j][pos][4:])]
            min_pressure_times.append(min_pressure_time)
        axs[i].semilogx(x, np.log10(min_pressure_times), marker='o', linestyle='-')
        axs[i].grid()
        axs[i].set_title(title.format(pos))
        axs[i].set_ylim((-2,1))
    plt.xlabel('Log (experiment)')
    plt.ylabel('Log Time of Minimum Pressure')

def plot_time_min_pressure_single_fixedH(data_list, time_list, positions, x):
    fig = plt.figure()
    for i, pos in enumerate(positions):
            min_pressure_times = []
            for j in range(len(data_list)):
                min_pressure_time = time_list[j][np.argmin(data_list[j][pos][4:])]
                min_pressure_times.append(min_pressure_time)
            plt.semilogx(x, np.log10(min_pressure_times), marker='o', linestyle='-', label=f"{i+1}")
    plt.legend()
    plt.xlabel("Log of Re")
    plt.ylabel("Log of the time of the min Cp")
    plt.title(f"Log(T_min) with Log(Re) at different position along the wall with h={h_value}")

def plot_min_pressure_single_fixedRe(data_list, time_list, positions, x):
    fig = plt.figure()
    for i, pos in enumerate(positions):
        min_pressures = []
        for j in range(len(data_list)):
            min_pressure = np.min(data_list[j][pos][4:])
            min_pressures.append(min_pressure)
        plt.plot(x, np.log10(np.abs(min_pressures)), marker='o', linestyle='-', label=f"{i+1}")
    plt.legend()
    plt.xlabel("h")
    plt.ylabel("Log of the min Cp")
    plt.title(f"Log(T_min) with h at different position along the wall with Re={re_value}")

# Lists to store data from all experiments
all_cylinder_data = []
all_wall_data = []
all_times = []

# Read and collect data from all experiments with Re=1000
# for folder in re_1000_subfolders:
for folder in h_6_subfolders:
    file_path = os.path.join(folder, "pressureMotion.txt")
    cylinder_data, wall_data, time = read_cp_data(file_path)
    all_cylinder_data.append(cylinder_data)
    all_wall_data.append(wall_data)
    all_times.append(time)

cylinder_positions = range(16)
# plot_cp(all_cylinder_data, all_times, cylinder_positions, "Cp on the cylinder at θ=π+{}π/8", (-2.25, 1.5))
mean_cps = [[np.mean(cylinder_data[i][4:]) for i in range(len(cylinder_data.columns))] for cylinder_data in all_cylinder_data]
# plot_mean_cp(np.linspace(180, 540, 17), mean_cps)

wall_positions = range(21)
plot_wall_cp(all_wall_data, all_times, wall_positions, "Cp on the wall at n+{}L [h]", (-1, 0.4))
# plot_min_pressure(all_wall_data, all_times, wall_positions, "Min Cp on the wall at n+{}L [h]", re_values)
# plot_time_min_pressure(all_wall_data, all_times, wall_positions, "Min Cp time on the wall at n+{}L [h]", re_values)
plot_time_min_pressure_single_fixedH(all_wall_data, all_times, wall_positions, re_values)
plt.show()

# Lists to store data from all experiments
all_cylinder_data = []
all_wall_data = []
all_times = []

for folder in re_1000_subfolders:
    file_path = os.path.join(folder, "pressureMotion.txt")
    cylinder_data, wall_data, time = read_cp_data(file_path)
    all_cylinder_data.append(cylinder_data)
    all_wall_data.append(wall_data)
    all_times.append(time)

wall_positions = range(21)
plot_wall_cp(all_wall_data, all_times, wall_positions, "Cp on the wall at n+{}L [Re]", (-1, 0.4))
# plot_min_pressure(all_wall_data, all_times, wall_positions, "Min Cp on the wall at n+{}L [Re]", h_values)
plot_min_pressure_single_fixedRe(all_wall_data, all_times, wall_positions, h_values)
# plot_time_min_pressure(all_wall_data, all_times, wall_positions, "Min Cp time on the wall at n+{}L [Re]", h_values)
plt.show()