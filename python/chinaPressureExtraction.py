import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import os
import re
import scipy.signal as signal
from scipy.stats import linregress, t


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

n = 2**7
h = 1/n
nu = 1/1000000
L = n / 20

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
    plt.suptitle("Evolution of Cp with time (s) at different positions")
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

def apply_high_pass_filter(signal_data, fs, cutoff_freq):
    # Normalize cutoff frequency to Nyquist frequency
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist

    # Design high-pass Butterworth filter
    order = 4
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)

    # Apply the filter to the signal
    filtered_signal = signal.filtfilt(b, a, signal_data)

    return filtered_signal

# Function to plot Cp on the wall at different positions
def plot_wall_cp(data_list, time_list, positions, title, ylim, fixedRe):
    fig, axs = plt.subplots(4, 6, figsize=(18, 12))
    axs = axs.flatten()

    for i, pos in enumerate(positions):
        for j in range(len(data_list)):
            filtered_signal = apply_high_pass_filter(data_list[j][pos][4:], fs=1/(time_list[j][1] - time_list[j][0]), cutoff_freq=0.004*re_value/1000)
            axs[i].plot(time_list[j][4:], filtered_signal, label=f"{j+1}")
        axs[i].grid()
        axs[i].set_title(title.format(pos))
        axs[i].set_ylim(ylim)
    for i, pos in enumerate(positions):
        for j in range(len(data_list)):
            time_cyl = np.linspace(0, time_list[j][-1], 500)
            pos_cyl = np.zeros(time_cyl.shape)
            for k in range(len(time_cyl)-1):
                if fixedRe:
                    u = 0.02 * re_value/1000
                else:
                    u = 0.02 * re_values[j]/1000
                if u*time_cyl[k] > 2**7*h - i*L*h - L*h/2 and u*time_cyl[k] < 2**7*h - i*L*h + L*h/2:
                    pos_cyl[k] = 0.2
                
            axs[i].plot(time_cyl, pos_cyl-1)
        axs[i].grid()
        axs[i].set_title(title.format(pos))
        axs[i].set_ylim(ylim)
    axs[i].legend()
    plt.suptitle("Evolution of Cp with time (s) at different positions on the wall")
    plt.tight_layout()

re_values = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
h_values = np.round(np.linspace(1,5,10),2)
h_values_extended = [1, 1.44, 1.89, 2.33, 2.78, 3.22, 3.67, 4.11, 4.56, 5, 6, 7, 8, 9, 10]
main_folder_path = "E:/simuChina/cartesianBDD_original"
re_value = 1000  # Example Re value
h_value = 30  # Example h value [6, 8, 11, 13, 16, 19, 22, 24, 27, 30]
re_1000_subfolders = find_subfolders_with_parameters(main_folder_path, re_value, None)
# temp = re_1000_subfolders[0]
# re_1000_subfolders.pop(0)
# re_1000_subfolders.append(temp)
print(re_1000_subfolders)
h_6_subfolders = find_subfolders_with_parameters(main_folder_path, None, h_value)
print(h_6_subfolders)

# Function to plot minimum pressure values for each experiment
def plot_min_pressure(data_list, time_list, positions, title, x):
    fig, axs = plt.subplots(4, 6, figsize=(18, 12))
    axs = axs.flatten()
    for i, pos in enumerate(positions):
        min_pressures = []
        for j in range(len(data_list)):
            min_pressure = np.max(data_list[j][pos][4:]) - np.min(data_list[j][pos][4:])
            min_pressures.append(min_pressure)
        axs[i].semilogx(x, np.log10(np.abs(min_pressures)), marker='o', linestyle='-')
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

# Function to plot time instances when minimum pressure is reached for each experiment on a single graph with a fixed h
def plot_time_min_pressure_single_fixedH(data_list, time_list, positions, x):
    plt.rc('font', size=20)
    plt.rc('legend', fontsize=16)
    fig = plt.figure()
    min_pressure_first_point = []
    for i, pos in enumerate(positions):
            min_pressure_times = []
            for j in range(len(data_list)):
                min_pressure_time = time_list[j][np.argmin(data_list[j][pos][4:])]
                min_pressure_times.append(min_pressure_time)
            plt.plot(np.log10(x), np.log10(min_pressure_times), marker='o', linestyle='-', label=f"{i+1}")
            min_pressure_first_point.append(np.log10(min_pressure_times[0]))
    plt.legend()
    plt.xlabel(r"$Log(Re)$")
    plt.ylabel(r"$Log(argmin(C_p))$")
    plt.title(r"$Log(argmin(C_p))$ with $Log(Re)$"+ f" at different sensing position along the wall with h={h_value}")
    print("mean of the first point: ", np.mean(min_pressure_first_point))

# Function to plot minimum pressure values for each experiment on a single graph with a fixed Re
def plot_min_pressure_single_fixedRe(data_list, time_list, positions, x):
    plt.rc('font', size=20)
    plt.rc('legend', fontsize=16)
    fig = plt.figure()
    for i, pos in enumerate(positions):
        min_pressures = []
        for j in range(1,len(data_list)):
            filtered_signal = apply_high_pass_filter(data_list[j][pos][4:], fs=1/(time_list[j][1] - time_list[j][0]), cutoff_freq=0.004*re_value/1000)
            min_pressure = np.max(filtered_signal) - np.min(filtered_signal)
            min_pressures.append(min_pressure)
        if len(min_pressures) > len(x):
            x = h_values_extended
        plt.plot(np.log10(x[1:]), np.log10(np.power(np.abs(min_pressures),-4)), marker='o', linestyle='-', label=f"{i+1}")
    plt.legend()
    plt.xlabel(r"$Log(h)$")
    plt.ylabel(r"$Log(\frac{1}{(max(C_p)-min(C_p))^4})$")
    plt.title(r"$Log(\frac{1}{(max(C_p)-min(C_p))^4})$ with $log(h)$"+f" at different sensing position along the wall with Re={re_value}")

# # Function to plot the drag frequency for each experiment on a single graph
# def plot_drag(subfolders):
#     fig = plt.figure()
#     for i, folder in enumerate(subfolders):
#         file_path = os.path.join(folder, "Motion.csv")
#         Re = extract_re_from_file_path(file_path)
#         _nu = L / Re
#         df = pd.read_csv(file_path, header=None)
#         time = df[0] * h * h * _nu / nu

#         plt.plot(time, np.mean(np.array(df[2]))*np.ones(time.shape), label=f"{i+1}")

        # lift_fft = fft(np.array(df[2]))
        # N = len(df[2])
        # xf = fftfreq(N, df[0][1] - df[0][0])[:N//2]
        # plt.plot(xf, 2.0/N * np.abs(lift_fft[0:N//2]))
    # plt.legend()

# # Function to compute the correlation between signals of differents sensors
# def compute_correlation(data_list, time_list, positions, x, ref):
#     fig, axs = plt.subplots(4, 6, figsize=(18, 12))
#     axs = axs.flatten()
#     all_correlations = []
#     for i, pos in enumerate(positions):
#         correlations = []
#         for j in range(len(data_list)):
#             correlation_coefficient = np.corrcoef(data_list[j][pos][4:], data_list[j][ref][4:])[0,1]
#             correlations.append(correlation_coefficient)
#         if len(correlations) > len(x):
#             x = h_values_extended
#         all_correlations.append(correlations)
#         axs[i].plot(x, np.log2(np.power(np.array(correlations)+0.6,-1)), marker='o', linestyle='-')
#         axs[i].plot(x,[y*2.87/10 for y in x])
#         axs[i].grid()
#     plt.xlabel("h")
#     plt.suptitle(f"Evolution of ln(1/(correlation+0.6)) coefficient between sensors and last sensor for varying h for Re={re_value}")

    # fig = plt.figure()
    # for j in range(len(data_list)):
    #     plt.plot(positions, [sensor[j] for sensor in all_correlations], marker='o', linestyle='-', label=f"{j+1}")
    # plt.legend()

# Lists to store data from all experiments, with a fixed h
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

# # View data on the cylinder
# cylinder_positions = range(16)
# plot_cp(all_cylinder_data, all_times, cylinder_positions, "Cp on the cylinder at θ=π+{}π/8", (-2.25, 1.5))
# mean_cps = [[np.mean(cylinder_data[i][4:]) for i in range(len(cylinder_data.columns))] for cylinder_data in all_cylinder_data]
# plot_mean_cp(np.linspace(180, 540, 17), mean_cps)

wall_positions = range(21)
# plot_wall_cp(all_wall_data, all_times, wall_positions, "Cp on the wall at n/2+{}L"+f"[h={h_value}]", (-1, 0.7), False)
plot_time_min_pressure_single_fixedH(all_wall_data, all_times, wall_positions, re_values)
# plt.show()

# Lists to store data from all experiments, with a fixed Re
all_cylinder_data = []
all_wall_data = []
all_times = []
file_path2 = None

for folder in re_1000_subfolders:
    file_path = os.path.join(folder, "pressureMotion.txt")
    cylinder_data, wall_data, time = read_cp_data(file_path)
    all_cylinder_data.append(cylinder_data)
    all_wall_data.append(wall_data)
    all_times.append(time)

wall_positions = range(21)
# plot_wall_cp(all_wall_data, all_times, wall_positions, "Cp on the wall at n/2+{}L"+f"[Re={re_value}]", (-1, 0.75), True)
# plot_min_pressure(all_wall_data, all_times, wall_positions, "Min Cp on the wall at n+{}L [Re]", h_values)
plot_min_pressure_single_fixedRe(all_wall_data, all_times, wall_positions, h_values)
# plot_time_min_pressure_single_fixedH(all_wall_data, all_times, wall_positions, h_values)
# plot_drag(re_1000_subfolders)
# compute_correlation(all_wall_data, all_times, wall_positions, h_values, 20)
plt.show()

def plot_affine_regression(data_list, time_list, positions, x, r, show):
    slopes = []  # List to store slopes from linear regressions
    intercepts = []  # List to store intercepts from linear regressions
    
    fig = plt.figure()
    
    for i, pos in enumerate(positions):
        min_pressures = []
        for j in range(len(data_list)):
            # Apply high-pass filter to the signal
            filtered_signal = apply_high_pass_filter(data_list[j][pos][4:], fs=1/(time_list[j][1] - time_list[j][0]), cutoff_freq=0.004*r/1000)
            min_pressure = np.max(filtered_signal) - np.min(filtered_signal)
            min_pressures.append(min_pressure)
        
        # Compute log-transformed values
        if len(x) < len(min_pressures):
            x = h_values_extended
        log_h = np.log10(x)
        log_p = np.log10(np.power(np.abs(min_pressures), -4))# - np.min(np.log10(np.power(np.abs(min_pressures), -4)))
        
        # Perform linear regression (affine)
        slope, intercept, r_value, p_value, std_err = linregress(log_h, log_p)
        slopes.append(slope)  # Store the slope
        intercepts.append(intercept)  # Store the intercept
        
        # Calculate confidence intervals for the slope and intercept
        confidence_level = 0.99
        alpha = 1 - confidence_level
        n = len(log_h)
        se_slope = std_err
        se_intercept = std_err * np.sqrt(np.mean(log_h**2))
        t_value = np.abs(t.ppf(alpha/2, n - 2))  # Use t-distribution to get t-value
        
        slope_ci = (slope - t_value * se_slope, slope + t_value * se_slope)
        intercept_ci = (intercept - t_value * se_intercept, intercept + t_value * se_intercept)
        
        # Plot log-transformed data and fitted line with uncertainty intervals
        plt.plot(log_h, log_p, marker='o', linestyle='-', label=f"Position {i+1}")
        # plt.plot(log_h, slope * log_h + intercept, color='red', label='Fitted line')
        # plt.fill_between(log_h, (slope_ci[0] * log_h + intercept_ci[0]), (slope_ci[1] * log_h + intercept_ci[1]),
        #                  color='gray', alpha=0.2, label='Confidence interval')
    
    # Calculate mean slope and intercept
    mean_slope = np.mean(slopes)
    mean_intercept = np.mean(intercepts)
    
    # Calculate uncertainty interval for the mean slope
    se_mean_slope = np.std(slopes) / np.sqrt(len(slopes))
    t_value_mean = np.abs(t.ppf(alpha/2, len(slopes) - 1))
    mean_slope_ci = (mean_slope - t_value_mean * se_mean_slope, mean_slope + t_value_mean * se_mean_slope)
    
    # Plot mean regression line with uncertainty interval
    plt.plot(log_h, mean_slope * log_h + mean_intercept, color='blue', linestyle='--', label='Mean Regression')
    plt.fill_between(log_h, (mean_slope_ci[0] * log_h + mean_intercept), (mean_slope_ci[1] * log_h + mean_intercept),
                     color='lightblue', alpha=0.4, label='Mean Slope Confidence Interval')
    
    plt.legend()
    plt.xlabel("Log(h)")
    plt.ylabel("Log(1/(P_max - P_min)^4)")
    plt.title(f"Log(1/(P_max - P_min)^4) vs. Log(h) at different positions along the wall with Re={re_value}")
    plt.grid(True)
    if show:
        plt.show()
    
    return mean_slope, mean_slope_ci, slopes, mean_intercept


# mean_slop, _, _, mean_intercept = plot_affine_regression(all_wall_data, all_times, wall_positions, h_values, re_value, True)
# print(mean_slop, mean_intercept)