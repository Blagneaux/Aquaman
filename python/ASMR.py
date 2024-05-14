# -----------------------------------------------------------------------------------------------

# ASMR (Automatic Scenarization of Multi-fidelity Representation)

# -----------------------------------------------------------------------------------------------


import glob
import os
import queue
import re
import subprocess
import threading
import time
from matplotlib import pyplot as plt
import nidaqmx
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from readSensors import SensorReading


input_folder_simu = "E:/simuChina"
input_folder_exp = "E:/sensorExpChina/wall"
file_name_simu = "pressureMotion.txt"
file_name_exp = "pressureMotion.xlsx"

# Sensors
serial_port = None
data_buffer = []
task = nidaqmx.Task()
# task.ai_channels.add_ai_voltage_chan("Dev2/ai0:5")
# sensors_reader = SensorReading(task)

# Parameters of the simulation
n = 2**7
h = 1/n
nu = 1/1000000
L = n/20
wall_sensors = range(21)
wall_sensors_exp = range(8)

# Initialize a queue to store the result from run_lilypad_simulation
result_queue = queue.Queue()

# Create the whole space of X
re_values = np.linspace(start=1000, stop=10000, num=40)
h_values = np.linspace(start=6, stop=60, num=550)
X1, X2 = np.meshgrid(re_values, h_values)
X = np.column_stack([X1.ravel(), X2.ravel()])

# Create theorical Y values
theorical_y1 = -np.log10(re_values)+3.55
theorical_y2 = 7.69*np.log10(h_values/6)-0.87
theorical_Y1, theorical_Y2 = np.meshgrid(theorical_y1, theorical_y2)
theorical_Y = np.column_stack([theorical_Y1.ravel(), theorical_Y2.ravel()])

rng = np.random.RandomState(1)
training_indices = [0, len(re_values)-1, len(re_values)*(len(h_values)-1), len(re_values)*len(h_values)-1]

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

def extract_h_from_file_path(file_path):
    # Define a regular expression pattern to match the Re value
    pattern = r"Re(\d+)_h(\d+\.\d+)"

    # Search for the pattern in the file path
    match = re.search(pattern, file_path)

    # If a match is found, extract and return the Re value
    if match:
        return float(match.group(2))
    else:
        return None  # Return None if no match is found

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

def is_empty_csv(file_path):
    with open(file_path, 'r') as file:
        first_line = file.readline()
        return len(first_line.strip()) == 0
    
def read_cp_data(file_path):
    data = []

    Re = extract_re_from_file_path(file_path)
    _nu = L / Re

    with open(file_path, 'r') as pressure:
        for line in pressure:
            data.append(line)

    cp = data[4:]
    wall_cp, time = [], []
    for i in range(len(cp)):
        to_add = cp[i].split()
        if to_add[-1] == ';':
            to_add.pop()
        to_add = np.array(to_add)
        to_add = to_add.astype(np.float32)
        if i % 2 == 0:
            time.append(to_add[0] * h * h *_nu / nu)
        else:
            wall_cp.append(to_add)
    wall_cp_dataframe = pd.DataFrame(wall_cp)

    return wall_cp_dataframe, time

def read_pressure_data(file_path):
    wall_data, time = [], []

    df = pd.read_excel(file_path, header=None)
    raw_data = df.iloc[:, :-1]
    raw_time = df.iloc[:, -1]

    # Adapt data to Pascals
    min_binary_value = -32768
    max_binary_value = 32768
    min_pressure_value = -500
    max_pressure_value = 500
    raw_data = raw_data - raw_data.mean()
    scaled_data = (raw_data - min_binary_value) * (max_pressure_value - min_pressure_value) / (max_binary_value - min_binary_value) + min_pressure_value
    for i in range(len(df[0])-1):
        wall_data.append(scaled_data.iloc[i, :])
        time.append(i/500)
    
    wall_pressure_dataframe = pd.DataFrame(wall_data)

    return wall_pressure_dataframe, time

def compute_Re(data, time, positions):
    min_pressure_points = []
    
    for i, pos in enumerate(positions):
        min_pressure_time = time[np.argmin(data[pos][4:])]
        min_pressure_points.append(np.log10(min_pressure_time))
    
    return np.mean(min_pressure_points)

def compute_Re_exp(data, time, positions):
    min_pressure_points = []

    for i, pos in enumerate(positions):
        min_pressure_time = time[np.argmin(data[pos])]
        min_pressure_points.append(np.log10(min_pressure_time))

    return np.max([np.mean(min_pressure_points), -1])

def compute_h(data, time, positions, file_path):
    Re = extract_re_from_file_path(file_path)
    amp_pressures = []

    for i, pos in enumerate(positions):
        filtered_signal = apply_high_pass_filter(data[pos][4:], fs=1/(time[1] - time[0]), cutoff_freq=0.004*Re/1000)
        amp_pressure = np.max(filtered_signal) - np.min(filtered_signal)
        amp_pressures.append(np.log10(np.power(np.abs(amp_pressure), -4)))

    return np.mean(amp_pressures)

def compute_h_exp(data, time, positions, file_path):
    amp_pressures = []

    for i, pos in enumerate(positions):
        amp_pressure = np.max(data[pos]) - np.min(data[pos])
        amp_pressures.append(np.log10(np.power(np.abs(amp_pressure), -4)))

    return np.mean(amp_pressures)

def find_newest_exp_path(folder, source):
    if source == "simu":
        file_name = file_name_simu
    elif source == "exp":
        file_name = file_name_exp
    # Find the newest experiment folder and select the pressure file
    new_exp_sub_folder = max(glob.glob(os.path.join(folder, "*/")), key=os.path.getmtime)
    new_exp_file_path = os.path.join(new_exp_sub_folder, file_name)

    return new_exp_file_path

def cp_txt2metrics_csv(file_path, metric_file, source):
    print(file_path)

    # Read the pressure file and extract pressure and time
    if source == "simu":
        wall_data, time_data = read_cp_data(file_path)
    elif source == "exp":
        wall_data, time_data = read_pressure_data(file_path)
    
    re = extract_re_from_file_path(file_path)
    h = extract_h_from_file_path(file_path)

    # Compute the metrics related to the pressure which are linked to Re and h
    if source == "simu":
        f_re = compute_Re(wall_data, time_data, wall_sensors)
        f_h = compute_h(wall_data, time_data, wall_sensors, file_path)
    elif source == "exp":
        f_re = compute_Re_exp(wall_data, time_data, wall_sensors_exp)
        f_h = compute_h_exp(wall_data, time_data, wall_sensors_exp, file_path)

    # Add the metrics to the metric_file
    new_data = pd.DataFrame([(re, h, f_re, f_h)], columns=['Re', 'h', 'f_re', 'f_h'])
    if is_empty_csv(metric_file):
        df = new_data
    else:
        df = pd.read_csv(metric_file)
        df = pd.concat([df, new_data], ignore_index=True)

    # Write the new metric_file
    df.to_csv(metric_file, index=False)

def find_next_exp_param(metric_file):
    metric_file_simu = input_folder_simu+metric_file
    metric_file_exp = input_folder_exp+metric_file
    next_re, next_h = None, None

    # Create the Y element for the GPR y reading the metric_file
    if not is_empty_csv(metric_file_simu) and not is_empty_csv(metric_file_exp):
        df = pd.read_csv(metric_file_simu)
        x1_train = df['Re']
        x2_train = df['h']
        y1 = df['f_re']
        y2 = df['f_h']
        X_train = np.array([[x1_train[i], x2_train[i]] for i in range(len(x1_train))])
        Y = np.array([[y1[i], y2[i]] for i in range(len(y1))])

        print("Also uncomment when wall sensors are on")
        # df_exp = pd.read_csv(metric_file_exp)
        # x1_train_exp = df_exp['Re']
        # x2_train_exp = df_exp['h']
        # y1_exp = df_exp['f_re']
        # y2_exp = df_exp['f_h']
        # X_train_exp = np.array([[x1_train_exp[i], x2_train_exp[i]] for i in range(len(x1_train_exp))])
        # Y_exp = np.array([[y1_exp[i], y2_exp[i]] for i in range(len(y1_exp))])

    else:
        return "No metrics to use, please do a few experiments first"
    
    # Create the Gaussian process model based on the simulation
    kernel = RBF(length_scale=np.array([1000, 1]), length_scale_bounds=[(10, 1e6), (1e-2, 1e3)])
    gaussian_process = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=9
    )
    gaussian_process.fit(X_train, Y)
    mean_prediction, std_prediction = gaussian_process.predict(X, return_std=True)

    # Create the Gaussian process model based on the simulation and noise
    kernel2 = RBF(length_scale=np.array([1000, 1]), length_scale_bounds=[(10, 1e6), (1e-2, 1e3)])
    gaussian_process2 = GaussianProcessRegressor(
        kernel=kernel2, n_restarts_optimizer=9
    )
    print("And here too")
    # gaussian_process2.fit(X_train_exp, Y_exp)
    # mean_prediction2, std_prediction2 = gaussian_process2.predict(X, return_std=True)
    std_prediction2 = std_prediction*0
    mean_prediction2, X_train_exp, Y_exp = 0, 0, 0
    
    # Select the most uncertain point
    listDiff = [np.linalg.norm(np.abs(s1-s2)) for s1, s2 in zip(std_prediction, std_prediction2)]
    new_index = np.argmax(listDiff)
    while new_index in training_indices:
        print("New index is already in the list, finding the second most uncertain value...")
        listDiff[np.argmax(listDiff)] = -1
        new_index = np.argmax(listDiff)

    print(len(training_indices), training_indices)
    training_indices.append(new_index)
    next_re, next_h = X[new_index]

    return int(next_re), np.round(next_h,2), std_prediction, mean_prediction, X_train, Y, mean_prediction2, X_train_exp, Y_exp

def create_metric_file(name, fromZero=False):
    df = pd.DataFrame()
    df.to_csv(name, index=False)
    cp_txt2metrics_csv(os.path.join(input_folder_simu,"cartesianBDD/Re1000_h6.0/"+file_name_simu), input_folder_simu+'/metric_test.csv', "simu")
    cp_txt2metrics_csv(os.path.join(input_folder_simu,"cartesianBDD/Re1000_h60.0/"+file_name_simu), input_folder_simu+'/metric_test.csv', "simu")
    cp_txt2metrics_csv(os.path.join(input_folder_simu,"cartesianBDD/Re10000_h60.0/"+file_name_simu), input_folder_simu+'/metric_test.csv', "simu")
    cp_txt2metrics_csv(os.path.join(input_folder_simu,"cartesianBDD/Re10000_h6.0/"+file_name_simu), input_folder_simu+'/metric_test.csv', "simu")

    countFiles = 0
    if not fromZero:
        # for folder in os.listdir(input_folder_simu):
        #     if folder[0:2] == "Re" and folder[2] != "a":
        #         countFiles += 1
        #         print("Already existing file n⁰: ", countFiles)
        #         cp_txt2metrics_csv(os.path.join(input_folder_simu,folder+"/"+file_name), input_folder_simu+'/metric_test.csv')
        return "To modify to include the exp data"

def write_new_parameters(param_file_path, re, h):
    df = pd.DataFrame([(re, h)], columns=['Re', 'h'])
    df.to_csv(param_file_path, index=False)

    traj_file_path = "E:/sensorExpChina/trajectory.csv"
    tank_width = 1318       # in mm
    wall_width = 315        # in mm
    h_max = tank_width / 2 - wall_width

    x_0 = 1000          # in mm
    dist = 1000         # in mm
    speed = re * 1e-6 / 0.05    # in m/s
    Delta_t = dist / speed     # in ms
    delta_t = 1000          # in ms
    step = Delta_t // delta_t
    print(Delta_t)

    y_tank_pos = int(h_max - h*50/6)

    X_tank_pos, Y_tank_pos, T_tank, Z_tank_pos, C_tank_pos =[], [], [], [], []
    for i in range(int(step)+1):
        Y_tank_pos.append(y_tank_pos)
        T_tank.append(delta_t)
        X_tank_pos.append(x_0 + i*speed*delta_t)
        Z_tank_pos.append(0)
        C_tank_pos.append(-150)

    data_tank_traj = {'X': X_tank_pos,
                      'T': T_tank,
                      'Y': Y_tank_pos,
                      'T2': T_tank,
                      'X2': Z_tank_pos,
                      'T3': T_tank,
                      'C': C_tank_pos,
                      'T4': T_tank}
    df_traj = pd.DataFrame(data_tank_traj)
    df_traj.to_csv(traj_file_path, index=False, header=False)

class TimeoutExpired(Exception):
    pass

def run_with_timeout(command, timeout=3):
    process = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE, shell=False, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    timer = threading.Timer(timeout, lambda: subprocess.run(['taskkill', '/F', '/T', '/PID', str(process.pid)], shell=True))

    try:
        timer.start()
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command, stderr)
        return stdout, stderr
    except subprocess.CalledProcessError as e:
        raise e
    except Exception as e:
        raise e
    finally:
        timer.cancel()

def run_lilypad_simulation(queue, n=0, time=300):
    if n > 0:
        print(f"Retrying to run the process for the {n}th time")

    if n > 10:
        result = "Something went wrong"
        queue.put(result)

    try:
        stdout, stderr = run_with_timeout([
            'C:/Users/blagn771/Desktop/software/processing-4.2/processing-java.exe', 
            '--sketch=C:/Users/blagn771/Documents/Aquaman/Aquaman/lily-pad-master/LilyPad',
            '--run'
        ], timeout=time)

        # Process stdout and stderr if needed
        print(stdout.decode())
        print(stderr.decode())

    except subprocess.CalledProcessError as e:
        # Handle subprocess errors
        print(f"Subprocess error: {e}")
        print("-------------------------restarting---------------------------------")
        run_lilypad_simulation(queue, n+1)

    except TimeoutExpired:
        print("Timeout expired. Terminating the subprocess...")

    except Exception as e:
        print(f"An error occurred: {e}")

# ------------------------------------------------------------------------------------------
# Moved to readSensors.py
# ------------------------------------------------------------------------------------------
# def run_read_pressure_from_sensors():
#     reader = SerialReader()
#     next_param_df = pd.read_csv("E:/simuChina/metric_test_next_param.csv")
#     Re = next_param_df['Re'][0]
#     duration = 0.05*1e6/Re  # Duration of reading in seconds
#     h = next_param_df['h'][0]
#     reader.start_reading(duration, Re, h)

# def run_read_force_from_sensor():
#     force_reader = DataProcessor()
#     force_reader.read_data(task)
# ------------------------------------------------------------------------------------------

def run_read_from_sensors():
    sensors_reader.run_sensor_reading()

def run_tank():
    command = ["C:\\path_to_your_exe\\MyConsoleApp.exe"]

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    stdout, stderr = process. communicate()
    print("Output: ", stdout.decode())
    print("Error: ", stderr.decode())

def run_autoexperiments(iteration, already=0, threshold=0.01, debug=False):
    count = 0 + already
    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(231, projection='3d')
    ax2 = fig.add_subplot(234, projection='3d')
    ax3 = fig.add_subplot(232, projection='3d')
    ax4 = fig.add_subplot(235, projection='3d')
    ax5 = fig.add_subplot(233, projection='3d')
    ax6 = fig.add_subplot(236, projection='3d')

    while count < iteration+already:
        ax.clear()
        ax2.clear()
        ax3.clear()
        ax4.clear()
        ax5.clear()
        ax6.clear()
        count += 1
        print("iteration n⁰: ", count)

        # Run the towing tank experiment
        sensors_thread = threading.Thread(target=run_tank)
        
        # Run a Lilypad simulation
        lilypad_thread = threading.Thread(target=run_lilypad_simulation, args=(result_queue,))

        # Start threads
        sensors_thread.start()
        lilypad_thread.start()

        # Wait for all threads to finish before rocessing to the next iteration
        sensors_thread.join()
        lilypad_thread.join()

        # Check if any error occurred in any of the threads
        if not result_queue.empty():
            e = result_queue.get()
            break

        # Wait for the water to calm down
        print("Going to sleep")
        time.sleep(30)
        print("Waking up")

        # Get the lattest simulation subfolder
        new_subfolder = find_newest_exp_path(input_folder_simu, "simu")
        new_input_folder_simu = os.path.join(input_folder_simu, new_subfolder)

        # Compute the metrics
        cp_txt2metrics_csv(new_input_folder_simu, input_folder_simu+'/metric_test.csv', "simu")

        # Get the lattest experiment subfolder
        print("To uncomment when wall sensors are on")
        # new_subfolder = find_newest_exp_path(input_folder_exp, "exp")
        # new_input_folder_exp = os.path.join(input_folder_exp, new_subfolder)

        # Compute the metrics
        # cp_txt2metrics_csv(new_input_folder_exp, input_folder_exp+'/metric_test.csv', "exp")

        # Run a new GPR with all the existing values to find the next exp
        new_re, new_h, std_predictions, mean_prediction, X_train, Y, mean_prediction_exp, X_train_exp, Y_exp = find_next_exp_param(metric_file="/metric_test.csv")

        # Check if the max std_prediction is under a given threshold
        if np.max(std_predictions) < threshold:
            break

        # Rewrite the file for the next parameters
        write_new_parameters(input_folder_simu+'/metric_test_next_param.csv', new_re, new_h)

        ax.set_xlabel('log(Re)')
        ax.set_ylabel('h (in pixels)')
        ax.set_zlabel('f(Re, h)')
        ax.set_title("Prediction VS theory for f(Re) simulated")

        ax2.set_xlabel('log(Re)')
        ax2.set_ylabel('h (in pixels)')
        ax2.set_zlabel('f(Re, h)')
        ax2.set_title("Prediction VS theory for f(h) simulated")

        ax3.set_xlabel('log(Re)')
        ax3.set_ylabel('h (in pixels)')
        ax3.set_zlabel('f(Re, h)')
        ax3.set_title("f(Re) simulated minus f(Re) measured")

        ax4.set_xlabel('log(Re)')
        ax4.set_ylabel('h (in pixels)')
        ax4.set_zlabel('f(Re, h)')
        ax4.set_title("f(h) simulated minus f(h) measured")

        ax5.set_xlabel('log(Re)')
        ax5.set_ylabel('h (in pixels)')
        ax5.set_zlabel('f(Re, h)')
        ax5.set_title("Prediction VS theory for f(Re) measured")

        ax6.set_xlabel('log(Re)')
        ax6.set_ylabel('h (in pixels)')
        ax6.set_zlabel('f(Re, h)')
        ax6.set_title("Prediction VS theory for f(h) measured")

        ax.scatter(np.log10(X_train[:, 0]), X_train[:, 1], Y[:, 0], color='blue', label='Observations')
        ax.plot_trisurf(np.log10(X[:, 0]), X[:, 1], mean_prediction[:, 0], linewidth=0.2, antialiased=True, cmap='viridis', alpha=0.5, label='Mean prediction')
        ax.plot_surface(np.log10(X1), X2, theorical_Y[:, 0].reshape(X1.shape), color='tab:orange', alpha=0.3, label='Real surface')  # Adding real surface
        
        # ax5.scatter(np.log10(X_train_exp[:, 0]), X_train_exp[:, 1], Y_exp[:, 0], color='blue', label='Observations')
        # ax5.plot_trisurf(np.log10(X[:, 0]), X[:, 1], mean_prediction_exp[:, 0], linewidth=0.2, antialiased=True, cmap='viridis', alpha=0.5, label='Mean prediction')
        # ax5.plot_surface(np.log10(X1), X2, theorical_Y[:, 0].reshape(X1.shape), color='tab:orange', alpha=0.3, label='Real surface')  # Adding real surface
        
        # ax3.scatter(np.log10(X_train_exp[:, 0]), X_train_exp[:, 1], Y[:, 0] - Y_exp[:, 0], color='blue', label='Observations')
        # ax3.plot_trisurf(np.log10(X[:, 0]), X[:, 1], [m1-m2 for m1,m2 in zip(mean_prediction[:,0], mean_prediction_exp[:,0])], linewidth=0.2, antialiased=True, cmap='viridis', alpha=0.5, label='Mean prediction')

        ax2.scatter(np.log10(X_train[:, 0]), X_train[:, 1], Y[:, 1], color='blue', label='Observations')
        ax2.plot_trisurf(np.log10(X[:, 0]), X[:, 1], mean_prediction[:, 1], linewidth=0.2, antialiased=True, cmap='viridis', alpha=0.5, label='Mean prediction')
        ax2.plot_surface(np.log10(X1), X2, theorical_Y[:, 1].reshape(X1.shape), color='tab:orange', alpha=0.3, label='Real surface')  # Adding real surface
        
        # ax6.scatter(np.log10(X_train_exp[:, 0]), X_train_exp[:, 1], Y_exp[:, 1], color='blue', label='Observations')
        # ax6.plot_trisurf(np.log10(X[:, 0]), X[:, 1], mean_prediction_exp[:, 1], linewidth=0.2, antialiased=True, cmap='viridis', alpha=0.5, label='Mean prediction')
        # ax6.plot_surface(np.log10(X1), X2, theorical_Y[:, 1].reshape(X1.shape), color='tab:orange', alpha=0.3, label='Real surface')  # Adding real surface
        
        # ax4.scatter(np.log10(X_train_exp[:, 0]), X_train_exp[:, 1], Y[:, 1] - Y_exp[:, 1], color='blue', label='Observations')
        # ax4.plot_trisurf(np.log10(X[:, 0]), X[:, 1], [m1-m2 for m1,m2 in zip(mean_prediction[:,1], mean_prediction_exp[:,1])], linewidth=0.2, antialiased=True, cmap='viridis', alpha=0.5, label='Mean prediction')

        plt.tight_layout() 
        if debug:
            plt.show()
        else:
            plt.savefig(input_folder_simu+"/gpr_plots/"+str(1000+count)[1:]+".png")
            plt.pause(0.1)

        if debug:
            cont = input("Do you want to continue? (y/n): ")
            if cont.lower() != 'y':
                break


create_metric_file(input_folder_simu+"/metric_test.csv", True)
new_re, new_h, _, _, _, _, _, _, _ = find_next_exp_param(metric_file="/metric_test.csv")
write_new_parameters(input_folder_simu+'/metric_test_next_param.csv', new_re, new_h)

run_autoexperiments(2, debug=False)
# run_autoexperiments(100, already=101, debug=False)
run_autoexperiments(1, debug=True)