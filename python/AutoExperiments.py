import glob
import os
import re
import socket
import subprocess
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Exponentiation


input_folder = "D:/simuChina"
file_name = "pressureMotion.txt"

# Parameters of the simulation
n = 2**7
h = 1/n
nu = 1/1000000
L = n/20
wall_sensors = range(21)

# Create the whole space of X
re_values = np.linspace(start=1000, stop=10000, num=10)
h_values = np.linspace(start=6, stop=60, num=60)
X1, X2 = np.meshgrid(re_values, h_values)
X = np.column_stack([X1.ravel(), X2.ravel()])

# Create theorical Y values
theorical_y1 = -np.log10(re_values)+3.71
theorical_y2 = 7.69*np.log10(h_values/6)-0.87
theorical_Y1, theorical_Y2 = np.meshgrid(theorical_y1, theorical_y2)
theorical_Y = np.column_stack([theorical_Y1.ravel(), theorical_Y2.ravel()])


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
    pattern = r"Re(\d+)_h(\d+)"

    # Search for the pattern in the file path
    match = re.search(pattern, file_path)

    # If a match is found, extract and return the Re value
    if match:
        return int(match.group(2))
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

def compute_Re(data, time, positions):
    min_pressure_points = []
    
    for i, pos in enumerate(positions):
        min_pressure_time = time[np.argmin(data[pos][4:])]
        min_pressure_points.append(np.log10(min_pressure_time))
    
    return np.mean(min_pressure_points)

def compute_h(data, time, positions, file_path):
    Re = extract_re_from_file_path(file_path)
    amp_pressures = []

    for i, pos in enumerate(positions):
        filtered_signal = apply_high_pass_filter(data[pos][4:], fs=1/(time[1] - time[0]), cutoff_freq=0.004*Re/1000)
        amp_pressure = np.max(filtered_signal) - np.min(filtered_signal)
        amp_pressures.append(np.log10(np.power(np.abs(amp_pressure), -4)))

    return np.mean(amp_pressures)

def find_newest_exp_path(folder):
    # Find the newest experiment folder and select the pressure file
    new_exp_sub_folder = max(glob.glob(os.path.join(folder, "*/")), key=os.path.getmtime)
    new_exp_file_path = os.path.join(new_exp_sub_folder, file_name)

    return new_exp_file_path

def cp_txt2metrics_csv(file_path, metric_file):
    # Read the pressure file and extract pressure and time
    wall_data, time_data = read_cp_data(file_path)
    re = extract_re_from_file_path(file_path)
    h = extract_h_from_file_path(file_path)

    # Compute the metrics related to the pressure which are linked to Re and h
    f_re = compute_Re(wall_data, time_data, wall_sensors)
    f_h = compute_h(wall_data, time_data, wall_sensors, file_path)

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
    next_re, next_h = None, None

    # Create the Y element for the GPR y reading the metric_file
    if not is_empty_csv(metric_file):
        df = pd.read_csv(metric_file)
        x1_train = df['Re']
        x2_train = df['h']
        y1 = df['f_re']
        y2 = df['f_h']
        # X1_train, X2_train = np.meshgrid(x1_train, x2_train)
        # Y1, Y2 = np.meshgrid(y1, y2)
        # X_train = np.column_stack([X1_train.ravel(), X2_train.ravel()])
        # Y = np.column_stack([Y1.ravel(), Y2.ravel()])
        X_train = np.array([[x1_train[i], x2_train[i]] for i in range(len(x1_train))])
        Y = np.array([[y1[i], y2[i]] for i in range(len(y1))])

    else:
        return "No metrics to use, please do a few experiments first"
    
    # Create the Gaussian process model
    # kernel = RBF(length_scale=np.array([1000, 1]), length_scale_bounds=(1, 10))
    kernel = Exponentiation(1 * RBF(length_scale=np.array([1000, 1]), length_scale_bounds=(1, 10)), exponent=4)
    gaussian_process = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=9
    )
    gaussian_process.fit(X_train, Y)
    mean_prediction, std_prediction = gaussian_process.predict(X, return_std=True)
    
    # Select the most uncertain point
    new_index = np.argmax([np.linalg.norm(s) for s in std_prediction])
    next_re, next_h = X[new_index]

    return int(next_re), np.round(next_h,2), std_prediction, mean_prediction, X_train, Y

def create_metric_file(name):
    df = pd.DataFrame()
    df.to_csv(name, index=False)
    cp_txt2metrics_csv(os.path.join(input_folder,"cartesianBDD/Re1000_h6/"+file_name), input_folder+'/metric_test.csv')
    cp_txt2metrics_csv(os.path.join(input_folder,"cartesianBDD/Re1000_h60/"+file_name), input_folder+'/metric_test.csv')
    cp_txt2metrics_csv(os.path.join(input_folder,"cartesianBDD/Re10000_h60/"+file_name), input_folder+'/metric_test.csv')
    cp_txt2metrics_csv(os.path.join(input_folder,"cartesianBDD/Re10000_h6/"+file_name), input_folder+'/metric_test.csv')

def write_new_parameters(param_file_path, re, h):
    df = pd.DataFrame([(re, h)], columns=['Re', 'h'])
    df.to_csv(param_file_path, index=False)

def run_lilypad_simulation():
    n = 0
    try:
        subprocess.run([
            'C:/Users/blagn771/Desktop/software/processing-4.2/processing-java.exe', 
            '--sketch=C:/Users/blagn771/Documents/Aquaman/Aquaman/lily-pad-master/LilyPad',
            '--run'
        ], check=True, stderr=subprocess.PIPE, timeout=3)
    except subprocess.CalledProcessError as e:
        # Check if the error message indicates a port conflict
        if b'Address already in use' in e.stderr:
            print("Error: Address already in use. Restarting the subprocess with a different configuration...")
            # Handle the port conflict by choosing a different port and restart
            restart_lilypad_simulation(n)
        else:
            # Handle other subprocess errors
            print(f"Subprocess error: {e}")
            run_lilypad_simulation()

    except subprocess.TimeoutExpired as e:
        print(f"Timeout of 5min as expired")

def find_available_port():
    """Find an available port by binding to a socket and then releasing it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('localhost', 0))  # Bind to a random available port
        return s.getsockname()[1]  # Return the assigned port
    
def restart_lilypad_simulation(n):
    n += 1
    print(f"Restarting Lilypad simulation with a different configuration... {n}/n")
    try:
        # Find an available port dynamically
        port = find_available_port()
        print(f"Using port: {port}")

        # Construct subprocess command with dynamic port
        subprocess.run([
            'C:/Users/blagn771/Desktop/software/processing-4.2/processing-java.exe', 
            '--sketch=C:/Users/blagn771/Documents/Aquaman/Aquaman/lily-pad-master/LilyPad',
            f'--run --port={port}'
        ], check=True, stderr=subprocess.PIPE, timeout=300)

    except subprocess.CalledProcessError as e:
        # Check if the error message indicates a port conflict
        if b'Address already in use' in e.stderr:
            print("Error: Address already in use. Restarting the subprocess with a different port...")
            restart_lilypad_simulation(n)  # Restart with a new port
        else:
            # Handle other subprocess errors
            print(f"Subprocess error: {e}")

    except subprocess.TimeoutExpired as e:
        print(f"Timeout of 5min as expired")

def run_autoexperiments(iteration, threshold=0.01, debug=False):
    count = 0
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(211, projection='3d')
    ax2 = fig.add_subplot(212, projection='3d')

    while count < iteration:
        ax.clear()
        ax2.clear()
        count += 1
        print("iteration n⁰: ", count)
        
        # Run a Lilypad simulation
        run_lilypad_simulation()

        # Get the lattest exp subfolder
        new_subfolder = find_newest_exp_path(input_folder)
        new_input_folder = os.path.join(input_folder, new_subfolder)

        # Compute the metrics
        cp_txt2metrics_csv(new_input_folder, input_folder+'/metric_test.csv')

        # Run a new GPR with all the existing values to find the next exp
        new_re, new_h, std_predictions, mean_prediction, X_train, Y = find_next_exp_param(metric_file=input_folder+"/metric_test.csv")

        # Check if the max std_prediction is under a given threshold
        if np.max(std_predictions) < threshold:
            break

        # Rewrite the file for the next parameters
        write_new_parameters(input_folder+'/metric_test_next_param.csv', new_re, new_h)

        ax.set_xlabel('log(Re)')
        ax.set_ylabel('h (in pixels)')
        ax.set_zlabel('f(Re, h)')

        ax2.set_xlabel('log(Re)')
        ax2.set_ylabel('h (in pixels)')
        ax2.set_zlabel('f(Re, h)')

        ax.scatter(np.log10(X_train[:, 0]), X_train[:, 1], Y[:, 0], color='blue', label='Observations')
        ax.plot_trisurf(np.log10(X[:, 0]), X[:, 1], mean_prediction[:, 0], linewidth=0.2, antialiased=True, cmap='viridis', alpha=0.5, label='Mean prediction')
        ax.plot_surface(np.log10(X1), X2, theorical_Y[:, 0].reshape(X1.shape), color='tab:orange', alpha=0.3, label='Real surface')  # Adding real surface

        ax2.scatter(np.log10(X_train[:, 0]), X_train[:, 1], Y[:, 1], color='blue', label='Observations')
        ax2.plot_trisurf(np.log10(X[:, 0]), X[:, 1], mean_prediction[:, 1], linewidth=0.2, antialiased=True, cmap='viridis', alpha=0.5, label='Mean prediction')
        ax2.plot_surface(np.log10(X1), X2, theorical_Y[:, 1].reshape(X1.shape), color='tab:orange', alpha=0.3, label='Real surface')  # Adding real surface

        plt.tight_layout() 
        if debug:
            plt.show()
        else:
            plt.savefig(input_folder+"/gpr_plots/"+str(1000+count)[1:]+".png")
            plt.pause(0.1)

        if debug:
            cont = input("Do you want to continue? (y/n): ")
            if cont.lower() != 'y':
                break


create_metric_file(input_folder+"/metric_test.csv")
new_re, new_h, _, _, _, _ = find_next_exp_param(metric_file=input_folder+"/metric_test.csv")
write_new_parameters(input_folder+'/metric_test_next_param.csv', new_re, new_h)

run_autoexperiments(100, debug=False)
run_autoexperiments(1, debug=True)
