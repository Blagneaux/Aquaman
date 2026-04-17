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
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


@dataclass
class PathsConfig:
    simu_root: Path = Path("E:/simuChina")
    exp_root: Path = Path("E:/sensorExpChina/wall")
    file_name_simu: str = "pressureMotion.txt"
    file_name_exp: str = "pressureMotion.xlsx"
    metric_name: str = "metric_test.csv"
    next_param_csv: Path = Path("E:/simuChina/metric_test_next_param.csv")
    gpr_plot_dir: Path = Path("E:/simuChina/gpr_plots")
    gcode_output_dir: Path = Path("E:/sensorExpChina/gcode")


@dataclass
class SensorConfig:
    wall_sensor_count_sim: int = 21
    wall_sensor_count_exp: Optional[int] = 8  # None uses all columns in the file
    cylinder_sensor_count_exp: Optional[int] = 8


@dataclass
class TankConfig:
    length_mm: float = 750.0  # New tank length (75 cm)
    width_mm: float = 690.0   # New tank width (69 cm)
    wall_width_mm: float = 14.0  # Wall thickness / clearance in mm
    y0_mm: float = 0.0  # Offset from machine origin to wall reference (see compute_tank_y_mm)
    h_mm_per_unit: float = 50.0 / 6.0
    x_start_mm: float = 0.0
    travel_margin_mm: float = 0.0
    accel_dist_mm: float = 125.0
    z_mm: float = 0.0
    c_mm: float = -155.0
    return_to_start: bool = False
    axis_x: str = "X"
    axis_y: str = "Y"
    axis_z: str = "Z"
    axis_c: str = "C"


@dataclass
class CNCjsConfig:
    cli: Sequence[str] = ("cncjs",)  # Change to ("cncjs-cli",) if needed
    port: Optional[str] = None
    baud: int = 115200
    controller: Optional[str] = None
    extra_args: Sequence[str] = ()  # Add CLI args required by your CNCjs install
    dry_run: bool = False


PATHS = PathsConfig()
SENSORS = SensorConfig()
TANK = TankConfig()
CNCJS = CNCjsConfig()
RUN_TANK = False  # Set True to enable towing tank motion
TRAJECTORY_MODE = "cncjs"  # "cncjs" or "csv"

# Example usage:
# 1) Set CNCJS.port (e.g., "COM5" or "/dev/ttyUSB0") and adjust CNCJS.baud/controller if needed.
# 2) Set RUN_TANK = True to enable tank motion.
# 3) Choose TRAJECTORY_MODE = "cncjs" (gcode + CNCjs) or "csv" (legacy trajectory file).
# 4) Run: python python/simulationCorrelationDTtooling/ASMR.py
# This will loop:
# - Run the Lilypad simulation in parallel with the tank motion.
# - Compute metrics on simulated vs measured pressure.
# - Fit a GPR and pick the next (Re, h) where the two models differ most.
# - Generate a new trapezoidal tow profile and repeat.

#             ┌──────────────────────────────┐
#             │   Select current (Re, h)     │
#             │   from GPR suggestion        │
#             └──────────────┬───────────────┘
#                            │
#         ┌──────────────────┴──────────────────┐
#         │                                     │
#         │  Parallel execution                 │
#         │                                     │
# ┌───────▼────────┐                    ┌───────▼────────┐
# │ Lilypad sim    │                    │ Tank run       │
# │ (pressure sim) │                    │ (CNCjs gcode)  │
# └───────┬────────┘                    └───────┬────────┘
#         │                                     │
#         └──────────────┬──────────────────────┘
#                        │
#             ┌──────────▼──────────┐
#             │ Compute metrics:    │
#             │ f_re, f_h (sim+exp) │
#             └──────────┬──────────┘
#                        │
#             ┌──────────▼──────────┐
#             │ Fit GPR models:     │
#             │ sim vs exp metrics  │
#             └──────────┬──────────┘
#                        │
#             ┌──────────▼──────────┐
#             │ Select next (Re, h) │
#             │ at max model diff   │
#             └──────────┬──────────┘
#                        │
#             ┌──────────▼──────────┐
#             │ Generate trapezoid  │
#             │ tow profile (gcode) │
#             └──────────┬──────────┘
#                        │
#                        └───────(loop)───────▶


# Parameters of the simulation
n = 2**7
h = 1 / n
nu = 1 / 1_000_000
L = n / 20

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

training_indices = [0, len(re_values)-1, len(re_values)*(len(h_values)-1), len(re_values)*len(h_values)-1]


def resolve_positions(dataframe, positions: Optional[Sequence[int]]):
    if positions is None:
        return range(dataframe.shape[1])
    return positions

def extract_re_from_file_path(file_path):
    file_path = str(file_path)
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
    file_path = str(file_path)
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
    file_path = Path(file_path)
    if not file_path.exists():
        return True
    if file_path.stat().st_size == 0:
        return True
    with file_path.open('r') as file:
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
    # raw_time column exists but timestamps are recomputed below

    # Adapt data to Pascals
    min_binary_value = -32768
    max_binary_value = 32768
    min_pressure_value = -500
    max_pressure_value = 500
    raw_data = raw_data - raw_data.mean()
    scaled_data = (raw_data - min_binary_value) * (max_pressure_value - min_pressure_value) / (max_binary_value - min_binary_value) + min_pressure_value
    for i in range(raw_data.shape[0]):
        wall_data.append(scaled_data.iloc[i, :])
        time.append(i / 500)
    
    wall_pressure_dataframe = pd.DataFrame(wall_data)

    return wall_pressure_dataframe, time

def compute_Re(data, time, positions=None):
    positions = resolve_positions(data, positions)
    min_pressure_points = []
    
    for i, pos in enumerate(positions):
        min_pressure_time = time[np.argmin(data[pos][4:])]
        min_pressure_points.append(np.log10(min_pressure_time))
    
    return np.mean(min_pressure_points)

def compute_Re_exp(data, time, positions=None):
    positions = resolve_positions(data, positions)
    min_pressure_points = []

    for i, pos in enumerate(positions):
        min_pressure_time = time[np.argmin(data[pos])]
        min_pressure_points.append(np.log10(min_pressure_time))

    return np.max([np.mean(min_pressure_points), -1])

def compute_h(data, time, positions=None, file_path=None):
    positions = resolve_positions(data, positions)
    Re = extract_re_from_file_path(file_path)
    amp_pressures = []

    for i, pos in enumerate(positions):
        filtered_signal = apply_high_pass_filter(data[pos][4:], fs=1/(time[1] - time[0]), cutoff_freq=0.004*Re/1000)
        amp_pressure = np.max(filtered_signal) - np.min(filtered_signal)
        amp_pressures.append(np.log10(np.power(np.abs(amp_pressure), -4)))

    return np.mean(amp_pressures)

def compute_h_exp(data, time, positions=None, file_path=None):
    positions = resolve_positions(data, positions)
    amp_pressures = []

    for i, pos in enumerate(positions):
        amp_pressure = np.max(data[pos]) - np.min(data[pos])
        amp_pressures.append(np.log10(np.power(np.abs(amp_pressure), -4)))

    return np.mean(amp_pressures)

def find_newest_exp_path(folder, source):
    if source == "simu":
        file_name = PATHS.file_name_simu
    elif source == "exp":
        file_name = PATHS.file_name_exp
    # Find the newest experiment folder and select the pressure file
    candidates = glob.glob(os.path.join(folder, "*/"))
    if not candidates:
        raise FileNotFoundError(f"No experiment folders found in {folder}")
    new_exp_sub_folder = max(candidates, key=os.path.getmtime)
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
        positions = range(SENSORS.wall_sensor_count_sim)
        f_re = compute_Re(wall_data, time_data, positions)
        f_h = compute_h(wall_data, time_data, positions, file_path)
    elif source == "exp":
        positions = None
        if SENSORS.wall_sensor_count_exp is not None:
            positions = range(SENSORS.wall_sensor_count_exp)
        f_re = compute_Re_exp(wall_data, time_data, positions)
        f_h = compute_h_exp(wall_data, time_data, positions, file_path)

    # Add the metrics to the metric_file
    new_data = pd.DataFrame([(re, h, f_re, f_h)], columns=['Re', 'h', 'f_re', 'f_h'])
    if is_empty_csv(metric_file):
        df = new_data
    else:
        df = pd.read_csv(metric_file)
        df = pd.concat([df, new_data], ignore_index=True)

    # Write the new metric_file
    df.to_csv(metric_file, index=False)

def find_next_exp_param(metric_file=None):
    metric_name = metric_file or PATHS.metric_name
    metric_name = str(metric_name).lstrip("/\\")
    metric_file_simu = PATHS.simu_root / metric_name
    metric_file_exp = PATHS.exp_root / metric_name
    next_re, next_h = None, None

    # Create the Y element for the GPR y reading the metric_file
    if is_empty_csv(metric_file_simu):
        return "No simulation metrics to use, please do a few simulations first"

    df = pd.read_csv(metric_file_simu)
    x1_train = df['Re']
    x2_train = df['h']
    y1 = df['f_re']
    y2 = df['f_h']
    X_train = np.array([[x1_train[i], x2_train[i]] for i in range(len(x1_train))])
    Y = np.array([[y1[i], y2[i]] for i in range(len(y1))])

    use_exp = not is_empty_csv(metric_file_exp)
    X_train_exp, Y_exp = None, None
    if use_exp:
        df_exp = pd.read_csv(metric_file_exp)
        x1_train_exp = df_exp['Re']
        x2_train_exp = df_exp['h']
        y1_exp = df_exp['f_re']
        y2_exp = df_exp['f_h']
        X_train_exp = np.array([[x1_train_exp[i], x2_train_exp[i]] for i in range(len(x1_train_exp))])
        Y_exp = np.array([[y1_exp[i], y2_exp[i]] for i in range(len(y1_exp))])
    
    # Create the Gaussian process model based on the simulation
    kernel = RBF(length_scale=np.array([1000, 1]), length_scale_bounds=[(10, 1e6), (1e-2, 1e3)])
    gaussian_process = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=9
    )
    gaussian_process.fit(X_train, Y)
    mean_prediction, std_prediction = gaussian_process.predict(X, return_std=True)

    # Create the Gaussian process model based on the simulation and noise
    mean_prediction2, std_prediction2 = None, None
    if use_exp:
        kernel2 = RBF(length_scale=np.array([1000, 1]), length_scale_bounds=[(10, 1e6), (1e-2, 1e3)])
        gaussian_process2 = GaussianProcessRegressor(
            kernel=kernel2, n_restarts_optimizer=9
        )
        gaussian_process2.fit(X_train_exp, Y_exp)
        mean_prediction2, std_prediction2 = gaussian_process2.predict(X, return_std=True)

    # Select the point of maximum disagreement between simulated and measured models
    if use_exp and mean_prediction2 is not None:
        listDiff = [np.linalg.norm(s1 - s2) for s1, s2 in zip(mean_prediction, mean_prediction2)]
    else:
        # Fallback: maximize uncertainty if no experimental model is available yet
        listDiff = [np.linalg.norm(s1) for s1 in std_prediction]
    new_index = np.argmax(listDiff)
    while new_index in training_indices:
        print("New index is already in the list, finding the second most uncertain value...")
        listDiff[np.argmax(listDiff)] = -1
        new_index = np.argmax(listDiff)

    print(len(training_indices), training_indices)
    training_indices.append(new_index)
    next_re, next_h = X[new_index]

    return int(next_re), np.round(next_h, 2), std_prediction, mean_prediction, X_train, Y, mean_prediction2, X_train_exp, Y_exp

def create_metric_file(name, fromZero=False):
    metric_path = Path(name)
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame()
    df.to_csv(metric_path, index=False)
    cp_txt2metrics_csv(PATHS.simu_root / "init/Re1000_h6.0" / PATHS.file_name_simu, metric_path, "simu")
    cp_txt2metrics_csv(PATHS.simu_root / "init/Re1000_h60.0" / PATHS.file_name_simu, metric_path, "simu")
    cp_txt2metrics_csv(PATHS.simu_root / "init/Re10000_h60.0" / PATHS.file_name_simu, metric_path, "simu")
    cp_txt2metrics_csv(PATHS.simu_root / "init/Re10000_h6.0" / PATHS.file_name_simu, metric_path, "simu")

    countFiles = 0
    if not fromZero:
        # for folder in os.listdir(PATHS.simu_root):
        #     if folder[0:2] == "Re" and folder[2] != "a":
        #         countFiles += 1
        #         print("Already existing file n⁰: ", countFiles)
        #         cp_txt2metrics_csv(PATHS.simu_root / folder / PATHS.file_name_simu, metric_path, "simu")
        return "To modify to include the exp data"

def write_new_parameters(param_file_path, re, h):
    param_path = Path(param_file_path)
    param_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([(re, h)], columns=['Re', 'h'])
    df.to_csv(param_path, index=False)

def write_new_traj_csv(re, h):
    base_dir = Path(PATHS.exp_root).parent
    traj_file_path = base_dir / "trajectory.csv"
    traj_dir = base_dir / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)
    traj_file_save_path = traj_dir / f"Re{re}_h{h}.csv"
    positions, dts = build_trapezoid_profile(re, h)
    y_tank_pos = compute_tank_y_mm(h)

    Y_tank_pos, T_tank, Z_tank_pos, C_tank_pos = [], [], [], []

    for dt in dts:
        Y_tank_pos.append(y_tank_pos)
        T_tank.append(dt * 1000)
        Z_tank_pos.append(TANK.z_mm)
        C_tank_pos.append(TANK.c_mm)

    # Positions list includes the starting point; drop it to align with dt steps
    X_tank_full = positions[1:]

    data_tank_traj = {
        'X': X_tank_full,
        'T': T_tank,
        'Y': Y_tank_pos,
        'T2': T_tank,
        'X2': Z_tank_pos,
        'T3': T_tank,
        'C': C_tank_pos,
        'T4': T_tank,
    }
    df_traj = pd.DataFrame(data_tank_traj)
    df_traj.to_csv(traj_file_path, index=False, header=False)
    df_traj.to_csv(traj_file_save_path, index=False, header=False)


def compute_speed_m_s(re):
    return re * 1e-6 / 0.05


def compute_speed_mm_s(re):
    return compute_speed_m_s(re) * 1000.0


def compute_tank_y_mm(h_value):
    # y0_mm shifts the origin so the commanded Y is referenced to the wall side of the tank.
    # h_value is the experimental "h" parameter scaled by h_mm_per_unit.
    h_max = TANK.width_mm / 2 - TANK.wall_width_mm
    return -(TANK.y0_mm + h_max - h_value * TANK.h_mm_per_unit)


def build_trapezoid_profile(re, h, cruise_steps=101):
    speed_mm_s = compute_speed_mm_s(re)
    if speed_mm_s <= 0:
        raise ValueError("Computed speed must be positive.")

    total_len = max(TANK.length_mm - 2 * TANK.travel_margin_mm, 0.0)
    accel_len = min(TANK.accel_dist_mm, total_len / 2.0)
    cruise_len = max(total_len - 2 * accel_len, 0.0)

    if total_len <= 0:
        raise ValueError("Total travel length must be positive.")

    # Constant-acceleration trapezoid
    accel = speed_mm_s * speed_mm_s / (2.0 * accel_len) if accel_len > 0 else 0.0
    t_accel = speed_mm_s / accel if accel > 0 else 0.0
    t_cruise = cruise_len / speed_mm_s if cruise_len > 0 else 0.0

    if cruise_len > 0 and t_cruise > 0:
        dt = t_cruise / cruise_steps
        n_accel = max(1, int(round(t_accel / dt)))
        n_cruise = cruise_steps
    else:
        n_accel = max(1, cruise_steps // 2)
        dt = t_accel / n_accel if t_accel > 0 else 0.01
        n_cruise = 0
    n_decel = n_accel

    positions = [0.0]
    dts = []

    # Acceleration
    for i in range(1, n_accel + 1):
        t = i * dt
        x = 0.5 * accel * t * t if accel > 0 else 0.0
        positions.append(x)
        dts.append(dt)

    # Cruise
    for i in range(1, n_cruise + 1):
        t = i * dt
        x = accel_len + speed_mm_s * t
        positions.append(x)
        dts.append(dt)

    # Deceleration
    for i in range(1, n_decel + 1):
        t = i * dt
        x = accel_len + cruise_len + speed_mm_s * t - 0.5 * accel * t * t
        positions.append(x)
        dts.append(dt)

    positions[-1] = total_len
    x_start = TANK.x_start_mm
    positions = [x_start + x for x in positions]
    return positions, dts


def build_tow_gcode(re, h):
    positions, dts = build_trapezoid_profile(re, h)
    y_pos = compute_tank_y_mm(h)
    z_pos = TANK.z_mm
    c_pos = TANK.c_mm
    axis_x = TANK.axis_x
    axis_y = TANK.axis_y
    axis_z = TANK.axis_z
    axis_c = TANK.axis_c

    lines = [
        "; ASMR tow sequence",
        "G21",  # mm units
        "G90",  # absolute positioning
        "G94",  # feed per minute
        f"G0 {axis_x}{positions[0]:.3f} {axis_y}{y_pos:.3f} {axis_z}{z_pos:.3f} {axis_c}{c_pos:.3f}",
    ]

    for i in range(1, len(positions)):
        dt = dts[i - 1]
        dx = positions[i] - positions[i - 1]
        feed_mm_min = (dx / dt) * 60.0 if dt > 0 else 0.0
        lines.append(
            f"G1 {axis_x}{positions[i]:.3f} {axis_y}{y_pos:.3f} {axis_z}{z_pos:.3f} {axis_c}{c_pos:.3f} F{feed_mm_min:.3f}"
        )

    if TANK.return_to_start:
        lines.append(
            f"G1 {axis_x}{positions[0]:.3f} {axis_y}{y_pos:.3f} {axis_z}{z_pos:.3f} {axis_c}{c_pos:.3f}"
        )
    lines.append("M2")
    return lines


def write_new_gcode(re, h):
    PATHS.gcode_output_dir.mkdir(parents=True, exist_ok=True)
    gcode_path = PATHS.gcode_output_dir / f"Re{re}_h{h}.gcode"
    lines = build_tow_gcode(re, h)
    gcode_path.write_text("\n".join(lines) + "\n")
    return gcode_path


def run_cncjs_gcode(gcode_path):
    if CNCJS.dry_run:
        print(f"[DRY RUN] CNCjs command would run with {gcode_path}")
        return
    if CNCJS.port is None:
        raise ValueError("CNCJS.port must be set to a serial port (e.g., COM3 or /dev/ttyUSB0).")

    cmd = list(CNCJS.cli)
    cmd += ["-p", CNCJS.port, "-b", str(CNCJS.baud)]
    if CNCJS.controller:
        cmd += ["-m", CNCJS.controller]
    cmd += list(CNCJS.extra_args)
    cmd += [str(gcode_path)]

    subprocess.run(cmd, check=True)


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

# def run_read_from_sensors():
#     sensors_reader.run_sensor_reading()
        
# ------------------------------------------------------------------------------------------

def load_next_params(param_file=PATHS.next_param_csv):
    df = pd.read_csv(param_file)
    return int(df['Re'][0]), float(df['h'][0])


def run_tank_motioncontrol():
    # Legacy: Path to the compiled C# file
    exe_path = "D:/dehan/Proj/Motioncontrol/bin/Release/net8.0/Motioncontrol.exe"

    # Run the executable
    result = subprocess.run([exe_path], capture_output=True, text=True)

    # Print the output and errors
    print("Output:")
    print(result.stdout)
    print("Errors:")
    print(result.stderr)


def run_tank_cncjs(re, h):
    print(
        f"[CONFIG] y0_mm is {TANK.y0_mm:.3f}. Verify this offset on the real tank before trusting Y positioning."
    )
    print(
        f"[CONFIG] h_mm_per_unit is {TANK.h_mm_per_unit:.6f}. Update after the simulation is rescaled to the new setup."
    )
    gcode_path = write_new_gcode(re, h)
    run_cncjs_gcode(gcode_path)

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

        # Run the towing tank experiment (CNCjs or legacy CSV)
        tank_thread = None
        if RUN_TANK:
            current_re, current_h = load_next_params()
            tank_thread = threading.Thread(target=run_tank_cncjs, args=(current_re, current_h))
        
        # Run a Lilypad simulation
        lilypad_thread = threading.Thread(target=run_lilypad_simulation, args=(result_queue,))

        # Start threads
        if tank_thread:
            tank_thread.start()
        lilypad_thread.start()

        # Wait for all threads to finish before rocessing to the next iteration
        if tank_thread:
            tank_thread.join()
        lilypad_thread.join()

        # Check if any error occurred in any of the threads
        if not result_queue.empty():
            e = result_queue.get()
            break

        # Wait for the water to calm down
        print("Going to sleep, waiting for water to get still")
        time.sleep(300)
        print("Waking up")

        # Get the lattest simulation subfolder
        new_input_folder_simu = find_newest_exp_path(PATHS.simu_root, "simu")

        # Compute the metrics
        cp_txt2metrics_csv(new_input_folder_simu, PATHS.simu_root / PATHS.metric_name, "simu")

        # Get the latest experiment subfolder and compute metrics
        try:
            new_input_folder_exp = find_newest_exp_path(PATHS.exp_root, "exp")
            cp_txt2metrics_csv(new_input_folder_exp, PATHS.exp_root / PATHS.metric_name, "exp")
        except FileNotFoundError:
            print("No experimental folders found yet; skipping experimental metrics this iteration.")

        # Run a new GPR with all the existing values to find the next exp
        new_re, new_h, std_predictions, mean_prediction, X_train, Y, mean_prediction_exp, X_train_exp, Y_exp = find_next_exp_param(metric_file=PATHS.metric_name)

        # Check if the max std_prediction is under a given threshold
        if np.max(std_predictions) < threshold:
            break

        # Rewrite the file for the next parameters
        write_new_parameters(PATHS.next_param_csv, new_re, new_h)
        if TRAJECTORY_MODE == "cncjs":
            write_new_gcode(new_re, new_h)
        else:
            write_new_traj_csv(new_re, new_h)

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
            PATHS.gpr_plot_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(PATHS.gpr_plot_dir / f"{str(1000 + count)[1:]}.png")
            plt.pause(0.1)

        if debug:
            cont = input("Do you want to continue? (y/n): ")
            if cont.lower() != 'y':
                break
def main():
    metric_path = PATHS.simu_root / PATHS.metric_name
    create_metric_file(metric_path, True)
    new_re, new_h, _, _, _, _, _, _, _ = find_next_exp_param(metric_file=PATHS.metric_name)
    write_new_parameters(PATHS.next_param_csv, new_re, new_h)
    if TRAJECTORY_MODE == "cncjs":
        write_new_gcode(new_re, new_h)
    else:
        write_new_traj_csv(new_re, new_h)

    run_autoexperiments(200, debug=False)
    # run_autoexperiments(100, already=101, debug=False)
    # run_autoexperiments(1, debug=True)


if __name__ == "__main__":
    main()
