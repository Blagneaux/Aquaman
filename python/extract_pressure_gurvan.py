import numpy as np
import pandas as pd
import os
import re
import matplotlib.pyplot as plt

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
    # Define a regular expression pattern to match the h value with possible decimals
    pattern = r"Re\d+_h(\d+\.\d+|\d+)"

    # Search for the pattern in the file path
    match = re.search(pattern, file_path)

    # If a match is found, extract and return the h value as a float or integer
    if match:
        h_value = match.group(1)  # Extract the captured group which includes decimal part
        return float(h_value) if '.' in h_value else int(h_value)
    else:
        return None  # Return None if no match is found
    
def read_cp_data(file_path, Re):
    output = []
    _nu = L / Re

    with open(file_path, 'r') as pressure:
        for line in pressure:
            output.append(line)

    data = output[4:]
    wall, time = [], []
    for i in range(len(data)):
        to_add = data[i].split()
        if to_add[-1] == ';':
            to_add.pop()
        to_add = np.array(to_add)
        to_add = to_add.astype(np.float32)
        if i % 2 == 0:
            time.append((to_add[0]) * h * h * _nu / nu)
        else:
            wall.append(to_add)

    wall_dataframe = pd.DataFrame(wall)

    return wall_dataframe, time

# Fixed parameters for the experiments
width = 2**7
L = width / 20
nu = 1 / 1000000
origin = 3*width/2
h = 1/width

def compute_metrics_for_gurvan(input_folder):
    # Extract the variables from the name of the file
    Re_value = extract_re_from_file_path(input_folder)
    h_value = extract_h_from_file_path(input_folder)
    print(Re_value, h_value)

    # Get the time evolution during the experiments
    file_path = os.path.join(input_folder, "pressureMotion.txt")
    _, time = read_cp_data(file_path, Re_value)

    # Load the whole pressure map
    data_file_path = os.path.join(input_folder, "FullMap.csv")
    full_data = pd.read_csv(data_file_path)

    # Extract the pressure at the center sensor so it has an optimal approching and going away part
    mid_data = [full_data[i][width*(width-1)] * 1000 * (Re_value*nu/0.05)**2 for i in full_data.columns]

    # Compute the distances
    y = (h_value - L/2) / width   # in meter
    Y = [y for t in time]
    X = [-(0.5 - t*(Re_value*nu/0.05)) for t in time]
    dist = [np.sqrt(x*x + y*y) for (x,y) in zip(X, Y)]

    # Create a single DataFrame from the lists
    df = pd.DataFrame({
        "Pressure (Pa)": mid_data,
        "Vertical distance (m)": Y,
        "Horizontal distance (m)": X,
        "Euclidean distance (m)": dist,
        "Time (s)": time
    })

    # Save the DataFrame to a CSV file
    output_file_path = os.path.join(input_folder, "metrics_for_gurvan.csv")
    df.to_csv(output_file_path, index=False)


main_folder = "E:\simuChina\cartesianBDD_FullPressureMap"
sub_folders = []
for root, dirs, files in os.walk(main_folder):
        # 'root' is the path to the directory, 'dirs' is a list of the subdirectories in 'root'
        for dir in dirs:
            sub_folders.append(os.path.join(root, dir))
        break  # This break ensures that only the first level of directories is listed

for sub_folder in sub_folders:
    input_folder = os.path.join(main_folder, sub_folder)
    compute_metrics_for_gurvan(input_folder=input_folder)