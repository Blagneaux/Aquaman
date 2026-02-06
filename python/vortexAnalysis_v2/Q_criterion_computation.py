import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import os
import shutil
import sys

def compute_Q_criterion(vx, vy, dx, dy):
    """
    Compute the Q-criterion for a 2D velocity field.

    Parameters:
    vx : 2D numpy array
        Velocity component in the x-direction.
    vy : 2D numpy array
        Velocity component in the y-direction.
    dx : float
        Grid spacing in the x-direction.
    dy : float
        Grid spacing in the y-direction.

    Returns:
    Q : 2D numpy array
        Q-criterion values at each grid point.
    """
    # Unflatten the velocity fields
    
    # Compute velocity gradients
    dvx_dx = np.gradient(vx, dx, axis=0)
    dvx_dy = np.gradient(vx, dy, axis=1)
    dvy_dx = np.gradient(vy, dx, axis=0)
    dvy_dy = np.gradient(vy, dy, axis=1)

    # Compute components of the rate of strain tensor S and rotation tensor Ω
    s11 = dvx_dx
    s22 = dvy_dy
    s12 = 0.5 * (dvx_dy + dvy_dx)
    o12 = 0.5 * (dvx_dy - dvy_dx)

    # Frobenius norms squared: ||S||^2 = s11^2 + s22^2 + 2*s12^2, ||Omega||^2 = 2*o12^2
    Q = 0.5 * (2.0 * o12**2 - (s11**2 + s22**2 + 2.0 * s12**2))
    return Q

def compute_Q_map(vx_path, vy_path, dx, dy):
    """
    Compute the Q-criterion for a 2D velocity field evolving with time.

    Parameters:
    vx_path : string
        Velocity component in the x-direction for every time step.
    vy_path : string
        Velocity component in the y-direction for every time step.
    dx : float
        Grid spacing in the x-direction.
    dy : float
        Grid spacing in the y-direction.

    Returns:
    Q : 2D numpy array
        Q-criterion values at each grid point (flatten) and each time step.
    """
    vx = pd.read_csv(vx_path, header=None).to_numpy(dtype=np.float64)
    vy = pd.read_csv(vy_path, header=None).to_numpy(dtype=np.float64)

    Q_criterion = np.empty_like(vx)

    # Unflatten the velocity fields to work with the spatial 2D arrays
    for i in range(len(vx[0])):
        vxi = vx[:,i].reshape(256,128)
        vyi = vy[:,i].reshape(256,128)
        # Compute Q-criterion for the current snapshot
        Q_criterion[:,i] = compute_Q_criterion(vxi, vyi, dx, dy).ravel()

    del vx, vy
    return Q_criterion

# Set the grid spacing (LilyPad should be 1x1 units)
dx = 1
dy = 1

# Prepare paths from all the configurations
path_nadia = "D:/crop_nadia/"
path_thomas = "D:/thomas_files/"
path_boai = "D:/boai_files/"
path_full = "D:/full_files/"

ux_path = "ux_map.csv"
ux_circle_path = "circle_ux_map.csv"
uy_path = "uy_map.csv"
uy_circle_path = "circle_uy_map.csv"
Q_criterion_path = "Q_criterion_map.csv"
Q_criterion_circle_path = "circle_Q_criterion_map.csv"

# List the experiments where the sensors did work
list_experiments = [1, 2, 3, 6, 7, 8, 9, 10, 11, 14, 15, 16, 17, 18, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 40]
# list_experiments = [3, 6, 7, 8, 9, 10, 11, 14, 15, 16, 17, 18, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 40]

all = True  # Set to True to compute Q-criterion for all experiments and snapshots

if all:
    # For a given experiment, select the snapshots where the fish was detected
    for experiment in list_experiments:
        subpath = path_boai + str(experiment)
        subfolders = []
        for dirpath, dirnames, filenames in os.walk(subpath):
            subfolders.extend(dirnames)
        
        for experiment_snapshot in subfolders:
            full_nadia_path = path_nadia + str(experiment) + "/" + experiment_snapshot + "/"
            full_thomas_path = path_thomas + str(experiment) + "/" + experiment_snapshot + "/"
            full_boai_path = path_boai + str(experiment) + "/" + experiment_snapshot + "/"
            full_full_path = path_full + str(experiment) + "/" + experiment_snapshot + "/"

            total, used, free = shutil.disk_usage("D:/")
            if free < 10 * 1024**3:  # Check if free space is less than 10 GB
                print(f"Warning: Low disk space. Stopped computation at experiment {experiment}, snapshot {experiment_snapshot}.")
                sys.exit(1)

            Q_to_save = compute_Q_map(full_nadia_path + ux_path, full_nadia_path + uy_path, dx, dy)
            pd.DataFrame(Q_to_save).to_csv(full_nadia_path + Q_criterion_path, index=False, header=False)
            del Q_to_save

            Q_to_save = compute_Q_map(full_nadia_path + ux_circle_path, full_nadia_path + uy_circle_path, dx, dy)
            pd.DataFrame(Q_to_save).to_csv(full_nadia_path + Q_criterion_circle_path, index=False, header=False)
            del Q_to_save

            Q_to_save = compute_Q_map(full_thomas_path + ux_path, full_thomas_path + uy_path, dx, dy)
            pd.DataFrame(Q_to_save).to_csv(full_thomas_path + Q_criterion_path, index=False, header=False)
            del Q_to_save

            Q_to_save = compute_Q_map(full_thomas_path + ux_circle_path, full_thomas_path + uy_circle_path, dx, dy)
            pd.DataFrame(Q_to_save).to_csv(full_thomas_path + Q_criterion_circle_path, index=False, header=False) 
            del Q_to_save  

            Q_to_save = compute_Q_map(full_boai_path + ux_path, full_boai_path + uy_path, dx, dy)
            pd.DataFrame(Q_to_save).to_csv(full_boai_path + Q_criterion_path, index=False, header=False)
            del Q_to_save

            Q_to_save = compute_Q_map(full_boai_path + ux_circle_path, full_boai_path + uy_circle_path, dx, dy)
            pd.DataFrame(Q_to_save).to_csv(full_boai_path + Q_criterion_circle_path, index=False, header=False)
            del Q_to_save

            Q_to_save = compute_Q_map(full_full_path + ux_path, full_full_path + uy_path, dx, dy)
            pd.DataFrame(Q_to_save).to_csv(full_full_path + Q_criterion_path, index=False, header=False)
            del Q_to_save

            Q_to_save = compute_Q_map(full_full_path + ux_circle_path, full_full_path + uy_circle_path, dx, dy)
            pd.DataFrame(Q_to_save).to_csv(full_full_path + Q_criterion_circle_path, index=False, header=False)
            del Q_to_save

        print(f"Completed Q-criterion computation for experiment {experiment}.")

    print("Q-criterion computation completed for all experiments and snapshots.")

else:
    Q_criterion = compute_Q_map(
        vx_path="velocity_x_map.csv",
        vy_path="velocity_y_map.csv",
        dx=dx,
        dy=dy,
    )
    pd.DataFrame(Q_criterion).to_csv("Q_criterion_test_map.csv", index=False, header=False)