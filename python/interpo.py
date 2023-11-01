import numpy as np
from matplotlib.pyplot import *
import pandas as pd
from scipy import interpolate

i = 0  # fonctionne
# i = 2 # ne fonctionne pas

# --- Generate and store knot points in matrix P
X_data = pd.read_csv("x.csv", header=None)
Y_data = pd.read_csv("y.csv", header=None)

n = len(X_data[i])
P = np.zeros([n, 2])
P[:, 0] = X_data[i]
P[:, 1] = Y_data[i]

x_interp = P[:, 0]
y_interp = P[:, 1]

x_interp2 = X_data[i+1]
y_interp2 = Y_data[i+1]


def find_duplicates_two_arrays(array1, array2):
    # Combine the two arrays into pairs of elements using zip
    combined_array = list(zip(array1, array2))
    seen = {}
    duplicates = []

    for item in combined_array:
        # Convert each pair of elements to a tuple for hashing
        key = tuple(item)

        # If the combination is already in the dictionary, it's a duplicate
        if key in seen:
            duplicates.append(item)
        else:
            seen[key] = True

    return duplicates


def remove_duplicates_from_arrays(array1, array2):
    combined_array = list(zip(array1, array2))
    seen = {}
    unique_array1 = []
    unique_array2 = []

    for item in combined_array:
        key = tuple(item)
        if key not in seen:
            seen[key] = True
            unique_array1.append(item[0])
            unique_array2.append(item[1])

    return unique_array1, unique_array2


x, y = remove_duplicates_from_arrays(x_interp, y_interp)
x = np.r_[x, x[0]]
y = np.r_[y, y[0]]
tck, _ = interpolate.splprep([x, y], s=len(x) // 2, per=True)
xi0, yi0 = interpolate.splev(np.linspace(0, 1, 100), tck)

min_x = np.min(xi0)
rotation_index = np.where(xi0 == min_x)[0]
rotation_init_index = rotation_index.astype(np.int64)
xi0 = np.roll(xi0, shift=-rotation_index, axis=0)
yi0 = np.roll(yi0, shift=-rotation_index, axis=0)

xi0, yi0 = remove_duplicates_from_arrays(xi0, yi0)
xi0 = np.r_[xi0, xi0[0]]
yi0 = np.r_[yi0, yi0[0]]
tck, _ = interpolate.splprep([xi0, yi0], s=len(xi0) // 2, per=True)
xi0, yi0 = interpolate.splev(np.linspace(0, 1, 50), tck)

x2, y2 = remove_duplicates_from_arrays(x_interp2, y_interp2)
x2 = np.r_[x2, x2[0]]
y2 = np.r_[y2, y2[0]]
tck2, _ = interpolate.splprep([x2, y2], s=len(x2) // 2, per=True)
xi02, yi02 = interpolate.splev(np.linspace(0, 1, 100), tck2)

min_x2 = np.min(xi02)
rotation_index2 = np.where(xi02 == min_x2)[0]
rotation_init_index2 = rotation_index2.astype(np.int64)
xi02 = np.roll(xi02, shift=-rotation_index2, axis=0)
yi02 = np.roll(yi02, shift=-rotation_index2, axis=0)

xi02, yi02 = remove_duplicates_from_arrays(xi02, yi02)
xi02 = np.r_[xi02, xi02[0]]
yi02 = np.r_[yi02, yi02[0]]
tck2, _ = interpolate.splprep([xi02, yi02], s=len(xi02) // 2, per=True)
xi02, yi02 = interpolate.splev(np.linspace(0, 1, 50), tck2)

# -------------------------------------------------------------------------------
# Init figures
# -------------------------------------------------------------------------------
fig, ax = subplots()
# ax.plot(P_gen[:,0], P_gen[:,1], 'y-', lw=2 ,label='Generating Curve')
ax.plot(P[:, 0], P[:, 1], 'ks', label='Knot points P')
ax.set_title('View X-Y')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal', 'datalim')
ax.margins(.1, .1)
ax.grid()

# --- Print plots
# hp = ax.plot(f_uu[:,0], f_uu[:,1], 'bo')
ax.plot(xi0, yi0, 'b-o')
ax.plot(xi0[0], yi0[0], 'go')
ax.plot(xi0[len(xi0) // 2], yi0[len(xi0) // 2], 'ro')
ax.plot(xi02, yi02, 'r-o')
ax.plot(xi02[0], yi02[0], 'go')
ax.plot(xi02[len(xi02) // 2], yi02[len(xi02) // 2], 'bo')
show()