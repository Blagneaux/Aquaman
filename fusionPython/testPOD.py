import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import os
import imageio
import io

plt.rcParams.update({'font.size' : 18, 'font.family' : 'Times New Roman', "text.usetex": False})

Path = 'C:/Users/blagn771/Desktop/DATA/FLUIDS/'
save_path = 'E:/Blog_Posts/OpenFOAM/ROM_Series/Post22/'
Files = os.listdir(Path)
print(Files)

# Contents = sp.io.loadmat(Path + Files[1])
Contents = sp.io.loadmat("C:/Users/blagn771/Documents/Aquaman/Aquaman/FullVorticityMapRe100_h60.mat")

Contents.keys()

m = Contents['nx'][0][0]
n = Contents['ny'][0][0]
Vort = Contents['VORTALL']
print(m, n)
print(Vort.shape)

Circle = plt.Circle((128*4/7, 64), 128/20, ec='k', color='white', zorder=2)

fig, ax = plt.subplots(figsize=(11, 4))

p = ax.contourf(np.reshape(Vort[:,0],(n,m)).T, levels = 1001, vmin=-0.5, vmax=0.5, cmap = "seismic_r")
q = ax.contour(np.reshape(Vort[:,0],(n,m)).T, 
               levels = [-2.4, -2, -1.6, -1.2, -0.8, -0.4, -0.2, 0.2, 0.4, 0.8, 1.2, 1.6, 2, 2.4], 
               colors='k', linewidths=1)
ax.add_patch(Circle)
ax.set_aspect('equal')

plt.show()

### Data Matrix
X = Vort

### Mean-removed Matrix
X_mean = np.mean(X, axis = 1)
Y = X - X_mean[:,np.newaxis]

### Covariance Matrix
C = np.dot(Y.T, Y)/(Y.shape[1]-1)
print("C:", C.shape)

### SVD of Covariance Matrix
U, S, V = np.linalg.svd(C)

### POD modes
Phi = np.dot(Y, U)
print("Y:", Y.shape)
print("U:", U.shape)
print("Phi:", Phi.shape)

### Temporal coefficients
a = np.dot(Phi.T, Y)

fig, axs = plt.subplots(3, 2, figsize=(15, 15))

# Loop through the subplots and modes
for i in range(3):
    for j in range(2):
        ax = axs[i, j]
        Mode = i * 2 + j
        # Reshape and plot the contourf plot for each Mode
        p = ax.contourf(np.reshape(Phi[:,Mode], (n, m)).T, levels=1001, vmin=-0.5, vmax=0.5, cmap="seismic_r")
        # Create and add the circle patch
        Circle = plt.Circle((128*4/7, 64), 128/20, ec='k', color='white', zorder=2)
        ax.add_patch(Circle)
        ax.set_aspect('equal')
        ax.set_title(f"Plot for Mode {Mode+1}", fontsize=12)  # Plain text title

# Improve layout to prevent subplot overlap
fig.subplots_adjust(hspace=0.5, wspace=0.5)

# Optionally, you can add a colorbar for the plots
fig.colorbar(p, ax=axs, orientation='vertical', fraction=0.01, pad=0.04)

plt.show()

fig, axs = plt.subplots(1, 2, figsize=(12, 4))

ax = axs[0]
p = ax.contourf(np.reshape(Y[:,0],(n,m)).T, levels = 1001, vmin=-0.5, vmax=0.5, cmap="seismic_r")
Circle = plt.Circle((128*4/7, 64), 128/20, ec='k', color='white', zorder=2)
ax.add_patch(Circle)
ax.xaxis.set_tick_params(direction='in', which='both')
ax.yaxis.set_tick_params(direction='in', which='both')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.set_aspect('equal')

ax = axs[1]
nModes = 10
sum = np.sum(Phi[:,:nModes], axis = 1)
p = ax.contourf(np.reshape(sum[:,],(n,m)).T, levels = 1001, vmin=-0.5, vmax=0.5, cmap="seismic_r")
Circle = plt.Circle((128*4/7, 64), 128/20, ec='k', color='white', zorder=2)
ax.add_patch(Circle)
ax.xaxis.set_tick_params(direction='in', which='both')
ax.yaxis.set_tick_params(direction='in', which='both')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.set_aspect('equal')

plt.show()

Energy = np.zeros((len(S),1))
for i in np.arange(0,len(S)):
    Energy[i] = S[i]/np.sum(S)

X_Axis = np.arange(Energy.shape[0])
heights = Energy[:,0]

fig, axes = plt.subplots(1, 2, figsize = (12,4))
ax = axes[0]
ax.bar(X_Axis, heights, width=0.5)
ax.set_xlim(-0.25, 25)
ax.set_xlabel('Modes')
ax.set_ylabel('Energy Content')

ax = axes[1]
cumulative = np.cumsum(S)/np.sum(S)
ax.plot(cumulative, marker = 'o', markerfacecolor = 'none', markeredgecolor = 'k', ls='-', color = 'k')
ax.set_xlabel('Modes')
ax.set_ylabel('Cumulative Energy')
ax.set_xlim(0, 25)

plt.show()

fig, axs = plt.subplots(3, 2, figsize=(15, 15))

# Loop through the subplots and modes
for i in range(3):
    for j in range(2):
        ax = axs[i, j]
        Mode = i * 2 + j
        ax.plot(a[Mode, :])
        ax.xaxis.set_tick_params(direction='in', which='both')
        ax.yaxis.set_tick_params(direction='in', which='both')
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.set_title(f"Plot for Mode {Mode+1}", fontsize=12)  # Plain text title

plt.tight_layout()
plt.show()