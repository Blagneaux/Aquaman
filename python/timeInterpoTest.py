import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import pandas as pd
import pysindy as ps

a = [0, 0.2, -0.1]
c = int(2 ** 6 / 3)
xc = 2 ** 6 / 4 - 0.25 * c
k = 2 * np.pi / c
omega = 1.2 * k
T = 2 * np.pi / omega
s = 0
for ai in a:
    s += ai
if s == 0:
    s = 1
    a[0] = 1
for i in range(len(a)):
    a[i] *= 0.25 * T / s


def Ax(x):
    amp = a[0]
    for i in range(1, len(a)):
        amp += a[i] * (((x - xc) / c) ** i)
    return amp


def h(x, t):
    return Ax(x) * np.sin(k * (x - xc) - omega * t)


def hdot(x, t):
    return -Ax(x) * omega * np.cos(k * (x - xc) - omega * t)


def dhdx(x, t):
    amp = a[1] / c
    for i in range(2, len(a)):
        amp += a[i] * i / c * ((x - xc) / c)**(i-1)
    return Ax(x) * k * np.cos(k * (x - xc) - omega * t) + amp * np.sin(k * (x - xc) - omega * t)

def custom_deriv(x, t):
    x_dot = []
    index = 0
    for i in range(len(x)):
        if i < 3:
            sub_x = x[:5]
            sub_t = t[:5]
            index = 1
        elif i > len(x) - 3:
            sub_x = x[-5:]
            sub_t = t[-5:]
            index = 5 - len(x) + i
        else:
            sub_x = x[i-2:i+3]
            sub_t = t[i-2:i+3]
            index = 3
        spl = interpolate.splrep(sub_t, sub_x)
        interpolated_x = interpolate.splev(sub_x, spl, der=1)
        print(interpolated_x)
        x_dot.append(interpolated_x[index])

    return x_dot


X_data = pd.read_csv("x.csv", header=None)
Y_data = pd.read_csv("y.csv", header=None)

print(len(Y_data.columns))

x0, y0 = [], []
for i in range(len(X_data.columns)):
    x0.append(X_data[i][247])
    y0.append(Y_data[i][247])

for i in Y_data[499]:
    print(i)

for i in range(len(Y_data[0])):
    x_mean = np.mean([X_data[j][i] for j in range(len(X_data.columns))])
    # print(i, np.round(x_mean,2),hdot(x_mean, 250))

X = np.mean(x0)
Y = np.mean(y0)
T = np.linspace(0, 499.5, 1000)

print(hdot(30.5, 249.25))


# fig, ax = plt.subplots()
# ax.plot(y0)
# ax.plot([h(X, t) for t in T] + Y)

# spl_y = interpolate.splrep(T, y0, k=3, s=0.5)
# # y_dot = custom_deriv([h(X, t) for t in T], T)
# # y_dot = interpolate.splev(np.linspace(T[0], T[-1], 20000), spl_y)

# sfd = ps.SmoothedFiniteDifference(smoother_kws={'window_length': 11})
# y_dot = sfd._differentiate(y0, T)

# fig, ax = plt.subplots()
# ax.plot(y_dot)

# ax2 = ax.twiny()
# # ax2.plot(y0[:40], 'g')
# # ax2.plot([h(X, t) for t in T[:40]] + Y, 'r')
# ax2.plot([hdot(X, t) for t in T], 'r')
# plt.show()