import matplotlib.pyplot as plt
import numpy as np

D=16
dt = 0.430
len_dt = 5533

# data = np.genfromtxt("C:/Users/blagn771/Desktop/benchmark_WaterliLy/Re100_D16/excels/Lift.csv", delimiter=",")
# data2 = np.genfromtxt("C:/Users/blagn771/Desktop/benchmark_WaterliLy/Re100_D16/excels/v4D.csv", delimiter=",")
# sp = np.fft.fft(data)
# sp2 = np.fft.fft(data2)
# freq = np.fft.fftfreq(len(data),d=dt*len_dt/1536)
# plt.plot(freq[:100]*D,abs(sp)[:100]/max(abs(sp)))
# plt.plot(freq[:100]*D,abs(sp2)[:100]/max(abs(sp2)))
# plt.plot(freq[np.argmax(abs(sp))]*D,[1], "*")
# plt.legend(["frequency of the C_L", "frequency of the Y velocity at 8D,4D",f"{round(freq[np.argmax(abs(sp))]*D,4)}"])
# plt.show()

# R = [100, 150, 200, 250]
# lowR = [0.1805, 0.2012, 0.2083, 0.2152]
# highR = [0.1865, 0.2002, 0.2151, 0.2214]
# plt.plot(R, lowR)
# plt.plot(R, highR)
# plt.legend(["D=16", "D=32"])
# plt.xlim(50,300)
# plt.ylim(0.15,0.25)
# plt.xlabel("Re")
# plt.ylabel("St")
# plt.show()

P = [-0.2, -0.22, -0.21, -0.16, -0.12, -0.22, -0.20, -0.16, -0.12, -0.09, -0.08]
V = [0.9, 0.86, 0.9, 0.97, 0.96, 0.86, 0.9, 0.96, 0.96, 1.02, 1.06]
Y = [2.5, 2, 1.5, 1, 0.5, 0, -0.5, -1, -1.5, -2, -2.5]
plt.plot(V,Y)
plt.legend("C = 64")
plt.xlabel("Vx")
plt.ylabel("vertical position")
plt.show()