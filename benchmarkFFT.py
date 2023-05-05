import matplotlib.pyplot as plt
import numpy as np

data = np.genfromtxt("C:/Users/blagn771/Desktop/benchmark_WaterliLy/Re250_D32/Lift.csv", delimiter=",")
data2 = np.genfromtxt("C:/Users/blagn771/Desktop/benchmark_WaterliLy/Re250_D32/v4D.csv", delimiter=",")
sp = np.fft.fft(data)
sp2 = np.fft.fft(data2)
freq = np.fft.fftfreq(len(data),d=0.297*16175/1536)
plt.plot(freq[:100]*32,abs(sp)[:100]/max(abs(sp)))
plt.plot(freq[:100]*32,abs(sp2)[:100]/max(abs(sp2)))
plt.plot(freq[np.argmax(abs(sp))]*32,[1], "*")
plt.legend(["frequency of the C_L", "frequency of the Y velocity at 8D,4D",f"{round(freq[np.argmax(abs(sp))]*32,4)}"])
plt.show()