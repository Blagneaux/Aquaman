import matplotlib.colors
import matplotlib.pyplot as plt
font = {'family': 'Arial',
        'weight': 'normal',
        'size': 22}
matplotlib.rc('font', **font)

Re = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

E99_no_zoom = [68, 73.16666666666667, 74.66666666666667, 75.58333333333333, 76.08333333333333, 76.41666666666667, 76.5, 76.58333333333333, 76.75, 76.91666666666667]
E95_no_zoom = [37.166666666666664, 40.333333333333336, 41.333333333333336, 41.666666666666664, 42, 42.416666666666664, 42.5, 42.416666666666664, 42.666666666666664, 42.75]
E1_no_zoom = [0.24437598395716248, 0.23633661771641962, 0.23168221096351066, 0.22958369304433865, 0.2290971287104474, 0.22798702782399843, 0.22784124356296928, 0.22863644504299463, 0.22730408541960823, 0.2276861340832362]   
RMSE_no_zoom = [310.9190067032547, 341.79906877178416, 350.36097931396097, 355.27007137474686, 358.9603342941534, 359.9374516425791, 361.7658838629103, 364.3656815502671, 363.9603724285063, 365.1966627806381]

E99_zoom_128 = [45.916666666666664, 48.666666666666664, 49.75, 50.25, 50.5, 50.75, 50.916666666666664, 51.083333333333336, 51.25, 51.25]
E95_zoom_128 = [23, 24, 24.666666666666668, 24.833333333333332, 25.166666666666668, 25.333333333333332, 25.333333333333332, 25.166666666666668, 25.416666666666668, 25.333333333333332]
E1_zoom_128 = [0.2989855228928919, 0.2955803919726431, 0.2939422398014973, 0.29253562354642526, 0.2925336418519424, 0.29177785958162955, 0.29177627924460814, 0.2935883775447157, 0.2926863769490475, 0.29294075138074294]
RMSE_zoom_128 = [501.8349829213442, 558.3406364439164, 576.5499960509667, 584.4999305988322, 590.435460794475, 592.5634417267412, 595.7232769380064, 600.3371849453739, 600.1802731466522, 601.8581909590671]

E99_zoom_64 = [31.083333333333332, 32.666666666666664, 33.083333333333336, 33.5, 33.666666666666664, 33.916666666666664, 33.916666666666664, 33.833333333333336, 33.916666666666664, 33.916666666666664]
E95_zoom_64 = [14.166666666666666, 14.833333333333334, 15.083333333333334, 15.083333333333334, 15.25, 15.416666666666666, 15.333333333333334, 15.166666666666666, 15.25, 15.25]
E1_zoom_64 = [0.32458026127010275, 0.3221808915240456, 0.32123176663796743, 0.32067419675453057, 0.32069632492928624, 0.3203158794953943, 0.32027502213196996, 0.3217247807537288, 0.3217453382687754, 0.3216882122897046]
RMSE_zoom_64 = [565.6788376119529, 625.8193320875258, 647.4783236036147, 657.4920870611613, 664.4399926585533, 667.1579667114611, 670.7669023252238, 674.7065047344074, 675.6222306857693, 677.1418279759215]

E99_zoom_64_posteriori = [68, 73.16666666666667, 74.66666666666667, 75.58333333333333, 76.08333333333333, 76.41666666666667, 76.5, 76.58333333333333, 76.75, 76.91666666666667]
E95_zoom_64_posteriori = [37.166666666666664, 40.333333333333336, 41.333333333333336, 41.666666666666664, 42, 42.416666666666664, 42.5, 42.416666666666664, 42.666666666666664, 42.75]
E1_zoom_64_posteriori = [0.24437598395716248, 0.23633661771641962, 0.23168221096351066, 0.22958369304433865, 0.2290971287104474, 0.22798702782399843, 0.22784124356296928, 0.22863644504299463, 0.22730408541960823, 0.2276861340832362]       
RMSE_zoom_64_posteriori = [920.3933810663701, 1013.3464282392248, 1041.886363718374, 1056.3648100876885, 1067.1293728787539, 1070.2398904919698, 1075.7720611519665, 1083.3035210704704, 1082.7563649725964, 1086.0476933733864]

fig = plt.figure()
plt.title("Mean of the number of modes to keep at least 99% of the energy of the system over the\n distance between the center of the cylinder and the wall, with respect to the Reynolds number")
plt.plot(Re, E99_no_zoom, marker='o', label="No zoom, width=256", linestyle='dashed')
plt.plot(Re, E99_zoom_128, marker='o', label="Zoom to width=128", linestyle='dashed')
plt.plot(Re, E99_zoom_64, marker='o', label="Zoom to width=64", linestyle='dashed')
plt.plot(Re, E99_zoom_64_posteriori, marker='o', label="Zoom a posteriori to width=64", linestyle='dashed')
plt.xlabel("Reynolds number")
plt.ylabel("Mean number of modes")
plt.legend()

fig = plt.figure()
plt.title("Mean of the number of modes to keep at least 95% of the energy of the system over the\n distance between the  center of the cylinder and the wall, with respect to the Reynolds number")
plt.plot(Re, E95_no_zoom, marker='o', label="No zoom, width=256", linestyle='dashed')
plt.plot(Re, E95_zoom_128, marker='o', label="Zoom to width=128", linestyle='dashed')
plt.plot(Re, E95_zoom_64, marker='o', label="Zoom to width=64", linestyle='dashed')
plt.plot(Re, E95_zoom_64_posteriori, marker='o', label="Zoom a posteriori to width=64", linestyle='dashed')
plt.xlabel("Reynolds number")
plt.ylabel("Mean number of modes")
plt.legend()

fig = plt.figure()
plt.title("Mean energy of the most energetic mode of the system over the distance between the center\n of the cylinder  and the wall, with respect to the Reynolds number")
plt.plot(Re, E1_no_zoom, marker='o', label="No zoom, width=256", linestyle='dashed')
plt.plot(Re, E1_zoom_128, marker='o', label="Zoom to width=128", linestyle='dashed')
plt.plot(Re, E1_zoom_64, marker='o', label="Zoom to width=64", linestyle='dashed')
plt.plot(Re, E1_zoom_64_posteriori, marker='o', label="Zoom a posteriori to width=64", linestyle='dashed')
plt.xlabel("Reynolds number")
plt.ylabel("Mean number of modes")
plt.legend()

fig = plt.figure()
plt.title("Mean RMSE of the reconstruction of the flow with 99% of the energy of the system over\n the distance between  the center of the cylinder and the wall, with respect to the Reynolds number")
plt.plot(Re, RMSE_no_zoom, marker='o', label="No zoom, width=256", linestyle='dashed')
plt.plot(Re, RMSE_zoom_128, marker='o', label="Zoom to width=128", linestyle='dashed')
plt.plot(Re, RMSE_zoom_64, marker='o', label="Zoom to width=64", linestyle='dashed')
plt.plot(Re, RMSE_zoom_64_posteriori, marker='o', label="Zoom a posteriori to width=64", linestyle='dashed')
plt.xlabel("Reynolds number")
plt.ylabel("Mean number of modes")
plt.legend()

plt.show()