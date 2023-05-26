from numpy import genfromtxt
import numpy as np
my_data = genfromtxt("C:/Users/blagn771/Desktop/FullPressure.csv", delimiter=",")
my_data_str = [[str(i).replace(".",",").replace("\n","") for i in row] for row in my_data]
print("Creating new file")
np.savetxt("C:/Users/blagn771/Desktop/FullPressure_norm.csv", my_data_str, delimiter=";", fmt="%s")
