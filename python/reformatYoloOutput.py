import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

file_path = "C:/Users/blagn771/Documents/Aquaman/Aquaman/runs/segment/predict5/labels/test_4.txt"

# Step 1: Open the text file in read mode
try:
    with open(file_path, 'r') as file:
        # Step 2: Read the file line by line and store values in an array
        values_array = []
        for line in file:
            # Step 3: Append each line (or value) to the array
            value = line.strip()  # Remove leading/trailing whitespace
            values_array.append(value)

except FileNotFoundError:
    print(f"The file '{file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {str(e)}")

# Now, 'values_array' contains the values from the text file
print("Values from the text file:")
for value in values_array:
    print(value)

# Assuming values are stored as a single string with blank space separation
values_str = values_array[0]  # Get the single string element
split_values = values_str.split()  # Split the string into individual values

# Convert the split values to floats
numeric_values = [float(value) for value in split_values[1:]]
max_value = max(numeric_values)
min_value = min(numeric_values)

# Create x-axis values (assuming the values are plotted sequentially)
x_values = np.linspace(min_value, max_value, len(numeric_values))

# Create the plot
plt.plot(x_values, numeric_values, marker='o', linestyle='-')
plt.title('Plot of Values from Text File')
plt.xlabel('Data Point Index')
plt.ylabel('Value')
plt.grid(True)

# Display the plot
plt.show()