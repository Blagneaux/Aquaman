import threading
import nidaqmx
from readPressure import SerialReader
from forceSensor_connect import DataProcessor
import pandas as pd

class SensorReading():
    def __init__(self, task):
        self.task = task

    def run_read_pressure_from_sensors(self):
        reader = SerialReader()
        next_param_df = pd.read_csv("E:/simuChina/metric_test_next_param.csv")
        Re = next_param_df['Re'][0]
        duration = 0.05 * 1e6 / Re  # Duration of reading in seconds
        h = next_param_df['h'][0]
        reader.start_reading(duration, Re, h)

    def run_read_cylinder_from_sensor(self):
        cylinder_reader = DataProcessor()
        cylinder_reader.read_data(self.task)

    def run_sensor_reading(self):
        # Create the threads
        pressure_thread = threading.Thread(target=self.run_read_pressure_from_sensors)
        cylinder_thread = threading.Thread(target=self.run_read_cylinder_from_sensor)

        # Starts the threads
        pressure_thread.start()
        cylinder_thread.start()

        # Wait for all threads to finish
        pressure_thread.join()
        cylinder_thread.join()

# Example usage
if __name__ == "__main__":
    task = nidaqmx.Task()
    task.ai_channels.add_ai_voltage_chan("Dev2/ai0:5")
    sensor_reading_instance = SensorReading(task)
    sensor_reading_instance.run_sensor_reading()
