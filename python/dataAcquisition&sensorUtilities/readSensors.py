import threading
try:
    import nidaqmx
except ImportError:  # Allow pressure-only setups without NI-DAQ installed
    nidaqmx = None

from readPressure import SerialReader
import pandas as pd

class SensorReading:
    def __init__(
        self,
        task=None,
        *,
        read_pressure=True,
        read_cylinder=False,
        num_pressure_sensors=8,
        pressure_output_folder="E:/sensorExpChina/wall",
    ):
        self.task = task
        self.read_pressure = read_pressure
        self.read_cylinder = read_cylinder
        self.num_pressure_sensors = num_pressure_sensors
        self.pressure_output_folder = pressure_output_folder

    def run_read_pressure_from_sensors(self):
        reader = SerialReader(
            num_sensors=self.num_pressure_sensors,
            output_folder=self.pressure_output_folder,
        )
        next_param_df = pd.read_csv("E:/simuChina/metric_test_next_param.csv")
        Re = next_param_df['Re'][0]
        duration = 0.05 * 1e6 / Re  # Duration of reading in seconds
        h = next_param_df['h'][0]
        reader.start_reading(duration, Re, h)

    def run_read_cylinder_from_sensor(self):
        if nidaqmx is None:
            raise RuntimeError("nidaqmx is not available; cannot read cylinder sensors.")
        from forceSensor_connect import DataProcessor

        cylinder_reader = DataProcessor()
        cylinder_reader.read_data(self.task)

    def run_sensor_reading(self):
        threads = []
        if self.read_pressure:
            threads.append(threading.Thread(target=self.run_read_pressure_from_sensors))
        if self.read_cylinder:
            threads.append(threading.Thread(target=self.run_read_cylinder_from_sensor))

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

# Example usage
if __name__ == "__main__":
    task = None
    if nidaqmx is not None:
        task = nidaqmx.Task()
        task.ai_channels.add_ai_voltage_chan("Dev2/ai0:21")

    sensor_reading_instance = SensorReading(
        task,
        read_pressure=True,
        read_cylinder=False,
        num_pressure_sensors=8,
    )
    sensor_reading_instance.run_sensor_reading()
