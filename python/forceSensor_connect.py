import time
import pandas as pd
import os
import datetime
import numpy as np
import nidaqmx


class DataProcessor:
    def __init__(self, desktop_dir='E:/sensorExpChina/forceSensor'):
        next_param_df = pd.read_csv("E:/simuChina/metric_test_next_param.csv")
        Re = next_param_df['Re'][0]
        self.duration = 0.05*1e6/Re  # Duration of reading in seconds
        h = next_param_df['h'][0]
        self.filename = f"Re{Re}_h{h}.csv"
        self.desktop_dir = desktop_dir
        self.filepath1 = os.path.join(desktop_dir, self.filename)
        title = pd.DataFrame(columns=['world time', 'force_x_n', 'force_y', 'force_z', 'tx_lbfin', 'ty', 'tz', 'co_drag', 'pressure_30', 'pressure_100', 'pressure_150', 'pressure_260', 'pressure_330'])
        title.to_csv(self.filepath1, index=0, encoding="utf-8")

        # self.task = nidaqmx.Task()
        # self.task.ai_channels.add_teds_ai_voltage_chan("Dev2/ai0:5")

        # ft12607
        self.transfer_matrix = np.array([
            [-0.00325, 0.00015, 0.09822, -1.82505, -0.09847, 1.94002],
            [-0.10322, 2.17422, 0.05485, -1.05390, 0.05019, -1.12259],
            [3.34623, -0.19080, 3.40461, -0.09898, 3.34600, -0.15760],
            [-0.07841, 1.03241, -3.81689, -0.40255, 3.85454, -0.69811],
            [4.36689, -0.24081, -2.30034, 0.92527, -2.13793, -0.81700],
            [0.11835, -2.30599, 0.10847, -2.24479, 0.12206, -2.39381]])

    def add2file(self, force, co_drag):
        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        rightnow = [formatted_time]
        data_all = [
            [rightnow, force[0], force[1], force[2], force[3], force[4], force[5], co_drag, force[6], force[7], force[8], force[9], force[10]]]
        data_all = pd.DataFrame(data_all)
        data_all.to_csv(self.filepath1, mode="a+", index=0, header=0)

    def read_data(self, task):
        print("Start reading force data")
        duration = self.duration
        start_time = time.time()
        while (time.time() - start_time) < duration:
            data2 = np.array(task.read())
            data = data2
            co_drag = 0
            force_matrix = self.process_data(data)
            self.add2file(force_matrix, co_drag)
            # print(force_matrix)
        print("Finished reading force data")

    def process_data(self, data):
        transfer_matrix1 = self.transfer_matrix.T
        force_matrix_lbf = np.dot(data, transfer_matrix1)
        force_matrix = force_matrix_lbf
        force_matrix[0] = force_matrix_lbf[0] * 4.44822162
        force_matrix[1] = force_matrix_lbf[1] * 4.44822162
        force_matrix[2] = force_matrix_lbf[2] * 4.44822162
        force_matrix[6] = force_matrix_lbf[6] * 689476
        force_matrix[7] = force_matrix_lbf[7] * 689476
        force_matrix[8] = force_matrix_lbf[8] * 689476
        force_matrix[9] = force_matrix_lbf[9] * 689476
        force_matrix[10] = force_matrix_lbf[10] * 689476
        return force_matrix


if __name__ == "__main__":
    data_processor = DataProcessor()
    task = nidaqmx.Task()
    task.ai_channels.add_ai_voltage_chan("Dev2/ai0:5")
    data_processor.read_data(task)
