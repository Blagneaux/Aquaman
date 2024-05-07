import os
import serial
import serial.tools.list_ports
import threading
from openpyxl import Workbook
import pandas as pd
from openpyxl.styles import NamedStyle

class SerialReader:
    def __init__(self):
        self.serial_port = None
        self.is_reading = False
        self.data_filename = None
        self.data_buffer = []
        self.output_folder = "D:/sensorExpChina/wall"

    def start_reading(self, duration=6, Re=1000, h=6):
        self.data_buffer.clear()  # Clear buffer before starting
        filename = "pressureMotion.xlsx"
        directory = f"Re{Re}_h{h}"
        full_output_folder = os.path.join(self.output_folder, directory)
        os.mkdir(full_output_folder)
        self.data_filename = os.path.join(full_output_folder,filename)
        self.serial_port = self.recup_port_Arduino()
        if self.serial_port:
            self.is_reading = True
            threading.Thread(target=self.read_serial).start()
            threading.Timer(duration, self.stop_reading).start()

    def stop_reading(self):
        if self.is_reading:
            self.is_reading = False
            self.serial_port.close()
            self.write_to_excel()
            print("Lecture arrêtée avec succès.")

    def read_serial(self):
        while self.is_reading:
            try:
                if self.serial_port and self.serial_port.is_open:
                    data = self.serial_port.readline().decode("utf-8", errors="replace")
                    # print(data.strip())
                    pressure_data, timestamp = self.extract_data(data)
                    if pressure_data is not None and timestamp is not None:
                        self.data_buffer.append((pressure_data, timestamp))
            except (serial.SerialException, UnicodeDecodeError) as e:
                print(f"Erreur de lecture depuis le port série : {e}")
                self.stop_reading()
            except AttributeError as e:
                print(f"Erreur: {e}")
                self.stop_reading()

    def extract_data(self, data):
        parts = data.split()
        if len(parts) == 9:
            try:
                pressure_data = [float(part) for part in parts[:8]]
                timestamp = float(parts[-1])
                return pressure_data, timestamp
            except ValueError:
                print(f"Erreur de conversion des données : {data}")
                return None, None
        else:
            return None, None

    def write_to_excel(self):
        if self.data_filename:
            workbook = Workbook()
            sheet = workbook.active
            sheet.title = "Serial Data"

            integer_style = NamedStyle(name="integer", number_format="0")

            for row_index, (pressure_data, timestamp) in enumerate(
                self.data_buffer, start=1
            ):
                for col_index, value in enumerate(pressure_data + [timestamp], start=1):
                    cleaned_value = int(value)
                    sheet.cell(
                        row=row_index, column=col_index, value=cleaned_value
                    ).style = integer_style

            workbook.save(self.data_filename)
            print(f"Données enregistrées dans {self.data_filename}")

    @staticmethod
    def recup_port_Arduino():
        ports = list(serial.tools.list_ports.comports())
        for p in ports:
            if "USB-SERIAL CH340" in p.description:
                return serial.Serial(p.device, 1000000)
        return None


if __name__ == "__main__":
    reader = SerialReader()
    next_param_df = pd.read_csv("D:/simuChina/metric_test_next_param.csv")
    Re = next_param_df['Re'][0]
    duration = 0.05*1e6/Re  # Duration of reading in seconds
    h = next_param_df['h'][0]
    reader.start_reading(duration, Re, h)
