from concurrent.futures import ThreadPoolExecutor
import subprocess
import pandas as pd
import numpy as np

# Define the function to run a Processing sketch with a specific input
def run_processing(uTurn, startIndex, haato, haachama, fish, sketch_path):
    try:
        # Launch using processing-java
        cmd = [
            "C:/Users/blagn771/Desktop/software/processing-4.2/processing-java.exe",
            f"--sketch={sketch_path}",
            "--run",
            f"--args", {uTurn}, {startIndex}, {haato}, {haachama}, {fish}
        ]
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running sketch with input {uTurn}, {startIndex}, {haato}, {haachama}, {fish}: {e}")

parameters = pd.read_csv("D:/crop_nadia/list_for_automation.csv")
uTurn_list = parameters["uTurn"]
startIndex_list = parameters["start_frame"]
haato_list = parameters["video"]
haachama_list = parameters["sample"]
fish_list = ["true" for i in haachama_list]
fish2_list = ["false" for i in haachama_list]

uTurn_list = pd.concat([uTurn_list, uTurn_list], ignore_index=True)
startIndex_list = pd.concat([startIndex_list, startIndex_list], ignore_index=True)
haato_list = pd.concat([haato_list, haato_list], ignore_index=True)
haachama_list = pd.concat([haachama_list, haachama_list], ignore_index=True)
fish_list = fish_list + fish2_list

# List of inputs to try
sketch_path = "C:/Users/blagn771/Documents/Aquaman/Aquaman/lily-pad-master/LilyPad"  # Replace with actual path

# Use a pool of 3 threads
with ThreadPoolExecutor(max_workers=3) as executor:
    for uTurn, startIndex, haato, haachama, fish in zip(uTurn_list, startIndex_list, haato_list, haachama_list, fish_list):
        executor.submit(run_processing, uTurn, startIndex, haato, haachama, fish, sketch_path)

print("All sketches completed.")