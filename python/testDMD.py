import numpy as np
import pandas as pd

raw_csv_path = 'C:/Users/blagn771/Documents/Aquaman/Aquaman/lily-pad-master/LilyPad/chinaBenchmark/FullMap.csv'

raw_df = pd.read_csv(raw_csv_path, header=None)
print(np.array(raw_df))