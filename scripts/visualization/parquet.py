
import os
import pandas as pd

from pathlib import Path
import pandas as pd

# 1. Get the directory of parquet.py (.../scripts/visualization)
script_dir = Path(__file__).resolve().parent

# 2. Go up two levels to the 'Thesis' folder, then down into 'results/runs'
data_dir = script_dir.parent.parent / 'results' / 'runs' /'sac_v27_best_model'

# 3. Target the specific file
file_path = data_dir / 'sac_perfect_det_wet_rice_100pct_seed1.parquet'


# Using repr() to catch any hidden whitespace, zero-width spaces, or newline characters in the string
print(f"Exact string: {repr(str(file_path))}")
print(f"Does OS see it?: {os.path.exists(file_path)}")

# 4. Load the dataframe
df = pd.read_parquet(file_path, engine='pyarrow')


# Display first few rows
print(df.head())

# Basic operations
print(df.columns)  # Column names
print(df.describe())  # Summary statistics
df.to_csv(r'C:\Users\Tara\Documents\Thesis\results\runs\data_preview.csv', index=False)