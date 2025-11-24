import pandas as pd
from pathlib import Path

BASE_DIR = Path(".").absolute()
CSV_PATH = BASE_DIR / "general_dataset" / "groundtruth" / "csv" / "train_big_size_A_B_E_K_WH_WB.csv"

try:
    df = pd.read_csv(CSV_PATH)
    print("Unique labels in CSV:", df['Label'].unique())
except Exception as e:
    print(f"Error reading CSV: {e}")
