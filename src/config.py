from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "m4_monthly_dataset.tsf" # https://forecastingdata.org/
PROCESSED_DATA_PATH = DATA_DIR / "sampled_m4_150.csv"

# Параметры выборки
N_SAMPLES = 150
RANDOM_SEED = 42

# Горизонт прогнозирования
HORIZON = 18
