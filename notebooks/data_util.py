"""Data Util to support load data or instances in these notebooks."""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

NOTEBOOK_DIR = "."
WORK_DIR = ".."
DATA_DIR = f"{WORK_DIR}/tests/data"
assert os.path.exists(NOTEBOOK_DIR), f"Not found folder {NOTEBOOK_DIR}"
assert os.path.exists(WORK_DIR), f"Not found folder {WORK_DIR}"
assert os.path.exists(DATA_DIR), f"Not found folder {DATA_DIR}"

# Data Functions
def load_data_set_bejin():
    data_link = f"{DATA_DIR}/pollution.csv"
    assert os.path.exists(data_link), f"Not found folder {data_link}"
    df = pd.read_csv(data_link)

    # Set date-time as index
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    # Encoding wind_direction to integer
    encoder = LabelEncoder()
    df["wind_direction"] = encoder.fit_transform(df["wind_direction"])
    return df   

def get_xy_scalers(df, independents, dependent):
    # Normalization
    x_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaler = MinMaxScaler(feature_range=(0, 1))
    x_scaler.fit(df[independents].values)
    y_scaler.fit(df[dependent].values.reshape(-1, 1))
    return x_scaler, y_scaler

def get_instance_x(df, n_steps, independents):
    start = np.random.randint(0, len(df.index) - n_steps - 1)
    end = start + n_steps
    x_df = df[start:end].copy()
    x_df = x_df.reset_index()
    x_df = x_df.loc[:, independents]
    return x_df