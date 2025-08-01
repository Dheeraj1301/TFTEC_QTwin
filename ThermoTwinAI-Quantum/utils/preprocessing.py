# utils/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_split_data(path, window_size=15):
    df = pd.read_csv(path)
    data = df['CoP'].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i, 0])
        y.append(data[i, 0])

    X, y = np.array(X), np.array(y)

    split = int(len(X) * 0.8)
    return X[:split], y[:split], X[split:], y[split:]
