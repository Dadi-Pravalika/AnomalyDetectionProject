
import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess(file_path, seq_len):
    df = pd.read_csv(file_path)
    df = df.select_dtypes(include=['float64', 'int64'])
    scaler = MinMaxScaler()
    data = scaler.fit_transform(df)
    X, Y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        Y.append(data[i+1:i+seq_len+1])
    X = torch.tensor(np.array(X), dtype=torch.float32)
    Y = torch.tensor(np.array(Y), dtype=torch.float32)
    return X, Y
import pandas as pd
import numpy as np
import torch

def load_and_preprocess(path, seq_len):
    df = pd.read_csv(path)
    data = df.values.astype(np.float32)

    X = []
    Y = []

    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        Y.append(data[i+seq_len])

    X = torch.tensor(X)
    Y = torch.tensor(Y)

    return X, Y
def normalize_data(data):
    """
    Normalize input features using Min-Max scaling
    """
    return (data - data.min()) / (data.max() - data.min() + 1e-8)
