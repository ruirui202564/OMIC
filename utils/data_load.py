import numpy as np
import pandas as pd
from scipy.io import loadmat

def seed_everything(seed: int):
    np.random.seed(seed)

seed_everything(42)
def capricious(X, ratio):
    X_haphazard, mask = mask_types_nan(X, ratio, seed=1)
    return X_haphazard, mask

def mask_types_nan(X, mask_num, seed):
    X_masked = np.copy(X)
    mask_indices = []
    mask = np.ones_like(X)
    num_rows = X_masked.shape[0]
    num_cols = X_masked.shape[1]

    for i in range(num_rows):
        np.random.seed(seed * num_rows - i)

        nan_mask = np.isnan(X[i, :])
        nan_indices = np.where(nan_mask)[0]

        for nan_idx in nan_indices:
            X_masked[i, nan_idx] = np.nan
            mask[i, nan_idx] = 0
            mask_indices.append((i, nan_idx))

        num_mask = int(mask_num * num_cols)

        num_already_masked = len(nan_indices)
        num_additional_mask = max(0, num_mask - num_already_masked)

        if num_additional_mask > 0:
            non_nan_indices = np.where(~nan_mask)[0]

            if len(non_nan_indices) >= num_additional_mask:
                rand_idx = np.random.choice(non_nan_indices, num_additional_mask, replace=False)
            else:
                rand_idx = non_nan_indices

            for idx in rand_idx:
                X_masked[i, idx] = np.nan
                mask[i, idx] = 0
                mask_indices.append((i, idx))

    return X_masked, mask

def data_load_imbalance_binary(data_name, ratio):
    file = './data/binary/' + data_name + '.xlsx'
    df = pd.read_excel(file)
    data = df.values

    X = data[:, :-1]
    X = X.astype(float)
    X_haphazard, mask = capricious(X, ratio)

    Y = data[:, -1]
    Y = Y.reshape(-1, 1)
    unique_values = np.unique(Y)
    if len(unique_values) != 2:
        print(f"警告: 发现{len(unique_values)}个类别值，期望是2个")
    if set(unique_values) == {1, 2}:
        Y[np.where(Y == 2)] = 0
    Y = Y.astype(int)

    return X, Y, X_haphazard, mask


def data_load_imbalance_binary_other(data_name, ratio):
    file = './data/binary/' + data_name + '.mat'
    mat_data = loadmat(file)
    data = mat_data['data']
    X = data[:, 1:data.shape[1]]
    X = X.astype(float)

    X_haphazard, mask = capricious(X, ratio)

    Y = data[:, 0]
    Y = Y.reshape(-1, 1)
    Y[np.where(Y == -1)] = 0
    Y = Y.astype(int)

    return X, Y, X_haphazard, mask

def data_load_imbalance_multi(data_name, ratio):
    file = './data/multi-class/' + data_name + '.xlsx'
    df = pd.read_excel(file)
    data = df.values

    X = data[:, :-1]
    X = X.astype(float)
    X_haphazard, mask = capricious(X, ratio)

    Y = data[:, -1]
    Y = Y.reshape(-1, 1)
    unique_values = np.unique(Y)
    if np.min(Y) >0:
        min_val = np.min(Y)
        Y = Y-min_val
    Y = Y.astype(int)

    return X, Y, X_haphazard, mask

def data_load_imbalance_multi_syn(data_name, ratio):
    file = './data/syn/' + data_name + '.xlsx'
    df = pd.read_excel(file)
    data = df.values

    if 'dynamic' in data_name:
        X = data[:, 1:-1]
    else:
        X = data[:, :-1]
    X = X.astype(float)
    X_haphazard, mask = capricious(X, ratio)

    Y = data[:, -1]
    Y = Y.reshape(-1, 1)
    unique_values = np.unique(Y)
    if np.min(Y) >0:
        min_val = np.min(Y)
        Y = Y-min_val
    Y = Y.astype(int)

    return X, Y, X_haphazard, mask