from utils import generate_time_lags, split_values, clean_data, cyclical_encoder
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import random


def dataframe_fragmentation(df, lag_dim=25):
    """
    take a dataframe and lag dimension to create more dataframes
    :param df: input dataframe (with Open, Close, High and Low)
    :param lag_dim: dimensions of lags in days
    :return: a list of dataframes (Open, Close, High and Low divided)
    """
    try:
        df = clean_data(df, columns=['Adj Close', 'Volume'])
    except:
        print("dataframe already cleaned")
    list_df = split_values(df=df)
    lag_data = []
    for df_feature in list_df:
        lag_df = generate_time_lags(df_feature, lag_dim, feature=df_feature.columns[1])
        # Reverse the order of columns -> the lag method reverse the order, that's why we reverse back
        reversed_df = lag_df.iloc[:, ::-1]
        lag_data.append(reversed_df)
    return lag_data


def merge_feature(lagged_list, label, c_e):
    """
    take the list with lags produced by dataframe_fragmentation and unify all the columns (Open, Close, High and Low
    lags) that have the same Date.
    :param lagged_list: list of dataframes
    :return: df of lags
    """
    if c_e:
        df1, df2, df3, df4, df5, df6 = lagged_list
    else:
        df1, df2, df3, df4 = lagged_list
    # Merge dataframes based on the "Date" column
    merged_df = df1.merge(df2, on="Date", how="inner")
    merged_df = merged_df.merge(df3, on="Date", how="inner")
    merged_df = merged_df.merge(df4, on="Date", how="inner")
    if c_e:
        merged_df = merged_df.merge(df5, on="Date", how="inner")
        merged_df = merged_df.merge(df6, on="Date", how="inner")
    merged_df["company_label"] = [label] * merged_df.shape[0]
    return merged_df


def feature_label_split(df, target_cols):
    y = df[target_cols]
    X = df.drop(columns=target_cols)
    return X, y


def train_val_test_split(df, target_col, test_ratio):
    val_ratio = test_ratio / (1 - test_ratio)
    X, y = feature_label_split(df, target_col)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio, shuffle=False)
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_val_split(df, target_cols, val_ratio):
    X, y = feature_label_split(df, target_cols)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_ratio, shuffle=False)
    return X_train, X_val, y_train, y_val


def normalize(x_train, y_train, x_val=None, y_val=None, x_test=None, y_test=None, mode="minmax", a=0, b=1):
    types = ["minmax", "standard"]
    if mode not in types:
        print(f"ERROR: {mode} is not valid. Please use: {types}")
    if mode == "minmax":
        if a == 0 and b == 1:
            print("WARNING: You can set min and max value using 'a' and 'b' argument. Default: a=0, b=1")
        x_scaler = MinMaxScaler(feature_range=(a, b))
        y_scaler = MinMaxScaler(feature_range=(a, b))
    elif mode == "standard":
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
    # # # # # # #
    # init x scaler
    x_scaler.fit(X=x_train, y=y_train)
    # init y scaler
    y_scaler.fit(X=y_train)
    X = x_scaler.fit_transform(X=x_train)
    if (x_val is not None) and (y_val is not None):
        X_val = x_scaler.transform(x_val)
        if (x_test is not None) and (y_test is not None):
            x_test = x_scaler.transform(x_test)
    Y = y_scaler.fit_transform(y_train)
    if (x_val is not None) and (y_val is not None):
        Y_val = y_scaler.transform(y_val)
        if (x_test is not None) and (y_test is not None):
            y_test = y_scaler.transform(y_test)
            return X, Y, X_val, Y_val, x_test, y_test, x_scaler, y_scaler
        return X, Y, X_val, Y_val, x_scaler, y_scaler
    return X, Y, x_scaler, y_scaler


def data_loader(x_train, y_train, x_val=None, y_val=None, x_test=None, y_test=None, batch_size=64, shuffle=False):
    train_features = torch.Tensor(x_train)
    train_targets = torch.Tensor(y_train)
    train = TensorDataset(train_features, train_targets)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    if (x_val is not None) and (y_val is not None):
        val_features = torch.Tensor(x_val)
        val_targets = torch.Tensor(y_val)
        val = TensorDataset(val_features, val_targets)
        val_loader = DataLoader(val, batch_size=batch_size, shuffle=shuffle, drop_last=True)
        if (x_test is not None) and (y_test is not None):
            test_features = torch.Tensor(x_test)
            test_targets = torch.Tensor(y_test)
            test = TensorDataset(test_features, test_targets)
            test_loader = DataLoader(test, batch_size=batch_size, shuffle=shuffle, drop_last=True)
            return train_loader, val_loader, test_loader
        return train_loader, val_loader
    return train_loader


def noise_sample_gen(data, y, gaussian_std=0.1, data_duplicate_prob=0.3):
    """
    Function that add noisy data based on data_duplicate_probability: handled with a uniform distribution prob
    :param data_duplicate_prob: probability to add a duplicate row perturbed with gaussian noise
    :param data: input matrix
    :param y: output value associated with i-th matrix row
    :param gaussian_std: gaussian standard deviation
    :return: updated data and y
    """
    num_samples, features = data.shape
    noisy_data_matrix = data
    new_y = y
    for i in range(num_samples):
        try:
            np.random.seed(round(i * num_samples * gaussian_std * np.random.uniform(0, 1)))
        except:
            np.random.seed(i)
        if np.random.uniform(0, 1) <= data_duplicate_prob:
            original_line = data[i, :]
            original_y = y[i, :]
            # Duplicate the original line
            duplicated_line = original_line.copy()
            # generate a vector with normal distribution based on std and feature len
            gaussian_noise = np.random.normal(0, gaussian_std, features - 1)  # not adding noise to company label
            gaussian_noise = np.concatenate((gaussian_noise, [0]), axis=0)  # stack 0 to sum with company label
            duplicated_line += gaussian_noise

            # append duplicate (copying the y data too) to noisy data matrix
            noisy_data_matrix = np.vstack((noisy_data_matrix, duplicated_line))
            new_y = np.vstack((new_y, original_y))
    return noisy_data_matrix, new_y


def cyclic_encoding_adapter(df, embed_start=1000):
    # adapt the cyclical value in a single one, to reduce dimensions from 9*lags to 2*lags
    exceptions_columns = ["Open", "High", "Low", "Close", "year", "Embed", "Date"]
    embed_column = []
    values = df.values
    for i in range(values.shape[0]):
        embed = embed_start
        for j in range(4, values.shape[1] - 1):
            embed = embed * values[i][j]
        embed_column.append(embed)
    df["Embed"] = embed_column
    columns = df.columns
    del_col = []
    for col in columns:
        if col not in exceptions_columns:
            del_col.append(col)
    df = clean_data(df=df, columns=del_col)
    return df

def data_cyclic_encoding(df, encode=True):
    """
    transform data timestamp into cyclic encoding  --> catch seasonality
    :param encode: if true encode Data field, else delete Data field only
    :param df: pandas dataframe
    :return: pandas dataframe
    """
    if encode:
        # Convert timestamp to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        # Extract relevant time features
        df['day_of_week'] = df['Date'].dt.dayofweek
        df['month'] = df['Date'].dt.month
        df['day'] = df['Date'].dt.day
        df['week_of_year'] = df['Date'].dt.isocalendar().week

        # Cyclic encoding
        df = cyclical_encoder(df, 'day_of_week', 7, 0)
        df = cyclical_encoder(df, 'month', 12, 0)
        df = cyclical_encoder(df, 'day', 31, 0)
        df = cyclical_encoder(df, 'week_of_year', 52, 0)

        # Normalize time features
        ref_year = float(max(df['Date'].dt.year))
        df['year'] = df['Date'].dt.year #/ ref_year  # Normalize to a reference year
        df = clean_data(df=df, columns=['Adj Close', 'Volume'])
        df = cyclic_encoding_adapter(df=df)
    return df
