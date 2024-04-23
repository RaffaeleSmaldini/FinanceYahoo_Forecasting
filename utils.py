from tqdm import tqdm
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

def load_csv(dirpath=os.path.join(os.getcwd(), 'data')):
    """
    load from directory csv files only
    :param dirpath:
    :return: dictionary of dataframes
    """
    files = os.listdir(dirpath)
    print("Files found: ", files)
    data = {}
    for file in files:
        name, ext = os.path.splitext(file)
        if "_" in name:
            name = name.split("_")
        if ext == '.csv':
            try:
                df = pd.read_csv(os.path.join(dirpath, file))
                data[name] = df
            except Exception as e:
                print(e)
        else:
            print(f"{file} is not .csv; pass")
    return data


def trend_plot(df):
    df = clean_data(df=df)
    for column_name in df.columns:
        plt.plot(df[column_name], label=column_name)
    plt.legend()
    plt.show()

def clean_data(df, columns=['Date', 'Adj Close', 'Volume']):
    df = df.drop(columns=columns)
    return df


def show_trend(data, mode='s'):
    types = ['s', 'm']
    """
    mode = "s" -- single file : show one trend; data must be a dataframe
    mode = "m" -- multiple files : show all data trend; data must be a dictionary of dataframes
    :return: 0
    """
    if mode == 's':
        try:
            if isinstance(data, pd.DataFrame):
                trend_plot(df=data)
            else:
                print(f"with {mode} --mode, data must be a pandas dataframe")
        except Exception as e:
            print(e)
    elif mode == 'm':
        try:
            if isinstance(data, dict):
                # Create subplots
                num_subplots = len(data)
                fig, axes = plt.subplots(1, num_subplots, figsize=(15, 5))

                for i, (name, df) in enumerate(data.items()):
                    df = clean_data(df=df)
                    ax = axes[i]
                    df.plot(ax=ax)
                    ax.set_title(name)
                plt.tight_layout()
                plt.show()
            else:
                print(f"with {mode} --mode, data must be a dictionary of pandas dataframes")
        except Exception as e:
            print(e)
    else:
        print(f"no mode {mode} defined: please use {types}")


def cyclical_encoder(df, col_name, period, start_num=0):
    kwargs = {
        f'sin_{col_name}' : lambda x: np.sin(2*np.pi*(df[col_name]-start_num)/period),
        f'cos_{col_name}' : lambda x: np.cos(2*np.pi*(df[col_name]-start_num)/period)
             }
    return df.assign(**kwargs).drop(columns=[col_name])


def split_values(df):
    df_list = []
    for col in df.columns:
        if col != 'Date':
            data = {"Date": df["Date"], f"{col}": df[col]}
            new_df = pd.DataFrame(data=data)
            df_list.append(new_df)
    return df_list

def generate_time_lags(df, n_lags, feature):
    df_n = df.copy()
    for n in range(1, n_lags + 1):
        df_n[f"lag{n}_{feature}"] = df_n[feature].shift(n)
    df_n = df_n.iloc[n_lags:]
    return df_n
