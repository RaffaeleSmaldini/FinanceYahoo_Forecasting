"""
                                        Smaldini Raffaele || Ardillo Michele
                                                Politecnico di Bari
NOTE:
The architectures and training method were designed exclusively to predict the day immediately following the sequence
of lags. An autoregressive approach is used to predict up to the 7th day. This kind of approach helps the model to predict
trends and values patterns, because the attention is focused in 4 (Open High, Low, Close) values rather than 12
(4 for the day after, 4 for the next 2 day, 4 for the seventh day).
The code is designed to use CUDA, if errors are generated due to pytorch.cuda, we recommend modifying the code for
"cpu" only by removing the ".to(device)" code parts in each class.
"""
from utils import load_csv, show_trend
# torch dep
import torch
torch.cuda.empty_cache()
import torch.nn as nn
import torch.optim as optim
from BiL import BiL
from EncoderBiL import EBiL
from OptimizedTraining import TrainMethod
from ToM import ToM
from BiLPET import BiLPET
import numpy as np
#
import pandas as pd
from preprocess import dataframe_fragmentation, merge_feature, train_val_test_split, normalize, \
    data_loader, noise_sample_gen, data_cyclic_encoding
from warnings import filterwarnings

filterwarnings(category=pd.errors.PerformanceWarning, action="ignore")
pd.set_option('display.max_columns', None)
print(torch.__version__)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")
# ---------------
import os

# load data
data = load_csv(dirpath=os.path.join(os.getcwd(), "data_10Y"))     # 4 companies, 10 years each
# data = load_csv()  # LOAD MORE DATA
# show trend
show_trend(mode="m", data=data)

#
c_e = True  # see cyclic encoding function in preprocess **
lags = 45
targets = ["Open", "Close", "High", "Low"]  # the values in these columns correspond to the last value of the series
# if we use 25 lags, the 26th value will be ["Open", "Close", "High", "Low"] == value to predict --> [**2]
dataset = []
company_label = []
company_dict = {}  # generate and save a dictionary with company labels

# The for cycle iterates over a dictionary in which company is the key and file the variable (dataframe)
for company, file in data.items():
    print("Pre-processing: ", company)
    # dictionary save the preprocessed firms to use for autoregressive test
    if company not in company_label:
        company_label.append(company)
        company_dict[company] = company_label.index(company)
    # cyclic_encoding add 2 types of feature: Embed and year (in lags): helps to catch seasonality
    # dataframe_fragmentation divide the dataframe in lagged list of feature to elaborate
    # merge_feature unify the lists and generate a single dataframe adding the company label index too
    file = data_cyclic_encoding(df=file, encode=c_e)  # **
    frame = dataframe_fragmentation(df=file, lag_dim=lags)
    frame = merge_feature(lagged_list=frame, label=company_label.index(company), c_e=c_e)
    if c_e:
        # remove this value because we don't need them --> [**2]
        frame = frame.drop(columns=['year', 'Embed'])
    frame = frame.drop(columns=['Date'])
    dataset.append(frame)

# scaling x and y using a scaler for each company help to catch pattern better
X_train = []
X_val = []
X_test = []
y_train = []
y_val = []
y_test = []
x_scalers = {}
y_scalers = {}
i = 0

# # # # # defining parameters for dataset creation and normalization
normalization_mode = "minmax"
a_minmax = 0
b_minmax = 1
test_ratio = 0.05
use_gaussian_duplicate = False
NOTE = "If false, following parameters are irrelevant: "
gaussian_std = 0.1
gaussian_data_duplicate_prob = 0.5
# # # # #
for dat in dataset:
    # scales each feature independently and save the scalers to use them in the autoregressive test
    X_train_dat, X_val_dat, X_test_dat, y_train_dat, y_val_dat, y_test_dat = train_val_test_split(df=dat,
                                                                                                  target_col=targets,
                                                                                                  test_ratio=test_ratio)
    X_train_dat, y_train_dat, X_val_dat, y_val_dat, X_test_dat, y_test_dat, x_scaler, y_scaler = normalize(X_train_dat,
                                                                                                           y_train_dat,
                                                                                                           X_val_dat,
                                                                                                           y_val_dat,
                                                                                                           X_test_dat,
                                                                                                           y_test_dat,
                                                                                                           mode=normalization_mode,
                                                                                                           a=a_minmax,
                                                                                                           b=b_minmax)
    X_train.append(X_train_dat)
    X_val.append(X_val_dat)
    X_test.append(X_test_dat)
    y_train.append(y_train_dat)
    y_val.append(y_val_dat)
    y_test.append(y_test_dat)
    x_scalers[f"{company_label[i]}"] = x_scaler
    y_scalers[f"{company_label[i]}"] = y_scaler
    i += 1
# concatenate in a single np.array
X_train = np.concatenate(X_train)
X_val = np.concatenate(X_val)
X_test = np.concatenate(X_test)
y_train = np.concatenate(y_train)
y_val = np.concatenate(y_val)
y_test = np.concatenate(y_test)

# add duplicate with gaussian noise (only x)
if use_gaussian_duplicate:  # regularization technique
    X_train, y_train = noise_sample_gen(data=X_train, y=y_train,
                                        gaussian_std=gaussian_std, data_duplicate_prob=gaussian_data_duplicate_prob)
    # doesn't work well, validation could increase
# #
print("x: ", X_train.shape, "|| y: ", y_train.shape)
print("x val: ", X_val.shape, "|| y val: ", y_val.shape)
# #
batch_size = 128
n_epochs = 90
learning_rate = 1e-4
weight_decay = 1e-1
# # data loader
"""train_loader, val_loader = data_loader(x_train=X_train, x_val=X_val, y_train=y_train, y_val=y_val,
                                      shuffle=True, batch_size=batch_size)  # the time sequences are stored as raw (NOT COLUMNS!) so I can shuffle"""

train_loader, val_loader, test_loader = data_loader(x_train=X_train, x_val=X_val, y_train=y_train, y_val=y_val,
                                                    x_test=X_test, y_test=y_test,
                                                    shuffle=True,
                                                    batch_size=batch_size)  # the time sequences are stored as raw (NOT COLUMNS!) so I can shuffle"""

#loss_fn = nn.MSELoss(reduction="mean")
loss_fn = nn.SmoothL1Loss()

prep_params = {"cyclic_encoding": c_e,
               "lags": str(lags),
               "learning_rate": str(learning_rate),
               "batch": str(batch_size),
               "epochs": str(n_epochs),
               "input_dim": str(X_train.shape[1]),
               "weight_decay": str(weight_decay),
               "normalization_mode": "minimax",
               "a_minmax": str(a_minmax),
               "b_minmax": str(b_minmax),
               "test_ratio": str(test_ratio),
               "use_gaussian_duplicate": use_gaussian_duplicate,
               "NOTE: ": NOTE,
               "gaussian_std": str(gaussian_std),
               "gaussian_data_duplicate_prob": str(gaussian_data_duplicate_prob)
               }


# #

def BiL_starter():  # start BiL
    name = "BiL"
    # #
    model_params = {"input_dim": X_train.shape[1],
                    "bottleneck": 32,
                    "hidden_dim": 64,
                    "layer_dim": 6,
                    "output_dim": 4,
                    "self_attention_heads": 8,
                    "regularize": True
                    }
    model = BiL(**model_params).to(device)
    return model, model_params, name


def EBiL_starter():  # start EBIL
    name = "EBiL"
    # #
    model_params = {"input_dim": X_train.shape[1],
                    "n_head": 4,
                    "dim_forward": 64,
                    "embedding_dim": 32,
                    "n_layers": 3,
                    "output_dim": 4,
                    "hidden_dim": 64,
                    "layer_dim": 6,
                    "regularize": True}

    model = EBiL(**model_params).to(device)
    return model, model_params, name


def ToM_starter():  # start ToM
    name = "ToM"
    # #
    model_params = {"input_dim": X_train.shape[1],
                    "n_head": 4,
                    "dim_forward": 64,
                    "embedding_dim": 32,
                    "n_layers": 3,
                    "output_dim": 4,
                    "regularize": True}

    model = ToM(**model_params).to(device)
    return model, model_params, name

def BiLPET_starter():  # start BiLPET
    name = "BiLPET"
    # #
    model_params = {"input_dim": X_train.shape[1],
                    "bottleneck": 8,   # LSTM
                    "n_head": 2,    # Transformer
                    "dim_forward": 16,  # Transformer
                    "embedding_dim": 16,  # Transformer
                    "n_layers": 2,  # Transformer
                    "hidden_dim": 16,   # LSTM
                    "layer_dim": 2,     # LSTM
                    "output_dim": 4,
                    "cross_attention_heads": 2,
                    "regularize": True}

    model = BiLPET(**model_params).to(device)
    return model, model_params, name


ans = int(input(
    "1)BiL - use a Bidirectional LSTM with Self Attention\n2)EBiL - use an encoders based Transformer BiL model\n3)ToM "
    "- use and encoder based model with positional encoder\n4)BiLPET - "
    "Positional Encoder Based Transformer with Cross Attention between BiLSTM output; (q = out_transformer)\n:"))
if ans == 1:
    print("Using BiL")
    model, model_params, name = BiL_starter()
elif ans == 2:
    print("Using EBiL")
    model, model_params, name = EBiL_starter()
elif ans == 3:
    print("Using ToM")
    model, model_params, name = ToM_starter()
elif ans == 4:
    print("Using BiLPET")
    model, model_params, name = BiLPET_starter()
else:
    exit()
print(model)
# training process
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
opt = TrainMethod(model=model, loss_fn=loss_fn, optimizer=optimizer, device=device, model_params=model_params,
                  x_scaler=x_scalers, y_scaler=y_scalers, prep_params=prep_params, companies=company_dict)
opt.train(train_loader, val_loader=val_loader, batch_size=batch_size, n_epochs=n_epochs, n_features=X_train.shape[1],
          model_name=name)
opt.plot_losses()
# predictions test
predictions, values = opt.evaluate(test_loader, batch_size=batch_size, n_features=X_train.shape[1])
