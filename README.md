# FinanceYahoo_Forecasting
Forecasting of Yahoo finance (https://finance.yahoo.com) data using Bi-LSTM and Transformer model. Exploring auto-regressive prediction using trained model on single target "value" [Open, High, Low, Close]. 
## Forecasting project basics
The aim of the project concerns the design, implementation and evaluation of deep learning models for stock price prediction in the finance domain. The dataset consists of historical stock prices of four related companies: Google,
Microsoft, Amazon, and Apple in the NasdaqGS market. The goal is to predict the next-day, the 2-days-after, and the next-week stock prices for Open, High, Low, and Close values for each company.
The datasets used for this porpouse are the: Apple (APPL), Amazon (AMZN), Google (GOOGL), and Microsoft (MSFT) .csv files (You can find them at *https://finance.yahoo.com*). You can find all the data used in *"data"* dir (smaller dataset) or for a bigger dataset in *"data_10Y"* dir.
The *"45lag_test_data"*  and *"test_data"* dirs will be used for the autoregressive tests. The difference between these 2 directories of files concerns the length of the "lags", i.e. the days observed useful for the model's predictions. 

### Model Architecture
The model architecture are "fixed" since this was a project for our Deep Learning course exam. We had to build at least three models, a transformer-like model, a Bi-LSTM model, and a model that combines both of them.
We decided to use an Encoder based approach for those involving transformers. 
The primary focus was on discovering the optimal blend of "light models" (in parameters), effective metrics incorporating a short trend window (lags) of Open, High, Low, Close values, and models adept at capturing trend performance beyond just MSE and MAE. The architectures were tailored specifically for predicting the subsequent day following the lag sequence, with an autoregressive method chosen to forecast up to the 7th day. Due to the diverse model architectures, two distinct evaluations were conducted:
1. The initial assessment occurred immediately post-training to identify models exhibiting strong metrics (MAE, MSE) in predicting the day following the lags.
2. The second evaluation utilized an autoregressive technique to extend forecasting capability up to the 7th day.

***Although these models possess the capability to forecast beyond this timeframe, it's advised not to surpass lags/2 days to prevent potential declines in model accuracy (both in trend and metrics).***

> [!NOTE]
>***You can find all the architectures in our .pdf file [Report 2023](reports/SMALDINI_ARDILLO_Report_IIB.pdf).*** 
It explains our testing, design process, and metrics for a deeper understanding of each of the models we considered.
In the pdf file there are also several results of the autoregression test. I planned to put the different tests on a simplified table here on GitHub.
Our models simultaneously predict the values of [Open, High, Low, Close] and require the label of the company whose trend you wish to forecast (such as APPL, AMZN, GOOGL, MSFT). Each model undergoes training using all available data, excluding testing data. 
> [!WARNING]
>However, we are cognizant of a potential issue related to the training process: the utilization of training and prediction time lags (specifically, a single following day) not in chronological order. To elaborate, when predicting stock data, it's typical to utilize data from a time period preceding the testing data. In our case, the training and the first type of testing data do not follow this chronological ordering convention.

The following is an example of the issue: 
>Suppose we have historical stock data for a company like Apple (APPL) for the month of January. We want to predict the stock prices for February.
In a typical scenario with chronological ordering:
We would train our model using historical data from January (e.g., from January 1st to January 28th).
Then, we would test our model's performance using data from February (e.g., from February 1st to February 28th). The model would make predictions for February based on what it learned from January.
However, if our training and testing data are not in chronological order (since we decided to random mix our tensors):
We included some data from February in our training set, which the model should not have access to during training.
This could lead to a scenario where the model is inadvertently "peeking into the future" during training, potentially biasing its performance evaluation and leading to overly optimistic results.
To rectify this, we would need to ensure that our training data strictly precedes our testing data in terms of chronological order, as this aligns with the real-world scenario where we predict future values based on past data.  In the first type of testing this was  not considered but in the second one (auto-regressive) the data are unseen from the model during the training and they are from the actual "future".

> [!NOTE]
> **We found good performances of the autoregressive tests despite this problem with the first training (technically if the models had biases the autoregression results should be very bad). _I will train new models that follow conventions and use data with "time gaps" between training and training data._**

## Installation
To use my code, follow these steps (choose one between 1a and 1b):
### 1a. Clone this repository
```
git clone https://github.com/RaffaeleSmaldini/FinanceYahoo_Forecasting.git
```
### 1b. Download this repository
Download the repository and extract it.

### 2. Install requirements.txt
```
pip install -r requirements.txt
```
**Make sure to run this command in the same directory where the requirements.txt file is located.**
### 3. Install PyTorch CUDA
This project uses CUDA, so we advice to install pytorch CUDA using **_https://pytorch.org/_** and following the site instructions;
In our case we used the following command to install the CUDA version.: 
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
``` 
If you do not install CUDA version, you may need to modify the code to run cpu only.
All the others modules/library are included in the requirments.txt.
### 4. Train models 
All the code to train a model is located in *"main.py"*, if you want to solve *the problem described in the previous paragraphs* you will currently have to modify the code slightly.

## Autoregressive models usage
You will find the different trained models in "models.rar", you will have to unzip in the same location (you can also use a model you train). Make sure the models are in a *"models"* directory. To start the autoregression test you need to use *"run_testing.py"*; here you can find: 
```
ates = AutoTest(model_dir=["LARGE_ToM", "LARGE_BiLPET", "SMALL_ToM", "SMALL_EBiL"], model_type=["ToM", "BiLPET", "ToM", "EBiL"])
ates.start_autoregressive_bagging(data=test_data[lab], company=lab, autoregression_days=7, plot_all_graphs=False)
```
*"start_autoregressive_begging"* method will predict for the same test lags, based on *"autoregressive_days"* set value, the different values for each model and then mediate them based on model presence. 

```
ates = AutoTest(model_dir=["45LAG_SMALL_ToM", "45LAG_SMALL_BiL", "45LAG_SMALL_ToM"], model_type=["ToM", "BiL", "ToM"])
ates.start_autoregressive_bagging(data=test_data[lab], company=lab, autoregression_days=7, plot_all_graphs=False)
```
*In the same way as the previous one, but uses data with a greater number of lags.*

```
ates = AutoTest(model_dir="45LAG_SMALL_BiL", model_type="BiL")
ates.start_autoregression(data=test_data[lab], company=lab, autoregression_days=7)
```
*It uses a single model to predict the values for the next 7 days (since autoregression_days=7 in this case).*

If you want to use different test data than the ones we use, you will have to download them from *https://finance.yahoo.com* and place them in the appropriate folders, modifying the names of the .csv files as required.

## Citing Our Work :+1: 

If you use our work or code in your research, please cite our repository:

[RaffaeleSmaldini/FinanceYahoo_Forecasting](https://github.com/RaffaeleSmaldini/FinanceYahoo_Forecasting)


