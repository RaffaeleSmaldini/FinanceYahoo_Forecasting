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
You can find all the architectures in our .pdf file. It explains our testing, design process, and metrics for a deeper understanding of each of the models we considered.
In the pdf file there are also several results of the autoregression test. I planned to put the different tests on a simplified table here on GitHub.
Our models simultaneously predict the values of [Open, High, Low, Close] and require the label of the company whose trend you wish to forecast (such as APPL, AMZN, GOOGL, MSFT). Each model undergoes training using all available data, excluding testing data. However, we are cognizant of a potential issue related to the training process: the utilization of training and prediction time lags (specifically, a single following day) not in chronological order. To elaborate, when predicting stock data, it's typical to utilize data from a time period preceding the testing data. In our case, the training and the first type of testing data do not follow this chronological ordering convention. The following is an example of the issue: 
>Suppose we have historical stock data for a company like Apple (APPL) for the month of January. We want to predict the stock prices for February.
In a typical scenario with chronological ordering:
We would train our model using historical data from January (e.g., from January 1st to January 28th).
Then, we would test our model's performance using data from February (e.g., from February 1st to February 28th). The model would make predictions for February based on what it learned from January.
However, if our training and testing data are not in chronological order (since we decided to random mix our tensors):
We included some data from February in our training set, which the model should not have access to during training.
This could lead to a scenario where the model is inadvertently "peeking into the future" during training, potentially biasing its performance evaluation and leading to overly optimistic results.
To rectify this, we would need to ensure that our training data strictly precedes our testing data in terms of chronological order, as this aligns with the real-world scenario where we predict future values based on past data.  In the first type of testing this was  not considered but in the second one (auto-regressive) the data are unseen from the model during the training and they are from the actual "future".

...

