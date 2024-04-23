# FinanceYahoo_Forecasting
Forecasting of Yahoo finance (https://finance.yahoo.com) data using Bi-LSTM and Transformer model. Exploring auto-regressive prediction using trained model on single target "value" [Open, High, Low, Close]. 
## Forecasting project basics
The aim of the project concerns the design, implementation and evaluation of deep learning models for stock price prediction in the finance domain. The dataset consists of historical stock prices of four related companies: Google,
Microsoft, Amazon, and Apple in the NasdaqGS market. The goal is to predict the next-day, the 2-days-after, and the next-week stock prices for Open, High, Low, and Close values for each company.
The datasets used for this porpouse are the: Apple (APPL), Amazon (AMZN), Google (GOOGL), and Microsoft (MSFT) .csv files (You can find them at *https://finance.yahoo.com*)
### Model Architecture
The model architecture are "fixed" since this was a project for our Deep Learning course exam. We had to build at least three models, a transformer-like model, a Bi-LSTM model, and a model that combines both of them.
We decided to use an Encoder based approach for those involving transformers. 
The primary focus was on discovering the optimal blend of "light models" (in parameters), effective metrics incorporating a short trend window (lags) of Open, High, Low, Close values, and models adept at capturing trend performance beyond just MSE and MAE. The architectures were tailored specifically for predicting the subsequent day following the lag sequence, with an autoregressive method chosen to forecast up to the 7th day. Due to the diverse model architectures, two distinct evaluations were conducted:
1. The initial assessment occurred immediately post-training to identify models exhibiting strong metrics (MAE, MSE) in predicting the day following the lags.
2. The second evaluation utilized an autoregressive technique to extend forecasting capability up to the 7th day.
***Although these models possess the capability to forecast beyond this timeframe, it's advised not to surpass lags/2 days to prevent potential declines in model accuracy (both in trend and metrics).***
You can find all the architectures in our .pdf file. It explains our testing, design process, and metrics for a deeper understanding of each of the models we considered.
In the pdf file there are also several results of the autoregression test. I planned to put the different tests on a simplified table here on GitHub.





