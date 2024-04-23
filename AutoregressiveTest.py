import os
import torch
import joblib
import json
import pandas as pd
from BiL import BiL
from EncoderBiL import EBiL
from ToM import ToM
from BiLPET import BiLPET
import numpy as np
from tqdm import tqdm
from utils import clean_data
from preprocess import data_cyclic_encoding
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
device = "cuda" if torch.cuda.is_available() else "cpu"
from warnings import filterwarnings

filterwarnings(category=pd.errors.PerformanceWarning, action="ignore")
filterwarnings(category=UserWarning, action="ignore")


class AutoTest:
    def __init__(self, models_dir=os.path.join(os.getcwd(), "models"), model_dir=None, model_type="BiL"):
        """
        :param models_dir: models' directory
        :param model_dir: model's directory
        :param model_type: model name
        """
        if isinstance(model_dir, list):
            print("WARNING: model_dir argument found a list, you are trying to do bagging with more model;"
                  " model_type must be a list of ordered element!")
            if len(model_dir) <= 1:
                print("WARNING: You are using 1 or less model; input the model "
                      "as a string instead and use 'start_autoregression'.")
                exit()
            if isinstance(model_type, list):
                print("WARNING: Model Bagging work with 'start_autoregressive_bagging'. ")
                self.bagging = True
                self.models_dir = models_dir
                self.bagging_dirs = model_dir
                self.models_type = model_type
            else:
                self.bagging = False
        else:
            self.bagging = False
        self.bool_acc = True    # -> need to modify the code for regression without future data
        # add a way to embed the "Data" using preprocess encoder adapter + use the timestamp year
        # || order: [Open, High, Low , Close, year, Embed]
        if not self.bagging:
            if model_dir is None:
                print("ERROR: model_dir is unfilled, please add a valid model directory")
                exit()
            self.model_path = os.path.join(models_dir, model_dir)
            # #
            with open(os.path.join(self.model_path, "companies.json"), 'r') as json_file:
                self.companies = json.load(json_file)
            print("Registered companies during training: ", self.companies)
            # # load scalers
            self.x_scalers = joblib.load(os.path.join(self.model_path, "xscaler.pkl"))
            self.y_scalers = joblib.load(os.path.join(self.model_path, "yscaler.pkl"))
            # # load params
            with open(os.path.join(self.model_path, "model_params.json"), 'r') as json_file:
                params = json.load(json_file)
            with open(os.path.join(self.model_path, "prep_params.json"), 'r') as json_file:
                prep_params = json.load(json_file)
                self.max_train_lag = int(prep_params["lags"])
            self.labels = ["Open", "High", "Low", "Close"]  # label order
            self.c_e = prep_params["cyclic_encoding"]
            # # load model
            if model_type == "BiL":
                self.model = BiL(**params).to(device)
            elif model_type == "EBiL":
                self.model = EBiL(**params).to(device)
            elif model_type == "ToM":
                self.model = ToM(**params).to(device)
            elif model_type == "BiLPET":
                self.model = BiLPET(**params).to(device)
            else:
                print(f"No model types with name: {model_type}")
                exit()
            self.model.load_state_dict(torch.load(os.path.join(self.model_path, model_type)))
            self.model.eval()

    def __process_pipe__(self, data):
        data = data_cyclic_encoding(df=data, encode=self.c_e)
        if not self.c_e:
            data = clean_data(data)
        else:
            data = clean_data(data, columns=["Date"])
        if not self.bagging:
            print(data)
        # Select the first "lags" values of each column
        selected_data = data.iloc[:self.max_train_lag, :].values
        # concatenate columns -------
        original_array = np.array(selected_data)
        # Get the number of columns in the original array
        num_columns = original_array.shape[1]
        # Initialize an empty list to store the concatenated values
        concatenated_values = []
        # Iterate through each column and concatenate its values
        for i in range(num_columns):
            concatenated_values.extend(original_array[:, i])
        # Convert the result to a NumPy array if needed
        result_array = np.array(concatenated_values)
        # ----------------------------
        result_array = np.append(arr=result_array, values=self.company_label)
        #
        if self.bool_acc:
            # Store the remaining values in separate arrays to use for predictions
            remaining_data = data.iloc[self.max_train_lag:, :]
            # Separate arrays for the remaining columns
            self.open_ = remaining_data['Open'].values
            self.high_ = remaining_data['High'].values
            self.low_ = remaining_data['Low'].values
            self.close_ = remaining_data['Close'].values
            if self.c_e:
                self.year = remaining_data['year'].values
                self.em = remaining_data['Embed'].values
        return result_array

    def regen_input(self, inr, p):
        regen_in = self.x_scaler.inverse_transform(inr.cpu().detach().numpy().reshape(1, -1))
        regen_in = regen_in[0][:-1]
        sep1 = regen_in[0:self.max_train_lag]
        sep2 = regen_in[self.max_train_lag: self.max_train_lag * 2]
        sep3 = regen_in[self.max_train_lag * 2: self.max_train_lag * 3]
        sep4 = regen_in[self.max_train_lag * 3: self.max_train_lag * 4]
        if self.c_e:
            sep5 = regen_in[self.max_train_lag * 4: self.max_train_lag * 5]
            sep6 = regen_in[self.max_train_lag * 5: self.max_train_lag * 6]
            stack = [sep1, sep2, sep3, sep4, sep5, sep6]
        else:
            stack = [sep1, sep2, sep3, sep4]
        stack2 = []
        concat_in = []
        i = 0
        for s in stack:
            g = s[1:]
            if i <= 3:
                g = np.append(arr=g, values=p[0][i])
            elif i == 4:
                g = np.append(arr=g, values=self.year[0])
                self.year = np.delete(self.year, [0])
            elif i == 5:
                g = np.append(arr=g, values=self.em[0])
                self.em = np.delete(self.em, [0])
            i += 1
            stack2.append(g)
        for s in stack2:
            for i in range(len(s)):
                concat_in.append(s[i])
        concat_in.append(self.company_label)
        concat_in = np.array(concat_in).reshape(1, -1)
        input_data = self.x_scaler.transform(X=concat_in)
        # add 2 dimensions to fit the model shape
        input_data = torch.Tensor(input_data).unsqueeze(0).to(device)
        return input_data

    def __metrics__(self, values, predictions):
        # Concatenate the lists into numpy arrays
        true_values = np.concatenate(values, axis=0)
        predicted_values = np.concatenate(predictions, axis=0)
        print(true_values.shape, predicted_values.shape)
        true_values = self.y_scaler.transform(true_values)
        predicted_values = self.y_scaler.transform(predicted_values)
        #
        #  MAE for each column
        mae = mean_absolute_error(true_values, predicted_values, multioutput='raw_values')

        # MSE for each column
        mse = mean_squared_error(true_values, predicted_values, multioutput='raw_values')

        # PDA (Percentage of Direction Accuracy) for each column
        pdas = []
        for col in range(true_values.shape[1]):
            true_series = true_values[:, col]  # Extract the true series for the column
            predicted_series = predicted_values[:, col]
            pda = self.pda_calculation(true=true_series, predicted=predicted_series)
            pdas.append(pda)
        mae = list(mae); mse = list(mse); pda = list(pdas)
        label = ["Open", "High", "Low", "Close"]
        save_dict = {}
        for i in range(len(mae)):
            save_dict[f"{label[i]}_mae"] = str(mae[i])
            save_dict[f"{label[i]}_mse"] = str(mse[i])
            save_dict[f"{label[i]}_pda[%]"] = str(pda[i])
        print(save_dict)
        if self.bagging:
            self.bagging_metrics.append(save_dict)

    def pda_calculation(self, true, predicted):
        # Calculate the difference between each price and the previous one
        true_movements = np.diff(true)
        predicted_movements = np.diff(predicted)

        # Determine the signs of the price movements
        true_directions = np.sign(true_movements)
        predicted_directions = np.sign(predicted_movements)
        # (1 for positive, -1 for negative) --> the direction will be 1 if the value is increasing and -1 if decreasing

        # Count the number of correctly predicted directions, using the sign function result
        correct_predictions = np.sum(true_directions == predicted_directions)

        # PDA (in %) formula
        total_predictions = len(true_directions)
        pda = (correct_predictions / total_predictions) * 100

        return pda

    def start_autoregression(self, data, company, autoregression_days=7):
        """
        :param data: must a pandas dataframe (first "lag" value will be used as input of the model,
             the others will be used to calculate accuracy)
        :param company_label: label of the company
        :param autoregression_days: days of recursive prediction
        :return:
        """
        try:
            label = self.companies[company]
            print(f"{company}: {label}")
        except:
            print(f"The label '{company}' is not in companies label")
        self.x_scaler = self.x_scalers[company]
        self.y_scaler = self.y_scalers[company]
        self.company_label = label
        predictions = []
        values = []
        if ((data.shape[0] - self.max_train_lag) < autoregression_days) and self.bool_acc:
            print(f"ERROR: days of autoregression has been set to {autoregression_days}, "
                  f"but the length of data to use for calculate accuracy is {len(data) - self.max_train_lag}")
            exit()
        if autoregression_days >= self.max_train_lag // 2:
            print("WARNING: You will predict using half or more data that are generated from the model, "
                  "so the accuracy may decrease based on the previous predictions.")
        input_data = self.__process_pipe__(data=data).reshape(1, -1)
        print(f"Data input shape: {input_data.shape} || autoregressive prediction len: {autoregression_days} ")
        input_data = self.x_scaler.transform(X=input_data)
        # add 1 dimensions to fit the model shape
        input_data = torch.Tensor(input_data).unsqueeze(0).to(device)
        for i in tqdm(range(autoregression_days)):
            pred = self.model(input_data).cpu().detach().numpy()
            p = self.y_scaler.inverse_transform(pred)
            predictions.append(p)
            print(f" ------------------------------- DAY: {i+1}")
            print("NORMALIZED PRED: ", pred)
            print("PRED: ", p)  # debug -----------
            if self.bool_acc:
                print("REAL: ", self.open_[i], self.high_[i], self.low_[i], self.close_[i])  # debug -----------
                values.append([[self.open_[i], self.high_[i], self.low_[i], self.close_[i]]])
            # start regen input
            input_data = self.regen_input(input_data.squeeze(0).squeeze(0), p=p)
        print(f" --------------- METRICS ------------------" )
        self.__metrics__(values=values, predictions=predictions)
        if self.bagging:
            if self.plt_all:
                self.plot_autoregression(values, predictions)
        else:
            self.plot_autoregression(values, predictions)
        if self.bagging:
            self.bagging_pred.append(predictions)
            self.values = values

    def plot_autoregression(self, values, predictions):
        # Create a figure and axes for the plot
        values = np.array(values).squeeze(1)
        predictions = np.array(predictions).squeeze(1)

        # Create a figure and axes for the plot
        fig, ax = plt.subplots()

        # Define line styles and colors
        line_styles = ['--', '-.', ':', '-.']
        colors = ['blue', 'red', 'green', 'purple']

        # Plot real values as dashed lines
        for i in range(predictions.shape[1]):
            ax.plot(values[:, i], label=f'Real {self.labels[i]}', linestyle=line_styles[i], color=colors[i])

        # Plot predicted values as continuous lines
        for i in range(predictions.shape[1]):
            ax.plot(predictions[:, i], label=f'Predicted {self.labels[i]}', linestyle='-', color=colors[i])

        # Set labels and legend
        ax.set_xlabel('Data Points')
        ax.set_ylabel('Values')
        ax.set_title('Real vs. Predicted Values')
        ax.legend()

        # Show the plot
        plt.show()

    def start_autoregressive_bagging(self, data, company, autoregression_days=7, plot_all_graphs=False):
        # __init__ copy but works with bagging/ensemble mode
        self.plt_all = plot_all_graphs
        print(f"WARNING: {company} must be in the companies.json of all the bagging models or the code will crush")
        # modified __init__ for iterative bagging purpose
        # init variables for saving results
        self.bagging_pred = []
        self.bagging_metrics = []
        for n in range(len(self.bagging_dirs)):
            self.model_path = os.path.join(self.models_dir, self.bagging_dirs[n])
            # #
            with open(os.path.join(self.model_path, "companies.json"), 'r') as json_file:
                self.companies = json.load(json_file)
            print(f"Registered companies during {self.model_path} training: ", self.companies)
            try:
                label = self.companies[company]
                print(f"{company}: {label}")
            except:
                print(f"The label '{company}' is not in companies label")
            # # load scalers
            self.x_scalers = joblib.load(os.path.join(self.model_path, "xscaler.pkl"))
            self.y_scalers = joblib.load(os.path.join(self.model_path, "yscaler.pkl"))
            # # load params
            with open(os.path.join(self.model_path, "model_params.json"), 'r') as json_file:
                params = json.load(json_file)
            with open(os.path.join(self.model_path, "prep_params.json"), 'r') as json_file:
                prep_params = json.load(json_file)
                self.max_train_lag = int(prep_params["lags"])
            self.labels = ["Open", "High", "Low", "Close"]  # label order
            self.c_e = prep_params["cyclic_encoding"]
            # # load model
            if self.models_type[n] == "BiL":
                self.model = BiL(**params).to(device)
            elif self.models_type[n] == "EBiL":
                self.model = EBiL(**params).to(device)
            elif self.models_type[n] == "ToM":
                self.model = ToM(**params).to(device)
            elif self.models_type[n] == "BiLPET":
                self.model = BiLPET(**params).to(device)
            else:
                print(f"No model types with name: {self.models_type[n]}")
                exit()
            self.model.load_state_dict(torch.load(os.path.join(self.model_path, self.models_type[n])))
            self.model.eval()
            # start autoregression for each model
            self.start_autoregression(data, company, autoregression_days=autoregression_days)
        # use appended value of self.bagging_pred, self.bagging_metrics to calculate final metrics and predictions
        # this kind of prediction is like the MIXTURE for output probability model
        print(" --------------- BAGGING PREDICTIONS ------------------ ")
        # Calculate the element-wise mean for predictions
        ew_pred_mean = np.mean(self.bagging_pred, axis=0)
        print(ew_pred_mean)
        self.plot_autoregression(values=self.values, predictions=ew_pred_mean)
        print(" ---------------- BAGGING METRICS ------------------- ")
        self.__metrics_mean__()


    def __metrics_mean__(self):
        res = {}
        for el in self.bagging_metrics:
            for k, v in el.items():
                if k not in res.keys():
                    res[k] = float(v)
                else:
                    res[k] += float(v)
        for k, v in res.items():
            res[k] = float(res[k]) / int(len(self.bagging_metrics))
        print(res)




