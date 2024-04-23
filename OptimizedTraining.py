import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import json
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error


class TrainMethod:
    def __init__(self, model, loss_fn, optimizer, device, model_params, x_scaler, y_scaler, prep_params, companies):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []
        self.device = device
        self.params = model_params
        self.xscaler = x_scaler
        self.yscaler = y_scaler
        self.pp = prep_params
        self.companies = companies

    def train_step(self, x, y):
        # Sets model to train mode
        self.model.train()

        # Makes predictions
        yhat = self.model(x)

        # Computes loss
        loss = self.loss_fn(y, yhat)

        # Computes gradients
        loss.backward()

        # Updates parameters and zeroes gradients
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Returns the loss
        return loss.item()

    def store_model(self, model_name):
        models_dir = os.path.join(os.getcwd(), "models")
        datatime_model = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        if not os.path.exists(models_dir):
            os.mkdir(models_dir)
        model_dir = os.path.join(models_dir, f"{datatime_model}_{model_name}")
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        model_path = os.path.join(model_dir, f'{model_name}')
        return model_path, model_dir

    def train(self, train_loader, val_loader=None, batch_size=64, n_epochs=50, n_features=1, model_name="base_model"):
        self.noval=True if val_loader is None else False
        model_path, self.model_dir = self.store_model(model_name=model_name)
        for epoch in tqdm(range(1, n_epochs + 1)):
            batch_losses = []
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.view([batch_size, -1, n_features]).to(self.device)
                y_batch = y_batch.to(self.device)
                loss = self.train_step(x_batch, y_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)
            if val_loader is not None:
                with torch.no_grad():
                    batch_val_losses = []
                    for x_val, y_val in val_loader:
                        x_val = x_val.view([batch_size, -1, n_features]).to(self.device)
                        y_val = y_val.to(self.device)
                        self.model.eval()
                        yhat = self.model(x_val)
                        val_loss = self.loss_fn(y_val, yhat).item()
                        batch_val_losses.append(val_loss)
                    validation_loss = np.mean(batch_val_losses)
                    self.val_losses.append(validation_loss)

            if (epoch <= 10) or (epoch % 50 == 0) or (epoch % 5 == 0):
                if val_loader is not None:
                    print(
                        f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f}"
                    )
                else:
                    print(
                        f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}"
                    )
        torch.save(self.model.state_dict(), model_path)
        self.save_utils()

    def save_utils(self):
        # Save as JSON
        try:
            json_path = os.path.join(self.model_dir, 'model_params.json')
            with open(json_path, 'w') as json_file:
                json.dump(self.params, json_file)
        except:
            print("An error occurred during params saving")
        # # save architecture
        try:
            model_architecture = repr(self.model)
            arc_path = os.path.join(self.model_dir, 'model_architecture.json')
            with open(arc_path, 'w') as f:
                f.write(model_architecture)
        except:
            print("An error occurred during model architecture saving")
        # # save scaler
        try:
            # Save scalers using joblib
            scaler_path = os.path.join(self.model_dir, 'xscaler.pkl')
            joblib.dump(self.xscaler, scaler_path)
            scaler_path = os.path.join(self.model_dir, 'yscaler.pkl')
            joblib.dump(self.yscaler, scaler_path)
        except:
            print("An error occurred during scaler saving")
        # # save prep. parameters
        try:
            json_path = os.path.join(self.model_dir, 'prep_params.json')
            with open(json_path, 'w') as json_file:
                json.dump(self.pp, json_file)
        except:
            print("An error occurred during prep_params.json saving")
        try:
            # save dict in current working dir
            json_path = os.path.join(self.model_dir, 'companies.json')
            with open(json_path, 'w') as f:
                json.dump(self.companies, f)
        except:
            print("An error occurred during companies.json saving")

    def plot_losses(self):
      plt.plot(self.train_losses, label="Training loss")
      if not self.noval:
        plt.plot(self.val_losses, label="Validation loss")
      plt.legend()
      plt.title("Losses")
      try:
        plt.savefig(os.path.join(self.model_dir, 'losses.png'))
      except:
          print("An error occurred during losses.png saving")
      plt.show()
      plt.close()

    def save_test_metrics(self, metrics):
        try:
            met_path = os.path.join(self.model_dir, 'test_metrics.json')
            with open(met_path, 'w') as f:
                json.dump(metrics, f)
        except Exception as e:
            print("An error occurred during test_metrics.json saving")
            print(e)


    def evaluate(self, test_loader, batch_size=1, n_features=1, metrics=True):
      with torch.no_grad():
        predictions = []
        values = []
        for x_test, y_test in test_loader:
            x_test = x_test.view([batch_size, -1, n_features]).to(self.device)
            y_test = y_test.to(self.device)
            self.model.eval()
            yhat = self.model(x_test)
            predictions.append(yhat.cpu().detach().numpy())
            values.append(y_test.cpu().detach().numpy())
        if metrics:
            # Concatenate the lists into numpy arrays
            true_values = np.concatenate(values, axis=0)
            predicted_values = np.concatenate(predictions, axis=0)

            #  (MAE) for each column
            mae = mean_absolute_error(true_values, predicted_values, multioutput='raw_values')

            # (MSE) for each column
            mse = mean_squared_error(true_values, predicted_values, multioutput='raw_values')
            mae = list(mae); mse = list(mse)
            label = ["Open", "High", "Low", "Close"]
            save_dict = {}
            for i in range(len(mae)):
                save_dict[f"{label[i]}_mae"] = str(mae[i])
                save_dict[f"{label[i]}_mse"] = str(mse[i])
            print(save_dict)
            self.save_test_metrics(metrics=save_dict)
      return predictions, values

