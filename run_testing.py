from AutoregressiveTest import AutoTest
from utils import load_csv
import os
"""
# This class can do bagging/ensemble with more models, but:
#   - models must be in the same dir
#   - must use an ordered list of 'model_dir' and 'model_type'
#   - all models must be trained on the company you are trying to predict
#   - must use 'start_autoregressive_bagging' instead of 'start_autoregression'
                                                                                """
"""
# Use 45lag_test_data directory only with "45LAG" models
test_data = load_csv(dirpath=os.path.join(os.getcwd(), "45lag_test_data"))
# The testing will work anyway, but results could be compromised.
                                                                            """
test_data = load_csv(dirpath=os.path.join(os.getcwd(), "45lag_test_data"))
#test_data = load_csv(dirpath=os.path.join(os.getcwd(), "test_data"))

lab = input("Enter company name: ")

#   #   # AUTOREGRESSIVE BAGGING/ENSEMBLE
"""ates = AutoTest(model_dir=["LARGE_ToM", "LARGE_BiLPET", "SMALL_ToM", "SMALL_EBiL"], model_type=["ToM", "BiLPET", "ToM", "EBiL"])
ates.start_autoregressive_bagging(data=test_data[lab], company=lab, autoregression_days=7, plot_all_graphs=False)"""

#   #   # AUTOREGRESSION BAGGING/ENSEMBLE WITH 45 LAGS MODELS
ates = AutoTest(model_dir=["45LAG_SMALL_ToM", "45LAG_SMALL_BiL", "45LAG_SMALL_ToM"], model_type=["ToM", "BiL", "ToM"])
ates.start_autoregressive_bagging(data=test_data[lab], company=lab, autoregression_days=7, plot_all_graphs=False)

#   #   # SINGLE AUTOREGRESSION
"""ates = AutoTest(model_dir="45LAG_SMALL_BiL", model_type="BiL")
ates.start_autoregression(data=test_data[lab], company=lab, autoregression_days=7)"""

