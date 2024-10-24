from modeling_lib import *
from DatasetSimulator import DatasetSimulator
from InferenceClass import InferenceClass
import numpy as np
import pandas as pd
from scipy.stats import norm, expon, multinomial
from scipy.special import expit, softmax
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
import joblib
import tensorflow as tf
import os
import json
from configs.paths import variables_info_file, correlations_file

with open(variables_info_file, 'r') as f:
    variables_info = json.load(f)

for key, value in variables_info.items():
    if value['distribution'] == "norm":
        value['distribution'] = norm
    elif value['distribution'] == "expon":
        value['distribution'] = expon

with open(correlations_file, 'r') as f:
    correlations = np.array(json.load(f))
target_variable_name = 'all_cause_mortality'

var_name_2_col = {var_name: i for i, var_name in enumerate(variables_info.keys())}
simulator = DatasetSimulator(target_variable_name, variables_info, correlations, n_samples=5000, var_name_2_col=var_name_2_col)
simulated_data = simulator.create_simulated_dataset()

lognormal_data = simulated_data[1]
simulated_data = simulated_data[0]

X = simulated_data.drop(columns=['all_cause_mortality'])
y = simulated_data['all_cause_mortality']


# X = simulated_data.drop(columns=['all_cause_mortality'])
# y = simulated_data['all_cause_mortality']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# xgb = XGBoost_ModelingClass(simulated_data, target_variable_name, variables_info)
# xgb.train_model()
# xgb.inference(X[:4])
# xgb.cross_validation_fcn()
# xgb.tune_parameters(3)

# lgb = LightGBM_ModelingClass(simulated_data, target_variable_name, variables_info)
# lgb.train_model()
# lgb.inference(new_data=X[:30])
# lgb.cross_validation_fcn()
# lgb.tune_parameters(3)

# svm = SVM_ModelingClass(simulated_data, target_variable_name, variables_info)
# svm.train_model()
# svm.cross_validation_fcn()
# svm.inference()
# svm.tune_parameters(3)

# rf = RF_ModelingClass(simulated_data, target_variable_name, variables_info)
# rf.train_model()
# rf.inference()
# rf.cross_validation_fcn()
# rf.tune_parameters(3)

# nn = NeuralNetwork_ModelingClass_TF(simulated_data, target_variable_name, variables_info)
# nn.train_model()
# nn.inference()


# inference = InferenceClass()
# new_data = X.sample(30)
# model_name = 'rf'
# inference.load_model(model_name)
# predictions = inference.predict(new_data, model_name)