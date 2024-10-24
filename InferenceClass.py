from modeling_lib import *
from DatasetSimulator import DatasetSimulator
import numpy as np
import pandas as pd
from scipy.stats import norm, expon, multinomial
from scipy.special import expit, softmax
from sklearn.model_selection import train_test_split
from utils.file_utils import save_model, load_model, save_preprocessor, load_preprocessor
from configs.paths import variables_info_file, correlations_file
import xgboost as xgb
import lightgbm as lgb
import joblib
import tensorflow as tf
import json
import os

class InferenceClass:
    def __init__(self):
        """
        Initialize the inference class by loading the saved model_paths and preprocessor_paths
        """
        self.model = None
        self.preprocessor = None

    def reconstruct_dataset(self):
        with open(variables_info_file, 'r') as f:
            variables_info = json.load(f)

        for key, value in variables_info.items():
            if value['distribution'] == 'norm':
                value['distribution'] = norm
            elif value['distribution'] == 'expon':
                value['distribution'] = expon

        with open(correlations_file, 'r') as f:
            correlations = np.asarray(json.load(f))
        
        self.variables_info = variables_info
        self.correlations = correlations
        self.target_variable_name = 'all_cause_mortality'
        n_samples = 1000        
        var_name_2_col = {var_name: i for i, var_name in enumerate(self.variables_info.keys())}
        simulator = DatasetSimulator(self.target_variable_name, self.variables_info, self.correlations, n_samples, var_name_2_col)
        simulated_data = simulator.create_simulated_dataset()
        lognormal_data = simulated_data[1]
        self.simulated_data = simulated_data[0]
        X = self.simulated_data.drop(columns=['all_cause_mortality'])
        y = self.simulated_data['all_cause_mortality']
        return self.simulated_data, self.target_variable_name, self.variables_info, X  #return X for test Prediction method.
    
    def train_model(self, model_name):
        """
        Get a trained model if you don't want to load model from model_paths.
        """
        # Get simulated_date from reconstruct_dataset() method.
        simulated_data, target_variable_name, variables_info, _ = self.reconstruct_dataset()
        if model_name == 'svm':
            svm = SVM_ModelingClass(simulated_data, target_variable_name, variables_info)
            self.model = svm.train_model()
        if model_name == 'lightgbm':
            lgb = LightGBM_ModelingClass(simulated_data, target_variable_name, variables_info)
            self.model = lgb.train_model()
        if model_name == 'xgboost':
            xgb = XGBoost_ModelingClass(simulated_data, target_variable_name, variables_info)
            self.model = xgb.train_model()
        if model_name == 'rf':
            rf = RF_ModelingClass(simulated_data, target_variable_name, variables_info)
            self.model = rf.train_model()
        if model_name == 'nn':
            nn = NeuralNetwork_ModelingClass_TF(simulated_data, target_variable_name, variables_info)
            self.preprocessor = nn.preprocess_data()
            self.model = nn.train_model()
        print(f"\n{'='*60}\n{model_name} model has been trained and is ready for predictions\n{'='*60}")
        return self.model, self.preprocessor

    def load_model(self, model_name):
        """
        Load the trained model from file.
        """
        if model_name == 'svm':
            model = load_model(model_name)
            # For pipelines like SVM, the preprocessor is integrated
            self.preprocessor = None 

        elif model_name == 'lightgbm':
            model = load_model(model_name)

        elif model_name == 'xgboost':
            model = load_model(model_name)

        elif model_name == 'rf':
            model = load_model(model_name)

        elif model_name == 'nn':
            model = load_model(model_name)
            preprocessor = load_preprocessor(model_name)
            self.preprocessor = preprocessor

        else:
            raise ValueError(f"Unknown model type: {model_name}")

        self.model = model
        print(f"\n{'='*60}\n{model_name} model is loaded and ready for predictions\n{'='*60}")
        
        return self.model, self.preprocessor

    def predict(self, new_data, model_name):
        """
        Make predictions on new data using the loaded model.
        """        
        self.simulated_data, self.target_variable_name, self.variables_info, _ = self.reconstruct_dataset()
        if self.model is None:
            raise ValueError(f"Model for {model_name} is not loaded. Call load_model(model_name) or train_model(model_name) first.")
        if model_name == 'svm':
            predictions = self.model.predict(new_data)
        elif model_name == 'lightgbm':
            predictions = self.model.predict(new_data)
        elif model_name == 'xgboost':
            dmatrix_data = xgb.DMatrix(new_data)
            predictions = self.model.predict(dmatrix_data)
        elif model_name == 'rf':
            predictions = self.model.predict(new_data)
        elif model_name == 'nn':
            if self.preprocessor is None:
                raise ValueError(f"Preprocessor for {model_name} is not loaded. Call load_model(model_name) or train_model(model_name) first.")
            new_data_preprocessed = self.preprocessor.transform(new_data)
            predictions = self.model.predict(new_data_preprocessed).squeeze()

        else:
            raise ValueError(f"Unknown model type: {model_name}")

        if self.variables_info[self.target_variable_name]['type'] == 'lognormal':
            predictions = np.exp(predictions)

        print(f'Predictions:\n {predictions}')
        return predictions


if __name__ == '__main__':
    """
    model_name are: nn, svm, xgboost, lightgbm, rf
    """

    inference = InferenceClass()
    _, _, _, X = inference.reconstruct_dataset() # return X for tesing prediction.
    model_name = 'nn' 
    # inference.train_model(model_name) # Train a model for predictions if you do not have a trained model to load.
    inference.load_model(model_name)   # If you have trained model, then you can just load it.
    inference.predict(X[:30], model_name)  # HYere use X[:30] as "new_data" to test