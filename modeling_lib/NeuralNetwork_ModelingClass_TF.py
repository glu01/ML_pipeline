import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from utils.file_utils import save_model, load_model, save_preprocessor, load_preprocessor
import numpy as np
import pandas as pd
import joblib
import os

class NeuralNetwork_ModelingClass_TF:
    def __init__(self, simulated_data, target_variable_name, variables_info):
        self.simulated_data = simulated_data
        self.target_variable_name = target_variable_name
        self.variables_info = variables_info
        self.seed = 42

        # Splitting the data
        self.X = self.simulated_data.drop(columns=[self.target_variable_name])
        self.y = self.simulated_data[self.target_variable_name]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        
        if self.variables_info[target_variable_name]['type'] == 'lognormal':
            self.y_train = np.log(self.y_train)
            self.y_test = np.log(self.y_test)

    def create_preprocessor(self):
        continuous_vars = [var for var, info in self.variables_info.items() if info['type'] == 'continuous']
        ordinal_vars = [var for var, info in self.variables_info.items() if info['type'] == 'ordinal']

        continuous_transformer = StandardScaler()
        ordinal_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

        preprocessor = ColumnTransformer(
            transformers=[
                ('cont', continuous_transformer, continuous_vars),
                ('ord', ordinal_transformer, ordinal_vars)
            ]
        )
        return preprocessor

    def preprocess_data(self):
        preprocessor = self.create_preprocessor()

        if isinstance(self.X_train, np.ndarray):
            self.X_train = pd.DataFrame(self.X_train, columns=self.X.columns)
            self.X_test = pd.DataFrame(self.X_test, columns=self.X.columns)

        self.X_train = preprocessor.fit_transform(self.X_train)
        self.X_test = preprocessor.transform(self.X_test)

        return preprocessor

    def create_model(self):
        input_size = self.X_train.shape[1]

        model = models.Sequential()
        model.add(layers.Dense(64, input_shape=(input_size,), activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(1))  

        model.compile(optimizer='adam', loss='mean_squared_error')

        return model

    def train_model(self, epochs=50, batch_size=32):
        """
        train nn model and saved it
        """
        preprocessor = self.preprocess_data()
        self.model = self.create_model()

        self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, validation_data=(self.X_test, self.y_test))
        y_pred = self.model.predict(self.X_test)
        rmse = np.sqrt(mean_squared_error(y_pred, self.y_test))
        save_model(self.model, rmse, 'nn')
        save_preprocessor(preprocessor, 'nn')
        return self.model

    def inference(self):
        self.model = load_model('nn')
        preprocessor = load_preprocessor('nn')
        if self.model is None:
            raise ValueError("Call train_model() to train a neural network model first")

        y_pred = self.model.predict(self.X_test).squeeze()
        if self.variables_info[self.target_variable_name]['type'] == 'lognormal':
            y_pred = np.exp(y_pred)

        print(f'Predictions:\n{y_pred}')
        return y_pred
