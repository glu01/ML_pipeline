import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from utils.file_utils import save_model, load_model, save_preprocessor, load_preprocessor
import optuna
import os
import joblib

class SVM_ModelingClass:
    def __init__(self, simulated_data, target_variable_name, variables_info):
        self.simulated_data = simulated_data
        self.target_variable_name = target_variable_name
        self.variables_info = variables_info
        self.seed = 42

        self.X = self.simulated_data.drop(columns=[self.target_variable_name])
        self.y = self.simulated_data[self.target_variable_name]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        
        if self.variables_info[target_variable_name]['type'] == 'lognormal':
            self.y_train = np.log(self.y_train)
            self.y_test = np.log(self.y_test)

        self.pipeline = None

    def create_params(self, custom_params=None):
        default_params = {
            'C': 1.0,  
            'kernel': 'rbf',  
            'gamma': 'scale', 
            'tol': 1e-3, 
        }
        if custom_params:
            default_params.update(custom_params)
        return default_params

    def create_model(self, custom_params=None):
        """Create the SVM model"""
        params = self.create_params(custom_params)
        return SVR(**params)

    def create_pipeline(self, custom_params=None):
        """
        Create a pipeline that includes preprocessing and the SVM model.
        """
        # Identify continuous and ordinal variables
        continuous_vars = [var for var, info in self.variables_info.items() if info['type'] == 'continuous']
        ordinal_vars = [var for var, info in self.variables_info.items() if info['type'] == 'ordinal']

        continuous_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        ordinal_transformer = Pipeline(steps=[('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))])

        preprocessor = ColumnTransformer(
            transformers=[
                ('cont', continuous_transformer, continuous_vars),
                ('ord', ordinal_transformer, ordinal_vars)
            ]
        )

        self.pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('svr', self.create_model(custom_params))])
        
        return self.pipeline, preprocessor

    def train_model(self, custom_params=None):
        """
        Train the SVM model using the preprocessed data
        """

        self.pipeline, preprocessor = self.create_pipeline(custom_params)
        self.pipeline.fit(self.X_train, self.y_train)    
        y_pred = self.pipeline.predict(self.X_test)
        rmse = np.sqrt(mean_squared_error(y_pred, self.y_test))

        save_model(self.pipeline, rmse, 'svm')
        save_preprocessor(preprocessor, 'svm')
        return self.pipeline

    def inference(self, new_data=None):
        """Make predictions on the test set"""
        self.pipeline = load_model('svm')
        if self.pipeline is None:
            raise ValueError("Call train_model() to train a svm model first")
        if new_data:
            y_pred = self.pipeline.predict(new_data)
            if self.variables_info[self.target_variable_name]['type'] == 'lognormal':
                y_pred = np.exp(y_pred)
            print(f'prediction:\n {y_pred}')
            return y_pred
        else:
            y_pred = self.pipeline.predict(self.X_test)
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            if self.variables_info[self.target_variable_name]['type'] == 'lognormal':
                y_pred = np.exp(y_pred)
            print(f'prediction:\n {y_pred}')
            print(f'rmse: {rmse}')
            return y_pred

    def cross_validation_fcn(self):
        """
        Perform cross-validation.
        """
        if self.pipeline is None:
            self.pipeline, _ = self.create_pipeline()
        mean_score = -cross_val_score(self.pipeline, self.X_train, self.y_train, cv=5, scoring='neg_mean_squared_error').mean()
        rmse = np.sqrt(mean_score)
        print(f'rmse for cross validation: {rmse}')
        return rmse

    def tune_parameters(self, n_trials=50):
        def objective(trial):
            """
            Objective function for Optuna optimization for SVM.
            """
            # Hyperparameters for SVM
            param = {
                'svr__C': trial.suggest_float('svr__C', 0.1, 10.0),
                'svr__epsilon': trial.suggest_float('svr__epsilon', 0.01, 0.2),
                'svr__kernel': trial.suggest_categorical('svr__kernel', ['linear', 'rbf'])
            }
            pipeline, _ = self.create_pipeline()
            pipeline.set_params(**param)
            mean_score = self.cross_validation_fcn()
            RMSE_score = np.sqrt(mean_score)

            return RMSE_score
    
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        best_params = study.best_params
        print(f"Best parameters found: {best_params}")

        return best_params
