from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from utils.file_utils import save_model, load_model, save_preprocessor, load_preprocessor
import optuna
import joblib
import numpy as np
import os

class RF_ModelingClass:
    def __init__(self, simulated_data, target_variable_name, variables_info):
        self.simulated_data = simulated_data
        self.target_variable_name = target_variable_name
        self.variables_info = variables_info
        self.seed = 42
        self.X = self.simulated_data.drop(columns=[self.target_variable_name])
        self.y = self.simulated_data[self.target_variable_name]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.model = None 

    def create_params(self, custom_params=None):
        """
        Create the default parameters for the RandomForestRegressor.
        """
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': self.seed
        }
        if custom_params:
            default_params.update(custom_params)
        return default_params
    
    def create_model(self, custom_params=None):
        params = self.create_params(custom_params)
        return RandomForestRegressor(**params)
    
    def train_model(self, custom_params=None):
        """
        Train the RF model.
        """
        if self.model is None:
            self.model = self.create_model(custom_params)
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)
        rmse = np.sqrt(mean_squared_error(y_pred, self.y_test))
        save_model(self.model, rmse, 'rf')

        return self.model
    
    def inference(self, new_data=None):
        """
        Make predictions on the test set using the trained model.
        """
        self.model = load_model('rf')
        if self.model is None:
            raise ValueError("Call train_model() to train a rf model first")
        if new_data is not None:
            y_pred = self.model.predict(new_data)
            print(f'prediction:\n {y_pred}')
        else:   
            y_pred = self.model.predict(self.X_test)
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            print(f'prediction:\n {y_pred}')
            print(f"RMSE: {rmse}")
            return y_pred, rmse
    
    def cross_validation_fcn(self, custom_params=None):
        """
        Perform cross-validation and return the mean RMSE.
        """
        rf_model = self.create_model(custom_params)  
        cv_scores = cross_val_score(rf_model, self.X_train, self.y_train, cv=5, scoring='neg_mean_squared_error')
        mean_rmse = np.sqrt(-cv_scores.mean()) 
        print(f"Cross-Validation RMSE: {mean_rmse}")
        return mean_rmse
    
    def tune_parameters(self, n_trials=100):
        """
        hyperparameter tuning using Optuna.
        """
        def objective(trial):
            param = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 5, 50),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'random_state': self.seed
            }

            rf_model = RandomForestRegressor(**param)
            cv_scores = cross_val_score(rf_model, self.X_train, self.y_train, cv=5, scoring='neg_mean_squared_error')
            mean_rmse = np.sqrt(-cv_scores.mean()) 
            return mean_rmse
        
        study = optuna.create_study(direction='minimize')  
        study.optimize(objective, n_trials=n_trials)
        best_params = study.best_params
        print(f"Best parameters found: {best_params}")

        self.model = RandomForestRegressor(**best_params)
        self.model.fit(self.X_train, self.y_train)
        return best_params
