import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from utils.file_utils import save_model, load_model, save_preprocessor, load_preprocessor
import optuna
import joblib
import numpy as np
import os

class XGBoost_ModelingClass:
    def __init__(self, simulated_data, target_variable_name, variables_info):
        self.simulated_data = simulated_data
        self.target_variable_name = target_variable_name
        self.variables_info = variables_info
        self.seed = 42

        self.X = self.simulated_data.drop(columns=[self.target_variable_name])
        self.y = self.simulated_data[self.target_variable_name]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        
        self.dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        self.dtest = xgb.DMatrix(self.X_test, label=self.y_test)
        self.model = None

    def create_params(self, custom_params=None):
        """
        Create default or custom parameters for xgb.
        """
        default_params = {
            'booster': 'gbtree',
            'objective': 'reg:squarederror', 
            'eta': 0.1, 
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': self.seed
        }
        if custom_params:
            default_params.update(custom_params)
        return default_params
    
    def train_model(self, custom_params=None):
        """
        Train the xgb model, optionally with custom parameters.
        """
        
        params = self.create_params(custom_params)

        self.model = xgb.train(
            params,
            self.dtrain,
            num_boost_round=500,
            evals=[(self.dtrain, 'train'), (self.dtest, 'eval')],
            early_stopping_rounds=30,
            verbose_eval=100
        )
        y_pred = self.model.predict(self.dtest)
        rmse = np.sqrt(mean_squared_error(y_pred, self.y_test))
        save_model(self.model, rmse, 'xgboost')
        return self.model
    
    def inference(self, new_data=None):
        """
        Make predictions on the test set using the trained model.
        Optionally, pass a custom DMatrix for inference.
        """
        self.model = load_model('xgboost')
        if self.model is None:
            raise ValueError("Call train_model() to train a xgb model first")
        if new_data is not None:
            dtest = xgb.DMatrix(new_data)
            y_pred = self.model.predict(dtest)
            print(f'prediction:\n {y_pred}')
        else:
            y_pred = self.model.predict(self.dtest)
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            print(f"RMSE: {rmse}")
            return y_pred, rmse

    def cross_validation_fcn(self, custom_params=None):
        """
        Perform cross-validation with optional custom parameters.
        """
        params = self.create_params(custom_params)
        
        cv_results = xgb.cv(
            params,
            self.dtrain,
            num_boost_round=500,
            nfold=5,
            metrics="rmse",
            early_stopping_rounds=30,
            seed=self.seed,
            verbose_eval=100
        )
        mean_rmse = cv_results['test-rmse-mean'].min()
        print(f"Cross-Validation RMSE: {mean_rmse}")
        return mean_rmse
    
    def tune_parameters(self, n_trials=10):
        """
        Perform hyperparameter tuning using Optuna.
        """
        def objective(trial):
            param = {
                'booster': 'gbtree',
                'objective': 'reg:squarederror',  
                'eta': trial.suggest_float('eta', 0.01, 0.2),  
                'max_depth': trial.suggest_int('max_depth', 3, 10),  
                'subsample': trial.suggest_float('subsample', 0.5, 1.0), 
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),  
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 1.0), 
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 1.0),  
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),  
                'seed': self.seed
            }
            
            cv_results = xgb.cv(
                param,
                self.dtrain,
                num_boost_round=500,
                nfold=5,
                metrics="rmse",
                early_stopping_rounds=30,
                seed=self.seed,
                verbose_eval=False
            )
  
            return cv_results['test-rmse-mean'].min()

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        best_params = study.best_params
        print(f"Best parameters found: {best_params}")


        # self.train_model(custom_params=best_params)
        
        return best_params
