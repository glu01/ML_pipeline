import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, train_test_split
from utils.file_utils import save_model, load_model, save_preprocessor, load_preprocessor
import joblib
import optuna
import numpy as np
import os

class LightGBM_ModelingClass:
    def __init__(self, simulated_data, target_variable_name, variables_info):
        self.simulated_data = simulated_data
        self.target_variable_name = target_variable_name
        self.variables_info = variables_info
        self.seed = 42

        self.X = self.simulated_data.drop(columns=[self.target_variable_name])
        self.y = self.simulated_data[self.target_variable_name]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        
        self.lgb_train = lgb.Dataset(self.X_train, self.y_train)
        self.lgb_eval = lgb.Dataset(self.X_test, self.y_test, reference=self.lgb_train)
        self.model = None
        
    def create_params(self, custom_params=None):
        """
        Create a LightGBM model with predefined parameters.
        """
        default_params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'num_leaves': 40,
            'learning_rate': 0.09,
            'feature_fraction': 0.9,
        }
        if custom_params:
            default_params.update(custom_params)
        return default_params

    def train_model(self, custom_params=None):
        """
        Train the lgb model.
        """
        params = self.create_params(custom_params)
        self.model = lgb.train(
            params,
            self.lgb_train,
            num_boost_round=200,
            valid_sets=[self.lgb_train, self.lgb_eval],
            valid_names=['train', 'valid'],
            callbacks=[lgb.early_stopping(10)]
        )
        
        y_pred = self.model.predict(self.X_test)
        rmse = np.sqrt(mean_squared_error(y_pred, self.y_test))
        save_model(self.model, rmse, 'lightgbm')
        return self.model

    def inference(self, new_data=None):
        """
        Make predictions on the test set.
        """
        # Use the best iteration to make predictions
        self.model = load_model('lightgbm')
        if self.model is None:
            raise ValueError("Call train_model() to train a lgb model first")
        if new_data is not None:
            y_pred = self.model.predict(new_data, num_iteration=self.model.best_iteration)
            print(f'Predictions:\n {y_pred}')
        else:
            y_pred = self.model.predict(self.X_test, num_iteration=self.model.best_iteration)
            rmse = sqrt(mean_squared_error(self.y_test, y_pred))
            print(f'prediction:\n {y_pred}')
            print(f"RMSE: {rmse}")
            return y_pred, rmse

    def cross_validation_fcn(self, custom_params=None):
        """
        Perform cross-validation using lgb.cv.
        """
        ### lgb.cv() does not work here, so we manully do the cross validation.
        params = self.create_params(custom_params)
        kf = KFold(n_splits=5, shuffle=True, random_state=self.seed)
        rmse_scores = []
        for train_index, val_index in kf.split(self.X_train):
            X_train_fold, y_train_fold = self.X_train.iloc[train_index], self.y_train.iloc[train_index]
            X_val_fold, y_val_fold = self.X_train.iloc[val_index], self.y_train.iloc[val_index]

            train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
            val_data = lgb.Dataset(X_val_fold, label=y_val_fold)

            model = lgb.train(
                        params,
                        train_data,
                        num_boost_round=200,
                        valid_sets=[train_data, val_data],
                        callbacks=[lgb.early_stopping(10)]
            )
            y_pred = model.predict(X_val_fold, num_iteration=model.best_iteration)
            rmse = np.sqrt(mean_squared_error(y_pred, y_val_fold))
            rmse_scores.append(rmse)

        rmse = np.mean(rmse_scores)
        print(f'Mean RMSE for cross validation: {rmse}')
        return rmse

    def tune_parameters(self, n_trials=10):
        """
        Perform hyperparameter tuning using Optuna for LightGBM.
        Returns the best hyperparameters after tuning.
        """
        def objective(trial):
            """
            Objective function for Optuna.
            """
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True), 
                'num_leaves': trial.suggest_int('num_leaves', 40, 100),  
                'max_depth': trial.suggest_int('max_depth', 4, 8), 
                'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),  
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),  
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),  
                'subsample': trial.suggest_float('subsample', 0.5, 1.0), 
                'cat_smooth': trial.suggest_int('cat_smooth', 1, 100), 
                'seed': self.seed
            }

            kf = KFold(n_splits=5, shuffle=True, random_state=self.seed)
            rmse_scores = []
            for train_index, val_index in kf.split(self.X_train):
                X_train_fold, y_train_fold = self.X_train.iloc[train_index], self.y_train.iloc[train_index]
                X_val_fold, y_val_fold = self.X_train.iloc[val_index], self.y_train.iloc[val_index]

                train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
                val_data = lgb.Dataset(X_val_fold, label=y_val_fold)

                model = lgb.train(
                            params,
                            train_data,
                            num_boost_round=200,
                            valid_sets=[train_data, val_data],
                            callbacks=[lgb.early_stopping(10)]
                )
                y_pred = model.predict(X_val_fold, num_iteration=model.best_iteration)
                rmse = np.sqrt(mean_squared_error(y_pred, y_val_fold))
                rmse_scores.append(rmse)

            rmse = np.mean(rmse_scores)
            return rmse

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        best_params = study.best_params
        print(f"Best parameters found: {best_params}")

        return best_params
