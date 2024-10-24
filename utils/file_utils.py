import os
import joblib
import tensorflow as tf
from configs.paths import models_dir, preprocessor_dir, best_score_file
import json


def save_model(model, score, model_name):
    """
    Save the model based on its model name (svm, rf, lightgbm, xgboost, nn).
    Args:
        model: The model to save.
        model_name: The name of the model.
    """
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    latest_model_path = os.path.join(models_dir, f'{model_name}_latest_model')
    
    if model_name == 'nn':  # Neural Network (TensorFlow/Keras model)
        latest_model_path += '.h5'
        model.save(latest_model_path)
        print(f"Latest Neural Network model saved to {latest_model_path}")

    elif model_name == 'svm' or model_name == 'rf':  # SVM or RandomForest (joblib pkl model)
        latest_model_path += '.pkl'
        joblib.dump(model, latest_model_path)
        print(f"Latest {model_name} model saved to {latest_model_path}")

    elif model_name == 'xgboost':  # XGBoost model
        latest_model_path += '.model'
        model.save_model(latest_model_path)
        print(f"Latest XGBoost model saved to {latest_model_path}")

    elif model_name == 'lightgbm':  # LightGBM model
        latest_model_path += '.model'
        model.save_model(latest_model_path)
        print(f"Latest LightGBM model saved to {latest_model_path}")

    else:
        raise ValueError(f"Unrecognized model name: {model_name}")
    

    if os.path.exists(best_score_file):
        with open(best_score_file, 'r') as f:
            best_scores = json.load(f)
    else:
        best_scores = {}

    best_score = best_scores.get(model_name, float('inf'))

    if score < best_score:

        best_model_path = os.path.join(models_dir, f'{model_name}_best_model')
        
        if model_name == 'nn':  
            best_model_path += '.h5'
            model.save(best_model_path)
        elif model_name == 'svm' or model_name == 'rf': 
            best_model_path += '.pkl'
            joblib.dump(model, best_model_path)
        elif model_name == 'xgboost':  
            best_model_path += '.model'
            model.save_model(best_model_path)
        elif model_name == 'lightgbm':  
            best_model_path += '.model'
            model.save_model(best_model_path)

        best_scores[model_name] = score
        print(f"New best {model_name} model saved with score: {score}")

        with open(best_score_file, 'w') as f:
            json.dump(best_scores, f)


def load_model(model_name, best=True):
    """
    Load the model based on its model name (svm, randomforest, lightgbm, xgboost, neuralnetwork).
    Args:
        model_name: The name of the model.
    Returns:
        The loaded model.
    """

    model_version = 'best' if best else 'latest'

    if model_name == 'nn':  # Neural Network (TensorFlow/Keras model)
        model_load_path = os.path.join(models_dir, f'{model_name}_{model_version}_model.h5')
        if not os.path.exists(model_load_path):
            raise FileNotFoundError(f"Neural Network model file {model_load_path} does not exist.")
        print(f"Loading Neural Network model from {model_load_path}")
        return tf.keras.models.load_model(model_load_path)

    elif model_name == 'svm' or model_name == 'rf':  # SVM or RandomForest (joblib pkl model)
        model_load_path = os.path.join(models_dir,  f'{model_name}_{model_version}_model.pkl')
        if not os.path.exists(model_load_path):
            raise FileNotFoundError(f"{model_name} model file {model_load_path} does not exist.")
        print(f"Loading {model_name} model from {model_load_path}")
        return joblib.load(model_load_path)

    elif model_name == 'xgboost':  # XGBoost model
        model_load_path = os.path.join(models_dir,  f'{model_name}_{model_version}_model.model')
        if not os.path.exists(model_load_path):
            raise FileNotFoundError(f"XGBoost model file {model_load_path} does not exist.")
        print(f"Loading XGBoost model from {model_load_path}")
        import xgboost as xgb
        model = xgb.Booster()
        model.load_model(model_load_path)
        return model

    elif model_name == 'lightgbm':  # LightGBM model
        model_load_path = os.path.join(models_dir,  f'{model_name}_{model_version}_model.model')
        if not os.path.exists(model_load_path):
            raise FileNotFoundError(f"LightGBM model file {model_load_path} does not exist.")
        print(f"Loading LightGBM model from {model_load_path}")
        import lightgbm as lgb
        model = lgb.Booster(model_file=model_load_path)
        return model

    else:
        raise ValueError(f"Unrecognized model name: {model_name}")



def save_preprocessor(preprocessor, model_name):
    """
    Save the preprocessor based on its model name (svm, neuralnetwork).
    Args:
        preprocessor: The preprocessor to save.
        model_name: The name of the model.
    """
    if not os.path.exists(preprocessor_dir):
        os.makedirs(preprocessor_dir)
    
    if model_name == 'svm':
        preprocessor_save_path = os.path.join(preprocessor_dir, f'{model_name}_preprocessor.pkl')
        joblib.dump(preprocessor, preprocessor_save_path)
        print(f"Preprocessor of {model_name} saved to {preprocessor_save_path}")
    
    elif model_name == 'nn':
        preprocessor_save_path = os.path.join(preprocessor_dir, f'{model_name}_preprocessor.pkl')
        joblib.dump(preprocessor, preprocessor_save_path)
        print(f"Preprocessor of {model_name} saved to {preprocessor_save_path}")
    
    else:
        raise ValueError(f"Unrecognized model name: {model_name}")


def load_preprocessor(model_name):
    """
    Load the preprocessor based on its model name (svm, neuralnetwork).
    Args:
        model_name: The name of the model.
    Returns:
        Loaded preprocessor object.
    """

    if model_name == 'svm':
        preprocessor_load_path = os.path.join(preprocessor_dir, f"{model_name}_preprocessor.pkl")
        if not os.path.exists(preprocessor_load_path):
            raise FileNotFoundError(f"{model_name} preprocessor file {preprocessor_load_path} does not exist.")
        print(f"Loading {model_name} preprocessor from {preprocessor_load_path}")
        return joblib.load(preprocessor_load_path)
    
    elif model_name == 'nn':
        preprocessor_load_path = os.path.join(preprocessor_dir, f"{model_name}_preprocessor.pkl")
        if not os.path.exists(preprocessor_load_path):
            raise FileNotFoundError(f"{model_name} preprocessor file {preprocessor_load_path} does not exist.")
        print(f"Loading {model_name} preprocessor from {preprocessor_load_path}")
        return joblib.load(preprocessor_load_path)

    else:
        raise ValueError(f"Unrecognized model name: {model_name}")
