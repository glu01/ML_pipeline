import numpy as np
import pandas as pd
from scipy.stats import norm, expon, multinomial
from scipy.special import expit, softmax
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from configs.paths import variables_info_file, correlations_file
import json

class DatasetSimulator:
    def __init__(self, target_variable_name, variables_info, correlations, n_samples, var_name_2_col):
        self.target_variable_name = target_variable_name
        self.variables_info, self.correlations, self.n_samples = variables_info, correlations, n_samples
        self.var_name_2_col = var_name_2_col
    
    def generate_continuous_variable(self, col_indices = None, indices = False):
        """
        Simulates continuous variables based on the given parameters.
        
        Parameters:
        - variables_info (dict): Dictionary containing variable names as keys and their info as values.
          Example: {'alcohol': {'type': 'continuous', 'distribution': norm, 'mean': 0, 'std_dev': 1}}
        - correlations (np.array): Correlation matrix for all variables.
        - n_samples (int): Number of samples to generate.
        
        Returns:
        pd.DataFrame: DataFrame containing the generated synthetic data for continuous variables.
        """
        variable_names = []
        if indices == False:
            col_indices = []
        
        for var_name, var_info in self.variables_info.items():
            if var_info['type'] in ['continuous', 'ordinal', 'lognormal']:
                variable_names.append(var_name)
                if indices == False:
                    col_indices.append(var_name_2_col[var_name])
        
        n_continuous = len(variable_names)
        if n_continuous == 0:
            return pd.DataFrame([])
        

        
        means = np.zeros(n_continuous)
        std_devs = np.ones(n_continuous)

        # Create covariance matrix based on correlations
        continuous_correlations = self.correlations[col_indices, col_indices]
        cov_continuous = np.outer(std_devs, std_devs) * continuous_correlations
        
        # Generate continuous data
        continuous_data = np.random.multivariate_normal(means, cov_continuous, self.n_samples)
        
        # Transform continuous data based on distributions
        transformed_continuous_data = np.zeros_like(continuous_data)
        for i, var in enumerate(variable_names):
            distribution = self.variables_info[var]['distribution']
            transformed_continuous_data[:, i] = distribution.ppf(norm.cdf(continuous_data[:, i]))

        # Extract mean and std_dev from variables_info
        means = np.array([variables_info[var]['mean'] for var in variable_names])
        std_devs = np.array([variables_info[var]['std_dev'] for var in variable_names])
        
        
        # Apply mean and std_dev
        transformed_continuous_data = transformed_continuous_data * std_devs + means
        
        # Create DataFrame with appropriate column names
        data_created_df = pd.DataFrame(transformed_continuous_data, columns=variable_names)
        
        return data_created_df
    
    def generate_binary_categorical_variable(self):
        """
        Generates binary categorical variables based on the given parameters.
        
        Parameters:
        - variables_info (dict): Dictionary containing variable names as keys and their info as values.
        - correlations (np.array): Correlation matrix for all variables.
        - n_samples (int): Number of samples to generate.
        - p_1 (float): Marginal probability of categorical = 1.
        
        Returns:
        pd.DataFrame: DataFrame containing the generated synthetic data for binary categorical variables.
        """
        
        variable_names = []
        categorical_data = None
        
        for var_name, var_info in self.variables_info.items():
            if var_info['type'] == 'binary_categorical':
                
                coefficients = correlations[var_name_2_col[var_name]]
                coefficients = np.delete(coefficients, np.where(coefficients == 1))

                # Generate binary categorical data
                p_1 = np.asarray(var_info['p_1s'])
                intercept = np.log(p_1 / (1 - p_1))
                log_odds = self.continuous_data @ coefficients + intercept
                prob = expit(log_odds)
                categorical_data = np.random.binomial(1, prob)
        
        if categorical_data != None:
            # Create DataFrame with appropriate column names
            data_created_df = pd.DataFrame(categorical_data, columns=variable_names)

            return data_created_df
        else:
            return pd.DataFrame([])
    
    def generate_multiclass_variable(self):
        """
        Generates multi-class categorical variables based on the given parameters.
        
        Parameters:
        
        Returns:
        pd.DataFrame: DataFrame containing the generated synthetic data for multi-class categorical variables.
        """
        variable_names = []
        
        for var_name, var_info in self.variables_info.items():
            if var_info['type'] == 'multiclass_categorical':
                variable_names.append(var_name)
                
        
        n_variables = len(variable_names)
        
        if n_variables == 0:
            return pd.DataFrame([])  # Return an empty DataFrame if no ordinal variables
    
        for var_name in variable_names:
            var_info = self.variables_info[var_name]
            
    
    def generate_ordinal_variable(self):
        """
        Generates ordinal variables based on the given parameters.
        
        Parameters:
         - data_created_df : list of all continuous simulations. Contains all ordinals as well
        
        Returns:
        pd.DataFrame: DataFrame containing the generated synthetic data for ordinal variables.
        """
        variable_names = []
        
        for var_name, var_info in self.variables_info.items():
            if var_info['type'] == 'ordinal':
                variable_names.append(var_name)
        
        n_variables = len(variable_names)
        
        if n_variables == 0:
            return pd.DataFrame([])  # Return an empty DataFrame if no ordinal variables
    
        for var_name in variable_names:
            var_info = self.variables_info[var_name]
            thresholds = np.asarray(var_info['thresholds'])  # eg. [1, 2, 3, 4, 5]

            values = self.continuous_data[var_name]
            closest = [thresholds[np.argmin(np.abs(thresholds - i))] for i in values]

            self.continuous_data[var_name] = closest
            
        return self.continuous_data
    
    def generate_lognormal_variable(self):
        """
        Generates lognormal variables based on the given parameters.
        
        Parameters:
        - variables_info (dict): Dictionary containing variable names as keys and their info as values.
        - correlations (np.array): Correlation matrix for all variables.
        - n_samples (int): Number of samples to generate.
        
        Returns:
        pd.DataFrame: DataFrame containing the generated synthetic data for lognormal variables.
        """
        variable_names = []
        
        for var_name, var_info in self.variables_info.items():
            if var_info['type'] == 'lognormal':
                variable_names.append(var_name)
        
        n_variables = len(variable_names)
        
        if n_variables == 0:
            return pd.DataFrame([])  # Return an empty DataFrame if no ordinal variables
    
        for var_name in variable_names:
            var_info = self.variables_info[var_name]
            
            normal_data = self.ordinal_data[var_name]

            # Extract mean and std_dev from variables_info
#             means = np.array([variables_info[var]['mean'] for var in variable_names])
#             std_devs = np.array([variables_info[var]['std_dev'] for var in variable_names])
            
            lognormal_data = np.exp(normal_data)
            
#             print(np.mean(normal_data), np.mean(lognormal_data))
            
#             plt.hist(normal_data, bins = 100)
            
        
            # Create DataFrame with appropriate column names
            
            data_created_df = pd.DataFrame(lognormal_data, columns=variable_names)
        
        self.ordinal_data[var_name] = lognormal_data
        return data_created_df
            
    def create_simulated_dataset(self):
        """
        Creates a simulated dataset combining continuous, binary, multi-class, ordinal, and lognormal variables.
        
        Parameters:
        - variables_info (dict): Dictionary containing variable names as keys and their info as values.
        - correlations (np.array): Correlation matrix for all variables.
        - n_samples (int): Number of samples to generate.
        
        Returns:
        pd.DataFrame: DataFrame containing the simulated dataset with appropriate column names.
        """
        # Generate continuous variables
        self.continuous_data = self.generate_continuous_variable()
        
#         print("continuous data:\n", self.continuous_data.shape)
        
        # Generate binary categorical variables
        self.binary_categorical_data = self.generate_binary_categorical_variable()
        
#         print("binary categorical data:\n", self.binary_categorical_data.shape)

        
        # Generate ordinal variables
        self.ordinal_data = self.generate_ordinal_variable()
        
#         print("ordinal data:\n", self.ordinal_data.shape)
        
        
        # Generate multi-class categorical variables
        self.multiclass_data = self.generate_multiclass_variable()
        
#         print("multi class data:\n", self.multiclass_data.shape)
        
        # Generate lognormal variables
        self.lognormal_data = self.generate_lognormal_variable()
        
        l = [self.ordinal_data, self.lognormal_data, self.binary_categorical_data]
        
        return l
    
# # Example usage:
# variables_info = {
#     'alcohol': {'type': 'continuous', 'distribution': norm, 'mean': 5, 'std_dev': 1},
#     'grain_unrefined': {'type': 'continuous', 'distribution': norm, 'mean': 3, 'std_dev': 1},
#     'grain_refined': {'type': 'continuous', 'distribution': norm, 'mean': 4, 'std_dev': 1},
#     'meat_unprocessed': {'type': 'continuous', 'distribution': norm, 'mean': 5, 'std_dev': 1},
#     'meat_processed': {'type': 'continuous', 'distribution': norm, 'mean': 5, 'std_dev': 1},
#     'fruits_and_veggies': {'type': 'continuous', 'distribution': norm, 'mean': 2, 'std_dev': 1},
#     'water': {'type': 'continuous', 'distribution': norm, 'mean': 2000, 'std_dev': 1},
#     'refined_sugar': {'type': 'continuous', 'distribution': norm, 'mean': 50, 'std_dev': 1},
#     'artificial_sweetener': {'type': 'continuous', 'distribution': norm, 'mean': 10, 'std_dev': 1},
#     'cardio': {'type': 'continuous', 'distribution': norm, 'mean': 100, 'std_dev': 1},
#     'strength_training': {'type': 'continuous', 'distribution': norm, 'mean': 100, 'std_dev': 1},
#     'sleep_duration': {'type': 'continuous', 'distribution': norm, 'mean': 8, 'std_dev': 1},
#     'calcium': {'type': 'continuous', 'distribution': norm, 'mean': 500, 'std_dev': 100},
#     'fish_oil_omega_3': {'type': 'continuous', 'distribution': norm, 'mean': 10, 'std_dev': 1},
#     'green_tea': {'type': 'continuous', 'distribution': norm, 'mean': 100, 'std_dev': 1},
#     'legumes': {'type': 'continuous', 'distribution': norm, 'mean': 10, 'std_dev': 1},
#     'fat_trans': {'type': 'continuous', 'distribution': norm, 'mean': 2, 'std_dev': 1},
#     'smoking_frequency': {'type': 'continuous', 'distribution': norm, 'mean': 10, 'std_dev': 1},
#     'sauna_frequency': {'type': 'ordinal', 'distribution': norm, 'mean': 1, 'std_dev': 1, 'thresholds': [1,2,3,4,5]},
#     'stress_level': {'type': 'ordinal', 'distribution': norm, 'mean': 2, 'std_dev': 1, 'thresholds': [1,2,3,4,5]},
#     'sleep_quality': {'type': 'ordinal', 'distribution': norm, 'mean': 3, 'std_dev': 1, 'thresholds': [1,2,3,4,5]},
#     'all_cause_mortality': {'type': 'lognormal', 'distribution': norm, 'mean': 0, 'std_dev': 0.5}
# }

# # Example correlation matrix (dummy values)
# n_variables = len(variables_info)
# correlations = np.asarray(
#       [[1.   , 0.075, 0.139, 0.114, 0.052, 0.101, 0.079, 0.129, 0.194, 0.032, 0.051, 0.104, 0.194, 0.079, 0.18 , 0.065, 0.128, 0.099, 0.173, 0.087, 0.065, 0.03 ],
#        [0.107, 1.   , 0.194, 0.096, 0.103, 0.036, 0.012, 0.058, 0.173, 0.104, 0.082, 0.09 , 0.047, 0.177, 0.123, 0.009, 0.116, 0.174, 0.127, 0.134, 0.157, 0.154],
#        [0.174, 0.064, 1.   , 0.012, 0.094, 0.086, 0.062, 0.06 , 0.025, 0.022, 0.145, 0.163, 0.077, 0.058, 0.184, 0.151, 0.037, 0.02 , 0.084, 0.126, 0.125, 0.077],
#        [0.046, 0.074, 0.118, 1.   , 0.079, 0.095, 0.148, 0.133, 0.172, 0.123, 0.044, 0.065, 0.008, 0.091, 0.14 , 0.124, 0.194, 0.111, 0.049, 0.019, 0.11 , 0.058],
#        [0.127, 0.169, 0.127, 0.082, 1.   , 0.087, 0.164, 0.174, 0.126, 0.089, 0.074, 0.104, 0.104, 0.035, 0.15 , 0.072, 0.026, 0.006, 0.199, 0.117, 0.028, 0.066],
#        [0.19 , 0.048, 0.065, 0.034, 0.146, 1.   , 0.19 , 0.059, 0.148, 0.067, 0.165, 0.145, 0.168, 0.173, 0.082, 0.134, 0.176, 0.081, 0.113, 0.189, 0.108, 0.174],
#        [0.199, 0.068, 0.122, 0.077, 0.159, 0.183, 1.   , 0.193, 0.143, 0.102, 0.057, 0.174, 0.073, 0.169, 0.11 , 0.079, 0.083, 0.15 , 0.129, 0.018, 0.079, 0.095],
#        [0.119, 0.145, 0.195, 0.134, 0.154, 0.099, 0.181, 1.   , 0.143, 0.061, 0.064, 0.028, 0.11 , 0.126, 0.161, 0.01 , 0.094, 0.032, 0.067, 0.093, 0.078, 0.082],
#        [0.066, 0.015, 0.171, 0.16 , 0.139, 0.149, 0.198, 0.111, 1.   , 0.012, 0.173, 0.119, 0.073, 0.006, 0.167, 0.153, 0.079, 0.146, 0.089, 0.083, 0.035, 0.141],
#        [0.021, 0.006, 0.155, 0.198, 0.088, 0.18 , 0.096, 0.126, 0.007, 1.   , 0.099, 0.064, 0.165, 0.134, 0.108, 0.188, 0.153, 0.198, 0.018, 0.034, 0.029, 0.135],
#        [0.016, 0.043, 0.076, 0.014, 0.141, 0.181, 0.027, 0.19 , 0.095, 0.184, 1.   , 0.197, 0.001, 0.115, 0.101, 0.19 , 0.172, 0.064, 0.028, 0.065, 0.045, 0.12 ],
#        [0.049, 0.027, 0.122, 0.115, 0.185, 0.12 , 0.005, 0.015, 0.057, 0.104, 0.104, 1.   , 0.191, 0.08 , 0.112, 0.066, 0.126, 0.036, 0.103, 0.098, 0.134, 0.144],
#        [0.033, 0.137, 0.186, 0.173, 0.186, 0.098, 0.057, 0.028, 0.132, 0.085, 0.159, 0.112, 1.   , 0.064, 0.177, 0.185, 0.007, 0.155, 0.05 , 0.173, 0.137, 0.178],
#        [0.097, 0.16 , 0.053, 0.104, 0.075, 0.179, 0.117, 0.196, 0.031, 0.095, 0.025, 0.167, 0.19 , 1.   , 0.091, 0.1  , 0.122, 0.072, 0.007, 0.178, 0.18 , 0.158],
#        [0.044, 0.185, 0.031, 0.135, 0.162, 0.166, 0.143, 0.072, 0.154, 0.044, 0.15 , 0.178, 0.04 , 0.105, 1.   , 0.055, 0.175, 0.176, 0.188, 0.098, 0.102, 0.079],
#        [0.011, 0.139, 0.045, 0.063, 0.009, 0.189, 0.158, 0.067, 0.137, 0.186, 0.143, 0.119, 0.13 , 0.193, 0.043, 1.   , 0.049, 0.071, 0.024, 0.059, 0.038, 0.057],
#        [0.023, 0.159, 0.116, 0.1  , 0.095, 0.102, 0.015, 0.17 , 0.186, 0.153, 0.07 , 0.06 , 0.019, 0.167, 0.183, 0.054, 1.   , 0.055, 0.092, 0.194, 0.178, 0.04 ],
#        [0.172, 0.122, 0.186, 0.148, 0.111, 0.   , 0.174, 0.13 , 0.186, 0.057, 0.188, 0.149, 0.119, 0.074, 0.182, 0.015, 0.151, 1.   , 0.144, 0.145, 0.18 , 0.15 ],
#        [0.063, 0.124, 0.068, 0.12 , 0.054, 0.021, 0.081, 0.054, 0.121, 0.166, 0.093, 0.125, 0.123, 0.01 , 0.039, 0.043, 0.132, 0.115, 1.   , 0.024, 0.072, 0.115],
#        [0.124, 0.139, 0.087, 0.017, 0.153, 0.175, 0.155, 0.101, 0.195, 0.072, 0.1  , 0.061, 0.197, 0.109, 0.01 , 0.034, 0.184, 0.169, 0.048, 1.   , 0.146, 0.023],
#        [0.044, 0.086, 0.05 , 0.084, 0.194, 0.007, 0.035, 0.055, 0.081, 0.195, 0.12, 0.182, 0.093, 0.161, 0.138, 0.071, 0.076, 0.021, 0.104, 0.15 , 1.   , 0.096],
#        [0.056, 0.156, 0.03 , 0.179, 0.094, 0.126, 0.198, 0.005, 0.09 , 0.199, 0.137, 0.012, 0.105, 0.042, 0.125, 0.049, 0.128, 0.141, 0.117, 0.009, 0.192, 1.   ]])



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
l = list(variables_info.keys())
var_name_2_col = {l[i] : i for i in range(len(l))}

n_samples = 10000

# Create instance of DatasetSimulator
simulator = DatasetSimulator(target_variable_name, variables_info, correlations, n_samples, var_name_2_col)

# Generate simulated dataset
simulated_data = simulator.create_simulated_dataset()
lognormal_data = simulated_data[1]
simulated_data = simulated_data[0]
simulated_data


# Split the data into features and target
X = simulated_data.drop(columns=['all_cause_mortality'])
y = simulated_data['all_cause_mortality']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Show the first few rows of the training data
X_train.head(), y_train.head(), variables_info