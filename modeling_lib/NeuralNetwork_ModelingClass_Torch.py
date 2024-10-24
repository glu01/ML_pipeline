import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import numpy as np
import joblib
import pandas as pd

class CustomNeuralNet(nn.Module):
    def __init__(self, input_size):
        super(CustomNeuralNet, self).__init__()
        
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(32)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = nn.functional.relu(x)

        x = self.fc3(x)
        return x

class NeuralNetwork_ModelingClass_Torch:
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

        self.X_train_tensor = torch.tensor(self.X_train, dtype=torch.float32)
        self.X_test_tensor = torch.tensor(self.X_test, dtype=torch.float32)
        self.y_train_tensor = torch.tensor(self.y_train.values, dtype=torch.float32).unsqueeze(1)
        self.y_test_tensor = torch.tensor(self.y_test.values, dtype=torch.float32).unsqueeze(1)

        return preprocessor

    def create_model(self):
        """
        :param input_size: Size of input features
        """
        input_size = self.X_train.shape[1]
        model = CustomNeuralNet(input_size)

        return model

    def train_model(self, model_save_path='nn_model.pth', preprocessor_save_path='nn_preprocessor.pkl', epochs=10, batch_size=32, lr=0.001):

        preprocessor = self.preprocess_data()
        self.model = self.create_model()

        criterion = nn.MSELoss() 
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        train_dataset = TensorDataset(self.X_train_tensor, self.y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}')

        torch.save(self.model, model_save_path)
        joblib.dump(preprocessor, preprocessor_save_path)
        print(f'Model and preprocessor saved to {model_save_path} and preprocessor.pkl')
        return self.model

    def inference(self):
        if self.model is None:
            raise ValueError("Call train_model() to train a nn model first")
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(self.X_test_tensor).squeeze(1).numpy()
        if self.variables_info[target_variable_name]['type'] == 'lognormal':
            y_pred = np.exp(y_pred)
        print(f'Prediction:\n {y_pred}')
        return y_pred
