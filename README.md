```
Model_Training_and_Inference/
│
├── configs/
│   ├── paths.py          # Configuration for file paths (models, preprocessors, scores, etc.)
│   ├── best_scores.json  # Stores the best model scores for each model
│
├── model/                # Contains the trained model files (latest and best models)
│
├── modeling_lib/         # Contains individual model classes (SVM, RF, XGBoost, LightGBM, NeuralNetwork)
│   ├── SVM_ModelingClass.py
│   ├── RF_ModelingClass.py
│   ├── XGBoost_ModelingClass.py
│   ├── LightGBM_ModelingClass.py
│   ├── NeuralNetwork_ModelingClass_TF.py
│
├── preprocessor/         # Preprocessor files for each model (e.g., scaling and transformation pipelines)
│
├── utils/
│   ├── file_utils.py     # Utility functions for saving/loading models and preprocessors
│
├── DatasetSimulator.py   # Script to simulate dataset for testing/training purposes
├── InferenceClass.py     # Main class to handle model training and inference
├── test_inference.py     # Script to test the inference process
├── README.md             # This file
├── requirements.txt    
```
```python
=======================================================================
# How to train a model:
=======================================================================

from InferenceClass import InferenceClass

inference = InferenceClass()

# Train the SVM model
model_name = 'svm'
inference.train_model(model_name)

# After training, the model will be saved


=======================================================================
# How to make an inference:
=======================================================================
from InferenceClass import InferenceClass

inference = InferenceClass()

# Load the pre-trained model
model_name = 'svm'
inference.load_model(model_name)

# Run inference on new data
# Here we use simulated data for inference; you can replace it with your own data
data = inference.reconstruct_dataset()[0]  # Get the X values from the dataset
predictions = inference.predict(data[:30], model_name)  # Predict on the first 30 rows

