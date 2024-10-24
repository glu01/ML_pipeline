import os

base_dir = os.path.dirname(os.path.dirname(__file__))

models_dir = os.path.join(base_dir, 'model')
preprocessor_dir = os.path.join(base_dir, 'preprocessor')

configs_dir = os.path.join(base_dir, 'configs')

best_score_file = os.path.join(configs_dir, 'best_scores.json')
variables_info_file = os.path.join(configs_dir, 'variables_info.json')
correlations_file = os.path.join(configs_dir, 'correlations.json')



# logs_dir = os.path.join(base_dir, 'logs')
# data_dir = os.path.join(base_dir, 'data')
