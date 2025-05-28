# Steffen Mauceri
# 05/2024
#
# visualize feature importance for bias correction for publication

import json
import joblib
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
import numpy as np
import shap
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

import paths
from util import load_data,load_data_by_year, bias_correct



# make changes #############################
# the model we evaluate
path = paths.MODELS_DIR / 'V15_3_2.6_prec_xco2raw_SA_biasLndNDGL_lnd_RF0'
# precorrect data with this model. This model is not used for evaluation
path_prec = paths.MODELS_DIR / 'V15_3_2.6_xco2_TCCON_biasLndNDGL_lnd_RF0'
output_dir = paths.FIGURE_DIR / 'feature_importance'
precorrect_IO = True    # Correct data with model in path_prec
var_tp = 'xco2raw_SA_bias'  # 'xco2_TCCON_bias' or 'xco2raw_SA_bias'  #TODO set to SA bias for SA model
save_fig = True    # save figures to hard drive
TCCON = False # only TCCON data
max_samples = 10**6
name = ''
data_type = 'all' #'train', 'test', 'all'
#stop make changes ###########################

name = path.split('/')[-2] + name
print(name)

# load model parameters
with open(path / 'normalization_params.json', 'r') as f:
    norm_params = json.load(f)
features = norm_params['features']
model = norm_params['model_type']
X_mean = norm_params['X_mean']
X_std = norm_params['X_std']
y_mean = norm_params['y_mean']
y_std = norm_params['y_std']
features = norm_params['features']
qf = norm_params['qf']
mode = norm_params['mode']

# load trained model
M = joblib.load(path / 'trained_model.joblib')


# load data
if data_type == 'train':
    # Train set
    data = load_data_by_year(2015, 2021, mode, qf=qf)
    
elif data_type == 'all':
    # Train+Val+Test set
    data = load_data_by_year(2015, 2022, mode, qf=qf)
elif data_type == 'test':
    # Test set
    data = load_data(2022, mode, qf=qf,  TCCON=TCCON)
else:
    print('wrong data type')


if var_tp == 'xco2_TCCON_bias':
    data = data[data['xco2tccon'] > 0]
    data.loc[:, var_tp] = data['xco2_raw'] - data['xco2tccon']

if len(data) > max_samples:
    data = data.sample(max_samples, replace=False)
print(str(len(data)/1000) + 'k samples loaded')

if precorrect_IO:
    data = bias_correct(path_prec, data, ['xco2_raw'])


# make input output pair
X = data.loc[:, features]
y = data.loc[:, var_tp]

M.max_samples = 100000

# get permutation importance of input features
result = permutation_importance(M, X, y, n_repeats=10, n_jobs=10)
sorted_idx = result.importances_mean.argsort()

# plot importances
if mode == 'SeaGL':
    color = 'skyblue'
elif mode == 'LndNDGL':
    color = 'limegreen'

# sea TCCON 3.2,3
# land TCCON 2.8,3
# sea SA 3.7,2
# land SA 2.6,3
fig, ax = plt.subplots(figsize=(2.8, 3))
ax.barh(X.columns[sorted_idx], result.importances_mean[sorted_idx], color=color)
ax.set_xlabel('Feature Importance')
ax.set_xlim(0,0.6)
fig.tight_layout()
if save_fig:
    plt.savefig(output_dir / 'Imp_' + name + '.png', dpi=300)
plt.show()
plt.close()


print('Done >>>')