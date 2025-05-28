# Steffen Mauceri
# 02/2022
#
# make various plots to visualize bias correction vs nearest cloud distance
# Note, this a collection of scripts and code snippets.

# make sure you have all of those packages installed in your conda envirnoment
import pandas as pd
import numpy as np
import os

import paths

from util import load_data_by_year, scatter_hist, calc_SA_bias_clean, bias_correct, plot_map


# make changes #############################
path_lnd_1 = paths.TC_LND_CORR_MODEL 
path_sea_1 = paths.TC_OCN_CORR_MODEL 
path_lnd_2 = paths.SA_LND_CORR_MODEL 
path_sea_2 = paths.SA_OCN_CORR_MODEL 


save_fig = True    # save figures to hard drive
verbose_IO = False
var_tp = 'xco2raw_SA_bias'  # 'xco2_SA_bias', 'xco2_TCCON_bias' or 'xco2raw_SA_bias'  # uncorrected bias based on SA # O'dell bias correction applied -> remaining biases based on SA
max_samples = 10**7
name = '_'
#stop make changes ###########################

# Output path for figures and results
path = paths.MODEL_SAVE_DIR / 'B11' / 'V15_3_2.6_combined_RF0' / 'all' 
if not os.path.exists(path):
    os.makedirs(path)

name = path.split('/')[-2] + name

mode = 'all'
qf = None

print(name)

# load data
data = load_data_by_year(2016, 2019, mode, qf=qf)

# remove data with no cld_dist
data = data[data['cld_dist'].notna()]

if var_tp == 'xco2_TCCON_bias':
    data = data[data['xco2tccon'] > 0]
    data.loc[:, var_tp] = data['xco2_raw'] - data['xco2tccon']

if len(data) > max_samples:
    data = data.sample(max_samples, replace=False)

print(str(len(data)/1000) + 'k samples loaded')

# make a copy of the orignial raw xco2 for later plots and analysis
data.loc[:,'xco2_raw_orig'] = data.loc[:,'xco2_raw'].copy()
# perfrom bias correction
# Always use two-step, land/sea split bias correction

data.loc[:,'xco2MLcorr1'] = data.loc[:,'xco2_raw'].copy()

data_lnd = data[data['land_water_indicator'] == 0]
data_sea = data[data['land_water_indicator'] == 1]

data_lnd = bias_correct(path_lnd_1, data_lnd, ['xco2_raw'])
data_lnd = bias_correct(path_lnd_2, data_lnd, ['xco2_raw'])
data_sea = bias_correct(path_sea_1, data_sea, ['xco2_raw'])
data_sea = bias_correct(path_sea_2, data_sea, ['xco2_raw'])

data = pd.concat([data_lnd, data_sea])

data.loc[:,'xco2MLcorr'] = data.loc[:,'xco2_raw']
data.loc[:,'xco2_raw'] = data.loc[:,'xco2_raw_orig']
data.loc[:,'xco2MLbias'] = data.loc[:,'xco2_raw_orig'] - data.loc[:,'xco2MLcorr']

# recalculate small area bias with cloud free soundings as truth proxy
data.sort_values('SA', inplace=True) # sort data by SA
XCO2 = data['xco2MLcorr'].to_numpy()
SA = data['SA'].to_numpy()
qf = data['cld_dist'].to_numpy() < 5
data.loc[:, 'xco2_SA_bias-ML'] = calc_SA_bias_clean(XCO2, SA, qf) # will only use soundings more than 5 km away from clouds as truth proxy
XCO2 = data['xco2'].to_numpy()
data.loc[:, 'xco2_SA_bias'] = calc_SA_bias_clean(XCO2, SA, qf)
# remove values where don't have enough samples to calc SA bias
data = data.loc[~pd.isna(data['xco2_SA_bias-ML']),:]

#****         show biases vs nearest cloud distance          ************************************************************
# performance with respect to clouds
data = data[data['cld_dist'] < 10]
# get soundings where we have TCCON data
data_t = data[data['xco2tccon'] > 0]
# compare bias correction to tccon
diff_ML = data_t['xco2MLcorr'] - data_t['xco2tccon']
diff_B11 = data_t['xco2'] - data_t['xco2tccon']

var = 'cld_dist'
scatter_hist(data_t[var].to_numpy(), diff_B11.to_numpy(), var, 'OCO2 - TCCON [ppm]', name + 'B11', path,save_IO=save_fig)
scatter_hist(data_t[var].to_numpy(), diff_ML.to_numpy(),  var, 'OCO2 - TCCON [ppm]', name + 'ML', path,save_IO=save_fig)
scatter_hist(data[var].to_numpy(), data['xco2_SA_bias-ML'].to_numpy(), var, 'XCO2 Bias [ppm]', name+ 'ML', path,save_IO=save_fig)
scatter_hist(data[var].to_numpy(), data['xco2_SA_bias'].to_numpy(), var, 'XCO2 Bias [ppm]', name+ 'B11', path,save_IO=save_fig)


#plot cloud dist
plot_map(data, ['cld_dist'], save_fig=False, path=path, name=name + '_cld', pos_neg_IO=False, min = 2, max=4)

print('Done >>>')