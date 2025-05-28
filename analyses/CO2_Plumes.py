# script to investigate impact of bias correction on real plumes (e.g. from power plants)
# Steffen Mauceri
# 09/22

# load data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import paths
from util import load_data, bias_correct
import joblib
import json
from pathlib import Path



# make changes #############################
# Use paths defined in paths.py
path1 = paths.TC_LND_CORR_MODEL # Corresponds to xco2_TCCON_biasLndNDGL_lnd_RF0
path2 = paths.SA_LND_CORR_MODEL # Corresponds to prec_xco2raw_SA_biasLndNDGL_lnd_RF0

precorrect_IO = True    # Correct data with two models, first the one in path1 then the one in path2
save_fig = True    # save figures to hard drive
preload_IO = False
var_tp = 'xco2raw_SA_bias' # 'xco2_SA_bias', 'xco2_TCCON_bias' or 'xco2raw_SA_bias'  # uncorrected bias based on SA # O'dell bias correction applied -> remaining biases based on SA
name = '_'
#stop make changes ###########################
p_lat = [39.285563,38.747527,38.935591, 23.978,-23.668333, -23.668333, -23.668333, -23.668333, -23.668333, -23.668333,-23.668333] #List from 10.1002/2017GL074702
p_lon = [-96.117011, -85.034330,-82.118461,82.627612, 27.610556,27.610556,27.610556,27.610556,27.610556,27.610556,27.610556]
p_time = [20151204, 20150813,20150730,20141023,20181118,20190504,20200716,20200817,20201022,20180711,20191105]


if precorrect_IO:
    path = path2
else:
    path = path1


name = path.name + name # Get name directly from Path object
print(name)

print('working on:', name)
path = Path(path)
model_dir = path / name

# Load model using joblib
model_path = model_dir / 'trained_model.joblib'
M = joblib.load(model_path)

# Load normalization parameters from JSON
params_path = model_dir / 'normalization_params.json'
with open(params_path, 'r') as f:
    params = json.load(f)

features = params['features']
model_type = params.get('model_type', params.get('model')) # Use .get for flexibility
X_mean = pd.Series(params['X_mean'])
X_std = pd.Series(params['X_std'])
y_mean = params.get('y_mean') # Use .get as y_mean/y_std might not always be present
y_std = params.get('y_std')


# load data
data = load_data(2021, mode, qf=qf, preload_IO=preload_IO, max_n=2*10**7)
data1 = load_data(2019, mode, qf=qf,preload_IO=preload_IO, max_n=2*10**7)
data2 = load_data(2020, mode, qf=qf,preload_IO=preload_IO, max_n=2*10**7)
data = pd.concat([data,  data1, data2])

print(str(len(data)/1000) + 'k samples loaded')

# remove footprints 1-3 and 6-8 to mitigate scatter
data = data[(data['footprint'] == 4) | (data['footprint'] == 5)]

# make a copy of the orignial raw xco2 for later plots and analysis
data.loc[:,'xco2_raw_orig'] = data.loc[:,'xco2_raw'].copy()
# perfrom bias correction
data = bias_correct(path1, data, ['xco2_raw'])
if precorrect_IO:
    data = bias_correct(path2, data, ['xco2_raw'])
data.loc[:,'xco2MLcorr'] = data.loc[:,'xco2_raw']
data.loc[:,'xco2_raw'] = data.loc[:,'xco2_raw_orig']


#****************************************************************
# find soundings that contain plumes
l = 0.6   # range for lat long [deg]
t = 1     # range for time [day]
for i in range(len(p_lon)):
    lat = p_lat[i]
    lon = p_lon[i]
    time = p_time[i]

    data_i = data[(data['latitude'] >= lat-l) & (data['latitude'] <= lat+l) &
                  (data['longitude'] >= lon-l) & (data['longitude'] <= lon+l) &
                  ((data['sounding_id']*10**-8).astype(int) >= time-t) & ((data['sounding_id']*10**-8).astype(int) <= time+t)]

    # plot data that contain plumes with and without bias correction
    print('found ' + str(len(data_i)))
    if len(data_i) > 0:
        print('Plume ' + str(i) + ' at lat: ' + str(lat) + ' lon: ' + str(lon) + ' time: ' + str(time))
        data_i.sort_values('latitude', inplace=True)
        plt.figure(figsize=(5,3))
        plt.plot(data_i['latitude'], data_i['xco2'] - data_i['xco2'].mean(),color='k',marker='+', linestyle='--', label='OCO-2')
        plt.plot(data_i['latitude'], data_i['xco2MLcorr']- data_i['xco2MLcorr'].mean(),color='r',marker='.', linestyle=':', markersize=4, label='OCO-2 corr.')
        plt.title('Lat: ' + str(np.round(lat, 2)) + '° | Lon: ' + str(np.round(lon, 2))+ '°' )
        plt.legend()
        plt.xlabel('Latitude [°]')
        plt.ylabel('XCO2 anomaly [ppm]')
        plt.tight_layout()
        if save_fig:
            # Construct save path using Path object methods
            save_path = path.parent / (name + '_Plume_lat_' + str(i) + '.png') # Save in parent dir of model
            plt.savefig(save_path, dpi=300)
        plt.show()
        plt.close()



print('Done')