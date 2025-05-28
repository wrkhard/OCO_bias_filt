# Calculate the distance of each sounding to the nearest TCCON sounding in feature space and plot the results on a map

import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
import paths
from util import load_data_and_concat_years, plot_map

# make changes #############################
name = ''
save_fig = True    # save figures to hard drive
path = paths.PLOTS_DIR / 'TCCON'
mode = 'all'
#stop make changes ###########################


data = load_data_and_concat_years(2015, 2022, mode)

# randomly donwn sample to half the data
data = data.sample(2* 10**7, replace=False)


features = ['co2_grad_del','dp', 'sensor_zenith_angle']


X = data[features].copy()
X_mean = X.mean()
X_std = X.std()
X = (X - X_mean) / X_std


# get TCCON stations
TC_data = X.loc[data['xco2tccon'].to_numpy() > 0,:].sample(10**5, replace=False)


# calculate distance of individual Sounding to nearest TCCON soundings in feature space
print('fit tree')
kdt = KDTree(TC_data, leaf_size=50, metric='euclidean')


print('calc distances')
distances, _ = kdt.query(X, k=10, sort_results=False, dualtree=True)
mean_dist = np.mean(distances, axis=1)
data['TCCON_dist'] = mean_dist


plot_map(data, ['TCCON_dist'], save_fig=save_fig, path=path, name=name + 'TCCON State Space Distance_3', pos_neg_IO=False)


print('Done >>>')