# make a few plots of the exported lite files to make sure they are correct
# 10/2022 Steffen Mauceri

import numpy as np
import pandas as pd
import glob
import netCDF4 as nc
from tqdm import tqdm
import os
from util import plot_map, dist

def get_all_headers(f):
    headers = list(f.variables.keys())

    # get groups
    groups = list(f.groups.keys())

    for g in groups:
        vars = list(f[g].variables.keys())
        for v in vars:
            headers.append(g+ '/' + v )

    return headers


year = 2015
frac = 0.5 # fraction of data to load
name_all = 'LiteFileExport_' + str(year) + '_'
save_fig = True
Lite_file_path = '/Volumes/OCO/LiteFiles/B11_gamma/'
save_path = '/Volumes/OCO/LiteFiles/B11_gamma_val/'

# check that we exported each file from B111 to B112
import os

# Paths to the two directories
B111_dir = '/Volumes/OCO/LiteFiles/B111/'
B112_dir = Lite_file_path

# Function to extract file identifiers from file names
def extract_identifier(filename):
    try:
        test = filename.split('.nc4')[1]
        filename = filename.split('_B11')[0]
        filename = filename.split('LtCO2_')[1]
    except:
        pass
    return filename

# List of file identifiers in each directory
files_in_dir1 = {extract_identifier(f) for f in os.listdir(B111_dir) if os.path.isfile(os.path.join(B111_dir, f))}
files_in_dir2 = {extract_identifier(f) for f in os.listdir(B112_dir) if os.path.isfile(os.path.join(B112_dir, f))}

# Identifiers unique to each directory
unique_to_dir1 = files_in_dir1 - files_in_dir2
unique_to_dir2 = files_in_dir2 - files_in_dir1

print("Files unique to B111 directory:", unique_to_dir1)
print("Files unique to B11gamma directory:", unique_to_dir2)







# read in Lite files ************************************************************
# make path to save figures if it does not exist
if save_fig:
    if not os.path.exists(save_path):
        os.makedirs(save_path)

fill_value = 999999
counts=0
print(year)
data_all = []
# get LiteFile data
Lite_files = glob.glob(Lite_file_path + 'oco2_LtCO2_' + str(year-2000) + '*.nc4')

# get Lite vars
l_ds = nc.Dataset(Lite_files[0])
l_vars = get_all_headers(l_ds)
# remove vars we dont need
l_vars = [e for e in l_vars if e not in ('bands', 'footprints','levels', 'vertices','Retrieval/iterations','file_index','vertex_latitude','vertex_longitude',
                                         'date','source_files','pressure_levels', 'co2_profile_apriori', 'xco2_averaging_kernel','Preprocessors/co2_ratio_offset_per_footprint',
                                         'Preprocessors/h2o_ratio_offset_per_footprint','Retrieval/SigmaB',
                                         'pressure_weight', 'xco2_qf_simple_bitflag', 'xco2_qf_bitflag', 'Sounding/l1b_type', 'Sounding/orbit')]

# read in data
for j in tqdm(range(len(Lite_files))):
    l = Lite_files[j]
    l_ds = nc.Dataset(l)
    l_ids = l_ds['sounding_id'][:]
    # initialize array
    data = np.ones((len(l_ids ), len(l_vars))) * fill_value

    # load cloud data
    i = -1
    for v in l_vars:
        i += 1
        data[:, i] = l_ds[v][:]

    # change missing values to nan
    data[data == fill_value] = np.nan

    # downsample data to save RAM
    interval = int(1 / frac)
    data = data[::interval, :]

    data_all.append(data)
    counts+=1


data = np.concatenate(data_all)
data = pd.DataFrame(data, columns=l_vars)
data['sounding_id'] = data['sounding_id'].astype(int)

# clean up features
Features = data.columns
feature_dict = {}
for f in Features:
    features_clean = f.split('/')[-1]
    if f.split('/')[0] != f.split('/')[-1]:
        feature_dict[f] = features_clean
data.rename(columns=feature_dict, inplace=True)


# calculate some stats *********************************************************
# SA bias
lat = data['latitude'].to_numpy()
lon = data['longitude'].to_numpy()
SA = np.zeros_like(lat)*np.nan
SA_bias_b112 = np.zeros_like(lat)*np.nan
XCO2b112 = data['xco2_gamma'].to_numpy()
SA_bias = np.zeros_like(lat) * np.nan
XCO2 = data['xco2'].to_numpy()

s=1
i=0
j = 1
max_dist = 100 # in km
while i+j < len(data):
    j+=10
    lats = lat[i:i+j]
    lons = lon[i:i+j]
    # calculate maximum distance
    dist_max = dist(lats.max(), lats.min(), lons.max(), lons.min())
    # check if SA is to big
    if dist_max >= max_dist :
        if j < 20: # if SA is too small we leave the nan value there
            i = i + j
            j = 0
        else:
            # assign SA
            SA[i:i+j-10] = s + year * 10**8 # this should keep SAs unique across years

            # calculate SA bias
            SA_bias_b112[i:i+j-10] = XCO2b112[i:i+j-10] - np.median(XCO2b112[i:i+j-10])
            SA_bias[i:i + j - 10] = XCO2[i:i + j - 10] - np.median(XCO2[i:i + j - 10])

            s+=1
            i = i+j-10
            j=0
            print(str(np.round(i/len(data)*100,3)) + '%')


with open(save_path + 'SA_bias.txt', 'w') as f:
    for qf in [0, 1, 2]:
        # calculate mean and std of SA bias
        xco2_SA_bias_mean = np.nanmean(SA_bias[data['xco2_quality_flag_gamma']==qf])
        xco2b112_SA_bias_mean = np.nanmean(SA_bias_b112[data['xco2_quality_flag_gamma']==qf])
        xco2_SA_bias_std = np.nanstd(SA_bias[data['xco2_quality_flag_gamma']==qf])
        xco2b112_SA_bias_std = np.nanstd(SA_bias_b112[data['xco2_quality_flag_gamma']==qf])

        # write to txt file
        f.write('Quality Flag: ' + str(qf) + '\n')
        f.write('xco2_gamma_SA_bias_mean: ' + str(xco2b112_SA_bias_mean) + '\n')
        f.write('xco2_SA_bias_mean: ' + str(xco2_SA_bias_mean) + '\n')
        f.write('xco2_gamma_SA_bias_std: ' + str(xco2b112_SA_bias_std) + '\n')
        f.write('xco2_SA_bias_std: ' + str(xco2_SA_bias_std) + '\n')


# visualize data ***************************************************************
data.loc[:,'ML-B11'] = data.loc[:,'xco2_gamma'] - data.loc[:,'xco2']
data.loc[:,'ML-Raw'] = data.loc[:,'xco2_gamma'] - data.loc[:,'xco2_raw']
data.loc[:,'B11-Raw'] = data.loc[:,'xco2'] - data.loc[:,'xco2_raw']
data_all = data.copy()
for qf in [0, 1, 2]:
    data = data_all.loc[data_all['xco2_quality_flag_gamma'] == qf]
    name = name_all + 'QF' + str(qf)
    plot_map(data, ['ML-Raw', 'ML-B11', 'B11-Raw'], save_fig=save_fig, path=save_path, name=name, pos_neg_IO=True, min=-1,max=1)
    plot_map(data, ['bias_correction_uncert'], save_fig=save_fig, path=save_path, name=name, pos_neg_IO=False)




print('Done >>> ')
