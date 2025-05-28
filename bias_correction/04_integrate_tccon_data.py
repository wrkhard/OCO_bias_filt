# Steffen Mauceri
# add TCCON data and combine with OCO-2 soundings


import numpy as np
import pandas as pd
import glob
import netCDF4 as nc
from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot as plt


import paths
from util import dist, load_data


# citation for TCCON
#Total Carbon Column Observing Network (TCCON) Team. (2022). 2020 TCCON Data Release (Version GGG2020) [Data set].
# CaltechDATA. https://doi.org/10.14291/TCCON.GGG2020



def apply_averaging_kernel_correction(xco2_tccon, a, co2_ap, xco2_ap):
    """
    Apply averaging kernel correction to TCCON XCO2 data.

    Parameters:
    - xco2_tccon: Retrieved TCCON XCO2 (including bias correction)
    - a: Un-normalized (raw) XCO2 averaging kernel from the OCO algorithm (array of 20 levels)
    - co2_ap: OCO CO2 prior (array of 20 levels)
    - xco2_ap: OCO2 XCO2 prior (scalar)

    Returns:
    - xco2_tccon_ak: Corrected TCCON XCO2 including the OCO-2 averaging kernel
    """
    xco2_ak_sum = np.sum(a * co2_ap)
    gamma = xco2_tccon / xco2_ap
    delta = (gamma - 1) * (xco2_ak_sum - xco2_ap)
    xco2_tccon_ak = xco2_tccon + delta
    return xco2_tccon_ak



## Read all TCCON data *******************************************
time_max = 1*60*60 # sec [1 hours]
lat_lon_max = 2 #2


# TCCON_folder = '/Volumes/OCO/TCCON/ggg2020' # folder that contains TCCON data files
# Use paths.TCCON_FILES_DIR. It's assumed TCCON files are directly in this dir or a known subdir.
# If TCCON files are in a specific subdirectory (e.g., 'ggg2020') within TCCON_FILES_DIR, adjust accordingly:
# tccon_data_path = paths.TCCON_FILES_DIR / "ggg2020"
tccon_data_path = paths.TCCON_FILES_DIR # Assuming .nc files are directly in TCCON_FILES_DIR

#ggg2020
t_vars = ['lat','long', 'xco2_x2019', 'time', 'year','day']


# load TCCON data
print('load TCCON data')
data_all = []
name_all = []
# TCCON_files = glob.glob(TCCON_folder + '/*.nc')
TCCON_files = sorted(list(Path(tccon_data_path).glob('*.nc')))
for f in TCCON_files:
    t_ds = nc.Dataset(f)
    #t_ids = t_ds['lat_deg'][:]
    t_ids = t_ds['lat'][:]
    # initialize array
    t_data = np.ones((len(t_ids), len(t_vars))) * np.nan

    i = -1
    for v in t_vars:
        i += 1
        t_data[:, i] = t_ds[v][:]

    # append data to list
    data_all.append(t_data)
    # add name
    name = t_ds.long_name
    #name = t_ds.longName
    # remove numbers from name
    name = name.split('0')[0]
    # add to name_all list
    name_all.append([name] * len(t_data))

# merge TCCON data and names
data_all = np.concatenate(data_all)
name_all = np.concatenate(name_all)

t_data = pd.DataFrame(data_all, columns= t_vars) #make into a DataFrame
t_data['tccon_name'] = name_all

# rename xc02_x2019 to xco2
t_data.rename(columns={'xco2_x2019': 'xco2'}, inplace=True)



## Match OCO-2 to TCCON ******************************************************

years = np.arange(2014, 2025)
for year in years:
    print(f"Processing year: {year}")
    # read in OCO-2 data
    # path = '/Volumes/OCO/Pkl_OCO2_B112/LiteB112_' +str(year)+ '.pkl'  # path to data
    input_parquet_file = paths.PAR_DIR / f'LiteB112_{year}.parquet' #

    if not input_parquet_file.exists():
        print(f"Warning: Input OCO-2 parquet file {input_parquet_file} not found. Skipping year {year}.")
        continue
    data = pd.read_parquet(input_parquet_file) 

    # sort data by sounding_id
    data = data.sort_values('sounding_id')

    xco2tccon_all = []
    tccon_name_all = []
    tccon_dist_all = []


    # itterate over a 10th of a day of OCO-2 soundings to speed up
    for day in tqdm(range(370*10)):
        day_time = day*24*60*6 + data['time'].iloc[0] # in seconds since 1970
        # subset data (for faster processing)
        # TCCON data
        t_data_y = t_data[(t_data['time'] >= day_time - time_max) & (t_data['time'] < day_time + 24 * 60 * 6 + time_max)]
        # OCO-2 data
        data_y = data[(data['time'] >= day_time) & (data['time'] < day_time + 24 * 60 * 6)]
        # check if we have soundings that match in time

        if len(data_y) > 0:
            xco2tccon = np.zeros((len(data_y), 1)) * np.nan
            tccon_name = np.zeros((len(data_y), 1), dtype='<U15')
            tccon_dist = np.zeros((len(data_y), 1)) * np.nan

            if len(t_data_y) > 0:

                # ggg2020
                t_lat = t_data_y['lat'].to_numpy()
                t_lon = t_data_y['long'].to_numpy()
                t_xco2 = t_data_y['xco2'].to_numpy()
                t_time = t_data_y['time'].to_numpy()
                t_name = t_data_y['tccon_name'].to_numpy()

                # subset OCO-2 data for faster computing
                lat = data_y['latitude'].to_numpy()
                lon = data_y['longitude'].to_numpy()
                time = data_y['time'].to_numpy()

                i = 0
                for i in range(len(data_y)): #itterate over OCO-2 data
                    # find oco-2 soundings that have min dist to any tccon of less than dist<max
                    if (np.min(np.abs(lat[i] - t_lat)) < lat_lon_max) & (np.min(np.abs(lon[i] - t_lon)) < lat_lon_max):
                        # find tccon data with similar time, and dist
                        xco2tccon_collection = []
                        tccon_name_collection = []
                        tccon_dist_collection = []

                        for j in range(len(t_data_y)): #itterate over TCCON data
                            if (np.abs(lat[i] - t_lat[j]) < lat_lon_max) & (np.abs(lon[i] - t_lon[j]) < lat_lon_max) & (np.abs(time[i] - t_time[j])  < time_max):
                                # collect all TCCON fits for a given OCO-2 sounding
                                xco2tccon_collection.append(t_xco2[j])
                                tccon_name_collection.append(t_name[j])
                                tccon_dist_collection.append(dist(lat[i], t_lat[j], lon[i] , t_lon[j]))


                        if len(xco2tccon_collection) == 0:
                            continue

                        # cast lists to arrays
                        xco2tccon_collection = np.array(xco2tccon_collection)
                        tccon_name_collection = np.array(tccon_name_collection)
                        tccon_dist_collection = np.array(tccon_dist_collection)


                        # check that all our match ups come from one tccon site
                        tccon_name_un = np.unique(tccon_name_collection)
                        if len(tccon_name_un) > 1:
                            # find the closest tccon site
                            avg_dist = []
                            for t in tccon_name_un:
                                avg_dist.append(np.nanmean(tccon_dist_collection[np.where(tccon_name_collection == t)]))
                            tccon_name_closest = tccon_name_un[np.argmin(avg_dist)]
                            # remove data from all other stations
                            xco2tccon_collection = np.array(xco2tccon_collection)[np.where(tccon_name_collection == tccon_name_closest)]
                            tccon_dist_collection = np.array(tccon_dist_collection)[np.where(tccon_name_collection == tccon_name_closest)]
                            tccon_name_un[0] = tccon_name_closest


                        # average over all TCCON fits for a given OCO-2 sounding
                        xco2tccon[i] = np.nanmedian(xco2tccon_collection)
                        tccon_name[i] = tccon_name_un[0]
                        tccon_dist[i] = np.nanmean(tccon_dist_collection)

                        # correct TCCON data with averaging kernel
                        pressure_weight = data_y['pressure_weight'].iloc[i]
                        xco2_ak_norm = data_y['xco2_averaging_kernel'].iloc[i] #normalized averaging kernel
                        co2_ap = data_y['co2_profile_apriori'].iloc[i]
                        xco2_ap = data_y['xco2_apriori'].iloc[i]

                        # Un-Normalize the averaging kernel
                        a = xco2_ak_norm * pressure_weight

                        # Apply averaging kernel correction
                        xco2_tccon_ak = apply_averaging_kernel_correction(xco2tccon[i], a, co2_ap, xco2_ap)


                        # diff = xco2_tccon_ak - xco2tccon[i]
                        # print('xco2tccon: ' + str(xco2tccon[i]) + ' xco2tccon_ak: ' + str(xco2_tccon_ak) + ' diff: ' + str(diff))
                        xco2tccon[i] = xco2_tccon_ak


            # collect all matches
            xco2tccon_all.append(xco2tccon)
            tccon_name_all.append(tccon_name)
            tccon_dist_all.append(tccon_dist)


    # transform lists into numpy array
    xco2tccon_all = np.concatenate(xco2tccon_all)
    tccon_name_all = np.concatenate(tccon_name_all)
    tccon_dist_all = np.concatenate(tccon_dist_all)


    # add to DataFrame
    data['xco2tccon'] = xco2tccon_all
    data['tccon_name'] = tccon_name_all
    data['tccon_dist'] = tccon_dist_all

    # # remove ram intensive variables
    data = data.drop(columns=['pressure_weight', 'xco2_averaging_kernel', 'co2_profile_apriori', 'xco2_apriori'])

    print('finished year: '+ str(year))

    data.to_parquet(input_parquet_file) # Changed to to_parquet

print('>>> Done')