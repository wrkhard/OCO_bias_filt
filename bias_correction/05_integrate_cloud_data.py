# Steffen Mauceri
# 02/2023
# add nearest cloud distance and combine with OCO-2 soundings
# cloud data available for 2014 to 2019


import numpy as np
import pandas as pd
import glob
import netCDF4 as nc
from tqdm import tqdm
from pathlib import Path

import paths


years = np.arange(2014, 2025)
for year in years:
    print(f"Processing year: {year}")
    # read in OCO-2 data
    # path = '/Volumes/OCO/Pkl_OCO2_B112/LiteB112_' +str(year)+ '.pkl'  # path to data
    input_parquet_file = paths.PAR_DIR / f'LiteB112_{year}.parquet'
    if not input_parquet_file.exists():
        print(f"Warning: Input OCO-2 parquet file {input_parquet_file} not found. Skipping year {year}.")
        continue
    data = pd.read_parquet(input_parquet_file)

    # load cloud data
    # get 3D cloud data
    # cloud_files = glob.glob('/Volumes/OCO/3D_cloud_metrics_OCO2_V9/3d_cloud_metrics_oco2_v9_' + str(year) + '/oco2_*.nc4')
    cloud_year_dir = paths.CLOUD_DATA_DIR / f'3d_cloud_metrics_oco2_v9_{year}' # Construct path to year-specific subdir
    if not cloud_year_dir.exists():
        print(f"Warning: Cloud data directory {cloud_year_dir} not found. Skipping year {year} for cloud data.")
        # Decide if to continue processing the year without cloud data or skip entirely
        # For now, just add NaNs for cld_dist and save.
        data['cld_dist'] = np.nan 
        data.to_parquet(input_parquet_file)
        continue

    cloud_files = sorted(list(cloud_year_dir.glob('oco2_*.nc4')))
    if not cloud_files:
        print(f"Warning: No cloud files found in {cloud_year_dir} for pattern 'oco2_*.nc4'. Skipping year {year} for cloud data.")
        data['cld_dist'] = np.nan
        data.to_parquet(input_parquet_file)
        continue

    c_ids = []
    c_dists = []
    for c in cloud_files:
        c_ds = nc.Dataset(c)
        # get sounding id and nearest cloud distance of cloud file
        c_ids.append(c_ds['sounding_id'][:])
        c_dists.append(c_ds['cld_dist'][:])
    c_ids = np.hstack(c_ids)
    c_dists = np.hstack(c_dists)
    # remove missing values
    c_ids = c_ids[c_dists != -999999]
    c_dists = c_dists[c_dists != -999999]

    # make dictionary of ids
    print('make dict of ids')
    c_ids_dict = {k: v for v, k in enumerate(c_ids)}
    OCO_ids_dict = {k: v for v, k in enumerate(data['sounding_id'].to_numpy())}

    cld_dist = np.zeros((len(data), 1)) * np.nan
    for c_id in tqdm(c_ids_dict.keys()):
        # for c_id in c_ids_dict.keys():
        if c_id in OCO_ids_dict.keys():
            c_index = c_ids_dict[c_id]
            OCO_index = OCO_ids_dict[c_id]

            # get values for all vars
            cld_dist[OCO_index] = c_dists[c_index]

    data['cld_dist'] = cld_dist


    data.to_parquet(input_parquet_file)


print('Done >>>')