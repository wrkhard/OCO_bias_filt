# Steffen Mauceri
# add Model data compiled by Chris


import numpy as np
import pandas as pd
import glob
import h5py as h5
from pathlib import Path
from tqdm import tqdm

import paths
from util import dist


# OCO-2 Sounding_id: YYYYMMDDhhmmssmf. {YYYY=year, MM=month 1-12, DD=day 1-31, hh=hour 0-23, mm=minute 0-59, ss=seconds 0-59, m=hundreds of milliseconds 0-9, f=footprint number 1-8}



time_max = 10 # sec [+- 5 seconds]

# Define the model filename
model_data_filename = "oco2_b11b_odell_4modelRef_20230424.h5"
models_filepath = paths.MODEL_INPUT_DATA_DIR / model_data_filename

if not models_filepath.exists():
    print(f"CRITICAL ERROR: Models data file {models_filepath} not found. Exiting.")
    exit() # Or raise an error

# load Model data
file = h5.File(models_filepath)
m_names = file['model_names'][:]
m_ids = file['id'][:]
m_xco2 = file['xco2'][:]
# get date from sounding_id
m_date = m_ids//10**6



years = np.arange(2014, 2025)
for year in years:
    print(f"Processing year: {year}")

    # Load OCO-2 data from Parquet file
    input_parquet_file = paths.PAR_DIR / f'LiteB112_{year}.parquet' # Changed extension and variable name
    if not input_parquet_file.exists():
        print(f"Warning: Input OCO-2 parquet file {input_parquet_file} not found. Skipping year {year}.")
        continue
    data = pd.read_parquet(input_parquet_file)
    # sort data by sounding id
    data.sort_values('sounding_id', inplace=True)
    o_ids = data['sounding_id'].to_numpy()
    # remove footprint and milliseconds
    o_ids = o_ids//100
    # get date from sounding_id
    o_date = o_ids//10**6

    xco2_model_all = []

    # itterate over dates of OCO-2 soundings to speed up
    dates = np.unique(o_date)
    for day in tqdm(range(len(dates))):
        day_time = dates[day]
        # subset data (for faster processing)
        # Model data
        m_xco2_y = m_xco2[m_date == day_time , :]
        m_ids_y = m_ids[m_date == day_time]
        # OCO-2 data
        o_ids_y = o_ids[o_date == day_time]


        xco2_model = np.zeros((len(o_ids_y), len(m_names))) * np.nan
        # check if we have soundings that match in time
        if len(m_xco2_y) > 0:
            # find oco-2 soundings that have a match to model
            for i in range(len(o_ids_y)):  # itterate over OCO-2 data
                # for j in range(len(m_ids_y)):  # itterate over Model data
                #     if np.abs(o_ids_y[i] - m_ids_y[j]) <= time_max:
                #         xco2_model[i,:] = m_xco2_y[j,:]
                j = np.argmin(np.abs(o_ids_y[i] - m_ids_y))
                if o_ids_y[i] - m_ids_y[j] <= time_max:
                    xco2_model[i, :] = m_xco2_y[j, :]

        # collect all matches
        xco2_model_all.append(xco2_model)

    # transform lists into numpy array
    xco2_model_all = np.concatenate(xco2_model_all)

    # print how many matches we have as percentage
    print('Matched soundings: ' + str(np.sum(~np.isnan(xco2_model_all[:,0]))/len(xco2_model_all[:,0])*100) + '%')

    # add to DataFrame
    for i in range(len(m_names)):
        name = m_names[i].decode("utf-8")
        data[name] = xco2_model_all[:,i]

    print(f'Memory usage: {data.memory_usage(deep=True).sum() / 1024**3:.2f} GB')
    print(f'Processed {len(data)} soundings for year {year}.')
    data.to_parquet(input_parquet_file) # Overwrite the input file. Changed to to_parquet

print(">>> Done.")