# Steffen Mauceri
# makes SA based on distance and calculcate SA bias


import numpy as np
import pandas as pd
from pathlib import Path

import paths
from util import dist

max_dist = 100 # in km

years = np.arange(2014, 2025)
# load data
for year in years:
    print(f"Processing year: {year}")

    input_parquet_file = paths.PAR_DIR / f'LiteB112_{year}.parquet'
    if not input_parquet_file.exists():
        print(f"Warning: Input file {input_parquet_file} not found. Skipping year {year}.")
        continue
    data = pd.read_parquet(input_parquet_file)

    lat = data['latitude'].to_numpy()
    lon = data['longitude'].to_numpy()
    SA = np.zeros(len(lat))*np.nan
    SA_bias_raw = np.zeros_like(lat)*np.nan
    XCO2_raw = data['xco2_raw'].to_numpy()
    SA_bias = np.zeros_like(lat) * np.nan
    XCO2 = data['xco2'].to_numpy()
    s=1
    i=0
    j = 1
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
                SA_bias_raw[i:i+j-10] = XCO2_raw[i:i+j-10] - np.median(XCO2_raw[i:i+j-10])
                SA_bias[i:i + j - 10] = XCO2[i:i + j - 10] - np.median(XCO2[i:i + j - 10])

                s+=1
                i = i+j-10
                j=0
                print(str(np.round(i/len(data)*100,3)) + '%')

    print('add SA and biases to data')
    data['SA'] = SA
    data['xco2_SA_bias'] = SA_bias
    data['xco2raw_SA_bias'] = SA_bias_raw


    print('saving '+str(year)+' file')
    # save everything to csv
    data.to_parquet(input_parquet_file) # Changed to to_parquet
    print(f'>>> Done with {year}')

print('>>> All years processed')
