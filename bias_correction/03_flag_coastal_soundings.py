# flag small areas that cross from land to ocean

import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

import paths

years = np.arange(2014, 2025)

# flag coast
print('flag coast areas')
for year in years:
    print(f"Processing year: {year}")

    input_parquet_file = paths.PAR_DIR / f'LiteB112_{year}.parquet'
    if not input_parquet_file.exists():
        print(f"Warning: Input file {input_parquet_file} not found. Skipping year {year}.")
        continue
    data = pd.read_parquet(input_parquet_file)
    # if 'coast' in data.
    #     continue
    # else:
    SA = list(pd.unique(data['SA']))

    idx_flag = []
    data_SA = data['SA'].to_numpy()
    land_frac = data['land_fraction'].to_numpy()
    coast = np.zeros(len(data))
    # itterate over SAs
    for i in tqdm(range(len(SA))):
        a = SA[i]
        idx = data_SA == a
        # check if it mixes land and ocean
        land_types = len(np.unique(land_frac[idx]))

        if land_types > 1:
            # check that we min altitude is 0
            if data['altitude'][idx].min() == 0:
                coast[idx] = 1


    # make flag
    data['coast'] = coast

    print('flaged ' + str(data['coast'].mean()*100) + '% of soundings')

    # save output
    print(f"Saving updated data to {input_parquet_file}")
    data.to_parquet(input_parquet_file)
    print(f'>>> Done with {year}')

print('>>> All years processed')