# Steffen Mauceri
# make a faster to load version of data set. Purely helpful for speed

from pathlib import Path

import paths
from util import load_data

balanced = True


if balanced:
    balance_str = ''
else:
    balance_str = 'un'


for mode in ['all','LndNDGL', 'SeaGL']: #['all','LndNDGL', 'SeaGL']
    for year in range(2014, 2025):
        data = load_data(year, mode, preload_IO=False, clean_IO=True, balanced=balanced)
        
        if data is None or data.empty:
            print(f"No data loaded for year {year}, mode {mode}. Skipping preload file creation.")
            continue

        output_preload_filename = f'PreLoadB112v2_{balance_str}balanced_5M_{mode}_{year}.parquet'
        output_preload_filepath = paths.PRELOAD_DIR / output_preload_filename
        
        paths.ensure_dir_exists(paths.PRELOAD_DIR)
        data.to_parquet(output_preload_filepath)
        print(f"Saved: {output_preload_filepath}")

# versioning
# v12 variable tccon distance with wollengong and darwin
# v13 old chema for variable tccon distance
# v14 simplified tccon distance  with wollengong and darwin (only constrain mega cities)
# v15 using TCCON X_2019 and AK adjustment

# v1 B11.2 switch to B11.2
# v2 B11.2 added quality flag for strong emitters


print('Done >>>')