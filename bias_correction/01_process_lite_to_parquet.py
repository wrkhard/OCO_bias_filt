# Makes pkl files out of Lite files and removes unnecessary variables, cleans up naming
# Optimized for performance
# 03/2022 Steffen Mauceri

# LiteFiles are available on the GES DISC: https://disc.gsfc.nasa.gov/datasets?keywords=oco2

import numpy as np
import pandas as pd
import glob
import netCDF4 as nc
from tqdm import tqdm
from pathlib import Path

import paths

def get_all_headers_with_dims(f):
    headers = []
    dims = []
    # Get variables in the root group
    for var_name in f.variables.keys():
        var = f.variables[var_name]
        headers.append(var_name)
        dims.append(var.ndim)
    # Get variables in subgroups
    groups = list(f.groups.keys())
    for g in groups:
        for var_name in f[g].variables.keys():
            full_var_name = g + '/' + var_name
            var = f[g].variables[var_name]
            headers.append(full_var_name)
            dims.append(var.ndim)
    return headers, dims

fill_value = 999999

years = np.arange(2014, 2025)
for year in years:
    counts = 0
    print(year)
    data_dict = {}
    # Get LiteFile data
    lite_file_pattern = f'oco2_LtCO2_{year - 2000:02d}*B11210Ar*.nc4'
    Lite_files = sorted(list(Path(paths.OCO_LITE_FILES_DIR).glob(lite_file_pattern)))

    # randomly remove 2/3 of the data to save RAM
    Lite_files = Lite_files[::3]

    # Get Lite vars
    l_ds = nc.Dataset(Lite_files[0])
    l_vars, l_dims = get_all_headers_with_dims(l_ds)
    # Remove vars we don't need
    vars_to_remove = [
        'bands', 'footprints', 'levels', 'vertices', 'Retrieval/iterations', 'file_index', 'vertex_latitude',
        'vertex_longitude', 'date', 'source_files', 'pressure_levels',
        'Preprocessors/co2_ratio_offset_per_footprint', 'Preprocessors/h2o_ratio_offset_per_footprint',
        'Retrieval/SigmaB', 'xco2_qf_simple_bitflag', 'xco2_qf_bitflag',
        'Sounding/l1b_type', 'Sounding/orbit', 'frames',
    ]
    l_vars = [e for e in l_vars if e not in vars_to_remove]
    # remove any vars that start with 'L1b'
    l_vars = [e for e in l_vars if not e.startswith('L1b')]

    # Separate variables by dimension
    l_vars_1d = []
    l_vars_2d = []
    l_vars_nd = []
    for v in l_vars:
        var = l_ds[v]
        ndim = var.ndim
        if ndim == 1:
            l_vars_1d.append(v)
        elif ndim == 2:
            l_vars_2d.append(v)
        else:
            l_vars_nd.append(v)

    # Initialize data_dict keys
    for v in l_vars_1d + l_vars_2d + l_vars_nd:
        data_dict[v] = []

    # Read in data
    for l in tqdm(Lite_files):
        l_ds = nc.Dataset(l)

        # Read 1D variables
        for v in l_vars_1d:
            val = l_ds[v][:]
            val = np.where(val == fill_value, np.nan, val)
            data_dict[v].append(val)

        # Read 2D variables
        for v in l_vars_2d:
            val = l_ds[v][:]  # shape: (num_soundings, dim2)
            val = np.where(val == fill_value, np.nan, val)
            data_dict[v].append(val)

        # Handle variables with ndim > 2
        for v in l_vars_nd:
            val = l_ds[v][:]  # shape: (num_soundings, dim2, dim3, ...)
            val = np.where(val == fill_value, np.nan, val)
            data_dict[v].append(val)

        counts += 1

    # Concatenate data from all files
    for v in data_dict.keys():
        data_dict[v] = np.concatenate(data_dict[v], axis=0)

    # Create DataFrame
    data_all_df = pd.DataFrame()

    # Add 1D variables to DataFrame
    for v in l_vars_1d:
        print(v)
        data_all_df[v] = data_dict[v]

    # Add 2D variables to DataFrame (each entry is an array)
    for v in l_vars_2d:
        data_all_df[v] = list(data_dict[v])

    # Handle variables with ndim > 2 if needed (each entry is an array)
    for v in l_vars_nd:
        data_all_df[v] = list(data_dict[v])

    # Clean up features
    Features = data_all_df.columns
    feature_dict = {}
    for f in Features:
        features_clean = f.split('/')[-1]
        if f.split('/')[0] != f.split('/')[-1]:
            feature_dict[f] = features_clean
    data_all_df.rename(columns=feature_dict, inplace=True)

    # Ensure 'sounding_id' is int
    data_all_df['sounding_id'] = data_all_df['sounding_id'].astype(int)

    # sort data by sounding_id
    data_all_df = data_all_df.sort_values('sounding_id')

    # Save the combined DataFrame to a Parquet file
    print(f"Saving combined data for year {year} to Parquet...")
    paths.ensure_dir_exists(paths.PAR_DIR)
    output_filename = paths.PAR_DIR / f'LiteB112_{year}.parquet'
    data_all_df.to_parquet(output_filename)
    print(f"Successfully saved to {output_filename}")

print('>>> Done') 