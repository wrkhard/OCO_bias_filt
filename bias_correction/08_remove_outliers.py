# Steffen Mauceri
# 09/22
# removes some severe outliers and sets their QF=2 (data cleaning).


from pathlib import Path
import numpy as np
import pandas as pd
import netCDF4 as nc
from tqdm import tqdm
from scipy.spatial import cKDTree

import paths
from util import load_data

def filter_strong_emitters(data, year, emission_threshold=500000, distance_threshold=0.2):
    """
    Filter out soundings that are close to strong CO2 emitters.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing OCO-2/3 soundings with latitude and longitude
    year : int
        Year to match with emission data
    emission_threshold : float
        Threshold for strong emitters in tonnes (default: 5,000,000)
    distance_threshold : float
        Maximum distance in degrees to consider an emitter (default: 0.2)
    
    Returns:
    --------
    pandas.DataFrame
        Original DataFrame with added 'strong_emitter' quality flag
    """

    if year == 2024:
        year = 2023
    
    # Load emission data for the given year
    emission_filename = f'EDGAR_2024_GHG_CO2_{year}_TOTALS_emi.nc'
    emission_filepath = paths.EMISSION_DATA_DIR / emission_filename
    
    if not emission_filepath.exists():
        print(f"Warning: Emission file {emission_filepath} not found for year {year}. Skipping strong emitter filtering for this year.")
        data['strong_emitter'] = np.zeros(len(data), dtype=int)
        return data

    with nc.Dataset(emission_filepath, 'r') as ds:
        emissions = ds['emissions'][:]
        lat_emi = ds['lat'][:]
        lon_emi = ds['lon'][:]
    
    # Initialize quality flag
    data['strong_emitter'] = np.zeros(len(data), dtype=int)
    
    # Find strong emitters
    strong_emitter_mask = emissions > emission_threshold
    
    # Expand lat and lon to 2D arrays
    lat_2d, lon_2d = np.meshgrid(lat_emi, lon_emi, indexing='ij')
    
    # Get coordinates of strong emitters
    strong_emitter_lats = lat_2d[strong_emitter_mask]
    strong_emitter_lons = lon_2d[strong_emitter_mask]
    
    # Create a KDTree for efficient nearest neighbor search
    emitters_tree = cKDTree(np.column_stack((strong_emitter_lats, strong_emitter_lons)))
    # Query the tree for all soundings at once
    coords = np.column_stack((data['latitude'].values, data['longitude'].values))
    distances, _ = emitters_tree.query(coords)
    # Set flag for soundings within distance threshold
    data['strong_emitter'] = (distances <= distance_threshold).astype(int)
    
    return data


years = np.arange(2014, 2025)

for year in years:
    print(f"Processing OCO-2 data for year {year}...")

    # Construct the input Parquet file path
    input_parquet_file = paths.PAR_DIR / f'LiteB112_{year}.parquet'

    # Check if the input file exists
    if not input_parquet_file.exists():
        print(f"Warning: Input OCO-2 parquet file {input_parquet_file} not found. Skipping year {year}.")
        continue

    # Load the data from the Parquet file
    data = pd.read_parquet(input_parquet_file)

    # rename h2o_ratio_bc column to h2o_ratio
    data.rename(columns={'LoFI_m2ccv1bsim  ': 'LoFI_m2ccv1bsim', 'UnivEd_v5.2      ': 'UnivEd_v5.2', 'MACC_v21r1       ':'MACC_v21r1'}, inplace=True)

    print(str(len(data)) + ' samples loaded')
    # filter data
    drop = np.zeros_like(data['xco2'], dtype=bool)

    drop[data['h2o_ratio_bc'] >= 1.1] = True
    drop[(data['co2_grad_del'] <= -100)] = True
    drop[(data['co2_grad_del'] >= 100)] = True
    print(str(np.sum(drop)))
    #land
    print('lnd')
    drop[(data['snow_flag'] == 1) & (data['land_fraction'] == 100)] = True  # remove snow
    drop[(data['co2_ratio_bc'] >= 1.04) & (data['land_fraction'] == 100)] = True
    drop[(data['co2_ratio_bc'] <= 0.99) & (data['land_fraction'] == 100)] = True
    print(str(np.sum(drop))) # to keep track of which filters removes a lot of data
    drop[(data['dpfrac'] >= 5) & (data['land_fraction'] == 100)] = True
    drop[(data['dpfrac'] <= -7.5) & (data['land_fraction'] == 100)] = True
    drop[(data['aod_ice'] >= 0.2) & (data['land_fraction'] == 100)] = True
    drop[(data['dws'] >= 1.0) & (data['land_fraction'] == 100)] = True
    print(str(np.sum(drop)))
    drop[(data['dust_height'] <= 0.7) & (data['land_fraction'] == 100)] = True
    drop[(data['dust_height'] >= 1.75) & (data['land_fraction'] == 100)] = True
    drop[(data['rms_rel_sco2'] >= 1) & (data['land_fraction'] == 100)] = True
    drop[(data['snr_sco2'] <= 50) & (data['land_fraction'] == 100)] = True
    print(str(np.sum(drop)))
    drop[(data['max_declocking_wco2'] >= 1.5) & (data['land_fraction'] == 100)] = True
    print('max_declocking ' + str(np.sum(drop)))
    drop[(data['h_continuum_wco2'] >= 100) & (data['land_fraction'] == 100)] = True
    print('h_continuum_wco2 ' + str(np.sum(drop)))
    drop[(data['deltaT'] >= 1.5) & (data['land_fraction'] == 100)] = True
    print('deltaT ' + str(np.sum(drop)))
    drop[(data['albedo_slope_wco2'] >= 0) & (data['land_fraction'] == 100)] = True
    print('albedo_slope_wco2 ' + str(np.sum(drop)))
    drop[(data['albedo_quad_sco2'] >= 0.000005) & (data['land_fraction'] == 100)] = True
    drop[(data['albedo_quad_sco2'] <= -0.000005) & (data['land_fraction'] == 100)] = True
    print('albedo_quad_sco2 ' + str(np.sum(drop)))
    #sea
    print('sea')
    drop[(data['co2_ratio_bc'] <= 0.99) & (data['land_fraction'] == 0)] = True
    drop[(data['co2_ratio_bc'] >= 1.03) & (data['land_fraction'] == 0)] = True
    print(str(np.sum(drop)))
    drop[(data['deltaT'] <= -0.8) & (data['land_fraction'] == 0)] = True
    print('deltaT ' + str(np.sum(drop)))
    drop[(data['rms_rel_sco2'] >= 0.5) & (data['land_fraction'] == 0)] = True
    print('rms ' + str(np.sum(drop)))
    drop[(data['snr_sco2'] <= 200) & (data['land_fraction'] == 0)] = True
    print('snr ' + str(np.sum(drop)))
    drop[(data['max_declocking_wco2'] >= 0.75) & (data['land_fraction'] == 0)] = True
    print('max_declocking ' + str(np.sum(drop)))

    xco2_MLquality_flag = np.zeros(len(data))
    xco2_MLquality_flag[drop] = 2
    data['xco2_MLquality_flag'] = xco2_MLquality_flag

    print(str(np.sum(1-drop)) + ' filtered samples left')

    # mark small areas that are close to strong CO2 emitters so we can remove them later
    print('Filtering strong emitters...')
    data = filter_strong_emitters(data, year)
    n_flagged = data['strong_emitter'].sum()
    print(f"Flagged {n_flagged} soundings ({n_flagged/len(data)*100:.2f}%) as near strong emitters")
    drop[data['strong_emitter'] == 1] = True
    print(str(np.sum(1-drop)) + ' filtered samples left after strong emitter filtering')

    ## remove TCCON matches that are too far away from OCO-2
    # dictionary of selected TCCON sites that have a different max_dist than the default
    TCCON_dist_select = {'bremen': 35,
                         'burgos': 100,
                         'easttroutlake': 75,
                         'edwards': 35,
                         'eureka': 100,
                         'fourcorners': 35,
                         'garmisch': 20,
                         'harwell': 100,
                         'hefei': 20,
                         'indianapolis': 50,
                         'jpl': 20,
                         'karlsruhe': 50,
                         'nicosia': 20,
                         'nyalesund':20,
                         'orleans': 50,
                         'paris': 100,
                         'pasadena': 20,
                         'reunion': 20,
                         'rikubetsu': 75,
                         'saga': 75,
                         'sodankyla': 150,
                         'xianghe':35}


    drop = np.zeros_like(data['xco2'], dtype=bool)
    # remove all TCCON matches that are more than 200 km away from OCO-2
    drop[data['tccon_dist'] > 100] = True
    # itterate over TCCON sites in TCCON_dist_select
    for site in TCCON_dist_select.keys():
        #print(site)
        # find the TCCON matches that are too far away from OCO-2
        drop[(data['tccon_name'] == site) & (data['tccon_dist'] > TCCON_dist_select[site])] = True
    # set their QF to 1
    TCCON_dist_quality_flag = np.zeros(len(data))
    TCCON_dist_quality_flag[drop] = 1
    data['tccon_dist_quality_flag'] = TCCON_dist_quality_flag
    print(str(np.sum(drop)) + ' TCCON matches removed')


    # drop diagnostic vars to save RAM
    data.drop(columns={'xco2_weak_idp', 'xco2_strong_idp', 'fs', 'fs_rel', 's31', 's32', 'eof3_1_rel','diverging_steps',
                       'polarization_angle', 'brdf_weight_slope_wco2', 'brdf_weight_slope_sco2', 'albedo_quad_o2a',
                       'albedo_quad_wco2', 'albedo_quad_sco2',
                       'snow_flag', 'psurf_apriori_o2a', 'psurf_apriori_wco2', 'psurf_apriori_sco2', 'surface_type'}, inplace=True, errors='ignore')


    # Save filtered data to a new Parquet file
    paths.ensure_dir_exists(paths.PAR_DIR)
    output_filtered_parquet_file = paths.PAR_DIR / f'LiteB112v2_filt_{year}.parquet'
    data.to_parquet(output_filtered_parquet_file)

    print(f"Finished processing year {year}. Filtered data saved to {output_filtered_parquet_file}")

print('Done >>>')
