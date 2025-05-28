import numpy as np
import pandas as pd
import netCDF4 as nc
import glob
from tqdm import tqdm
from pathlib import Path

import paths

def filter_strong_emitters(data, year, emission_threshold=5000000, distance_threshold=0.2):
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
    strong_emitter_lats = lat_emi[strong_emitter_mask]
    strong_emitter_lons = lon_emi[strong_emitter_mask]
    
    # For each sounding, check if it's close to any strong emitter
    for i in tqdm(range(len(data))):
        lat = data['latitude'].iloc[i]
        lon = data['longitude'].iloc[i]
        
        # Calculate distances to all strong emitters
        lat_dist = np.abs(strong_emitter_lats - lat)
        lon_dist = np.abs(strong_emitter_lons - lon)
        
        # Check if any emitter is within the distance threshold
        if np.any((lat_dist <= distance_threshold) & (lon_dist <= distance_threshold)):
            data.loc[i, 'strong_emitter'] = 1
    
    return data

def main():
    # Example usage
    years = np.arange(2014, 2025)
    
    for year in years:
        print(f"Processing year: {year}")
        
        # Construct the input Parquet file path
        input_parquet_file = paths.PAR_DIR / f'LiteB112_{year}.parquet'

        if not input_parquet_file.exists():
            print(f"Warning: Input OCO-2 parquet file {input_parquet_file} not found. Skipping year {year}.")
            continue

        try:
            data = pd.read_parquet(input_parquet_file)
            
            # Apply strong emitter filter
            data = filter_strong_emitters(data, year)
            
            # Save results
            data.to_parquet(input_parquet_file)
            
            # Print statistics
            n_flagged = data['strong_emitter'].sum()
            print(f"Flagged {n_flagged} soundings ({n_flagged/len(data)*100:.2f}%) as near strong emitters")

            print(f"Finished processing OCO-2 data for year {year}.")
        except Exception as e:
            print(f"Error processing year {year}: {e}")

if __name__ == "__main__":
    main() 