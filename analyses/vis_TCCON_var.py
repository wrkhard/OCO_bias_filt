import numpy as np
import pandas as pd
import netCDF4 as nc
from util import dist, load_data
from tqdm import tqdm
from matplotlib import pyplot as plt
import paths

# Script configuration
verbose = False # Define verbose flag, as it was used but not defined

# Define a specific output path for this analysis script, relative to FIGURE_DIR
analysis_specific_output_path = paths.FIGURE_DIR / 'vis_TCCON_var_outputs'
paths.ensure_dir_exists(analysis_specific_output_path)

# citation for TCCON
#Total Carbon Column Observing Network (TCCON) Team. (2022). 2020 TCCON Data Release (Version GGG2020) [Data set].
# CaltechDATA. https://doi.org/10.14291/TCCON.GGG2020

## Read all TCCON data *******************************************
max_time = 1*60*60 # sec [1 hours]
# lat_lon_max = 2.5
max_dist_oco_tccon_match = 200 # km Maximum distance considered for OCO-2 TCCON match. Renamed from max_dist to avoid conflict with loop variable.

TCCON_folder = paths.TCCON_FILES_DIR # Use TCCON_FILES_DIR from paths.py
t_vars = ['lat','long', 'xco2', 'time', 'year','day']
t_profiles = ['prior_gravity', 'prior_co2','ak_xco2', 'xh2o', 'prior_pressure']

# load TCCON data
print('load TCCON data')
data_all_tccon = [] # Renamed to avoid conflict with OCO-2 data_all later
name_all_tccon = [] # Renamed for clarity
# Use .rglob to find .nc files recursively if they are in subdirectories, or .glob for current directory only.
# Assuming TCCON files are directly in TCCON_FILES_DIR, not subfolders.
TCCON_files = list(TCCON_folder.glob('*.nc'))

if not TCCON_files:
    print(f"Warning: No .nc files found in {TCCON_folder}. TCCON data loading will be skipped.")
    # Initialize t_data as an empty DataFrame with expected columns if no files are found
    # to prevent errors later in the script if it expects t_data to exist.
    t_data = pd.DataFrame(columns=t_vars + ['tccon_name'])
    TCCON_stations = np.array([]) # No stations if no data
else:
    for f in tqdm(TCCON_files, desc="Loading TCCON files"):
        try:
            with nc.Dataset(f) as t_ds:
                t_ids = t_ds['lat'][:] # Assuming 'lat' can be used to get the number of records
                # initialize array
                current_t_data_values = np.ones((len(t_ids), len(t_vars))) * np.nan

                i = -1
                for v in t_vars:
                    i += 1
                    if v in t_ds.variables:
                        current_t_data_values[:, i] = t_ds[v][:]
                    else:
                        print(f"Warning: Variable '{v}' not found in {f}. Filling with NaNs.")
                
                # append data to list
                data_all_tccon.append(current_t_data_values)
                # add name
                name = getattr(t_ds, 'long_name', f.stem) # Use file stem if long_name attribute is missing
                # remove numbers from name if a common pattern like "StationName01" exists
                # This is a heuristic and might need adjustment based on actual naming patterns.
                name = name.split('0')[0] if '0' in name else name 
                name = name.split(' ')[0] # Take first part if space in name
                name_all_tccon.append([name] * len(current_t_data_values))
        except Exception as e:
            print(f"Error loading TCCON file {f}: {e}")
            continue

    if data_all_tccon:
        # merge TCCON data and names
        data_all_tccon_np = np.concatenate(data_all_tccon)
        name_all_tccon_np = np.concatenate(name_all_tccon)
        t_data = pd.DataFrame(data_all_tccon_np, columns= t_vars) #make into a DataFrame
        t_data['tccon_name'] = name_all_tccon_np
        TCCON_stations = np.unique(name_all_tccon_np)
    else:
        print("No TCCON data successfully loaded.")
        t_data = pd.DataFrame(columns=t_vars + ['tccon_name'])
        TCCON_stations = np.array([])

## Determine matching criteria for each TCCON stations ***************************
# For each TCCON stations find soundings that are close to that station in space

# get OCO-2 data
years = range(2015, 2021)
columns = ['latitude', 'longitude', 'time', 'xco2_raw', 'xco2', 'xco2_quality_flag']
data_list_oco = [] # Renamed for clarity
print(f"Loading OCO-2 data from {paths.PAR_DIR}...")
for year in years:
    year_file = paths.PAR_DIR / f'LiteB111_{year}.parquet'
    if verbose:
        print(f"Loading data for year {year} from {year_file}")
    if year_file.exists():
        df = pd.read_parquet(year_file)[columns]
        data_list_oco.append(df)
    else:
        print(f"Warning: OCO-2 file not found: {year_file}")

if not data_list_oco:
    print("Error: No OCO-2 data loaded. Exiting script.")
    exit()

data_all_oco = pd.concat(data_list_oco, axis=0) # Renamed for clarity

# only keep QF = 0
data_all_oco = data_all_oco.loc[data_all_oco['xco2_quality_flag'] == 0, :]

# find matches
print('determine variability of xco2 for TCCON stations')
# TCCON_stations defined above after loading TCCON data


RMSEs_TCCON_dist = []

if not TCCON_stations.size: # Check if TCCON_stations is empty
    print("No TCCON stations loaded to process. Skipping variability calculation.")
else:
    for TCCON_station in tqdm(TCCON_stations, desc="Processing TCCON stations"):
        print(TCCON_station)
        # get TCCON lat lon
        station_data = t_data[t_data['tccon_name'] == TCCON_station]
        if station_data.empty:
            print(f"No data for TCCON station: {TCCON_station}. Skipping.")
            RMSEs_TCCON_dist.append([np.nan] * len([20,35,50,75,100,150,200])) # Append NaNs to keep array sizes consistent
            continue
            
        t_lat = station_data['lat'].iloc[0]
        t_lon = station_data['long'].iloc[0]
        
        # Create a working copy for OCO-2 data for this station to avoid modifying the full dataset repeatedly
        current_oco_data = data_all_oco.copy()
        # save the distance of each sounding to the TCCON station
        current_oco_data['spatial_dist'] = dist(current_oco_data['latitude'].values, t_lat, current_oco_data['longitude'].values, t_lon)

        data_subset_for_station = current_oco_data.loc[current_oco_data['spatial_dist'] < max_dist_oco_tccon_match, :].copy()

        if data_subset_for_station.empty:
            print(f"No OCO-2 data found within {max_dist_oco_tccon_match} km of {TCCON_station}. Skipping.")
            RMSEs_TCCON_dist.append([np.nan] * len([20,35,50,75,100,150,200]))
            continue

        # cluster data into groups when they are close in time
        time_oco = data_subset_for_station['time'].values # Renamed from time to avoid conflict
        SA = np.zeros_like(time_oco) * np.nan
        s = 1 # sounding group ID
        i = 0 # start index of current window
        j = 0 # current window size search offset (from i)
        
        # Simplified window logic for grouping soundings by time
        current_group_indices = []
        last_time = -np.inf
        
        sorted_indices = np.argsort(time_oco)
        time_sorted = time_oco[sorted_indices]
        original_indices = np.arange(len(time_oco))[sorted_indices]

        for k_idx in range(len(time_sorted)):
            current_time = time_sorted[k_idx]
            original_idx = original_indices[k_idx]

            if not current_group_indices or (current_time - time_sorted[np.argmin(time_sorted[current_group_indices])] <= max_time):
                current_group_indices.append(k_idx) # Add index in sorted array
            else:
                # Finalize previous group if it's large enough
                if len(current_group_indices) >= 10: # Minimum size for a group
                    SA[original_indices[current_group_indices]] = s
                    s += 1
                # Start new group
                current_group_indices = [k_idx]
        
        # Process the last group
        if len(current_group_indices) >= 10:
            SA[original_indices[current_group_indices]] = s
            s += 1

        # calculate variability of xco2 in each cluster for different distance
        data_subset_for_station['SA'] = SA

        # calculate variability in space ****************************************************
        # xco2_var_space_station = [] # This seems unused locally, xco2_vars_space is global but also unused
        dists_km = [20,35,50,75,100,150,200] # Renamed from dists to avoid conflict

        RMSEs_dist_station = [] # Renamed for clarity
        for dist_i in dists_km:
            data_dist_subset = data_subset_for_station.loc[data_subset_for_station['spatial_dist'] < dist_i, :]
            
            if data_dist_subset.empty:
                RMSEs_dist_station.append(np.nan)
                continue

            # Define reference data as soundings within the smallest distance bin (dists_km[0])
            data_in_ref = data_dist_subset.loc[data_dist_subset['spatial_dist'] < dists_km[0], :]
            data_out_eval = data_dist_subset # Evaluate all data within current dist_i against the reference

            RMSEs_sa = []
            valid_SAs = np.unique(data_in_ref.loc[~np.isnan(data_in_ref['SA']), 'SA'])

            if not valid_SAs.size:
                RMSEs_dist_station.append(np.nan) # No valid SAs with reference data
                continue

            for sa_val in valid_SAs:
                # Define the truth as the median of data_in_ref for the current SA
                sa_ref_data = data_in_ref.loc[data_in_ref['SA'] == sa_val, 'xco2']
                if sa_ref_data.empty or sa_ref_data.isnull().all():
                    continue # Skip SA if no valid reference data
                data_in_median = np.nanmedian(sa_ref_data)
                
                # Calculate the RMSE of data_out_eval - data_in_median for each SA
                sa_out_data = data_out_eval.loc[data_out_eval['SA'] == sa_val, 'xco2']
                if sa_out_data.empty or sa_out_data.isnull().all():
                    continue # Skip SA if no valid evaluation data
                
                RMSE = np.sqrt(np.nanmean((sa_out_data - data_in_median)**2))
                RMSEs_sa.append(RMSE)

            if not RMSEs_sa: # If no RMSEs were calculated for any SA for this dist_i
                RMSEs_dist_station.append(np.nan)
            else:
                RMSE_dist_i = np.nanmean(RMSEs_sa)
                RMSEs_dist_station.append(RMSE_dist_i)

        RMSEs_TCCON_dist.append(RMSEs_dist_station)

# make array from RMSEs_TCCON_dist, ensuring consistent shape for plotting
max_len_dists = len(dists_km) # dists_km defined above
RMSEs_TCCON_dist_np = np.array([row + [np.nan]*(max_len_dists - len(row)) if len(row) < max_len_dists else row for row in RMSEs_TCCON_dist])


# plot variability vs distance
# n_stations = len(RMSEs_TCCON_dist_np) # Renamed from n
if not RMSEs_TCCON_dist_np.size or np.all(np.isnan(RMSEs_TCCON_dist_np)):
    print("No valid RMSE data to plot. Skipping plot generation.")
else:
    n_stations = RMSEs_TCCON_dist_np.shape[0]
    # data_refs_np = np.array(data_refs) # data_refs is unused
    # xco2_vars_space_np = np.array(xco2_vars_space) # xco2_vars_space is unused

    change_percent = ((RMSEs_TCCON_dist_np / RMSEs_TCCON_dist_np[:,0:1]) - 1)*100
    threshold_percent = 25 # Renamed from threashold

    # calculate when the threshold is reached for each station
    max_dist_at_threshold = []
    valid_station_indices_for_plot = []
    for i in range(n_stations):
        if np.all(np.isnan(RMSEs_TCCON_dist_np[i,:])) or np.all(np.isnan(change_percent[i,:])):
            print(f"Skipping station {TCCON_stations[i]} for max_dist calculation due to all NaN data.")
            max_dist_at_threshold.append(np.nan)
            continue
        
        # Find where change is less than threshold_percent
        # np.where returns a tuple of arrays, take the first one for indices
        valid_indices = np.where(change_percent[i,:] < threshold_percent)[0]
        if valid_indices.size > 0:
            max_dist_val = dists_km[np.max(valid_indices)]
            valid_station_indices_for_plot.append(i)
        else: # If threshold is never met (e.g., always above or all NaNs after first point)
            # Check if first point itself is already above threshold or NaN
            if np.isnan(change_percent[i,0]) or change_percent[i,0] >= threshold_percent:
                max_dist_val = np.nan # Cannot determine a valid max_dist_km
            else: # First point is below threshold, but no other points are (or they are NaN)
                max_dist_val = dists_km[0] # Only the first distance bin meets criteria
                valid_station_indices_for_plot.append(i)
        
        max_dist_at_threshold.append(max_dist_val)
        if TCCON_stations.size > i : # Ensure TCCON_stations is not empty and i is a valid index
             print(f"{TCCON_stations[i]} max dist ({threshold_percent}% RMSE increase): {max_dist_val} km")
        else:
             print(f"Station index {i} out of bounds for TCCON_stations list. Max dist: {max_dist_val} km")

    plt.figure(figsize=(12, 10))
    markers = ['o', 's', 'v', 'd', 'p', 'h', '8', '>', '<', 'D', 'P', 'H', 'X', '1', '2', '3', '4', 'x', '+'] * ( (n_stations // 18) +1)
    plt.xscale('log')
    for i in valid_station_indices_for_plot: # Plot only stations that had some valid data for this
        if TCCON_stations.size > i:
            plt.plot(dists_km, RMSEs_TCCON_dist_np[i,:], label=TCCON_stations[i], marker=markers[i % len(markers)], markersize=7)
    
    plt.xticks(dists_km, dists_km)
    plt.xlabel('Distance [km]', fontsize=14)
    plt.ylabel('XCO2 RMSE [ppm]', fontsize=14) # Changed label to reflect RMSE not std increase
    plt.title('OCO-2 XCO2 RMSE vs. Distance from TCCON site (within co-located soundings)', fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
    plt.savefig(analysis_specific_output_path / 'xco2_rmse_vs_distance.png', dpi=300)
    # plt.show() # Typically call show after all figures for a script run, or not at all if saving
    plt.close()


    plt.figure(figsize=(12, 10))
    plt.xscale('log')
    for i in valid_station_indices_for_plot:
        if TCCON_stations.size > i:
            plt.plot(dists_km, change_percent[i,:], label=TCCON_stations[i], marker=markers[i % len(markers)], markersize=7)
    plt.hlines(threshold_percent, dists_km[0], dists_km[-1], linestyles='dashed', colors='k', label=f'{threshold_percent}% RMSE threshold')
    plt.xticks(dists_km, dists_km)
    plt.xlabel('Distance [km]', fontsize=14)
    plt.ylabel('Increase in XCO2 RMSE from closest bin [%]', fontsize=14)
    plt.title('Relative Increase in OCO-2 XCO2 RMSE vs. Distance from TCCON site', fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(analysis_specific_output_path / 'xco2_rmse_increase_percent_vs_distance.png', dpi=300)
    # plt.show()
    plt.close()

print('done >>>')