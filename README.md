# OCO-2/3 Bias Correction and Filtering Pipeline

## Overview

This project implements and explores the effect of applying a new bias correction and quality filtering approach to increase the accuracy of atmospheric CO2 measurements derived from the Orbiting Carbon Observatory-2 (OCO-2) satellite. This is not an official OCO-2 data product. For details on the approach, please refer to:


https://doi.org/10.22541/essoar.174164198.80749970/v1
and
https://doi.org/10.22541/essoar.174164203.37422284/v1

## License

The source code is licensed under the terms found in the `LICENSE` file.

## Setup

To set up the required Python environment, use the provided `environment.yml` file with Conda:

```bash
conda env create -f environment.yml
conda activate bias_filt
```

Data paths are managed via `paths.py`. For external data, you may need to set the `OCO_DATA_BASE` environment variable to point to your data directory. See `paths.py` for details.

## Running the Pipeline

The main way to execute the processing pipeline is by using the `run_bias_correction_pipeline.py` script. 
This script manages the execution of individual processing steps in the correct order and can resume from the last completed step.

To run the pipeline starting from the beginning or resuming from the last completed step:
```bash
python run_bias_correction_pipeline.py
```

To clean the pipeline status file (forcing the pipeline to start from scratch on the next run):
```bash
python run_bias_correction_pipeline.py --clean-status
```

## Processing Scripts Details

The following scripts constitute the processing pipeline and are called by `run_bias_correction_pipeline.py`. They are generally intended to be run in the order listed below if run manually.

# Processing steps of data for bias correction and filtering of OCO2/3 data

#merge OCO2 Lite files to Parquet
bias_correction/01_create_initial_parquet.py
- Converts OCO-2/3 Lite files (netCDF format) to parquet files
- Removes unnecessary variables and cleans up naming conventions
- Optimizes for performance by removing redundant data
- Handles different data dimensions and formats them into a pandas DataFrame

# make SA and calculate SA bias
bias_correction/02_create_small_areas.py
- Creates Small Area (SA) groupings of soundings
- Calculates SA biases in XCO2 retrievals
- Helps identify systematic biases in small geographic regions

# flag SA on coast lines
bias_correction/03_flag_coastal_soundings.py
- Flags small areas that cross from land to ocean
- Identifies coastal regions where land-water transitions occur
- Helps handle special cases in bias correction near coastlines

# add TCCON data to dataset
bias_correction/04_integrate_tccon_data.py
- Adds TCCON (Total Carbon Column Observing Network) data to the dataset
- Matches OCO-2/3 soundings with nearby TCCON stations
- Calculates distances to TCCON stations
- Adds TCCON XCO2 values and station names to the dataset

# add clouds
bias_correction/05_integrate_cloud_data.py
- Adds cloud information to the dataset
- Includes cloud distance and cloud fraction data
- Helps identify and filter out cloud-contaminated soundings

# add model to dataset
bias_correction/06_integrate_flux_model_data.py
- Adds model data (like GEOS-Chem) to the dataset
- Matches model output with OCO-2/3 soundings in time and space
- Provides additional context for bias correction

# flag strong emitters
bias_correction/07_filter_strong_emission_sources.py

# data cleaning
bias_correction/08_remove_outliers.py
- Performs initial data cleaning and quality filtering
- Applies various quality flags based on retrieval parameters
- Removes problematic soundings (e.g., snow-covered areas, high aerosol loading)
- Has different filtering criteria for land and ocean soundings

# allow for faster data loading
bias_correction/09_prepare_model_input_data.py
- Creates preloaded data files for faster processing
- Optimizes data loading for subsequent analysis
- Reduces memory usage and processing time

# (Optional) perform feature selection to optimize model inputs
bias_correction/10_feature_selection.py
- Analyzes feature importance and selects an optimal set for model training
- Helps improve model performance and reduce complexity

# train model
bias_correction/11_train_bias_correction_model.py
bias_correction/12_train_bias_correction_model_spatially_weighted.py
bias_correction/13_train_bias_correction_model_kfold_validation.py
bias_correction/14_train_bias_correction_model_spatially_weighted_kfold_validation.py
- Trains machine learning models for bias correction
- Uses Random Forest and other ML algorithms
- Corrects systematic biases in XCO2 retrievals
- Handles both TCCON and Small Area biases

# train filter models (should be run in directory where you want to save optuna trials and plots)
optimize_filter.py 
- Optimizes quality filtering parameters
- Uses Optuna for hyperparameter optimization
- Balances data quality and throughput
- Has separate optimization for land and ocean soundings

# additional plots (optional)
visualization_scripts/vis_bias_corr.py
- Creates visualization plots of bias correction results
- Compares corrected data with TCCON measurements
- Shows spatial patterns of biases
- Analyzes performance at land-water crossings

The pipeline follows a logical flow:
1. Data preparation (bias_correction/01_create_initial_parquet.py, bias_correction/02_create_small_areas.py, bias_correction/03_flag_coastal_soundings.py)
2. Integration of external datasets (bias_correction/04_integrate_tccon_data.py, bias_correction/05_integrate_cloud_data.py, bias_correction/06_integrate_flux_model_data.py)
3. Removing outliers (bias_correction/07_filter_strong_emission_sources.py, bias_correction/08_remove_outliers.py)
4. Data Preloading for ML (bias_correction/09_prepare_model_input_data.py)
5. (Optional) Feature Selection (bias_correction/10_feature_selection.py)
6. Model Training (e.g., bias_correction/11_train_bias_correction_model.py and its variants)
7. Filter Optimization (optimize_filter.py)
8. Visualization (visualization_scripts/vis_bias_corr.py)


## Citation

If you use this code or the resulting data, please cite the following preprints:

https://doi.org/10.22541/essoar.174164198.80749970/v1 and 
https://doi.org/10.22541/essoar.174164203.37422284/v1
