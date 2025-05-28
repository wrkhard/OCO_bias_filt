# Contact: William<dot>R<dot>Keely<at>jpl<dot>nasa<dot>gov OR William<dot>Keely<at>gmail<dot>com
#
# Optimize filter returns potential model parameters and choices for abstention threshold that are pareto optimal solutions.
#
# May 2, 2024

import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import precision_recall_fscore_support, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.inspection import PartialDependenceDisplay

from tqdm import tqdm
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree
import warnings

# import umap
# import pynndescent

import seaborn as sns

# change until to util_current when working locally
from util import *


# multi objective optimization
import optuna
import plotly.express as px


# arument parser
import argparse

# suppress warnings
warnings.filterwarnings("ignore")



def train_classifier(X_train, y_train, features, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', random_state=42, n_jobs=10, class_weight={1.0, 1.0}, sample_weights=None, model_type='RandomForest'):
    """
    Train a classifier on the given data and return the trained model
    :param X_train: pandas DataFrame, the data to train the classifier on
    :param y_train: pandas Series, the target variable
    :param features: list of strings, the features to use for training
    :param model_type: string, the type of model to train, default is RandomForest
    :param n_estimators: int, number of trees in the forest
    :param max_depth: int, maximum depth of the tree
    :param min_samples_split: int, minimum number of samples required to split an internal node
    :param min_samples_leaf: int, minimum number of samples required to be at a leaf node
    :param max_features: int, number of features to consider when looking for the best split
    :param random_state: int, random state for reproducibility
    :param n_jobs: int, number of jobs to run in parallel
    :param class_weight: dict, class weights for the classifier
    :return: trained model
    """
    X_train = X_train[features]
    qf_0_weight = class_weight[0]
    qf_1_weight = class_weight[1]

    M = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split,
                               min_samples_leaf=min_samples_leaf, max_features=max_features, random_state=random_state,
                               n_jobs=n_jobs, class_weight={0: qf_0_weight, 1: qf_1_weight})
    
    # fit classifier filter
    if sample_weights is not None:
        M.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        M.fit(X_train, y_train)

    preds = M.predict(X_train)

    return M, preds

def sounding_fraction(data_train, res = 4):
    """
    Calculate the sounding fraction for the given gridding resolution
    :param data_train: pandas DataFrame, the data to calculate the sounding fraction for
    :param res: int, the resolution of the sounding fraction
    :return: pandas DataFrame, the data with the sounding fraction added
    """
    print('calculating sounding fraction ...')
    # raster data
    Lat = data_train['latitude'].to_numpy()
    Lon = data_train['longitude'].to_numpy()
    weights = np.zeros((len(data_train)))
    # n = len(data_train)
    # n_month is the average number of soundings per month
    n = len(data_train)/12



    for i in tqdm(range(len(np.arange(-90, 90, res)))):
        Lat_i = np.arange(-90, 90, res)[i]
        for Lon_i in range(-180, 180, res):
            # check if we have measurements in bin
            match = (Lat >= Lat_i) & (Lat < Lat_i + res) & (Lon >= Lon_i) & (Lon < Lon_i + res)
            bin = np.sum(match)
            if bin > 0:
                weights[match] = bin/n 
            else:
                weights[match] = 0

    return weights / weights.max() * 100

def SA_label_density_weighting(data_train, gamma = 0, res = 4):
    """
    Calculate the sounding density weighting for the SA labels
    :param data_train: pandas DataFrame, the data to calculate the SA label density weighting for
    :param alpha: float, the alpha value for the SA label density weighting
    :return: pandas DataFrame, the data with the SA label density weighting added
    """
    print('calculating SA label density weighting ...')

    if 'sounding_fraction' not in data_train.columns:
        data_train['sounding_fraction'] = sounding_fraction(data_train, res=res)

    # SA density weighting
    data_train['SA_label_density_weighting'] = np.abs(data_train['xco2raw_SA_bias']) * (1 + gamma *data_train['sounding_fraction'])

    return data_train

def land_filter_trial(trial):
    """
    Define the parameters for the land filter
    :param trial: optuna trial object
    :return: dict, the obective scores for the land filter
    """
    
    # *********** MAKE CHANGES IF OCO-3 ****************************************************************************
    data_train = load_and_concat_years(2015, 2021, 'all', verbose_IO=True, qf=None, preload_IO=True)
    data_test = load_data(2022, 'all', verbose_IO=True, qf=None, preload_IO=True)


    data_train = data_train[data_train['land_fraction'] == 100]
    data_test = data_test[data_test['land_fraction'] == 100]

    # data_train = data_train.sample(n = int(5E+6), axis=0,random_state=42)
    # data_test = data_test.sample(n = int(3E+6), axis=0,random_state=42)
    
    # bias correct -- current paths are for Will's local machine
    TC_lnd = '/Users/williamkeely/Desktop/B11/current/bias_models_current/current_lnd/TCCON_V12_LndNDGL_RF.p' # change to your path
    SA_lnd = '/Users/williamkeely/Desktop/B11/current/bias_models_current/current_lnd/V12_prec_xco2raw_SA_LndNDGL_RF.p'
    # ************************************************************************************************************************

    data_train = bias_correct(TC_lnd,data_train, ['xco2_raw'], uq=True,proxy_name='TCCON')
    data_train = bias_correct(SA_lnd,data_train, ['xco2_raw'], uq=True,proxy_name='SA')
    data_test = bias_correct(TC_lnd,data_test, ['xco2_raw'], uq=True,proxy_name='TCCON')
    data_test = bias_correct(SA_lnd,data_test, ['xco2_raw'], uq=True,proxy_name='SA')

    data_train['xco2MLcorr'] = data_train['xco2_raw']
    data_test['xco2MLcorr'] = data_test['xco2_raw']
    data_train['h2o_ratio_bc'] = data_train['h2o_ratio']
    data_test['h2o_ratio_bc'] = data_test['h2o_ratio']
    data_train['co2_ratio_bc'] = data_train['co2_ratio']
    data_test['co2_ratio_bc'] = data_test['co2_ratio']
    data_train['aod_diff'] = data_train['aod_total'] - data_train['aod_total_apriori']
    data_test['aod_diff'] = data_test['aod_total'] - data_test['aod_total_apriori']

    # define features ************************************************ MAKE CAHNGES HERE TO FEATURES ************************************************
    feats_tc_lnd = ['altitude_stddev',  'bias_correction_uncert', 'aod_sulfate',  't700', 'xco2_uncertainty',
                    'albedo_wco2', 'h2o_ratio_bc', 'tcwv_uncertainty', 'albedo_sco2', 'h2o_scale', 
                    'aod_total', 'co2_ratio_bc', 'zlo_o2a', 'dp_abp', 'rms_rel_sco2',
                    'glint_angle', 'dust_height', 'dp_o2a', 'snr_sco2', 'dp_sco2', 'water_height', 
                    'aod_dust', 'ice_height', 'dws', 'dpfrac', 'aod_water', ]


    feats_sa_lnd = ['bias_correction_uncert', 'co2_ratio_bc', 'rms_rel_sco2', 'altitude_stddev', 'xco2_uncertainty', 'h2o_ratio_bc', 'dust_height', 
                    'albedo_sco2', 'aod_diff', 'water_height', 'snr_sco2', 'dpfrac', 'ice_height', 'aod_water', 'dp_o2a', 'albedo_wco2', 
                    'dp_sco2', 't700', 'aod_dust', 'tcwv_uncertainty']
    # **************************************************************************************************************************************

    # PRIORS for Bayesian Optimization

    # define priors for the training label weights
    gamma = trial.suggest_loguniform('gamma', 0.001, 1000.0)
    data_train = SA_label_density_weighting(data_train, gamma = gamma)
    tccon_weight_bool = trial.suggest_categorical('tccon_weight_bool', [True, False])

    # define priors for the defining filter labels
    abstention_threshold = trial.suggest_float('abstention_threshold', 0.5, 2.0, step = 0.05)
    percentile = trial.suggest_int('percentile', 30, 90, step = 5)

    # define the prior for the RF hyperparameters
    if percentile >= 70:
        class_weight = (trial.suggest_float('class_weight_0', 1.0, 1.2, step = 0.1), trial.suggest_float('class_weight_1', 1.0, 1.2, step = 0.1))
    if percentile < 70:
        class_weight = (trial.suggest_float('class_weight_0', 1.0, 1.5, step = 0.1), trial.suggest_float('class_weight_1', 1.0, 1.1, step = 0.1))
    n_estimators = trial.suggest_int('n_estimators', 100, 500)
    max_depth = trial.suggest_int('max_depth', 10, 50)
    # min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    # min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    # max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])

    rf_args = {'n_estimators': n_estimators, 'max_depth': max_depth, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'random_state': 42, 'n_jobs': 10, 'class_weight': class_weight}

    # do optimize filter
    N_TOTAL_SOUNDS = data_test.shape[0]

    TC_data = data_train[data_train['xco2tccon'] > 0]
    TCCON_names = TC_data['tccon_name'].unique()
    TCCON_names = TCCON_names[TCCON_names != 'pasadena01']
    TCCON_names = TCCON_names[TCCON_names != 'xianghe01']
    TC_data = TC_data[TC_data['tccon_name'].isin(TCCON_names)]    

    # y_tc = ((TC_data.diff_density_weighting_TCCON <= np.nanpercentile(TC_data.diff_density_weighting_TCCON,percentile))) # if the throuput weighting is to be applied to TCCON training labels
    # y_tc.replace({False: 1, True: 0}, inplace=True)
    y_tc = ((np.abs(TC_data.xco2MLcorr - TC_data.xco2tccon) <= np.nanpercentile(np.abs(TC_data.xco2MLcorr - TC_data.xco2tccon),percentile)))
    y_tc.replace({False: 1, True: 0}, inplace=True)

    y_sa = ((data_train.SA_label_density_weighting <= np.nanpercentile(data_train.SA_label_density_weighting,percentile)))
    y_sa.replace({False: 1, True: 0}, inplace=True)

    if tccon_weight_bool:
        weight_TCCON(data_train,features=None)
        M_tc, _ = train_classifier(TC_data, y_tc, feats_tc_lnd, sample_weights=data_train['weights'], **rf_args)
    else:
        M_tc, _ = train_classifier(TC_data, y_tc, feats_tc_lnd, **rf_args)

    M_sa, _ = train_classifier(data_train, y_sa, feats_sa_lnd, **rf_args)

    preds_tc = M_tc.predict(data_test[feats_tc_lnd])
    preds_sa = M_sa.predict(data_test[feats_sa_lnd])

    # create tccon and small area filter bit columns
    data_test['tccon_filter'] = preds_tc
    data_test['sa_filter'] = preds_sa

    # build the ternary filter
    data_test['xco2_quality_flag_gamma'] = 2
    data_test.loc[(data_test['sa_filter'] == 0) | (data_test['tccon_filter'] == 0), 'xco2_quality_flag_gamma'] = 1
    data_test.loc[(data_test['sa_filter'] == 0) & (data_test['tccon_filter'] == 0) & (data_test['bias_correction_uncert'] <= abstention_threshold), 'xco2_quality_flag_gamma'] = 0    

    qf0 = data_test[data_test['xco2_quality_flag_gamma'] == 0]

    # TODO : finish this part and find a good way to incorporate a way to save either models or their params
    trial_results_dir = Path("optuna_trial_results") # Define a directory for trial outputs
    trial_results_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists

    if qf0.shape[0] < 100:
        print('Filter is too strict, not enough QF=0 data points, no stats recorded')
        trial_metrics = {
            'tccon_error' : np.nan,
            'throughput' : np.nan,
            'bias_uncert' : np.nan,
            'gamma' : gamma,
            'abstention_threshold' : abstention_threshold,
            'percentile' : percentile,
            'class_weight_0': class_weight[0] if class_weight else None,
            'class_weight_1': class_weight[1] if class_weight else None,
            'TCCON_weight_bool' : tccon_weight_bool,
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'trial_number': trial.number,
            'status': 'failed_strict_filter'
        }
        # Save M_tc and M_sa models even if the filter is too strict, for potential analysis
        joblib.dump(M_tc, trial_results_dir / f'M_tc_trial_{trial.number:03d}.joblib')
        joblib.dump(M_sa, trial_results_dir / f'M_sa_trial_{trial.number:03d}.joblib')
        print(f"Saved M_tc and M_sa for trial {trial.number} (strict filter) to {trial_results_dir}")

    else:
        trial_metrics = {
            'tccon_error' : get_TCCON_Error(qf0),
            'throughput' : get_throuput(qf0, N_TOTAL_SOUNDS),
            'bias_uncert' : np.nanstd(qf0.bias_correction_uncert),
            'gamma' : gamma,
            'abstention_threshold' : abstention_threshold,
            'percentile' : percentile,
            'class_weight_0': class_weight[0] if class_weight else None,
            'class_weight_1': class_weight[1] if class_weight else None,
            'TCCON_weight_bool' : tccon_weight_bool,
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'trial_number': trial.number,
            'status': 'success'
        }
        # Save the models using joblib
        joblib.dump(M_tc, trial_results_dir / f'M_tc_trial_{trial.number:03d}.joblib')
        joblib.dump(M_sa, trial_results_dir / f'M_sa_trial_{trial.number:03d}.joblib')
        print(f"Saved M_tc and M_sa for trial {trial.number} to {trial_results_dir}")

    # Save the trial parameters and scalar results to a JSON file
    # Using a more descriptive filename including trial number
    trial_json_filename = trial_results_dir / f'trial_{trial.number:03d}_params_results.json'
    with open(trial_json_filename, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        serializable_metrics = {k: (v.item() if hasattr(v, 'item') else v) for k, v in trial_metrics.items()}
        json.dump(serializable_metrics, f, indent=4)
    print(f"Saved trial parameters and results to {trial_json_filename}")

    return trial_metrics['tccon_error'], trial_metrics['throughput'], trial_metrics['bias_uncert']

def ocean_filter_trial(trial):
    """
    Define the parameters for the ocean filter
    :param trial: optuna trial object
    :return: dict, the obective scores for the ocean filter
    """
    d1 = load_data(2015, 'all', verbose_IO=True, qf=None, preload_IO=True, footprint=0, Steffen_IO=False)
    d2 = load_data(2016, 'all', verbose_IO=True, qf=None, preload_IO=True, footprint=0, Steffen_IO=False)
    d3 = load_data(2017, 'all', verbose_IO=True, qf=None, preload_IO=True, footprint=0, Steffen_IO=False)
    d4 = load_data(2018, 'all', verbose_IO=True, qf=None, preload_IO=True, footprint=0, Steffen_IO=False)
    d5 = load_data(2019, 'all', verbose_IO=True, qf=None, preload_IO=True, footprint=0, Steffen_IO=False)
    d6 = load_data(2020, 'all', verbose_IO=True, qf=None, preload_IO=True, footprint=0, Steffen_IO=False)
    d7 = load_data(2021, 'all', verbose_IO=True, qf=None, preload_IO=True, footprint=0, Steffen_IO=False)

    # data_train = pd.concat([d1, d2, d3, d4, d5, d6], ignore_index=True)
    data_train = d1
    data_test = d7

    data_train = data_train[data_train['land_fraction'] < 100]
    data_test = data_test[data_test['land_fraction'] < 100]

    # data_train = data_train.sample(n = int(5E+6), axis=0,random_state=42)
    # data_test = data_test.sample(n = int(3E+6), axis=0,random_state=42)

    # bias correct
    # bias correct
    TC_ocn = '/Users/williamkeely/Desktop/B11/current/bias_models_current/current_ocn/TCCON_V12_OcnGL_RF.p'
    SA_ocn = '/Users/williamkeely/Desktop/B11/current/bias_models_current/current_ocn/V12_prec_xco2raw_SA_OcnGL_RF.p'

    data_train = bias_correct(TC_ocn,data_train, ['xco2_raw'], uq=True,proxy_name='TCCON')
    data_train = bias_correct(SA_ocn,data_train, ['xco2_raw'], uq=True,proxy_name='SA')
    data_test = bias_correct(TC_ocn,data_test, ['xco2_raw'], uq=True,proxy_name='TCCON')
    data_test = bias_correct(SA_ocn,data_test, ['xco2_raw'], uq=True,proxy_name='SA')

    data_train['xco2MLcorr'] = data_train['xco2_raw']
    data_test['xco2MLcorr'] = data_test['xco2_raw']
    data_train['h2o_ratio_bc'] = data_train['h2o_ratio']
    data_test['h2o_ratio_bc'] = data_test['h2o_ratio']
    data_train['co2_ratio_bc'] = data_train['co2_ratio']
    data_test['co2_ratio_bc'] = data_test['co2_ratio']
    data_train['aod_diff'] = data_train['aod_total'] - data_train['aod_total_apriori']
    data_test['aod_diff'] = data_test['aod_total'] - data_test['aod_total_apriori']


    # define features ************************************************ MAKE CAHNGES HERE TO FEATURES ************************************************
    feats_tc_ocn = ['bias_correction_uncert', 'tcwv_uncertainty', 'xco2_uncertainty',
                    'co2_ratio_bc', 'albedo_sco2', 'albedo_wco2', 't700', 'h2o_ratio_bc', 
                    'zlo_o2a', 'aod_sulfate', 'aod_total', 'h2o_scale', 'water_height', 'dp_abp',
                    'rms_rel_sco2', 'snr_sco2', 'glint_angle',  'dpfrac',
                        'ice_height',  'dp_o2a', 'aod_dust', 'dust_height',
                        'aod_water', 'dp_sco2', 'dws', ]






    feats_sa_ocn = ['max_declocking_sco2', 'h_continuum_wco2', 'color_slice_noise_ratio_o2a', 'color_slice_noise_ratio_sco2', 'dp_abp', 'aod_diff',
                    'h_continuum_sco2', 'aod_total', 'rms_rel_wco2', 'h_continuum_o2a', 'bias_correction_uncert', 'max_declocking_o2a', 'water_height',
                    'color_slice_noise_ratio_wco2', 'rms_rel_sco2', 'max_declocking_wco2', 'co2_ratio_bc', 'xco2_uncertainty', 'dws', 'snr_sco2', 't700', ]
    
    # **************************************************************************************************************************************

    # PRIORS for Bayesian Optimization
    # define priors for the training label weights
    gamma = trial.suggest_loguniform('gamma', 0.001, 1000.0)
    data_train = SA_label_density_weighting(data_train, gamma = gamma)
    tccon_weight_bool = trial.suggest_categorical('tccon_weight_bool', [True, False])

    # define priors for the defining filter labels
    abstention_threshold = trial.suggest_float('abstention_threshold', 0.5, 2.0, step = 0.05)
    percentile = trial.suggest_int('percentile', 30, 90, step = 5)

    # define the prior for the RF hyperparameters
    if percentile >= 70:
        class_weight = (trial.suggest_float('class_weight_0', 1.0, 1.2, step = 0.1), trial.suggest_float('class_weight_1', 1.0, 1.2, step = 0.1))
    if percentile < 70:
        class_weight = (trial.suggest_float('class_weight_0', 1.0, 1.5, step = 0.1), trial.suggest_float('class_weight_1', 1.0, 1.1, step = 0.1))
    n_estimators = trial.suggest_int('n_estimators', 100, 500)
    max_depth = trial.suggest_int('max_depth', 10, 50)
    # min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    # min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    # max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])

    rf_args = {'n_estimators': n_estimators, 'max_depth': max_depth, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'random_state': 42, 'n_jobs': 10, 'class_weight': class_weight}

    # do optimize filter
    N_TOTAL_SOUNDS = data_test.shape[0]

    TC_data = data_train[data_train['xco2tccon'] > 0]
    TCCON_names = TC_data['tccon_name'].unique()
    TCCON_names = TCCON_names[TCCON_names != 'pasadena01']
    TCCON_names = TCCON_names[TCCON_names != 'xianghe01']
    TC_data = TC_data[TC_data['tccon_name'].isin(TCCON_names)]    

    # y_tc = ((TC_data.diff_density_weighting_TCCON <= np.nanpercentile(TC_data.diff_density_weighting_TCCON,percentile))) # if the throuput weighting is to be applied to TCCON training labels
    # y_tc.replace({False: 1, True: 0}, inplace=True)
    y_tc = ((np.abs(TC_data.xco2MLcorr - TC_data.xco2tccon) <= np.nanpercentile(np.abs(TC_data.xco2MLcorr - TC_data.xco2tccon),percentile)))
    y_tc.replace({False: 1, True: 0}, inplace=True)


    y_sa = ((data_train.SA_label_density_weighting <= np.nanpercentile(data_train.SA_label_density_weighting,percentile)))
    y_sa.replace({False: 1, True: 0}, inplace=True)

    if tccon_weight_bool:
        weight_TCCON(data_train,features=None)
        M_tc, _ = train_classifier(TC_data, y_tc, feats_tc_ocn, sample_weights=data_train['weights'], **rf_args)
    else:
        M_tc, _ = train_classifier(TC_data, y_tc, feats_tc_ocn, **rf_args)
    
    M_sa, _ = train_classifier(data_train, y_sa, feats_sa_ocn, **rf_args)

    preds_tc = M_tc.predict(data_test[feats_tc_ocn])
    preds_sa = M_sa.predict(data_test[feats_sa_ocn])

    # create tccon and small area filter bit columns
    data_test['tccon_filter'] = preds_tc
    data_test['sa_filter'] = preds_sa

    # build the ternary filter
    data_test['xco2_quality_flag_gamma'] = 2
    data_test.loc[(data_test['sa_filter'] == 0) | (data_test['tccon_filter'] == 0), 'xco2_quality_flag_gamma'] = 1
    data_test.loc[(data_test['sa_filter'] == 0) & (data_test['tccon_filter'] == 0) & (data_test['bias_correction_uncert'] <= abstention_threshold), 'xco2_quality_flag_gamma'] = 0    

    qf0 = data_test[data_test['xco2_quality_flag_gamma'] == 0]

    # TODO : finish this part and find a good way to incorporate a way to save either models or their params
    trial_results_dir = Path("optuna_trial_results_ocean") # Define a directory for trial outputs
    trial_results_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists

    if qf0.shape[0] < 100:
        print('Filter is too strict, not enough QF=0 data points, no stats recorded')
        trial_metrics = {
            'tccon_error' : np.nan,
            'throughput' : np.nan,
            'bias_uncert' : np.nan,
            'gamma' : gamma,
            'abstention_threshold' : abstention_threshold,
            'percentile' : percentile,
            'class_weight_0': class_weight[0] if class_weight else None,
            'class_weight_1': class_weight[1] if class_weight else None,
            'TCCON_weight_bool' : tccon_weight_bool,
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'trial_number': trial.number,
            'status': 'failed_strict_filter'
        }
        # Save M_tc and M_sa models even if the filter is too strict, for potential analysis
        joblib.dump(M_tc, trial_results_dir / f'M_tc_trial_{trial.number:03d}.joblib')
        joblib.dump(M_sa, trial_results_dir / f'M_sa_trial_{trial.number:03d}.joblib')
        print(f"Saved M_tc and M_sa for trial {trial.number} (strict filter) to {trial_results_dir}")
    else:
        trial_metrics = {
            'tccon_error' : get_TCCON_Error(qf0),
            'throughput' : get_throuput(qf0, N_TOTAL_SOUNDS),
            'bias_uncert' : np.nanstd(qf0.bias_correction_uncert),
            'gamma' : gamma,
            'abstention_threshold' : abstention_threshold,
            'percentile' : percentile,
            'class_weight_0': class_weight[0] if class_weight else None,
            'class_weight_1': class_weight[1] if class_weight else None,
            'TCCON_weight_bool' : tccon_weight_bool,
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'trial_number': trial.number,
            'status': 'success'
        }
        # Save the models using joblib
        joblib.dump(M_tc, trial_results_dir / f'M_tc_trial_{trial.number:03d}.joblib')
        joblib.dump(M_sa, trial_results_dir / f'M_sa_trial_{trial.number:03d}.joblib')
        print(f"Saved M_tc and M_sa for trial {trial.number} to {trial_results_dir}")

    # Save the trial parameters and scalar results to a JSON file
    trial_json_filename = trial_results_dir / f'trial_{trial.number:03d}_params_results.json'
    with open(trial_json_filename, 'w') as f:
        serializable_metrics = {k: (v.item() if hasattr(v, 'item') else v) for k, v in trial_metrics.items()}
        json.dump(serializable_metrics, f, indent=4)
    print(f"Saved trial parameters and results to {trial_json_filename}")

    return trial_metrics['tccon_error'], trial_metrics['throughput'], trial_metrics['bias_uncert']

    

# OBJECTIVE FUNCTIONS *************************************************************************
def get_TCCON_Error(data):
    data = data[data['xco2tccon'] > 0]
    data = data[data['tccon_name'] != 'pasadena01']
    data = data[data['tccon_name'] != 'xianghe01']
    data = data[data['xco2_quality_flag_gamma'] == 0]
    return np.sqrt(np.nanmean((data['xco2_raw'] - data['xco2tccon'])**2))

def get_SA_Error(data):
    data = data[data['xco2_quality_flag_gamma'] == 0]
    return np.nanstd(data.xco2raw_SA_bias)

def get_throuput(data, N_TOTAL_SOUNDS):
    data = data[data['xco2_quality_flag_gamma'] == 0]
    return data.shape[0]/N_TOTAL_SOUNDS

def b11_land_constraint_func(result, threshold = 55):
    # check if filter has similar global data availability for QF=0 and B11.1=0
    if (result['throughput'] >= threshold) & (result['tccon_error'] <= 0.93):
        return 1
    else:
        return 0  
    
def seasonal_bin_count(data, threshold = 0, res = 2):
    # Seasonal bin counts
    data = get_season(data)
    # get lat lon bin counts for each season
    fall, _ ,_ = np.histogram2d(data.loc[data['season'] == 'SON', 'longitude'], data.loc[data['season'] == 'SON', 'latitude'], bins= (360//res, 180//res))
    winter, _ ,_ = np.histogram2d(data.loc[data['season'] == 'DJF', 'longitude'], data.loc[data['season'] == 'DJF', 'latitude'], bins= (360//res, 180//res))
    spring, _ ,_ = np.histogram2d(data.loc[data['season'] == 'MAM', 'longitude'], data.loc[data['season'] == 'MAM', 'latitude'], bins= (360//res, 180//res))
    summer, _ ,_ = np.histogram2d(data.loc[data['season'] == 'JJA', 'longitude'], data.loc[data['season'] == 'JJA', 'latitude'], bins= (360//res, 180//res))
    
    # number of bins in each season that are below threshold
    n_bins = 0
    n_bins += np.sum(fall < threshold)
    n_bins += np.sum(winter < threshold)
    n_bins += np.sum(spring < threshold)
    n_bins += np.sum(summer < threshold)
    return n_bins

def seasonal_avg_bin_count(data, threshold = 0, res = 2):
    data = get_season(data)
   # get lat lon bin counts for each season
    fall, _ ,_ = np.histogram2d(data.loc[data['season'] == 'SON', 'longitude'], data.loc[data['season'] == 'SON', 'latitude'], bins= (360//res, 180//res))
    winter, _ ,_ = np.histogram2d(data.loc[data['season'] == 'DJF', 'longitude'], data.loc[data['season'] == 'DJF', 'latitude'], bins= (360//res, 180//res))
    spring, _ ,_ = np.histogram2d(data.loc[data['season'] == 'MAM', 'longitude'], data.loc[data['season'] == 'MAM', 'latitude'], bins= (360//res, 180//res))
    summer, _ ,_ = np.histogram2d(data.loc[data['season'] == 'JJA', 'longitude'], data.loc[data['season'] == 'JJA', 'latitude'], bins= (360//res, 180//res))

    # percentage of bins in each season that are below threshold
    # fall_bins = (fall.size - np.sum(fall > threshold))/fall.size
    # winter_bins = (winter.size - np.sum(winter > threshold))/winter.size
    # spring_bins = (spring.size - np.sum(spring > threshold))/spring.size
    # summer_bins = (summer.size - np.sum(summer > threshold))/summer.size
    fall_bins = np.sum(fall < threshold)/fall.size
    winter_bins = np.sum(winter < threshold)/winter.size
    spring_bins = np.sum(spring < threshold)/spring.size
    summer_bins = np.sum(summer < threshold)/summer.size

    # calculate the average percentage
    avg = np.mean([fall_bins, winter_bins, spring_bins, summer_bins]) * 100
    return avg

# plotting functions
def plot_best_trials(study):
    all_trials = study.trials
    best = study.best_trials


    # get params, and values for all_trials and put in DataFrame
    throughput = []
    rmse = []
    bias_uncert = []
    study_id = []

    for i in range(len(all_trials)):
        throughput.append(all_trials[i].values[0])
        rmse.append(all_trials[i].values[1])
        bias_uncert.append(all_trials[i].values[2])
        study_id.append(all_trials[i].number)

    # get params, values for best trials and put in DataFrame
    throughput_best = []
    rmse_best = []
    bias_uncert_best = []
    study_id_best = []

    for i in range(len(best)):
        throughput_best.append(best[i].values[0])
        rmse_best.append(best[i].values[1])
        bias_uncert_best.append(best[i].values[2])
        study_id_best.append(best[i].number)


    # create dataframe for all trials
    df_all = pd.DataFrame({'throughput': throughput, 'rmse': rmse, 'bias_uncert': bias_uncert, 'trial_id': study_id})
    # add 1 to each value in study_id
    # df_all['study_id'] = df_all['study_id'] + 1


    # add column for best trials
    df_all['best'] = 0
    df_all.loc[df_all['trial_id'].isin(study_id_best), 'best'] = 1

    df_all.head()

    # filts df to only best trials
    df_best = df_all[df_all['best'] == 1]



    # plot pareto front
    fig = df_best.plot.scatter(x='throughput', y='rmse', c='trial_id', colormap='nipy_spectral', colorbar=True, title='Pareto Front', xlabel='Throughput [%]', ylabel='RMSE [ppm]', figsize=(10,6)).get_figure()
    fig.savefig('pareto_front_throughput_tcconrmse.png')
    fig = df_best.plot.scatter(x='throughput', y='bias_uncert', c='trial_id', colormap='nipy_spectral', colorbar=True, title='Pareto Front', xlabel='Throughput [%]', ylabel='Bias Uncert. [ppm]', figsize=(10,6)).get_figure()
    fig.savefig('pareto_front_throughput_biasuncert.png')
    fig = df_best.plot.scatter(x='rmse', y='bias_uncert', c='trial_id', colormap='nipy_spectral', colorbar=True, title='Pareto Front', xlabel='RMSE [ppm]', ylabel='Bias Uncert. [ppm]', figsize=(10,6)).get_figure()
    fig.savefig('pareto_front_rmse_biasuncert.png')

# **************ARG PARSER****************
def parse_args():
    parser = argparse.ArgumentParser(description='Optimize filter')
    parser.add_argument('--land', action=argparse.BooleanOptionalAction,default = True,  help='Optimize land filter')
    parser.add_argument('--ocean', action=argparse.BooleanOptionalAction, default=True, help='Optimize ocean filter')
    parser.add_argument('--n_trials', type=int, default=10, help='Number of trials')
    parser.add_argument('--viz', action=argparse.BooleanOptionalAction, default=True, help='Visualize the results')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.land == True:
        # optimize land filter
        print('Optimizing land filter ...')
        study = optuna.create_study(directions=['minimize','maximize','minimize',], sampler=optuna.samplers.MOTPESampler(consider_prior = True, seed=10), pruner=optuna.pruners.HyperbandPruner())
        study.optimize(land_filter_trial,n_trials=args.n_trials)
        # print('Best params for land filter')
        # print(study.best_trials.params)
        # save the params
        if args.viz == True:
            plot_best_trials(study)
    if args.ocean == True:
        # optimize ocean filter
        print('Optimizing ocean filter ...')
        study = optuna.create_study(directions=['minimize','maximize','minimize',], sampler=optuna.samplers.MOTPESampler(consider_prior = True, seed=10), pruner=optuna.pruners.HyperbandPruner())
        study.optimize(ocean_filter_trial, n_trials=args.n_trials)
        # print('Best params for ocean filter')
        # print(study.best_trials.params)
        # save the params
        if args.viz == True:
            plot_best_trials(study)


if __name__ == '__main__':
    main()



