# Steffen Mauceri

# correct biases in XCO2 retrieval with ML
# takes in a list of state vector elements from XCO2 retrievals that are informative of XCO2 biases
# Grows a random forest to predict biases in XCO2

import os
import pandas as pd
import numpy as np
import json
import joblib
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF
import xgboost as xgb
from pathlib import Path

import paths
from util import load_data, \
    get_variability_reduction, get_importance,plot_tccon, plot_map, make_prediction, feature_selector, \
    get_RMSE, filter_TCCON, calc_SA_bias, bias_correct, calc_SA_bias_clean, load_and_concat_years

# Set random seeds for reproducibility
np.random.seed(42)

def precorrect(path, data):
    ''' correct data with an existing model
    :param path: str, path to model
    :param data: pd Dataframe, data to be corrected
    :return: pd DataFrame, corrected data
    '''
    # make a copy of the orignial raw xco2 for later plots and analysis
    data.loc[:,'xco2_raw_orig'] = data.loc[:,'xco2_raw'].copy()
    data.loc[:, 'xco2raw_SA_bias_orig'] = data.loc[:, 'xco2raw_SA_bias'].copy()
    #perfrom bias correction
    data = bias_correct(path, data, ['xco2_raw'])
    #update bias to TCCON
    data.loc[:, 'xco2_TCCON_bias'] = data['xco2_raw'] - data['xco2tccon']
    # recalculate small area bias
    data.sort_values('SA', inplace=True)
    XCO2 = data['xco2_raw'].to_numpy()
    SA = data['SA'].to_numpy()

    data.loc[:,'xco2raw_SA_bias'] = calc_SA_bias_clean(XCO2, SA, data['strong_emitter'])

    print('soundings total: ' + str(len(data)))
    # data = data.loc[~pd.isna(data['xco2raw_SA_bias']),:]  # Commenting out NaN removal
    print('soundings remaining after SA calc: ' + str(len(data)))
    return data


def main(name, model, save_fig, feature_select, mode, var_tp, qf, max_n, max_depth, min_weight_fraction_leaf, prec_model_path = None, verbose_IO=False):
    ''' Main function to train a model and make some preliminary plots
    :param name: str, name of model
    :param model: str, kind of model ['RF', 'Ridge', 'GPR']
    :param save_fig: bool,
    :param feature_select:
    :param mode: str, ['LndND', 'SeaGL', 'LndGL']
    :param var_tp: str, variable to predict
    :param qf: int, quality flag, use 'None' to not filter based on QF
    :param max_n: int, maximum number of samples to work with (to save RAM)
    :param max_depth: int, max depth of RF
    :param min_weight_fraction_leaf: float, min weight fraction leaf for RF
    :param prec_model_path: Path or None, path to model to precorrect with
    :param verbose_IO: bool, extra output
    :return:
    '''
    # get features
    features, feature_n = feature_selector(feature_select)

    name = name + var_tp  + mode + '_' +  feature_n + model
    print(name)
    # make folder for model output
    model_output_dir = paths.MODEL_SAVE_DIR / name

    print(model_output_dir.name)
    # make dir if it does not exist
    paths.ensure_dir_exists(model_output_dir)


    # load data
    if var_tp == 'xco2_TCCON_bias':
        TCCON = True
    else:
        TCCON = False
    preload_IO = True
    RIW = False

    hold_out_year = 2022

    data_train = load_and_concat_years(2015, 2024, hold_out_year=hold_out_year, mode=mode, qf=qf, preload_IO=preload_IO, TCCON=TCCON, remove_inland_water=RIW)
    data_test = load_data(hold_out_year, mode, verbose_IO=verbose_IO, qf=qf, preload_IO=preload_IO, TCCON=TCCON, remove_inland_water=RIW)

    # get TCCON station names
    TCCON_names = np.unique(data_train['tccon_name'].to_numpy())


    # perform train test split on station names
    # TCCON_names_train, TCCON_names_test = train_test_split(TCCON_names, test_size=0.4, random_state=1)
    TCCON_names_train = TCCON_names
    TCCON_names_test = TCCON_names


    if var_tp == 'xco2_TCCON_bias':

        # only do a spatial split
        data_test = data_test[data_test['xco2tccon'] > 0]
        data_train = data_train[data_train['xco2tccon'] > 0]

        data_train.loc[:, var_tp] = data_train['xco2_raw'] - data_train['xco2tccon']
        data_test.loc[:, var_tp] = data_test['xco2_raw'] - data_test['xco2tccon']

        # get testing TCCON stations
        data_test = filter_TCCON(data_test, TCCON_names_test)
        # get training TCCON stations
        data_train = filter_TCCON(data_train, TCCON_names_train)

    else:
        # if we are working with the small area bias, we remove any soundings that have a TCCON match up. 
        data_train = data_train[pd.isna(data_train['xco2tccon'])]


    print(TCCON_names_test)

    print(str(len(data_train)/1000) + 'k samples loaded')
    # reduce size of train-set to max_n samples
    if len(data_train) > max_n:
        data_train = data_train.sample(max_n, random_state=1)
    if len(data_test) > max_n:
        data_test = data_test.sample(max_n, random_state=1)
    print(str(len(data_train) / 1000) + 'k samples downsampled')

    # precorrect_IO is now checked based on whether prec_model_path is provided
    # if precorrect_IO:
    if prec_model_path is not None:
        print(f'Correcting data with existing model: {prec_model_path}')
        # correct data with an existing model
        precorrect_model_load_path = prec_model_path # Use the passed Path object directly
 
        if not precorrect_model_load_path.exists() or not precorrect_model_load_path.is_dir():
            raise FileNotFoundError(f"Precorrect model directory {precorrect_model_load_path} not found.")
        else:
            data_train = precorrect(precorrect_model_load_path, data_train)
            data_test = precorrect(precorrect_model_load_path, data_test)

    if var_tp in features:
        features.remove(var_tp)


    # subsample data to have a more balanced data set for GPR, NN, Ridge
    if model != 'RF':
        if len(data_train) > 10 ** 4:
            data_train = data_train.sample(10 ** 4, random_state=1)#, weights=data_train['weights'])

    # Remove NaN values only for training
    train_mask = ~pd.isna(data_train[var_tp])
    X_train, y_train = data_train[features][train_mask], data_train[var_tp][train_mask]
    
    # Keep NaN values in test set for analysis
    X_test, y_test = data_test[features], data_test[var_tp]

    # calc mean std of data (using only non-NaN values)
    X_mean = X_train.mean()
    y_mean = y_train.mean()
    X_std = X_train.std()
    y_std = y_train.std()

    if model == 'RF':

        # set up Random Forest parameters
        M = RandomForestRegressor(n_estimators=100,
                                  max_depth=max_depth,
                                  max_samples=0.5,
                                  min_weight_fraction_leaf=min_weight_fraction_leaf,
                                  max_features=0.5,
                                  min_impurity_decrease=0.0,
                                  bootstrap=True,
                                  oob_score=False,
                                  n_jobs=16,
                                  random_state=42,
                                  verbose=0,
                                  warm_start=False)


    if model == 'GPR':

        if len(X_train) > 10**4:
            X_train = X_train.sample(10**4, random_state=1, weights=data_train['weights'])
            y_train = y_train.sample(10**4, random_state=1, weights = data_train['weights'])

        # normalize data to 0 mean unit standard deviation
        X_train = (X_train - X_mean) / X_std
        X_test = (X_test - X_mean) / X_std
        y_train = (y_train - y_mean) / y_std
        y_test = (y_test - y_mean) / y_std

        # set up GPR parameters
        kernel = 1.0 * RBF(length_scale=1, length_scale_bounds=(0.5,5)) + WhiteKernel(noise_level=0.2, noise_level_bounds=(0.1, 1))
        M = GaussianProcessRegressor(kernel=kernel, alpha=0.8, n_restarts_optimizer=5)

    if model == 'NN':

        # normalize data to 0 mean unit standard deviation
        X_train = (X_train - X_mean) / X_std
        X_test = (X_test - X_mean) / X_std
        y_train = (y_train - y_mean) / y_std
        y_test = (y_test - y_mean) / y_std

        M = MLPRegressor(hidden_layer_sizes=(64,2),
                         max_iter=1000,
                         verbose=True,
                         learning_rate_init=0.0005,
                         alpha = 0.01,
                         random_state=42)

    if model == 'Ridge':
        # normalize data to 0 mean unit standard deviation
        X_train = (X_train - X_mean) / X_std
        X_test = (X_test - X_mean) / X_std
        y_train = (y_train - y_mean) / y_std
        y_test = (y_test - y_mean) / y_std

        # set Ridge parameters
        M = Ridge(alpha=10**-4, random_state=42)

    if model == 'BayesianRidge':
        # normalize data to 0 mean unit standard deviation
        X_train = (X_train - X_mean) / X_std
        X_test = (X_test - X_mean) / X_std
        y_train = (y_train - y_mean) / y_std
        y_test = (y_test - y_mean) / y_std

        # set Ridge parameters
        M = BayesianRidge(alpha_1=10**-4, alpha_2=10**-4, lambda_1=10**-4, lambda_2=10**-4)
        
    if model == 'XGB':
        X_train = (X_train - X_mean) / X_std
        X_test = (X_test - X_mean) / X_std
        y_train = (y_train - y_mean) / y_std
        y_test = (y_test - y_mean) / y_std
        
        n_jobs = 16 # change based on local machine reqs.
        
        # B11 parameters
        if n_jobs is not None:
            params = {'learning_rate': 0.018807, 'n_estimators': 100, 'max_depth': 15, 'min_child_weight': 22, 'seed': 0, 'num_feature' : len(features),
                'subsample': 0.6768, 'gamma': 7.65, 'reg_alpha': 0, 'reg_lambda': 1, 'tree_method' : 'auto', 'n_jobs' : n_jobs}
        else:
            params = {'learning_rate': 0.018807, 'n_estimators': 100, 'max_depth': 15, 'min_child_weight': 22, 'seed': 0, 'num_feature' : len(features),
                'subsample': 0.6768, 'gamma': 7.65, 'reg_alpha': 0, 'reg_lambda': 1, 'tree_method' : 'auto'}
            
        M = xgb.XGBRegressor(**params)

    # train model
    print('fitting model')
    M.fit(X_train, y_train)

    if model == 'GPR':
        print(M.kernel_)

    # Save the trained model M using joblib
    model_save_filepath = model_output_dir / 'trained_model.joblib'
    joblib.dump(M, model_save_filepath)
    print(f"Saved trained model to: {model_save_filepath}")

    # Save feature importance if available (example for RF)
    if hasattr(M, 'feature_importances_') and model == 'RF':
        feature_imp = pd.Series(M.feature_importances_, index=features).sort_values(ascending=False)
        feature_imp_filepath = model_output_dir / 'feature_importance.csv'
        feature_imp.to_csv(feature_imp_filepath)
        print(f"Saved feature importance to: {feature_imp_filepath}")

    # Always save X_mean, y_mean, X_std, y_std, features, and model_type
    norm_params = {
        'X_mean': X_mean.to_dict(),
        'X_std': X_std.to_dict(),
        'y_mean': float(y_mean),
        'y_std': float(y_std),
        'features': features,
        'model_type': model,
        'mode': mode,
        'qf': qf
    }
    norm_params_filepath = model_output_dir / 'normalization_params.json'
    with open(norm_params_filepath, 'w') as f:
        json.dump(norm_params, f, indent=4)
    print(f"Saved normalization parameters to: {norm_params_filepath}")

    # calculate R^2. If train and test are very different try reducing the depth of the random forest
    print('calculate R^2')
    R2_train = M.score(X_train, y_train); print('R2 Train:',R2_train)
    # Calculate R2 only on non-NaN test values
    test_mask = ~pd.isna(y_test)
    R2_test = M.score(X_test[test_mask], y_test[test_mask]); print('R2 Test :',R2_test)

    bias_test, bias_std_test = make_prediction(M, X_test, model,UQ=False,X_train=X_train, y_mean=y_mean, y_std=y_std)
    bias_train, bias_std_train = make_prediction(M, X_train, model, UQ=False, X_train=X_train, y_mean=y_mean, y_std=y_std)

    data_test.loc[:,'xco2MLcorr'] = data_test.loc[:,'xco2_raw'] - bias_test
    data_test.loc[:,'xco2MLbias'] = bias_test

    data_train.loc[train_mask, 'xco2MLcorr'] = data_train.loc[train_mask,'xco2_raw'] - bias_train


    # ****************************************************************
    # make plots
    if prec_model_path is not None:
        data_test['xco2raw_SA_bias'] = data_test['xco2raw_SA_bias_orig']

    if model == 'RF':
        get_importance(M, X_test, name, model_output_dir, save_IO=save_fig)

    # make plots and calculate performance for each quality flag
    data_test_all = data_test
    name_all = name
    for qf in [0, 1, None]:
        print('making plots for QF=' + str(qf))
        if qf is not None:
            data_test = data_test_all.loc[data_test_all['xco2_quality_flag'] == qf]
        else:
            data_test = data_test_all

        name = name_all + '_QF' + str(qf)

        # recalculate small area bias
        data_test.sort_values('SA', inplace=True)
        XCO2 = data_test.loc[:, 'xco2MLcorr'].to_numpy()
        SA = data_test.loc[:, 'SA'].to_numpy()
        data_test.loc[:, 'xco2raw_SA_bias-ML'] = calc_SA_bias(XCO2, SA)

        ## compare to TCCON
        xco2ML_std, xco2ML_median, xco2B11_std, xco2B11_median, xco2raw_std, xco2raw_median, xco2ML_RMSE, xco2B11_RMSE,xco2raw_RMSE = plot_tccon(data_test, TCCON_names_test, save_fig=save_fig, path=model_output_dir, name=name, qf=qf, precorrect_IO=prec_model_path is not None)

        # plot variability reduction
        get_variability_reduction(data_test, var_tp, name, path=model_output_dir, save_fig=save_fig, qf=qf)


        # calculate RMSE before and after bias correction
        print('get RMSE reduction')
        RMSE_Raw = get_RMSE(data_test['xco2raw_SA_bias'].to_numpy())
        RMSE_ML = get_RMSE(data_test['xco2raw_SA_bias-ML'].to_numpy(), ignore_nan=True)
        RMSE_B11 = get_RMSE(data_test['xco2_SA_bias'].to_numpy())

        # write results out to file
        d = {'RMSE_Raw_SA': RMSE_Raw,
         'RMSE_ML_SA': RMSE_ML,
         'RMSE_B11_SA': RMSE_B11,
             'RMSE_Raw_TC': xco2raw_RMSE,
             'RMSE_ML_TC': xco2ML_RMSE,
             'RMSE_B11_TC': xco2B11_RMSE,
            'xco2raw_std': xco2raw_std,
             'xco2raw_median': xco2raw_median,
             'xco2ML_std': xco2ML_std,
             'xco2ML_median': xco2ML_median,
             'xco2B11_std': xco2B11_std,
             'xco2B11_median': xco2B11_median,
            'R2_train': R2_train,
            'R2_test': R2_test,
             'RF_depth': max_depth
             }
        df = pd.DataFrame(data=d, index=[0])
        df.to_csv(model_output_dir / (name + '_error.txt'), index=False, float_format='%.4f')


        ## plot maps
        # plot ML bias
        plot_map(data_test,['xco2MLbias'] ,save_fig=save_fig, path=model_output_dir, name=name)

        # plot difference to B11
        data_test.loc[:,'xco2MLcorr-B11'] = data_test['xco2MLcorr'] - data_test['xco2']
        plot_map(data_test, ['xco2MLcorr-B11'], save_fig=save_fig, path=model_output_dir, name=name + '_diff_B11', pos_neg_IO=True)

        if prec_model_path is not None:
            # plot difference to original raw XCO2 before precorrection
            data_test.loc[:,'xco2MLcorr-raw_xco2'] = data_test['xco2MLcorr'] - data_test['xco2_raw_orig']
            plot_map(data_test, ['xco2MLcorr-raw_xco2'], save_fig=save_fig, path=model_output_dir, name=name + '_diff_orig_rawXCO2', pos_neg_IO=True)

            # visualize bias correction of precorrect step
            # plot difference to original raw XCO2 before precorrection
            data_test.loc[:,'precor-raw_xco2'] = data_test['xco2_raw'] - data_test['xco2_raw_orig']
            plot_map(data_test, ['precor-raw_xco2'], save_fig=save_fig, path=model_output_dir, name=name + '_diff_precor', pos_neg_IO=True)

        # compare to TCCON
        data_test['xco2-TCCON'] = data_test['xco2MLcorr'] - data_test['xco2tccon']
        print('Visualize TCCON bias vs Latitude ...')
        tccon_median = data_test.groupby('tccon_name').median()
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 7))
        for var in ['latitude', 'airmass']:
            plt.scatter(tccon_median[var], tccon_median['xco2-TCCON'], marker='x')
            for i, txt in enumerate(tccon_median.index):
                plt.annotate(txt, (tccon_median[var][i], tccon_median['xco2-TCCON'][i] + 0.005))
            plt.ylabel('OCO-2 bias by TCCON Station')
            plt.xlabel(var)
            h_max = np.nanmax(tccon_median[var])
            h_min = np.nanmin(tccon_median[var])
            plt.hlines(0, h_min, h_max, colors='k', linestyles='dashed')
            plt.title('OCO-2 - TCCON [ppm] [test]')
            plt.tight_layout()
            if not save_fig:
                plt.show()
            plt.close()





# make changes #############################
name = 'V3_11.2_2.6_prec_' # name of generated model
save_fig = True         # save figures to hard drive
qf = None               # what quality flag data: 0: best quality; 1:lesser quality; none: all data
max_n = 2*10**7          # max number of samples for training
precorrect_IO = True    # precorrect data with another model (defined in main program)
min_weight_fraction_leaf = 0.0005 # minimum weight fraction leaf for RF
max_depth = 15          # max depth of RF
prec_model_path = None # Initialize path variable

for model in ['RF']: #'RF', 'Ridge', 'Ransac', 'XGB' What model to use
    for var_tp in ['xco2raw_SA_bias']:#['xco2_TCCON_bias', 'xco2raw_SA_bias']:
        for mode in ['LndNDGL']: #'all', 'LndGL', 'LndND', 'LndNDGL', 'SeaGL'
            # stop make changes ###########################
            if mode == 'LndNDGL':
                if var_tp == 'xco2_TCCON_bias':
                    feature_select = 'TCCON_bias_Lnd'
                    max_depth = 20
                else:
                    feature_select = 'SA_bias_LndNDGL'
                    max_depth = 20
                    min_weight_fraction_leaf = 0.0001

                    prec_model_path = paths.TC_LND_CORR_MODEL
            if mode == 'LndND':
                if var_tp == 'xco2_TCCON_bias':
                    feature_select = 'TCCON_bias_Lnd'
                else:
                    feature_select = 'SA_bias_LndND'
            elif mode == 'LndGL':
                if var_tp == 'xco2_TCCON_bias':
                    feature_select = 'TCCON_bias_Lnd'
                else:
                    feature_select = 'SA_bias_LndGL'
            elif mode == 'SeaGL':
                if var_tp == 'xco2_TCCON_bias':
                    feature_select = 'TCCON_bias_Sea'
                    max_depth = 10
                else:
                    feature_select = 'SA_bias_Sea'
                    max_depth = 20
                    min_weight_fraction_leaf = 0.0001

                    prec_model_path = paths.TC_OCN_CORR_MODEL

            # Only pass the path if precorrection is enabled and path is set
            current_prec_model_path = prec_model_path if precorrect_IO else None
            main(name, model,save_fig,feature_select,mode,var_tp,qf,max_n,max_depth, min_weight_fraction_leaf, prec_model_path=current_prec_model_path)

print('Done >>>')
