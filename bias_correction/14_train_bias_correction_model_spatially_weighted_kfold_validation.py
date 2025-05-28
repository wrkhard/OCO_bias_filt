# Steffen Mauceri
#
# correct biases in XCO2 retrieval with ML
# takes in a list of state vector elements from XCO2 retrievals that are informative of XCO2 biases
# Grows a random forest to predict biases in XCO2
# similar to ML_2.6 but performs K-fold cross validation over the TCCON stations
# added weighting based on numbers of soundings per TCCON site


import os
import pandas as pd
import numpy as np

import json
import joblib
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge, RANSACRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF
import xgboost as xgb
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import KFold

import paths # Added import paths
from util import load_data, scatter_density, scatter_3D, train_test_split_Time, \
    get_variability_reduction, plot_tccon, plot_map, make_prediction, normalize_per_SA, \
    feature_selector, \
    get_RMSE, filter_TCCON, calc_SA_bias, bias_correct, weight_TCCON, load_and_concat_years


def precorrect(path, data):
    ''' correct data with an existing model
    :param path: str, path to model
    :param data: pd Dataframe, data to be corrected
    :return: pd DataFrame, corrected data
    '''
    # make a copy of the orignial raw xco2 for later plots and analysis
    data.loc[:,'xco2_raw_orig'] = data.loc[:,'xco2_raw'].copy()
    #perfrom bias correction
    data = bias_correct(path, data, ['xco2_raw'])
    data['xco2MLcorr'] = data['xco2_raw']
    # recalculate small area bias
    data.sort_values('SA', inplace=True)
    XCO2 = data['xco2_raw'].to_numpy()
    SA = data['SA'].to_numpy()
    data.loc[:,'xco2raw_SA_bias'] = calc_SA_bias(XCO2, SA)

    return data

def main(name, save_fig, mode, qf, max_n,  footprint, precorrect_IO):
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
    :param footprint: int, which footprint to work on, use 0 to work on all simultaneously
    :param verbose_IO: bool, extra output
    :return:
    '''

    # params for FIRST model
    max_depth = 20  #20 LndNDGL, 10 SeaGL
    min_weight_fraction_leaf = 0.005 # 0.0005 LndNDGL, 0.005 SeaGL
    var_tp = 'xco2_TCCON_bias'
    feature_select = 'TCCON_bias_all'
    model = 'RF'
    # ********** Train first Model *********************************


    name = name + var_tp + mode + '_' + model
    print(name)
    # make folder for model output
    model_dir = paths.MODEL_SAVE_DIR / name
    
    print(model_dir.name)
    # make dir if it does not exist
    paths.ensure_dir_exists(model_dir)

    # get TCCON station names
    preload_IO = True
    var_tp = 'xco2_TCCON_bias'

    data = load_data(2022, mode, qf=qf, preload_IO=preload_IO, footprint=footprint, Steffen_IO=Steffen_IO, TCCON = True)
    TCCON_names = np.unique(data['tccon_name'].to_numpy())

    if mode == 'SeaGL':
        TCCON_keep = ['burgos', 'darwin', 'nyalesund', 'eureka', 'izana', 'lauder', 'reunion', 'rikubetsu',
                      'saga', 'tsukuba', 'wollongong']
        TCCON_names = TCCON_names[np.isin(TCCON_names, TCCON_keep)]


    # perform train test split on station names
    data_test_folds = []
    kf = KFold(n_splits=6, shuffle=True, random_state=1)
    for train_indx, test_indx in kf.split(TCCON_names):

        var_tp = 'xco2_TCCON_bias'
        if var_tp == 'xco2_TCCON_bias':
            TCCON = True
        else:
            TCCON = False

        TCCON_names_test = TCCON_names[test_indx]
        TCCON_names_train = TCCON_names[train_indx]
        # save TCCON test stations
        # write results out to file
        d = {'TCCON_test': TCCON_names_test}
        df = pd.DataFrame(data=d)
        df.to_csv(model_dir / (name + '_TC_names.txt'), index=False)

        # get features
        features, feature_n = feature_selector(feature_select)
        
        hold_out_year = 2022
        data_train = load_and_concat_years(2015, 2023, mode=mode, qf=qf, preload_IO=preload_IO, TCCON=TCCON, hold_out_year=hold_out_year)
        data_test = load_data(hold_out_year, mode, qf=qf, preload_IO=preload_IO, footprint=footprint, TCCON = TCCON)


        data_train = data_train[data_train['xco2tccon'] > 0]

        data_train.loc[:, var_tp] = data_train['xco2_raw'] - data_train['xco2tccon']
        data_test.loc[:, var_tp] = data_test['xco2_raw'] - data_test['xco2tccon']

        # get testing TCCON stations
        data_test = filter_TCCON(data_test, TCCON_names_test)
        # get training TCCON stations
        data_train = filter_TCCON(data_train, TCCON_names_train)

        # # remove TCCON stations where we don't have enough measurements
        # for t_name in TCCON_names:
        #     if data_train.loc[data_train['tccon_name'] == t_name, 'xco2tccon'].count() <= 50:
        #         print('removing ' + t_name)
        #         TCCON_names = TCCON_names[TCCON_names != t_name]

        # calculate weights for each TCCON station based on number of samples and add to data_train
        data_train = weight_TCCON(data_train, features)

        print(TCCON_names_test)

        print(str(len(data_train) / 1000) + 'k samples loaded')
        # reduce size of train-set to max_n samples
        if len(data_train) > max_n:
            data_train = data_train.sample(max_n, random_state=1)
        if len(data_test) > max_n:
            data_test = data_test.sample(max_n, random_state=1)
        print(str(len(data_train) / 1000) + 'k samples downsampled')

        if var_tp in features:
            features.remove(var_tp)

        X_train, y_train = data_train[features], data_train[var_tp]
        X_test, y_test = data_test[features], data_test[var_tp]
        weights_train = data_train['weights']

        # calc mean std of data
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
                                      random_state=None,
                                      verbose=0,
                                      warm_start=False)

        if model == 'GPR':
            # normalize data to 0 mean unit standard deviation
            X_train = (X_train - X_mean) / X_std
            X_test = (X_test - X_mean) / X_std
            y_train = (y_train - y_mean) / y_std
            y_test = (y_test - y_mean) / y_std

            # set up GPR parameters
            kernel = 1.0 * RBF(length_scale=1,
                               length_scale_bounds=(0.5, 1))  # + WhiteKernel(noise_level=0.1, noise_level_bounds=(0.1, 1))
            M = GaussianProcessRegressor(kernel=kernel, alpha=0.8, n_restarts_optimizer=5)

        if model == 'NN':
            # normalize data to 0 mean unit standard deviation
            X_train = (X_train - X_mean) / X_std
            X_test = (X_test - X_mean) / X_std
            y_train = (y_train - y_mean) / y_std
            y_test = (y_test - y_mean) / y_std

            # set up GPR parameters
            M = MLPRegressor(hidden_layer_sizes=(128, 2), max_iter=100, verbose=True)

        if model == 'Ridge':
            # normalize data to 0 mean unit standard deviation
            X_train = (X_train - X_mean) / X_std
            X_test = (X_test - X_mean) / X_std
            y_train = (y_train - y_mean) / y_std
            y_test = (y_test - y_mean) / y_std

            # set Ridge parameters
            M = Ridge(alpha=10 ** -4)

        if model == 'Ransac':
            # normalize data to 0 mean unit standard deviation
            X_train = (X_train - X_mean) / X_std
            X_test = (X_test - X_mean) / X_std
            y_train = (y_train - y_mean) / y_std
            y_test = (y_test - y_mean) / y_std

            # set Ridge parameters
            M = RANSACRegressor(min_samples=0.2, loss='squared_error')

        if model == 'XGB':
            X_train = (X_train - X_mean) / X_std
            X_test = (X_test - X_mean) / X_std
            y_train = (y_train - y_mean) / y_std
            y_test = (y_test - y_mean) / y_std

            n_jobs = 16  # change based on local machine reqs.

            # B10 parameters
            if n_jobs is not None:
                params = {'learning_rate': 0.018807, 'n_estimators': 1000, 'max_depth': 11, 'min_child_weight': 22,
                          'seed': 0, 'num_feature': len(features),
                          'subsample': 0.6768, 'gamma': 7.65, 'reg_alpha': 0, 'reg_lambda': 1, 'tree_method': 'auto',
                          'n_jobs': n_jobs}
            else:
                params = {'learning_rate': 0.018807, 'n_estimators': 1000, 'max_depth': 11, 'min_child_weight': 22,
                          'seed': 0, 'num_feature': len(features),
                          'subsample': 0.6768, 'gamma': 7.65, 'reg_alpha': 0, 'reg_lambda': 1, 'tree_method': 'auto'}

            M = xgb.XGBRegressor(**params)

        # train model
        print('fitting model')
        M.fit(X_train, y_train, sample_weight=weights_train)

        if model == 'GPR':
            print(M.kernel_)

        # save model
        model_save_filepath = model_dir / 'trained_model.joblib'
        joblib.dump(M, model_save_filepath)
        print(f"Saved first trained model to: {model_save_filepath}")

        norm_params = {
            'X_mean': X_mean.to_dict(),
            'X_std': X_std.to_dict(),
            'y_mean': float(y_mean),
            'y_std': float(y_std),
            'features': features,
            'model_type': model, # 'model' here is the model type string like 'RF'
            'mode': mode,      # 'mode' here is the data mode like 'LndNDGL'
            'qf': qf,
            'footprint': footprint
        }
        norm_params_filepath = model_dir / 'normalization_params.json'
        with open(norm_params_filepath, 'w') as f:
            json.dump(norm_params, f, indent=4)
        print(f"Saved first model normalization parameters to: {norm_params_filepath}")

        # calculate R^2. If train and test are very different try reducing the depth of the random forest
        print('calculate R^2 for 1st model')
        R2_train = M.score(X_train, y_train);
        print('R2 Train:', R2_train)
        R2_test = M.score(X_test, y_test);
        print('R2 Test :', R2_test)


        if precorrect_IO:
            # SECOND MODEL
            hold_out_year = 2022
            data_train = load_and_concat_years(2015, 2022, mode=mode, qf=qf, preload_IO=preload_IO, TCCON=TCCON, hold_out_year=hold_out_year)
            data_test = load_data(hold_out_year, mode, qf=qf, preload_IO=preload_IO, footprint=footprint, TCCON=TCCON)

            # remove soundings with TCCON matches for train set to balance data
            data_train = data_train[pd.isna(data_train['xco2tccon'])]

            print('correcting data with 1st model')
            # correct data with an existing model
            data_train = precorrect(model_dir, data_train)
            data_test = precorrect(model_dir, data_test)

            # *********** Train second Model *********************************
            # params for second model
            max_depth = 20
            var_tp = 'xco2raw_SA_bias'
            feature_select = 'SA_bias_all'
            model = 'RF'
            # get features
            features, feature_n = feature_selector(feature_select)
            # remove NaN in var_tp
            data_train = data_train.dropna(subset=[var_tp])
            data_test = data_test.dropna(subset=[var_tp])

            X_train, y_train = data_train[features], data_train[var_tp]
            X_test, y_test = data_test[features], data_test[var_tp]

            # calc mean std of data
            X_mean = X_train.mean()
            y_mean = y_train.mean()
            X_std = X_train.std()
            y_std = y_train.std()

            if model == 'RF':
                # set up Random Forest parameters
                M = RandomForestRegressor(n_estimators=100,
                                          max_depth=max_depth,
                                          max_samples=0.5,
                                          min_weight_fraction_leaf=0.0001,
                                          max_features=0.5,
                                          min_impurity_decrease=0.0,
                                          bootstrap=True,
                                          oob_score=False,
                                          n_jobs=16,
                                          random_state=None,
                                          verbose=0,
                                          warm_start=False)

            if model == 'GPR':
                # normalize data to 0 mean unit standard deviation
                X_train = (X_train - X_mean) / X_std
                X_test = (X_test - X_mean) / X_std
                y_train = (y_train - y_mean) / y_std
                y_test = (y_test - y_mean) / y_std

                # set up GPR parameters
                kernel = 1.0 * RBF(length_scale=1,
                                   length_scale_bounds=(0.5, 1))  # + WhiteKernel(noise_level=0.1, noise_level_bounds=(0.1, 1))
                M = GaussianProcessRegressor(kernel=kernel, alpha=0.8, n_restarts_optimizer=5)

            if model == 'NN':
                # normalize data to 0 mean unit standard deviation
                X_train = (X_train - X_mean) / X_std
                X_test = (X_test - X_mean) / X_std
                y_train = (y_train - y_mean) / y_std
                y_test = (y_test - y_mean) / y_std

                # set up GPR parameters
                M = MLPRegressor(hidden_layer_sizes=(128, 2), max_iter=100, verbose=True)

            if model == 'Ridge':
                # normalize data to 0 mean unit standard deviation
                X_train = (X_train - X_mean) / X_std
                X_test = (X_test - X_mean) / X_std
                y_train = (y_train - y_mean) / y_std
                y_test = (y_test - y_mean) / y_std

                # set Ridge parameters
                M = Ridge(alpha=10 ** -4)

            if model == 'Ransac':
                # normalize data to 0 mean unit standard deviation
                X_train = (X_train - X_mean) / X_std
                X_test = (X_test - X_mean) / X_std
                y_train = (y_train - y_mean) / y_std
                y_test = (y_test - y_mean) / y_std

                # set Ridge parameters
                M = RANSACRegressor(min_samples=0.5, loss='squared_error')

            if model == 'XGB':
                X_train = (X_train - X_mean) / X_std
                X_test = (X_test - X_mean) / X_std
                y_train = (y_train - y_mean) / y_std
                y_test = (y_test - y_mean) / y_std

                n_jobs = 16  # change based on local machine reqs.

                # B11 parameters
                if n_jobs is not None:
                    params = {'learning_rate': 0.018807, 'n_estimators': 100, 'max_depth': 15, 'min_child_weight': 22,
                              'seed': 0, 'num_feature': len(features),
                              'subsample': 0.6768, 'gamma': 7.65, 'reg_alpha': 0, 'reg_lambda': 1, 'tree_method': 'auto',
                              'n_jobs': n_jobs}
                else:
                    params = {'learning_rate': 0.018807, 'n_estimators': 100, 'max_depth': 15, 'min_child_weight': 22,
                              'seed': 0, 'num_feature': len(features),
                              'subsample': 0.6768, 'gamma': 7.65, 'reg_alpha': 0, 'reg_lambda': 1, 'tree_method': 'auto'}

                M = xgb.XGBRegressor(**params)

            # train model
            print('fitting model')
            M.fit(X_train, y_train)

            if model == 'GPR':
                print(M.kernel_)

            # save model
            model_stage2_save_filepath = model_dir / 'trained_model_stage2.joblib'
            joblib.dump(M, model_stage2_save_filepath)
            print(f"Saved second trained model (stage2) to: {model_stage2_save_filepath}")

            norm_params_stage2 = {
                'X_mean': X_mean.to_dict(),
                'X_std': X_std.to_dict(),
                'y_mean': float(y_mean),
                'y_std': float(y_std),
                'features': features,
                'model_type': model,
                'mode': mode, # This mode corresponds to the overall script mode, not necessarily specific to stage2 inputs if they differ
                'qf': qf,
                'footprint': footprint # As above
            }
            norm_params_stage2_filepath = model_dir / 'normalization_params_stage2.json'
            with open(norm_params_stage2_filepath, 'w') as f:
                json.dump(norm_params_stage2, f, indent=4)
            print(f"Saved second model (stage2) normalization parameters to: {norm_params_stage2_filepath}")

            # calculate R^2. If train and test are very different try reducing the depth of the random forest
            print('calculate R^2 for 2nd model')
            R2_train = M.score(X_train, y_train);
            print('R2 Train:', R2_train)
            R2_test = M.score(X_test, y_test);
            print('R2 Test :', R2_test)


        bias_test, bias_std_test = make_prediction(M, X_test, model, UQ=False, X_train=X_train, y_mean=y_mean, y_std=y_std)
        data_test.loc[:, 'xco2MLcorr'] = data_test.loc[:, 'xco2_raw'] - bias_test
        data_test.loc[:, 'xco2MLbias'] = bias_test

        data_test_folds.append(data_test)

    data_test = pd.concat(data_test_folds)

    # Save test data for later analysis
    # Ensure the directory exists
    model_dir.mkdir(parents=True, exist_ok=True)
    data_test.to_parquet(model_dir / 'data_test.parquet')

    # ****************************************************************

    # make plots and calculate performance for each quality flag
    data_test_all = data_test
    name_all = name
    #analyze performance by Viewing Mode
    if mode == 'all':
        for m in ['LndGL', 'LndND', 'SeaGL']:
            print(m)
            if m == 'LndND':
                print('removing ocean')
                d_m = data.loc[data['land_fraction'] == 100, :]
                d_m = d_m.loc[d_m['operation_mode'] == 0, :]
            elif m == 'LndGL':
                print('removing ocean')
                d_m = data.loc[data['land_fraction'] == 100, :]
                d_m = d_m.loc[d_m['operation_mode'] == 1, :]
            elif m == 'SeaGL':
                print('removing land')
                d_m = data.loc[data['land_fraction'] == 0, :]
            else:
                d_m = data_test_all

            for qf in [0, 1]:
                print('making plots for QF=' + str(qf))
                d_mq = d_m.loc[d_m['xco2_quality_flag'] == qf]
                name = name_all + '_' + m + '_QF' + str(qf)

                ## compare to TCCON
                _, _, _, _, _, _, _, _, _ = plot_tccon(
                    d_mq, TCCON_names, save_fig=save_fig, path=model_dir, name=name, qf=qf,
                    precorrect_IO=precorrect_IO)

                for t_name in TCCON_names:
                    _, _, _, _, _, _, _, _, _ = plot_tccon(
                        d_mq, t_name, save_fig=save_fig, path=model_dir, name=name + t_name, qf=qf,
                        precorrect_IO=precorrect_IO)
    else:
        for qf in [0, 1]:
            print('making plots for QF=' + str(qf))
            data_test = data_test_all.loc[data_test_all['xco2_quality_flag'] == qf]
            name = name_all + '_QF' + str(qf)

            ## compare to TCCON
            _, _, _, _, _, _, _, _, _ = plot_tccon(
                data_test, TCCON_names, save_fig=save_fig, path=model_dir, name=name, qf=qf,
                precorrect_IO=precorrect_IO)

            for t_name in TCCON_names:
                _, _, _, _, _, _, _, _, _ = plot_tccon(
                    data_test, t_name, save_fig=save_fig, path=model_dir, name=name + t_name, qf=qf,
                    precorrect_IO=precorrect_IO)



# make changes #############################
name = 'V15_3_k_2022_prec_'  # name of generated model
save_fig = True  # save figures to hard drive
qf = None  # what quality flag data: 0: best quality; 1:lesser quality; none: all data
max_n = 2*10 ** 7  # max number of samples for training
Steffen_IO = True  # is this Steffen's computer
footprint = 0
mode = 'LndNDGL' # Viewing Mode: 'LndGL', 'LndND', 'SeaGL', 'all'
precorrect_IO = True


main(name, save_fig, mode, qf, max_n, footprint, precorrect_IO)

print('Done >>>')