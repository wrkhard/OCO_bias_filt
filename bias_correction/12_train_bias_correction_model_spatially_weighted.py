# Steffen Mauceri
#
# correct biases in XCO2 retrieval with ML
# takes in a list of state vector elements from XCO2 retrievals that are informative of XCO2 biases
# Grows a random forest to predict biases in XCO2
# change to 2.4 calculated SA bias only from high quality data to remove 3D could biases
# change to 2.5 added weighting based on numbers of soundings per TCCON site



import pandas as pd
import numpy as np
import joblib
import json
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge, RANSACRegressor,  BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF
import xgboost as xgb


import paths
from util import load_data, \
    get_variability_reduction, get_importance,plot_tccon, plot_map, make_prediction, feature_selector, \
    get_RMSE, filter_TCCON, calc_SA_bias, bias_correct,  weight_TCCON, load_and_concat_years



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
    # recalculate small area bias
    data.sort_values('SA', inplace=True)
    XCO2 = data['xco2_raw'].to_numpy()
    SA = data['SA'].to_numpy()

    data.loc[:, 'xco2raw_SA_bias'] = calc_SA_bias(XCO2, SA)
    data = data.loc[~data['xco2raw_SA_bias'].isna(),:]

    return data


def main(name, model, save_fig, feature_select, mode, var_tp, qf, max_n, max_depth, min_weight_fraction_leaf, verbose_IO=False):
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
    RIW = True

    # Define years and load_data arguments
    hold_out_year = 2022 # Year used for testing
    load_args = dict(qf=qf, preload_IO=preload_IO, TCCON=TCCON, remove_inland_water=RIW)

    # Load training data using the new function
    data_train = load_and_concat_years(2015, 2023, mode=mode, hold_out_year=hold_out_year, **load_args)
    # Load test data (single year)
    data_test = load_data(hold_out_year, mode=mode, verbose_IO=verbose_IO, **load_args)

    #TODO: remove this
    data_test = data_train.copy()


    # get TCCON station names
    TCCON_names = np.unique(data_train['tccon_name'].to_numpy())

    # count number of samples per station
    for n in TCCON_names:
        print(n + ' : ' + str(len(data_train[data_train['tccon_name'] == n])))

    # remove TCCON stations where we don't have enough measurements
    for t_name in TCCON_names:
        if data_train.loc[data_train['tccon_name'] == t_name, 'xco2tccon'].count() <= 50:
            print('removing ' + t_name)
            TCCON_names = TCCON_names[TCCON_names != t_name]

    # perform train test split on station names
    # TCCON_names_train, TCCON_names_test = train_test_split(TCCON_names, test_size=0.4, random_state=1)
    TCCON_names_train = TCCON_names
    TCCON_names_test = TCCON_names

    # save TCCON test stations
    # write results out to file
    d = {'TCCON_test': TCCON_names_test}
    df = pd.DataFrame(data=d)
    df.to_csv(model_output_dir / (name + '_TC_names.txt'), index=False)


    if var_tp == 'xco2_TCCON_bias':
        # # only do a spatial split
        # data_train = pd.concat([data_train, data_test])
        # data_train = data_train[data_train['xco2tccon'] > 0]
        # data_train.loc[:, var_tp] = data_train['xco2_raw'] - data_train['xco2tccon']
        # # data_test.loc[:, var_tp] = data_test['xco2_raw'] - data_test['xco2tccon']
        # data_test = data_train
        # # get testing TCCON stations
        # data_test = filter_TCCON(data_test, TCCON_names_test)
        # # get training TCCON stations
        # data_train = filter_TCCON(data_train, TCCON_names_train)

        data_train = data_train[data_train['xco2tccon'] > 0]
        data_test = data_test[data_test['xco2tccon'] > 0]
        data_train.loc[:, var_tp] = data_train['xco2_raw'] - data_train['xco2tccon']
        data_test.loc[:, var_tp] = data_test['xco2_raw'] - data_test['xco2tccon']

        # remove small inland water bodies
        if mode == 'SeaGL':
            TCCON_keep = ['burgos','eureka','izana', 'nyalesund', 'saga','lauder',  'reunion', 'rikubetsu',  'tsukuba', 'wollongong',  'darwin']
            data_test = filter_TCCON(data_test, TCCON_keep)
            data_train = filter_TCCON(data_train, TCCON_keep)


        # calculate weights for each TCCON station based on number of samples and add to data_train
        data_train = weight_TCCON(data_train, features)

        # get testing TCCON stations
        data_test = filter_TCCON(data_test, TCCON_names_test)
        # get training TCCON stations
        data_train = filter_TCCON(data_train, TCCON_names_train)
    else: # only use data where we have no TCCON data
        # calculate weights for land ocean crossings
        data_train = weight_coast(data_train, multiplier=1)
        data_train = data_train[pd.isna(data_train['xco2tccon'])]


    print(TCCON_names_test)

    print(str(len(data_train)/1000) + 'k samples loaded')
    # reduce size of train-set to max_n samples
    if len(data_train) > max_n:
        data_train = data_train.sample(max_n)
    if len(data_test) > max_n:
        data_test = data_test.sample(max_n)
    print(str(len(data_train) / 1000) + 'k samples downsampled')

    if precorrect_IO:
        print('correcting data with existing model')
        # correct data with an existing model
        precorrect_model_name = "V11_4_2.4_xco2_TCCON_biasall_all_RF"
        precorrect_model_load_path = paths.MODEL_SAVE_DIR / precorrect_model_name 

        if not precorrect_model_load_path.exists() or not precorrect_model_load_path.is_dir():
            raise FileNotFoundError(f"Warning: Precorrect model directory {precorrect_model_load_path} not found. Skipping precorrection.")
        else:
            data_train = precorrect(precorrect_model_load_path, data_train)
            data_test = precorrect(precorrect_model_load_path, data_test)

    if var_tp in features:
        features.remove(var_tp)

    X_train, y_train = data_train[features], data_train[var_tp]
    X_test,  y_test  = data_test[features], data_test[var_tp]
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
                                  n_jobs=-1,
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
        kernel = 1 * RBF(length_scale=1, length_scale_bounds=(1,3)) + WhiteKernel(noise_level=0.5, noise_level_bounds=(0.05, 0.5))
        M = GaussianProcessRegressor(kernel=kernel, alpha=0.2, n_restarts_optimizer=20)

    if model == 'NN':
        # normalize data to 0 mean unit standard deviation
        X_train = (X_train - X_mean) / X_std
        X_test = (X_test - X_mean) / X_std
        y_train = (y_train - y_mean) / y_std
        y_test = (y_test - y_mean) / y_std

        # set up GPR parameters
        M = MLPRegressor(hidden_layer_sizes=(256,2), max_iter=200,verbose=True, learning_rate_init=0.0001)

    if model == 'Ridge':
        # normalize data to 0 mean unit standard deviation
        X_train = (X_train - X_mean) / X_std
        X_test = (X_test - X_mean) / X_std
        y_train = (y_train - y_mean) / y_std
        y_test = (y_test - y_mean) / y_std

        # set Ridge parameters
        M = Ridge(alpha=10**-4)

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
        # B10 parameters
        # if n_jobs is not None:
        #     params = {'learning_rate': 0.018807, 'n_estimators': 1000, 'max_depth': 11, 'min_child_weight': 22, 'seed': 0, 'num_feature' : len(features),
        #         'subsample': 0.6768, 'gamma': 7.65, 'reg_alpha': 0, 'reg_lambda': 1, 'tree_method' : 'auto', 'n_jobs' : n_jobs}
        # else:
        #     params = {'learning_rate': 0.018807, 'n_estimators': 1000, 'max_depth': 11, 'min_child_weight': 22, 'seed': 0, 'num_feature' : len(features),
        #         'subsample': 0.6768, 'gamma': 7.65, 'reg_alpha': 0, 'reg_lambda': 1, 'tree_method' : 'auto'}
            
        M = xgb.XGBRegressor(**params)


    print('fitting model')
    M.fit(X_train, y_train, sample_weight=weights_train)

    if model == 'GPR':
        print(M.kernel_)

    # Save model and parameters using joblib
    model_save_filepath = model_output_dir / 'trained_model.joblib'
    joblib.dump({
        'X_mean': X_mean, 
        'y_mean': y_mean, 
        'X_std': X_std, 
        'y_std': y_std, 
        'TrainedModel': M,
        'features': features, 
        'model': model, 
        'qf': qf, 
        'mode': mode
    }, model_save_filepath)
    print(f"Saved trained model to: {model_save_filepath}")

    # Additionally save normalization parameters in JSON format for easier access
    norm_params = {
        'X_mean': X_mean.to_dict(),
        'X_std': X_std.to_dict(),
        'y_mean': float(y_mean),
        'y_std': float(y_std),
        'features': features,
        'model': model,
        'qf': qf if qf is not None else "None",
        'mode': mode
    }
    norm_params_filepath = model_output_dir / 'normalization_params.json'
    with open(norm_params_filepath, 'w') as f:
        json.dump(norm_params, f, indent=4)
    print(f"Saved normalization parameters to: {norm_params_filepath}")

    # calculate R^2. If train and test are very different try reducing the depth of the random forest
    print('calculate R^2')
    R2_train = M.score(X_train, y_train); print('R2 Train:',R2_train)
    R2_test  = M.score(X_test, y_test);   print('R2 Test :',R2_test)

    bias_test, bias_std_test = make_prediction(M, X_test, model,UQ=False,X_train=X_train, y_mean=y_mean, y_std=y_std)
    # bias_train, bias_std_train = make_prediction(M, X_train, model,UQ=False,X_train=X_train, y_mean=y_mean, y_std=y_std)

    data_test.loc[:,'xco2MLcorr'] = data_test.loc[:,'xco2_raw'] - bias_test
    data_test.loc[:,'xco2MLbias'] = bias_test

    # data_train.loc[:,'xco2MLcorr'] = data_train.loc[:,'xco2_raw'] - bias_train


    # ****************************************************************
    # make plots
    if precorrect_IO:
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
        name = name_all + '_QF_' + str(qf)

        # recalculate small area bias
        data_test.sort_values('SA', inplace=True)
        XCO2 = data_test.loc[:, 'xco2MLcorr'].to_numpy()
        SA = data_test.loc[:, 'SA'].to_numpy()
        data_test.loc[:, 'xco2raw_SA_bias-ML'] = calc_SA_bias(XCO2, SA)

        ## compare to TCCON

        xco2ML_std, xco2ML_median, xco2B11_std, xco2B11_median, xco2raw_std, xco2raw_median, xco2ML_RMSE, xco2B11_RMSE,xco2raw_RMSE = plot_tccon(data_test, TCCON_names_test, save_fig=save_fig, path=model_output_dir, name=name, qf=qf, precorrect_IO=precorrect_IO)

        # plot variability reduction
        get_variability_reduction(data_test, var_tp, name, path=model_output_dir, save_fig=save_fig, qf=qf)


        # calculate RMSE before and after bias correction
        print('get RMSE reduction')
        RMSE_Raw = get_RMSE(data_test['xco2raw_SA_bias'].to_numpy())
        RMSE_ML = get_RMSE(data_test['xco2raw_SA_bias-ML'].to_numpy())
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

        if precorrect_IO:
            # plot difference to original raw XCO2 before precorrection
            data_test.loc[:,'xco2MLcorr-raw_xco2'] = data_test['xco2MLcorr'] - data_test['xco2_raw_orig']
            plot_map(data_test, ['xco2MLcorr-raw_xco2'], save_fig=save_fig, path=model_output_dir, name=name + '_diff_orig_rawXCO2', pos_neg_IO=True)

            # visualize bias correction of precorrect step
            # plot difference to original raw XCO2 before precorrection
            data_test.loc[:,'precor-raw_xco2'] = data_test['xco2_raw'] - data_test['xco2_raw_orig']
            plot_map(data_test, ['precor-raw_xco2'], save_fig=save_fig, path=model_output_dir, name=name + '_diff_precor', pos_neg_IO=True)



# make changes #############################
name = 'V2_11.2_2.6_del_' # name of generated model
save_fig = True         # save figures to hard drive
qf = None               # what quality flag data: 0: best quality; 1:lesser quality; none: all data
max_n = 2*10**7          # max number of samples for training
max_depth = 15          # depth of decission tree. Higher numbers corresbond to a more complex function. Try values [5 to 15]
min_weight_fraction_leaf = 0.0005 # minimum weight fraction leaf for RF
precorrect_IO = False  # precorrect data with another model (defined in main program)

for model in ['RF']: #'RF', 'Ridge', 'Ransac', 'XGB' What model to use
    for var_tp in ['xco2_TCCON_bias']:#['xco2_TCCON_bias', 'xco2_SA_bias', 'xco2raw_SA_bias']:
        for mode in ['SeaGL', 'LndNDGL']: #'all', 'LndGL', 'LndND', 'LndNDGL', 'SeaGL'
# stop make changes ###########################
            if mode == 'LndNDGL':
                if var_tp == 'xco2_TCCON_bias':
                    feature_select = 'TCCON_bias_Lnd'
                    max_depth = 20
                else:
                    feature_select = 'SA_bias_LndNDGL'
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
                    min_weight_fraction_leaf = 0.005
                else:
                    feature_select = 'SA_bias_Sea'
            elif mode == 'all':
                if var_tp == 'xco2_TCCON_bias':
                    feature_select = 'TCCON_bias_all'
                else:
                    feature_select = 'SA_bias_all'

            main(name, model,save_fig,feature_select,mode,var_tp,qf,max_n,max_depth, min_weight_fraction_leaf)

print('Done >>>')

# run vis_bias_corr.py script to visualize results
