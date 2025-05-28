# Steffen Mauceri & William Keely
#
# Filter bias corrected XCO2 using ML
# takes in a list of state vector elements from XCO2 retrievals that are informative of XCO2 biases
# Grows a random forest to predict biases in XCO2 or a cross validated user defined model.

# TODO: Add support for mode = 'all'
# TODO: Add support for SVC etc
# TODO: Add gridSearch args and experiment loop.
# TODO: add in CO2_Plume.py 
# TODO: Finish decision boundary plots.
# TODO: Combine SA & TCCON ...


import os
import pandas as pd
import numpy as np
import json
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm
import joblib
from pathlib import Path

from models import NNClassifier



from util import load_data, scatter_density, scatter_3D, subsample, train_test_split_Time,\
    get_variability_reduction, get_importance,plot_tccon, plot_map, make_prediction,normalize_per_SA, feature_selector, \
    get_RMSE, filter_TCCON, bias_correct, calc_SA_bias, plot_qf_map, hex_plot, plot_decision_surface

def generate_training_set(train_years,tccon_flag = False,mode='LndNDGL',qf=None,preload_IO=True,footprint=0,Steffen_IO=False):
    '''
    :param years: list of years for training
    :param tccon_flag: bool, if True use tccon with SA for training
    '''
    data = []
    # for year in train_years:
    #     if not tccon_flag:
    #         data.append(load_data(year,mode,qf=qf,preload_IO=preload_IO,footprint=footprint,Steffen_IO=Steffen_IO))
    #     else:
    #         d = load_data(year,mode,qf=qf,preload_IO=preload_IO, footprint=footprint,Steffen_IO=Steffen_IO)
    #         data_tccon = d[d.xco2tccon.notnull()]
    #         data_tccon.loc[:,'xco2_raw'] = data_tccon.loc[:,'xco2tccon']
    #         d = pd.concat([d,data_tccon])
    #         data.append(d)
    for year in train_years:
        data.append(load_data(year,mode,qf=qf,preload_IO=preload_IO,footprint=footprint,Steffen_IO=Steffen_IO))
    return pd.concat(data)
            

def main( model, save_fig, feature_select, mode, train_years, var_tp, qf, max_n, max_depth, footprint, error_threshold,sample_weights = True,correction_type = 'all',plot_verbose=True, verbose_IO=False, Steffen_IO = False, tccon_flag = True):
    ''' Main function to train a model and make some preliminary plots

    :param name: str, name of model
    :param model: str, kind of model ['RF', 'Ridge', 'GPR']
    :param save_fig: bool,
    :param mode: str, ['LndND', 'SeaGL', 'LndGL', 'all', 'LndNDGL']
    :param train_years, [2018,2020,2022]
    :param var_tp: str, variable to predict
    :param qf: int, quality flag, use 'None' to not filter based on QF
    :param max_n: int, maximum number of samples to work with (to save RAM)
    :param max_depth: int, max depth of RF
    :param footprint: int, which footprint to work on, use 0 to work on all simultaneously
    :param error_threshold: float, ppm threshold for ML QF.
    :param sample_weights: bool, use weighting strategy?
    :param correction_type: string, 'all', 'LndND', 'SeaGL'
    :param plot_verbose: bool, output additional plots?
    :param verbose_IO: bool, extra output
    :param Steffen_IO: is this Steffen's computer?
    :param tccon_flag: bool, use tccon in training?
    :return:
    '''

    # path strings for bias correction models and qf dir.
    if Steffen_IO:
        tcconRidgeDirString = "" # for Steffen's computer.
        rfDirString = ""
        qfModelDirString = ""
        
    else: # William's local 
        if correction_type == 'all':
            tcconRidgeDirString = '/Users/williamkeely/Desktop/B11/current/all_correction/V3_7_nofoot_noTCsplit_xco2_TCCON_biasall_all_Ridge0/'  # path to 1st bias model directory
            rfDirString = '/Users/williamkeely/Desktop/B11/current/all_correction/V3_7_precRidge_nofoot_noTCsplit_xco2raw_SA_biasall_all_RF0/'  # path to 2nd bias model directory
            qfModelDirString = '/Users/williamkeely/Desktop/B11/current/MLQF_models' 
        elif correction_type == 'LndND':
            tcconRidgeDirString = '/Users/williamkeely/Desktop/B11/current/land_correction/V3_7_nofoot_noTCsplit_xco2_TCCON_biasLndND_lnd_Ransac0/'  # path to 1st bias model directory
            rfDirString = '/Users/williamkeely/Desktop/B11/current/land_correction/V3_7_precRansac_nofoot_noTCsplit_xco2raw_SA_biasLndND_lnd_RF0/'  # path to 2nd bias model directory
            qfModelDirString = '/Users/williamkeely/Desktop/B11/current/MLQF_models'
        elif correction_type == 'SeaGL':
            tcconRidgeDirString = '/Users/williamkeely/Desktop/B11/current/all_correction/V3_7_nofoot_noTCsplit_xco2_TCCON_biasall_all_Ridge0/'  # path to 1st bias model directory
            rfDirString = '/Users/williamkeely/Desktop/B11/current/all_correction/V3_7_precRidge_nofoot_noTCsplit_xco2raw_SA_biasall_all_RF0/'  # path to 2nd bias model directory
            qfModelDirString = '/Users/williamkeely/Desktop/B11/current/MLQF_models' 
            
        else:
            print('correction type must be LndND, all, or SeaGL')
            
    # TODO: Move string and folder creation to a function.
    sample_weight_string = 'true'
    if not sample_weights:
        sample_weight_string = 'false'

    name = "V2_"+str(max_depth)+"_QFmode_"+str(mode)+"_biascorr_"+correction_type+"_weight_"+sample_weight_string+"_nofoot_noTCsplit_MLQF"+model+"_thesh"+str(error_threshold)
    print(name)
    
    # generate model folder 
    model_folder = qfModelDirString + "/" + name
    try:
        os.mkdir(model_folder)
    except FileExistsError:
       # directory already exists
        print("folder already exists...")
        pass 
    model_folder = model_folder + "/"
    
    # load data
    preload_IO = True
    
    data = []
    # generate_training_set() ............
    data_train = generate_training_set(train_years=train_years,tccon_flag = tccon_flag,mode=mode,qf=qf,preload_IO=preload_IO,footprint=footprint,Steffen_IO=Steffen_IO)

        

    data_test = load_data(2021, mode, verbose_IO=verbose_IO, qf=qf, preload_IO=preload_IO, footprint=footprint, Steffen_IO=Steffen_IO)
    data_qf = data_test.copy(deep=True) # deep copy for plotting
    data_qf = data_qf[data_qf['xco2_quality_flag'] == 0] # use to evaluate difference between MLQF and B11 QF


    # make bias correction 1
    vars = ['xco2_raw']
    data_train = bias_correct(tcconRidgeDirString, data_train, vars)
    data_test = bias_correct(tcconRidgeDirString, data_test, vars)

    # make bias correction 2 # if we have a two step bias corr model
    data_train = bias_correct(rfDirString, data_train, vars)
    data_test = bias_correct(rfDirString, data_test, vars)

    data_train['xco2MLcorr'] =  data_train['xco2_raw']
    data_test['xco2MLcorr'] = data_test['xco2_raw']
    
    # recalculate small area bias
    data_test.sort_values('SA', inplace=True)
    XCO2 = data_test.loc[:, 'xco2MLcorr'].to_numpy()
    SA = data_test.loc[:, 'SA'].to_numpy()
    data_test.loc[:, 'xco2raw_SA_bias'] = calc_SA_bias(XCO2, SA)
    
    data_test_qf_rmse_compare = data_test.copy(deep=True) # use to compare RMSE of MLBC between MLQF and B11 QF.
    data_test_qf_rmse_compare = data_test_qf_rmse_compare[data_test_qf_rmse_compare['xco2_quality_flag'] == 0]
    
    # get features for filtering
    features, feature_n = feature_selector(feature_select)
    X_train = data_train[features].to_numpy()
    X_test = data_test[features].to_numpy()
    
    print("Number of features for filtering : ", X_train.shape)


    y_train_c = (data_train.loc[:, 'xco2raw_SA_bias']  >= error_threshold) | (data_train.loc[:,'xco2raw_SA_bias'] <= -error_threshold)# threshold for what classification.
    y_test_c = (data_test.loc[:, 'xco2raw_SA_bias']  >= error_threshold) | (data_test.loc[:,'xco2raw_SA_bias'] <= -error_threshold)
                               
    y_train_c = y_train_c.astype(int)
    y_test_c = y_test_c.astype(int)
                               
    y_train_c = y_train_c.to_numpy()
    y_test_c = y_test_c.to_numpy()
    
    print("y_train_c : ", y_train_c)
    

    # add more weight to the QF=1 data. This will compensate for our unbalanced training set and make QF=0 more pure.
    # TODO add weights depending on how far individual soundings are away from decision boundary
    if sample_weights:
        weights_train = y_train_c * 1/np.mean(y_train_c) + 1
        

    print(str(np.round(np.mean(y_train_c) * 100)) + '% outliers [train]' ); print(str(np.round(np.mean(y_test_c) * 100)) + '% outliers [test]' )

    if model == 'RF':
        # set up Random Forest parameters
        M = RandomForestClassifier(n_estimators=500,
                                   max_depth=max_depth,
                                   max_samples=len(X_train)//2,
                                   min_samples_split=40,
                                   min_samples_leaf=10,
                                   min_weight_fraction_leaf=0.0,
                                   max_features='sqrt',
                                   max_leaf_nodes=None,
                                   min_impurity_decrease=0.0,
                                   bootstrap=True,
                                   oob_score=False,
                                   n_jobs=16,
                                   random_state=None,
                                   verbose=0,
                                   warm_start=False)
    if model == 'NN':
        # set up MLP parameters for sklearn MLPClassifier
        # M = MLPClassifier(hidden_layer_sizes=(128,2),
        #                   activation = 'logistic',
        #                   solver='adam',
        #                   alpha = 0.001,
        #                   batch_size = 1000,
        #                   shuffle = True,
        #                   learning_rate = 'adaptive',
        #                   n_iter_no_change = 20,
        #                   max_iter=1000,
        #                   verbose=True)
        M = NNClassifier(n_features = len(features),
                         n_outputs = 1, 
                         n_layers = [512,512], 
                         activation='sigmoid', 
                         lrate=0.001
                    )


        
        
    # train model
    print('fitting model')
    if model == 'NN' and sample_weights:
        # M.fit(X_train, y_train_c) # sample_weight not supported by sklearn MLPClassifier .. look at keras implementation.
        # M.fit(X_train, y_train_c,
        #     epochs = 1,
        #     batch_size = 1000,
        #     class_weight=None,
        #     sample_weight=weights_train,
        #     workers=5,
        #     verbose = 2)
        M.fit(X_train, y_train_c, class_weight = {0 : 1.0, 1 : 1.0}, batch_size = 128, shuffle = True, epochs = 200)
    if model == 'RF' and sample_weights:    
        M.fit(X_train, y_train_c, sample_weight=weights_train)
    elif model =='RF' and not sample_weights:
        M.fit(X_train, y_train_c)


    # Save the trained model M using joblib
    model_output_path = Path(model_folder) # Ensure model_folder is a Path object
    paths.ensure_dir_exists(model_output_path) # Ensure the directory exists

    model_save_filepath = model_output_path / 'trained_model.joblib'
    joblib.dump(M, model_save_filepath)
    print(f"Saved trained model to: {model_save_filepath}")

    # Save parameters to normalization_params.json
    # Gather parameters that were defined and used in the function scope
    # For classifiers, X_mean, X_std, y_mean, y_std might not be relevant as for regressors,
    # but include if they are computed and used (e.g., for some NN inputs).
    # Based on the provided code, X_mean/std are not computed for the classifier inputs.
    model_params = {
        'features': features, # list of feature names
        'model_type': model, # e.g., 'RF', 'NN'
        'mode': mode, # e.g., 'LndNDGL'
        'qf': qf, # quality flag used
        'error_threshold': error_threshold,
        'train_years': train_years,
        'max_n_samples_training': max_n, # max_n was a parameter to the main function
        'correction_type': correction_type,
        'feature_select_name': feature_select, # name of the feature selection group used
        'sample_weights_used': sample_weights
    }
    if model == 'RF':
        model_params['max_depth'] = max_depth
        # Potentially other RF params like min_samples_split, min_samples_leaf if needed for reproducibility/info
        model_params['rf_n_estimators'] = M.get_params().get('n_estimators')
        model_params['rf_max_samples'] = M.get_params().get('max_samples')
        model_params['rf_min_samples_split'] = M.get_params().get('min_samples_split')
        model_params['rf_min_samples_leaf'] = M.get_params().get('min_samples_leaf')
        model_params['rf_max_features'] = M.get_params().get('max_features')

    # If X_mean and X_std were calculated and used (e.g. for NN), add them
    # if 'X_mean' in locals() and 'X_std' in locals():
    #     model_params['X_mean'] = X_mean.to_dict() if isinstance(X_mean, pd.Series) else X_mean
    #     model_params['X_std'] = X_std.to_dict() if isinstance(X_std, pd.Series) else X_std

    params_save_filepath = model_output_path / 'normalization_params.json'
    with open(params_save_filepath, 'w') as f:
        json.dump(model_params, f, indent=4)
    print(f"Saved model parameters to: {params_save_filepath}")

    # calculate R^2. If train and test are very different try reducing the depth of the random forest
    print('calculate Accuracy')
    # If model is not a tf NN
    if model != 'NN':
        Accuracy_train = M.score(X_train, y_train_c); print('Accuracy Train:',Accuracy_train)
        Accuracy_test  = M.score(X_test, y_test_c);   print('Accuracy Test :',Accuracy_test)
    




    QF_test, QF_std_test = make_prediction(M, X_test, model)

    data_test['xco2_MLquality_flag'] = QF_test

    data_test_QF0 = data_test.loc[data_test['xco2_MLquality_flag'] == 0]
    data_test_QF1 = data_test.loc[data_test['xco2_MLquality_flag'] == 1]
    print('QF = 0 : ', data_test_QF0.head())
    print('QF = 1 : ', data_test_QF1.head())
    prec, recal, F1, _ = precision_recall_fscore_support(y_test_c, QF_test)
    print('[QF=0, QF=1]')
    print('Precision Test :', prec)
    print('Recall Test :', recal)
    print('F1 Test :', F1)



    # calculate RMSE before and after bias correction
    print('get RMSE reduction')
    RMSE_SA_QF0 = get_RMSE(data_test_QF0['xco2raw_SA_bias'])    ; print(RMSE_SA_QF0)
    RMSE_SA_QF1 = get_RMSE(data_test_QF1['xco2raw_SA_bias'])    ; print(RMSE_SA_QF1)
    RMSE_SA_QF01 = get_RMSE(data_test['xco2raw_SA_bias'])       ; print(RMSE_SA_QF01)
    RMSE_SA_B11QF0 = get_RMSE(data_test_qf_rmse_compare['xco2raw_SA_bias']) ; print(RMSE_SA_B11QF0)

    #
    # # # compare to TCCON
    # get TCCON station names
    TCCON_names = np.unique(data_train['tccon_name'].to_numpy())
    TCCON_names = TCCON_names[TCCON_names != 'pasadena01']
    TCCON_names = TCCON_names[TCCON_names != 'xianghe01']
    print('TCCON Stations : ', TCCON_names)
    # perform train test split on station names
    TCCON_names_train, TCCON_names_test = train_test_split(TCCON_names, test_size=0.4, shuffle=False,
                                                           random_state=1)
     
    # TCCON plots
    xco2ML_std, xco2ML_mean, xco2B11_std, xco2B11_mean, xco2raw_std, xco2raw_mean, xco2ML_RMSE_QF0, xco2B11_RMSE, xco2raw_RMSE = plot_tccon(data_test_QF0,
                                                                                               TCCON_names_test,
                                                                                               save_fig=save_fig,
                                                                                               path=model_folder,
                                                                                               name=name + '_MLQF0_'  + str(error_threshold))
    xco2ML_std, xco2ML_mean, xco2B11_std, xco2B11_mean, xco2raw_std, xco2raw_mean, xco2ML_RMSE_QF1, xco2B11_RMSE, xco2raw_RMSE = plot_tccon(data_test_QF1,
                                                                                               TCCON_names_test,
                                                                                               save_fig=save_fig,
                                                                                               path=model_folder,
                                                                                               name=name+ '_MLQF1_' + str(error_threshold))


    # save metrics to json
    QF1_pc = data_test['xco2_MLquality_flag'].mean() * 100
    QF0_pc = 100-QF1_pc
    print("Percent passing = " + str(QF0_pc) + "%")
    d = {
        'RMSE_SA_MLQF0': RMSE_SA_QF0,
        'RMSE_SA_MLQF1': RMSE_SA_QF1,
        'RMSE_SA_MLQF01': RMSE_SA_QF01,
        'QF=1_pc': QF1_pc,
        'RMSE_SA_B11QF0': RMSE_SA_B11QF0,
        'RMSE_TCCON_MLQF0': xco2ML_RMSE_QF0,
        'RMSE_TCCON_B11': xco2B11_RMSE,
        'B11_QF0_Pass': 59
    }
    json_fp = model_folder / (name + "_results.json")
    with open(json_fp, 'w') as f:
        json.dump(d, f, indent=4)
    print(f'Results saved to {json_fp}')

    # ****************************************************************
    print('making plots')
    

    

    if model == 'RF':
        get_importance(M, X_test, name, model_folder, save_IO=save_fig)

    # plot maps
    # plot difference to B11
    # QF = 0   
    data_test_QF0['xco2MLcorr-B11'] = data_test_QF0['xco2MLcorr'] - data_test_QF0['xco2']
    plot_map(data_test_QF0, ['xco2MLcorr-B11'], save_fig=save_fig, path=model_folder, name=name + '_diff_B11 ' + '_MLQF0_' + str(error_threshold), pos_neg_IO=True)
    # QF = 1
    data_test_QF1['xco2MLcorr-B11'] = data_test_QF1['xco2MLcorr'] - data_test_QF1['xco2']
    plot_map(data_test_QF1, ['xco2MLcorr-B11'], save_fig=save_fig, path=model_folder, name=name + '_diff_B11'+ '_MLQF1_' + str(error_threshold), pos_neg_IO=True)
    
    # plot SA remaining biases
    # QF = 0
    plot_map(data_test_QF0, ['xco2raw_SA_bias'], save_fig=save_fig, path=model_folder, name=name + '_remaining_biases_SA_ ' + '_MLQF0_' + str(error_threshold), pos_neg_IO=True)
    # QF = 1
    plot_map(data_test_QF1, ['xco2raw_SA_bias'], save_fig=save_fig, path=model_folder, name=name + '_remaining_biases_SA_'+ '_MLQF1_' + str(error_threshold), pos_neg_IO=True)
    
    # plot difference in % passing B11QF and MLQF
    plot_qf_map(data_qf,data_test_QF0, 'xco2raw_SA_bias',year = '2021', save_fig=True, path=model_folder, name='None',diff = True)
    
    if plot_verbose:
        # TODO: update hex bin plots for additional variables.
        # hexbin plots (3 plots min)
        # QF = 0
        hex_plot(data_test_QF0,name='MLQF=0',path=model_folder,save_fig = True, bias = 'xco2raw_SA_bias')
        # QF = 1
        hex_plot(data_test_QF1,name='MLQF=1',path=model_folder,save_fig = True,bias = 'xco2raw_SA_bias')

        # confusion matrix

        # decision boundary (3 plots min)
    




# make changes #############################
model = 'RF' #RF or GPR
save_fig = True    # save figures to hard drive
modes = ['LndNDGL']
train_years = [2018,2019,2020,2022] # remove 2019 for now..
var_tp = 'xco2raw_SA_bias'#'xco2_SA_bias'# 'xco2_SA_bias' or 'xco2raw_SA_bias', 'xco2_TCCON_bias'  # uncorrected bias based on SA # O'dell bias correction applied -> remaining biases based on SA
qf = None            # what quality flag data: 0: best quality; 1:lesser quality; none: all data
max_n = 10**7           # max number of samples for training
max_depth = 12          # depth of decission tree. Higher numbers corresbond to a more complex function. Try values [5 to 15]
debug = False
sample_weights = True
correction_type = 'LndND'
error_threshold = [1.0] # threshold for QF
# error_threshold = [1.7]
plot_verbose=True
Steffen_IO = False # is this Steffen's computer
tccon_flag = True
feature_select = 'SA_filt_Lnd'

#stop make changes ###########################
for threshold in error_threshold:
    for model in ['NN']:
        for var_tp in ['xco2raw_SA_bias']:#['xco2_TCCON_bias', 'xco2_SA_bias', 'xco2raw_SA_bias']:
            for mode in modes:
                #for footprint in range(1,9):
                footprint = 0
                if mode == 'LndNDGL':
                    feature_select = 'SA_filt_Lnd'
                    print(feature_select)
                elif mode == 'LndGL':
                    #feature_select = 0
                    max_depth = 8
                elif mode == 'SeaGL':
                    feature_select = 'SA_filt_Sea'
                elif mode == 'all':
                    feature_select = 'SA_filt_all'

                if model == 'RF':
                    max_n = 10 ** 7
                if model == 'Ridge':
                    max_n = 10 ** 6

                if var_tp == 'xco2_TCCON_bias':
                    max_depth = 5

                main(model=model,save_fig=save_fig,feature_select=feature_select,mode=mode,train_years=train_years,var_tp=var_tp,qf=qf,max_n=max_n,max_depth=max_depth, footprint=footprint,correction_type = correction_type, sample_weights = sample_weights,plot_verbose=plot_verbose, error_threshold = threshold, Steffen_IO = Steffen_IO, tccon_flag = tccon_flag)

print('Done >>>')
