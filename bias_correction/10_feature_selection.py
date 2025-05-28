# Steffen Mauceri
# Recursive feature elimination to find the smallest set of features for bias correction


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

import paths
from util import bias_correct, calc_SA_bias, filter_TCCON, get_season, weight_TCCON, load_and_concat_years


def precorrect(path, data):
    ''' correct data with an existing model
    :param path: str, path to model
    :param data: pd Dataframe, data to be corrected
    :return: pd DataFrame, corrected data
    '''

    # make a copy of the orignial raw xco2 for later plots and analysis
    data.loc[:,'xco2_raw_orig'] = data.loc[:,'xco2_raw'].copy()
    data = bias_correct(path, data, ['xco2_raw'])
    # recalculate small area bias
    data.sort_values('SA', inplace=True)
    XCO2 = data['xco2_raw'].to_numpy()
    SA = data['SA'].to_numpy()

    data.loc[:, 'xco2raw_SA_bias'] = calc_SA_bias(XCO2, SA)
    data = data.loc[~data['xco2raw_SA_bias'].isna(),:]

    return data

# make changes #########################################################
for mode in ['LndNDGL']:#['LndNDGL' ,'SeaGL', 'LndGL', 'all']
    for i in range(1,13,1):  # perform multiple runs with one TCCON held out station. Use var_tp = 'xco2_TCCON_bias'
    #for year in range(2016,2022): #perform multiple runs with one year held out. Use var_tp = 'xco2raw_SA_bias'
        
        name = mode + '_V_test_split_' + str(i) #+ str(year)
        # Construct plot save path using paths.py
        plot_save_dir = paths.FIGURE_DIR / "feature_selection" / name 
        # The actual filename for plots will be determined later in the script, e.g., plt.savefig(plot_save_dir / 'my_plot.png')
        model = 'RF'    #RF or GPR
        save_fig = True    # save figures to hard drive
        var_tp = 'xco2_TCCON_bias'  # 'xco2raw_SA_bias', 'xco2_TCCON_bias'
        qf = None            # what quality flag data: 0: best quality; 1:lesser quality; none: all data
        max_n = 2* 10**6           # max number of samples for training
        max_depth = 10           # depth of decission tree. Higher numbers corresbond to a more complex function. Try values [5 to 15]
        precorrect_IO = False
        permutation_importance_IO = True
#stop make changes #########################################################


        if var_tp == 'xco2_TCCON_bias':
            TCCON = True
        else:
            TCCON = False

        # get features
        if mode == 'SeaGL':
            # SeaGL
            features = ['solar_zenith_angle',
                        'sensor_zenith_angle'
                        ,'windspeed_u_met'
                        ,'windspeed_v_met'
                        ,'co2_ratio_bc'
                        ,'h2o_ratio_bc'
                        ,'max_declocking_o2a'
                        # ,'max_declocking_wco2'
                        ,'max_declocking_sco2'
                        ,'color_slice_noise_ratio_o2a'
                        # ,'color_slice_noise_ratio_wco2'
                        ,'color_slice_noise_ratio_sco2'
                        ,'h_continuum_o2a'
                        # ,'h_continuum_wco2'
                        ,'h_continuum_sco2'
                        ,'dp_abp'
                        #,'surface_type'
                        ,'psurf'
                        ,'windspeed'
                        #,'windspeed_apriori'
                        #,'psurf_apriori'
                        ,'t700'
                        ,'tcwv'
                        ,'tcwv_apriori'
                        ,'dp'
                        ,'dp_o2a'
                        ,'dp_sco2'
                        #,'dpfrac'
                        ,'co2_grad_del'
                        # ,'dws'
                        #,'eof3_1_rel'
                        #,'snow_flag'
                        ,'aod_dust'
                        #,'aod_bc'
                        #,'aod_oc'
                        ,'aod_seasalt'
                        ,'aod_sulfate'
                        ,'aod_strataer'
                        ,'aod_water'
                        ,'aod_ice'
                        # ,'aod_total'
                        ,'dust_height'
                        ,'ice_height'
                        ,'water_height'
                        #,'aod_total_apriori'
                        #,'dws_apriori'
                        #,'aod_fine_apriori'
                        ,'h2o_scale'
                        ,'deltaT'
                        ,'albedo_o2a'
                        # ,'albedo_wco2'
                        ,'albedo_sco2'
                        ,'albedo_slope_o2a'
                        ,'albedo_slope_wco2'
                        ,'albedo_slope_sco2'
                        # ,'albedo_quad_o2a'
                        # ,'albedo_quad_wco2'
                        # ,'albedo_quad_sco2'
                        # ,'brdf_weight_slope_wco2'
                        # ,'brdf_weight_slope_sco2'
                        # ,'chi2_o2a'
                        # ,'chi2_wco2'
                        # ,'chi2_sco2'
                        ,'rms_rel_o2a'
                        ,'rms_rel_wco2'
                        ,'rms_rel_sco2'
                        #,'solar_azimuth_angle'
                        #,'sensor_azimuth_angle'
                        #,'polarization_angle'
                        #,'land_fraction'
                        ,'glint_angle'
                        ,'airmass'
                        ,'snr_o2a'
                        #,'snr_wco2'
                        ,'snr_sco2'
                        #,'path'
                        ,'footprint'
                        #,'land_water_indicator'
                        ,'altitude']
        elif mode == 'all':
            features = [
                'solar_zenith_angle',
                        'sensor_zenith_angle'
                # , 'windspeed_u_met'
                # , 'windspeed_v_met'
                , 'co2_ratio'
                , 'h2o_ratio'
                 # , 'max_declocking_o2a'
                 # ,'max_declocking_wco2'
                 # , 'max_declocking_sco2'
                , 'color_slice_noise_ratio_o2a'
                        # ,'color_slice_noise_ratio_wco2'
                , 'color_slice_noise_ratio_sco2'
                , 'h_continuum_o2a'
                        # ,'h_continuum_wco2'
                , 'h_continuum_sco2'
                , 'dp_abp'
                        # ,'surface_type'
                # , 'psurf'
                # , 'psurf_apriori'
                , 't700'
                        # ,'fs'
                        # ,'fs_rel'
                , 'tcwv'
                        # ,'tcwv_apriori'
                , 'dp'
                , 'dp_o2a'
                        # ,'dp_sco2'
                , 'dpfrac'
                , 'co2_grad_del'
                , 'dws'
                # , 'eof3_1_rel'
                        # ,'snow_flag'
                , 'aod_dust'
                # , 'aod_bc'
                # , 'aod_oc'
                # , 'aod_seasalt'
                # , 'aod_sulfate'
                , 'aod_strataer'
                , 'aod_water'
                , 'aod_ice'
                        # ,'aod_total'
                , 'dust_height'
                , 'ice_height'
                , 'water_height'
                # , 'aod_total_apriori'
                # , 'dws_apriori'
                # , 'aod_fine_apriori'
                , 'h2o_scale'
                , 'deltaT'
                , 'albedo_o2a'
                        # ,'albedo_wco2'
                , 'albedo_sco2'
                , 'albedo_slope_o2a'
                , 'albedo_slope_wco2'
                , 'albedo_slope_sco2'
                # , 'albedo_quad_o2a'
                # , 'albedo_quad_wco2'
                # , 'albedo_quad_sco2'
                        # ,'brdf_weight_slope_wco2'
                        # ,'brdf_weight_slope_sco2'
                        # ,'chi2_o2a'
                        # ,'chi2_wco2'
                        # ,'chi2_sco2'
                , 'solar_azimuth_angle'
                , 'sensor_azimuth_angle'
                # , 'polarization_angle'
                # ,'land_fraction'
                , 'glint_angle'
                , 'airmass'
                , 'snr_o2a'
                        # ,'snr_wco2'
                , 'snr_sco2'
                        # ,'path'
                , 'footprint'
                # ,'land_water_indicator'
                , 'altitude'
                , 'altitude_stddev']
        else:
            # LndND
            features = ['solar_zenith_angle',
                        'sensor_zenith_angle'
                        ,'co2_ratio_bc'
                        ,'h2o_ratio_bc'
                        ,'max_declocking_o2a'
                        #,'max_declocking_wco2'
                        ,'max_declocking_sco2'
                        ,'color_slice_noise_ratio_o2a'
                        #,'color_slice_noise_ratio_wco2'
                        ,'color_slice_noise_ratio_sco2'
                        ,'h_continuum_o2a'
                        #,'h_continuum_wco2'
                        ,'h_continuum_sco2'
                        ,'dp_abp'
                        #,'surface_type'
                        # ,'psurf'
                        # ,'psurf_apriori'
                        ,'t700'
                        # ,'fs'
                        #,'fs_rel'
                        ,'tcwv'
                        # ,'tcwv_apriori'
                        ,'dp'
                        ,'dp_o2a'
                        #,'dp_sco2'
                        ,'dpfrac'
                        ,'co2_grad_del'
                        ,'dws'
                        #,'snow_flag'
                        ,'aod_dust'
                        # ,'aod_bc'
                        # ,'aod_oc'
                        ,'aod_seasalt'
                        ,'aod_sulfate'
                        ,'aod_strataer'
                        ,'aod_water'
                        ,'aod_ice'
                        #,'aod_total'
                        ,'dust_height'
                        ,'ice_height'
                        ,'water_height'
                        # ,'aod_total_apriori'
                        ,'dws_apriori'
                        # ,'aod_fine_apriori'
                        ,'h2o_scale'
                        ,'deltaT'
                        ,'albedo_o2a'
                        #,'albedo_wco2'
                        ,'albedo_sco2'
                        ,'albedo_slope_o2a'
                        ,'albedo_slope_wco2'
                        ,'albedo_slope_sco2'
                        # ,'albedo_quad_o2a'
                        # ,'albedo_quad_wco2'
                        # ,'albedo_quad_sco2'
                        #,'brdf_weight_slope_wco2'
                        #,'brdf_weight_slope_sco2'
                        #,'chi2_o2a'
                        #,'chi2_wco2'
                        #,'chi2_sco2'
                        ,'rms_rel_o2a'
                        ,'rms_rel_wco2'
                        ,'rms_rel_sco2'
                        # ,'solar_azimuth_angle'
                        # ,'sensor_azimuth_angle'
                        # ,'polarization_angle'
                        #,'land_fraction'
                        ,'glint_angle'
                        ,'airmass'
                        ,'snr_o2a'
                       # ,'snr_wco2'
                        ,'snr_sco2'
                        #,'path'
                        ,'footprint'
                        #,'land_water_indicator'
                        ,'altitude'
                        ,'altitude_stddev']



        # load data
        preload_IO = True
        data = load_and_concat_years(2015, 2021, mode=mode, qf=qf, preload_IO=preload_IO, TCCON = TCCON)


        if var_tp == 'xco2raw_SA_bias':
            data = get_season(data) # get year variable
            data_train = data[data['year'] != year]
            data_test = data[data['year'] == year]
        else:
            # if we use TCCON we split by tccon station not time!
            data_train = data
            data_test = data


        if precorrect_IO:
            print('correcting data with existing model')
            # correct data with an existing model
            # Define the name of the precorrect model (needs to be configured)
            precorrect_model_path = paths.SA_OCN_CORR_MODEL # Get the Path object from paths.py
            path = precorrect_model_path # Use the path directly
            print(f"Using precorrect model path: {path}")
            if not path.exists() or not path.is_dir():
                 raise FileNotFoundError(f"Precorrect model directory not found: {path}")
            data_train = precorrect(path, data_train)
            data_test = precorrect(path, data_test)

        print(str(len(data_train)/1000) + 'k samples loaded')

        if var_tp == 'xco2_TCCON_bias':

            # get data where we have a match with TCCON
            data_train = data_train[data_train['xco2tccon'] > 0]
            data_test = data_test[data_test['xco2tccon'] > 0]
            data_train.loc[:, var_tp] = data_train['xco2_raw'] - data_train['xco2tccon']
            data_test.loc[:, var_tp] = data_test['xco2_raw'] - data_test['xco2tccon']

            # perform k-fold train test split on station names
            # get TCCON station names
            TCCON_names = np.unique(data_train['tccon_name'].to_numpy())

            if mode == 'SeaGL':
                TCCON_keep = ['burgos', 'darwin', 'nyalesund', 'eureka', 'izana', 'lauder', 'reunion', 'rikubetsu',
                              'saga', 'tsukuba', 'wollongong']
                TCCON_names = TCCON_names[np.isin(TCCON_names, TCCON_keep)]

            # remove TCCON stations where we don't have enough measurements
            for t_name in TCCON_names:
                if data_train.loc[data_train['tccon_name'] == t_name, 'xco2tccon'].count() <= 100:
                    print('removing ' + t_name)
                    TCCON_names = TCCON_names[TCCON_names != t_name]


            TCCON_names_test = TCCON_names[i:i + 1]
            TCCON_names_train = np.delete(TCCON_names, i)


            # weight tccon obs equally for each station
            data_train = weight_TCCON(data_train, features)


            # get testing TCCON stations
            data_test = filter_TCCON(data_test, TCCON_names_test)
            # get training TCCON stations
            data_train = filter_TCCON(data_train, TCCON_names_train)


            # save TCCON test stations
            d = {'TCCON_test': TCCON_names_test}
            df = pd.DataFrame(data=d)
            paths.ensure_dir_exists(plot_save_dir)
            df.to_csv(plot_save_dir / 'TC_names.txt', index=False)


        # reduce size of train-set to max_n samples
        if len(data_train) > max_n:
            data_train = data_train.sample(max_n)

        if len(data_test) > max_n//4:
            data_test = data_test.sample(max_n//4)

        print(str(len(data_train) / 1000) + 'k samples downsampled')


        if var_tp in features:
            features.remove(var_tp)

        all_features = features.copy()

        ## start feature elimination
        order = []
        R2 = []

        # set up Random Forest parameters
        M = RandomForestRegressor(n_estimators=100,
                                  max_depth=max_depth,
                                  max_samples=len(data_train) // 2,
                                  min_samples_split=20,
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

        for i in tqdm(range(len(features)-1)):

            X_train, y_train = data_train[features], data_train[var_tp]
            X_test,  y_test  = data_test[features], data_test[var_tp]

            if permutation_importance_IO:
                # train model
                M.fit(X_train, y_train, sample_weight=data_train['weights'].to_numpy())
                # calculate feature importance
                # importance = permutation_importance(M, X_test, y_test, sample_weight=weights_test)
                importance = permutation_importance(M, X_test, y_test, n_repeats=10)
                sorted_idx = importance.importances_mean.argsort()
                # remove least important feature
                id_remove = sorted_idx[0]
                order.append(features[id_remove])
                features.remove(features[id_remove])
            else:
                # calculate reduction in R2 for each feature
                R2_f = []
                # make a copy of X_train and X_test
                X_train_copy = X_train.copy()
                X_test_copy = X_test.copy()
                for f in features:
                    X_train_tmp = X_train_copy.drop(f, axis=1)
                    X_test_tmp = X_test_copy.drop(f, axis=1)
                    M_tmp = M.fit(X_train_tmp, y_train)
                    R2_tmp = M_tmp.score(X_test_tmp, y_test)
                    R2_f.append(R2_tmp)
                sorted_idx = np.argsort(R2_f)
                # remove least important feature
                id_remove = sorted_idx[-1]
                order.append(features[id_remove])
                features.remove(features[id_remove])
                # train model
                M.fit(X_train, y_train)


            # save R2 for visualization
            # R2_train = M.score(X_train, y_train, sample_weight=weights_train)
            # R2_test = M.score(X_test, y_test, sample_weight=weights_test)
            R2_train = M.score(X_train, y_train)
            R2_test = M.score(X_test, y_test)
            print('R2 Train:' + str(R2_train) + ' R2 Test :' + str(R2_test))
            R2.append(R2_test)

        # add remaining feature to order for plotting
        order.append(features[0])
        # plot result
        y = -np.arange(len(R2)+1)
        plt.figure(figsize=(4,11))
        plt.plot(R2+[np.nan], y)
        plt.yticks(y, order)
        plt.grid()
        # plt.title(name)
        if var_tp == 'xco2raw_SA_bias':
            plt.title(str(year))
        else:
            plt.title(str(TCCON_names_test) + mode)
        plt.tight_layout()
        paths.ensure_dir_exists(plot_save_dir)
        plt.savefig(plot_save_dir / ('feature_selection_' + name + mode + '.png'), dpi=300)



print('Done >>>')
