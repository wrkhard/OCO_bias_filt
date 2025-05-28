# Steffen Mauceri
# 03/2023
#
# make various plots to visualize the effect of bias correction and filtering
# Note, this a collection of scripts and code snippets.

# make sure you have all of those packages installed in your conda envirnoment
import json
import pandas as pd
import numpy as np
import os
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import paths # Import the paths module
from util import load_data, plot_tccon, plot_map, get_season, normalize_per_SA, scatter_hist, \
    get_variability_reduction, scatter_density, calc_SA_bias, bias_correct, construct_filter, get_RMSE, load_data_and_concat_years


# make changes #############################
# Bias correction model directory paths (Now using paths from paths.py)
TC_LND_CORR_PATH = paths.TC_LND_CORR_MODEL
TC_OCN_CORR_PATH = paths.TC_OCN_CORR_MODEL
SA_LND_CORR_PATH = paths.SA_LND_CORR_MODEL
SA_OCN_CORR_PATH = paths.SA_OCN_CORR_MODEL

# filter model paths (Now using paths from paths.py)
TC_LND_FILT_PATH = paths.TC_LND_FILTER_MODEL
TC_OCN_FILT_PATH = paths.TC_OCN_FILTER_MODEL
SA_LND_FILT_PATH = paths.SA_LND_FILTER_MODEL
SA_OCN_FILT_PATH = paths.SA_OCN_FILTER_MODEL

path = '/Volumes/OCO/LiteFiles/B11_gamma/'

# Define a specific output path for this analysis script, relative to FIGURE_DIR
analysis_specific_output_path = paths.FIGURE_DIR / 'vis_bias_filt_corr_outputs'

# abstention filtering threshold on bias_correciton_uncert DEFAULT is 1.23 [ppm]
# value greater than 1.23 increases throughput in largely dusty regions such as N. Africa. Smaller values will remove more of the tropics.
ABSTENTION_THRESHOLD_LND = 1.45
ABSTENTION_THRESHOLD_OCN = 0.85

TCCON = True  # use TCCON data only
save_fig = False   # save figures to hard drive
verbose_IO = False
var_tp = 'xco2raw_SA_bias'  # 'xco2_SA_bias', 'xco2_TCCON_bias' or 'xco2raw_SA_bias'  # uncorrected bias based on SA # O'dell bias correction applied -> remaining biases based on SA
max_samples = 10**7
name = 'B11_gamma_TCCON_'  # name of the output files
data_type = 'all' #'train', 'test', 'all', 'model'
holdout_year = 2022
#stop make changes ###########################


paths.ensure_dir_exists(analysis_specific_output_path) # Use paths.ensure_dir_exists


qf = None
mode = 'all'
print(name)


# load data
if data_type == 'train':
    #Train set'
    data = load_data_and_concat_years(2018, 2022, holdout_year=holdout_year, mode=mode, qf=qf, TCCON=TCCON)

elif data_type == 'all':
    #Train+Val+Test set
    data = load_data_and_concat_years(2016, 2022, mode=mode, qf=qf, TCCON=TCCON)

elif data_type == 'test':
    #Test set
    data = load_data(holdout_year, mode, qf=qf, TCCON=TCCON)

elif data_type == 'model':
    #Test set
    data = load_data(2019, mode, qf=qf)
    data1 = load_data(2020, mode,qf=qf)
    data = pd.concat([data, data1])
else:
    print('wrong data type')


if var_tp == 'xco2_TCCON_bias':
    data = data[data['xco2tccon'] > 0]
    data.loc[:, var_tp] = data['xco2_raw'] - data['xco2tccon']

if len(data) > max_samples:
    data = data.sample(max_samples, replace=False, )

print(str(len(data)/1000) + 'k samples loaded')

# make a copy of the orignial raw xco2 for later plots and analysis
data.loc[:,'xco2_raw_orig'] = data.loc[:,'xco2_raw'].copy()

# perfrom bias correction
print('Performing bias correction ... ')
# split data into land and sea
data_lnd = data[(data['land_water_indicator'] == 0) | (data['land_water_indicator'] == 3)]
data_sea = data[(data['land_water_indicator'] == 1)]
# perform bias correction on land and sea
if len(data_lnd) > 0:
    data_lnd = bias_correct(TC_LND_CORR_PATH, data_lnd, ['xco2_raw'], uq=True)
    data_lnd = bias_correct(SA_LND_CORR_PATH, data_lnd, ['xco2_raw'], uq=False)
if len(data_sea) > 0:
    data_sea = bias_correct(TC_OCN_CORR_PATH, data_sea, ['xco2_raw'], uq=True)
    data_sea = bias_correct(SA_OCN_CORR_PATH, data_sea, ['xco2_raw'], uq=False)

    # adjust sea based on land-ocean crossings
    data_sea['xco2_raw'] = data_sea['xco2_raw'] - 0.15

# combine data
if len(data_lnd) > 0 and len(data_sea) > 0:
    data = pd.concat([data_lnd, data_sea])
elif len(data_lnd) > 0:
    data = data_lnd
elif len(data_sea) > 0:
    data = data_sea
# sort data by sounding id
data.sort_values('sounding_id', inplace=True)

#  add ternary quality flag
#  path dict and abstention threshold
kwargs = {
    'path_tc_lnd': TC_LND_FILT_PATH,
    'path_tc_ocn': TC_OCN_FILT_PATH,
    'path_sa_lnd': SA_LND_FILT_PATH,
    'path_sa_ocn': SA_OCN_FILT_PATH,
    'abstention_threshold_lnd': ABSTENTION_THRESHOLD_LND,
    'abstention_threshold_ocn': ABSTENTION_THRESHOLD_OCN}
#TODO: remove this line for next update
data['h2o_ratio_bc'] = data['h2o_ratio']
data['co2_ratio_bc'] = data['co2_ratio']

data = construct_filter(data, **kwargs)


# recalculate small area bias
# sort data by SA
# data.sort_values('SA', inplace=True)
# XCO2 = data['xco2_raw'].to_numpy()
# SA = data['SA'].to_numpy()
# data.loc[:, 'xco2raw_SA_bias-ML'] = calc_SA_bias(XCO2, SA)

data.loc[:,'xco2MLcorr'] = data.loc[:,'xco2_raw']
data.loc[:,'xco2_raw'] = data.loc[:,'xco2_raw_orig']
data.loc[:,'xco2MLbias'] = data.loc[:,'xco2_raw_orig'] - data.loc[:,'xco2MLcorr']


#****         show how model works          ************************************************************
# show how much of the correction comes from TCCON vs SA
data.loc[:,'MLcorr_ratio'] = ((data.loc[:,'xco2MLcorr1'] - data.loc[:,'xco2_raw_orig']) / (data.loc[:,'xco2MLcorr'] - data.loc[:,'xco2_raw_orig'])).abs()
plot_map(data, ['MLcorr_ratio'], save_fig=save_fig, path=analysis_specific_output_path, name=name + '_ML_ratio', pos_neg_IO=False, min=0, max=1)




#****           compare to TCCON            ************************************************************
#
# get TCCON station names
TCCON_names = np.unique(data['tccon_name'].to_numpy())

# remove pasadena and xianghe
TCCON_names = TCCON_names[TCCON_names != '']
# TCCON_names = TCCON_names[TCCON_names != 'pasadena01']
# TCCON_names = TCCON_names[TCCON_names != 'xianghe01']




data_T = data[data['xco2tccon'] > 0].copy()
data_T['xco2-TCCON'] = data_T['xco2MLcorr'] - data_T['xco2tccon']
data_T = data_T.loc[data_T['xco2_quality_flag'] == 0]

data_T_lnd = data_T.loc[data_T['land_fraction'] == 100]
data_T_sea = data_T.loc[data_T['land_fraction'] == 0]

# remove tccon stations from data_T_sea that are not near the ocean
TCCON_keep = ['burgos', 'darwin', 'nyalesund', 'eureka', 'izana', 'lauder', 'reunion', 'rikubetsu',
              'saga', 'tsukuba', 'wollongong']
data_T_sea = data_T_sea[data_T_sea['tccon_name'].isin(TCCON_keep)]


# calculate average error for each TCCON station
lnd_TCCON_error = data_T_lnd.groupby('tccon_name')['xco2-TCCON'].agg(['std', 'median'])
sea_TCCON_error = data_T_sea.groupby('tccon_name')['xco2-TCCON'].agg(['std', 'median'])

# calculate std of all medians
lnd_TCCON_std = np.std(lnd_TCCON_error['median'])
sea_TCCON_std = np.std(sea_TCCON_error['median'])
# print out
print('Land TCCON std: ' + str(lnd_TCCON_std))
print('Sea TCCON std: ' + str(sea_TCCON_std))








#compare new vs old QF
data_all = data
name_all = name
for qf in [0, 1]:
    print('making plots for QF=' + str(qf))
    for qf_type in ['ML', 'B11']:
        if qf_type == 'ML':
            data = data_all.loc[data_all['xco2_MLquality_flag'] == qf]
            name = name_all + '_MLQF' + str(qf)
        else:
            data = data_all.loc[data_all['xco2_quality_flag'] == qf]
            name = name_all + '_QF' + str(qf)

        xco2ML_std, xco2ML_median, xco2B11_std, xco2B11_median, xco2raw_std, xco2raw_median, xco2ML_RMSE, xco2B11_RMSE,xco2raw_RMSE = plot_tccon(data, TCCON_names, save_fig=save_fig, path=analysis_specific_output_path, name=name, qf=qf)


# # write out to file
# d = {'xco2ML_std': xco2ML_std,
#      'xco2ML_mean': xco2ML_median,
#      'xco2B11_std': xco2B11_std,
#      'xco2B11_mean': xco2B11_median}
# df = pd.DataFrame(data=d, index=[0])
# df.to_csv(analysis_specific_output_path + name + '_TCCONerror.txt', index=False, float_format='%.6f')

#
# ****      model interpretation       **************************************************************************

# for var in features:
#     scatter_hist(data[var].to_numpy(), data['xco2MLbias'].to_numpy(), var, 'RF Bias [ppm]', name, path,save_IO=save_fig, bias_IO=False)


# explain bias correction
#plot bias correction vs each variable with other variables varying naturally
# load features of model

# if precorrect_IO:
#     path = path2
# else:
#     path = path1
# with open(paths.MODEL_SAVE_DIR / path / 'normalization_params.json', 'r') as f: # Assuming path1/path2 are subdirs in MODEL_SAVE_DIR
#     norm_params = json.load(f)
# features = norm_params['features']
# model = norm_params['model_type']

# for var in features:
#     print(var)
#     scatter_hist(data[var].to_numpy(), data['xco2MLbias'].to_numpy(), var, 'xco2 ML bias [ppm]', name + model, analysis_specific_output_path, save_IO=save_fig, bias_IO=True)

# plot bias correction vs each variable with other variables set to their mean
# X = data
# X_mean = X.mean()
# for var in features:
#     print(var)
#     steps = 10000
#     # make Df with mean of features
#     X_i = X.iloc[:steps].copy()
#     X_i[features] = X_mean[features]
#     X_i[var] = np.linspace(X[var].quantile(0.01), X[var].quantile(0.99), num=steps)
#     # make prediction while keeping other vars at mean
#     X_i = bias_correct(path1, X_i, ['xco2_raw'])
#     if precorrect_IO:
#         X_i = bias_correct(path2, X_i, ['xco2_raw'])
#     X_i.loc[:, 'xco2MLbias'] = X_i.loc[:, 'xco2_raw_orig'] - X_i.loc[:, 'xco2_raw']
#
#     scatter_hist(X_i[var].to_numpy(), X_i['xco2MLbias'].to_numpy(), var, 'xco2 ML bias [ppm]', name + model + 'mean', analysis_specific_output_path, save_IO=save_fig, bias_IO=True)





# show how all model perform for individual viewing mode
# name_all = name
# if mode == 'all':
#     for m in ['LndGL', 'LndND', 'SeaGL']:
#         print(m)
#         if m == 'LndND':
#             print('removing ocean')
#             d_m = data.loc[data['land_fraction'] == 100, :]
#             d_m = d_m.loc[d_m['sensor_zenith_angle'] < 5, :]
#         elif m == 'LndGL':
#             print('removing ocean')
#             d_m = data.loc[data['land_fraction'] == 100, :]
#             d_m = d_m.loc[d_m['sensor_zenith_angle'] > 5, :]
#         elif m == 'SeaGL':
#             print('removing land')
#             d_m = data.loc[data['land_fraction'] == 0, :]
#             d_m = d_m.loc[d_m['sensor_zenith_angle'] > 5, :]
#
#
#         d_m_all = d_m
#         for qf in [0,1]:
#             print('making plots for QF=' + str(qf))
#             d_m = d_m_all.loc[d_m_all['xco2_quality_flag'] == qf]
#             name = name_all + '_' +m+ '_QF' + str(qf)
            # comparison to TCCON
            # _, _, _, _, _, _, _, _, _ = plot_tccon(d_m, TCCON_names, save_fig=save_fig, path=path2, name=name, qf=qf)
            # # SA variability reduction histogram
            # get_variability_reduction(d_m, var_tp, name, path2, save_fig=save_fig, qf=qf)
            #
            # # calculate RMSE before and after bias correction
            # print('get RMSE reduction')
            # RMSE_Raw = get_RMSE(d_m['xco2raw_SA_bias'].to_numpy())
            # RMSE_ML = get_RMSE(d_m['xco2raw_SA_bias-ML'].to_numpy())
            # RMSE_B11 = get_RMSE(d_m['xco2_SA_bias'].to_numpy())
            #
            # # write results out to file
            # d = {'RMSE_Raw_SA': RMSE_Raw,
            #      'RMSE_ML_SA': RMSE_ML,
            #      'RMSE_B11_SA': RMSE_B11
            #      }
            # df = pd.DataFrame(data=d, index=[0])
            # df.to_csv(path2 + name + '_error.txt', index=False, float_format='%.4f')

#
# # show how model performs over snow
# print('get variability over snow')
# d_s = data.loc[data['snow_flag'] == 1]  # only keep snow
# get_variability_reduction(d_s, var_tp, name + '_snow', path2, save_fig=save_fig)
# d_s = data.loc[data['snow_flag'] != 1]  # only keep snow
# get_variability_reduction(d_s, var_tp, name + '_NOsnow', path2, save_fig=save_fig)




# #****           Comparison to Models            ***************************************************
# show comparison to Models [2019 - 2020]
# models = ['CT_2019B+NRT2022-1', 'Jena_s10oc-v2021', 'LoFI_m2ccv1sim', 'MACC_v20r2', 'UnivEd_v5']
# # select data for model comparison
# for qf in [0, 1]:
#     if qf == 0:
#         d_m = data.loc[(data[models[0]] > 0) & (data['xco2_quality_flag'] == 0),:]
#     elif qf == 1:
#         d_m = data.loc[(data[models[0]] > 0) & (data['xco2_quality_flag'] == 1), :]
#     else:
#         d_m = data.loc[(data[models[0]] > 0), :]
#     # calc model mean
#     d_m.loc[:,'Model_mean'] = d_m.loc[:, 'CT_2019B+NRT2022-1':'UnivEd_v5'].mean(axis=1)
#     d_m.loc[:,'Model_std'] = d_m.loc[:, 'CT_2019B+NRT2022-1':'UnivEd_v5'].std(axis=1)
#     models.append('Model_mean')
#     # calc differences
#     for m in models:
#         d_m.loc[:,'B11-' + m] = d_m.loc[:,'xco2'] - d_m.loc[:,m]
#         d_m.loc[:,'ML-'+ m ] = d_m.loc[:,'xco2MLcorr'] - d_m.loc[:,m]
#
#     #quantify differences to models
#     M_std = d_m.loc[:,['ML-Model_mean', 'B11-Model_mean', 'B11-CT_2019B+NRT2022-1', 'B11-Jena_s10oc-v2021',
#                        'B11-LoFI_m2ccv1sim', 'B11-MACC_v20r2', 'B11-UnivEd_v5', 'ML-CT_2019B+NRT2022-1',
#                        'ML-Jena_s10oc-v2021', 'ML-LoFI_m2ccv1sim', 'ML-MACC_v20r2', 'ML-UnivEd_v5']].std()
#     M_mean = d_m.loc[:,['ML-Model_mean', 'B11-Model_mean', 'B11-CT_2019B+NRT2022-1', 'B11-Jena_s10oc-v2021',
#                         'B11-LoFI_m2ccv1sim', 'B11-MACC_v20r2', 'B11-UnivEd_v5', 'ML-CT_2019B+NRT2022-1',
#                         'ML-Jena_s10oc-v2021', 'ML-LoFI_m2ccv1sim', 'ML-MACC_v20r2', 'ML-UnivEd_v5']].mean()
#     M = pd.concat([M_mean, M_std], axis=1)
#     M.rename(columns={0:'Mean', 1:'Std'}, inplace=True)
#     M.to_csv(path + name +  '_diff_M_QF' + str(qf) + '.csv')
#
#     modelsB11 = models.copy()
#     modelsML = models.copy()
#     for i in range(len(models)):
#         modelsB11[i] = 'B11-' + models[i]
#         modelsML[i] = 'ML-' + models[i]
#     # plot differences on map
#     # plot_map(d_m, modelsB11, save_fig=save_fig, path=path2, name=name + '_B11_diff_M', pos_neg_IO=True)
#     # plot_map(d_m, modelsML, save_fig=save_fig, path=path, name=name + '_ML_diff_M', pos_neg_IO=True)
#     plot_map(d_m, ['ML-Model_mean', 'B11-Model_mean'], save_fig=save_fig, path=path, name=name + '_diff_M_QF' + str(qf), pos_neg_IO=True, min=-2, max=2)
#
#
# # #
#     # plot difference vs std
#     d_m.loc[:,'B11-Model_mean_std'] = np.abs(d_m['xco2'] - d_m['Model_mean'])/d_m['Model_std']
#     d_m.loc[:,'ML-Model_mean_std'] = np.abs(d_m['xco2MLcorr'] - d_m['Model_mean'])/d_m['Model_std']
#     plot_map(d_m, ['B11-Model_mean_std', 'ML-Model_mean_std'], save_fig=save_fig, path=path, name=name + '_ratio_M_QF' + str(qf), pos_neg_IO=False, min=1, max=5)
#
#     # plot differences for individual seasons
#     # d_m = get_season(d_m)
#     # for season in ['MAM', 'JJA', 'SON', 'DJF']:
#     #     plot_map(d_m[d_m['season'] == season], ['B11-Model_mean', 'ML-Model_mean'], save_fig=save_fig, path=path, name=name + 'Model_mean_diff_QF' + str(qf) + season, pos_neg_IO=True)
#
#
# # #****           Visualize biases at land water crossings            ***************************************************
# # identify small areas that cross from ocean to land
#
# for qf in [0,1]:
#     d = data.loc[data['coast'] == 1, :]
#     d = d.loc[d['xco2_quality_flag'] == qf]
#     # for each small area calculate bias as if ocean would be the truth (difference to ocean)
#     SAs = pd.unique(d['SA'])
#     coast_bias_ML = np.zeros_like(d['SA']) * np.nan
#     coast_bias_B11 = np.zeros_like(d['SA'])* np.nan
#
#     for SA in SAs:
#         # find soundings that belong to SA
#         id = d['SA'] == SA
#         d_SA = d.loc[id]
#         d_SA_sea = d_SA.loc[d_SA['land_fraction'] == 0, :]
#         d_SA_sea = d_SA_sea.loc[d_SA_sea['sensor_zenith_angle'] > 5, :]
#         d_SA_lnd = d_SA.loc[d_SA['land_fraction'] == 100, :]
#         # check that we have enough soundings for a robust bias estimation
#         if (len(d_SA_lnd) >= 5) & (len(d_SA_sea) >= 5):
#             # calc median sea and lnd XCO2
#             coast_bias_B11[id] = d_SA_lnd['xco2'].median() - d_SA_sea['xco2'].median()
#             coast_bias_ML[id] = d_SA_lnd['xco2MLcorr'].median() - d_SA_sea['xco2MLcorr'].median()
#     # add bias to dataframe
#     d['coast_bias_B11'] = coast_bias_B11
#     d['coast_bias_ML'] = coast_bias_ML
#
#     # plot histogram
#     plt.figure(figsize=(8, 4))
#     bins = np.arange(np.nanpercentile(coast_bias_B11, 2), np.nanpercentile(coast_bias_B11, 98), 0.1)
#     n = plt.hist(coast_bias_ML, bins=bins, label='OCO-2 corr.', histtype='step', color='k')
#     plt.hist(coast_bias_B11, bins=bins, label='OCO-2 B11 ', histtype='step', color='r')
#
#     plt.vlines(np.median(coast_bias_ML), 0, np.max(n[0]), colors='k',
#                label='OCO-2 corr. = ' + str(np.round(np.nanmedian(coast_bias_ML), 2)) + r'$\pm$' + str(np.round(np.nanstd(coast_bias_ML), 2)))
#     plt.vlines(np.median(coast_bias_B11), 0, np.max(n[0]), colors='r',
#                label='OCO-2 B11 = ' + str(np.round(np.nanmedian(coast_bias_B11), 2)) + r'$\pm$' + str(np.round(np.nanstd(coast_bias_B11), 2)))
#
#     plt.title('OCO-2 lnd - sea QF=' + str(qf))
#     plt.xlabel('XCO2 [ppm]')
#     plt.ylabel('#soundings')
#     plt.legend()
#     plt.tight_layout()
#     if save_fig:
#         plt.savefig(analysis_specific_output_path / (f'lnd-sea_hist_QF={str(qf)}{name}.png'))
#     else:
#         plt.show()
#
#     # plot by latitude
#     # aggregate by latitude
#     coast_bias_B11_l = []
#     coast_bias_ML_l = []
#     step = 5
#     lats = np.arange(-60,60+step,step)
#     for l in lats:
#         coast_bias_B11_l.append(np.nanmedian(d['coast_bias_B11'].loc[(d['latitude'] >= l) & (d['latitude'] <= l+step)]))
#         coast_bias_ML_l.append(np.nanmedian(d['coast_bias_ML'].loc[(d['latitude'] >= l) & (d['latitude'] <= l + step)]))
#
#     plt.figure(figsize=(4, 6))
#     plt.title('OCO-2 lnd - sea QF=' + str(qf))
#     # plt.plot(coast_bias_B11_l, lats, label='OCO-2 B11')
#     # plt.plot(coast_bias_ML_l,lats,  label='OCO-2 corr.')
#     # plt.vlines(0, -60, 60)
#     plt.fill_betweenx(lats,  coast_bias_B11_l, label='OCO-2 B11', alpha=0.5)
#     plt.fill_betweenx(lats,  coast_bias_ML_l,  label='OCO-2 corr.', alpha=0.5)
#     # plt.barh(lats-1, coast_bias_B11_l,2.0, label='OCO-2 B11')
#     # plt.barh(lats+1, coast_bias_ML_l,2.0, label='OCO-2 corr.')
#     plt.xlabel('XCO2 lnd - sea [ppm]')
#     plt.ylabel('Latitude [$^\circ$]')
#     plt.legend()
#     plt.tight_layout()
#     if save_fig:
#         plt.savefig(analysis_specific_output_path / (f'lnd-sea_Lat QF={str(qf)}{name}.png'))
#     else:
#         plt.show()
#
#     # plot on a map
#     d_clean = d[d['coast_bias_B11'].notna()]
#     plot_map(d_clean, ['coast_bias_B11', 'coast_bias_ML'], save_fig=save_fig, path=path,
#              name=name + 'Coast_diff_QF' + str(qf), pos_neg_IO=True, min=-1, max=1)





# #****           Visualize footprint biases             ***************************************************
# itterate over footprints and calculate bias to median
# SAs = pd.unique(data['SA'])
# find soundings that belong to SA
#         id = d['SA'] == SA
#         d_SA = d.loc[id]
# bias_f = []
# for SA in SAs:
#     d_SA = data.loc[data['SA'] == SA,:]
#     #check that we have at least each footprint once in SA
#     footprints = pd.unique(d_SA['footprint'])
#     if len(footprints) == 8:
#         bias_f_SA = np.zeros((8, 2)) * np.nan
#         for f in np.arange(1,9):
#             bias_f_SA[f-1,0] = d_SA.loc[d_SA['footprint'] == f,'xco2MLcorr'].mean() - d_SA.loc[:,'xco2MLcorr'].mean()
#             bias_f_SA[f-1,1] = d_SA.loc[d_SA['footprint'] == f, 'xco2'].mean() - d_SA.loc[:,'xco2'].mean()
#         bias_f.append(bias_f_SA)
# bias_f = np.stack(bias_f,0)
#
# #visualiz mean bias
# bias_f_mean = np.nanmean(bias_f,0)
# # Create the figure and axis
# fig, ax = plt.subplots()
# # Create the bar plot
# ax.bar(np.arange(1, 9)-0.2, bias_f_mean[:,0],width=0.4, label='OCO-2 corr.')
# ax.bar(np.arange(1, 9)+0.2, bias_f_mean[:,1],width=0.4, label='B11')
# # Set the x-axis label
# ax.set_xlabel('footprint')
# # Set the y-axis label
# ax.set_ylabel('XCO2 offset [ppm]')
# # Add a horizontal line at y=0
# ax.axhline(y=0, color='k', linestyle='--')
# # Add a legend
# ax.legend()
#
# plt.tight_layout()
# if save_fig:
#     plt.savefig(analysis_specific_output_path / (f'Footprint_offset QF={str(qf)}{name}.png'))
# else:
#     plt.show()
#
#
# #visualiz RMSE
# bias_f_RMSE = np.nanmean(bias_f**2,0)**0.5
# # Create the figure and axis
# fig, ax = plt.subplots()
# # Create the bar plot
# ax.bar(np.arange(1, 9)-0.2, bias_f_RMSE[:,0],width=0.4, label='OCO-2 corr.')
# ax.bar(np.arange(1, 9)+0.2, bias_f_RMSE[:,1],width=0.4, label='B11')
# # Set the x-axis label
# ax.set_xlabel('footprint')
# # Set the y-axis label
# ax.set_ylabel('XCO2 RMSE [ppm]')
# # Add a legend
# ax.legend()
#
# plt.tight_layout()
# if save_fig:
#     plt.savefig(analysis_specific_output_path / (f'Footprint_RMSE QF={str(qf)}{name}.png'))
# else:
#     plt.show()

print('Done >>>')