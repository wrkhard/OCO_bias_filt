# Steffen Mauceri
# 08/2020
#
# make various plots to visualize bias correction
# Note, this a collection of scripts and code snippets.

# make sure you have all of those packages installed in your conda envirnoment
import pandas as pd
import numpy as np
from pathlib import Path # Import Path
import json # Import json for loading model parameters

import matplotlib.pyplot as plt
import seaborn as sns
import paths

from util import load_data, plot_tccon, plot_map, get_season, scatter_hist, plot_histogram, \
    get_variability_reduction,  bias_correct, get_RMSE, load_and_concat_years


# make changes #############################
# Default model paths from paths.py (can be overridden locally if needed)
# if we don't use split_lnd_sea_IO
path1 = paths.TC_LND_CORR_MODEL
path2 = paths.SA_LND_CORR_MODEL 

# if we use split_lnd_sea_IO
path_lnd_1 = paths.TC_LND_CORR_MODEL
path_sea_1 = paths.TC_OCN_CORR_MODEL
path_lnd_2 = paths.SA_LND_CORR_MODEL
path_sea_2 = paths.SA_OCN_CORR_MODEL

precorrect_IO = True    # Correct data with two models, first the one in path1 then the one in path2
split_lnd_sea_IO = False# split data into land and sea and correct seperately

save_fig = True   # save figures to hard drive
verbose_IO = False
var_tp = 'xco2raw_SA_bias' # 'xco2_SA_bias', 'xco2_TCCON_bias' or 'xco2raw_SA_bias'
max_samples = 2 * 10**7
name = ''
UQ = False # very slow. Consider reducing max_samples to 5 * 10**6
TCCON = False
RIW = False # remove inland water
data_type = 'all' #'train', 'test', 'all'
hold_out_year = 2022  # Year for 'test' data_type
#stop make changes ###########################

# Determine output directory based on paths used
if precorrect_IO:
    if split_lnd_sea_IO:
        # Create a specific directory name for combined land/sea models
        output_base_path = paths.MODEL_SAVE_DIR / (path2.name + '_combined') / data_type
    else:
        # Use the directory of the second model if correcting with two
        output_base_path = path2 / data_type # path2 is already a Path
else:
    # Use the directory of the first model if only using one
    output_base_path = path1 / data_type # path1 is already a Path

paths.ensure_dir_exists(output_base_path)
path = output_base_path # Use this as the main output path

# make extra folder for TCCON plots
path_TCCON = path / 'TCCON' # Use Path object
paths.ensure_dir_exists(path_TCCON)

name = path.parent.name + name # Get name from parent directory of the output path


params_json_path = path1 / 'normalization_params.json'

if not split_lnd_sea_IO:
    if not params_json_path.exists():
        raise FileNotFoundError(f"Normalization parameters file not found: {params_json_path}")
    with open(params_json_path, 'r') as f:
        p = json.load(f)
    features = p['features']
    mode = p['mode'] 
else:
    # When splitting land/sea, qf and mode might be common or need specific handling.
    # For now, let's try to load from path_lnd_1 and assume others are consistent or handled by bias_correct.
    # This part might need refinement based on how these are set for split models.
    params_lnd_1_json_path = path_lnd_1 / 'normalization_params.json'
    if not params_lnd_1_json_path.exists():
        raise FileNotFoundError(f"Normalization parameters file not found: {params_lnd_1_json_path}")
    with open(params_lnd_1_json_path, 'r') as f:
        p_lnd_1 = json.load(f)
    features = None # Features will be collected from all models later
    mode = 'all' # Mode is set to all when splitting, data loading handles specifics


print(f"Run Name: {name}")
print(f"Output Path: {path}")
print(f"Mode: {mode}")

# Define common arguments for data loading
load_args = dict(TCCON=TCCON, remove_inland_water=RIW, preload_IO=True)

# load data using the new function or single load_data call
if data_type == 'train':
    # Load training data (hold out the test year)
    data = load_and_concat_years(2015, 2021, mode=mode, hold_out_year=hold_out_year, **load_args)

elif data_type == 'all':
    # Load all data (no hold out year)
    data = load_and_concat_years(2015, 2023, mode=mode, **load_args)

elif data_type == 'test':
    # Load only the test year
    data = load_data(hold_out_year, mode=mode, **load_args)
else:
    raise ValueError('Error: Invalid data_type specified.')

if data.empty:
    raise ValueError("No data loaded. Exiting.")

if var_tp == 'xco2_TCCON_bias':
    data = data[data['xco2tccon'] > 0]
    data.loc[:, var_tp] = data['xco2_raw'] - data['xco2tccon']


if len(data) > max_samples:
    data = data.sample(int(max_samples), replace=False, random_state=1)

print(str(len(data)/1000) + 'k samples loaded')



# make a copy of the orignial raw xco2 for later plots and analysis
data.loc[:,'xco2_raw_orig'] = data.loc[:,'xco2_raw'].copy()

if precorrect_IO: # if we use the two step bias correction approach
    if split_lnd_sea_IO: # if we further have a model for land and one for ocean

        data_lnd = data[(data['land_water_indicator'] == 0) | (data['land_water_indicator'] == 3)]
        data_sea = data[(data['land_water_indicator'] == 1)]

        print('correcting TCCON Lnd ...')
        data_lnd = bias_correct(path_lnd_1, data_lnd, ['xco2_raw'], uq=UQ)
        data_lnd = bias_correct(path_lnd_2, data_lnd, ['xco2_raw'])
        print('correcting TCCON Sea ...')
        data_sea = bias_correct(path_sea_1, data_sea, ['xco2_raw'], uq=UQ)
        data_sea = bias_correct(path_sea_2, data_sea, ['xco2_raw'])

        # value needs to be adjusted based on land - ocean crossings on training set
        # adjust ocean XCO2 to match land using land - ocean crossings
        #11.1 value for publication: -0.2
        data_sea['xco2_raw'] = data_sea['xco2_raw'] - 0.2  #is the median bias of land - ocean crossings derived from sea-lnd_hist_QF0 on training set


        data = pd.concat([data_lnd, data_sea])

        # get all the features - Load features from each model file
        features_list = []
        for model_path_dir in [path_lnd_1, path_lnd_2, path_sea_1, path_sea_2]: # model_path is already a directory
            params_file = model_path_dir / 'normalization_params.json' # New json path
            
            if params_file.exists():
                with open(params_file, 'r') as f:
                    p_params = json.load(f)
                p_features = p_params['features']
                features_list.extend(p_features)
            else:
                raise FileNotFoundError(f"Model parameters file not found at {params_file}, cannot load features.")

        features = list(set(features_list)) # Get unique features across all models


    else:
        # perfrom bias correction
        data = bias_correct(path1, data, ['xco2_raw'])
        data.loc[:,'xco2MLcorr1'] = data.loc[:,'xco2_raw'].copy()
        data = bias_correct(path2, data, ['xco2_raw'])
else:
    # perfrom bias correction
    data = bias_correct(path1, data, ['xco2_raw'])

# recalculate small area bias
# sort data by SA
# data.sort_values('SA', inplace=True)
# XCO2 = data['xco2_raw'].to_numpy()
# SA = data['SA'].to_numpy()
# data.loc[:, 'xco2raw_SA_bias-ML'] = calc_SA_bias(XCO2, SA)

data['xco2MLcorr'] = data['xco2_raw']
data['xco2_raw'] = data['xco2_raw_orig']
data['xco2MLbias'] = data['xco2_raw_orig'] - data['xco2MLcorr']


# save data to csv
# data.to_csv(path / (name + '_data.csv'), index=False, float_format='%.6f') # Use Path object - Consider if saving this large file is needed


# #****         show how model works          ************************************************************
# # show how much of the correction comes from TCCON vs SA
# # data.loc[:,'MLcorr_ratio'] = ((data.loc[:,'xco2MLcorr1'] - data.loc[:,'xco2_raw_orig']) / (data.loc[:,'xco2MLcorr'] - data.loc[:,'xco2_raw_orig'])).abs()
# # plot_map(data, ['MLcorr_ratio'], save_fig=save_fig, path=path, name=name + '_ML_ratio', pos_neg_IO=False, min=0, max=1)

#****         show Uncertainties in model prediction   ****************************************************
if UQ:
    plot_map(data, ['bias_correction_uncert'], save_fig=save_fig, path=path, name=name + '_UQ', pos_neg_IO=False)

#****         compare ML to B11 and Raw          ************************************************************
print('Compare to B11 and Raw ...')
data.loc[:,'ML-Raw'] = data.loc[:,'xco2MLcorr'] - data.loc[:,'xco2_raw_orig']
data.loc[:,'ML-B11'] = data.loc[:,'xco2MLcorr'] - data.loc[:,'xco2']
data.loc[:,'B11-Raw'] = data.loc[:,'xco2'] - data.loc[:,'xco2_raw_orig'] # compare B11 to Raw (for comparison)



data_all = data.copy()
for qf in [0, 1]:
    data = data_all.loc[data_all['xco2_quality_flag'] == qf]
    name_i = name + '_QF' + str(qf)
    plot_map(data, ['ML-Raw', 'ML-B11', 'B11-Raw'], save_fig=save_fig, path=path, name=name_i , pos_neg_IO=True, min=-1, max=1)

# compare ML-B11 for land and sea
plot_histogram(
    data_all[data_all['land_water_indicator'] == 0],
    column='ML-B11',
    bins=np.linspace(-1.5, 1.5, 50),
    save_fig=save_fig,
    path=path,
    name=name + '_lnd',
    xlabel='OCO-2 ML - Operational [ppm]',
    ylabel='Soundings',
    title='OCO-2 ML - Operational over Land'
)

plot_histogram(
    data_all[data_all['land_water_indicator'] == 1],
    bins=np.linspace(-1.5, 1.5, 50),
    column='ML-B11',
    save_fig=save_fig,
    path=path,
    name=name + '_sea',
    xlabel='OCO-2 ML - Operational [ppm]',
    ylabel='Soundings',
    title='OCO-2 ML - Operational over Ocean'
)

data = data_all.copy()



#****           compare to TCCON            ************************************************************
print('Compare to TCCON ...')
# # get TCCON station names
TCCON_names = np.unique(data['tccon_name'].to_numpy())
# remove empty string
TCCON_names = TCCON_names[TCCON_names != '']



name_all = name
for qf in [0,1]:
    print('making plots for QF=' + str(qf))
    data = data_all.loc[data_all['xco2_quality_flag'] == qf]
    name = name_all + '_QF' + str(qf) #+ 'TC_' + t_name

    if mode == 'SeaGL':
        # remove tccon stations from data_T_sea that are not near the ocean
        TCCON_keep = ['burgos', 'darwin', 'nyalesund', 'eureka', 'izana', 'lauder', 'reunion', 'rikubetsu',
                        'saga', 'tsukuba', 'wollongong']
        data = data[data['tccon_name'].isin(TCCON_keep)]

    xco2ML_std, xco2ML_median, xco2B11_std, xco2B11_median, xco2raw_std, xco2raw_median, xco2ML_RMSE, xco2B11_RMSE,xco2raw_RMSE = plot_tccon(data, TCCON_names, save_fig=save_fig, path=path, name=name, qf=qf)

data = data_all.copy()

# write out to file
d = {'xco2ML_std': xco2ML_std,
        'xco2ML_mean': xco2ML_median,
        'xco2B10_std': xco2B11_std,
        'xco2B10_mean': xco2B11_median}
df = pd.DataFrame(data=d, index=[0])
df.to_csv(path / (name + '_TCCONerror.txt'), index=False, float_format='%.6f') # Use Path object


# make a plot for each TCCON station as box and whisker plot ****************************************************
# get soundings where we have TCCON data
data_T = data[data['xco2tccon'] > 0].copy()
data_T['xco2-TCCON'] = data_T['xco2MLcorr'] - data_T['xco2tccon']
data_T = data_T.loc[data_T['xco2_quality_flag'] == 0]

data_T_lnd = data_T.loc[data_T['land_fraction'] == 100]
data_T_sea = data_T.loc[data_T['land_fraction'] == 0]

# remove tccon stations from data_T_sea that are not near the ocean
TCCON_keep = ['burgos', 'darwin', 'nyalesund', 'eureka', 'izana', 'lauder', 'reunion', 'rikubetsu',
                'saga', 'tsukuba', 'wollongong']
data_T_sea = data_T_sea[data_T_sea['tccon_name'].isin(TCCON_keep)]

# sort data by 'latitude'
data_T_lnd.sort_values('latitude', inplace=True)
data_T_sea.sort_values('latitude', inplace=True)

try:
    plt.figure(figsize=(10, 4))
    sns.boxplot(x='tccon_name', y='xco2-TCCON', data=data_T_lnd, width=0.5, showfliers=False, color='limegreen')
    plt.legend().remove()
    plt.title('OCO-2 bias by TCCON Station over Land')
    plt.xlabel('TCCON Station Names')
    plt.ylabel('OCO-2 - TCCON [ppm]')
    plt.xticks(rotation=90)
    plt.ylim(-5, 5)
    plt.grid(axis='y')
    plt.tight_layout()
    # plt.savefig(path_TCCON +name+ 'tccon_bias_lnd_boxplot.png', dpi=300)
    plt.savefig(path_TCCON / (name + 'tccon_bias_lnd_boxplot.png'), dpi=300) # Use Path object
    plt.close()
except Exception as e:
    print('no land data')

try:
    plt.figure(figsize=(4, 4))
    sns.boxplot(x='tccon_name', y='xco2-TCCON', data=data_T_sea, width=0.5, showfliers=False, color='skyblue')
    plt.legend().remove()
    plt.title('OCO-2 bias by TCCON Station over Ocean')
    plt.xlabel('TCCON Station Names')
    plt.ylabel('OCO-2 - TCCON [ppm]')
    plt.xticks(rotation=90)
    plt.ylim(-5, 5)
    plt.grid(axis='y')
    plt.tight_layout()
    # plt.savefig(path_TCCON +name+ 'tccon_bias_sea_boxplot.png', dpi=300)
    plt.savefig(path_TCCON / (name + 'tccon_bias_sea_boxplot.png'), dpi=300) # Use Path object
    plt.close()
except Exception as e:
    print('no sea data')



# #****           Visualize TCCON bias vs Latitude             ***************************************************
print('Visualize TCCON bias vs Latitude ...')
tccon_median = data_T.groupby('tccon_name').median()

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
    plt.title('OCO-2 - TCCON [ppm]')
    plt.tight_layout()
    # plt.savefig(path_TCCON + name+ 'tccon_bias_vs_' + var + '.png', dpi=300)
    plt.savefig(path_TCCON / (name + 'tccon_bias_vs_' + var + '.png'), dpi=300) # Use Path object
    plt.close()


# #****           Visualize TCCON bias vs state variables          ***************************************************
vars_to_plot = features + ['latitude', 'airmass']
for var in vars_to_plot:
    scatter_hist(data_T[var].to_numpy(), data_T['xco2-TCCON'].to_numpy(), var, 'OCO-2 - TCCON [ppm]', name, path_TCCON,save_IO=save_fig, bias_IO=True)

# # ****      model interpretation       **************************************************************************
print('model interpretation ...')
for var in features:
    scatter_hist(data[var].to_numpy(), data['xco2MLbias'].to_numpy(), var, 'RF Bias [ppm]', name, path,save_IO=save_fig, bias_IO=False)


# explain bias correction
# plot bias correction vs each variable with other variables varying naturally


# load normalization parameters
with open(path / 'normalization_params.json', 'r') as f:
    norm_params = json.load(f)
features = norm_params['features']
model = norm_params['model_type']
for var in features:
    print(var)
    scatter_hist(data[var].to_numpy(), data['xco2MLbias'].to_numpy(), var, 'xco2 ML bias [ppm]', name + model, path, save_IO=save_fig, bias_IO=True)

# plot bias correction vs each variable with other variables set to their mean
X = data
X_mean = X.mean()
for var in features:
    print(var)
    steps = 10000
    # make Df with mean of features
    X_i = X.iloc[:steps].copy()
    X_i[features] = X_mean[features]
    X_i[var] = np.linspace(X[var].quantile(0.01), X[var].quantile(0.99), num=steps)
    # make prediction while keeping other vars at mean
    X_i = bias_correct(path1, X_i, ['xco2_raw'])
    if precorrect_IO:
        X_i = bias_correct(path2, X_i, ['xco2_raw'])
    X_i.loc[:, 'xco2MLbias'] = X_i.loc[:, 'xco2_raw_orig'] - X_i.loc[:, 'xco2_raw']

    scatter_hist(X_i[var].to_numpy(), X_i['xco2MLbias'].to_numpy(), var, 'xco2 ML bias [ppm]', name + model + 'mean', path, save_IO=save_fig, bias_IO=True)


# show how all model perform for individual viewing mode
print('show how all model perform for individual viewing mode ...')
if mode == 'all':
    for m in ['LndGL', 'LndND', 'LndNDGL', 'SeaGL']:
        print(m)
        if m == 'LndND':
            print('removing ocean')
            d_m = data.loc[data['land_fraction'] == 100, :]
            d_m = d_m.loc[d_m['operation_mode'] == 0, :]

        elif m == 'LndGL':
            print('removing ocean')
            d_m = data.loc[data['land_fraction'] == 100, :]
            d_m = d_m.loc[d_m['operation_mode'] == 1, :]

        elif m == 'LndNDGL':
            print('removing ocean')
            d_m = data.loc[data['land_fraction'] == 100, :]
            d_m = d_m.loc[d_m['operation_mode'] != 2 , :]
        elif m == 'SeaGL':
            print('removing land')
            d_m = data.loc[data['land_fraction'] == 0, :]


        d_m_all = d_m.copy()
        for qf in [0,1]:
            print('making plots for QF=' + str(qf))
            d_m = d_m_all.loc[d_m_all['xco2_quality_flag'] == qf]
            name_i = name + '_' +m+ '_QF' + str(qf)

            #comparison to TCCON
            _, _, _, _, _, _, _, _, _ = plot_tccon(d_m, TCCON_names, save_fig=save_fig, path=path, name=name_i, qf=qf)
            # SA variability reduction histogram
            get_variability_reduction(d_m, var_tp, name_i, path, save_fig=save_fig, qf=qf)

            # calculate RMSE before and after bias correction
            print('get RMSE reduction')
            RMSE_Raw = get_RMSE(d_m['xco2raw_SA_bias'].to_numpy())
            RMSE_ML = get_RMSE(d_m['xco2raw_SA_bias-ML'].to_numpy(), ignore_nan=True)
            RMSE_B11 = get_RMSE(d_m['xco2_SA_bias'].to_numpy())

            # write results out to file
            d = {'RMSE_Raw_SA': RMSE_Raw,
                    'RMSE_ML_SA': RMSE_ML,
                    'RMSE_B11_SA': RMSE_B11
                    }
            df = pd.DataFrame(data=d, index=[0])
            df.to_csv(path / (name_i + '_error.txt'), index=False, float_format='%.4f')


#****           Comparison to Models            ***************************************************
#show comparison to Models
if data_type != 'test':
    print('Comparison to Models ...')
    models = data.loc[:,'CT_2022+NRT2023-1':'MACC_v21r1'].columns.to_list()
    # select data for model comparison
    for qf in [0]:
        if qf == 0:
            d_m = data.loc[(data[models[0]] > 0) & (data['xco2_quality_flag'] == 0),:]
        elif qf == 1:
            d_m = data.loc[(data[models[0]] > 0) & (data['xco2_quality_flag'] == 1), :]
        else:
            d_m = data.loc[(data[models[0]] > 0), :]
        # calc model mean
        d_m.loc[:,'Model_mean'] = d_m.loc[:, 'CT_2022+NRT2023-1':'MACC_v21r1'].mean(axis=1)
        d_m.loc[:,'Model_std'] = d_m.loc[:, 'CT_2022+NRT2023-1':'MACC_v21r1'].std(axis=1)
        models.append('Model_mean')
        # calc differences
        model_diffs = []
        for m in models:
            d_m.loc[:,'B11-' + m] = d_m.loc[:,'xco2'] - d_m.loc[:,m]
            d_m.loc[:,'ML-'+ m ] = d_m.loc[:,'xco2MLcorr'] - d_m.loc[:,m]
            d_m.loc[:, 'raw-' + m] = d_m.loc[:, 'xco2_raw'] - d_m.loc[:, m]
            model_diffs.append('B11-' + m)
            model_diffs.append('ML-' + m)
            model_diffs.append('raw-' + m)


        #quantify differences to models
        M_std = d_m.loc[:,model_diffs].std()
        M_mean = d_m.loc[:,model_diffs].mean()
        M = pd.concat([M_mean, M_std], axis=1)
        M.rename(columns={0:'Mean', 1:'Std'}, inplace=True)
        M.to_csv(path / (name + '_diff_M_QF' + str(qf) + '.csv'))

        modelsB11 = models.copy()
        modelsML = models.copy()
        for i in range(len(models)):
            modelsB11[i] = 'B11-' + models[i]
            modelsML[i] = 'ML-' + models[i]
        # plot differences on map
        plot_map(d_m, modelsB11, save_fig=save_fig, path=path, name='B11_diff_M_QF' + str(qf), pos_neg_IO=True, min=-1, max=1)
        plot_map(d_m, modelsML, save_fig=save_fig, path=path, name='ML_diff_M_QF' + str(qf), pos_neg_IO=True, min=-1, max=1)


        # # plot difference vs std
        d_m.loc[:,'B11-Model_mean_std'] = np.abs(d_m['xco2'] - d_m['Model_mean'])/d_m['Model_std']
        d_m.loc[:,'ML-Model_mean_std'] = np.abs(d_m['xco2MLcorr'] - d_m['Model_mean'])/d_m['Model_std']
        plot_map(d_m, ['B11-Model_mean_std', 'ML-Model_mean_std'], save_fig=save_fig, path=path, name='ratio_M', pos_neg_IO=False, min=1, max=5)

        # # plot differences for individual seasons
        d_m = get_season(d_m)
        for season in ['MAM', 'JJA', 'SON', 'DJF']:
            plot_map(d_m[d_m['season'] == season], ['B11-Model_mean', 'ML-Model_mean', 'raw-Model_mean'], save_fig=save_fig, path=path, name='Model_mean_diff_' + season, pos_neg_IO=True, min=-1, max=1)


# #****           Visualize biases at land water crossings            ***************************************************
if mode == 'all':
    # # identify small areas that cross from ocean to land
    print('Visualize biases at land water crossings ...')
    for qf in [0]:
        d = data.loc[data['coast'] == 1, :]
        d = d.loc[d['xco2_quality_flag'] == qf]
        # for each small area calculate bias as if Land would be the truth (difference to ocean)
        SAs = pd.unique(d['SA'])
        coast_bias_ML = np.zeros_like(d['SA']) * np.nan
        coast_bias_B11 = np.zeros_like(d['SA'])* np.nan

        for SA in SAs:
            # find soundings that belong to SA
            id = d['SA'] == SA
            d_SA = d.loc[id]
            d_SA_sea = d_SA.loc[d_SA['land_fraction'] == 0, :]
            d_SA_lnd = d_SA.loc[d_SA['land_fraction'] == 100, :]
            # check that we have enough soundings for a robust bias estimation
            if (len(d_SA_lnd) >= 5) & (len(d_SA_sea) >= 5):
                # calc median sea and lnd XCO2
                coast_bias_B11[id] = d_SA_sea['xco2'].median() - d_SA_lnd['xco2'].median()
                coast_bias_ML[id] = d_SA_sea['xco2MLcorr'].median() - d_SA_lnd['xco2MLcorr'].median()

        # add bias to dataframe
        d['coast_bias_B11'] = coast_bias_B11
        d['coast_bias_ML'] = coast_bias_ML

        # reduce redundant values. Only keep one value per SA
        d = d.drop_duplicates(subset='SA')

        # drop rows with 'coast_bias_B11' == nan
        d = d[d['coast_bias_B11'].notna()]

        coast_bias_B11 = d['coast_bias_B11'].to_numpy()
        coast_bias_ML = d['coast_bias_ML'].to_numpy()


        # plot histogram
        plt.figure(figsize=(8, 4))
        bins = np.arange(np.nanpercentile(coast_bias_B11, 2), np.nanpercentile(coast_bias_B11, 98), 0.1)
        n = plt.hist(coast_bias_ML, bins=bins, label='OCO-2 corr.', histtype='step', color='k')
        plt.hist(coast_bias_B11, bins=bins, label='OCO-2 B11 ', histtype='step', color='r')

        plt.vlines(np.median(coast_bias_ML), 0, np.max(n[0]), colors='k',
                    label='OCO-2 corr. = ' + str(np.round(np.nanmedian(coast_bias_ML), 2)) + r'$\pm$' + str(np.round(np.nanstd(coast_bias_ML), 2)))
        plt.vlines(np.median(coast_bias_B11), 0, np.max(n[0]), colors='r',
                    label='OCO-2 B11 = ' + str(np.round(np.nanmedian(coast_bias_B11), 2)) + r'$\pm$' + str(np.round(np.nanstd(coast_bias_B11), 2)))

        plt.title('OCO-2 sea - lnd QF=' + str(qf))
        plt.xlabel('XCO2 [ppm]')
        plt.ylabel('#soundings')
        plt.legend()
        plt.tight_layout()
        if save_fig:
            plt.savefig(path / ('sea-lnd_hist_QF=' + str(qf) + name + '.png')) # Use Path object
        else:
            plt.show()

        # plot by latitude
        # aggregate by latitude
        coast_bias_B11_l = []
        coast_bias_ML_l = []
        step = 5
        lats = np.arange(-60,60+step,step)
        for l in lats:
            coast_bias_B11_l.append(np.nanmedian(d['coast_bias_B11'].loc[(d['latitude'] >= l) & (d['latitude'] <= l+step)]))
            coast_bias_ML_l.append(np.nanmedian(d['coast_bias_ML'].loc[(d['latitude'] >= l) & (d['latitude'] <= l + step)]))

        plt.figure(figsize=(4, 6))
        plt.title('OCO-2 sea - lnd QF=' + str(qf))
        # plt.plot(coast_bias_B11_l, lats, label='OCO-2 B11')
        # plt.plot(coast_bias_ML_l,lats,  label='OCO-2 corr.')
        # plt.vlines(0, -60, 60)
        plt.fill_betweenx(lats,  coast_bias_B11_l, label='OCO-2 B11', alpha=0.5)
        plt.fill_betweenx(lats,  coast_bias_ML_l,  label='OCO-2 corr.', alpha=0.5)
        # plt.barh(lats-1, coast_bias_B11_l,2.0, label='OCO-2 B11')
        # plt.barh(lats+1, coast_bias_ML_l,2.0, label='OCO-2 corr.')
        plt.xlabel('XCO2 sea - lnd [ppm]')
        plt.ylabel('Latitude [$^\circ$]')
        plt.legend()
        plt.tight_layout()
        if save_fig:

            plt.savefig(path / ('sea-lnd_Lat QF=' + str(qf) + name + '.png'))
        else:
            plt.show()

        # plot on a map
        d_clean = d[d['coast_bias_B11'].notna()]
        plot_map(d_clean, ['coast_bias_B11', 'coast_bias_ML'], save_fig=save_fig, path=path,
                    name=name + 'Coast_diff_QF' + str(qf), pos_neg_IO=True, min=-1, max=1)

# # #****           Visualize LndND LndGL biases             ***************************************************
if mode == 'all':
    print('Visualize biases at land water crossings ...')
    for qf in [0,1]:
        d = data.loc[data['xco2_quality_flag'] == qf]
        # bin all LndND and LndGL soundings by latitude
        d['latitude_bin'] = pd.cut(d['latitude'], np.arange(-60, 60, 5))
        # calculate median bias for each bin for LndND
        Lnd = d.loc[(d['land_water_indicator'] == 0), :]
        LndND = Lnd.loc[Lnd['operation_mode'] == 0, :]
        LndGL = Lnd.loc[Lnd['operation_mode'] == 1, :]
        # calculate median for each bin
        LndND_median = LndND.groupby('latitude_bin')['xco2MLcorr'].median()
        LndGL_median = LndGL.groupby('latitude_bin')['xco2MLcorr'].median()

        LndND_median_B11 = LndND.groupby('latitude_bin')['xco2'].median()
        LndGL_median_B11 = LndGL.groupby('latitude_bin')['xco2'].median()
        #calculate difference
        LndND_GL = LndND_median - LndGL_median
        LndND_GL_B11 = LndND_median_B11 - LndGL_median_B11
        # calculate std for each bin
        LndND_std = LndND.groupby('latitude_bin')['xco2MLcorr'].std()
        LndGL_std = LndGL.groupby('latitude_bin')['xco2MLcorr'].std()
        LndND_std_B11 = LndND.groupby('latitude_bin')['xco2'].std()
        LndGL_std_B11 = LndGL.groupby('latitude_bin')['xco2'].std()

        # average std
        LndND_GL_std = np.sqrt(LndND_std**2 + LndGL_std**2)
        LndND_GL_std_B11 = np.sqrt(LndND_std_B11**2 + LndGL_std_B11**2)

        # convert calculated values to numpy arrays
        LndND_GL = LndND_GL.to_numpy()
        LndND_GL_std = LndND_GL_std.to_numpy()
        LndND_GL_index = np.arange(-57.5, 57.5, 5)

        LndND_GL_B11 = LndND_GL_B11.to_numpy()
        LndND_GL_std_B11 = LndND_GL_std_B11.to_numpy()


        # plot
        plt.figure(figsize=(4, 10))
        plt.errorbar(LndND_GL, LndND_GL_index,  xerr=LndND_GL_std, fmt='o', label='OCO-2 corr.')
        plt.errorbar(LndND_GL_B11, LndND_GL_index, xerr=LndND_GL_std_B11, fmt='x', label='OCO-2 B11')
        plt.vlines(0, -60, 60, colors='k', linestyles='dashed')
        plt.title('OCO-2 LndND - LndGL QF=' + str(qf))
        plt.ylabel('Latitude [$^\circ$]')
        plt.xlabel('XCO2 [ppm]')
        plt.legend()
        plt.tight_layout()
        if save_fig:
            plt.savefig(path / ('LndND-LndGL QF=' + str(qf) + name + '.png'))
        else:
            plt.show()



#****           Visualize footprint biases             ***************************************************
print('Visualize footprint biases ...')
#itterate over footprints and calculate bias to median
SAs = pd.unique(data['SA'])
#take a subset of SAs to speed up computation
SAs = SAs[::100]
bias_f = []
for SA in SAs:
    d_SA = data.loc[(data['SA'] == SA) & (data['xco2_quality_flag'] == 0),:]
    #check that we have at least each footprint once in SA
    footprints = pd.unique(d_SA['footprint'])
    if len(footprints) == 8:
        bias_f_SA = np.zeros((8, 2)) * np.nan
        for f in np.arange(1,9):
            bias_f_SA[f-1,0] = d_SA.loc[d_SA['footprint'] == f,'xco2MLcorr'].mean() - d_SA.loc[:,'xco2MLcorr'].mean()
            bias_f_SA[f-1,1] = d_SA.loc[d_SA['footprint'] == f, 'xco2'].mean() - d_SA.loc[:,'xco2'].mean()
        bias_f.append(bias_f_SA)
bias_f = np.stack(bias_f,0)

#visualiz mean bias
bias_f_mean = np.nanmean(bias_f,0)
# Create the figure and axis
fig, ax = plt.subplots()
# Create the bar plot
ax.bar(np.arange(1, 9)-0.2, bias_f_mean[:,0],width=0.4, label='OCO-2 corr.')
ax.bar(np.arange(1, 9)+0.2, bias_f_mean[:,1],width=0.4, label='B11')
# Set the x-axis label
ax.set_xlabel('footprint')
# Set the y-axis label
ax.set_ylabel('XCO2 offset [ppm]')
# Add a horizontal line at y=0
ax.axhline(y=0, color='k', linestyle='--')
# Add a legend
ax.legend()
plt.tight_layout()
if save_fig:
    plt.savefig(path / ('Footprint_offset QF=' + str(qf) + name + '.png'))
else:
    plt.show()

#visualiz RMSE
bias_f_RMSE = np.nanmean(bias_f**2,0)**0.5
# Create the figure and axis
fig, ax = plt.subplots()
# Create the bar plot
ax.bar(np.arange(1, 9)-0.2, bias_f_RMSE[:,0],width=0.4, label='OCO-2 corr.')
ax.bar(np.arange(1, 9)+0.2, bias_f_RMSE[:,1],width=0.4, label='B11')
# Set the x-axis label
ax.set_xlabel('footprint')
# Set the y-axis label
ax.set_ylabel('XCO2 RMSE [ppm]')
# Add a legend
ax.legend()
plt.tight_layout()
if save_fig:
    plt.savefig(path / ('Footprint_RMSE QF=' + str(qf) + name + '.png'))
else:
    plt.show()


print('Done >>>')