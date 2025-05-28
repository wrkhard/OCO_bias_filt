
# make various plots to visualize the effect of bias correction and filtering
# Note, this a collection of scripts and code snippets.

# make sure you have all of those packages installed in your conda envirnoment
import pandas as pd
import numpy as np
import os
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from util import load_data_and_concat_years, plot_tccon, plot_map, get_season, normalize_per_SA, scatter_hist, \
    get_variability_reduction, scatter_density, calc_SA_bias, bias_correct, construct_filter, get_RMSE


def keep_min_overpasses(data, min_overpasses = 10):
    TCCON_names = np.unique(data['tccon_name'].to_numpy())
    TCCON_keep = []
    for name in TCCON_names:
        overpasses = np.sum(np.diff(data.loc[data['tccon_name'] == name, 'time']) > 10*60)
        if np.unique(overpasses) > min_overpasses:
            TCCON_keep.append(name)
    return data[data['tccon_name'].isin(TCCON_keep)]




# make changes #############################
# Bias correction model directory paths
TC_LND_CORR_PATH = 'bias_corr_models/V12_10_2.6_noTCCONcorr_xco2_TCCON_biasLndNDGL_lnd_RF0/'
TC_OCN_CORR_PATH = 'bias_corr_models/V12_11_2.6_noTCCONcorr_xco2_TCCON_biasSeaGL_sea_RF0/'
SA_LND_CORR_PATH = 'bias_corr_models/V12_10_2.6_noTCCONcorr_prec_xco2raw_SA_biasLndNDGL_lnd_RF0/'
SA_OCN_CORR_PATH = 'bias_corr_models/V12_11_2.6_noTCCONcorr_prec_xco2raw_SA_biasSeaGL_sea_RF0/'

# filter model paths
TC_LND_FILT_PATH = 'current_filters/V10_B11Gamma_TCCON_lndNDGL_filter_RF0.p'
TC_OCN_FILT_PATH = 'current_filters/V10_B11Gamma_TCCON_ocnGL_filter_RF0.p'
SA_LND_FILT_PATH = 'current_filters/V10_B11Gamma_small_area_lndNDGL_filter_RF0.p'
SA_OCN_FILT_PATH = 'current_filters/V10_B11Gamma_small_area_ocnGL_filter_RF0.p'

path = '/Volumes/OCO/LiteFiles/B11_gamma/'

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
hold_out_year = 2022
#stop make changes ###########################


if not os.path.exists(path):
    os.makedirs(path)


qf = None#p['qf']
mode = 'all'#p['mode']
print(name)


# load data
if data_type == 'train':
    #Train set
    data = load_data_and_concat_years(2018, 2022, hold_out_year=hold_out_year, mode=mode, qf=qf, TCCON=TCCON)

elif data_type == 'all':
    #Train+Val+Test set
    data = load_data_and_concat_years(2015, 2022, mode=mode, qf=qf, TCCON=TCCON)

elif data_type == 'test':
    #Test set
    data = load_data(hold_out_year, mode=mode, qf=qf, TCCON=TCCON)

elif data_type == 'model':
    #Test set
    data = load_data(2019, mode, qf=qf, TCCON=TCCON)
    data1 = load_data(2020, mode,qf=qf, TCCON=TCCON)
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


data = construct_filter(data, **kwargs)


data.loc[:,'xco2MLcorr'] = data.loc[:,'xco2_raw']
data.loc[:,'xco2_raw'] = data.loc[:,'xco2_raw_orig']
data.loc[:,'xco2MLbias'] = data.loc[:,'xco2_raw_orig'] - data.loc[:,'xco2MLcorr']



#****           compare to TCCON            ************************************************************
#
# get TCCON station names
TCCON_names = np.unique(data['tccon_name'].to_numpy())

# remove pasadena and xianghe
TCCON_names = TCCON_names[TCCON_names != '']
# TCCON_names = TCCON_names[TCCON_names != 'pasadena01']
# TCCON_names = TCCON_names[TCCON_names != 'xianghe01']


# calc for ML QF=0
data_T = data[data['xco2tccon'] > 0].copy()
data_T['xco2-TCCON'] = data_T['xco2MLcorr'] - data_T['xco2tccon']
data_T = data_T.loc[data_T['xco2_quality_flag_gamma'] == 0]

data_T_lnd = data_T.loc[data_T['land_fraction'] == 100]
data_T_sea = data_T.loc[data_T['land_fraction'] == 0]

# remove tccon stations from data_T_sea that are not near the ocean
TCCON_keep = ['burgos', 'darwin', 'nyalesund', 'eureka', 'izana', 'lauder', 'reunion', 'rikubetsu',
              'saga', 'tsukuba', 'wollongong']
data_T_sea = data_T_sea[data_T_sea['tccon_name'].isin(TCCON_keep)]


data_T_lnd = keep_min_overpasses(data_T_lnd)
data_T_sea = keep_min_overpasses(data_T_sea)


# calculate average error for each TCCON station
lnd_TCCON_error = data_T_lnd.groupby('tccon_name')['xco2-TCCON'].agg(['std', 'median'])
sea_TCCON_error = data_T_sea.groupby('tccon_name')['xco2-TCCON'].agg(['std', 'median'])

# remove pasadena
# lnd_TCCON_error = lnd_TCCON_error[lnd_TCCON_error.index != 'pasadena']

# calculate std of all medians
lnd_TCCON_std = np.std(lnd_TCCON_error['median'])
sea_TCCON_std = np.std(sea_TCCON_error['median'])
# print out
print('ML corr and ML QF=0')
print('Land TCCON std: ' + str(lnd_TCCON_std))
print('Sea TCCON std: ' + str(sea_TCCON_std))



# calc for ML QF=0+1
data_T = data[data['xco2tccon'] > 0].copy()
data_T['xco2-TCCON'] = data_T['xco2MLcorr'] - data_T['xco2tccon']
data_T = data_T.loc[data_T['xco2_quality_flag_gamma'] != 2]

data_T_lnd = data_T.loc[data_T['land_fraction'] == 100]
data_T_sea = data_T.loc[data_T['land_fraction'] == 0]

# remove tccon stations from data_T_sea that are not near the ocean
TCCON_keep = ['burgos', 'darwin', 'nyalesund', 'eureka', 'izana', 'lauder', 'reunion', 'rikubetsu',
              'saga', 'tsukuba', 'wollongong']
data_T_sea = data_T_sea[data_T_sea['tccon_name'].isin(TCCON_keep)]

data_T_lnd = keep_min_overpasses(data_T_lnd)
data_T_sea = keep_min_overpasses(data_T_sea)

# calculate average error for each TCCON station
lnd_TCCON_error = data_T_lnd.groupby('tccon_name')['xco2-TCCON'].agg(['std', 'median'])
sea_TCCON_error = data_T_sea.groupby('tccon_name')['xco2-TCCON'].agg(['std', 'median'])

# calculate std of all medians
lnd_TCCON_std = np.std(lnd_TCCON_error['median'])
sea_TCCON_std = np.std(sea_TCCON_error['median'])
# print out
print('ML corr and ML QF=0+1')
print('Land TCCON std: ' + str(lnd_TCCON_std))
print('Sea TCCON std: ' + str(sea_TCCON_std))




# calc for ML corr and B11 QF=0
data_T = data[data['xco2tccon'] > 0].copy()
data_T['xco2-TCCON'] = data_T['xco2MLcorr'] - data_T['xco2tccon']
data_T = data_T.loc[data_T['xco2_quality_flag'] == 0]

data_T_lnd = data_T.loc[data_T['land_fraction'] == 100]
data_T_sea = data_T.loc[data_T['land_fraction'] == 0]

# remove tccon stations from data_T_sea that are not near the ocean
TCCON_keep = ['burgos', 'darwin', 'nyalesund', 'eureka', 'izana', 'lauder', 'reunion', 'rikubetsu',
              'saga', 'tsukuba', 'wollongong']
data_T_sea = data_T_sea[data_T_sea['tccon_name'].isin(TCCON_keep)]

data_T_lnd = keep_min_overpasses(data_T_lnd)
data_T_sea = keep_min_overpasses(data_T_sea)

# calculate average error for each TCCON station
lnd_TCCON_error = data_T_lnd.groupby('tccon_name')['xco2-TCCON'].agg(['std', 'median'])
sea_TCCON_error = data_T_sea.groupby('tccon_name')['xco2-TCCON'].agg(['std', 'median'])

# calculate std of all medians
lnd_TCCON_std = np.std(lnd_TCCON_error['median'])
sea_TCCON_std = np.std(sea_TCCON_error['median'])
# print out
print('ML corr and B11 QF=0')
print('Land TCCON std: ' + str(lnd_TCCON_std))
print('Sea TCCON std: ' + str(sea_TCCON_std))



# calc for B11 corr and B11 QF=0
data_T = data[data['xco2tccon'] > 0].copy()
data_T['xco2-TCCON'] = data_T['xco2'] - data_T['xco2tccon']
data_T = data_T.loc[data_T['xco2_quality_flag'] == 0]

data_T_lnd = data_T.loc[data_T['land_fraction'] == 100]
data_T_sea = data_T.loc[data_T['land_fraction'] == 0]

# remove tccon stations from data_T_sea that are not near the ocean
TCCON_keep = ['burgos', 'darwin', 'nyalesund', 'eureka', 'izana', 'lauder', 'reunion', 'rikubetsu',
              'saga', 'tsukuba', 'wollongong']
data_T_sea = data_T_sea[data_T_sea['tccon_name'].isin(TCCON_keep)]

data_T_lnd = keep_min_overpasses(data_T_lnd)
data_T_sea = keep_min_overpasses(data_T_sea)

# calculate average error for each TCCON station
lnd_TCCON_error = data_T_lnd.groupby('tccon_name')['xco2-TCCON'].agg(['std', 'median'])
sea_TCCON_error = data_T_sea.groupby('tccon_name')['xco2-TCCON'].agg(['std', 'median'])

# calculate std of all medians
lnd_TCCON_std = np.std(lnd_TCCON_error['median'])
sea_TCCON_std = np.std(sea_TCCON_error['median'])
# print out
print('B11 corr and B11 QF=0')
print('Land TCCON std: ' + str(lnd_TCCON_std))
print('Sea TCCON std: ' + str(sea_TCCON_std))
