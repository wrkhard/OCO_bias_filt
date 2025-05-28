# Steffen Mauceri
# 08/2020
#
# correct 3D cloud effect on XCO2 retrieval with ML
#
# takes in a list of parameters from csv files that are belived to be informative of the 3D cloud effect affecting XCO2 retrievals
# Grows a random forest to predict the difference between Raw XCO2 or Bias Corrected XCO2 to TCCON stations
# Can also be run in 'Small Area mode' where it tries to predict changes in XCO2 indipendent of TCCON

# make sure you have all of those packages installed in your conda envirnoment
import os
import pandas as pd
import numpy as np
import paths

import matplotlib.pyplot as plt
from util import load_data_and_concat_years, plot_map, raster_data, feature_selector, normalize_per_SA, scatter_hist, scatter_density


# make changes #############################
name = ''
save_fig = True    # save figures to hard drive
verbose_IO = False

qf = None
max_samples = 10**7
path = paths.PLOTS_DIR / 'features'
mode = 'SeaGL'
preload_IO = True
TCCON = False
#stop make changes ###########################

# load data
data = load_data_and_concat_years(2015, 2022, mode=mode, qf=qf, TCCON=TCCON)


if len(data) > max_samples:
    data = data.sample(max_samples, replace=False)

print(str(len(data)/1000) + 'k samples loaded')

#****************************************************************
print('making plots')

#visualize density of soundings
# plot_map(data, ['xco2raw_SA_bias'], save_fig=save_fig, path=path, pos_neg_IO=False, aggregate = 'count', max=2000)

# plot all features on a map
# get features
# feature_select = 'SA_bias_all'
# features, feature_n = feature_selector(feature_select)
features = ['solar_zenith_angle',
                        'sensor_zenith_angle'
                # , 'windspeed_u_met'
                # , 'windspeed_v_met'
                , 'co2_ratio'
                , 'h2o_ratio'
                , 'max_declocking_o2a'
                        # ,'max_declocking_wco2'
                , 'max_declocking_sco2'
                , 'color_slice_noise_ratio_o2a'
                        # ,'color_slice_noise_ratio_wco2'
                , 'color_slice_noise_ratio_sco2'
                , 'h_continuum_o2a'
                        # ,'h_continuum_wco2'
                , 'h_continuum_sco2'
                , 'dp_abp'
                        # ,'surface_type'
                , 'psurf'
                , 'psurf_apriori'
                , 't700'
                        # ,'fs'
                        # ,'fs_rel'
                , 'tcwv'
                # , 'tcwv_apriori'
                , 'dp'
                , 'dp_o2a'
                        # ,'dp_sco2'
                , 'dpfrac'
                , 'co2_grad_del'
                , 'dws'
                # , 'eof3_1_rel'
                        # ,'snow_flag'
                , 'aod_dust'
                , 'aod_bc'
                , 'aod_oc'
                , 'aod_seasalt'
                , 'aod_sulfate'
                , 'aod_strataer'
                , 'aod_water'
                , 'aod_ice'
                        # ,'aod_total'
                , 'dust_height'
                , 'ice_height'
                , 'water_height'
                , 'aod_total_apriori'
                , 'dws_apriori'
                , 'aod_fine_apriori'
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
                , 'rms_rel_o2a'
                , 'rms_rel_wco2'
                , 'rms_rel_sco2'
                , 'solar_azimuth_angle'
                , 'sensor_azimuth_angle'
                , 'polarization_angle'
                ,'land_fraction'
                , 'glint_angle'
                , 'airmass'
                , 'snr_o2a'
                        # ,'snr_wco2'
                , 'snr_sco2'
                        # ,'path'
                , 'footprint'
                ,'land_water_indicator'
                , 'altitude'
                , 'altitude_stddev']
for f in features:
    plot_map(data, f, save_fig=save_fig, path=path, name=f , pos_neg_IO=False)


# put data on raster
# features, feature_n = feature_selector(1)


# # for var in ['cld_dist']:#features:
# #     raster = raster_data(data[var], data['latitude'], data['longitude'], res = 2)
# #
# #     MIN = np.nanpercentile(raster.flatten(), 5)#-0.3
# #     MAX = np.nanpercentile(raster.flatten(), 95)#0.3
# #
# #     var_name = var
# #     Earth_Map_Raster(raster, MIN, MAX, var_name, name, colormap = plt.cm.inferno, Save=save_fig, Save_Name=save_dir+var +'_LndND.png')
#
#
#plot variability vs cld_dist
# nbins = 15
# epsilon = 1e-10
# cld_dist = data['cld_dist'].to_numpy()
# for var in features:
#     print(var)
#     # bin error by variable
#     bin_mean = np.zeros(nbins) * np.nan
#     bin_5 = np.zeros(nbins) * np.nan
#     bin_95 = np.zeros(nbins) * np.nan
#     n, bin_edges = np.histogram(np.arange(0,16), bins=nbins)
#     gap = np.mean(np.diff(bin_edges))
#     x = bin_edges[:-1] + gap / 2
#     x[0] = bin_edges[0]
#     x[-1] = bin_edges[-1]
#
#     # itterate over bins
#     data_var = data[var].to_numpy()
#     i = -1
#     for bin in bin_edges[:-1]:
#         i += 1
#         t = data_var[(cld_dist > bin) & (cld_dist < bin + gap)]
#         if len(t) > 0:
#             bin_mean[i] = np.mean(t)
#             bin_5[i] = np.percentile(t, 5)
#             bin_95[i] = np.percentile(t, 95)
#     #plt.plot(cld_dist, imp[:, k], '.', color='gray', zorder=1)
#     plt.figure(figsize=(4, 3))
#     plt.hlines(100, 0, 16)
#     plt.fill_between(x, (bin_5/bin_mean[-1])*100, (bin_95/bin_mean[-1])*100, color='orange', zorder=2, alpha=0.5)
#     plt.plot(x, (bin_mean/bin_mean[-1])*100, color='red')
#     plt.xlim(x[0], x[-1])
#     plt.ylim(-1000, 1000)
#
#     plt.xlabel('cloud distance [km]')
#     plt.ylabel(var + ' change [%]')
#     plt.tight_layout()
#     plt.savefig('/Users/smauceri/Projects/OCO2/3D_clouds/plots/Mean_vs_clddist_SeaGL_' + var + '.png', dpi=300)
#     plt.show()
#     plt.close()
#
# #plot dp vs time
# # nbins = 35
# # epsilon = 1e-10
# # cld_dist = data['time'].to_numpy()
# # for var in ['xco2_TCCON_bias']:#['xco2', 'xco2_raw','xco2raw_SA_bias','xco2_SA_bias' ,'dp','psurf','psurf_apriori']:
# #     print(var)
# #     if var == 'xco2_TCCON_bias':
# #         data = data[data['xco2tccon'] > 0]
# #         data.loc[:, var] = data['xco2'] - data['xco2tccon']
# #
# #     # bin error by variable
# #     bin_mean = np.zeros(nbins) * np.nan
# #     bin_5 = np.zeros(nbins) * np.nan
# #     bin_95 = np.zeros(nbins) * np.nan
# #     n, bin_edges = np.histogram(cld_dist, bins=nbins)
# #     gap = np.mean(np.diff(bin_edges))
# #     x = bin_edges[:-1] + gap / 2
# #     x[0] = bin_edges[0]
# #     x[-1] = bin_edges[-1]
# #
# #     # itterate over bins
# #     # data_var = (data['psurf_apriori'] - data['psurf']).to_numpy()
# #     data_var = data[var].to_numpy()
# #     i = -1
# #     for bin in bin_edges[:-1]:
# #         i += 1
# #         t = data_var[(cld_dist > bin) & (cld_dist < bin + gap)]
# #         if len(t) > 0:
# #             bin_mean[i] = np.mean(t)
# #             bin_5[i] = np.percentile(t, 15)
# #             bin_95[i] = np.percentile(t, 85)
# #     #plt.plot(cld_dist, imp[:, k], '.', color='gray', zorder=1)
# #     plt.figure(figsize=(4, 3))
# #     #plt.hlines(100, 0, 16)
# #     plt.fill_between(x, (bin_5/bin_mean[-1])*100, (bin_95/bin_mean[-1])*100, color='orange', zorder=2, alpha=0.5)
# #     #plt.plot(x, (bin_mean/bin_mean[-1])*100, color='red')
# #     plt.plot(x, bin_mean, color='red')
# #     plt.xlim(x[0], x[-1])
# #     #plt.ylim(-1000, 1000)
# #
# #     plt.xlabel('time')
# #     plt.ylabel(var)
# #     plt.tight_layout()
# #     plt.savefig('/Users/smauceri/Projects/OCO2/3D_clouds/plots/Mean_p_vs_time_SeaGL_qf01_' + var + '.png', dpi=300)
# #     plt.show()
# #     plt.close()
#
# # plot 2 vars against each other

data['xco2_TCCON_bias'] = data['xco2'] - data['xco2tccon']

vars = ['airmass', 'sensor_zenith_angle']#['albedo_wco2', 'albedo_sco2','albedo_o2a']
for var1 in vars:
    for var2 in vars:
        if var1 == var2:
            continue
        scatter_density(data[var1].to_numpy(), data[var2].to_numpy(), var1, var2, name, path, save_IO=save_fig)



print('>>> Done')
