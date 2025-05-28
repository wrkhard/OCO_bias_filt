# Steffen Mauceri
# 12/2022
#
# make various plots to visualize feature importance for our bias correction
# Note, this a collection of scripts and code snippets.

# make sure you have all of those packages installed in your conda envirnoment
import pandas as pd
from multiprocessing import Pool
import numpy as np
import shap
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from util import load_data_by_year, plot_map, get_season, raster_data
import joblib
import json
import paths #


def main_process(data):
    ''' get feature importance. In a function to run in parallel

    :param data: DataFrame;
    :return: DataFrame; with columns for feature importance attached
    '''
    # Define model paths using the paths module from the B11.2 series
    path1_dir = paths.TC_LND_CORR_MODEL # Dir for TCCON Land Bias Correction Model
    path2_dir = paths.SA_LND_CORR_MODEL # Dir for Sounding Aspect Land Bias Correction Model
    
    # pathQF_dir should point to a directory containing QF model files (trained_model.joblib, normalization_params.json)
    # paths.py currently provides direct .joblib file paths for filters (e.g., paths.TC_LND_FILTER_MODEL)
    # Using a bias correction model path as a placeholder if QF_IO is True, adjust if a dedicated QF model dir exists.
    pathQF_dir = paths.TC_LND_CORR_MODEL # Placeholder QF model directory path

    QF_IO = False  # processing QF model. Set to False to work with bias correction model
    precorrect_IO = True

    # --------------------------------------------------------------
    data = data[0] # data is passed as a list containing the DataFrame

    # load model
    if QF_IO:
        model_load_path = pathQF_dir / 'trained_model.joblib'
        params_load_path = pathQF_dir / 'normalization_params.json'
        
        M = joblib.load(model_load_path)
        with open(params_load_path, 'r') as f:
            params = json.load(f)
        features = params['features']
        model_type = params.get('model_type', params.get('model')) 
        mode = params.get('mlqf_mode', params.get('mode')) 
        qf = params.get('qf', None)
    else:
        model_load_path = path1_dir / 'trained_model.joblib'
        params_load_path = path1_dir / 'normalization_params.json'
        M = joblib.load(model_load_path)
        with open(params_load_path, 'r') as f:
            params = json.load(f)
        X_mean = pd.Series(params['X_mean'])
        X_std = pd.Series(params['X_std'])
        y_mean = params.get('y_mean')
        y_std = params.get('y_std')
        features = params['features']
        model_type = params.get('model_type', params.get('model'))
        qf = params.get('qf')
        mode = params['mode']


    # make input output pair
    X = data.loc[:,features]

    if model_type == 'RF':
        if hasattr(M, 'max_samples') and M.max_samples is not None:
             M.max_samples = min(M.max_samples, 100000)
        else:
             M.max_samples = 100000
        M.n_jobs = 1
    elif model_type not in ['GPR']:
        X = (X - X_mean) / X_std

    features_imp = [item + '_imp' for item in features]

    print('shap calculations for 1st model ...')
    if model_type in ['RF', 'XGB'] or 'RandomForest' in str(type(M)) or 'XGB' in str(type(M)):
        explainer = shap.TreeExplainer(M)
        shap_values = explainer.shap_values(X)
    else: 
        print(f"Warning: SHAP values for model type {model_type} might be slow or require KernelExplainer.")
        data.loc[:, features_imp] = np.nan
        shap_values = None

    if shap_values is not None:
        if QF_IO and isinstance(shap_values, list) and len(shap_values) > 0:
            data.loc[:, features_imp] = shap_values[0]
        else:
            data.loc[:, features_imp] = shap_values


    if precorrect_IO and not QF_IO:
        model_load_path_2 = path2_dir / 'trained_model.joblib'
        params_load_path_2 = path2_dir / 'normalization_params.json'
        M2 = joblib.load(model_load_path_2)
        with open(params_load_path_2, 'r') as f:
            params2 = json.load(f)
        X_mean2 = pd.Series(params2['X_mean'])
        X_std2 = pd.Series(params2['X_std'])
        features2 = params2['features']
        model_type2 = params2.get('model_type', params2.get('model'))

        X2 = data.loc[:,features2]
        if model_type2 == 'RF':
            if hasattr(M2, 'max_samples') and M2.max_samples is not None:
                 M2.max_samples = min(M2.max_samples, 100000)
            else:
                 M2.max_samples = 100000
            M2.n_jobs = 1
        elif model_type2 not in ['GPR']:
            X2 = (X2 - X_mean2) / X_std2

        features_imp2 = [item + '_imp2' for item in features2]

        print('shap calculations for 2nd model ...')
        shap_values2 = None
        if model_type2 in ['RF', 'XGB'] or 'RandomForest' in str(type(M2)) or 'XGB' in str(type(M2)):
            explainer2 = shap.TreeExplainer(M2)
            shap_values2 = explainer2.shap_values(X2)
        else:
            print(f"Warning: SHAP values for model type {model_type2} (2nd model) might be slow or require KernelExplainer.")
            data.loc[:, features_imp2] = np.nan # Placeholder

        if shap_values2 is not None:
             data.loc[:, features_imp2] = shap_values2
    return data

if __name__ == "__main__":
    # Script configuration
    # Use paths from the paths.py module
    path1_model_dir = paths.TC_LND_CORR_MODEL  # TCCON Land Bias Correction Model directory
    path2_model_dir = paths.SA_LND_CORR_MODEL  # Sounding Aspect Land Bias Correction Model directory
    
    # As in main_process, pathQF_model_dir expects a directory.
    # Using a bias correction model path as a placeholder if QF_IO is True.
    pathQF_model_dir = paths.TC_LND_CORR_MODEL # Placeholder QF model directory

    # Output directory for figures generated by this script
    analysis_output_dir = paths.FIGURE_DIR / 'vis_feature_importance_map_outputs'
    paths.ensure_dir_exists(analysis_output_dir)

    QF_IO = False  # processing QF model. Set to False to work with bias correction model
    precorrect_IO = True    # Correct data with two models, first the one in path1_model_dir then path2_model_dir
    save_fig = True    # save figures to hard drive
    verbose_IO = False # This variable is defined but not used in the script. Consider removing if not needed.
    max_samples = 2*10**7
    plot_name_prefix = '_shap_' # Prefix for output plot names
    n_jobs = 14 # Number of parallel jobs for processing

    # Determine the base model directory for naming, can be simplified
    # The `current_model_name_segment` will be used to make plot names more specific
    if QF_IO:
        current_model_dir_for_naming = pathQF_model_dir
    elif precorrect_IO: # If precorrecting, path2_model_dir is the "final" model affecting the data
        current_model_dir_for_naming = path2_model_dir
    else: # Otherwise, path1_model_dir is the one used
        current_model_dir_for_naming = path1_model_dir
    
    # Extract a segment from the model path for unique naming of output files.
    # This uses the parent directory name of the model path.
    current_model_name_segment = current_model_dir_for_naming.name 
    name = current_model_name_segment + plot_name_prefix
    print(f"Output name prefix: {name}")

    # load model parameters (similar to main_process, but for the main script logic)
    if QF_IO:
        active_model_params_path = pathQF_model_dir
        model_file = active_model_params_path / 'trained_model.joblib'
        params_file = active_model_params_path / 'normalization_params.json'
        # M = joblib.load(model_file) # Model M is loaded in main_process, not strictly needed here for params only
        with open(params_file, 'r') as f:
            params = json.load(f)
        features = params['features']
        model_type = params.get('model_type', params.get('model'))
        mode = params.get('mlqf_mode', params.get('mode'))
        qf = params.get('qf', None)
    else:
        active_model_params_path = path1_model_dir # Parameters for the first model
        model_file = active_model_params_path / 'trained_model.joblib'
        params_file = active_model_params_path / 'normalization_params.json'
        # M = joblib.load(model_file) # Model M is loaded in main_process
        with open(params_file, 'r') as f:
            params = json.load(f)
        # X_mean = pd.Series(params['X_mean']) # Not used directly in this block
        # X_std = pd.Series(params['X_std'])   # Not used directly in this block
        # y_mean = params.get('y_mean')       # Not used directly in this block
        # y_std = params.get('y_std')         # Not used directly in this block
        features = params['features']
        model_type = params.get('model_type', params.get('model'))
        qf = params.get('qf')
        mode = params['mode']


    # Train+Val+Test set
    data = load_data_by_year(2015, 2022, mode)


    if len(data) > max_samples:
        data = data.sample(max_samples, replace=False)
    print(str(len(data)/1000) + 'k samples loaded')

    data.sort_values('SA', inplace=True)

    # split data into chunks for parallel processing
    data_all = []
    for i in range(n_jobs):
        data_i = data.iloc[i*len(data)//n_jobs:(i+1)*len(data)//n_jobs] # Corrected upper bound for iloc
        data_all.append([data_i])

    # Parallel execution of main_process
    with Pool(n_jobs) as p:
        result = p.map(main_process, data_all)
    data = pd.concat(result)

    if QF_IO:
        features_all = features
        features_imp = [item + '_imp' for item in features]
        features_imp_all = features_imp

    else:
        # Load parameters for the second model if precorrect_IO is True
        if precorrect_IO:
            params_load_path_2 = path2_model_dir / 'normalization_params.json'
            with open(params_load_path_2, 'r') as f:
                params2 = json.load(f)
            features2 = params2['features']
        else: 
            features2 = [] 

        # combine feature importances
        features_all_ = features + features2
        features_all = []
        [features_all.append(x) for x in features_all_ if x not in features_all] # Deduplicate
        
        features_imp_all = [item + '_impAll' for item in features_all]
        features_imp1 = [item + '_imp' for item in features]
        features_imp2 = [item + '_imp2' for item in features2] # features2 might be empty
        
        columns = data.columns

        for f in features_all:
            imp_all_col = f + '_impAll'
            imp1_col = f + '_imp'
            imp2_col = f + '_imp2'
            
            data[imp_all_col] = 0 # Initialize
            if imp1_col in columns and f in features: # Check if feature belongs to model 1
                data[imp_all_col] = data[imp1_col]
            if imp2_col in columns and f in features2: # Check if feature belongs to model 2
                if imp1_col in columns and f in features : # If already added from model 1
                     data[imp_all_col] += data[imp2_col]
                else: # If only in model 2
                     data[imp_all_col] = data[imp2_col]
            # If a feature_imp column (e.g. f+'_imp') was created but the feature was not in features1 or features2
            # (e.g. if features/features2 were modified after shap calculation)
            # this logic might need adjustment. Assuming features/features2 align with *_imp/*_imp2 columns.


    #****           visualize results            ************************************************************
    print('making plots')
    # Save SHAP summary plots
    shap.summary_plot(data.loc[::10, features_imp_all].to_numpy(), features = data.loc[::10, features_all],feature_names=features_all, show=False)
    plt.savefig(analysis_output_dir / (name + '_shap_summary_all.png'))
    plt.close()
    
    if not QF_IO:
        if features: # Check if features list is not empty
            shap.summary_plot(data.loc[::10, features_imp1].to_numpy(), features=data.loc[::10, features],feature_names=features, show=False)
            plt.savefig(analysis_output_dir / (name + '_shap_summary_1.png'))
            plt.close()
        if features2: # Check if features2 list is not empty for the second model
            shap.summary_plot(data.loc[::10, features_imp2].to_numpy(), features=data.loc[::10, features2], feature_names=features2, show=False)
            plt.savefig(analysis_output_dir / (name + '_shap_summary_2.png'))
            plt.close()

    # Plot feature importance maps
    pos_neg_IO = True # For plot_map, determines if positive/negative contributions are shown differently
    plot_map(data, features_imp_all, save_fig=save_fig, path=analysis_output_dir,
             name=name + '_IMP_rel', pos_neg_IO=pos_neg_IO)
    if not QF_IO:
        if features:
            plot_map(data, features_imp1, save_fig=save_fig, path=analysis_output_dir,
                     name=name + '_IMP1_rel', pos_neg_IO=pos_neg_IO)
        if features2:
            plot_map(data, features_imp2, save_fig=save_fig, path=analysis_output_dir,
                     name=name + '_IMP2_rel', pos_neg_IO=pos_neg_IO)

    # Plot feature importance for different seasons
    data = get_season(data)
    for season in ['MAM', 'JJA', 'SON', 'DJF']:
        plot_map(data.loc[(data['season'] == season),:], features_imp_all, save_fig=save_fig,
                 path=analysis_output_dir, name=name + '_IMP_rel_' + season, pos_neg_IO=pos_neg_IO)

    # Show most important feature for a given location
    rastered_imp = []
    for var in features_imp_all:
        rastered_imp.append(raster_data(data[var].to_numpy(), data['latitude'].to_numpy(), data['longitude'].to_numpy()))
    rastered_imp = np.stack(rastered_imp, axis=2)
    imp_max = np.argmax(rastered_imp, axis=2).astype(float)
    imp_max[np.isnan(rastered_imp[:,:,0])] = np.nan # Mask NaNs based on the first feature's raster
    
    most_imp_var_plot_name = 'Most_Important_Variable'

    def Earth_Map_Raster_cat(raster, MIN, MAX, Title, labels, Save=False, Save_Name_Base='None', res=1):
        ''' makes beautiful plot of data organized in a matrix representing lat lon of the globe
        similar to Earth_Map_Raster but with categorical labels

        :param raster: gridded data
        :param MIN: min value for plot
        :param MAX: max value for plot
        # :param var_name: Name of var to be plotted (covered by Title)
        :param Title: Title of the plot
        :param Save: whether to save the plot or show it
        :param Save_Name_Base: Base path and name for saving the plot (e.g., analysis_output_dir / (most_imp_var_plot_name + Title_suffix))
        :param res: resolution of map in deg
        # :param colormap: colormap is now derived dynamically (cmap = plt.get_cmap('tab20', len(labels)))
        # :param extend: extend parameter for colorbar (not used in this version)
        :return: matplotlib figure object
        '''
        # New version with Cartopy
        limits = [-180, 180, -90, 90]
        offset = res / 2 * np.array([0, 0, 2, 2]) # Offset for correct pixel centering
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
        ax.coastlines()
        gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
        
        # Ensure discrete colormap for categorical data
        cmap = plt.get_cmap('tab20', MAX - MIN +1) # MAX - MIN + 1 is the number of categories
        
        im = ax.imshow(np.flipud(raster), interpolation='nearest', origin='lower',
                       extent=np.array(limits) + offset, cmap=cmap, vmin=MIN - 0.5, vmax=MAX + 0.5, # Adjust vmin/vmax for discrete colorbar
                       transform=ccrs.PlateCarree(), alpha=0.9)

        ax.set_title(Title, fontsize=15, pad=10)
        
        # Custom colorbar for categorical data
        cbar = fig.colorbar(im, ticks=np.arange(MIN, MAX + 1), fraction=0.066, pad=0.08, spacing='proportional')
        cbar.set_ticklabels(labels[MIN:MAX+1]) # Ensure labels match the MIN, MAX range shown
        
        plt.tight_layout()

        if Save:
            # Save_Name_Base is now expected to be a Path object for the directory and base name
            plt.savefig(str(Save_Name_Base) + '.png', dpi=200) # Ensure Save_Name_Base is string for plt.savefig
            plt.close()
        else:
            plt.show()
        return fig

    if QF_IO:
        plot_labels = [s.rstrip('_imp') for s in features_imp_all]
    else:
        plot_labels = [s.rstrip('_impAll') for s in features_imp_all]

    # Ensure plot_labels is not empty before plotting
    if plot_labels:
        Earth_Map_Raster_cat(imp_max, 0, len(plot_labels) -1, # Max index is len-1
                             f'{most_imp_var_plot_name} ({current_model_name_segment})', 
                             plot_labels, 
                             Save=save_fig, 
                             Save_Name_Base=analysis_output_dir / (most_imp_var_plot_name + name)
                            )
    else:
        print("No features to plot for Earth_Map_Raster_cat.")


    print('Done >>>')