#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 19:41:13 2026

@author: jesseruijer
"""


#For the features some clock time, some event time, some combined as maybe a fraction to capture speed 
#Use SHAP on logistic and lgbm to look at all features and determine which are important
#Mabye make a partial dependence plot where the calibration curve updates as time passes
#maybe more complicated stuff regarding moments for other features i could add later make sure to add them to feature lists and corr plots
#The label doesnt matter for 88 since theres no bid and ask side there
#Mabye the vol of 88 at eod is the amount of vol and the price could maybe be midprice or some other price

#Importing libraries,classes, functions from other scripts
import pandas as pd
import config
import joblib
import os
import gc
import numpy as np

from pathlib import Path
from DataAndFeatureEngineering import import_data, clean_data, data_regressors
from LightGBMEngine import train_lgbm_model
from ModelEvaluation import test_model
from LogisticRegressionEngine import train_logistic_model
from FileManager import select_files_via_finder, get_data_paths, generate_dynamic_paths, get_ml_training_paths, get_batch_data_paths
from FeatureFinder import feature_finder
from sklearn.preprocessing import StandardScaler


#Just a display feature in console so all columns are printed in console
pd.set_option('display.max_columns', None)

def prep_data_daily(file_path, file_path_mo):
    print(f'Runs full pipeline for {os.path.basename(file_path)}')
    rawdata = import_data(file_path, file_path_mo)
    cleandata = clean_data(rawdata)
    matrices = data_regressors(rawdata, cleandata)
    
    del rawdata
    del cleandata
    gc.collect()
    
    return {
        'Binary Matrix': matrices['Binary Matrix'], 
        'Multi Matrix': matrices['Multi Matrix']
        }

def save_data():
    print('Builds parquet files for easy storage and optimisation')
    
    batches_to_save = get_batch_data_paths()
    
    if not batches_to_save:
        print("No files were selected, pipleline canceled")
        return
    
    print(f"\nSuccessfully queued {len(batches_to_save)} days for processing. Starting engine...")
    
    #Loop through the list of files you selected
    for main_path, mo_path in batches_to_save:
        binary_file_dest, multi_file_dest = generate_dynamic_paths(main_path)
        
        print(f'\n--- Processing: {os.path.basename(main_path)} ---')
        
        matrices = prep_data_daily(main_path, mo_path)

        matrices['Binary Matrix'].to_parquet(binary_file_dest)
        matrices['Multi Matrix'].to_parquet(multi_file_dest)
        
        print(f'Saved -> {os.path.basename(binary_file_dest)}')
        print(f'Saved -> {os.path.basename(multi_file_dest)}')
        
        #Once processed delete the giant matrices like cleandata etc from RAM and flush the memory 
        del matrices
        gc.collect()
        
    print('\nBatch processing complete!')
    

def run_project():
    
    #Selecting files to use
    
    paths = get_ml_training_paths()
    
    if not paths:
        print("Model training aborted. No training files selected.")
        return
    
    if paths['train_bin'] and paths['test_bin']:
        print('\n[DETECTED BINARY DATA] Executing Logistic Regression')
        
        train_matrix_bin = pd.read_parquet(paths['train_bin'])
        test_matrix_bin = pd.read_parquet(paths['test_bin'])
        
        logistic_regX = train_matrix_bin[config.LOGISTIC_MODEL_FEATURES]
        logistic_regY = train_matrix_bin[config.TARGET]
        
        fill_weights = train_matrix_bin['UnitWeight']
        
        print('Cleaning infinities') # if some infinities slipped through, kill them so scalar can do its job
        logistic_regX = logistic_regX.replace([np.inf,-np.inf],0)
        
        # # DIAGNOSTIC: Print Highly Correlated Pairs 
        # scalar = StandardScaler()
        # logistic_regX_scaled = pd.DataFrame(
        #     scalar.fit_transform(logistic_regX), 
        #     columns=logistic_regX.columns
        # )
       
        # print("\nCalculating Correlation Matrix...")
        # corr_matrix = logistic_regX_scaled.corr().abs()
        
        # # Grab the upper triangle to avoid printing duplicates (e.g., A<->B and B<->A)
        # upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # print("\n--- Highly Correlated Feature Pairs (> 0.95) ---")
        # high_corr_pairs = []
        
        # # Hunt down the >0.95 pairs
        # for col in upper_triangle.columns:
        #     correlated_rows = upper_triangle[col][upper_triangle[col] > 0.85].index
        #     for row in correlated_rows:
        #         high_corr_pairs.append((row, col, upper_triangle.loc[row, col]))
                

        # high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
        
        # if not high_corr_pairs:
        #     print("No features correlated > 0.85 found!")
        # else:
        #     for f1, f2, score in high_corr_pairs:
        #         print(f"[{score:.4f}] {f1}  <-->  {f2}")
        # print("------------------------------------------------\n")


    
        # subset = train_matrix_bin[['LOTrailingVolPlaced','LOTrailingVolCanceled']]
        # print(subset.corr())
        
        # # Or visually inspect the relationship
        # import matplotlib.pyplot as plt
        # plt.scatter(train_matrix_bin['LOTrailingVolPlaced'], train_matrix_bin['LOTrailingVolCanceled'], alpha=0.1)
        # plt.show()
        
        # print("Stopping script for manual feature review...")
        # return
                
        
        
        #Training model, can be commented when saved model
        #base_lr, calibrated_lr, scalar_lr = train_logistic_model(logistic_regX, logistic_regY, fill_weights)
        
        #Save final model to harddrive, comment after saving 
        
        # model_package = {
            
        #     'base_model': base_lr,
        #     'calibrated_model': calibrated_lr,
        #     'features': ['TimeTillMarketClose','TotalQueueSize','BASpread',   'AbsQImbalance', 'RollingStd_QImbalance', 'LOTrailingVolPlaced',
        #                   'DistanceToMicroprice','LogVolAhead', 'TimeSincePlacement','TimeSinceLastMO'],
        #     'scalar' : scalar_lr
        #     }
        
        # #Pathing logic, uses pathlib library so on different devices it still saves correctly, maybe i need to install the os create folder etc stuff for all teh oter things as well
        # #so that when i run it on an external computer maybe to train for ex, it automatically creates the right folders on that specific computer
        
        # script_dir = Path(__file__).resolve().parent # file is built in variable holder, resolve and parent to get a string that just contains the directory that contains the script
        # models_dir = script_dir.parent / 'models' # /  works as a path joiner
        # models_dir.mkdir(parents = True, exist_ok = True) #failsafe to create folder if it doesnt exist or does nothing if it already exists
        # model_filepath = models_dir / 'Logistic_Regression_Models_V1.joblib'
        
        # joblib.dump(model_package, model_filepath)
        
        # print(f'Succesfully saved logistic regression models to  {model_filepath}')
        
        # return 
        
    #Extracting the model from hard drive 
    #Rebuild the absolute path using pathlib
        script_dir = Path(__file__).resolve().parent 
        models_dir = script_dir.parent / 'models'
        model_filepath = models_dir / 'Logistic_Regression_Models_V1.joblib'
        
        print(f'Loading Logistic Models from {model_filepath}')
        
        #Load the package using the dynamic path
        loaded_model_package = joblib.load(model_filepath)
        
        #Extracting the contents from the dictionary
        features = loaded_model_package['features']
        scalar_lr = loaded_model_package['scalar']
        base_lr = loaded_model_package['base_model']
        calibrated_lr = loaded_model_package['calibrated_model']
        
        
    
        #comment this after best features have been found
        #feature_finder(base_lr, 'Logistic Regression', train_matrix_bin , config.LOGISTIC_MODEL_FEATURES, logistic_regY , fill_weights)
        # print(f'Features used: {features}')
        
        logistic_test = test_model(
            test_data = test_matrix_bin, 
            base_model = base_lr,
            calibrated_model = calibrated_lr,
            scalar = scalar_lr,
            model_name = 'Logistic Regression',
            features = features,
            is_multi = False
        )
        
    if paths['train_bin'] and paths['test_bin']:
        print('\n[DETECTED Bin DATA] Executing LightGBM')

        lgbm_X = train_matrix_bin[config.LGBM_MODEL_FEATURES]
        lgbm_Y = train_matrix_bin[config.TARGET]
        
        fill_weights = train_matrix_bin['UnitWeight']
    
    print('Cleaning infinities') # if some infinities slipped through, kill them so scalar can do its job
    logistic_regX = logistic_regX.replace([np.inf,-np.inf],0)
    
    # #Training model, can be commented when saved model    
    #base_lgbm, calibrated_lgbm = train_lgbm_model(lgbm_X, lgbm_Y, fill_weights)
    
    #comment this after best features have been found
    #feature_finder(base_lgbm,'Light Gradient Boosted Model', test_matrix_bin , config.LGBM_MODEL_FEATURES, None, None)
    #return 
    #Save final model to harddrive, comment after saving 
    
    # model_package = {
        
    #     'base_model': base_lgbm,
    #     'calibrated_model': calibrated_lgbm,
    #     'features': ['TimeTillMarketClose', 'TotalQueueSize','QImbalance','TotalVolImbalance','WeightedVolImbalance','EventDeltaMicroprice',
    #                 'RollingStd_BASpread','RollingStd_QImbalance', 'RollingStd_WeightedVolImbalance', 'RollingMax_OrderFlowImbalance','RollingMin_OrderFlowImbalance',
    #                 'LOTrailingVolPlaced','LOTrailingVolCanceled','LOTrailingPlaceExecuteRatio', 'DistanceToMicroprice','LogVolAhead','QueuePositionRatio','TimeSincePlacement',
    #                 'ClockDeltaLogVolAhead','TimeSinceLastMO']
    #     }
    
    # #Pathing logic, uses pathlib library so on different devices it still saves correctly, maybe i need to install the os create folder etc stuff for all teh oter things as well
    # #so that when i run it on an external computer maybe to train for ex, it automatically creates the right folders on that specific computer
    
    # script_dir = Path(__file__).resolve().parent # file is built in variable holder, resolve and parent to get a string that just contains the directory that contains the script
    # models_dir = script_dir.parent / 'models' # /  works as a path joiner
    # models_dir.mkdir(parents = True, exist_ok = True) #failsafe to create folder if it doesnt exist or does nothing if it already exists
    # model_filepath = models_dir / 'LGBM_Models_V1.joblib'
    
    # joblib.dump(model_package, model_filepath)
    
    # print(f'Succesfully saved LGBM model to  {model_filepath}')
    
    # Extracting the model from hard drive 


    # Rebuild the absolute path using pathlib
    script_dir = Path(__file__).resolve().parent 
    models_dir = script_dir.parent / 'models'
    model_filepath = models_dir / 'LGBM_Models_V1.joblib'
    
    print(f'Loading LGBM from {model_filepath}')
    
    #Load the package using the dynamic path
    loaded_model_package = joblib.load(model_filepath)
    
    #Extracting the contents from the dictionary
    features = loaded_model_package['features']
    base_lgbm = loaded_model_package['base_model']
    calibrated_lgbm = loaded_model_package['calibrated_model']
    
    print(f'Features used: {features}')
    lgbm_test = test_model(
        test_data = test_matrix_bin, 
        base_model = base_lgbm,
        calibrated_model = calibrated_lgbm,
        scalar = None,
        model_name = 'Light Gradient Boosted Model',
        features = features,
        is_multi = False
    )


    
if __name__ == "__main__":
    
    #Manually nukes matrices from previous runs we dont need in RAM anymore
    if 'rawdata' in locals(): del rawdata
    if 'cleandata' in locals(): del cleandata
    if 'Binary_Regression_Matrix' in locals(): del Binary_Regression_Matrix
    if 'Multi_Class_Regression_Matrix' in locals(): del Multi_Class_Regression_Matrix
    if 'X' in locals(): del X
    
    gc.collect()
    
    #When theres new data uncomment this below and run once to store the data, if running same data leave this commented
    
    #save_data()
    
    #Only run the whole project if explicitly call main.py
    run_project()





































