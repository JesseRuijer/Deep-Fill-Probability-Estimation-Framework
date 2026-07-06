#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 19:41:13 2026

@author: jesseruijer
"""


#Mabye make a partial dependence plot where the calibration curve updates as time passes where its like a few different plots during a normalised order lifetime

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
    matrices = data_regressors(rawdata, cleandata, dont_include_full_trading_day = True)
    
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
    
    train_files = paths.get('train_bin', [])
    test_file = paths.get('test_bin')
    
    if not train_files or not test_file:
        print('Error needs atleast one test and train file')
        return 
    
    print(f'Loading {len(train_files)} training days ')
    
    train_frames = []
    
    for f in train_files:
        train_frames.append(pd.read_parquet(f))
        
    train_matrix = pd.concat(train_frames, ignore_index = True)
    del train_frames
    gc.collect()
    
    test_matrix = pd.read_parquet(test_file)
    
    print("Performing Logistic Regression on Binary Data")
    
    #Copy Required for final safety check below
    lr_X = train_matrix[config.LOGISTIC_MODEL_FEATURES].copy()
    lr_Y = train_matrix[config.TARGET].copy()
    lr_w = train_matrix['UnitWeight'].copy()
    
    # Final safety rest
    lr_X.replace([np.inf, -np.inf], 0, inplace=True)

    
    #Training model, can be commented when saved model
    #base_lr, calibrated_lr, scalar_lr = train_logistic_model(lr_X, lr_Y, lr_w)
    
    
    #Save final model to harddrive, comment after saving 
    
    # model_package = {
        
    #     'base_model': base_lr,
    #     'calibrated_model': calibrated_lr,
    #     'features': config.LOGISTIC_MODEL_FEATURES,
    #     'scalar' : scalar_lr
    #     }
    
    # #Pathing logic, uses pathlib library so on different devices it still saves correctly, maybe i need to install the os create folder etc stuff for all teh oter things as well
    # #so that when iyh run it on an external computer maybe to train for ex, it automatically creates the right folders on that specific computer
    
    # script_dir = Path(__file__).resolve().parent # file is built in variable holder, resolve and parent to get a string that just contains the directory that contains the script
    # models_dir = script_dir.parent / 'models' # /  works as a path joiner
    # models_dir.mkdir(parents = True, exist_ok = True) #failsafe to create folder if it doesnt exist or does nothing if it already exists
    # model_filepath = models_dir / config.CURRENT_LR_MODEL
    
    # joblib.dump(model_package, model_filepath)
    
    # print(f'Succesfully saved logistic regression models to  {model_filepath}')
 
    #Extracting the model from hard drive 
    #Rebuild the absolute path using pathlib
    script_dir = Path(__file__).resolve().parent 
    models_dir = script_dir.parent / 'models'
    model_filepath = models_dir / config.CURRENT_LR_MODEL
    
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
    
    # logistic_test = test_model(
    #     test_data = test_matrix, 
    #     base_model = base_lr,
    #     calibrated_model = calibrated_lr,
    #     scalar = scalar_lr,
    #     model_name = 'Logistic Regression',
    #     features = features,
    #     is_multi = False
    # )
    
    del lr_X, lr_Y, lr_w
    gc.collect()
        
    print("Performing LGBM on binary data")
    
    #Copy Required for final safety check below
    lgbm_X = train_matrix[config.LGBM_MODEL_FEATURES].copy()
    lgbm_Y = train_matrix[config.TARGET].copy()
    lgbm_w = train_matrix['UnitWeight'].copy()
    
    # Free up the massive original train_matrix from RAM immediately
    del train_matrix
    gc.collect()
    
    # Final safety rest
    lgbm_X.replace([np.inf, -np.inf], 0, inplace=True)

    # #Training model, can be commented when saved model    
    base_lgbm, calibrated_lgbm = train_lgbm_model(lgbm_X, lgbm_Y, lgbm_w)
    
   
    #Save final model to harddrive, comment after saving 
    
    model_package = {
        
        'base_model': base_lgbm,
        'calibrated_model': calibrated_lgbm,
        'features': config.LGBM_MODEL_FEATURES
        }
    
    #Pathing logic, uses pathlib library so on different devices it still saves correctly, maybe i need to install the os create folder etc stuff for all teh oter things as well
    #so that when i run it on an external computer maybe to train for ex, it automatically creates the right folders on that specific computer
    
    script_dir = Path(__file__).resolve().parent # file is built in variable holder, resolve and parent to get a string that just contains the directory that contains the script
    models_dir = script_dir.parent / 'models' # /  works as a path joiner
    models_dir.mkdir(parents = True, exist_ok = True) #failsafe to create folder if it doesnt exist or does nothing if it already exists
    model_filepath = models_dir / config.CURRENT_LGBM_MODEL
    
    joblib.dump(model_package, model_filepath)
    
    print(f'Succesfully saved LGBM model to  {model_filepath}')
    
    # Extracting the model from hard drive 


    # Rebuild the absolute path using pathlib
    script_dir = Path(__file__).resolve().parent 
    models_dir = script_dir.parent / 'models'
    model_filepath = models_dir / config.CURRENT_LGBM_MODEL
    
    print(f'Loading LGBM from {model_filepath}')
    
    # #Load the package using the dynamic path
    loaded_model_package = joblib.load(model_filepath)
    
    #Extracting the contents from the dictionary
    features = loaded_model_package['features']
    base_lgbm = loaded_model_package['base_model']
    calibrated_lgbm = loaded_model_package['calibrated_model']
    
    
    # #comment this after best features have been found
    
    # train_matrix_small = train_matrix.sample(frac=0.01, random_state=69)
    
    # lgbm_X = train_matrix_small[config.LGBM_MODEL_FEATURES].replace([np.inf, -np.inf], 0)
    # lgbm_Y = train_matrix_small[config.TARGET]
    # lgbm_w = train_matrix_small['UnitWeight']
    
    # # 2. CREATE A FAST, DUMB ARCHITECTURE
    # fast_params = {
    #     'objective': 'binary',
    #     'boosting_type': 'gbdt', 
    #     'metric': 'binary_logloss', 
    #     'n_jobs': -1, 
    #     'random_state': 69, 
    #     'n_estimators': 150,    # Stop at 150 trees max
    #     'learning_rate': 0.1,   # Fast learning rate
    #     'num_leaves': 31        # Simple, shallow trees
    # }

    # # 3. TRAIN IT FAST (Use for_tuning=True to skip the heavy Isotonic Calibrator)
    # base_lgbm = train_lgbm_model(
    #     lgbm_X, lgbm_Y, lgbm_w, 
    #     params=fast_params, 
    #     for_tuning=True 
    # )
    
    # # 4. RUN SHAP (On a 1% slice of the test matrix)
    # val_data_small = test_matrix.sample(frac=0.01, random_state=69)
    # feature_finder(base_lgbm, 'Light Gradient Boosted Model', val_data_small, config.LGBM_MODEL_FEATURES, None, None)
    
    # val_data_small = test_matrix.sample(frac = 0.01, random_state = 69)
   
    # feature_finder(base_lgbm,'Light Gradient Boosted Model', val_data_small , config.LGBM_MODEL_FEATURES, None, None)
    # return 
    
    print(f'Features used: {features}')
    lgbm_test = test_model(
        test_data = test_matrix, 
        base_model = base_lgbm,
        calibrated_model = calibrated_lgbm,
        scalar = None,
        model_name = 'Light Gradient Boosted Model',
        features = features,
        is_multi = False
    )
    
    del lgbm_X, lgbm_Y, lgbm_w
    gc.collect()

    
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





































