#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 19:41:13 2026

@author: jesseruijer
"""


#For the features some clock time, some event time, some combined as maybe a fraction to capture speed 
#Use SHAP on logistic and lgbm to look at all features and determine which are important
#Mabye make a partial dependence plot where the calibration curve updates as time passes
#There are a lot more lob feature statistics i could add later make sure to add them to feature lists and corr plots
#The label doesnt matter for 88 since theres no bid and ask side there
#Mabye the vol of 88 at eod is the amount of vol and the price could maybe be midprice or some other price

#Importing libraries,classes, functions from other scripts
import pandas as pd
import config
import joblib
import os
import gc

from DataAndFeatureEngineering import import_data, clean_data, data_regressors
from LightGBMEngine import train_lgbm_model
from ModelEvaluation import test_model
from LogisticRegressionEngine import train_logistic_model
from FileManager import select_files_via_finder, get_data_paths, generate_dynamic_paths, get_ml_training_paths, get_batch_data_paths
from FeatureFinder import feature_finder


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
        
        base_lr, calibrated_lr, scalar_lr = train_logistic_model(logistic_regX, logistic_regY, fill_weights)
        
        #Uncomment this after best features have been found
        feature_finder(base_lr, 'Logistic Regression', train_matrix_bin , config.ALL_FEATURES, logistic_regY , fill_weights)
        
        logistic_test = test_model(
            test_data = test_matrix_bin, 
            base_model = base_lr,
            calibrated_model = calibrated_lr,
            scalar = scalar_lr,
            model_name = 'Logistic Regression',
            features = config.LOGISTIC_MODEL_FEATURES,
            is_multi = False
        )
        
    # if paths['train_bin'] and paths['test_bin']:
    #     print('\n[DETECTED Bin DATA] Executing LightGBM')

    #     lgbm_X = train_matrix_bin[config.LGBM_MODEL_FEATURES]
    #     lgbm_Y = train_matrix_bin[config.TARGET]
        
    #     fill_weights = train_matrix_bin['UnitWeight']
        
    #     base_lgbm, calibrated_lgbm = train_lgbm_model(lgbm_X, lgbm_Y, fill_weights)
    
       # #comment this after best features have been found
       # feature_finder(base_lgbm,'Light Gradient Boosted Model', test_matrix_bin , config.LGBM_MODEL_FEATURES)
        
    #     lgbm_test = test_model(
    #         test_data = test_matrix_bin, 
    #         base_model = base_lgbm,
    #         calibrated_model = calibrated_lgbm,
    #         scalar = None,
    #         model_name = 'Light Gradient Boosted Model',
    #         features = config.LGBM_MODEL_FEATURES,
    #         is_multi = False
    #     )


    
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





































