#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 19:41:13 2026

@author: jesseruijer
"""

"""
Main script for controlling the process, to access functionality, go to UserScript
"""

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
from FileManager import generate_dynamic_paths, get_ml_training_paths, get_batch_data_paths


#For some strange reason, due to C++ threading library managaemnt, importing torch causes lgbm to crash, thats why i had to serperately import that below in the script only when its necessary

def prep_data_daily(file_path:str, file_path_mo:str) -> dict[str, pd.DataFrame]:
    
    """
    prep data, i.e run dataandfeaturengineering script on it
    """
    
    print(f'Runs full pipeline for {os.path.basename(file_path)}')
    rawdata = import_data(file_path, file_path_mo)
    cleandata = clean_data(rawdata)
    
    matrices = data_regressors(
        rawdata, 
        cleandata, 
        clear_RAM=True, 
        dont_include_full_trading_day=config.DONT_INCLUDE_FULL_TRAINING_DAY
    )
    
    del rawdata
    del cleandata
    gc.collect()
    
    return {
        'Binary Matrix': matrices['Binary Matrix']
      
        }

def save_data() -> None:
    
    """
    Save feature data for ML training in parquet format
    """
    
    print('Builds parquet files for easy storage and optimisation')
    
    batches_to_save = get_batch_data_paths()
    
    if not batches_to_save:
        print("No files were selected, pipleline canceled")
        return
    
    print(f"\nSuccessfully queued {len(batches_to_save)} days for processing. Starting engine...")
    
    #Loop through the list of files you selected
    for main_path, mo_path in batches_to_save:
        binary_file_dest = generate_dynamic_paths(main_path)
        
        print(f'\n--- Processing: {os.path.basename(main_path)} ---')
        
        matrices = prep_data_daily(main_path, mo_path)

        matrices['Binary Matrix'].to_parquet(binary_file_dest)
        
        print(f'Saved -> {os.path.basename(binary_file_dest)}')
        
        #Once processed delete the giant matrices like cleandata etc from RAM and flush the memory 
        del matrices
        gc.collect()
        
    print('\nBatch processing complete!')
    

def run_project(model_choice: str, job: str) -> None:
    
    """
    Main script that creates training data from the imported files, trains the models on that data,
    stores the trained models so they can be easily reused later and can test the models by calling ModelEvaluation
    """

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
    
# =========================================================================
# 1. LOGISTIC REGRESSION ENGINE
# =========================================================================
    
    if model_choice == 'lr':
        cols_to_load = config.LOGISTIC_MODEL_FEATURES + [config.TARGET, 'UnitWeight', 'ID']
        print(f"Performing Logistic Regression on Binary Data to {job}")
        if job == 'train':
            train_frames = []
           
            for f in train_files:
                train_frames.append(pd.read_parquet(f, columns = cols_to_load))
            
            train_matrix = pd.concat(train_frames, ignore_index = True)
            del train_frames
            gc.collect()
            
            #Copy Required for final safety check below
            lr_X = train_matrix[config.LOGISTIC_MODEL_FEATURES].copy()
            lr_Y = train_matrix[config.TARGET].copy()
            lr_w = train_matrix['UnitWeight'].copy()
            
            del train_matrix
            gc.collect()
            
            # Final safety rest
            lr_X.replace([np.inf, -np.inf], 0, inplace=True)
            
            #use 80% for training and 20% to calibrate on, in chronological order since we have timeseries data 
            split_idx = int(len(lr_X) * .8)
        
            train_X = lr_X.iloc[:split_idx]
            train_Y = lr_Y[:split_idx]
            train_w = lr_w.iloc[:split_idx]
            
            calib_X = lr_X.iloc[split_idx:]
            calib_y = lr_Y.iloc[split_idx:]
            calib_weights = lr_w.iloc[split_idx:]
        
            
            # #Training model, can be commented when saved model
            base_lr, calibrated_lr, scalar_lr = train_logistic_model(train_X, train_Y, train_w, calib_X, calib_y, calib_weights)
              
            #Save final model to harddrive, comment after saving 
            model_package = {
                
                'base_model': base_lr,
                'calibrated_model': calibrated_lr,
                'features': config.LOGISTIC_MODEL_FEATURES,
                'scalar' : scalar_lr
                }
        
            #Pathing logic, uses pathlib library so on different devices it still saves correctly
            script_dir = Path(__file__).resolve().parent # file is built in variable holder, resolve and parent to get a string that just contains the directory that contains the script
            models_dir = script_dir.parent / 'models' # /  works as a path joiner
            models_dir.mkdir(parents = True, exist_ok = True) #failsafe to create folder if it doesnt exist or does nothing if it already exists
            model_filepath = models_dir / config.CURRENT_LR_MODEL
            
            joblib.dump(model_package, model_filepath)
            del train_X, train_Y, train_w, calib_X, calib_y, calib_weights
            gc.collect()
            
            print(f'Succesfully saved logistic regression models to  {model_filepath}')
         
        elif job == 'test':
            test_matrix = pd.read_parquet(test_file, columns = cols_to_load)
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
            #feature_finder(base_lr, 'Logistic Regression', train_matrix , config.LOGISTIC_MODEL_FEATURES, logistic_regY , fill_weights)
            print(f'Features used: {features}')
            
            test_model(
                test_data = test_matrix, 
                base_model = base_lr,
                calibrated_model = calibrated_lr,
                scalar = scalar_lr,
                model_name = 'Logistic Regression',
                features = features,
            )
            
# =========================================================================
# 2. LIGHT GRADIENT BOOSTING ENGINE 
# =========================================================================
        
    elif model_choice == 'lgbm':
        
          print(f"Performing LGBM on binary data to {job}")
          cols_to_load = config.LGBM_MODEL_FEATURES + [config.TARGET, 'UnitWeight', 'ID']
          if job == 'train':
              train_frames = []
             
              for f in train_files:
                  train_frames.append(pd.read_parquet(f, columns = cols_to_load))
              
              train_matrix = pd.concat(train_frames, ignore_index = True)
              del train_frames
              gc.collect()

              #Copy Required for final safety check below
              lgbm_X = train_matrix[config.LGBM_MODEL_FEATURES].copy()
              lgbm_Y = train_matrix[config.TARGET].copy()
              lgbm_w = train_matrix['UnitWeight'].copy()
            
              # Free up the massive original train_matrix from RAM immediately
              del train_matrix
              gc.collect()
            
              # Final safety check
              lgbm_X.replace([np.inf, -np.inf], 0, inplace=True)
            
              #use 80% for training and 20% to calibrate on, in chronological order since we have timeseries data 
              split_idx = int(len(lgbm_X) * .8)
        
              train_X = lgbm_X.iloc[:split_idx]
              train_Y = lgbm_Y[:split_idx]
              train_w = lgbm_w.iloc[:split_idx]
            
              calib_X = lgbm_X.iloc[split_idx:]
              calib_y = lgbm_Y.iloc[split_idx:]
              calib_weights = lgbm_w.iloc[split_idx:]
        
              #Training model, can be commented when saved model    
              base_lgbm, calibrated_lgbm = train_lgbm_model(train_X, train_Y, train_w, calib_X, calib_y, calib_weights)
            
            # Save final model to harddrive, comment after saving 
              model_package = {
                
                  'base_model': base_lgbm,
                  'calibrated_model': calibrated_lgbm,
                  'features': config.LGBM_MODEL_FEATURES
                  }
            
              #Pathing logic, uses pathlib library so on different devices it still saves correctly
            
              script_dir = Path(__file__).resolve().parent # file is built in variable holder, resolve and parent to get a string that just contains the directory that contains the script
              models_dir = script_dir.parent / 'models' # /  works as a path joiner
              models_dir.mkdir(parents = True, exist_ok = True) #failsafe to create folder if it doesnt exist or does nothing if it already exists
              model_filepath = models_dir / config.CURRENT_LGBM_MODEL
            
              joblib.dump(model_package, model_filepath)
            
              print(f'Succesfully saved LGBM model to  {model_filepath}')
              del train_X, train_Y, train_w, calib_X, calib_y, calib_weights
              gc.collect()
              
          if job == 'test':
              test_matrix = pd.read_parquet(test_file, columns = cols_to_load)
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
              #Create val_data small to be a small section of the data to be used as validation
              # feature_finder(base_lgbm,'Light Gradient Boosted Model', val_data_small , config.LGBM_MODEL_FEATURES, None, None)

              print(f'Features used: {features}')
              test_model(
                  test_data = test_matrix, 
                  base_model = base_lgbm,
                  calibrated_model = calibrated_lgbm,
                  scalar = None,
                  model_name = 'Light Gradient Boosted Model',
                  features = features,
              )
        
# =========================================================================
# 3. FEED-FORWARD NEURAL NETWORK ENGINE 
# =========================================================================
        
    elif model_choice == 'fnn':
        import torch
        from FNN import prepdata_and_train, PyTorchSklearnWrapper, NN
        cols_to_load = config.FNN_MODEL_FEATURES + [config.TARGET, 'UnitWeight', 'ID']
        print(f"Performing FNN on Binary Data to {job}")
        if job == 'train':
            #Training model, can be commented when saved model 
            #Giving it one big matrix was too much, let the model train sequentially on the list of days and after done on one day store results and delete from ram
              
            fnn_model, scalar = prepdata_and_train(train_files)
            
            fnn_calibrator = fnn_model.calibrator
            
              # Save final model to harddrive, comment after saving 
            
              # Pathing logic, uses pathlib library so on different devices it still saves correctly, maybe i need to install the os create folder etc stuff for all teh oter things as well
              # so that when i run it on an external computer maybe to train for ex, it automatically creates the right folders on that specific computer
            
            script_dir = Path(__file__).resolve().parent # file is built in variable holder, resolve and parent to get a string that just contains the directory that contains the script
            models_dir = script_dir.parent / 'models' # /  works as a path joiner
            models_dir.mkdir(parents = True, exist_ok = True) #failsafe to create folder if it doesnt exist or does nothing if it already exists
            
            model_filepath = models_dir / config.CURRENT_FNN_MODEL_WEIGHTS
            torch.save(fnn_model.model.state_dict(), model_filepath)  #The dict contains the model weights, since prepdata function alrady returns a wrapped FNN, add .model to the object to have the pure model 
            
            metadata_filepath = models_dir / config.CURRENT_FNN_MODEL_METADATA
            metadata_package = {
                'features': config.FNN_MODEL_FEATURES,
                'scalar': scalar,
                'calibrator': fnn_calibrator
                 }
            
            joblib.dump(metadata_package, metadata_filepath)
            del fnn_model
            gc.collect()
            
            print(f'Succesfully saved FNN weights to  {model_filepath}')
      
        if job == 'test':
             test_matrix = pd.read_parquet(test_file, columns = cols_to_load)
               # Extracting the model from hard drive 
             if torch.backends.mps.is_available():
                 device = torch.device('mps')
                 print('Training on Apple Silicon MPS')
             elif torch.cuda.is_available():
                 device = torch.device('cuda')
                 print('Training on NVIDIA GPU (CUDA)')
             else:
                 device = torch.device('cpu')
                 print('Training on CPU') 
        
               # Rebuild the absolute path using pathlib
             script_dir = Path(__file__).resolve().parent 
             models_dir = script_dir.parent / 'models'
             model_filepath = models_dir / config.CURRENT_FNN_MODEL_WEIGHTS
             metadata_filepath = models_dir / config.CURRENT_FNN_MODEL_METADATA
            
             print(f'Loading FNN weights from {model_filepath}')
            
               # #Load the package using the dynamic path
             metadata_package = joblib.load(metadata_filepath)
            
               #Extracting the contents from the dictionary
             features = metadata_package['features']
             scalar = metadata_package['scalar']
             fnn_calibrator = metadata_package['calibrator']
            
               #re initiaze the 'empty' model blue print
             input_size = len(config.FNN_MODEL_FEATURES)
             loaded_model = NN(input_size = input_size).to(device)
            
               #fill the empty model with my loaded weights from training before
             loaded_model.load_state_dict(torch.load(model_filepath, map_location = device))
            
             #set to eval before testing
             loaded_model.eval()
            
               #Wrap so can use test_model function 
             fnn_wrapped = PyTorchSklearnWrapper(loaded_model, device, fnn_calibrator)
                  
             print(f'Features used: {features}')
             test_model(
                   test_data = test_matrix, 
                   base_model = fnn_wrapped,
                   calibrated_model = fnn_wrapped,
                   scalar = scalar,
                   model_name = 'FNN',
                   features = features,
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
    
    print('What model to use')
    model_choice = input('"LR" , "LGBM", "FNN"').strip().lower()
    print('You MUST have atleast 2 training days (to allow for enough data for training and calibration)')
    print('Do you want to train or test model, you must train first then run again and test if you wish to test')
    job = input('"train", "test"')
    run_project(model_choice, job)