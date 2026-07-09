#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 14:22:54 2026

@author: jesseruijer
"""

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import gc
import random
from pathlib import Path
from sklearn.preprocessing import StandardScaler

import config

from ModelEvaluation import test_model
from FileManager import get_ml_training_paths, get_batch_data_paths, generate_dynamic_paths
from DataAndFeatureEngineering import import_data, clean_data, data_regressors

#Below is a fix to be able to run lgbm and torch in one script 
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

#####Use either lgbm or fnn to get outputs and performance on selected scripts and ID


def prep_data_daily(file_path, file_path_mo):
    print(f'Runs full pipeline for {os.path.basename(file_path)}')
    rawdata = import_data(file_path, file_path_mo)
    cleandata = clean_data(rawdata)
    matrices = data_regressors(rawdata, cleandata, config.DONT_INCLUDE_FULL_TRAINING_DAY)
    
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


def train(train_files, train_matrix, model):
    
    print('Starting Training')
    
    if model == 'FNN':
        
                
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
        from FNN import PyTorchSklearnWrapper, DataSet, UserFNN
    
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            print('Training on Apple Silicon MPS')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
            print('Training on NVIDIA GPU (CUDA)')
        else:
            device = torch.device('cpu')
            print('Training on CPU') 
        
        input_size = len(config.FNN_MODEL_FEATURES)
        
        LEARNING_RATE = 0.003143153725649377
        WEIGHT_DECAY = 2.5475947521348294e-06
        EPOCHS = 7
        BATCH_SIZE = 16384 
        
        scalar = StandardScaler()
        for f in train_files:
            df = pd.read_parquet(f)
            df.replace([np.inf, -np.inf], 0, inplace = True)

            X_train_raw = df[config.FNN_MODEL_FEATURES].values
            
            scalar.partial_fit(X_train_raw) #we cant fit the scalar to all days at once since it crashes RAM so use the partial_fit function that slowly updates the right scaling. It ends up getting the same result as applying a scalar to the whole day, it just doesnt crash RAM, and then later when we are training you just pass the scalar
            
            del df, X_train_raw
            gc.collect()

        model = UserFNN(input_size = input_size).to(device)
         
        criterion = nn.BCEWithLogitsLoss(reduction = 'none')    # BCE = Binary Cross Entropy = Logloss, this is just the scoring metric and saying reducion is none, it doesnt do any weighting by iteself it just spits out all the raw values and then with my manual weights i can do the weighint later
        optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY)
         
        for epoch in range(EPOCHS):
             model.train()
             random.shuffle(train_files)
             
             for f in train_files:
                 
                 df = pd.read_parquet(f)
                 df.replace([np.inf, -np.inf], 0, inplace = True)
                 
                 X_train_scaled = scalar.transform(df[config.FNN_MODEL_FEATURES].values)
                 y_train = df[config.TARGET].values
                 w_train = df['UnitWeight'].values
                 
                 train_dataset = DataSet(X_train_scaled, y_train, w_train)
                 train_loader = DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, shuffle = True)  #NOTE: for a FNN you must shuffle data, which works because at every stage the network has no memory of what happened before it, it does not introduce lookahead bias and helps the model converge faster
                 
                 
                 for features, labels, batch_weights in train_loader:
                     features, labels, batch_weights = features.to(device), labels.to(device), batch_weights.to(device)
                 
                     outputs = model(features)

                     #Custom weight loss
                     
                     raw_loss = criterion(outputs, labels)
                     weighted_batch_loss = (raw_loss * batch_weights).sum()
                     
                     loss = weighted_batch_loss / batch_weights.sum()
                     
                     optimizer.zero_grad()   #By default gradients accumulate in pytorch so zero them out here
                     loss.backward()
                     optimizer.step()

                 
                 del df, X_train_scaled, y_train, w_train, train_dataset, train_loader
                 gc.collect()
             print(f' Epoch {epoch + 1} / {EPOCHS} \n')       
        wrapped_fnn = PyTorchSklearnWrapper(model, device)
       
        
    
        script_dir = Path(__file__).resolve().parent # file is built in variable holder, resolve and parent to get a string that just contains the directory that contains the script
        models_dir = script_dir.parent / 'models' # /  works as a path joiner
        models_dir.mkdir(parents = True, exist_ok = True) #failsafe to create folder if it doesnt exist or does nothing if it already exists
      
      
        model_filepath = models_dir / config.USER_FNN_MODEL_WEIGHTS
        torch.save(model.state_dict(), model_filepath)  #The dict contains the model weights, since prepdata function alrady returns a wrapped FNN, add .model to the object to have the pure model 
      
        metadata_filepath = models_dir / config.USER_FNN_MODEL_METADATA
        metadata_package = {
            'features': config.FNN_MODEL_FEATURES,
            'scalar': scalar
            }
      
        joblib.dump(metadata_package, metadata_filepath)
      
        print(f'Succesfully saved FNN weights to  {model_filepath}')
        return wrapped_fnn, scalar
          
    elif model == 'LGBM':
          
        from LightGBMEngine import train_lgbm_model
            
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
        model_filepath = models_dir / config.USER_LGBM_MODEL
     
        joblib.dump(model_package, model_filepath)
     
        print(f'Succesfully saved LGBM model to  {model_filepath}')
        
    elif model == 'LR':
          
        from LogisticRegressionEngine import train_logistic_model
            
        #Copy Required for final safety check below
        lr_X = train_matrix[config.LOGISTIC_MODEL_FEATURES].copy()
        lr_Y = train_matrix[config.TARGET].copy()
        lr_w = train_matrix['UnitWeight'].copy()
        
        # Free up the massive original train_matrix from RAM immediately
        del train_matrix
        gc.collect()
        
      
        # Final safety check
        lr_X.replace([np.inf, -np.inf], 0, inplace=True)
        
        #use 80% for training and 20% to calibrate on, in chronological order since we have timeseries data 
        split_idx = int(len(lr_X) * .8)
    
        train_X = lr_X.iloc[:split_idx]
        train_Y = lr_Y[:split_idx]
        train_w = lr_w.iloc[:split_idx]
        
        calib_X = lr_X.iloc[split_idx:]
        calib_y = lr_Y.iloc[split_idx:]
        calib_weights = lr_w.iloc[split_idx:]
          
        base_lr, calibrated_lr, scalar_lr = train_logistic_model(train_X, train_Y, train_w, calib_X, calib_y, calib_weights)
         
        model_package = {
            
            'base_model': base_lr,
            'calibrated_model': calibrated_lr,
            'features': config.LOGISTIC_MODEL_FEATURES,
            'scalar' : scalar_lr
            }
     
      #Pathing logic, uses pathlib library so on different devices it still saves correctly, maybe i need to install the os create folder etc stuff for all teh oter things as well
      #so that when i run it on an external computer maybe to train for ex, it automatically creates the right folders on that specific computer
     
        script_dir = Path(__file__).resolve().parent # file is built in variable holder, resolve and parent to get a string that just contains the directory that contains the script
        models_dir = script_dir.parent / 'models' # /  works as a path joiner
        models_dir.mkdir(parents = True, exist_ok = True) #failsafe to create folder if it doesnt exist or does nothing if it already exists
        model_filepath = models_dir / config.USER_LR_MODEL
     
        joblib.dump(model_package, model_filepath)
     
        print(f'Succesfully saved LR model to  {model_filepath}')
 
    

def test_model_wrap(test_matrix, model):
    
    # Rebuild the absolute path using pathlib
    script_dir = Path(__file__).resolve().parent 
    models_dir = script_dir.parent / 'models'
    
 
    if model == 'FNN':
        
        import torch
        import torch.nn as nn
        from FNN import PyTorchSklearnWrapper, NN
        
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            print('Training on Apple Silicon MPS')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
            print('Training on NVIDIA GPU (CUDA)')
        else:
            device = torch.device('cpu')
            print('Training on CPU') 
        
       
        model_filepath = models_dir / config.USER_FNN_MODEL_WEIGHTS
        metadata_filepath = models_dir / config.USER_FNN_MODEL_METADATA
        
        print(f'Loading FNN weights from {model_filepath}')
        
        # #Load the package using the dynamic path
        metadata_package = joblib.load(metadata_filepath)
        
        #Extracting the contents from the dictionary
        features = metadata_package['features']
        scalar = metadata_package['scalar']
        
        #re initiaze the 'empty' model blue print
        input_size = len(config.FNN_MODEL_FEATURES)
        loaded_model = NN(input_size = input_size).to(device)
        
        #fill the empty model with my loaded weights from training before
        loaded_model.load_state_dict(torch.load(model_filepath, map_location = device))
        
        
        #set to eval before testing
        
        loaded_model.eval()
        
        #Wrap so can use test_model function 
        fnn_wrapped = PyTorchSklearnWrapper(loaded_model, device)
              
        print(f'Features used: {features}')
        fnn_test = test_model(
            test_data = test_matrix, 
            base_model = fnn_wrapped,
            calibrated_model = fnn_wrapped,
            scalar = scalar,
            model_name = 'FNN',
            features = features,
            is_multi = False
        )
    
    elif model == 'LGBM':
        model_filepath = models_dir / config.USER_LGBM_MODEL
        
        print(f'Loading LGBM from {model_filepath}')
        
        # #Load the package using the dynamic path
        loaded_model_package = joblib.load(model_filepath)
        
        #Extracting the contents from the dictionary
        features = loaded_model_package['features']
        base_lgbm = loaded_model_package['base_model']
        calibrated_lgbm = loaded_model_package['calibrated_model']
        
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
        
    elif model == 'LR':

        model_filepath = models_dir / config.USER_LR_MODEL
        
        print(f'Loading LR from {model_filepath}')
        
        # #Load the package using the dynamic path
        loaded_model_package = joblib.load(model_filepath)
        
        #Extracting the contents from the dictionary
        features = loaded_model_package['features']
        scalar_lr = loaded_model_package['scalar']
        base_lr = loaded_model_package['base_model']
        calibrated_lr = loaded_model_package['calibrated_model']
        
        print(f'Features used: {features}')
       
        logistic_test = test_model(
            test_data = test_matrix, 
            base_model = base_lr,
            calibrated_model = calibrated_lr,
            scalar = scalar_lr,
            model_name = 'Logistic Regression',
            features = features,
            is_multi = False
        )
       
 
def single_order_eval(ID, TSP, test_data, selected_model, features, scalar):
    
    order_data = test_data[test_data['ID'] == ID].copy()
    if len(order_data) == 0:
        print('ID Not Found')
        return
    
    order_data = order_data.sort_values('TimeSincePlacement')
    
    if TSP < order_data['TimeSincePlacement'].min():
        print('Order Does not exist yet at this time')
        return
    if TSP > order_data['TimeSincePlacement'].max():
        print('Order Does is dead already at this time')
        return
    else:
        state_at_tsp = order_data[order_data['TimeSincePlacement'] == TSP].iloc[-1:]
        
    X_raw = state_at_tsp[features].values
    
    if scalar is not None:
        X = scalar.transform(X_raw)
    else:
        X = X_raw
        
    prob = selected_model.predict_proba(X)[0,1]
    
    print(f'At {TSP}ms into this order ID life, the fill probability is {prob}')
    
    return prob
    
if __name__ == "__main__":
    
    print('Welcome Boss, Starting Programme')
    
    process_choice = input('Do you want to process Raw files (y/n) ').strip().lower()
    if process_choice == 'y':
        save_data()
    else:
        print('Moving on to preprocessed data')
    
    print('Please select TRAINING Data (atleast one day)')
    print('Please select Test Data (One day only and no overlap with test data (obviously))')

       
    paths = get_ml_training_paths()
    train_files = paths.get('train_bin', [])
    test_files = paths.get('test_bin', [])
    test_matrix = pd.read_parquet(test_files) 
  
    
    model_choice = ''
    while model_choice not in ['1', '2', '3']:
        print("\n MODEL SELECTION ")
        print("1. Logistic Regression")
        print("2. LGBM")
        print("3. Neural Network ")
        model_choice = input("Enter the number of the model you want to run (1/2/3): ").strip() #.strip just strips away whitespaces etc, so just the keyword remains
        
        if model_choice not in ['1', '2', '3']:
            print("Invalid input. Please enter 1, 2, or 3.")

    if model_choice == '1':
        selected_model = 'LR'
    elif model_choice == '2':
        selected_model = 'LGBM'
    elif model_choice == '3':
        selected_model = 'FNN'

    print(f"\n[System] You selected: {selected_model}")
    action_choice = input("Do you want to 'train' or 'test' or 'use' this model? ").strip().lower()

    if action_choice == 'train':
        
        train_frames = []
        
        for f in train_files:
            train_frames.append(pd.read_parquet(f))
            
        train_matrix = pd.concat(train_frames, ignore_index = True)
        train(train_files, train_matrix, selected_model)
        
        
    elif action_choice == 'test':
        
        test_model_wrap(test_matrix, selected_model)
        
        
    elif action_choice == 'use':
        print(f'Use the {selected_model} to experiment on orders, type "exit" if you want to stop')
        
        script_dir = Path(__file__).resolve().parent 
        models_dir = script_dir.parent / 'models'
        
        if selected_model == 'FNN':
            import torch
            import torch.nn as nn
            from FNN import PyTorchSklearnWrapper, UserFNN
            
            if torch.backends.mps.is_available():
                device = torch.device('mps')
                print('Training on Apple Silicon MPS')
            elif torch.cuda.is_available():
                device = torch.device('cuda')
                print('Training on NVIDIA GPU (CUDA)')
            else:
                device = torch.device('cpu')
                print('Training on CPU') 
            
           
            model_filepath = models_dir / config.USER_FNN_MODEL_WEIGHTS
            metadata_filepath = models_dir / config.USER_FNN_MODEL_METADATA
            # #Load the package using the dynamic path
            metadata_package = joblib.load(metadata_filepath)
            
            #Extracting the contents from the dictionary
            features = metadata_package['features']
            scalar = metadata_package['scalar']
            
            #re initiaze the 'empty' model blue print
            input_size = len(config.FNN_MODEL_FEATURES)
            loaded_model = UserFNN(input_size = input_size).to(device)
            
            #fill the empty model with my loaded weights from training before
            loaded_model.load_state_dict(torch.load(model_filepath, map_location = device))
        
        elif selected_model == 'LR':
            model_filepath = models_dir / config.USER_LR_MODEL
            
            print(f'Loading LR from {model_filepath}')
            
            # #Load the package using the dynamic path
            loaded_model_package = joblib.load(model_filepath)
            
            #Extracting the contents from the dictionary
            features = loaded_model_package['features']
            scalar = loaded_model_package['scalar']
            base_lr = loaded_model_package['base_model']
            calibrated_lr = loaded_model_package['calibrated_model']
            
        elif selected_model == 'LGBM':
            model_filepath = models_dir / config.USER_LGBM_MODEL
            print(f'Loading LGBM from {model_filepath}')
            
            # #Load the package using the dynamic path
            loaded_model_package = joblib.load(model_filepath)
            
            #Extracting the contents from the dictionary
            features = loaded_model_package['features']
            base_lgbm = loaded_model_package['base_model']
            calibrated_lgbm = loaded_model_package['calibrated_model']
        
        
        while True:
            print('Give Order ID from testdata you want to experiment on')
            print(f'Some Example IDs are {test_matrix["ID"].drop_duplicates().sample(5).values}')
            ID = input('Enter the ID here').strip()
            if ID.lower() == 'exit':
                break
            print('Enter time since placement you wish to evaluate the order at; t = 0 is placement')
            TSP = input('Enter time since placement here').strip()
            if TSP.lower() == 'exit':
                break
            try:
                single_order_eval(int(ID), int(TSP), test_matrix, selected_model, features, scalar)
                
            except ValueError:
                print('Enter a valid ID or TOD (Integer Form)')
    else:
        exit()
    


