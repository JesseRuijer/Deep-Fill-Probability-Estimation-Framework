#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 14:22:54 2026

@author: jesseruijer
"""

"""

UserScript that allows for the majority of the functionality of the framework to be easily accessed by the user

Contains for all models: training, testing, using, evaluation using walkforward window

"""

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import gc
import random
import matplotlib.gridspec as gridspec
import config

from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from matplotlib.ticker import PercentFormatter
from ModelEvaluation import test_model
from FileManager import get_ml_training_paths, get_batch_data_paths, generate_dynamic_paths
from DataAndFeatureEngineering import import_data, clean_data, data_regressors
from FNN import PyTorchSklearnWrapper



#Below is a fix to be able to run lgbm and torch in one script 
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def prep_data_daily(file_path:str, file_path_mo:str) -> dict[str, pd.DataFrame]:
    
    """
    prep data, i.e run dataandfeaturengineering script on it
    copy of function in main but didnt want to import it since UserScript is standalone
    """

    print(f'Runs full pipeline for {os.path.basename(file_path)}')
    rawdata = import_data(file_path, file_path_mo)
    cleandata = clean_data(rawdata)
    matrices = data_regressors(rawdata, cleandata, config.DONT_INCLUDE_FULL_TRAINING_DAY)
    
    del rawdata
    del cleandata
    gc.collect()
    
    return {
        'Binary Matrix': matrices['Binary Matrix']
        }

def save_data() -> None:
    
    """
    Save feature data for ML training in parquet format
    copy of function in main but didnt want to import it since UserScript is standalone
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
    

def get_raw_paths_from_parquet(file_path:str) -> tuple[str, str]:
    
    """
    Function that can be called to obatin the raw files from the parquet treated files, serves as a reverse lookup, 
    used in sequential training to immediately find info not in the parquet files
    
    """
    
    if isinstance(file_path, (list, tuple)):
        file_path = file_path[0]
    
    filename = os.path.basename(file_path)
    
    #example filename: "INTC_BINARY_2014_07_08.parquet"
    name_without_ext = filename.replace('.parquet', '')
    parts = name_without_ext.split('_')
    
    ticker = parts[0]
    #Reconstruct the original 8-digit date string (YYYYMMDD) from the split formatted date
    date = parts[2] + parts[3] + parts[4]
    
    script_dir = Path(__file__).resolve().parent
    raw_dir = script_dir.parent / 'data' / 'raw' / f'{ticker}_NASDAQ'
    
    main_raw_path = raw_dir / f'{ticker}_{date}_NASDAQ.mat'
    mo_raw_path = raw_dir / 'MO' / f'{ticker}_{date}.mat'
    
    if not main_raw_path.exists() or not mo_raw_path.exists():
        print(f"WARNING: Could not locate raw files for {filename}")
        
    return str(main_raw_path), str(mo_raw_path)


def train(train_files:list, train_matrix:pd.DataFrame, model:str) -> tuple[CalibratedClassifierCV | PyTorchSklearnWrapper , StandardScaler]:
    
    """
    Training for all 3 models
    Saving of trained model
    """
    
    print('Starting Training')
    
    if model == 'FNN':

        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
        from FNN import PyTorchSklearnWrapper, DataSet, UserFNN
        
        #Set seed to ensure reproducability for FNN
        SEED = config.RANDOM_SEED
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        
        #additional seeding for GPU and if user uses CUDA
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(SEED)
            
        elif torch.cuda.is_available():
            torch.cuda.manual_seed(SEED)
        
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
            
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
        
        LEARNING_RATE = 0.001    #0.003143153725649377 was the initial LR Optuna gave
        WEIGHT_DECAY = 2.5475947521348294e-06
        EPOCHS = 1
        BATCH_SIZE = 16384 
        
        idx = int(len(train_files) * .8)
                
        active_train_files  = train_files[: idx]
        active_val_files_df = train_files[idx:]
        
        scalar = StandardScaler()
        for f in active_train_files:
            df = pd.read_parquet(f)
            df.replace([np.inf, -np.inf], 0, inplace = True)

            X_train_raw = df[config.FNN_MODEL_FEATURES].values
            
            scalar.partial_fit(X_train_raw) #we cant fit the scalar to all days at once since it crashes RAM so use the partial_fit function that slowly updates the right scaling. It ends up getting the same result as applying a scalar to the whole day, it just doesnt crash RAM, and then later when we are training you just pass the scalar
            
            del df, X_train_raw
            gc.collect()
        print('Pass1')
        
        model = UserFNN(input_size = input_size).to(device)
         
        criterion = nn.BCEWithLogitsLoss(reduction = 'none')    # BCE = Binary Cross Entropy = Logloss, this is just the scoring metric and saying reducion is none, it doesnt do any weighting by iteself it just spits out all the raw values and then with my manual weights i can do the weighint later
        optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY)
        
        #training
        
        for epoch in range(EPOCHS):
             model.train()
             train_loss = 0.0
             total_train_weight = 0.0
             shuffled_files = active_train_files.copy()
             random.shuffle(shuffled_files)
             
             for f in shuffled_files:
                 
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
                     
                     train_loss += weighted_batch_loss.item()
                     total_train_weight += batch_weights.sum().item()

                 
                 del df, X_train_scaled, y_train, w_train, train_dataset, train_loader
                 gc.collect()
             
             
             avg_train_loss = train_loss / total_train_weight
            
            #Evaluation/calibration loop    
            
             model.eval() 
             val_loss = 0.0
             total_val_weight = 0.0
            
             with torch.no_grad():
                
                 for f in active_val_files_df: 
                     df = pd.read_parquet(f)
                     df.replace([np.inf, -np.inf], 0, inplace = True)
                    
                     X_val_scaled = scalar.transform(df[config.FNN_MODEL_FEATURES].values)
                     y_val = df[config.TARGET].values
                     w_val = df['UnitWeight'].values
                    
                     val_dataset = DataSet(X_val_scaled, y_val, w_val)
                     val_loader = DataLoader(dataset = val_dataset, batch_size = BATCH_SIZE, shuffle = False)  
                    
                     for features, labels, batch_weights in val_loader:
                         features, labels, batch_weights = features.to(device), labels.to(device), batch_weights.to(device)
           
                         outputs = model(features)
                         raw_loss = criterion(outputs, labels)
                        
                         val_loss += (raw_loss * batch_weights).sum().item()
                         total_val_weight += batch_weights.sum().item()
                    
                     del df, X_val_scaled, y_val, w_val, val_dataset, val_loader
                     gc.collect()
    
                 avg_val_logloss = val_loss / total_val_weight
                
             print(f' Epoch {epoch + 1} / {EPOCHS}')
             print(f' Weighted Train Logloss: {avg_train_loss:.3f}')
             print(f' Weighted Test Logloss:  {avg_val_logloss:.3f}\n')
        print('Pass2')
        
        print('Adding Calibrator')
        
        model.eval()
        val_raw_probs = []
        val_targets = []
        val_weights = []

        with torch.no_grad():
            for f in active_val_files_df:
                df = pd.read_parquet(f)
                df.replace([np.inf, -np.inf], 0, inplace=True)
                
                X_val_scaled = scalar.transform(df[config.FNN_MODEL_FEATURES].values)
                y_val = df[config.TARGET].values
                w_val = df['UnitWeight'].values
                
                val_dataset = DataSet(X_val_scaled, y_val, w_val)
                val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
                
                for features, labels, batch_weights in val_loader:
                    features = features.to(device)
                    logits = model(features)
                    probs = torch.sigmoid(logits).cpu().numpy().flatten()
                    
                    val_raw_probs.append(probs)
                    val_targets.append(labels.numpy().flatten())
                    val_weights.append(batch_weights.numpy().flatten())
                    
                del df, X_val_scaled, y_val, w_val, val_dataset, val_loader
                gc.collect()

        val_raw_probs = np.concatenate(val_raw_probs)
        val_targets = np.concatenate(val_targets)
        val_weights = np.concatenate(val_weights)

        fnn_calibrator = IsotonicRegression(out_of_bounds='clip')
        fnn_calibrator.fit(val_raw_probs, val_targets, sample_weight=val_weights)
        
        wrapped_fnn = PyTorchSklearnWrapper(model, device, fnn_calibrator)
    
        script_dir = Path(__file__).resolve().parent # file is built in variable holder, resolve and parent to get a string that just contains the directory that contains the script
        models_dir = script_dir.parent / 'models' # /  works as a path joiner
        models_dir.mkdir(parents = True, exist_ok = True) #failsafe to create folder if it doesnt exist or does nothing if it already exists
      
        model_filepath = models_dir / config.USER_FNN_MODEL_WEIGHTS
        torch.save(model.state_dict(), model_filepath)  #The dict contains the model weights, since prepdata function alrady returns a wrapped FNN, add .model to the object to have the pure model 
      
        metadata_filepath = models_dir / config.USER_FNN_MODEL_METADATA
        metadata_package = {
            'features': config.FNN_MODEL_FEATURES,
            'scalar': scalar,
            'calibrator': fnn_calibrator
            }
        print('Pass3')
        joblib.dump(metadata_package, metadata_filepath)
      
        print(f'Succesfully saved FNN weights to  {model_filepath}')
        return wrapped_fnn, scalar
          
    elif model == 'LGBM':
          
        from LightGBMEngine import train_lgbm_model
        
        needed_cols = config.LGBM_MODEL_FEATURES + ['UnitWeight', config.TARGET]
        
        if train_matrix is None:
            train_frames = [pd.read_parquet(f, columns = needed_cols).astype(np.float32) for f in train_files]
            train_matrix = pd.concat(train_frames, ignore_index = True).astype(np.float32)
            del train_frames
            gc.collect()
            
        #Copy Required for final safety check below
        print('Pass1')
        #it was ram spiking in the three lines below here when i used to use copy() so manage ram more efficiently now
        #use .pop gets item and then removes it from df, so by doing that, we dont need to copy anything since removing these two immediatley isolates lr_X
        lgbm_Y = train_matrix.pop(config.TARGET)
        lgbm_w = train_matrix.pop('UnitWeight')
        
        lgbm_X = train_matrix[config.LGBM_MODEL_FEATURES]
        del train_matrix
        
        gc.collect()
      
        # Final safety check
        print('Pass2')
        lgbm_X.replace([np.inf, -np.inf], 0, inplace=True)
        lgbm_X = lgbm_X.astype(np.float32, copy = False)
        print('Pass3')
        
        #use 80% for training and 20% to calibrate on, in chronological order since we have timeseries data 
        split_idx = int(len(lgbm_X) * .8)
    
        train_X = lgbm_X.iloc[:split_idx].copy()
        train_Y = lgbm_Y.iloc[:split_idx].copy()
        train_w = lgbm_w.iloc[:split_idx].copy()
        
        calib_X = lgbm_X.iloc[split_idx:].copy()
        calib_y = lgbm_Y.iloc[split_idx:].copy()
        calib_weights = lgbm_w.iloc[split_idx:].copy()
        
        del lgbm_X, lgbm_Y, lgbm_w
        gc.collect()
   
        base_lgbm, calibrated = train_lgbm_model(train_X, train_Y, train_w, calib_X, calib_y, calib_weights)
        print('Pass4')
        model_package = {
         
          'base_model': base_lgbm,
          'calibrated_model': calibrated,
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
        scalar = None # I know this is ugly code but its just easy if the three models return the same two things from this function 
        return calibrated, scalar
        
    elif model == 'LR':
          
        from LogisticRegressionEngine import train_logistic_model
        needed_cols = config.LOGISTIC_MODEL_FEATURES + ['UnitWeight', config.TARGET]
        
        if train_matrix is None:
            train_frames = [pd.read_parquet(f, columns = needed_cols).astype(np.float32) for f in train_files]
            train_matrix = pd.concat(train_frames, ignore_index = True)
            del train_frames
            gc.collect()
            
        #Copy Required for final safety check below
        print('Pass1')
        #it was ram spiking in the three lines below here when i used to use copy() so manage ram more efficiently now
        #use .pop gets item and then removes it from df, so by doing that, we dont need to copy anything since removing these two immediatley isolates lr_X
       
        lr_Y = train_matrix.pop(config.TARGET)
        lr_w = train_matrix.pop('UnitWeight')
        train_matrix = train_matrix[config.LOGISTIC_MODEL_FEATURES]
        
        lr_X = train_matrix

        gc.collect()
      
        # Final safety check
        print('Pass2')
        lr_X.replace([np.inf, -np.inf], 0, inplace=True)
        print('Pass3')
        #use 80% for training and 20% to calibrate on, in chronological order since we have timeseries data 
        split_idx = int(len(lr_X) * .8)
    
        train_X = lr_X.iloc[:split_idx]
        train_Y = lr_Y.iloc[:split_idx]
        train_w = lr_w.iloc[:split_idx]
        
        calib_X = lr_X.iloc[split_idx:]
        calib_y = lr_Y.iloc[split_idx:]
        calib_weights = lr_w.iloc[split_idx:]
          
        base_lr, calibrated, scalar = train_logistic_model(train_X, train_Y, train_w, calib_X, calib_y, calib_weights)
        print('Pass4')
        model_package = {
            
            'base_model': base_lr,
            'calibrated_model': calibrated,
            'features': config.LOGISTIC_MODEL_FEATURES,
            'scalar' : scalar
            }
     
      #Pathing logic, uses pathlib library so on different devices it still saves correctly, maybe i need to install the os create folder etc stuff for all teh oter things as well
      #so that when i run it on an external computer maybe to train for ex, it automatically creates the right folders on that specific computer
     
        script_dir = Path(__file__).resolve().parent # file is built in variable holder, resolve and parent to get a string that just contains the directory that contains the script
        models_dir = script_dir.parent / 'models' # /  works as a path joiner
        models_dir.mkdir(parents = True, exist_ok = True) #failsafe to create folder if it doesnt exist or does nothing if it already exists
        model_filepath = models_dir / config.USER_LR_MODEL
     
        joblib.dump(model_package, model_filepath)
     
        print(f'Succesfully saved LR model to  {model_filepath}')
        return calibrated, scalar
 
def test_model_wrap(test_matrix:pd.DataFrame, model:str) -> None:
    
    """
    Testing Models using ModelEvaluation Script
    """
    
    # Rebuild the absolute path using pathlib
    script_dir = Path(__file__).resolve().parent 
    models_dir = script_dir.parent / 'models'
    
    if model == 'FNN':
        
        import torch
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
        test_model(
            test_data = test_matrix, 
            base_model = base_lgbm,
            calibrated_model = calibrated_lgbm,
            scalar = None,
            model_name = 'Light Gradient Boosted Model',
            features = features,
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
       
        test_model(
            test_data = test_matrix, 
            base_model = base_lr,
            calibrated_model = calibrated_lr,
            scalar = scalar_lr,
            model_name = 'Logistic Regression',
            features = features,
        )
       
 
def single_order_eval(ID:int, TSP:float, test_data:pd.DataFrame, selected_model:PyTorchSklearnWrapper | CalibratedClassifierCV, features:list[str], scalar: StandardScaler) -> float:
    
    """
    Use Model to Return fill probability for a given order ID at a certain time since placement TSP
    """
    
    order_data = test_data[test_data['ID'] == ID].copy()
    if len(order_data) == 0:
        print('ID Not Found')
        return
    
    order_data = order_data.sort_values('TimeSincePlacement')
    print(f"Available range for ID {ID}: {order_data['TimeSincePlacement'].min()}ms to {order_data['TimeSincePlacement'].max()}ms")
    if TSP < order_data['TimeSincePlacement'].min():
        print('Order Does not exist yet at this time')
        return
    if TSP > order_data['TimeSincePlacement'].max():
        print('Order Does is dead already at this time')
        return
    
    state_at_tsp = order_data[order_data['TimeSincePlacement'] <= TSP].iloc[-1:]    #Get the closest entry now or in the past to the requested TSP
        
    X_raw = state_at_tsp[features]
    
    if scalar is not None:
        X = scalar.transform(X_raw)
    else:
        X = X_raw
        
    prob = selected_model.predict_proba(X)[0,1]
   
    print(f'At {TSP}ms into this order ID life, the fill probability is {prob*100:.2f}%')

    return prob

def calc_daily_qimbal(test_matrix:pd.DataFrame, use_model: PyTorchSklearnWrapper | CalibratedClassifierCV, scalar: StandardScaler, features:list[str], mo_data:pd.DataFrame, cleandata:pd.DataFrame) -> pd.DataFrame:
    
    """
    
    Calculates probqimbalance and qimbalance at any moment during trading hours for a trading day at any given price level
    
    Uses vectorized code using pandas to keep a running tally of Vol at price
    
    """
    
    X_raw = test_matrix[features].astype(np.float32, copy = False)
    X = scalar.transform(X_raw) if scalar else X_raw 
    
    test_matrix['fillprob'] = use_model.predict_proba(X)[:,1]
    
    #Calculate prob weighted vol from our model to create probqimbalance
    test_matrix['expvol'] = test_matrix['fillprob'] * test_matrix['Vol']  
    
    all_events = test_matrix[['ID', 'TOD', 'Type', 'SideOfBook', 'expvol', 'Price']].copy()
    
    # Type 68/70 are dropped upstream in data_regressors, so every order's real final
    # removal is invisible to test_matrix. Pull those events straight from the raw log --
    # no prediction needed, a confirmed-dead order's contribution is just 0.
    
    full_removals = cleandata['Event'][cleandata['Event']['Type'].isin([68, 70])][['ID', 'TOD', 'Type', 'SideOfBook', 'Price']].copy()
    full_removals['expvol'] = 0.0
    
   # Combine into one big chronological list of events
    combined = pd.concat([all_events, full_removals], ignore_index=True)
    combined = combined.sort_values(['ID', 'TOD'])
    
    # Using .shift(), we look at the row immediately above to see what this order's volume and price were BEFORE this event
    combined['prev_vol'] = combined.groupby('ID')['expvol'].shift(1).fillna(0.0)
    combined['prev_price'] = combined.groupby('ID')['Price'].shift(1)
    
   # The Negative Impact: Every event overwrites the old state, so we must subtract the previous volume
    subs = combined[['TOD', 'SideOfBook', 'prev_price', 'prev_vol']].copy()
    #just rename the columns from above
    subs.columns = ['TOD', 'SideOfBook', 'Price', 'Vol_Delta']
    subs['Vol_Delta'] = -subs['Vol_Delta'] 
    subs = subs.dropna(subset=['Price']) 
    
    # The Positive Impact: Every event applies a new state, so we add the new volume
    adds = combined[['TOD', 'SideOfBook', 'Price', 'expvol']].copy()
    adds.columns = ['TOD', 'SideOfBook', 'Price', 'Vol_Delta']
    
    # Combine adds and subs to get a master timeline of all volume changes
    impacts = pd.concat([adds, subs], ignore_index=True)
   # Group by exact microsecond and price to net out simultaneous adds/subs
    impacts = impacts.groupby(['TOD', 'SideOfBook', 'Price'])['Vol_Delta'].sum().reset_index()

    # Build the full Limit Order Book history using cumulative sum
    impacts = impacts.sort_values('TOD')
    impacts['Resting_Vol'] = impacts.groupby(['SideOfBook', 'Price'])['Vol_Delta'].cumsum()

    # Separate the book by side for cleaner merging
    bid_book = impacts[impacts['SideOfBook'] == 1].drop(columns=['SideOfBook'])
    ask_book = impacts[impacts['SideOfBook'] == 0].drop(columns=['SideOfBook'])
    
    #match datatypes so pd.merge can continue, as we scaled down data during processing before
    bid_book['TOD'] = bid_book['TOD'].astype('int64')
    ask_book['TOD'] = ask_book['TOD'].astype('int64')
    test_matrix['TOD'] = test_matrix['TOD'].astype('int64')

    bid_book['Price_Key'] = bid_book['Price'].abs().round().astype('int64')
    ask_book['Price_Key'] = ask_book['Price'].abs().round().astype('int64')

    test_matrix['BestBid_Key'] = test_matrix['BestBid'].abs().round().astype('int64')
    test_matrix['BestAsk_Key'] = test_matrix['BestAsk'].abs().round().astype('int64')

    # Map the resting volume back to the main test_matrix BBO
    test_matrix = test_matrix.sort_values('TOD')
    
   # Merge Bid Volume
    test_matrix = pd.merge_asof(
        test_matrix, 
        bid_book[['TOD', 'Price_Key', 'Resting_Vol']], 
        on='TOD', 
        left_by ='BestBid_Key', 
        right_by ='Price_Key', 
        direction='backward'
    )
    test_matrix.rename(columns={'Resting_Vol': 'Total_Prob_Bid_Vol'}, inplace=True)

    # Merge Ask Volume
    test_matrix = pd.merge_asof(
        test_matrix, 
        ask_book[['TOD', 'Price_Key', 'Resting_Vol']], 
        on='TOD', 
        left_by='BestAsk_Key', 
        right_by='Price_Key', 
        direction='backward'
    )
    test_matrix.rename(columns={'Resting_Vol': 'Total_Prob_Ask_Vol'}, inplace=True)
    

    # Clean up missing data (if a price level hasn't been established yet) and drop the extra merge columns
    test_matrix[['Total_Prob_Bid_Vol', 'Total_Prob_Ask_Vol']] = test_matrix[['Total_Prob_Bid_Vol', 'Total_Prob_Ask_Vol']].fillna(0)
    test_matrix.drop(columns=['Price_x', 'Price_y'], inplace=True, errors='ignore')
    
    test_matrix['Total_Prob_Bid_Vol'] = test_matrix['Total_Prob_Bid_Vol'].round(4).clip(lower=0)
    test_matrix['Total_Prob_Ask_Vol'] = test_matrix['Total_Prob_Ask_Vol'].round(4).clip(lower=0)
    
    prob_denom = test_matrix['Total_Prob_Bid_Vol'] + test_matrix['Total_Prob_Ask_Vol']
    
    # Calculate Imbalance, defaulting to 0 if the touch is completely empty
    test_matrix['ProbQImbal'] = np.where(
        prob_denom > 0, 
        (test_matrix['Total_Prob_Bid_Vol'] - test_matrix['Total_Prob_Ask_Vol']) / prob_denom,
        0.0
    )
    
    test_matrix['y_true'] = test_matrix[config.TARGET]
    mo_data['TOD'] = mo_data['TOD'].astype('int64')
    test_matrix['TOD'] = test_matrix['TOD'].astype('int64')
    
    #Create final df 
    merged = pd.merge_asof(
        mo_data.sort_values('TOD'), 
        test_matrix[['TOD', 'ProbQImbal', 'QImbalance', 'y_true', 'fillprob']].sort_values('TOD'), 
        on='TOD', 
        direction='backward'
    )
    
    #Drop NaNs for trades that were placed before opening hours
    merged = merged.dropna(subset=['ProbQImbal', 'QImbalance', 'fillprob'])
    
    return merged    
    
def walk_forward(all_data_paths:list[str], selected_model:str, train_window_days:int) -> None:
    
    """
    Walk forward model evaluation function
    Coordinates daily computation of graphs and metrics and rolls this forward
    Calls average plotting and metric results across entire walk forward testing set
    
    """
    
    from ModelEvaluation import compute_daily_performance_curve
    from ModelEvaluation import compute_daily_divergence
    from ModelEvaluation import compute_daily_alligator
    from ModelEvaluation import compute_daily_PR
    from ModelEvaluation import compute_daily_scores
    
    #initialize performance lists
    daily_curves = []
    divergence_curves = []
    alligator_curves = []
    PR_curves = []
    model_scores = []
    
    total_test_days = len(all_data_paths) - train_window_days
   
    for i in range(total_test_days):
        
        #obtain evaluation for each trainingset and corresponding following testday using ModelEvaluation script
        
        train_files = all_data_paths[i : i + train_window_days]
        test_file = all_data_paths[i + train_window_days]
       
        use_model, scalar = train(train_files, None, selected_model)
        
        gc.collect()
        
        test_matrix = pd.read_parquet(test_file)
        test_matrix.replace([np.inf, -np.inf], 0, inplace=True)
        
        tsp = test_matrix['TimeSincePlacement']
        order_ids = test_matrix['ID']
        
        if selected_model == 'FNN':
            features = config.FNN_MODEL_FEATURES
        elif selected_model == 'LR':
            features = config.LOGISTIC_MODEL_FEATURES
        elif selected_model == 'LGBM':
            features = config.LGBM_MODEL_FEATURES
            
        X_raw = test_matrix[features].astype(np.float32, copy = False)   
        X = scalar.transform(X_raw) if scalar else X_raw    
        y_pred_prob = use_model.predict_proba(X)[:,1]    
        y_true = test_matrix[config.TARGET]
        weights = test_matrix['UnitWeight']
        
        #performance curve
        day_pred, day_actual, day_vol = compute_daily_performance_curve(y_true, y_pred_prob, weights, mask = None)
        daily_curves.append((day_pred, day_actual, day_vol))

        #alligator curve
        day_fills, day_cancels = compute_daily_alligator(y_true, y_pred_prob, weights, tsp, order_ids)
        alligator_curves.append((day_fills, day_cancels))
        
        #pr curve
        day_precision, day_recall = compute_daily_PR(y_true, y_pred_prob, weights)
        PR_curves.append((day_precision, day_recall))
        
        #daily performance scores
        daily_scores = compute_daily_scores(y_true, y_pred_prob, weights)
        model_scores.append(daily_scores)
        
        #prepping and executing probqimbal vs qimbal graphs div graphs
        raw_data_path, raw_mo_path = get_raw_paths_from_parquet(test_file)
        rawdata = import_data(raw_data_path, raw_mo_path)
        cleandata = clean_data(rawdata)
        mo_data = rawdata['MO']
        daily_merged = calc_daily_qimbal(test_matrix, use_model, scalar, features, mo_data, cleandata)
        reg_buy, reg_tot, prob_buy, prob_tot = compute_daily_divergence(daily_merged)
        divergence_curves.append((reg_buy, reg_tot, prob_buy, prob_tot))
        
        del test_matrix, rawdata, cleandata, mo_data, daily_merged
        gc.collect()
    
    #having obtained results on all test days plot averaged results
    plot_walk_forward_curves(daily_curves, selected_model)
    plot_walk_forward_div(divergence_curves, selected_model)
    plot_walk_forward_alligator(alligator_curves, selected_model) 
    plot_walk_forward_PR(PR_curves, selected_model)
    prnt_daily_scores(model_scores, selected_model)
    
    
def plot_walk_forward_curves(daily_curves:list[tuple[np.ndarray, np.ndarray, np.ndarray]], model_name:str) -> None:
    
    """
    Plots the average performance curve alongside all individual test day performance curves over the testing set
    """

    #initialize 
    all_preds = []
    all_actuals = []
    all_vols = []

    for day_pred, day_actual, day_vol in daily_curves:
        all_preds.append(day_pred)
        all_actuals.append(day_actual)
        all_vols.append(day_vol)
       
    #compute mean of bins ignoring nan bins
    mean_pred = np.nanmean(all_preds, axis = 0)
    mean_acc = np.nanmean(all_actuals, axis = 0)
    total_vol = np.sum(all_vols, axis = 0)
    
    #Create bins for plot
    deltap = 0.01
    bins_low = np.arange(0, 0.401, deltap)
    bins_high = np.arange(0.43, 1, 3 * deltap)
    bins_custom = np.concatenate((bins_low, bins_high))
    middle = [(bins_custom[i] + bins_custom[i+1])/2 for i in range(len(bins_custom)-1)]
    
    #main performance plot
    plt.figure(figsize=(20, 10))
    gs1 = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
    ax1 = plt.subplot(gs1[0])

    for day_pred, day_actual, _ in daily_curves:
        #Filter nans i.e empty bins out, ~ is numpy NOT operator and we need more thatn one point to draw a line bewteen points, thats the following logic
        valid_mask = ~np.isnan(day_pred) & ~np.isnan(day_actual)
        if valid_mask.sum() > 1:
            ax1.plot(day_pred[valid_mask], day_actual[valid_mask], color='blue', alpha=0.15, linewidth=1)

    valid_mean = ~np.isnan(mean_acc) & ~np.isnan(mean_pred)
    ax1.plot(mean_pred[valid_mean], mean_acc[valid_mean], color='darkblue', linewidth=5, label=f'Avg of {model_name}')
    ax1.plot([0,1], [0,1], color='black', label='Perfect', linestyle='--')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax1.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax1.grid(True, alpha=0.3)
    ax1.set_ylabel('Actual Fill Prob')
    ax1.set_title(f'Performance of {model_name} on {config.TICK} ')
    ax1.legend()

    #subplot to see vol that appears in each bin used in main graph
    ax2 = plt.subplot(gs1[1], sharex=ax1)
    ax2.bar(middle, total_vol, width=deltap*0.8, color='black', alpha=0.8, label='Aggregated Volume')
    ax2.set_ylabel('Total Vol (Log)')
    ax2.set_xlabel('Predicted Fill Probability')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.subplots_adjust(hspace=0.1)
    plt.show()
 

def plot_walk_forward_div(divergence_curves:list[tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]], model_name:str) -> None:
    
    """
    Plots the performance for probqimbal vs qimbal alongside all individual test day performance curves over the testing set
    
    Is called div, since I was looking for divergence between the performance lines 
    """
    
    plt.figure(figsize=(9, 6))

    all_reg_buy = np.array([day[0] for day in divergence_curves], dtype=float)
    all_reg_tot = np.array([day[1] for day in divergence_curves], dtype=float)
    all_prob_buy = np.array([day[2] for day in divergence_curves], dtype=float)
    all_prob_tot = np.array([day[3] for day in divergence_curves], dtype=float)
    
    bins = np.linspace(-1.0, 1.0, 101)
    x_mids = (bins[:-1] + bins[1:]) / 2 
    
    # Calculate the mathematical average across the month per bin
    daily_reg_curve = np.divide(all_reg_buy, all_reg_tot, 
                                out=np.full_like(all_reg_buy, np.nan), 
                                where=all_reg_tot!=0)
    
    daily_prob_curve = np.divide(all_prob_buy, all_prob_tot, 
                                 out=np.full_like(all_prob_buy, np.nan), 
                                 where=all_prob_tot!=0)

    global_reg_tot = np.sum(all_reg_tot, axis=0)
    global_prob_tot = np.sum(all_prob_tot, axis=0)
    
    sum_reg_buy = np.sum(all_reg_buy, axis=0)
    sum_prob_buy = np.sum(all_prob_buy, axis=0)
    
    mean_reg = np.divide(sum_reg_buy, global_reg_tot, 
                         out=np.full_like(global_reg_tot, np.nan), 
                         where=global_reg_tot!=0)
                         
    mean_prob = np.divide(sum_prob_buy, global_prob_tot, 
                          out=np.full_like(global_prob_tot, np.nan), 
                          where=global_prob_tot!=0)

    bins = np.linspace(-1.0, 1.0, 101)
    x_mids = (bins[:-1] + bins[1:]) / 2 
    
    plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    ax1 = plt.subplot(gs[0])

    for i in range(len(daily_reg_curve)):
        valid = ~np.isnan(daily_reg_curve[i])
        if valid.sum() > 1:
            label = 'Daily Regular' if i == 0 else None
            ax1.plot(x_mids[valid], daily_reg_curve[i][valid], color='red', alpha=0.15, linewidth=1, label=label)
            
    for i in range(len(daily_prob_curve)):
        valid = ~np.isnan(daily_prob_curve[i])
        if valid.sum() > 1:
            label = 'Daily Improved' if i == 0 else None
            ax1.plot(x_mids[valid], daily_prob_curve[i][valid], color='blue', alpha=0.15, linewidth=1, label=label)

    valid_mean_reg = ~np.isnan(mean_reg)
    ax1.plot(x_mids[valid_mean_reg], mean_reg[valid_mean_reg], color='darkred', linewidth=5, label='Weighted Avg Regular')
    
    valid_mean_prob = ~np.isnan(mean_prob)
    ax1.plot(x_mids[valid_mean_prob], mean_prob[valid_mean_prob], color='darkblue', linewidth=5, label=f'Weighted Avg Prob ({model_name})')

    ax1.axvline(x=0.0, color='gray', linestyle='--', label='Neutral Book (0.0)')
    ax1.axhline(y=0.5, color='gray', linestyle=':', label='50/50 Split')
    
    ax1.set_title(f'Improved vs Reg Qimbal using {model_name} on {config.TICK}')
    ax1.set_ylabel('Proportion of MOs that are Buys')
    ax1.set_xlim(-1.0, 1.0)
    ax1.set_ylim(0, 1)
    ax1.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = plt.subplot(gs[1], sharex=ax1)
    
    
    #width to seperate bins a bit but dont wanna shift logic so check this in plot 
    width = (x_mids[1] - x_mids[0]) * 0.4
    ax2.bar(x_mids - width/2, global_reg_tot, width=width, color='darkred', alpha=0.5, label='Reg Vol')
    ax2.bar(x_mids + width/2, global_prob_tot, width=width, color='darkblue', alpha=0.5, label='Prob Vol')
    
    ax2.set_ylabel('Total MO Vol (Log)')
    ax2.set_xlabel('Imbalance (-1.0 to 1.0)')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.subplots_adjust(hspace=0.05)
    plt.show()
    
    #For bins above and below qimbal 0 how far are we from the 0.5 line, i.e how good of an indicator are we

    sign_mask = np.where(x_mids >= 0, 1.0, -1.0)

    daily_dev_reg  = np.nanmean((daily_reg_curve  - 0.5) * sign_mask, axis=1) #take average across bins for the 40 bins
    daily_dev_prob = np.nanmean((daily_prob_curve - 0.5) * sign_mask, axis=1)
    
    #Calculate the daily percentage improvement for each of the 15 test days
    with np.errstate(divide='ignore', invalid='ignore'): #silences warnings since we take care of nans later anyway 
        daily_improvement = (daily_dev_prob - daily_dev_reg) * 100
    
    daily_improvement = np.where(np.isfinite(daily_improvement), daily_improvement, np.nan)

    print("\n--- Signal Strength (Walk-Forward Mean +-Daily STD across Days) ---")
    print(f"Raw QImbal Mean Deviation:  {np.nanmean(daily_dev_reg):.4f} +- {np.nanstd(daily_dev_reg, ddof=1):.4f}") #ddof =1 is to get the 1/N-1 for std which we need since we work with a sample
    print(f"Prob QImbal Mean Deviation: {np.nanmean(daily_dev_prob):.4f} +- {np.nanstd(daily_dev_prob, ddof=1):.4f}")
    print(f"Signal Strength Gain:       {np.nanmean(daily_improvement):+.1f}% +- {np.nanstd(daily_improvement, ddof=1):.1f}%\n")
    
    
def plot_walk_forward_alligator(alligator_curves, model_name):
    plt.figure(figsize=(16, 9))

    # Force 2D float alignment using the ragged-array fix
    all_fills = pd.DataFrame([day[0] for day in alligator_curves]).values.astype(float)
    all_cancels = pd.DataFrame([day[1] for day in alligator_curves]).values.astype(float)
    
    x_axis = np.linspace(0, 1, 20)
    
    for i, fill in enumerate(all_fills):
        valid = ~np.isnan(fill)
        if valid.sum() > 1:
            label = 'Daily Filled' if i == 0 else None
            plt.plot(x_axis[valid], fill[valid], color='green', alpha=0.15, linewidth=1, label=label)
            
    for i, cancel in enumerate(all_cancels):
        valid = ~np.isnan(cancel)
        if valid.sum() > 1:
            label = 'Daily Canceled' if i == 0 else None
            plt.plot(x_axis[valid], cancel[valid], color='red', alpha=0.15, linewidth=1, label=label)
    
    # Calculate the mathematical average across the walk-forward days per bin
    mean_fills = np.nanmean(all_fills, axis=0)
    mean_cancels = np.nanmean(all_cancels, axis=0)

    plt.plot(x_axis, mean_fills, 'o-', color='green', label='Eventually filled')
    plt.plot(x_axis, mean_cancels, 'o-', color='red', label='Eventually canceled')
    
    plt.plot(x_axis, mean_cancels, color='darkred', linewidth=5, label='Average Cancel')
    plt.plot(x_axis, mean_fills, color='darkgreen', linewidth=5, label='Average Fill')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.title(f'Predicted Fill Probability over lifetime by eventual outcome {model_name} - {config.TICK}')
    plt.xlabel('Normalized order lifetime (0=placement, 1=death)')
    plt.ylabel('Average Fill Probability')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
def plot_walk_forward_PR(PR_curves, model_name):

    #Sklearn spits out arrays of PR of different sizes, so cant directly compare them like above where we had the fixed bins
    #Use fixed points where we evaluate the PR
    
    plt.figure(figsize=(16, 9))

    common_recall = np.linspace(0, 1, 101)
    all_interp_precisions = []
    
    for day_precision, day_recall in PR_curves:
        #The following is from SKlearn documentation
        # Sklearn returns recall in decreasing order (1 to 0). 
        # np.interp requires the x-axis to be strictly increasing.
        # use [::-1] slicing to reverse order in which we read array.
        rev_recall = day_recall[::-1]
        rev_precision = day_precision[::-1]
        # use np.interp to interpolate this days precision onto our common grid
        interp_prec = np.interp(common_recall, rev_recall, rev_precision)
        all_interp_precisions.append(interp_prec)
        
        # Plot the daily transparent ghost line (using original arrays)
        plt.plot(day_recall, day_precision, color='blue', alpha=0.15, linewidth=1)

    mean_precision = np.mean(all_interp_precisions, axis=0)
    plt.plot(common_recall, mean_precision, color='darkblue', linewidth=5, label=f'Avg {model_name} PR')

    plt.xlabel('Recall (Fills correctly identified as Fills / All Actual Fills)')
    plt.ylabel('Precision (Fills Correctly Identified / All Orders Predicted To Fill)')
    plt.title(f'Walk-Forward Precision-Recall Curve - {model_name} - {config.TICK}')
    plt.gca().xaxis.set_major_formatter(PercentFormatter(1.0))
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
def prnt_daily_scores(daily_scores, model_name):
    
    print(f'Dailiy Scores for {model_name} -{config.TICK}')
    
    for i, daily_score in enumerate(daily_scores):

        for metric_name, value in daily_score.items(): #.items() immediatley gets key string from dic and value attached to it
            print(f'{metric_name}: {value:.4f}')
            
    df_scores = pd.DataFrame(daily_scores)

    mean_scores = df_scores.mean() 
    std_scores = df_scores.std()
    
    for metric_name in mean_scores.index:
        print(f'Mean {metric_name}: {mean_scores[metric_name]:.3f} +- STD {std_scores[metric_name]}')


    
if __name__ == "__main__":
    
    print('Starting Programme')
    
    process_choice = input('Do you want to process Raw files (y/n) ').strip().lower()
    if process_choice == 'y':
        save_data()
    else:
        print('Moving on to preprocessed data')
    
    print('Please select TRAINING Data (atleast two days)')
    print('Please select Test Data (One day only and no overlap with test data (obviously))')

    print('Here just select again the training day from above, its for plotting ProbQImbal we need the MO data')  
    paths = get_ml_training_paths()
    #the second argument in .get is just an empty list it will proceed with if it cant find the list belonging to the key in the first argument for safety reasons
    train_files = paths.get('train_bin', [])
    test_files = paths.get('test_bin', [])
   
    
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
    print('WARNING: TRAINING IS REQUIRED BEFORE TESTING (whenever you select new data)')
    action_choice = input("Do you want to 'train' or 'test' or 'use' or 'eval' this model? ").strip().lower()
    
    if action_choice in ['test', 'qimbal', 'use']:
        test_matrix = pd.read_parquet(test_files) 
        data_path, mo_path = get_raw_paths_from_parquet(test_files)
        rawdata = import_data(data_path, mo_path)
        mo_data = rawdata['MO']
        cleandata = clean_data(rawdata)

    if action_choice == 'train':
        
        train(train_files, None, selected_model)
        
        
    elif action_choice == 'test':
        
        test_model_wrap(test_matrix, selected_model)
        
    elif action_choice == 'eval':
        #just manually force a list on the one item in tesst files so we can concatenate in sorted below 
        #sorted immediately gives the right order by itself due the naming of our data
        chronological_data = sorted(train_files + test_files)
        walk_forward(chronological_data, selected_model, train_window_days = config.TRAINDAYSWF)
        
   
    elif action_choice == 'use':
        print(f'Use the {selected_model} to experiment on orders, type "exit" if you want to stop')
        
        script_dir = Path(__file__).resolve().parent 
        models_dir = script_dir.parent / 'models'
        
        if selected_model == 'FNN':
            import torch
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
            fnn_calibrator = metadata_package['calibrator']
            
            #re initiaze the 'empty' model blue print
            input_size = len(config.FNN_MODEL_FEATURES)
            loaded_model = UserFNN(input_size = input_size).to(device)
            
            #fill the empty model with my loaded weights from training before
            loaded_model.load_state_dict(torch.load(model_filepath, map_location = device))
            loaded_model.eval()
            use_model = PyTorchSklearnWrapper(loaded_model, device, fnn_calibrator)
        
        elif selected_model == 'LR':
            model_filepath = models_dir / config.USER_LR_MODEL
            
            print(f'Loading LR from {model_filepath}')
            
            # #Load the package using the dynamic path
            loaded_model_package = joblib.load(model_filepath)
            
            #Extracting the contents from the dictionary
            features = loaded_model_package['features']
            scalar = loaded_model_package['scalar']
            base_lr = loaded_model_package['base_model']
            use_model = loaded_model_package['calibrated_model']
            
        elif selected_model == 'LGBM':
            model_filepath = models_dir / config.USER_LGBM_MODEL
            print(f'Loading LGBM from {model_filepath}')
            
            # #Load the package using the dynamic path
            loaded_model_package = joblib.load(model_filepath)
            
            #Extracting the contents from the dictionary
            features = loaded_model_package['features']
            base_lgbm = loaded_model_package['base_model']
            use_model = loaded_model_package['calibrated_model']
            scalar = None
    
        #This below is just temporary stuff to see if the use part works and yes it does, use the exploratory data part for that 
        X_raw = test_matrix[features].astype(np.float32, copy=False).values
        X_scaled = scalar.transform(X_raw) if scalar else X_raw 
        
        # 2. Isolate ONLY the Fill Probabilities (Column 1)
        fill_probs = use_model.predict_proba(X_scaled)[:, 1]
        
        # 3. Find the index of the absolute highest fill probability
        idx = np.argmax(fill_probs)
        
        # 4. Extract the exact ID, TSP, and Probability
        highest_id = int(test_matrix['ID'].iloc[idx])
        highest_tsp = int(test_matrix['TimeSincePlacement'].iloc[idx])
        highest_prob = fill_probs[idx]
        
        print("\n" + "="*50)
        print("MAX FILL PROBABILITY FOUND")
        print(f"Order ID: {highest_id}")
        print(f"Time Since Placement: {highest_tsp} ms")
        print(f"Fill Probability: {highest_prob * 100:.2f}%")
        print("="*50 + "\n")

        while True:
            
            print('Give Order ID from testdata you want to experiment on')
            print(f'Some Example IDs are {test_matrix["ID"].drop_duplicates().sample(5).values}')
            ID = input('Enter the ID here ').strip()
            if ID.lower() == 'exit':
                break
            print('Enter time since placement you wish to evaluate the order at; t = 0 is placement ')
            TSP = input('Enter time since placement here ').strip()
            if TSP.lower() == 'exit':
                break
            try:
                clean_ID = int(ID)
                clean_TSP = int(TSP)
                
            except ValueError:
                print('Enter a valid ID or TOD (Integer Form)')
                continue
            single_order_eval(clean_ID, clean_TSP, test_matrix, use_model, features, scalar)

    else:
        exit()
    


