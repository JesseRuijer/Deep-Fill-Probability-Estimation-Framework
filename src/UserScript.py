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
from sklearn.metrics import roc_auc_score
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
        'Binary Matrix': matrices['Binary Matrix']
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
        binary_file_dest = generate_dynamic_paths(main_path)
        
        print(f'\n--- Processing: {os.path.basename(main_path)} ---')
        
        matrices = prep_data_daily(main_path, mo_path)

        matrices['Binary Matrix'].to_parquet(binary_file_dest)
        #matrices['Multi Matrix'].to_parquet(multi_file_dest)
        
        print(f'Saved -> {os.path.basename(binary_file_dest)}')
        #print(f'Saved -> {os.path.basename(multi_file_dest)}')
        
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
        
        needed_cols = config.LGBM_MODEL_FEATURES + ['UnitWeight', config.TARGET]
        
        if train_matrix is None:
            train_frames = [pd.read_parquet(f, columns = needed_cols).astype(np.float32) for f in train_files]
            train_matrix = pd.concat(train_frames, ignore_index = True)
            del train_frames
            gc.collect()
            
        #Copy Required for final safety check below
        print('Pass1')
        #it was ram spiking in the three lines below here when i used to use copy() so manage ram more efficiently now
        #use .pop gets item and then removes it from df, so by doing that, we dont need to copy anything since removing these two immediatley isolates lr_X
        lgbm_Y = train_matrix.pop(config.TARGET)
        lgbm_w = train_matrix.pop('UnitWeight')
        
        
        lgbm_X = train_matrix
        
        gc.collect()
      
        # Final safety check
        print('Pass3')
        lgbm_X.replace([np.inf, -np.inf], 0, inplace=True)
        print('Pass4')
        
      
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
        scalar = None # I know this is ugly code but its just easy if the three models return the same two things from this function 
        return calibrated_lgbm, scalar
        
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
        
        
        lr_X = train_matrix

        gc.collect()
      
        # Final safety check
        print('Pass3')
        lr_X.replace([np.inf, -np.inf], 0, inplace=True)
        print('Pass4')
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
        return calibrated_lr, scalar_lr
 
def improve_qimbal(test_matrix, model, mo_data, cleandata):
    # Rebuild the absolute path using pathlib
    script_dir = Path(__file__).resolve().parent 
    models_dir = script_dir.parent / 'models'
    
    scalar = None
 
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
        
        #re initiaze the 'empty' model blue print
        input_size = len(config.FNN_MODEL_FEATURES)
        loaded_model = NN(input_size = input_size).to(device)
        
        #fill the empty model with my loaded weights from training before
        loaded_model.load_state_dict(torch.load(model_filepath, map_location = device))
        
        
        #set to eval before testing
        
        loaded_model.eval()
        
        #Wrap so can use test_model function 
        use_model = PyTorchSklearnWrapper(loaded_model, device)
        
    elif model == 'LGBM':
        model_filepath = models_dir / config.USER_LGBM_MODEL
        
        print(f'Loading LGBM from {model_filepath}')
        
        # #Load the package using the dynamic path
        loaded_model_package = joblib.load(model_filepath)
        
        #Extracting the contents from the dictionary
        features = loaded_model_package['features']
        base_lgbm = loaded_model_package['base_model']
        use_model = loaded_model_package['calibrated_model']
            
    elif model == 'LR':

        model_filepath = models_dir / config.USER_LR_MODEL
        
        print(f'Loading LR from {model_filepath}')
        
        # #Load the package using the dynamic path
        loaded_model_package = joblib.load(model_filepath)
        
        #Extracting the contents from the dictionary
        features = loaded_model_package['features']
        scalar_lr = loaded_model_package['scalar']
        base_lr = loaded_model_package['base_model']
        use_model = loaded_model_package['calibrated_model']
        
    X_raw = test_matrix[features].astype(np.float32, copy = False)
    X = scalar.transform(X_raw) if scalar else X_raw 
             
    #This code below is now state based and doesnt need type it just looks at the delta change in volume for each step 
    
    
    test_matrix['fillprob'] = use_model.predict_proba(X)[:,1]
    test_matrix['expvol'] = test_matrix['fillprob'] * test_matrix['Vol']  #Calculate prob weighted vol 
    
    all_events = test_matrix[['ID', 'TOD', 'Type', 'SideOfBook', 'expvol', 'Price']].copy()
    
    #at_touch = test_matrix[test_matrix['DistanceToTouch'] == 0].copy()
    
    # Type 68/70 are dropped upstream in data_regressors, so every order's real final
    # removal is invisible to test_matrix. Pull those events straight from the raw log --
    # no prediction needed, a confirmed-dead order's contribution is just 0.
    
    #touched_ids = set(at_touch['ID'].unique())
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

    # 3. Build the full Limit Order Book history using cumulative sum
    impacts = impacts.sort_values('TOD')
    impacts['Resting_Vol'] = impacts.groupby(['SideOfBook', 'Price'])['Vol_Delta'].cumsum()

    # Separate the book by side for cleaner merging
    bid_book = impacts[impacts['SideOfBook'] == 1].drop(columns=['SideOfBook'])
    ask_book = impacts[impacts['SideOfBook'] == 0].drop(columns=['SideOfBook'])
    
    #match datatypes so pd.merge can continue, as we scaled down data during processing before
# Force the merge keys to standard 64-bit types so the C-backend doesn't panic
    bid_book['TOD'] = bid_book['TOD'].astype('int64')
    ask_book['TOD'] = ask_book['TOD'].astype('int64')
    test_matrix['TOD'] = test_matrix['TOD'].astype('int64')

    bid_book['Price_Key'] = bid_book['Price'].abs().round().astype('int64')
    ask_book['Price_Key'] = ask_book['Price'].abs().round().astype('int64')

    test_matrix['BestBid_Key'] = test_matrix['BestBid'].abs().round().astype('int64')
    test_matrix['BestAsk_Key'] = test_matrix['BestAsk'].abs().round().astype('int64')

    # 4. Map the resting volume back to the main test_matrix BBO
    test_matrix = test_matrix.sort_values('TOD')
   # Merge Bid Volume
    # This magically finds the most recent Resting_Vol where the TOD matches AND the book Price matches the BestBid
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
    
    
    
    # 3. Calculate Imbalance, defaulting to 0 if the touch is completely empty
    test_matrix['ProbQImbal'] = np.where(
        prob_denom > 0, 
        (test_matrix['Total_Prob_Bid_Vol'] - test_matrix['Total_Prob_Ask_Vol']) / prob_denom,
        0.0
    )
  # 1. Map the calculated imbalances onto the actual Market Order timestamps
  
  # --- DIAGNOSTIC CHECK ---
    print("\n" + "="*40)
    print("1. AVERAGE FILL PROBABILITY BY SIDE (Model Check)")
    print('THIS IS HIGH SINCE MODEL ONLY EVALUATED AT BEST PRICES')
    print(test_matrix.groupby('SideOfBook')['fillprob'].mean())
    
    print("\n2. AVERAGE MAPPED VOLUME BY SIDE (Matching Check)")
    print(f"Mean Mapped Bid Vol: {test_matrix['Total_Prob_Bid_Vol'].mean():.2f}")
    print(f"Mean Mapped Ask Vol: {test_matrix['Total_Prob_Ask_Vol'].mean():.2f}")
    print("="*40 + "\n")
    
    # --- DEFINITIVE DIAGNOSTIC CHECK ---
    print("\n" + "="*50)
    print("1. RAW VOLUME IN TEST MATRIX (Input Check)")
    raw_vols = test_matrix.groupby('SideOfBook')['Vol'].sum()
    print(f"Total Raw Bid Vol (Side 1): {raw_vols.get(1.0, 0):,.0f}")
    print(f"Total Raw Ask Vol (Side 0): {raw_vols.get(0.0, 0):,.0f}")
    
    print("\n2. VOLUME IN REBUILT BOOK (Rebuilder Check)")
    print(f"Total Rebuilt Bid Vol: {bid_book['Resting_Vol'].sum():,.0f}")
    print(f"Total Rebuilt Ask Vol: {ask_book['Resting_Vol'].sum():,.0f}")
    
    print("\n3. EXACT MATCH RATE (Merge Check)")
    # Temporarily merge without fillna(0) to see how many rows actually found a match
    test_bid_merge = pd.merge_asof(test_matrix, bid_book[['TOD', 'Price_Key', 'Resting_Vol']], on='TOD', left_by='BestBid_Key', right_by='Price_Key', direction='backward')
    test_ask_merge = pd.merge_asof(test_matrix, ask_book[['TOD', 'Price_Key', 'Resting_Vol']], on='TOD', left_by='BestAsk_Key', right_by='Price_Key', direction='backward')
    
    bid_match_rate = (test_bid_merge['Resting_Vol'].notna().sum() / len(test_matrix)) * 100
    ask_match_rate = (test_ask_merge['Resting_Vol'].notna().sum() / len(test_matrix)) * 100
    print(f"Bid Price Match Rate: {bid_match_rate:.1f}%")
    print(f"Ask Price Match Rate: {ask_match_rate:.1f}%")
    print("="*50 + "\n")
  
    mo_data['TOD'] = mo_data['TOD'].astype('int64')
    test_matrix['TOD'] = test_matrix['TOD'].astype('int64')
    merged = pd.merge_asof(
        mo_data.sort_values('TOD'), 
        test_matrix[['TOD', 'ProbQImbal', 'QImbalance']].sort_values('TOD'), 
        on='TOD', 
        direction='backward'
    )
    
    #Drop NaNs for trades that were placed before opening hours
    merged = merged.dropna(subset=['ProbQImbal', 'QImbalance'])
    
    
    #NOTE: The previous plots where i used the hardcoded bins did show a difference in heightratios in favor of the model
    #but this was not right as our model is a nonlinear transformation so it could be that probqimbal is just a more squeezed or powered version of regular qimbal
    #so we must use quantiles to see if then the ratios are still in favor cuz then the model is reranking the data and improving the regular qimbal 
    bins = [-1.0, -1/3, 1/3, 1.0]
    labels = ['Sell-Heavy', 'Neutral', 'Buy-Heavy']
    # --- Plotting ProbQImbal ---
    merged['Imbalance_Bin'] = pd.cut(merged['ProbQImbal'], bins=bins, labels=labels, include_lowest=True)
    
    # For MO: BorS == 1 -> sell, BorS == 0 -> buy
    summary_prob = merged.groupby(['Imbalance_Bin', 'BorS'])['Vol'].sum().unstack(fill_value=0)
    summary_prob.columns = ['Market Buys', 'Market Sells']
        
    #Plotting ProbQImbal
    summary_prob.plot(kind='bar', figsize=(9, 6), color=['darkblue', 'darkred'], edgecolor='black')

    
    plt.title(f"Trade Vol vs. ProbQImbal - config.TICK for {model}")
    plt.xlabel("Imbalance Level")
    plt.ylabel("Vol of Trades")
    plt.xticks(rotation=0)
    plt.legend(["Market Buys", "Market Sells"])
    plt.tight_layout()
    plt.show()
    
    #Plotting RegularQimbal
    merged['Imbalance_Bin'] = pd.cut(merged['QImbalance'], bins=bins, labels=labels, include_lowest=True)

    #For MO BorS == 1 -> sell
    summary_reg = merged.groupby(['Imbalance_Bin', 'BorS'])['Vol'].sum().unstack(fill_value=0)
    summary_reg.columns = ['Market Buys', 'Market Sells'] 
      
    summary_reg.plot(kind='bar', figsize=(9, 6), color=['darkblue', 'darkred'], edgecolor='black')
    plt.title(f"Trade Vol vs. RegularQImbal - config.TICK for {model}")
    plt.xlabel("Imbalance Level")
    plt.ylabel("Vol of Trades")
    plt.xticks(rotation=0)
    plt.legend(["Market Buys", "Market Sells"])
    plt.tight_layout()
    plt.show()

    
    #plotting with quantiles
    #rank data to break ties, i.e if there are large blocks of 0 fill prob
    merged['ProbQImbal_Rank'] = merged['ProbQImbal'].rank(method = 'first')
    
    merged['Imbalance_Bin'] = pd.qcut(merged['ProbQImbal_Rank'], q=3, labels=labels)
    summary_prob = merged.groupby(['Imbalance_Bin', 'BorS'])['Vol'].sum().unstack(fill_value=0)
    summary_prob.columns = ['Market Buys', 'Market Sells']
    
    summary_prob.plot(kind='bar', figsize=(9, 6), color=['darkblue', 'darkred'], edgecolor='black')
    
    #Plotting ProbQImbal
    
    plt.title(f"Trade Vol vs. ProbQImbal - config.TICK for {model}")
    plt.xlabel("Imbalance Level")
    plt.ylabel("Vol of Trades")
    plt.xticks(rotation=0)
    plt.legend(["Market Buys", "Market Sells"])
    plt.tight_layout()
    plt.show()
    
    #Plotting RegularQimbal
    merged['QImbalance_Rank'] = merged['QImbalance'].rank(method = 'first')
    merged['Imbalance_Bin'] = pd.qcut(merged['QImbalance_Rank'], q=3, labels=labels)
    summary_reg = merged.groupby(['Imbalance_Bin', 'BorS'])['Vol'].sum().unstack(fill_value=0)
    summary_reg.columns = ['Market Buys', 'Market Sells']
    #For MO BorS == 1 -> sell
      
    summary_reg.plot(kind='bar', figsize=(9, 6), color=['darkblue', 'darkred'], edgecolor='black')
    plt.title(f"Trade Vol vs. RegularQImbal - config.TICK for {model}")
    plt.xlabel("Imbalance Level")
    plt.ylabel("Vol of Trades")
    plt.xticks(rotation=0)
    plt.legend(["Market Buys", "Market Sells"])
    plt.tight_layout()
    plt.show()
    
    #Now lastly here we can use roc auc unlike before, now we just care about does based on probqimbal a buy and a sell order which metric is better at ordering them
    from sklearn.metrics import roc_auc_score
    
    is_buy = (merged['BorS'] == -1).astype(int)
    
    auc_regular = roc_auc_score(is_buy, merged['QImbalance'])
    auc_probq = roc_auc_score(is_buy, merged['ProbQImbal']) 
    auc_increase = (auc_probq - auc_regular) / auc_regular
    print(f' Regular AUC is {auc_regular} and Improved AUC is {auc_probq}, this is a {(auc_increase) * 100}% increase')

    #Now just printing the initial ratios and the improved ratios between bins
    regshmb = summary_reg.loc['Sell-Heavy', 'Market Buys']
    regshms = summary_reg.loc['Sell-Heavy', 'Market Sells']
    regshratio = regshms / regshmb
    print(f'Regular QImbal bin ratio when market sell heavy is {regshratio}')
    
    regbhmb = summary_reg.loc['Buy-Heavy', 'Market Buys']
    regbhms = summary_reg.loc['Buy-Heavy', 'Market Sells']
    regbhratio = regbhmb / regbhms
    print(f'Regular QImbal bin ratio when market buy heavy is {regbhratio}')
    
    probshmb = summary_prob.loc['Sell-Heavy', 'Market Buys']
    probshms = summary_prob.loc['Sell-Heavy', 'Market Sells']
    probshratio = probshms / probshmb
    print(f'Prob QImbal bin ratio when market sell heavy is {probshratio}')
    
    probbhmb = summary_prob.loc['Buy-Heavy', 'Market Buys']
    probbhms = summary_prob.loc['Buy-Heavy', 'Market Sells']
    probbhratio = probbhmb / probbhms
    print(f'Prob QImbal bin ratio when marekt buy heavy is {probbhratio}')
    
    shincrease = (probshratio - regshratio)/ regshratio
    bhincrease = (probbhratio - regbhratio)/ regbhratio
    
    print(f'Increase in Sell heavy bin ratio is {(shincrease)*100} %')
    print(f'Increase in Buy heavy bin ratio is {(bhincrease)*100} %')
    
    return print((test_matrix['ProbQImbal'].describe()), test_matrix['QImbalance'].describe())

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
    
    #temporarily changed the below to output high prob of fill for other checks
    return prob

def calc_daily_qimbal(test_matrix, use_model, scalar, features, mo_data, cleandata):
    X_raw = test_matrix[features].astype(np.float32, copy = False)
    X = scalar.transform(X_raw) if scalar else X_raw 
             
    #This code below is now state based and doesnt need type it just looks at the delta change in volume for each step 
    
    
    test_matrix['fillprob'] = use_model.predict_proba(X)[:,1]
    test_matrix['expvol'] = test_matrix['fillprob'] * test_matrix['Vol']  #Calculate prob weighted vol 
    
    all_events = test_matrix[['ID', 'TOD', 'Type', 'SideOfBook', 'expvol', 'Price']].copy()
    
    #at_touch = test_matrix[test_matrix['DistanceToTouch'] == 0].copy()
    
    # Type 68/70 are dropped upstream in data_regressors, so every order's real final
    # removal is invisible to test_matrix. Pull those events straight from the raw log --
    # no prediction needed, a confirmed-dead order's contribution is just 0.
    
    #touched_ids = set(at_touch['ID'].unique())
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

    # 3. Build the full Limit Order Book history using cumulative sum
    impacts = impacts.sort_values('TOD')
    impacts['Resting_Vol'] = impacts.groupby(['SideOfBook', 'Price'])['Vol_Delta'].cumsum()

    # Separate the book by side for cleaner merging
    bid_book = impacts[impacts['SideOfBook'] == 1].drop(columns=['SideOfBook'])
    ask_book = impacts[impacts['SideOfBook'] == 0].drop(columns=['SideOfBook'])
    
    #match datatypes so pd.merge can continue, as we scaled down data during processing before
# Force the merge keys to standard 64-bit types so the C-backend doesn't panic
    bid_book['TOD'] = bid_book['TOD'].astype('int64')
    ask_book['TOD'] = ask_book['TOD'].astype('int64')
    test_matrix['TOD'] = test_matrix['TOD'].astype('int64')

    bid_book['Price_Key'] = bid_book['Price'].abs().round().astype('int64')
    ask_book['Price_Key'] = ask_book['Price'].abs().round().astype('int64')

    test_matrix['BestBid_Key'] = test_matrix['BestBid'].abs().round().astype('int64')
    test_matrix['BestAsk_Key'] = test_matrix['BestAsk'].abs().round().astype('int64')

    # 4. Map the resting volume back to the main test_matrix BBO
    test_matrix = test_matrix.sort_values('TOD')
   # Merge Bid Volume
    # This magically finds the most recent Resting_Vol where the TOD matches AND the book Price matches the BestBid
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
    
    # 3. Calculate Imbalance, defaulting to 0 if the touch is completely empty
    test_matrix['ProbQImbal'] = np.where(
        prob_denom > 0, 
        (test_matrix['Total_Prob_Bid_Vol'] - test_matrix['Total_Prob_Ask_Vol']) / prob_denom,
        0.0
    )
    
    #add y true cuz maybe need later in monthly eval
    
    test_matrix['y_true'] = test_matrix[config.TARGET]
    
    mo_data['TOD'] = mo_data['TOD'].astype('int64')
    test_matrix['TOD'] = test_matrix['TOD'].astype('int64')
    merged = pd.merge_asof(
        mo_data.sort_values('TOD'), 
        test_matrix[['TOD', 'ProbQImbal', 'QImbalance', 'y_true', 'fillprob']].sort_values('TOD'), 
        on='TOD', 
        direction='backward'
    )
    
    #Drop NaNs for trades that were placed before opening hours
    merged = merged.dropna(subset=['ProbQImbal', 'QImbalance', 'fillprob'])
    
    return merged

def plot_monthly_sum(monthly_merged, selected_model):
    
    is_buy = (monthly_merged['BorS'] == -1).astype(int)
    
    auc_regular = roc_auc_score(is_buy, monthly_merged['QImbalance'])
    auc_probq = roc_auc_score(is_buy, monthly_merged['ProbQImbal']) 
    auc_increase = (auc_probq - auc_regular) / auc_regular
    print(f' Monthly Regular AUC is {auc_regular} and Monthly Improved AUC is {auc_probq}, this is a {(auc_increase) * 100}% increase')

    labels = ['Sell-Heavy', 'Neutral', 'Buy-Heavy']
    #plotting with quantiles
    monthly_merged['Imbalance_Bin'] = pd.qcut(monthly_merged['ProbQImbal'], q=3, labels=labels)
    summary_prob = monthly_merged.groupby(['Imbalance_Bin', 'BorS'])['Vol'].sum().unstack(fill_value=0)
    summary_prob.columns = ['Market Buys', 'Market Sells']
    
    summary_prob.plot(kind='bar', figsize=(9, 6), color=['darkblue', 'darkred'], edgecolor='black')
    
    #Plotting ProbQImbal
    
    plt.title(f"Trade Vol vs. ProbQImbal Month - config.TICK for {selected_model}")
    plt.xlabel("Imbalance Level")
    plt.ylabel("Vol of Trades")
    plt.xticks(rotation=0)
    plt.legend(["Market Buys", "Market Sells"])
    plt.tight_layout()
    plt.show()
    
    #Plotting RegularQimbal
    monthly_merged['Imbalance_Bin'] = pd.qcut(monthly_merged['QImbalance'], q=3, labels=labels)
    summary_reg = monthly_merged.groupby(['Imbalance_Bin', 'BorS'])['Vol'].sum().unstack(fill_value=0)
    summary_reg.columns = ['Market Buys', 'Market Sells']
    #For MO BorS == 1 -> sell
      
    summary_reg.plot(kind='bar', figsize=(9, 6), color=['darkblue', 'darkred'], edgecolor='black')
    plt.title(f"Trade Vol vs. RegularQImbal Month - config.TICK for {selected_model}")
    plt.xlabel("Imbalance Level")
    plt.ylabel("Vol of Trades")
    plt.xticks(rotation=0)
    plt.legend(["Market Buys", "Market Sells"])
    plt.tight_layout()
    plt.show()
    
    #something going wrong in passing monthly merged to test_model 
    #Plotting the other performance stuff for the month 
    test_model_wrap(monthly_merged, selected_model)
    return None

def get_raw_paths_from_parquet(file_path):
    #in filemanager my raw path stuff was for one while which i wanted to keep for all my other stuff, but for the sequential training that needs to be in a function i can call in walkforward below
    filename = os.path.basename(file_path)
    
    # Example filename: "INTC_BINARY_2014_07_08.parquet"
    name_without_ext = filename.replace('.parquet', '')
    parts = name_without_ext.split('_')
    
    ticker = parts[0]
    # Reconstruct the original 8-digit date string (YYYYMMDD) from the split formatted date
    date = parts[2] + parts[3] + parts[4]
    
    script_dir = Path(__file__).resolve().parent
    raw_dir = script_dir.parent / 'data' / 'raw' / f'{ticker}_NASDAQ'
    
    main_raw_path = raw_dir / f'{ticker}_{date}_NASDAQ.mat'
    mo_raw_path = raw_dir / 'MO' / f'{ticker}_{date}.mat'
    
    if not main_raw_path.exists() or not mo_raw_path.exists():
        print(f"WARNING: Could not locate raw files for {filename}")
        
    return str(main_raw_path), str(mo_raw_path)
    
    
def walk_forward(all_data_paths, selected_model, train_window_days = 20):
    
    monthly_dataframes = []
    
    total_test_days = len(all_data_paths) - train_window_days
   
    for i in range(total_test_days):
        
        train_files = all_data_paths[i : i + train_window_days]
        test_file = all_data_paths[i + train_window_days]
       
        use_model, scalar = train(train_files, None, selected_model)
        
        gc.collect()
        
        test_matrix = pd.read_parquet(test_file)
        test_matrix.replace([np.inf, -np.inf], 0, inplace=True)
        
        if selected_model == 'FNN':
            features = config.FNN_MODEL_FEATURES
        elif selected_model == 'LR':
            features = config.LOGISTIC_MODEL_FEATURES
        elif selected_model == 'LGBM':
            features = config.LGBM_MODEL_FEATURES
        
        raw_data_path, raw_mo_path = get_raw_paths_from_parquet(test_file)
        
        rawdata = import_data(raw_data_path, raw_mo_path)
        cleandata = clean_data(rawdata)
        mo_data = rawdata['MO']
        
        daily_merged = calc_daily_qimbal(test_matrix, use_model, scalar, features, mo_data, cleandata)
        monthly_dataframes.append(daily_merged)
        del test_matrix, rawdata, cleandata, mo_data
        
    monthly_merged = pd.concat(monthly_dataframes, ignore_index = True)
    plot_monthly_sum(monthly_merged,selected_model)
        
    
if __name__ == "__main__":
    
    print('Welcome Boss, Starting Programme')
    
    process_choice = input('Do you want to process Raw files (y/n) ').strip().lower()
    if process_choice == 'y':
        save_data()
    else:
        print('Moving on to preprocessed data')
    
    print('Please select TRAINING Data (atleast one day)')
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
    print('TRAINING IS REQUIRED BEFORE TESTING (whenever you select new data obviously otherwise not required)')
    action_choice = input("Do you want to 'train' or 'test' or 'qimbal' or 'use' or 'eval' this model? ").strip().lower()
    
    if action_choice in ['test', 'qimbal', 'use']:
        test_matrix = pd.read_parquet(test_files) 
        from FileManager import get_data_paths
        data_path, mo_path = get_data_paths()
        rawdata = import_data(data_path, mo_path)
        mo_data = rawdata['MO']
        cleandata = clean_data(rawdata)

    if action_choice == 'train':
        
        train_frames = []
        
        for f in train_files:
            train_frames.append(pd.read_parquet(f))
            
        train_matrix = pd.concat(train_frames, ignore_index = True)
        train(train_files, None, selected_model)
        
        
    elif action_choice == 'test':
        
        test_model_wrap(test_matrix, selected_model)
        
    elif action_choice == 'qimbal':
        improve_qimbal(test_matrix, selected_model, mo_data, cleandata)
        
    elif action_choice == 'eval':
        #just manually force a list on the one item in tesst files so we can concatenate in sorted below 
        chronological_data = sorted(train_files + test_files)
        walk_forward(chronological_data, selected_model, train_window_days = 20)
        
        
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
            loaded_model.eval()
            use_model = PyTorchSklearnWrapper(loaded_model, device)
        
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
        print(f"MAX FILL PROBABILITY FOUND:")
        print(f"Order ID: {highest_id}")
        print(f"Time Since Placement: {highest_tsp} ms")
        print(f"Fill Probability: {highest_prob * 100:.2f}%")
        print("="*50 + "\n")
        while True:
            
            # order_events = test_matrix[test_matrix['ID'] == 43663]
            # placements = order_events[order_events['Type'].isin([66, 83])]
            
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
    


