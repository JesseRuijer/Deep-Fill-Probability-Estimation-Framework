#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 09:31:32 2026

@author: jesseruijer
"""


import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import config
from torch.utils.data import DataLoader
import random
import optuna 
import gc
from FNN import DataSet
from FileManager import get_ml_training_paths


class DynamicFNN(nn.Module):
    
    #Initialize new neural net thats modifiable per optuna trial 
    
    def __init__(self, trial, input_size):
        super(DynamicFNN, self).__init__()
              
        
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        n_layers = trial.suggest_int('n_layers', 1, 4)
        
        layers = []
        in_features = input_size
        
        for l in range(n_layers):
            out_features = trial.suggest_int(f'size_layer_{l}', 16, 256)
            
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.BatchNorm1d(out_features))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            
            in_features = out_features
            
        #add final output neuron since has to be 1
        
        layers.append(nn.Linear(out_features, 1))
        
        #Unpack list above in sequential container using sequential again like in FNN script so you dont have to manually write each loop
        
        self.network = nn.Sequential(*layers)   # * is for unpacking lists/tuples, ** is for unpacking dictionary
        
    def forward(self, x):
        return self.network(x)
        
        
        
def objective(trial, train_files, val_files, pre_fitted_scalar):   
    
    #optuna trial function 
    
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
    
    lr = trial.suggest_float('lr', 0.0001, 0.1, log = True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    
    BATCH_SIZE = 16384
    EPOCHS = 7
    
    scalar = pre_fitted_scalar
        
    model = DynamicFNN(trial, input_size = input_size).to(device)
     
    criterion = nn.BCEWithLogitsLoss(reduction = 'none')    # BCE = Binary Cross Entropy = Logloss, this is just the scoring metric and saying reducion is none, it doesnt do any weighting by iteself it just spits out all the raw values and then with my manual weights i can do the weighint later
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
     
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
             
         model.eval()
         val_loss = 0.0
         total_val_weight = 0.0

        
         with torch.no_grad():
            
             for f in val_files:   #Unlike above, no shuffling in validation files
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
            
            #pruning (early stopping) if no improvements in this specific NN
            
             trial.report(avg_val_logloss, epoch)
             if trial.should_prune():
                 raise optuna.exceptions.TrialPruned()
     
     
    return avg_val_logloss
     

if __name__ == '__main__':
    print("Starting Optuna Optimalisation")
    
    paths = get_ml_training_paths()
    
    all_files = paths.get('train_bin', [])
    idx = int(len(all_files) * .8)
    
    train_files = all_files[:idx]
    val_files = all_files[idx:]
    
    #Fitting scalar once before starting optuna
    
    print('Fitting Scalar incrementally across days using Scalar.partial fit')
    
    global_scalar = StandardScaler()
    
    for f in train_files:
        df = pd.read_parquet(f)
        df.replace([np.inf, -np.inf], 0, inplace = True)

        X_train_raw = df[config.FNN_MODEL_FEATURES].values
        
        global_scalar.partial_fit(X_train_raw) #we cant fit the scalar to all days at once since it crashes RAM so use the partial_fit function that slowly updates the right scaling. It ends up getting the same result as applying a scalar to the whole day, it just doesnt crash RAM, and then later when we are training you just pass the scalar
        
        del df, X_train_raw
        gc.collect()
        
    
    #Running search for optimal params
    
    study = optuna.create_study(
        study_name = 'fnn_tuning_INTC_0416-0423',
        storage = 'sqlite:///fnn_tuning.db', #Store the progress thusfar in optimsed sql light file, so you can just restart where you left off after stopping the script
        load_if_exists = True,
        direction = 'minimize'
        ) # Since our criterion for finetuning is average log loss we aim to minimize
    
    #below is just a workaround since optuna can only take one par as input and we have three so create a temprary lambda so we can pass just the one 
    study.optimize(lambda trial: objective(trial, train_files, val_files, global_scalar), n_trials = 40)
    
    print(f' Best average prediction score was {study.best_value:.3f}')
    print('Optimised structural pars')
    
    for key, value in study.best_params.items():
        print(f'{key} : {value}')    
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     

