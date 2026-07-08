#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 14:22:54 2026

@author: jesseruijer
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt
import gc
import random
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

import config
from FNN import PyTorchSklearnWrapper, DataSet
from ModelEvaluation import test_model
from FileManager import get_ml_training_paths

#####Use either lgbm or fnn to get outputs and performance on selected scripts and ID


#Fill in final values below after optuna finishes



class FinalFNN(nn.Module):
    def __init__(self, input_size):
        super(FinalFNN, self).__init__()
        
        dropout_rate = 0
        
        self.network = nn.Sequential(
            
            nn.Linear(input_size, ),
            nn.BatchNorm1d(),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
            
            
            
            
            )

    def forward(self, x):
        return self.network(x)



def train(train_files, model):
    
    print('Starting Training')
    
    if model == 'FNN':
    
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            print('Training on Apple Silicon MPS')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
            print('Training on NVIDIA GPU (CUDA)')
        else:
            device = torch.device('cpu')
            print('Training on CPU') 
        
        input_size = len()
        
        LEARNING_RATE = 0.001
        WEIGHT_DECAY = 0.0001 
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

        model = FinalFNN(input_size = input_size).to(device)
         
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
       
        return wrapped_fnn, scalar
          
    elif model == 'LGBM':
        




if __name__ == "__main__":
    print('Starting Programme')
    print('Please select TRAINING Data (atleast one day)')
    
    paths = get_ml_training_paths()
    
    train_files = paths.get('train_bin', [])
    
    print('Please select Test Data (One day only and no overlap with test data (obviously))')

    test_files = paths.get('test_bin', [])


