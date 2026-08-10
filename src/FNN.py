#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 15:18:19 2026

@author: jesseruijer
"""

"""

Feedforward Neural Net Training for Main and UserScript

Wrapper class to convert outputs to comply with ModelEvaluation function input requirements

"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
import config
from torch.utils.data import DataLoader, Dataset
import random
import gc

#Set seed to ensure reproducability
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

class PyTorchSklearnWrapper:   
    
    """
    Wrapper Class for converting tensors to arrays
    My test model function in ModelEvaluation expects a 2d array but the outputs of the FNN are in tensors, so have to wrap
    """
    
    def __init__(self, model:nn.Module, device:torch.device, calibrator:IsotonicRegression | None = None):
        self.model = model
        self.device = device
        self.calibrator = calibrator
        
    def predict_proba(self, X:pd.DataFrame | np.ndarray) -> np.ndarray:
        self.model.eval()
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        X_tensor = torch.tensor(X, dtype = torch.float32).to(self.device)

        with torch.no_grad():   #Forward pass but no need to track gradients, model already trainend, only interested in the probabilitstic output number
            logits = self.model(X_tensor)
            
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
        
        if self.calibrator is not None:
            probs = self.calibrator.predict(probs)
            
        cancel_probs = 1.0 - probs
        fill_probs = probs
    
        return np.vstack((cancel_probs, fill_probs)).T    #vertical stack safer for 1d arrays to stay in N,2 shape


class DataSet(Dataset):  
   
    """
    class for putting the LOB data in the right tensorformat so tensor flow can use it 
    """
    
    def __init__(self, features:np.ndarray, labels:np.ndarray, weights:np.ndarray):
        self.X = torch.tensor(features, dtype = torch.float32)
        self.y = torch.tensor(labels, dtype = torch.float32).reshape(-1,1)      #outputs are just 1d array, and this puts them into a fixed 1 column and for the amount of rows it just sorts it out itself, this reshaping is necessary to later compare it to the outputs of the NN
        self.w = torch.tensor(weights, dtype = torch.float32).reshape(-1,1)

    def __len__(self) -> int:
        return len(self.X)
        
    def __getitem__(self, idx:int) -> tuple[torch.Tensor, torch.Tensor,torch.Tensor]:
        return self.X[idx], self.y[idx], self.w[idx]


class NN(nn.Module):  #pass nn>module giving it all the pytorch capabilities
    
    """
    Neural net architecture
    """

    def __init__(self, input_size: int, dropout_rate :float= 0.16669353086758615): #Dropout rate ensures during every single training step x% of neurons get turned off to prevent overfitting
        super(NN, self).__init__() #to unlock the power of nn.Module
        
        self.network = nn.Sequential(       #Sequential does all the chaining together 
            nn.Linear(input_size, 70),     #First layer amount of neurons 
            nn.BatchNorm1d(70),    #Normalize the data in batch after each layer (to prevent gradient destabilisation during loss calcuation in backward prop )
            nn.ReLU(),  #apply nonlinearity
            nn.Dropout(dropout_rate),   #dropout, NOTE: pytorch automatically turns dropout off during testing so all neurons remain active when making real predictions, so dont have to manually turn dropout off later
            
            nn.Linear(70, 105), 
            nn.BatchNorm1d(105),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(105, 108),
            nn.BatchNorm1d(108),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(108, 53),
            nn.BatchNorm1d(53),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(53, 1)        #Final output is just 1 neuron ofc, NOTE: no sigmoid activation here, since thats already done by ptyorch inside the loss function later, so the outputs here are just raw logits
            )
    def forward(self, x:torch.Tensor) -> torch.Tensor:       #because we used the sequential above, the forward 
        return self.network(x)
    
def prepdata_and_train(train_files:list) ->tuple[PyTorchSklearnWrapper, StandardScaler ]:
    
    """
    Prep data and training of NN
    
    """
    
    ##Device configuration
    #Training is much faster on gpu, but apple silicon has MPS and if its on a different deivce you have to use NVIDIA CUDA if available (like on the cluster probs)
    #this if else statement make sure to try and train on gpu no matter the device
    #CPU is like a smart professor but just one guy and GPU is like 10000 dumb students, for neural net training whihc is just LAG, the dumb students are much faster

    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print('Training on Apple Silicon MPS')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print('Training on NVIDIA GPU (CUDA)')
    else:
        device = torch.device('cpu')
        print('Training on CPU') 
        
    #Split train and test data in 4:1 ratio in chronological order of the files for training and test data    
    idx = int(len(train_files) * .8)
            
    active_train_files  = train_files[: idx]
    active_val_files_df = train_files[idx:]
        
    
    print('Fitting Scalar incrementally across days using Scalar.partial fit')
    scalar = StandardScaler()
    for f in active_train_files:
        df = pd.read_parquet(f)
        df.replace([np.inf, -np.inf], 0, inplace = True)

        X_train_raw = df[config.FNN_MODEL_FEATURES].values
        
        scalar.partial_fit(X_train_raw) #we cant fit the scalar to all days at once since it crashes RAM so use the partial_fit function that slowly updates the right scaling. It ends up getting the same result as applying a scalar to the whole day, it just doesnt crash RAM, and then later when we are training you just pass the scalar
        
        del df, X_train_raw
        gc.collect()
    
    #Initialize model 
    input_size = len(config.FNN_MODEL_FEATURES)
    LEARNING_RATE = 0.001    #0.003143153725649377 was the initial LR Optuna gave
    WEIGHT_DECAY = 2.5475947521348294e-06 #L2 regularization to prevent overfitting
    EPOCHS = 2
    BATCH_SIZE = 16384 #power of two so easy to progress and batch size can be large since the tabular data is not super information dense (like a 4K image for example)
    
    model = NN(input_size = input_size).to(device)
    
    criterion = nn.BCEWithLogitsLoss(reduction = 'none')    # BCE = Binary Cross Entropy = Logloss, this is just the scoring metric and saying reducion is none, it doesnt do any weighting by iteself it just spits out all the raw values and then with my manual weights i can do the weighint later
    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY)
    
    #Training LOOP

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        total_train_weight = 0.0
        
        #Shuffle day order every epoch
        
        random.shuffle(active_train_files)
        
        for f in active_train_files:
            
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
        
        #Evaluation Loop
        
        model.eval() # set model to evaluation mode
        val_loss = 0.0
        total_val_weight = 0.0
        
        with torch.no_grad():
            
            for f in active_val_files_df:   #Unlike above, no shuffling in validation files
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
            
        print(f' Epoch {epoch + 1} / {EPOCHS} \n')
        print(f'Weighted Train Logloss: {avg_train_loss:.3f} \n')
        print(f'Weighted Test Logloss: {avg_val_logloss:.3f}')

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

    fnn_calibrator = IsotonicRegression(out_of_bounds='clip') #clip just makes sure out of bounds probabilities dont throw an error, call isotonicregression class and fit model to it
    fnn_calibrator.fit(val_raw_probs, val_targets, sample_weight=val_weights)
    
    wrapped_fnn = PyTorchSklearnWrapper(model, device, calibrator = fnn_calibrator)
    
    return wrapped_fnn, scalar


class UserFNN(nn.Module):
    
    #I am aware this is a 1:1 copy of the code above, but it was easier to save and use different neural nets by just seperating them by name for use in Main and use in UserScript
    #Similar neural net class as above but now for use in UserScript 
    
    def __init__(self, input_size:int):
        super(UserFNN, self).__init__()
        
        dropout_rate = 0.16669353086758615
        
        self.network = nn.Sequential(
            
            nn.Linear(input_size, 70),     
            nn.BatchNorm1d(70),    
            nn.ReLU(),  
            nn.Dropout(dropout_rate),   
            
            nn.Linear(70, 105), 
            nn.BatchNorm1d(105),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(105, 108),
            nn.BatchNorm1d(108),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(108, 53),
            nn.BatchNorm1d(53),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(53, 1)  
            )
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.network(x)