#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 15:18:19 2026

@author: jesseruijer
"""

import torch
import torch.nn as nn
import torchvision
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import config
from torch.utils.data import DataLoader, Dataset

import gc


class PyTorchSklearnWrapper:    #My test model function in ModelEvaluation expects a 2d array but the outputs of the FNN are in tensors, so have to wrap
    def __init__(self, model, device):
        self.model = model
        self.device = device
        
    def predict_proba(self, X):
        self.model.eval()
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        X_tensor = torch.tensor(X, dtype = torch.float32).to(self.device)

        with torch.no_grad():   #Forward pass but no need to track gradients, model already trainend, only interested in the probabilitstic output number
            logits = self.model(X_tensor)
            
            probs = torch.sigmoid(logits).cpu().numpy()


        class_0_probs = 1.0 - probs
        class_1_probs = probs
        
        return np.hstack((class_0_probs, class_1_probs))    #Horizontal stack to return a 2d array


class DataSet(Dataset):     #in a class what you pass in () is not a par like with functions, its the parent class you're inheriting from 
    #class for putting the LOB data in the right tensorformat so tensor flow can use it 
    def __init__(self, features, labels, weights):
        self.X = torch.tensor(features, dtype = torch.float32)
        self.y = torch.tensor(labels, dtype = torch.float32).reshape(-1,1)      #outputs are just 1d array, and this puts them into a fixed 1 column and for the amount of rows it just sorts it out itself, this reshaping is necessary to later compare it to the outputs of the NN
        self.w = torch.tensor(weights, dtype = torch.float32).reshape(-1,1)

    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.w[idx]


class NN(nn.Module):  #you need a class because classes function as blueprints that keep track of its internal state which is necessary bcecause it needs to remeber what it did before everytime you pass a new batch into it, then pass nn>module giving it all the pytorch capabilities
    #Neural net architecture
    def __init__(self, input_size, dropout_rate = 0.2): #Dropout rate ensures during every single training step 20% of neurons get turned off to prevent overfitting, can be increased later
        super(NN, self).__init__() #to unlock the power of nn.Module
        
        self.network = nn.Sequential(       #Sequential does all the chaining together i manually did in playground
            nn.Linear(input_size, 128),     #First layer amount of neurons 
            nn.BatchNorm1d(128),    #Normalize the data in batch after each layer (to prevent gradient destabilisation during loss calcuation in backward prop i think)
            nn.ReLU(),  #apply nonlinearity
            nn.Dropout(dropout_rate),   #dropout, NOTE: pytorch automatically turns dropout off during testing so all neurons remain active when making real predictions, so dont have to manually turn dropout off later
            
            nn.Linear(128, 64), #I believe a funnel shape in terms of hiddenlayer dimensions is usually the best approach, but can look over this more later
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(32, 1)        #Final output is just 1 neuron ofc, NOTE: no sigmoid activation here, since thats already done by ptyorch inside the loss function later, so the outputs here are just raw logits
            )
    def forward(self, x):       #because we used the sequential above, the forward 
        return self.network(x)
    
def prepdata_and_train(train_files):
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
            
    active_train_files  = train_files.iloc[: idx]
    active_val_files_df = train_files.iloc[idx:]
        
    
    
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
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.0001 #L2 regularization to prevent overfitting
    EPOCHS = 7
    BATCH_SIZE = 16384 #power of two so easy to progress and batch size can be large since the tabular data is not super information dense (like a 4K image for example)
    model = NN(input_size = input_size).to(device)
    
    criterion = nn.BCEWithLogitsLoss(reduction = 'none')    # BCE = Binary Cross Entropy = Logloss, this is just the scoring metric and saying reducion is none, it doesnt do any weighting by iteself it just spits out all the raw values and then with my manual weights i can do the weighint later
    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY)
    
    #Training LOOP

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        total_train_weight = 0.0
        
        for features, labels, batch_weights in train_loader:
            features, labels, batch_weights = features.to(device), labels.to(device), batch_weights.to(device)
            
            
            y_train = df[config.TARGET].values
            w_train = df['UnitWeight'].values
            
            X_test_raw = test_df[config.FNN_MODEL_FEATURES].values
            y_test = test_df[config.TARGET].values
            w_test = test_df['UnitWeight'].values

           
            X_train_scaled = scalar.fit_transform(X_train_raw)
            X_test_scaled = scalar.transform(X_test_raw)
            
           
            train_dataset = DataSet(X_train_scaled, y_train, w_train)
            train_loader = DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, shuffle = True)  #NOTE: for a FNN you must shuffle data, which works because at every stage the network has no memory of what happened before it, it does not introduce lookahead bias and helps the model converge faster
            
            test_dataset = DataSet(X_test_scaled, y_test, w_test)
            test_loader = DataLoader(dataset = test_dataset, batch_size = BATCH_SIZE, shuffle = False)


            
            

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
            
        avg_train_loss = train_loss / total_train_weight
        
        #Evaluation Loop
        
        model.eval() # set model to evaluation mode
        test_loss = 0.0
        total_test_weight = 0.0
        
        with torch.no_grad():
            for features, labels, batch_weights in test_loader:
                features, labels, batch_weights = features.to(device), labels.to(device), batch_weights.to(device)
        
                outputs = model(features)
                raw_loss = criterion(outputs, labels)
                
                test_loss += (raw_loss * batch_weights).sum().item()
                
                total_test_weight += batch_weights.sum().item()

            avg_out_of_sample_logloss = test_loss / total_test_weight
            
        print(f' Epoch {epoch + 1} / {EPOCHS} \n')
        print(f'Weighted Train Logloss: {avg_train_loss:.3f} \n')
        print(f'Weighted Test Logloss: {avg_out_of_sample_logloss:.3f}')


    
    
    wrapped_fnn = PyTorchSklearnWrapper(model, device)
    
    return wrapped_fnn, scalar













