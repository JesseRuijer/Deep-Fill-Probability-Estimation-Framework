#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 14:36:01 2026

@author: jesseruijer
"""

import pandas as pd
import numpy as np
import optuna 
import config
from LightGBMEngine import train_lgbm_model
from sklearn.metrics import log_loss
from pathlib import Path
from FileManager import get_ml_training_paths
import gc


#Loading data in 

paths = get_ml_training_paths()

if not paths:
    print("Model training aborted. No training files selected.")
    exit()

all_files = paths.get('train_bin', [])
train_files = all_files
val_file = paths.get('test_bin', []) #This is just for selecting the val file, once done with optuna, can chagne the test file back to another day in main 

train_frames = []

for f in train_files:
    train_frames.append(pd.read_parquet(f))
    
train_matrix = pd.concat(train_frames, ignore_index = True)
del train_frames
gc.collect()

train_matrix = train_matrix.sample(frac = 0.15, random_state = 67)
val_matrix = pd.read_parquet(val_file) #Validation data different from training and testing data ofc


X_train = train_matrix[config.LGBM_MODEL_FEATURES]
y_train = train_matrix[config.TARGET]
weights_train = train_matrix['UnitWeight']

X_val = val_matrix[config.LGBM_MODEL_FEATURES]
y_val = val_matrix[config.TARGET]
val_weights = val_matrix['UnitWeight']


def objective(trial):
    
    #Define search space
    
    params = {

    'objective' : 'binary', # Our Output var is multiclass for lgbm
    #'num_class' : 3, # How many diff classes (0,1,2 are the classes in our case)
    'boosting_type' : 'gbdt', #The default gradient boosting
    'metric' : 'binary_logloss', #Metric to measure perforance
    'n_jobs' : -1, #Using all available threads in cpu 
    'random_state' : 69 , #just set random seed for reproducability
    'n_estimators' : 5000, # number of sequential trees, i.e number of boosting rounds 
    
    #Fine tune these below, note that for example learning rate and n_estimators both influence number of sequential trees, so thats why im changingin only one at a time
    
    'learning_rate' : trial.suggest_float('learning_rate', 0.01, 0.1, log = True), # log is true to spend more time learning slowly which is useful with the amount of noise in lob data i think
    'num_leaves' : trial.suggest_int('num_leaves', 20, 150), #max num of leaves, i.e terminal nodes, allowed in each tree 
    'min_child_samples': trial.suggest_int('min_child_samples', 100, 3000) #Minimum number of data points required to create a new split in a leaf node
    
        }
    
    #Train
    
    model = train_lgbm_model(
        X_train, y_train, weights_train, 
        X_val = X_val, y_val = y_val, val_weights = val_weights,
        params = params, for_tuning = True
        )
    
    #Predict on validation data
    
    preds = model.predict_proba(X_val)[:,1]
    
    #Performance metrics
    #I believe for hyperpar tuning the precision recall is best for our situation i.e placing a lot of emphasis on TPs, true positives
    
    y_val_bin = np.where(y_val == 1, 1, 0)
    
    score = log_loss(y_val_bin, preds, sample_weight = val_weights)
    
    return score



if __name__ == '__main__':
    print("Starting Optuna Optimalisation")
    
    #Running search for optimal params
    
    study = optuna.create_study(direction = 'minimize') # Since our criterion for finetuning is average log loss we aim to minimize
    study.optimize(objective, n_trials = 60)
    
    print(f' Best average prediction score was {study.best_value:.3f}')
    print('Optimised structural pars')
    
    for key, value in study.best_params.items():
        print(f'{key} : {value}')
    
    
    
    
    
    
    
    
    
    
    
    
    
    