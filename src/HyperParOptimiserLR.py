#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 10:53:13 2026

@author: jesseruijer
"""


import pandas as pd
import numpy as np
import optuna 
import config
import gc
from LogisticRegressionEngine import train_logistic_model
from sklearn.metrics import log_loss
from FileManager import get_ml_training_paths

#Loading data in 

paths = get_ml_training_paths()

if not paths:
    print("Model training aborted. No training files selected.")
    exit()

train_files = paths.get('train_bin', [])
val_file = paths.get('test_bin', []) #This is just for selecting the val file, once done with optuna, can chagne the test file back to another day in main 

train_frames = []

for f in train_files:
    train_frames.append(pd.read_parquet(f))
    
train_matrix = pd.concat(train_frames, ignore_index = True)
del train_frames
gc.collect()

train_matrix = train_matrix.sample(frac = 0.1, random_state = 67)
val_matrix = pd.read_parquet(val_file) #Validation data different from training and testing data ofc

X_train = train_matrix[config.LOGISTIC_MODEL_FEATURES]
y_train = train_matrix[config.TARGET]
weights_train = train_matrix['UnitWeight']

X_val = val_matrix[config.LOGISTIC_MODEL_FEATURES]
y_val = val_matrix[config.TARGET]
val_weights = val_matrix['UnitWeight']

def objective(trial):
    
    penalty = trial.suggest_categorical('penalty', ['l2'])    #l1 too slow for large data and i sortof know which features i want to use but maybe if i have time i can run it again with including l1. l1 is lasso penalty, adds absolute value of weights to the errors, focusses on deleting weak features,  i.e assigning them weight zero
                                                                    #l2 is ridge, adds square of weights to error calculation, focusses on keeping all features weights relatively small and balanced, since i expect vol to have a massive impact, for our situation l1 might be better
    solver = trial.suggest_categorical('solver', ['lbfgs'])  #removed saga here since thats just too slow for large datasets
    
    #Define search space
    
    params = {
        #Basic 
        'max_iter': 5000, #Set very high for same reason as lgbm hyperpar, now we do early stopping using the tol parameter
        'random_state': 69,
        'n_jobs': 1,
        'tol': 0.001, #Force solver to stop early if it hits a plateau 
        
        #Optuna tuning pars
        'penalty': penalty,
        'solver': solver,
        'C': trial.suggest_float('C', 1e-4, 10.0, log=True),    #C stands for inverse regularization strength, tells the model how much its allowed to trust the training data, 10 means i trust it a lot 0 means i dont trust it at all
        
    
        }
    
    #Train
    
    model, _, scalar = train_logistic_model(X_train, y_train, weights_train, params = params, for_tuning = True)
    
    # Manually tell Optuna to extract and save the exact number of iterations scikit-learn took, because thats not in the default console output 
    trial.set_user_attr("actual_iters", int(model.n_iter_[0]))
   
    
    #Predict on validation data
    X_val_scaled = scalar.transform(X_val)
    
    preds = model.predict_proba(X_val_scaled)[:,1]
    
    #Performance metrics
    
    y_val_bin = np.where(y_val == 1, 1, 0)
    
    score = log_loss(y_val_bin, preds, sample_weight = val_weights)
    
    return score



if __name__ == '__main__':
    print("Starting Optuna Optimalisation")
    
    #Running search for optimal params
    
    study = optuna.create_study(direction = 'minimize') # Since our criterion for finetuning here is log loss we aim to minimize
    study.optimize(objective, n_trials = 40)
    
    print(f' Best average prediction score was {study.best_value:.3f}')
    print('Optimised pars')
    
    for key, value in study.best_params.items():
        print(f'{key} : {value}')
        
    #Printing best iteration count of best trial 
    best_iters = study.best_trial.user_attrs["actual_iters"]
    print(f'Optimal iterations used : {best_iters}')
    
    
    
    
    
    
    
    
    
    
    
    
    