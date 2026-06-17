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
from sklearn.metrics import average_precision_score

#Loading data in 

train_matrix = pd.read_parquet('/Users/jesseruijer/Documents/Summer Research/data/processed/INTC_MULTI_2014_04_01.parquet')
val_matrix = pd.read_parquet('/Users/jesseruijer/Documents/Summer Research/data/processed/INTC_MULTI_2014_04_02.parquet') #Validation data different from training and testing data ofc

X_train = train_matrix[config.LGBM_MODEL_FEATURES]
y_train = train_matrix[config.TARGET]
weights_train = train_matrix['UnitWeight']

X_val = val_matrix[config.LGBM_MODEL_FEATURES]
y_val = val_matrix[config.TARGET]


def objective(trial):
    
    #Define search space
    
    params = {

    'objective' : 'multiclass', # Our Output var is multiclass for lgbm
    'num_class' : 3, # How many diff classes (0,1,2 are the classes in our case)
    'boosting_type' : 'gbdt', #The default gradient boosting
    'metric' : 'multi_logloss', #Metric to measure perforance
    'n_jobs' : -1, #Using all available threads in cpu 
    'random_state' : 69 , #just set random seed for reproducability
    'n_estimators' : 150, # number of sequential trees, i.e number of boosting rounds 
    
    #Fine tune these below, note that for example learning rate and n_estimators both influence number of sequential trees, so thats why im changingin only one at a time
    
    'learning_rate' : trial.suggest_float('learning_rate', 0.01, 0.1, log = True), # log is true to spend more time learning slowly which is useful with the amount of noise in lob data i think
    'num_leaves' : trial.suggest_int('num_leaves', 20, 150), #max num of leaves, i.e terminal nodes, allowed in each tree 
    'min_child_samples': trial.suggest_int('min_child_samples', 100, 2000) #Minimum number of data points required to create a new split in a leaf node
    
        }
    
    #Train
    
    model = train_lgbm_model(X_train, y_train, weights_train, params = params, for_tuning = True)
    
    #Predict on validation data
    
    preds = model.predict_proba(X_val)[:,1]
    
    #Performance metrics
    #I believe for hyperpar tuning the precision recall is best for our situation i.e placing a lot of emphasis on TPs, true positives
    
    y_val_bin = np.where(y_val == 1, 1, 0)
    
    score = average_precision_score(y_val_bin, preds)
    
    return score



if __name__ == '__main__':
    print("Staring Optuna Optimalisation")
    
    #Running search for optimal params
    
    study = optuna.create_study(direction = 'maximize') # Since our criterion for finetuning is average precision score (AUC of Precision Recall) we aim to maximisze
    study.optimize(objective, n_trials = 60)
    
    print(f' Best average prediction score was {study.best_value:.3f}')
    print('Optimised structural pars')
    
    for key, value in study.best_params.items():
        print(f'{key} : {value}')
    
    
    
    
    
    
    
    
    
    
    
    
    
    