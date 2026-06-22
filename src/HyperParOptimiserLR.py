#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 10:53:13 2026

@author: jesseruijer
"""

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
from LogisticRegressionEngine import train_logistic_model
from sklearn.metrics import average_precision_score

#Loading data in 

train_matrix = pd.read_parquet('/Users/jesseruijer/Documents/Summer Research/data/processed/INTC_MULTI_2014_04_01.parquet')
val_matrix = pd.read_parquet('/Users/jesseruijer/Documents/Summer Research/data/processed/INTC_MULTI_2014_04_02.parquet') #Validation data different from training and testing data ofc

X_train = train_matrix[config.LOGISTIC_MODEL_FEATURES]
y_train = train_matrix[config.TARGET]
weights_train = train_matrix['UnitWeight']

X_val = val_matrix[config.LOGISTIC_MODEL_FEATURES]
y_val = val_matrix[config.TARGET]


def objective(trial):
    
    penalty = trial.suggest_categorical('penalty', ['l2'])    #l1 too slow for large data and i sortof know which features i want to use but maybe if i have time i can run it again with including l1. l1 is lasso penalty, adds absolute value of weights to the errors, focusses on deleting weak features,  i.e assigning them weight zero
                                                                    #l2 is ridge, adds square of weights to error calculation, focusses on keeping all features weights relatively small and balanced, since i expect vol to have a massive impact, for our situation l1 might be better
    if penalty == 'l1':
        solver = 'saga'
        
    else:
        solver = trial.suggest_categorical('solver', ['lbfgs', 'newton-cholesky'])  #removed saga here since thats just too slow for large datasets
    
    #Define search space
    
    params = {
        #Basic 
        'max_iter': 500, 
        'random_state': 69,
        'n_jobs': 1,
        
        #Optuna tuning pars
        'penalty': penalty,
        'solver': solver,
        'C': trial.suggest_float('C', 1e-4, 10.0, log=True),    #C stands for inverse regularization strength, tells the model how much its allowed to trust the training data, 10 means i trust it a lot 0 means i dont trust it at all
        
    
        }
    
    #Train
    
    model, _, scalar = train_logistic_model(X_train, y_train, weights_train, params = params, for_tuning = True)
    
    #Predict on validation data
    X_val_scaled = scalar.transform(X_val)
    
    preds = model.predict_proba(X_val_scaled)[:,1]
    
    #Performance metrics
    #I believe for hyperpar tuning the precision recall is best for our situation i.e placing a lot of emphasis on TPs, true positives
    
    y_val_bin = np.where(y_val == 1, 1, 0)
    
    score = average_precision_score(y_val_bin, preds)
    
    return score



if __name__ == '__main__':
    print("Staring Optuna Optimalisation")
    
    #Running search for optimal params
    
    study = optuna.create_study(direction = 'maximize') # Since our criterion for finetuning here is average precision score (AUC of Precision Recall) we aim to maximisze
    study.optimize(objective, n_trials = 40)
    
    print(f' Best average prediction score was {study.best_value:.3f}')
    print('Optimised pars')
    
    for key, value in study.best_params.items():
        print(f'{key} : {value}')
    
    
    
    
    
    
    
    
    
    
    
    
    
    