#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 10:42:47 2026

@author: jesseruijer
"""

"""
LR Engine

Trains & Calibrates
"""

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import config
import pandas as pd



## This code below was the original logistic code, which cant do sequential training if the data is too large

def train_logistic_model(X_train:pd.DataFrame, y_train:pd.Series, weights:pd.Series, X_calib:pd.DataFrame, y_calib:pd.Series, weights_calib:pd.Series, params:dict | None = None, for_tuning:bool = False) -> tuple[LogisticRegression, CalibratedClassifierCV | None,StandardScaler]:
    
    """
    Training and calibrating the logistic regression model
    """
    
    if params is None:
        params = {
            #By default logistic regressions performance metric is log loss, and it doesnt do anything else so thats why there's no performance metric specified since it uses logloss by default  
            'max_iter' : 172,
            'n_jobs' : 1, 
            'random_state' : config.RANDOM_SEED,
            'penalty' : 'l2', #Used to be l1 from optuna but after introducing heartbeats it was just too slow
            'solver': 'lbfgs', #Used to be saga from optuna but after introducing heartbeats it was just too slow
            'C': 9.130504371266555,
            'verbose' : 2 #Prints progress to console
            }
        
    scalar = StandardScaler(copy = False)   #Apply standardscalar without needing a copy, saves ram 
    X_train_standardised = scalar.fit_transform(X_train.astype(np.float32, copy = False)) #Here we fit and transform
    base_logistic_model = LogisticRegression(**params) # max iter higher then the standard 100 to take into account the noisy data we have
        
    #Saved the base model as well here, maybe thats useful for log odss not sure
    base_logistic_model.fit(X_train_standardised, y_train, sample_weight = weights)
    
    if for_tuning: #This is just for speed optimsation for using Optuna as for that you just need the base model so dont want to waste time calibrating
        return base_logistic_model, None, scalar
    
    X_calib_scaled = scalar.transform(X_calib.astype(np.float32, copy = False)) #Here we only transform and not fit. its cuz transform just applies the ruler and fit actually calculates it and we only want to calculate on the training data and not on the testing data
   
    calibrated_model = CalibratedClassifierCV(estimator=base_logistic_model, method='isotonic', 
                                              cv = 'prefit') # fit calibrator only on reserved data for calibration, since model has already been trained before this overrides cross validation and prevents us from training on past data since our data is in chronological order
  
    #Fit Scikit logistic regrssion
    
    calibrated_logistic_model = calibrated_model.fit(X_calib_scaled, y_calib, sample_weight = weights_calib)
    
    return base_logistic_model, calibrated_logistic_model, scalar