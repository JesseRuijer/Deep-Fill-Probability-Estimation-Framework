#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 10:42:47 2026

@author: jesseruijer
"""

#Importing libraries,classes, functions from other scripts
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

###################Starting logistic regression###########################

## This code below was the original logistic code, which cant do sequential training if the data is too large

def train_logistic_model(X_train, y_train, weights, X_calib, y_calib, weights_calib, params = None, for_tuning = False):
    
    if params == None:
        params = {
            
            #By default logistic regressions performance metric is log loss, and it doesnt do anything else so thats why there's no performance metric specified since it uses logloss by default  
            
            'max_iter' : 172,
            'n_jobs' : 1, #Used to be -1 which i believe is multicore, but not sure which one is better atm
            'random_state' : 69,
            
            'penalty' : 'l2', #Used to be l1 from optuna but after introducing heartbeats it was just too slow
            'solver': 'lbfgs', #Used to be saga from optuna but after introducing heartbeats it was just too slow
            'C': 9.130504371266555,
            'verbose' : 2 #Prints progress to console
            }
        
    scalar = StandardScaler(copy = False)   #Apply standardscalar without needing a copy, saves ram 
    X_train_standardised = scalar.fit_transform(X_train.astype(np.float32, copy = False)) #Here we fit and transform
    
    X_calib_scaled = scalar.transform(X_calib.astype(np.float32, copy = False)) #Here we only transform and not fit. its cuz transform just applies the ruler and fit actually calculates it and we only want to calculate on the training data and not on the testing data
    
   #Logistic regression 
   #Might use somethiing of platt scaling to wrap the log res model as log res doesnt work very well with data where the output is very skewed, i.e here we have much more cancels then fills
    base_logistic_model = LogisticRegression(**params) # max iter higher then the standard 100 to take into account the noisy data we have
        
    #Saved the base model as well here, maybe thats useful for log odss not sure
    base_logistic_model.fit(X_train_standardised, y_train, sample_weight = weights)
        
    if for_tuning: #This is just for speed optimsation for using Optuna as for that you just need the base model so dont want to waste time calibrating
        return base_logistic_model, None, scalar

    
    calibrated_model = CalibratedClassifierCV(estimator=base_logistic_model, method='isotonic', 
                                              cv = 'prefit') # fit calibrator only on reserved data for calibration, since model has already been trained before this overrides cross validation and prevents us from training on past data since our data is in chronological order
  
    
    #Look at the required assumptions for logistic regression, i think need iid and for example
    #below i included price related features but if the price changes they dont follow the same distribution
    #on a given day anymore and the whole model breaks, so now the model only looks at volume
    #and position dynamics

    
    
    
   
    #Fit Scikit logistic regrssion
    
    calibrated_logistic_model = calibrated_model.fit(X_calib_scaled, y_calib, sample_weight = weights_calib)
    
    return base_logistic_model, calibrated_logistic_model, scalar
    



