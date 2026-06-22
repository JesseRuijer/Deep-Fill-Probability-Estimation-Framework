#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 10:42:47 2026

@author: jesseruijer
"""

#Importing libraries,classes, functions from other scripts

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

###################Starting logistic regression###########################


def train_logistic_model(X_train, y_train, weights,params = None, for_tuning = False):
    
    if params == None:
        params = {
            'max_iter' : 200,
            'n_jobs' : 1, #Used to be -1 which i believe is multicore, but not sure which one is better atm
            'random_state' : 69,
            
            'penalty' : 'l2', #Used to be l1 from optuna but after introducing heartbeats it was just too slow
            'solver': 'lbfgs', #Used to be saga from optuna but after introducing heartbeats it was just too slow
            'C': 0.00010720311501308609,
            'verbose' : 2 #Prints progress to console
            }
        
    scalar = StandardScaler()
    X_train_standardised = scalar.fit_transform(X_train) #Here we fit and transform
   #Logistic regression 
   #Might use somethiing of platt scaling to wrap the log res model as log res doesnt work very well with data where the output is very skewed, i.e here we have much more cancels then fills
    base_logistic_model = LogisticRegression(**params) # max iter higher then the standard 100 to take into account the noisy data we have
        
    #Just saved the base model as well here, maybe thats useful for log odss not sure
    base_logistic_model.fit(X_train_standardised, y_train, sample_weight = weights)
        
    if for_tuning: #This is just for speed optimsation for using Optuna as for that you just need the base model so dont want to waste time calibrating
        return base_logistic_model, None, scalar
    
    
    
    
    calibrated_model = CalibratedClassifierCV(estimator=base_logistic_model, method='isotonic', cv=5 ) #Do some research if and why 5 is good value for cross validation in ML, its because that splits into 5 folds of 20% where each time we train on 80 and test on 20 and thats like the industry standard i think
  
    
    #Look at the required assumptions for logistic regression, i think need iid and for example
    #below i included price related features but if the price changes they dont follow the same distribution
    #on a given day anymore and the whole model breaks, so now the model only looks at volume
    #and position dynamics

    

    
   
    #Fit Scikit logistic regrssion
    
    calibrated_logistic_model = calibrated_model.fit(X_train_standardised, y_train, sample_weight = weights)
    
    return base_logistic_model, calibrated_logistic_model, scalar
    



