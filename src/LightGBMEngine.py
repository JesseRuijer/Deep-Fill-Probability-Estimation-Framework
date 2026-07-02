#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 10:42:59 2026

@author: jesseruijer
"""

#Importing libraries,classes, functions from other scripts


#Make it multi class, i.e first class is fill, second is active cancel and third is expired and also have to use softmax function then 
#But it probs uses that automatically (thats true it automatically uses softmax when you give it three distinct output vars), but still ahve to then change my output variable y to like have it seperate for logistic and with three levels for lightgbm and neural net

import lightgbm as lgb 
from sklearn.calibration import CalibratedClassifierCV



def train_lgbm_model(X_train, y_train, weights, params = None, for_tuning = False):

    # light GBM 
    
    if params is None:
        params = {
        
        #Structural
        'objective' : 'binary', # Our Output var is multiclass for lgbm
        #'num_class' : 3, # How many diff classes (0,1,2 are the classes in our case)
        'boosting_type' : 'gbdt', #The default gradient boosting 
        'metric' : 'binary_logloss', #Metric to measure perforance
        'n_jobs' : -1, #Using all available threads in cpu 
        'random_state' : 69 , #just set random seed for reproducability
        
        #Overall Tuning
        'n_estimators' : 150, # number of sequential trees, i.e number of boosting rounds 
        'learning_rate' :  0.017556499318855254, # scales contribution of each individual tree
        'num_leaves' : 37, #max num of leaves, i.e terminal nodes, allowed in each tree 
        'min_child_samples': 2469 #Minimum number of data points required to create a new split in a leaf node
        
        #Fine Tuning only use this after completion of tuning above and maybe not even at all, look at https://www.geeksforgeeks.org/machine-learning/lightgbm-regularization-parameters/
        
            }
    
    
    base_lgbm = lgb.LGBMClassifier(**params) #** is for unpacking the library from above  
    
    base_lgbm_model = base_lgbm.fit(X_train, y_train, sample_weight = weights)
    
    
    if for_tuning: #This is just for speed optimsation for using Optuna as for that you just need the base model so dont want to waste time calibrating
        return base_lgbm_model
    
    
    #calibrating the model, trees use stepfunctions, so use isotonic to match that or smooth it out??, do more research on this
    # im pretty sure trees dont require scalar and only care about relative ordering 

    calibrated_lgbm = CalibratedClassifierCV(
        estimator = base_lgbm,
        method = 'isotonic',
        cv = 5
        )
  
    calibrated_lgbm_model = calibrated_lgbm.fit(X_train, y_train, sample_weight = weights)
    
    return base_lgbm_model, calibrated_lgbm_model

