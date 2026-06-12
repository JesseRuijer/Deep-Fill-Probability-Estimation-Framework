#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 10:42:59 2026

@author: jesseruijer
"""

#Importing libraries,classes, functions from other scripts

import lightgbm as lgb 
from sklearn.calibration import CalibratedClassifierCV
def train_lgbm_model(X_train, y_train):

    # light GBM 

    base_lgbm = lgb.LGBMClassifier(
        n_jobs = -1, #Using all available threads in cpu 
        n_estimators = 150, # number of sequential trees  
        learning_rate = 0.05, # scales contribution of each individual tree
        num_leaves = 31, #max num of leaves, i.e terminal nodes, allowed in each tree 
        random_state = 69 , #just set random seed for reproducability
        #is_unbalance = True # deactivated this because use calibrated class later but look more into this telling the loss function about the skewed output var i think, not sure yet
        ) 

    #calibrating the model, trees use stepfunctions, so use isotonic to match that or smooth it out??, do more research on this
    # im pretty sure trees dont require scalar and only care about relative ordering 

    calibrated_lgbm = CalibratedClassifierCV(
        estimator = base_lgbm,
        method = 'isotonic',
        cv = 5
        )
    
    base_lgbm_model = base_lgbm.fit(X_train, y_train)
    
    calibrated_lgbm_model = calibrated_lgbm.fit(X_train, y_train)
    
    return base_lgbm_model, calibrated_lgbm_model

