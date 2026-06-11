#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 10:42:59 2026

@author: jesseruijer
"""

#Importing libraries,classes, functions from other scripts

import lightgbm as lgb 
from sklearn.calibration import CalibratedClassifierCV
def train_lgbm_model(regressormatrix):

    # light GBM 
    
    lgbm_mdl_features = ['AbsQImbalance', 'Weighted Vol Imbalance', 
                  "DistanceToTouch", 'LogVolAhead', "LookBackHiddenVol" ,'BASpread', 'QImbalance', 'TotalVolImbalance', 'Midprice', 'Microprice', 
                                            'MOTrailingVol100ms', 'MOTrailingOrders100ms', 'LOTrailingVolPlaced100ms', 'LOTrailingCountOrdersPlaced100ms', 
                                            'LOTrailingVolCanceled100ms', 'LOTrailingCountOrdersCanceled100ms', 'LOTrailingVolExecuted100ms',
                                            'LOTrailingCountOrdersExecuted100ms', 'VolAhead']
    
    X_train_lgbm = regressormatrix[lgbm_mdl_features]
    y_train = regressormatrix["Fill_NoFill"]
    
    #imbalance ratio since our outcome variable is heavily skewed
    dummy_fill_prob = regressormatrix['Fill_NoFill'].mean()
    imbalance_ratio = (1-dummy_fill_prob) / dummy_fill_prob

    base_lgb = lgb.LGBMClassifier(
        n_jobs = -1, #Using all available threads in cpu 
        n_estimators = 150, # number of sequential trees  
        learning_rate = 0.05, # scales contribution of each individual tree
        num_leaves = 31, #max num of leaves, i.e terminal nodes, allowed in each tree 
        random_state = 69 , #just set random seed for reproducability
        scale_pos_weight = imbalance_ratio #telling the loss function about the skewed output var i think, not sure yet
        ) 

    #calibrating the model, trees use stepfunctions, so use isotonic to match that or smooth it out??, do more research on this
    # im pretty sure trees dont require scalar and only care about relative ordering 

    calibrated_lgbm = CalibratedClassifierCV(
        estimator = base_lgb,
        method = 'isotonic',
        cv = 5
        )
    
    fitted_lgbm_model = calibrated_lgbm.fit(X_train_lgbm, y_train)
    
    return fitted_lgbm_model
