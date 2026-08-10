#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 10:42:59 2026

@author: jesseruijer
"""

"""
LGBM Engine

Trains & Calibrates
"""

import lightgbm as lgb 
import gc
from lightgbm import early_stopping
from sklearn.calibration import CalibratedClassifierCV
import config
import pandas as pd


def train_lgbm_model(X_train:pd.DataFrame, y_train:pd.Series, weights:pd.Series, X_calib:pd.DataFrame, y_calib:pd.Series, weights_calib:pd.Series , X_val:pd.DataFrame | None = None, y_val:pd.Series|None = None, val_weights:pd.Series|None = None,  params: dict|None = None, for_tuning: bool = False) -> lgb.LGBMClassifier | tuple[lgb.LGBMClassifier, CalibratedClassifierCV]:
    
    
    """
    Training LGBM
    """

    # lightGBM 
    
    if params is None:
        params = {
        
        #Structural
        'objective' : 'binary',
        'boosting_type' : 'gbdt', #The default gradient boosting 
        'metric' : 'binary_logloss', #Metric to ensure lgbm knows its goal is binary classification, and then the binary logloss is just for it to measrue performance
        'n_jobs' : -1, #Using all available threads in cpu 
        'random_state' : config.RANDOM_SEED , #just set random seed for reproducability
        
        #Overall Tuning
        'n_estimators' : 885, # number of sequential trees, i.e number of boosting rounds, set very high on purpose since early stopping below will catch it. as each step you test on the validation set its not allowed to train on and then if the model hasnt improved for 50 rounds then stop 
        'learning_rate' :  0.011286873546151143, # scales contribution of each individual tree
        'num_leaves' : 147, #max num of leaves, i.e terminal nodes, allowed in each tree 
        'min_child_samples': 2912, #Minimum number of data points required to create a new split in a leaf node
        'verbose': 1
            }
    
    
    base_lgbm = lgb.LGBMClassifier(**params) #** is for unpacking the library from above  

    
    fit_kwargs = {'sample_weight': weights}
    
    #Only need the below when running optuna 
    if X_val is not None and y_val is not None:
        fit_kwargs['eval_set'] = [(X_val, y_val)]
        fit_kwargs['callbacks'] = [early_stopping(stopping_rounds = 50)]
        if val_weights is not None:
            fit_kwargs['eval_sample_weight'] = [val_weights]
            
    base_lgbm_model = base_lgbm.fit(X_train, y_train, **fit_kwargs)
    
    if for_tuning: #This is just for speed optimsation for using Optuna as for that you just need the base model so dont want to waste time calibrating
        return base_lgbm_model
    
    del X_train, y_train, weights
    gc.collect()
    
    #calibrating base model to match event likelihood
    calibrated_lgbm = CalibratedClassifierCV(
        estimator = base_lgbm_model,
        method = 'isotonic',        #To calibrate raw lgbm output to LOB 
        cv = 'prefit' # fit calibrator only on reserved data for calibration, since model has already been trained before this overrides cross validation and prevents us from training on past data since our data is in chronological order
        )
  
    calibrated_lgbm_model = calibrated_lgbm.fit(X_calib, y_calib, sample_weight = weights_calib)
    
    del X_calib, X_val, weights_calib
    gc.collect()
    
    return base_lgbm_model, calibrated_lgbm_model