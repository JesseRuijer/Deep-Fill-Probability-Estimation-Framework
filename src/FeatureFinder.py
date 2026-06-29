#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 10:21:54 2026

@author: jesseruijer
"""
import shap
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

#For Shap use testdata, as its the same features as trained on
#For l1 finder for logistic, you have to use traindata

def feature_finder(model, model_name, data, features, y_train, weights):

    def shap_feature_finder(model, data, features):
        print("\n--- Calculating SHAP Values ---")
        
        # TreeExplainer is hyper-optimized for LightGBM/XGBoost
        explainer = shap.TreeExplainer(model)
        
        X_test = data[features]
        
        # Calculate the Shapley values for the test set
        shap_values = explainer.shap_values(X_test)
        
        # Generate the standard SHAP summary plot
        shap.summary_plot(shap_values, X_test, feature_names=features)
        
        #Write a script to obtain the top 10 features
        
        mean_abs_shap = np.abs(shap_values).mean(axis = 0)
        
        shap_importance = pd.DataFrame({
            'Feature': features,
            'Mean Abs Shap': mean_abs_shap}).sort_values(by = 'Mean Abs Shap', ascending = False)
        print(shap_importance)
        top_10_features = shap_importance.head(10)['Feature'].tolist()
        
        print(f'Final Top 10 Features for LGBM are: {top_10_features}')
        
        
    def logistic_best_feature_finder(model, data, features, y_train, weights):
        #Think liblinear optimised for binary large data
        
        X_train = data[features]
        scalar = StandardScaler()
        X_train_scaled = scalar.fit_transform(X_train)
        
        l1_model = LogisticRegression(
            penalty = 'l1',
            solver = 'liblinear',
            n_jobs = -1,
            C = 0.0001, #Strictness of allowing features, 1 is not strict, 0.1 is strict
            max_iter = 150,
            random_state = 68,
            tol = 0.01, #just another tuning parameter i could remove later, but it was nexessary to get l1 to work on the correlated features
            verbose = 1
            )
    
        l1_model.fit(X_train_scaled, y_train, sample_weight = weights)
        
        coefficients = l1_model.coef_[0]
        
        feature_importance = pd.DataFrame({
            'Feature': features,
            'Coefficient': coefficients,
            'Abs Coefficient': np.abs(coefficients)
            })
            
        print(feature_importance)
        survivor_features = feature_importance[feature_importance['Abs Coefficient'] > 0].copy()
        survivor_features = survivor_features.sort_values(by = 'Abs Coefficient', ascending = False)
        
        result = survivor_features['Feature'].tolist()
        
        print(result)
    
    if model_name == "Light Gradient Boosted Model":
         shap_feature_finder(model, data, features)
         
    if model_name == "Logistic Regression":
         logistic_best_feature_finder(model, data, features, y_train, weights)
         
         