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


#has to properly get the weights from dataengineering

def train_logistic_model(X_train, y_train):
    
   #Logistic regression 
   #Might use somethiing of platt scaling to wrap the log res model as log res doesnt work very well with data where the output is very skewed, i.e here we have much more cancels then fills
    base_logistic_model = LogisticRegression(max_iter=1000) # max iter higher then the standard 100 to take into account the noisy data we have
    calibrated_model = CalibratedClassifierCV(estimator=base_logistic_model, method='sigmoid', cv=5 ) #Do some research if and why 5 is good value for cross validation in ML
    scalar = StandardScaler()
    
    #Look at the required assumptions for logistic regression, i think need iid and for example
    #below i included price related features but if the price changes they dont follow the same distribution
    #on a given day anymore and the whole model breaks, so now the model only looks at volume
    #and position dynamics
    

    
    X_train_standardised = scalar.fit_transform(X_train) #Here we fit and transform
    #Fit Scikit logistic regrssion
    
    calibrated_logistic_model = calibrated_model.fit(X_train_standardised, y_train)
    
    #Just saved the base model as well here, maybe thats useful for log odss not sure
    base_logistic_model.fit(X_train_standardised, y_train)
    
    return base_logistic_model, calibrated_logistic_model, scalar
    



