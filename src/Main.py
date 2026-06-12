#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 19:41:13 2026

@author: jesseruijer
"""
#There are a lot more lob feature statistics i could add later make sure to add them to feature lists and corr plots
#Work on data cleaning from yesterday and soln to the fill no fill variable
#The label doesnt matter for 88 since theres no bid and ask side there
#Is there crossing in the graphs before the market opens maybe my graphs overlap
#Mabye the vol of 88 at eod is the amount of vol and the price could maybe be midprice or some other price
#Make something to access the feature matrix at a time of day and that it doesnt display empty df if the ms isnt right it should then round down to the nearest time
#maybe i also need to scale down the prices by dividing them by 10000 for the logistic regression but not sure ƒplots

#Importing libraries,classes, functions from other scripts
import pandas as pd
import config
import joblib

from DataAndFeatureEngineering import import_data, clean_data, data_regressors
from LightGBMEngine import train_lgbm_model
from ModelEvaluation import test_model
from LogisticRegressionEngine import train_logistic_model


#Just a display feature in console so all columns are printed in console
pd.set_option('display.max_columns', None)

def prep_data_daily(file_path, file_path_mo):
    print(f'Runs full pipeline for {file_path} and {file_path_mo}')
    rawdata = import_data(file_path, file_path_mo)
    cleandata = clean_data(rawdata)
    regressormatrix = data_regressors(rawdata, cleandata)
    
    return regressormatrix


def save_data():
    print('Builds parquet files for easy storage and optimisation')
    train_matrix = prep_data_daily(config.TRAIN_FILE_PATH, config.TRAIN_FILE_PATH_MO)
  
    test_matrix = prep_data_daily(config.TEST_FILE_PATH, config.TEST_FILE_PATH_MO)
    
    
    #Save matrix as a parquet file
    train_matrix.to_parquet("../data/processed/INTC_train_matrix_2014_04_01.parquet")
    test_matrix.to_parquet("../data/processed/INTC_test_matrix_2014_04_24.parquet")
   
def run_project():
    
    print('Loading Data')
    
    train_matrix = pd.read_parquet("../data/processed/INTC_train_matrix_2014_04_01.parquet")
    test_matrix = pd.read_parquet("../data/processed/INTC_test_matrix_2014_04_24.parquet")
    
 
    
    logistic_regX = train_matrix[config.LOGISTIC_MODEL_FEATURES]
    logistic_regY = train_matrix[config.TARGET]
    
    lgbm_X = train_matrix[config.LGBM_MODEL_FEATURES]
    lgbm_Y = train_matrix[config.TARGET]
    
    base_lr, calibrated_lr, scalar_lr = train_logistic_model(logistic_regX, logistic_regY)
    
    base_lgbm, calibrated_lgbm = train_lgbm_model(lgbm_X, lgbm_Y)
   
    print('Training and Testing Model')
    
    logistic_test = test_model(
        test_data = test_matrix, 
        base_model = base_lr,
        calibrated_model = calibrated_lr,
        scalar = scalar_lr,
        model_name = 'Logistic Regression',
        features = config.LOGISTIC_MODEL_FEATURES
        )
    
    lgbm_test = test_model(
        test_data = test_matrix, 
        base_model = base_lgbm,
        calibrated_model = calibrated_lgbm,
        scalar = None,
        model_name = 'Light Gradient Boosted Model Regression',
        features = config.LGBM_MODEL_FEATURES
        )


    
if __name__ == "__main__":
    
    #When theres new data uncomment this below and run once to store the data, if running same data leave this commented
    
    # save_data()
    
    #Only run the whole project if explicitly call main.py
    run_project()





































