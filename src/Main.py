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
   
def run_project():
    
    
    train_matrix = prep_data_daily(config.TRAIN_FILE_PATH, config.TRAIN_FILE_PATH_MO)
  
    test_matrix = prep_data_daily(config.TEST_FILE_PATH, config.TEST_FILE_PATH_MO)
    
    lgbm_trained = train_lgbm_model(train_matrix)
    
    lgbm_test = test_model(test_matrix)
    
    
    
    
    
if __name__ == "__main__":
    #Only run the whole project if explicitly call main.py
    run_project()





































