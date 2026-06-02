#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 19:41:13 2026

@author: jesseruijer
"""
import scipy.io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, roc_auc_score, brier_score_loss, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import seaborn as sns

from Functions import time_in_hours, plots, plot_feature, plot_corr_map


pd.set_option('display.max_columns', None)

file_path = '../STOCKS/INTC_NASDAQ/INTC_20140401_NASDAQ.mat'
file_path_MO = '../STOCKS/INTC_NASDAQ/Market Order/INTC_20140401.mat'


def import_data(file_path, file_path_MO):
    
    #Takes input matlab data and transforms it into a dictionary of Pandas Dataframes
    
    mat_data = scipy.io.loadmat(file_path)
    mat_data_MO = scipy.io.loadmat(file_path_MO)
    
    
    struct_data_E = mat_data['data'][0, 0]['Event']
    df_Event = pd.DataFrame(struct_data_E)
    df_Event.columns = ["TOD", "ID", "Type", "Vol", "Price", "unknown1", "unknown2"]

    struct_data_BV = mat_data['data'][0, 0]['BuyVolume']
    df_BV = pd.DataFrame(struct_data_BV)

    struct_data_SV = mat_data['data'][0, 0]['SellVolume']
    df_SV = pd.DataFrame(struct_data_SV)

    struct_data_BP = mat_data['data'][0, 0]['BuyPrice']
    df_BP = pd.DataFrame(struct_data_BP)

    struct_data_SP = mat_data['data'][0, 0]['SellPrice']
    df_SP = pd.DataFrame(struct_data_SP)
    
    
    struct_data_MO = mat_data_MO['MO']
    df_MO = pd.DataFrame(struct_data_MO)
    df_MO.rename(columns={
        0: "TOD",
        1: "BBP",
        2: "BAP",
        3: "BBV",
        4: "BAV",
        5: "APPS",
        6: "Vol",
        7: "BorS",
        8: "MME"}, inplace = True)
    
    #the dictionary
    data_set = {
        "Event" : df_Event,
        "BuyVol" : df_BV,
        "SellVol" : df_SV,
        "BuyPrice" : df_BP,
        "SellPrice" : df_SP,
        "MO" : df_MO
        }
    
    return data_set


def clean_data(raw_data):
    
    
    df_E = raw_data["Event"]
    df_BV = raw_data["BuyVol"]
    df_SV = raw_data["SellVol"]
    df_BP = raw_data["BuyPrice"]
    df_SP = raw_data["SellPrice"]
    df_MO = raw_data["MO"]
    
   #Cleans dataframes to not include first and last 30 min of trading hours and not include 88 and 84
    valid_row_mask = (
        (df_E["TOD"] >= 36000000) &
        (df_E["TOD"] <= 55800000) &
        (df_E["Type"] != 88) &
        (df_E["Type"] != 84)
        )
    
    df_E_without_noise = df_E[valid_row_mask]
    df_BV_without_noise = df_BV[valid_row_mask]
    df_SV_without_noise = df_SV[valid_row_mask]
    df_BP_without_noise = df_BP[valid_row_mask]
    df_SP_without_noise = df_SP[valid_row_mask]

    valid_row_mask_MO = (
        (df_MO["TOD"] >= 36000000) &
        (df_MO["TOD"] <= 55800000)
        )

    df_MO_without_noise = df_MO[valid_row_mask_MO]
    
    data_set_clean = {
        "Event" : df_E_without_noise,
        "BuyVol" : df_BV_without_noise,
        "SellVol" : df_SV_without_noise,
        "BuyPrice" : df_BP_without_noise,
        "SellPrice" : df_SP_without_noise,
        "MO" : df_MO_without_noise
        }
    
    return data_set_clean

def data_regressors(rawdata, cleandata):
    
    df_E = cleandata["Event"]
    df_E2 = rawdata["Event"]
    df_BV = cleandata["BuyVol"]
    df_SV = cleandata["SellVol"]
    df_BP = cleandata["BuyPrice"]
    df_SP = cleandata["SellPrice"]
    
    Regressors_df = pd.DataFrame()
    Regressors_df["TOD"] = df_E["TOD"]
    Regressors_df["BASpread"] = df_SP[0] - df_BP[0]
    #below fill.na(0) means it will fill imbalcne with 0 if theres a NaN situation
    #Q imbalance is just using the best bid and ask volumes
    Regressors_df["QImbalance"] = ((df_BV[0]-df_SV[0])/(df_BV[0]+ df_SV[0])).fillna(0)
    Regressors_df["AbsQImbalance"] = Regressors_df["QImbalance"].abs()
    #Total Vol imbalance uses sum of the 20 cols provided in the data
    #axis=1 does across cols, axis=0 does across rows
    Regressors_df["TotalVolImbalance"] = ((df_BV.sum(axis=1)-df_SV.sum(axis=1))/(df_BV.sum(axis=1)+ df_SV.sum(axis=1))).fillna(0)


    weights = [1/(i) for i in range(1,21)]

    Regressors_df["Weighted Vol Imbalance"] = (((weights*df_BV).sum(axis=1)-(weights*df_SV).sum(axis=1))/((weights*df_BV).sum(axis=1)+ (weights*df_SV).sum(axis=1))).fillna(0)
    Regressors_df["Midprice"] = (df_BP[0]+df_SP[0])/2
    Regressors_df["Microprice"] = ((df_BV[0]*df_SP[0])+(df_SV[0]*df_BP[0]))/(df_BV[0]+df_SV[0])
    
    
    #Calculates how far order is awaay from best bid or best ask
    Regressors_df['distance_to_best_price_ask'] = cleandata['SellPrice'][0] - cleandata[]
    
    train_mask = (
            (df_E2["Type"] != 88) &
            (df_E2["Type"] != 84) &
            #To remove after hour trading data
            (df_E2["TOD"] <= 57601000)
            )
    train_event_df = df_E2[train_mask]

    #Maps all the cancel or fill parts of LOs to df
    outcome_df = train_event_df[train_event_df["Type"].isin([67,68,69,70])]

    #Groups by ID and looks at what happens last, i.e that will be fill or cancel, but also looks if there was a partial fill at any time during its life 
    fill_map = outcome_df.groupby("ID")["Type"].apply(lambda x: x.isin([69,70]).any())

    #Converts it to binary 1 for fill to use in logistic regression
    fill_map = fill_map.astype(int)

    #Now map it to my Regressors df feature matrix i made above

    Regressors_df["Fill_NoFill"] = df_E["ID"].astype(int).map(fill_map)

    Clean_Regression_Data = Regressors_df.dropna(subset = ["Fill_NoFill"])

    
    return Clean_Regression_Data

rawdata = import_data(file_path, file_path_MO)
cleandata = clean_data(rawdata)
regressormatrix = data_regressors(rawdata, cleandata)

########## Replicating the slides
print(f" Total number of events on 1 April 2014 of INTC is {len(cleandata['Event'])}")
print(f" Amount of Market Orders on 1 April 2014 of INTC is {len(cleandata['MO'])}")
print(f' Percentage of MO per total number of events on 1 April 2014 INTC is {(len(cleandata["MO"])/len(cleandata["Event"])*100):.04f}%')

buy_no_walk = (cleandata['MO']['APPS'] == cleandata['MO']['BAP'] ) & (cleandata['MO']['BorS'] == -1)
sell_no_walk = (cleandata['MO']['APPS'] == cleandata['MO']['BBP'] ) & (cleandata['MO']['BorS'] == 1)
total_walk = buy_no_walk | sell_no_walk # | is OR operator

print(f" Percentage of orders that did not walk the book for INTC on April 1 2024 is {(total_walk.sum()/(len(cleandata['MO'])) * 100):.2f} % ")
######### 


#unknown 2 could very well be what side of the book the order appears on, but prove that below
#unknown 1 could be if its a visible order, proof that using 84 which is in raw and not in clean


print(cleandata["Event"][cleandata["Event"]["Type"] == 70])
#How many shares are sitting ahead of this in the queue
#Queue_pos





#try to recreate the graph at 11 am
plots(cleandata, 39600000)

#BASpread seems to not be relevant for INTC since the spread is pretty much one cent for the whole day
plot_feature(regressormatrix, "TotalVolImbalance")

plot_corr_map(regressormatrix)


#Starting logistic regression

#Must train model on filtered Data, but can search for its outcome on full data in terms of time, i.e an order might still get filled or not after 3:30 PM
#Filtered is already above, here below is not constrained on time but still constrained on not including 88 and 84

file_path2 = '../STOCKS/INTC_NASDAQ/INTC_20140424_NASDAQ.mat'
file_path_MO2 = '../STOCKS/INTC_NASDAQ/Market Order/INTC_20140424.mat'
rawdata2 = import_data(file_path2, file_path_MO2)
cleandata2 = clean_data(rawdata2)
regressormatrix2 = data_regressors(rawdata2, cleandata2)


#Below prints how many 1s and 0s we had
print("\n Total Fills vs Cancels")
print(regressormatrix["Fill_NoFill"].value_counts())

#Below prints how many counts of Types we had
print(cleandata["Event"]["Type"].value_counts())

print(regressormatrix.head())

#Logistic regression using statsmodels lib
lr_model = LogisticRegression()
scalar = StandardScaler()

y_train = regressormatrix["Fill_NoFill"]
features = ['QImbalance', 'AbsQImbalance', 'TotalVolImbalance', 'Weighted Vol Imbalance', 
            'Midprice', 'Microprice'] 
X_train = regressormatrix[features]

X_train_standardised = scalar.fit_transform(X_train) #Here we fit and transform
#Fit Scikit logistic regrssion

lr_model.fit(X_train_standardised, y_train)

y_true = regressormatrix2["Fill_NoFill"]

X_test = regressormatrix2[features]

X_test_scaled = scalar.transform(X_test) #Here we transform and not fit anymore i.e we use same scale as above so comparisons are valid

#Do some prediction using scikit learn
y_pred = lr_model.predict(X_test_scaled)
y_pred_prob = lr_model.predict_proba(X_test_scaled)[: , 1] # We are now only looking at the fill probabilities










model_coef_df = pd.DataFrame(
    
    {
     "Feature": features,
     "Coefficient (Log Odds)" : lr_model.coef_[0],  # We only predict binary classification so our model only has one row so access that with [0]
     "Odds Ratio": np.exp(lr_model.coef_[0])
     }
    
    )
print(model_coef_df.sort_values(by = "Odds Ratio", key=abs, ascending=False))


#Baseline fill percentage which i defined as the number of ones divided by number of ones and zeros in fill_map, which guarantees uniqueness by the fact i used .last in code before it
print(f"Baseline Fill percentage is {regressormatrix2['Fill_NoFill'].mean()*100:.2f} %")



print("Engine metrics")

brierscore = brier_score_loss(y_true, y_pred_prob)
print(f'Brier score is {brierscore:.3f}')

logloss = log_loss(y_true, y_pred_prob)
print(f'Logloss score is {logloss:.3f}')

aucscore = roc_auc_score(y_true, y_pred_prob)
print(f'AUC score is {aucscore:.3f}')


#######Work on LOB chronological order reconstruction 
    











