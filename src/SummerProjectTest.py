#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 19:41:13 2026

@author: jesseruijer
"""
#There are a lot more lob feature statistics i could add later
#Write something to classify if the stock is small or large tick and change analysis depending on that
#Finish prediction function
#Visualisation of performance vs dummy guessing
#Statistical evidence for my log regression model since now we just have the ORs but no p values or CIs
#maybe still use microprice in model but use it in some relative way or microprice relative to midprice, think of something
#Importing libraries,classes, functions from other scripts

import scipy.io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import precision_recall_curve, roc_auc_score, brier_score_loss, log_loss, roc_curve
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import seaborn as sns

from Functions import time_in_hours, plots, plot_feature, plot_corr_map, order_life

#Just a display feature in console so all columns are printed in console
pd.set_option('display.max_columns', None)

#Setting file paths
file_path = '../STOCKS/INTC_NASDAQ/INTC_20140401_NASDAQ.mat'
file_path_MO = '../STOCKS/INTC_NASDAQ/Market Order/INTC_20140401.mat'


def import_data(file_path, file_path_MO):
    
    #Takes input matlab data and transforms it into a dictionary of Pandas Dataframes
    
    mat_data = scipy.io.loadmat(file_path)
    mat_data_MO = scipy.io.loadmat(file_path_MO)
    
    
    struct_data_E = mat_data['data'][0, 0]['Event']
    df_Event = pd.DataFrame(struct_data_E)
    df_Event.columns = ["TOD", "ID", "Type", "Vol", "Price", "unknown1", "SideOfBook"] #BuySide = 1, Sellside = 0, note hidden orders can be pegged at midpoint so hard to classify those so maybe best to ignore this col for 84s
    
    #unknown 1 is literally just a row of ones and has no predictive power so remove it from df
    df_Event = df_Event.drop(columns= ["unknown1"])

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
    
   #Cleans dataframes to not include first and last 30 min of trading hours and not include 88 
    valid_row_mask = (
        (df_E["TOD"] >= 36000000) &
        (df_E["TOD"] <= 55800000) &
        (df_E["Type"] != 88)
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
   
    
    #For our machine to make accurate predictions we have to shift forward each row in the LOB BuySell Vol and Price
    #i.e for row 1 in event normally, row 1 in the LOB data would correspond to what happened immediately after 
    #the event in row 1, but to predict what happened to the event in row1 we need what the LOB looked like before that 
    #event happened so we need to shift row 0 from LOB down to row 1, can do this vectorize wise by .shift
    #So first we do that for the raw data so its all alligned, and then we can go to removing cols and stuff in cleandata
    
    df_BV = rawdata["BuyVol"].shift(1).loc[df_E.index]
    df_SV = rawdata["SellVol"].shift(1).loc[df_E.index]
    df_BP = rawdata["BuyPrice"].shift(1).loc[df_E.index]
    df_SP = rawdata["SellPrice"].shift(1).loc[df_E.index]
    
    
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
    
    hidden_vol = np.where(df_E["Type"] ==  84, df_E['Vol'], 0) #returns the vol of types 84 else zero in a new numpy array
    cum_vol_pad = np.pad(np.cumsum(hidden_vol), (1,0), constant_values = 0) #says add a zero to the start, nothing to the back and then we take cumsum of all the vols
    tod_values = df_E['TOD'].values
    lookback = 5000 #Lookback time in MS for hidden vol trades
    lookback_times = tod_values - lookback
    start_indices = np.searchsorted(tod_values, lookback_times, side='left')    #Does binary search st for every lookback time we calculate the row index it woud land on in tod_values
    current_indices = np.arange(1, len(tod_values) + 1)
    Regressors_df["LookBackHiddenVol"] = cum_vol_pad[current_indices] - cum_vol_pad[start_indices]
    
    weights = [1/(i) for i in range(1,21)]

    Regressors_df["Weighted Vol Imbalance"] = (((weights*df_BV).sum(axis=1)-(weights*df_SV).sum(axis=1))/((weights*df_BV).sum(axis=1)+ (weights*df_SV).sum(axis=1))).fillna(0)
    Regressors_df["Midprice"] = (df_BP[0]+df_SP[0])/2
    Regressors_df["Microprice"] = ((df_BV[0]*df_SP[0])+(df_SV[0]*df_BP[0]))/(df_BV[0]+df_SV[0])
    
    direction_of_order = df_E["SideOfBook"].values
    price_of_order = df_E["Price"].values
    best_bid = df_BP[0].values
    best_ask = df_SP[0].values
    
    distance_to_touch = np.where( #np.where works like an if condition, its this, else do this
        
        direction_of_order == 1,
        best_bid - price_of_order,
        price_of_order - best_ask
        
        )
    
    
    
    Regressors_df["DistanceToTouch"] = distance_to_touch   #How far a placed LO is from best bid or best ask
    
    #Calculates how far order is awaay from best bid or best ask
    
    #Vol Ahead looks at for a given placed limit order how much volume is ahead of it until best price
    #We create an empty array for all tods an event was placed, loop through all 20 levels of the order book prices we have and add volume to it if its in front of our order
    
    total_buy_vol_ahead = np.zeros(len(df_E))
    total_sell_vol_ahead = np.zeros(len(df_E))
    
    for i in range(20):
        lvl_buy_price = df_BP[i].values
        lvl_buy_vol = df_BV[i].values
        lvl_sell_price = df_SP[i].values
        lvl_sell_vol = df_SV[i].values
        
        total_buy_vol_ahead += np.where( lvl_buy_price >= price_of_order, lvl_buy_vol,0)
        total_sell_vol_ahead += np.where(lvl_sell_price <= price_of_order, lvl_sell_vol, 0)
        
    Vol_Ahead = np.where(
        direction_of_order == 1,
        total_buy_vol_ahead,
        total_sell_vol_ahead
        )
    
    Regressors_df['VolAhead'] = Vol_Ahead
    #we use log1p  which is log 1 + x since if an order is placed inside the spread it would have negative values and we cant take log of that 

    Regressors_df['LogVolAhead'] = np.log1p(Regressors_df['VolAhead'])
    train_mask = (
            (df_E2["Type"] != 88) &
            (df_E2["Type"] != 84) & #Here we do remove 84 since we cant train on something thats hidden i.e not in queue
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

#Adding the dataframes to variables for further use
rawdata = import_data(file_path, file_path_MO)
cleandata = clean_data(rawdata)
regressormatrix = data_regressors(rawdata, cleandata)


#############3########## Replicating the slides #############################
print(f" Total number of events on 1 April 2014 of INTC is {len(cleandata['Event'])}")
print(f" Amount of Market Orders on 1 April 2014 of INTC is {len(cleandata['MO'])}")
print(f' Percentage of MO per total number of events on 1 April 2014 INTC is {(len(cleandata["MO"])/len(cleandata["Event"])*100):.04f}%')

buy_no_walk = (cleandata['MO']['APPS'] == cleandata['MO']['BAP'] ) & (cleandata['MO']['BorS'] == -1)
sell_no_walk = (cleandata['MO']['APPS'] == cleandata['MO']['BBP'] ) & (cleandata['MO']['BorS'] == 1)
total_walk = buy_no_walk | sell_no_walk # | is OR operator

print(f" Percentage of orders that did not walk the book for INTC on April 1 2024 is {(total_walk.sum()/(len(cleandata['MO'])) * 100):.2f} % ")
#try to recreate the graph at 11 am
plots(cleandata, 39600000)
################################################################################3


################Proving Unknown 2 is what size of the book an event happens###########
#This shows whenever 66 or 83 we only get 1s and 0s respectively
print(cleandata["Event"][(cleandata["Event"]["Type"] == 66) & (cleandata["Event"]["SideOfBook"] == 0)])
print(cleandata["Event"][(cleandata["Event"]["Type"] == 83) & (cleandata["Event"]["SideOfBook"] == 1)])
#Must also check the side is correct for the other orders, most difficult will be the hidden orders
hidden0 = (rawdata['Event']['Type'] == 84) & (rawdata["Event"]["SideOfBook"] == 0)
hidden0index = rawdata['Event'][hidden0].index
look_up_index = hidden0index - 1 #To see what LOB looked like before LO was placed
#Adding .values is really important here because that removes the original indices from the df and just hands pandas lists of numbers without any pre fit indices
best_buy = rawdata["BuyPrice"].loc[look_up_index, 0].values # first col gives best buy price
best_sell = rawdata["SellPrice"].loc[look_up_index, 0].values # first col gives best sell price
trade_price84 = rawdata["Event"].loc[hidden0index, "Price"].values

res_df = pd.DataFrame(
    { "Trade price of 84": trade_price84,
     "Best Buy": best_buy,
     "Best Sell": best_sell}
    )

print(f'       Hidden Orders      \n {res_df.head(10)}')
#Here we may see orders exactly at mid price to be pegged there if they have to sell a lot they dont want to scare the market i think
#Sell LOs slightly above best buy is to ensure they can capture the incoming buyer immediately
#That gave some intuition about the location of the Hidden orders, now we can go on to show the sides are still correct
#For the other order types

target_event_types = [67, 68, 69, 70, 84]
mask = rawdata['Event']["Type"].isin(target_event_types)
index = rawdata['Event'][mask].index
unknown2val = rawdata['Event'].loc[index, 'SideOfBook'].values
eventprice = rawdata['Event'].loc[index, 'Price'].values

#Now look at LOB status before this order was placed
index_before = index - 1
best_bid = rawdata['BuyPrice'].loc[index_before, 0].values
best_ask = rawdata['SellPrice'].loc[index_before, 0].values
midprice_before = (best_ask + best_bid)/2
what_type = rawdata['Event'].loc[index, 'Type'].values

proof_df = pd.DataFrame(
    {'EventPrice': eventprice,
     'unknown2' : unknown2val,
     'Midprice': midprice_before,
     'Type': what_type
     })

#if price is below midprice its on the buy side and vice versa
proof_df['StrictlyBuySide%'] = (proof_df['EventPrice'] < proof_df['Midprice']).astype(int)*100 #astype just converts the boolean into 1 or 0
proof_df['StrictlyMidPrice%'] = (proof_df['EventPrice'] == proof_df['Midprice']).astype(int)*100
proof_df['StrictlySellSide%'] = (proof_df['EventPrice'] > proof_df['Midprice']).astype(int)*100

bucket_cols = ['Type', 'unknown2']
target_cols = ['StrictlyBuySide%', 'StrictlyMidPrice%', 'StrictlySellSide%']
grouped_buckets = proof_df.groupby(bucket_cols)
filtered_buckets = grouped_buckets[target_cols]
final_proof = filtered_buckets.mean()

#i.e for each type of event its calculated for the price of the order where the type was for whether that was on buy
#or sell side

print(f'                 Final Proof of unknown 2    \n {final_proof}')

#########################################################################################3


#BASpread seems to not be relevant for INTC since the spread is pretty much one cent for the whole day

#Some plots for exploratory data analysis and correlation matrices
plot_feature(regressormatrix, "QImbalance")

plot_corr_map(regressormatrix)


###################Starting logistic regression###########################

#Must train model on filtered Data, but can search for its outcome on full data in terms of time, i.e an order might still get filled or not after 3:30 PM
#Filtered is already above, here below is not constrained on time but still constrained on not including 88 and 84

#Importing training data
file_path2 = '../STOCKS/INTC_NASDAQ/INTC_20140424_NASDAQ.mat'
file_path_MO2 = '../STOCKS/INTC_NASDAQ/Market Order/INTC_20140424.mat'
rawdata2 = import_data(file_path2, file_path_MO2)
cleandata2 = clean_data(rawdata2)
regressormatrix2 = data_regressors(rawdata2, cleandata2)


#Logistic regression 
#Might use somethiing of platt scaling to wrap the log res model as log res doesnt work very well with data where the output is very skewed, i.e here we have much more cancels then fills
base_lr_model = LogisticRegression(max_iter=1000) # max iter higher then the standard 100 to take into account the noisy data we have
calibrated_model = CalibratedClassifierCV(estimator=base_lr_model, method='sigmoid', cv=5 ) #Do some research if and why 5 is good value for cross validation in ML
scalar = StandardScaler()
y_train = regressormatrix["Fill_NoFill"]

#Look at the required assumptions for logistic regression, i think need iid and for example
#below i included price related features but if the price changes they dont follow the same distribution
#on a given day anymore and the whole model breaks, so now the model only looks at volume
#and position dynamics

log_mdl_features = ['AbsQImbalance', 'Weighted Vol Imbalance', 
              "DistanceToTouch", 'LogVolAhead', "LookBackHiddenVol"] 
X_train = regressormatrix[log_mdl_features]

X_train_standardised = scalar.fit_transform(X_train) #Here we fit and transform
#Fit Scikit logistic regrssion

calibrated_model.fit(X_train_standardised, y_train)

base_lr_model.fit(X_train_standardised, y_train)

y_true = regressormatrix2["Fill_NoFill"]

X_test = regressormatrix2[log_mdl_features]

X_test_scaled = scalar.transform(X_test) #Here we transform and not fit anymore i.e we use same scale as above so comparisons are valid

#Do some prediction using scikit learn
y_pred = calibrated_model.predict(X_test_scaled)
y_pred_prob = calibrated_model.predict_proba(X_test_scaled)[: , 1] # We are now only looking at the fill probabilities



model_coef_df = pd.DataFrame(
    
    {
     "Feature": log_mdl_features,
     "Coefficient (Log Odds)" : base_lr_model.coef_[0],  # We only predict binary classification so our model only has one row so access that with [0]
     "Odds Ratio": np.exp(base_lr_model.coef_[0])
     }
    
    )
print(f'                  Logistic Regression ORs \n {model_coef_df.sort_values(by = "Odds Ratio", key=abs, ascending=False)}')


#Baseline fill percentage which i defined as the number of ones divided by number of ones and zeros in fill_map, which guarantees uniqueness by the fact i used .last in code before it
print(f"Baseline Fill percentage is {regressormatrix2['Fill_NoFill'].mean()*100:.2f} %")

#because were making a probability engine using logistic regression we must look at brier score and log loss to evaulte it
#Confusion matrices and precisions for ex dont make too much sense here since then you need to define a treshold for when a probability gets put in the category
#0 or 1 where for us that doesnt matter we're just interested in the pure probability of an order beig filled

print("Engine metrics")

brierscore = brier_score_loss(y_true, y_pred_prob)
print(f'Brier score is {brierscore:.3f}')

logloss = log_loss(y_true, y_pred_prob)
print(f'Logloss score is {logloss:.3f}')

aucscore = roc_auc_score(y_true, y_pred_prob)
print(f'AUC score is {aucscore:.3f}')



#Visualisiton of performance and comparison to baseline dummy model which just guesses a baseline percentage on each order for it being filled 
#Dummy y fill prob is just an array of length y true with all entries equal to dummy fill prob

dummy_fill_prob = regressormatrix['Fill_NoFill'].mean()
dummy_y_pred_prob = np.full(len(y_true), dummy_fill_prob)


print("Dummy metrics")

dummy_brierscore = brier_score_loss(y_true, dummy_y_pred_prob)
print(f'Dummy Brier score is {dummy_brierscore:.3f}')

dummy_logloss = log_loss(y_true, dummy_y_pred_prob)
print(f' Dummy Logloss score is {dummy_logloss:.3f}')

dummy_aucscore = roc_auc_score(y_true, dummy_y_pred_prob)
print(f'Dummy AUC score is {dummy_aucscore:.3f}')













# def feature_of_order():
    
#     return features_of_order

# def predict_order_fill_prob(features):
#     #predicts specific probability for a given limit order being filled using the logistic regression engine from above
    
    
#     input_df = pd.DataFrame({
#             'AbsQImbalance' : features[0],
#             'Weighted Vol Imbalance' : features[1],
#             'Microprice' : features[2],
#             'DistanceToTouch' : features[3],
#             'LogVolAhead' : features[4],
#             'LookBackHiddenVol' : features[-1] 
#             })
    
#     scaled_input  = scalar.transform(input_df)
#     fill_prob = lr_model.predict_log_proba(scaled_input)[0,1]
    
#     return fill_prob

# print(predict_order_fill_prob(feature_of_order()))








