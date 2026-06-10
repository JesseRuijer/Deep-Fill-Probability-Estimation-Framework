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

#Importing libraries,classes, functions from other scripts

import scipy.io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import precision_recall_curve, roc_auc_score, brier_score_loss, log_loss, roc_curve, average_precision_score
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import lightgbm as lgb 

from Functions import time_in_hours, plots, plot_feature, plot_corr_map, order_life, time_to_hours

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
    df_MO = cleandata['MO']
    
    opening_time = 34200000
    closing_time = 57600000
    
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
    ###########################Creating some non linear features#######################
    MO_tod_values = df_MO['TOD'].values #remeber .values makes it into np array since using for loops for this or pandas functions would take forever
    MO_vol_values = df_MO['Vol'].values
    
    cum_mo_vol = np.pad(np.cumsum(MO_vol_values), (1,0), constant_values = 0)      #pads to add a zero at thes start and then cumsum calculates the running total so to know the order arrival rate between two different times you just calculate the difference in their total running values
    
    lookback_intensity = 100 # time we want to look back for in ms
    event_tod_values = df_E['TOD'].values
    lookback_starting_times = event_tod_values - lookback_intensity
    
    mo_start_indices = np.searchsorted(MO_tod_values, lookback_starting_times, side = 'left')   #Finds the row indices where the lookback window starts and below where it finishes
    mo_end_indices = np.searchsorted(MO_tod_values, event_tod_values, side = 'right') 
    
    Regressors_df['MOTrailingVol100ms'] = cum_mo_vol[mo_end_indices] -  cum_mo_vol[mo_start_indices]
    Regressors_df['MOTrailingOrders100ms'] = mo_end_indices - mo_start_indices
    
    #Building something similar for the LOs, the adding and the cancelation activities
    mask_add = df_E['Type'].isin([66,83])
    mask_cancel = df_E['Type'].isin([67,68])
    mask_execute = df_E['Type'].isin([69,70])
    
    lo_add_vals_tod = df_E.loc[mask_add, 'TOD'].values
    lo_add_vals_vol = df_E.loc[mask_add, 'Vol'].values
    
    lo_cancel_vals_tod = df_E.loc[mask_cancel, 'TOD'].values
    lo_cancel_vals_vol = df_E.loc[mask_cancel, 'Vol'].values
    
    lo_execute_vals_tod = df_E.loc[mask_execute, 'TOD'].values
    lo_execute_vals_vol = df_E.loc[mask_execute, 'Vol'].values
    
    added_lo_cumsum = np.pad(np.cumsum(lo_add_vals_vol), (1,0) , constant_values = 0)
    cancel_lo_cumsum = np.pad(np.cumsum(lo_cancel_vals_vol), (1,0) , constant_values = 0)
    execute_lo_cumsum = np.pad(np.cumsum(lo_execute_vals_vol), (1,0) , constant_values = 0)
    
    added_lo_start_indices = np.searchsorted(lo_add_vals_tod, lookback_starting_times, side = 'left')   #Finds the row indices where the lookback window starts and below where it finishes
    added_lo_end_indices = np.searchsorted(lo_add_vals_tod, event_tod_values, side = 'right') 
   
    cancel_lo_start_indices = np.searchsorted(lo_cancel_vals_tod, lookback_starting_times, side = 'left')   #Finds the row indices where the lookback window starts and below where it finishes
    cancel_lo_end_indices = np.searchsorted(lo_cancel_vals_tod, event_tod_values, side = 'right') 
    
    execute_lo_start_indices = np.searchsorted(lo_execute_vals_tod, lookback_starting_times, side = 'left')   #Finds the row indices where the lookback window starts and below where it finishes
    execute_lo_end_indices = np.searchsorted(lo_execute_vals_tod, event_tod_values, side = 'right') 
    
    Regressors_df['LOTrailingVolPlaced100ms'] = added_lo_cumsum[added_lo_end_indices] -  added_lo_cumsum[added_lo_start_indices]
    Regressors_df['LOTrailingCountOrdersPlaced100ms'] = added_lo_end_indices - added_lo_start_indices
   
    Regressors_df['LOTrailingVolCanceled100ms'] = cancel_lo_cumsum[cancel_lo_end_indices] -  cancel_lo_cumsum[cancel_lo_start_indices]
    Regressors_df['LOTrailingCountOrdersCanceled100ms'] = cancel_lo_end_indices - cancel_lo_start_indices
    
    Regressors_df['LOTrailingVolExecuted100ms'] =  execute_lo_cumsum[execute_lo_end_indices] -   execute_lo_cumsum[execute_lo_start_indices]
    Regressors_df['LOTrailingCountOrdersExecuted100ms'] = execute_lo_end_indices - execute_lo_start_indices
    ###########################################
    
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

    #This could still be useful if we just want to look at if a random order will be filled partially or fully so dont remove this yet, maybe put it in another function
    #Groups by ID and looks at what happens last, i.e that will be fill or cancel, but also looks if there was a partial fill at any time during its life 
    fill_map = outcome_df.groupby("ID")["Type"].apply(lambda x: x.isin([69,70]).any())

    #Converts it to binary 1 for fill to use in logistic regression
    fill_map = fill_map.astype(int)

    #Now map it to my Regressors df feature matrix i made above

    Regressors_df["Fill_NoFill"] = df_E["ID"].astype(int).map(fill_map)

    Clean_Regression_Data = Regressors_df.dropna(subset = ["Fill_NoFill"])
    
    #Building a regime classifier which uses categorical variables to tell in what regime of day we are in
    Regressors_df['Regime'] = np.where(Regressors_df['TOD'] < opening_time , 0,    #Pre Market                         
                              np.where(Regressors_df['TOD'] < 36000000 , 1,    # 30 min vol after opening
                              np.where(Regressors_df['TOD'] < 55800000 , 2,    #Regular Market hours without first and last 30 min
                              np.where(Regressors_df['TOD'] < closing_time , 3,    #30 min high volatilitiy time before closing
                              4))))                                            #After market hours
    
    #need to alter the fill stuff maybe some sort of weighted splitting since the unit stuff if i were to use something like duplicating for that would just murder my RAM
    #might make a third classificaiton which is not filled at end of day and therefore cancelled, which is different then being canceled during day 
    
    #orders that made it to the end of the day without being filled or canceled 
    
    Regressors_df['TimeTillMarketClose'] = closing_time - df_E['TOD']
    
    #########Trying to make the continuous order tracker that correctly tracks partial fills and then lets the rest continue for the rest of the day as a new parent order
    
    Regressors_df['Type'] = df_E['Type']
    Regressors_df['ID'] = df_E['ID']
    Regressors_df['Vol'] = df_E['Vol']
    
    
    
    Regressors_df['PartialVols'] = np.where(Regressors_df['Type'].isin([67,68,69,70]), Regressors_df['Vol'] , 0)
    
    Regressors_df['OriginalVol'] = Regressors_df.groupby('ID')['Vol'].transform('first') #Looks at first entry and copies all entries to be like that, i.e the starting vol of any unique order id 
    Regressors_df['VolTaken'] = Regressors_df.groupby('ID')['PartialVols'].cumsum()
    
    #Vol of the new parent order
    Regressors_df['UnfilledAfter'] = Regressors_df['OriginalVol'] - Regressors_df['VolTaken']
    
    #Look if the remaining vol filled or canceled 
    Regressors_df['ExecutedVol'] = np.where(Regressors_df['Type'].isin([69,70]), Regressors_df['Vol'] , 0)
    Regressors_df['CanceledVol'] = np.where(Regressors_df['Type'].isin([67,68]), Regressors_df['Vol'] , 0)
    
    #How much vol was executed or canceled after this millisecond
    
    Regressors_df['TotalExecutedAfter'] = (Regressors_df.iloc[::-1].groupby('ID')['ExecutedVol'].cumsum().iloc[::-1] - Regressors_df['ExecutedVol'])
    Regressors_df['TotalCanceledAfter'] = ((Regressors_df.iloc[::-1].groupby('ID')['CanceledVol'].cumsum().iloc[::-1]) - (Regressors_df['CanceledVol']))
    
    #Trying to make some sort of weighting
    
    snapshot_state_df = Regressors_df[Regressors_df['Type'].isin([67,68,69,70])].copy()
    
    fills_df = snapshot_state_df[snapshot_state_df['TotalExecutedAfter'] > 0].copy()
    fills_df['FillNoFill'] = 1
    fills_df['UnitWeight'] = fills_df['TotalExecutedAfter']
    
    
    cancels_df = snapshot_state_df[snapshot_state_df['TotalCanceledAfter'] > 0].copy()
    fills_df['FillNoFill'] = 0
    fills_df['UnitWeight'] = fills_df['TotalCanceledAfter']
    
    Clean_Regression_Data = pd.concat([fills_df , cancels_df], ignore_index= True)
    
    #Has something partial happened to this order (1) or not (0)
    Clean_Regression_Data['OriginalOrNotOrder'] = np.where(Clean_Regression_Data['Type'].isin([67,69]) , 1, 0)
    Clean_Regression_Data['VolOfLastEvent'] = Clean_Regression_Data['Vol']
    
    #Cols that either dont have necessary info or to prevent data leaking i.e we cant train on totalexecuted after since that happnes in the future
    Cols_to_drop = ['ID', 'Type', 'Vol', 'TotalExecutedAfter', 'TotalCanceledAfter', 'ExecutedVol', 'CanceledVol']
    
    Clean_Regression_Data = Clean_Regression_Data.drop(columns = Cols_to_drop)
    
    
    return Clean_Regression_Data

#Adding the dataframes to variables for further use
rawdata = import_data(file_path, file_path_MO)
cleandata = clean_data(rawdata)
regressormatrix = data_regressors(rawdata, cleandata)

print(time_in_hours(57600062))
print(rawdata["Event"]["Type"].value_counts())

print(cleandata['Event'][cleandata['Event']['Type'] == 69])
print(order_life(32402869, cleandata))
print(regressormatrix[regressormatrix['MOTrailingVol100ms'] != 0].head(5))

print(f'Amount of final bulkorder cross section is \n {rawdata["Event"][rawdata["Event"]["Type"] == 88 ]}')
print(f' Amount of total volume of events in day is s{rawdata["Event"]["Vol"].sum()}')



#
#print(rawdata['BuyPrice']rawdata['BuyPrice']['TOD'] = 40000000])
    
# Revise this test logic its not quite correct yet
# df_E = pd.DataFrame({
#     'TOD': [36000055, 36000059, 36000062],
#     'ID': [32402869, 32402869, 32402869],
#     'Type': [66, 69, 70],  # Place, Partial Fill, Full Fill
#     'Vol': [300, 200, 100]  # Volumes from your console
# })

# # Initialize the Regressors DataFrame with dummy market features
# # We vary 'VolAhead' to simulate the order book changing in real-time
# Regressors_df = pd.DataFrame()
# Regressors_df['TOD'] = df_E['TOD']
# Regressors_df['VolAhead'] = [5000, 1200, 0]  # Queue clears out as time passes
# Regressors_df['DistanceToTouch'] = [2, 0, 0]   # Order moves to the touch line

# print("--- STEP 1: INITIAL COMPILING GRID ---")
# print(pd.concat([df_E[['ID', 'Type', 'Vol']], Regressors_df[['VolAhead', 'DistanceToTouch']]], axis=1))
# print("\n" + "="*60 + "\n")

# # =========================================================================
# # 2. RUN THE CONTINUOUS TARGET LOGIC
# # =========================================================================
# Regressors_df['Type'] = df_E['Type']
# Regressors_df['ID'] = df_E['ID']
# Regressors_df['Vol'] = df_E['Vol']

# # Isolate exact event behaviors
# Regressors_df['ExecutedVol'] = np.where(Regressors_df['Type'].isin([69, 70]), Regressors_df['Vol'], 0)
# Regressors_df['CanceledVol'] = np.where(Regressors_df['Type'].isin([67, 68]), Regressors_df['Vol'], 0)

# # Execute the double-flip reverse cumsum calculation
# Regressors_df['TotalExecutedAfter'] = (Regressors_df.iloc[::-1].groupby('ID')['ExecutedVol'].cumsum().iloc[::-1] - Regressors_df['ExecutedVol'])
# Regressors_df['TotalCanceledAfter'] = (Regressors_df.iloc[::-1].groupby('ID')['CanceledVol'].cumsum().iloc[::-1] - Regressors_df['CanceledVol'])

# print("--- STEP 2: REVERSE CUMSUM RESULTS (Looking into the future) ---")
# print(Regressors_df[['Type', 'Vol', 'ExecutedVol', 'TotalExecutedAfter']])
# print("\n" + "="*60 + "\n")

# # =========================================================================
# # 3. STATE-SPACE FILTERING & SNAPSHOT EXTRACTION
# # =========================================================================
# # Isolate rows where order states are born or transformed
# state_snapshots_df = Regressors_df[Regressors_df['Type'].isin([66, 83, 69])].copy()

# print("--- STEP 3: STATE SNAPSHOTS RETAINED (Type 70 is dropped) ---")
# print(state_snapshots_df[['TOD', 'Type', 'VolAhead', 'TotalExecutedAfter']])
# print("\n" + "="*60 + "\n")

# # =========================================================================
# # 4. TARGET GENERATION AND MATRIX PURGE
# # =========================================================================
# # Generate the Success Rows
# fills_df = state_snapshots_df[state_snapshots_df['TotalExecutedAfter'] > 0].copy()
# fills_df['Fill_NoFill'] = 1
# fills_df['Unit_Weight'] = fills_df['TotalExecutedAfter']

# # Generate the Failure Rows
# cancels_df = state_snapshots_df[state_snapshots_df['TotalCanceledAfter'] > 0].copy()
# cancels_df['Fill_NoFill'] = 0
# cancels_df['Unit_Weight'] = cancels_df['TotalCanceledAfter']

# # Combine into final training array
# Clean_Regression_Data = pd.concat([fills_df, cancels_df], ignore_index=True)

# # Feature extraction
# Clean_Regression_Data['Is_Partial_Fill_Now'] = np.where(Clean_Regression_Data['Type'] == 69, 1, 0)
# Clean_Regression_Data['Current_Event_Vol'] = Clean_Regression_Data['Vol']

# # Drop intermediate infrastructure tracking keys
# cols_to_drop = ['ID', 'Type', 'Vol', 'ExecutedVol', 'CanceledVol', 'TotalExecutedAfter', 'TotalCanceledAfter']
# Clean_Regression_Data = Clean_Regression_Data.drop(columns=cols_to_drop)

# print("--- STEP 4: FINAL CLEAN MACHINE LEARNING MATRIX ---")
# print(Clean_Regression_Data)


### check if 88/84 have ID, can i find order with similar ID to the 84 earlier in day 
#So the answer is no we cant find similar ID earlier in day, but they do both have an ID
#88 has an ID and you can immediately see that the VOL of those is massive
print(rawdata['Event'][rawdata['Event']['Type'] == 88].head())
print(rawdata['Event'][rawdata['Event']['Type'] == 84].head())
# Theres 84 if and only if ID is 0
print(rawdata['Event'][(rawdata['Event']['ID'] != 0) & (rawdata['Event']['Type'] == 84)])
print(rawdata['Event'][(rawdata['Event']['Type'] != 84) & (rawdata['Event']['ID'] == 0)])


print(cleandata["Event"].head(10))

#Bulk orders are probably opening and closing auctions
print(time_in_hours(15570644 ))
#####################Proving data only NOT ONLY has DAY type limit orders i.e. a LO placed during day will doesnt exist anymore after 4pm########################3
opening_time = 34200000
closing_time = 57600000
    
order_lifespan = rawdata['Event'].groupby('ID')['TOD'].agg(['min', 'max'])  #Creates pd df with buckets grouped by ID then only looks at TOD and then for each bucket calculates min and max and outputs that in pds df
mask6 = (order_lifespan['min'] < closing_time) & (order_lifespan['max'] > closing_time)
extract_orders = order_lifespan[mask6].index.values

print(extract_orders)
print(len(extract_orders)) #However only like 8 were put in much later then a few seconds past 4 where the cancelations are probably just do to latency 
print(order_life(147566 , rawdata))     #This was placed in pre market and canceled at end of day 

############################################################################


#############3########## Replicating the slides #############################
print(f" Total number of events on 1 April 2014 of INTC is {len(cleandata['Event'])}")
print(f" Amount of Market Orders on 1 April 2014 of INTC is {len(cleandata['MO'])}")
print(f' Percentage of MO per total number of events on 1 April 2014 INTC is {(len(cleandata["MO"])/len(cleandata["Event"])*100):.04f}%')

buy_no_walk = (cleandata['MO']['APPS'] == cleandata['MO']['BAP'] ) & (cleandata['MO']['BorS'] == -1)
sell_no_walk = (cleandata['MO']['APPS'] == cleandata['MO']['BBP'] ) & (cleandata['MO']['BorS'] == 1)
total_walk = buy_no_walk | sell_no_walk # | is OR operator

print(f" Percentage of orders that did not walk the book for INTC on April 1 2024 is {(total_walk.sum()/(len(cleandata['MO'])) * 100):.2f} % ")
#try to recreate the graph at 11 am
plots(rawdata, 57603600)
print(time_in_hours(33300000))
time_to_hours(16.001)
print(cleandata['Event'])
print(9.25*3600000)

################################################################################3

print(rawdata['Event'][rawdata['Event']['TOD'] == 0])
################Proving Unknown 2 is what size of the book an event happens###########
print(time_in_hours( 72000039))
#84 midprice half int, rounding might be wrong. Answer: no as the hidden orders are executed and that can be done at halfprice if agreed to. it is however true that regular LOs can only be placed in sizes of one cent at the smallest  

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
plot_feature(regressormatrix, "LOTrailingVolExecuted100ms")

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

lgbm_mdl_features = log_mdl_features + ['BASpread', 'QImbalance', 'TotalVolImbalance', 'Midprice', 'Microprice', 
                                        'MOTrailingVol100ms', 'MOTrailingOrders100ms', 'LOTrailingVolPlaced100ms', 'LOTrailingCountOrdersPlaced100ms', 
                                        'LOTrailingVolCanceled100ms', 'LOTrailingCountOrdersCanceled100ms', 'LOTrailingVolExecuted100ms',
                                        'LOTrailingCountOrdersExecuted100ms', 'VolAhead']

X_train = regressormatrix[log_mdl_features]

X_train_lgbm = regressormatrix[lgbm_mdl_features]

X_train_standardised = scalar.fit_transform(X_train) #Here we fit and transform
#Fit Scikit logistic regrssion



calibrated_model.fit(X_train_standardised, y_train)



base_lr_model.fit(X_train_standardised, y_train)

y_true = regressormatrix2["Fill_NoFill"]

X_test = regressormatrix2[log_mdl_features]
X_test_lgbm = regressormatrix2[lgbm_mdl_features]

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

#However i think AUC is only reliable on balanced data which this totally isnt, so the AUC is artificially inflated, why because we rarely ever have a fill and AUC is area under ROC, ROC formula is 

aucscore = roc_auc_score(y_true, y_pred_prob)
print(f'AUC score is {aucscore:.3f}')


avgprecision = average_precision_score(y_true, y_pred_prob)
print(f'Avg precision score is {avgprecision:.3f}')

#Visualisiton of performance and comparison to baseline dummy model which just guesses a baseline percentage on each order for it being filled 
#Dummy y fill prob is just an array of length y true with all entries equal to dummy fill prob

dummy_fill_prob = regressormatrix['Fill_NoFill'].mean()
dummy_y_pred_prob = np.full(len(y_true), dummy_fill_prob) #just creates an array of length y true with dummy fill probs


print("Dummy metrics")

dummy_brierscore = brier_score_loss(y_true, dummy_y_pred_prob)
print(f'Dummy Brier score is {dummy_brierscore:.3f}')

dummy_logloss = log_loss(y_true, dummy_y_pred_prob)
print(f' Dummy Logloss score is {dummy_logloss:.3f}')

dummy_aucscore = roc_auc_score(y_true, dummy_y_pred_prob)
print(f'Dummy AUC score is {dummy_aucscore:.3f}')

avgprecision_dummy = average_precision_score(y_true, dummy_y_pred_prob)
print(f'Avg precision score is {avgprecision_dummy:.3f}')

#Visualisation of performance vs dummy
fig, axes = plt.subplots(1,3, figsize = (24,8))
#calibration curve
engine_true, engine_prob_pred = calibration_curve(y_true, y_pred_prob, n_bins=10, strategy = 'quantile') #tuple unpacking since the function returns two variables, we just name them immediately in one line
axes[0].plot([0,1], [0,1], color = 'grey', label = "Perfect Calibration") # axes[0] means we're talking about the left figure then [0,1] , [0,1] are x list and y list and are read vertically so the first point is 0,0 and the second point is 1,1 and a line is drawn between them i.e the perfect prediction line i think but check this
axes[0].plot(engine_prob_pred, engine_true, color = 'b' ,label = 'Logistic Regression Engine')
axes[0].set_title('Calibration curve')
axes[0].legend()

#Roc curve

engine_fpr, engine_tpr, tresholds = roc_curve(y_true, y_pred_prob) #returns false postive rates and true positive rates, treshold which i think is the number or prob above or below it gives a certain classification
dummy_fpr, dummy_tpr, tresholds = roc_curve(y_true, dummy_y_pred_prob)
axes[1].plot(engine_fpr, engine_tpr, color = 'b', label = 'Logistic Regression Engine')
axes[1].plot(dummy_fpr, dummy_tpr, color = 'r', label = 'Dummy')
axes[1].legend()

#Precision recall curve
engine_precision, engine_recall, engine_treshold = precision_recall_curve(y_true, y_pred_prob)
axes[2].plot(engine_recall, engine_precision, color = 'b', label = 'Engine PR')
axes[2].set_xlabel('Recall')
axes[2].set_ylabel('Precision')
axes[2].plot([0,1], [dummy_fill_prob, dummy_fill_prob], color = 'red', label = 'Dummy') #the dummy PR is just the baseline fill rate i.e here just a horizontal line

plt.show()


def predict_order_fill_prob(features):
    #predicts specific probability for a given limit order being filled using the logistic regression engine from above
    #since pd dfs are slow its better to use np array here
    
    input_array = np.array(features).reshape(1,-1) #resshape needed 1 means passing 1 row, -1 means calculate the right dim for the columns so this creates a matrix which is whats needed for sci kit later
    scaled_input  = (input_array - scalar.mean_) / scalar.scale_ # I think the trailing _ tells sci kit to look at the fitted values and calculate mean and std of those                   using scalar so we do the standardizations for each value and not all at the same time
    fill_prob = calibrated_model.predict_proba(scaled_input)[0,1]
    
    return fill_prob

example_state = X_test.iloc[67].values # Expects numerical list with values corresponding to the entries above
print(predict_order_fill_prob(example_state))
print(X_test.iloc[67])
#Better to also create a function that extracts these features for maybe a given order ID?


# light GBM 

#imbalance ratio since our outcome variable is heavily skewed

imbalance_ratio = (1-dummy_fill_prob) / dummy_fill_prob

base_lgb = lgb.LGBMClassifier(
    n_jobs = -1, #Using all available threads in cpu 
    n_estimators = 150, # number of sequential trees  
    learning_rate = 0.05, # scales contribution of each individual tree
    num_leaves = 31, #max num of leaves, i.e terminal nodes, allowed in each tree 
    random_state = 69 , #just set random seed for reproducability
    scale_pos_weight = imbalance_ratio #telling the loss function about the skewed output var i think, not sure yet
    ) 

#calibrating the model, trees use stepfunctions, so use isotonic to match that, do more research on this

calibrated_lgbm = CalibratedClassifierCV(
    estimator = base_lgb,
    method = 'isotonic',
    cv = 5
    )

print('Training lgb') # im pretty sure trees dont require scalar and only care about relative ordering 
calibrated_lgbm.fit(X_train_lgbm, y_train)

y_pred_prob_lgbm = calibrated_lgbm.predict_proba(X_test_lgbm)[:, 1]

#Evaluate performance metrics 
print("Light GBM Engine metrics")

brierscore_lgbm = brier_score_loss(y_true, y_pred_prob_lgbm)
print(f'Brier score is {brierscore_lgbm:.3f}')

logloss_lgbm = log_loss(y_true, y_pred_prob_lgbm)
print(f'Logloss score is {logloss_lgbm:.3f}')

#However i think AUC is only reliable on balanced data which this totally isnt, so the AUC is artificially inflated, why because we rarely ever have a fill and AUC is area under ROC, ROC formula is 

aucscore_lgbm = roc_auc_score(y_true, y_pred_prob_lgbm)
print(f'AUC score is {aucscore_lgbm:.3f}')

avgprecision_lgbm = average_precision_score(y_true, y_pred_prob_lgbm)
print(f'Avg precision score is {avgprecision_lgbm:.3f}')































