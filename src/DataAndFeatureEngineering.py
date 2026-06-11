#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 10:45:38 2026

@author: jesseruijer
"""

#Importing libraries,classes, functions from other scripts

import scipy.io
import pandas as pd
import numpy as np

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
    
    canceled_vol_day = rawdata['Event'][rawdata['Event']['Type'].isin([67,68])]['Vol'].sum()
    added_vol_day = rawdata['Event'][rawdata['Event']['Type'].isin([66,83])]['Vol'].sum()
    Regressors_df['CancelationRatio'] = canceled_vol_day / added_vol_day
    
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