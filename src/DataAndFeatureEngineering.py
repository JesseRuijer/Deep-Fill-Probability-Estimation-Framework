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
import config
import gc
from Functions import order_life, find_order_pattern, time_in_hours, speedmetric, trailing_calc, calculate_rolling_moments
from FileManager import get_data_paths


#Just a display feature in console so all columns are printed in console
pd.set_option('display.max_columns', None)

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
    
    # The LOB Arrays are the heaviest things in RAM. Force them to 32-bit floats.
    df_BV = df_BV.astype('float32')
    df_SV = df_SV.astype('float32')
    df_BP = df_BP.astype('float32')
    df_SP = df_SP.astype('float32')
    
    # Force Event and MO to 32-bit  or less where applicable
    df_Event['TOD'] = df_Event['TOD'].astype('int32')
    df_Event['ID'] = df_Event['ID'].astype('int32')
    df_Event['Vol'] = df_Event['Vol'].astype('float32')
    df_Event['Price'] = df_Event['Price'].astype('float32')
    df_Event['Type'] = df_Event['Type'].astype('int8')
    df_Event['SideOfBook'] = df_Event['SideOfBook'].astype('int8')
    
    df_MO['TOD'] = df_MO['TOD'].astype('int32')
    df_MO['BBP'] = df_MO['BBP'].astype('float32')
    df_MO['BAP'] = df_MO['BAP'].astype('float32')
    df_MO['BBV'] = df_MO['BBV'].astype('float32')
    df_MO['BAV'] = df_MO['BAV'].astype('float32')
    df_MO['APPS'] = df_MO['APPS'].astype('float32')
    
    df_MO['Vol'] = df_MO['Vol'].astype('float32')
    df_MO['BorS'] = df_MO['BorS'].astype('int8')
    df_MO['MME'] = df_MO['MME'].astype('float32')
    
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
    
    #Cleans imported data so its ready for feature extraction
    
    
    df_E = raw_data["Event"]
    df_BV = raw_data["BuyVol"]
    df_SV = raw_data["SellVol"]
    df_BP = raw_data["BuyPrice"]
    df_SP = raw_data["SellPrice"]
    df_MO = raw_data["MO"]
    
    df_dictionary = {
        'Event': df_E,
        'BV': df_BV,
        'SV': df_SV,
        'BP': df_BP,
        'SP': df_SP,
        'MO': df_MO,
        }
    
    #Safety checks
    
    for name, df in df_dictionary.items():
        assert not df.isna().any().any(), f'NaN detected in {df}'  #Checks this and if condition true continue if  false then immediately stops and prints, the double .any() is just to see if anywhere in the whole df the condition is there 
    
    assert (df_BV.values >= 0).all() , 'Negative Vol detected'  
    assert (df_SV.values >= 0).all(), 'Negative Vol detected'
    assert (df_E["Vol"].values >=0).all(), 'Negative Vol detected'
    assert (df_MO["Vol"].values >=0).all(), 'Negative Vol detected'
    
   #Cleans dataframes to not include outside trading hours and not include 88 and 84
    valid_row_mask = (
        (df_E["TOD"] >= config.MARKET_OPEN_TIME) &
        (df_E["TOD"] <= (config.MARKET_CLOSE_TIME_INCLUDING_CANC_SPAM)) & #Let the closing time be 4:01 PM to account for the closing cancelations spam at eod
        (df_E["Type"] != 88) &
        (df_E["Type"] != 84)
        )
    
    df_E_without_noise = df_E[valid_row_mask].copy()
    df_BV_without_noise = df_BV[valid_row_mask].copy()
    df_SV_without_noise = df_SV[valid_row_mask].copy()
    df_BP_without_noise = df_BP[valid_row_mask].copy()
    df_SP_without_noise = df_SP[valid_row_mask].copy()

    valid_row_mask_MO = (
        (df_MO["TOD"] >= config.MARKET_OPEN_TIME) &
        (df_MO["TOD"] <= config.MARKET_CLOSE_TIME)      
        )

    df_MO_without_noise = df_MO[valid_row_mask_MO].copy()
    
    data_set_clean = {
        "Event" : df_E_without_noise,
        "BuyVol" : df_BV_without_noise,
        "SellVol" : df_SV_without_noise,
        "BuyPrice" : df_BP_without_noise,
        "SellPrice" : df_SP_without_noise,
        "MO" : df_MO_without_noise
        }
    
    return data_set_clean
    
def data_regressors(rawdata, cleandata, clear_RAM = True, dont_include_full_trading_day = True):
    
    #Builds the regression matrices used by the ML models
    
    #Importing Dataframes
    df_E = cleandata["Event"]
    df_MO = cleandata['MO']
    
    #For our machine to make accurate predictions we have to shift forward each row in the LOB BuySell Vol and Price
    #i.e for row 1 in event normally, row 1 in the LOB data would correspond to what happened immediately after 
    #the event in row 1, but to predict what happened to the event in row1 we need what the LOB looked like before that 
    #event happened so we need to shift row 0 from LOB down to row 1, can do this vectorize wise by .shift
    #So first we do that for the raw data so its all alligned, and then we can go to removing cols and stuff in cleandata
    
    df_BV = rawdata["BuyVol"].shift(1).loc[df_E.index]
    df_SV = rawdata["SellVol"].shift(1).loc[df_E.index]
    df_BP = rawdata["BuyPrice"].shift(1).loc[df_E.index]
    df_SP = rawdata["SellPrice"].shift(1).loc[df_E.index]
    
    #Some small calculations for which I still needed rawdata and cleandata
    canceled_vol_day = rawdata['Event'][rawdata['Event']['Type'].isin([67,68])]['Vol'].sum()
    added_vol_day = rawdata['Event'][rawdata['Event']['Type'].isin([66,83])]['Vol'].sum()
    
    #Removing all dictionary references after not needing them anymore
    if clear_RAM:
        rawdata.clear()
        cleandata.clear()
        gc.collect()
        
    #Creating masks
    mask_add = df_E['Type'].isin([66,83])
    mask_cancel = df_E['Type'].isin([67,68])
    mask_execute = df_E['Type'].isin([69,70])
    
    
    #Initializing main feature dataframe
    Regressors_df = pd.DataFrame()
    

    ########## Basic Features #######################
        
    Regressors_df['Type'] = df_E['Type']
    Regressors_df['ID'] = df_E['ID']
    Regressors_df['Vol'] = df_E['Vol']
    Regressors_df['Price'] = df_E['Price']
    Regressors_df["TOD"] = df_E["TOD"]
    Regressors_df['SideOfBook' ] = df_E["SideOfBook"]
    Regressors_df['BestBid'] = df_BP[0]
    Regressors_df['BestAsk'] = df_SP[0]
    Regressors_df['BidSize'] = df_BV[0]
    Regressors_df['AskSize'] = df_SV[0]
    Regressors_df['TotalQueueSize'] = Regressors_df['BidSize'] + Regressors_df['AskSize']
    Regressors_df["BASpread"] = Regressors_df['BestAsk'] - Regressors_df['BestBid']
    Regressors_df["Midprice"] = (Regressors_df['BestBid'] + Regressors_df['BestAsk'])/2
    Regressors_df["Microprice"] = ((Regressors_df['BidSize']*Regressors_df['BestAsk'])+(Regressors_df['AskSize']*Regressors_df['BestBid']))/(Regressors_df['BidSize'] + Regressors_df['AskSize'])
    Regressors_df['CancelationRatio'] = canceled_vol_day / added_vol_day
    Regressors_df['QImbalance'] = (Regressors_df['BidSize'] - Regressors_df['AskSize'] ) / (Regressors_df['BidSize'] + Regressors_df['AskSize'])
    Regressors_df["AbsQImbalance"] = Regressors_df["QImbalance"].abs()
    Regressors_df["TotalVolImbalance"] = ((df_BV.sum(axis=1)-df_SV.sum(axis=1))/(df_BV.sum(axis=1)+ df_SV.sum(axis=1))).fillna(0)   #Total Vol imbalance uses sum of the 20 cols provided in the data axis=1 does across cols, axis=0 does across rows
  
    weights = [1/(i) for i in range(1,21)]
    Regressors_df["WeightedVolImbalance"] = (((weights*df_BV).sum(axis=1)-(weights*df_SV).sum(axis=1))/((weights*df_BV).sum(axis=1)+ (weights*df_SV).sum(axis=1))).fillna(0)  
     
    #Building a regime classifier which uses categorical variables to tell in what regime of day we are in
    Regressors_df['Regime'] = np.where(Regressors_df['TOD'] < config.MARKET_OPEN_TIME , 0,    #Pre Market                         
                              np.where(Regressors_df['TOD'] < config.SOMARKET_NOISE , 1,    # 30 min vol after opening
                              np.where(Regressors_df['TOD'] < config.EOMARKET_NOISE , 2,    #Regular Market hours without first and last 30 min
                              np.where(Regressors_df['TOD'] < config.MARKET_CLOSE_TIME , 3,    #30 min high volatilitiy time before closing
                              4))))                                            #After market hours
    
    Regressors_df['TimeTillMarketClose'] = config.MARKET_CLOSE_TIME - Regressors_df['TOD']
    Regressors_df['IsFinalMinute'] = np.where(Regressors_df['TimeTillMarketClose'] <= 60000, 1, 0) #Just a handhold just for logistic regression to implement that cancelations at eod are not as valuable as cancelations during day  
    
    df_MO['SweepNoSweep'] = np.where(((df_MO['BorS'] == -1) & (df_MO['APPS'] > df_MO['BAP'])) | ((df_MO['BorS'] == 1) & (df_MO['APPS'] < df_MO['BBP'])) , 1, 0 ).astype('int8')

    
    ############ Trailing and Regime Features ##################
   #Some Event, Speed lookback features that dont require ID like further below

    market_past = Regressors_df[['TOD', 'Midprice', 'BestBid', 'BestAsk', 'BidSize', 'AskSize', 'Microprice', 
                                 #'MicroMidDeviation'
                                 ]].copy()
    market_past['TOD_+_1000'] = (market_past['TOD'] + config.FEATURE_DELTA).astype('int32')
    
    mid_arr = Regressors_df['Midprice'].values
    mid_past = Regressors_df['Midprice'].shift(config.EVENT_TIME_DELTA).values
    bestbid = Regressors_df['BestBid'].values
    bestbid_past = Regressors_df['BestBid'].shift(config.EVENT_TIME_DELTA).values
    
    bestask = Regressors_df['BestAsk'].values
    bestask_past = Regressors_df['BestAsk'].shift(config.EVENT_TIME_DELTA).values
    
    bidsize = Regressors_df['BidSize'].values
    bidsize_past = Regressors_df['BidSize'].shift(config.EVENT_TIME_DELTA).values
    
    asksize = Regressors_df['AskSize'].values
    asksize_past = Regressors_df['AskSize'].shift(config.EVENT_TIME_DELTA).values
    
    micro_arr = Regressors_df['Microprice'].values
    micro_past = Regressors_df['Microprice'].shift(config.EVENT_TIME_DELTA).values
   
    Event_delta_midprice = mid_arr - mid_past
    Regressors_df['EventDeltaMidprice'] = np.nan_to_num(Event_delta_midprice, nan=0.0, posinf=0.0, neginf=0.0)

   
    global_micro_delta = micro_arr - micro_past

    Regressors_df['EventDeltaMicroprice'] = np.nan_to_num(global_micro_delta, nan=0.0, posinf=0.0, neginf=0.0)

    distancetobid = bestbid - bestbid_past 
    distancetoask = bestask - bestask_past 
    Regressors_df['EventDeltaBestBid'] = np.nan_to_num(distancetobid, nan=0.0, posinf=0.0, neginf=0.0)
    Regressors_df['EventDeltaBestAsk'] = np.nan_to_num(distancetoask, nan=0.0, posinf=0.0, neginf=0.0)

    #OFI at best bid ask 
    # (Bid side demand change)
    deltadb = np.where(bestbid > bestbid_past, bidsize,
          np.where(bestbid == bestbid_past, bidsize - bidsize_past,
          - bidsize_past))

    #(Ask side supply change)
    deltada = np.where(bestask < bestask_past, asksize,
          np.where(bestask == bestask_past, asksize - asksize_past,
          - asksize_past))

    OrderFlowImbalance = deltadb - deltada
    Regressors_df['OrderFlowImbalance'] = np.nan_to_num(OrderFlowImbalance, nan=0.0, posinf=0.0, neginf=0.0)
    OrderFlowImbalance_past = Regressors_df['OrderFlowImbalance'].shift(config.EVENT_TIME_DELTA).values
    EventDeltaOrderFlowImbalance = OrderFlowImbalance - OrderFlowImbalance_past
    Regressors_df['EventDeltaOrderFlowImbalance'] = np.nan_to_num(EventDeltaOrderFlowImbalance, nan=0.0, posinf=0.0, neginf=0.0)

    
    #Immediately clear temporary arrays to save RAM
    del market_past, micro_arr, micro_past, global_micro_delta, mid_past, Event_delta_midprice, bestbid_past, bestask_past, bidsize_past, asksize_past, OrderFlowImbalance_past, OrderFlowImbalance
    gc.collect() 

    
    #Features for which during the trail i want to calculate some moments and extremes to summarize the behavior  over past 50 ticks
    features_for_std = ['Microprice', 'OrderFlowImbalance', 'BASpread', 'QImbalance', 'WeightedVolImbalance', 
                        ]
    features_for_skew = ['OrderFlowImbalance']
    features_for_extremes = ['OrderFlowImbalance', 'BASpread', 'Microprice']
    
    #Run the function for Standard Deviation
    std_dict = calculate_rolling_moments(
        Regressors_df, features_for_std, window=config.EVENT_TIME_DELTA, calc_std=True
    )
    Regressors_df = Regressors_df.assign(**std_dict)
    
    #Run the function for Skewness
    skew_dict = calculate_rolling_moments(
        Regressors_df, features_for_skew, window=config.EVENT_TIME_DELTA, calc_skew=True
    )
    Regressors_df = Regressors_df.assign(**skew_dict)
    
    #Run the function for Extremes
    extremes_dict = calculate_rolling_moments(
        Regressors_df, features_for_extremes, window=config.EVENT_TIME_DELTA, calc_extremes = True
    )
    Regressors_df = Regressors_df.assign(**extremes_dict)
    
    #Clean up temporary dicts
    del std_dict, skew_dict, extremes_dict 
    
    # #HiddenVol
    # hidden_vol = np.where(df_E["Type"] ==  84, df_E['Vol'], 0) #returns the vol of types 84 else zero in a new numpy array
    # cum_vol_pad = np.pad(np.cumsum(hidden_vol), (1,0), constant_values = 0) #says add a zero to the start, nothing to the back and then we take cumsum of all the vols
    # tod_values = df_E['TOD'].values
    # lookback = 5000 #Lookback time in MS for hidden vol trades
    # lookback_times = tod_values - lookback
    # start_indices = np.searchsorted(tod_values, lookback_times, side='left')    #Does binary search st for every lookback time we calculate the row index it woud land on in tod_values
    # current_indices = np.arange(1, len(tod_values) + 1)
    # Regressors_df["LookBackHiddenVol"] = cum_vol_pad[current_indices] - cum_vol_pad[start_indices]
    
    mo_buy = df_MO['BorS'] == -1
    mo_sell = df_MO['BorS'] == 1
    

    
    
    Regressors_df['MOTrailingVolBuy'] = trailing_calc(df_E['TOD'].values, df_MO.loc[mo_buy , 'TOD'].values, df_MO.loc[mo_buy , 'Vol'].values, config.LOOKBACK_WINDOW)[0]
    Regressors_df['MOTrailingVolSell'] = trailing_calc(df_E['TOD'].values, df_MO.loc[mo_sell , 'TOD'].values, df_MO.loc[mo_sell , 'Vol'].values, config.LOOKBACK_WINDOW)[0]
    Regressors_df['MOTrailingVolRatio'] = ((Regressors_df['MOTrailingVolSell'] - Regressors_df['MOTrailingVolBuy']) / (Regressors_df['MOTrailingVolBuy'] + Regressors_df['MOTrailingVolSell'])).fillna(0)
    
   #Qimbalance ratio sortof but then for trailing LOs for Placed and canceled
    Regressors_df['LOTrailingVolPlaced'] = trailing_calc(df_E['TOD'].values, df_E.loc[mask_add, 'TOD'].values,  df_E.loc[mask_add, 'Vol'].values, config.LOOKBACK_WINDOW)[0]
    Regressors_df['LOTrailingVolCanceled'] = trailing_calc(df_E['TOD'].values, df_E.loc[mask_cancel, 'TOD'].values,  df_E.loc[mask_cancel, 'Vol'].values, config.LOOKBACK_WINDOW)[0]
    Regressors_df['LOTrailingVolExecuted'] =  trailing_calc(df_E['TOD'].values, df_E.loc[mask_execute, 'TOD'].values,  df_E.loc[mask_execute, 'Vol'].values, config.LOOKBACK_WINDOW)[0]
    
    Regressors_df['LOTrailingPlaceCancelRatio'] = ((Regressors_df['LOTrailingVolPlaced'] - Regressors_df['LOTrailingVolCanceled']) / (Regressors_df['LOTrailingVolPlaced'] + Regressors_df['LOTrailingVolCanceled'])).fillna(0)
    Regressors_df['LOTrailingPlaceExecuteRatio'] = ((Regressors_df['LOTrailingVolPlaced'] - Regressors_df['LOTrailingVolExecuted']) / (Regressors_df['LOTrailingVolPlaced'] + Regressors_df['LOTrailingVolExecuted'])).fillna(0)



########## Dynamic Features #####################
    
    #Distance to touch. How far a placed LO is from best bid or best ask    
    Regressors_df["DistanceToTouch"] =  np.where(   
        Regressors_df["SideOfBook"].values == 1, # 1 is buy side, 0 is sell side
        Regressors_df['BestBid'].values - Regressors_df['Price'].values,
        Regressors_df['Price'].values - Regressors_df['BestAsk'].values
        ) 
    
    Regressors_df['DistanceToMicroprice'] = np.where(Regressors_df['SideOfBook'] == 1, Regressors_df['Microprice'] - Regressors_df['Price'] , Regressors_df['Price'] - Regressors_df['Microprice'] )
    
    #Vol Ahead looks at for a given placed limit order how much volume is ahead of it until best price
    #We create an empty array for all tods an event was placed, loop through all 20 levels of the order book prices we have and add volume to it if its in front of our order
    total_buy_vol_ahead = np.zeros(len(df_E))
    total_sell_vol_ahead = np.zeros(len(df_E))
    total_buy_vol_at_price = np.zeros(len(df_E))
    total_sell_vol_at_price = np.zeros(len(df_E))
    
    del df_E
    gc.collect()
    
    
    for i in range(20):
        lvl_buy_price = df_BP[i].values
        lvl_buy_vol = df_BV[i].values
        lvl_sell_price = df_SP[i].values
        lvl_sell_vol = df_SV[i].values
        
        total_buy_vol_ahead += np.where( lvl_buy_price >= Regressors_df['Price'], lvl_buy_vol,0)
        total_sell_vol_ahead += np.where(lvl_sell_price <= Regressors_df['Price'], lvl_sell_vol, 0)
        total_buy_vol_at_price += np.where( lvl_buy_price == Regressors_df['Price'], lvl_buy_vol,0)
        total_sell_vol_at_price += np.where(lvl_sell_price == Regressors_df['Price'], lvl_sell_vol, 0)
        
        
    Vol_Ahead = np.where(
        Regressors_df["SideOfBook"].values == 1,
        total_buy_vol_ahead,
        total_sell_vol_ahead
        )
    
    Vol_At_Price = np.where(
        Regressors_df["SideOfBook"].values == 1,
        total_buy_vol_at_price,
        total_sell_vol_at_price
        )
    
    Regressors_df['VolAhead'] = Vol_Ahead
    Regressors_df['LogVolAhead'] = np.log1p(Vol_Ahead)     #we use log1p  which is log 1 + x since if an order is placed inside the spread it would have negative values and we cant take log of that 
    
    Regressors_df['QueuePositionRatio'] = np.where(
        Vol_At_Price == 0,
        -1, #set this to -1 to give those values to orders who have 'fallen of the LOB', i.e the price levels have moved away from the 20 levels and we dont want to then give it a zero since then it would be the same signal for an order thats 30 levels in the book vs an order thats at the front of the queue at best price which is obviously terrible 
        Regressors_df['VolAhead'] / np.where(Vol_At_Price == 0, 1, Vol_At_Price)
        )
    
    
   ######### Target Generation, i.e targetting fills cancels etc #########################
    
    Regressors_df['InitialPlacementTime'] = Regressors_df.groupby('ID')['TOD'].transform('min')  #Time in ms since order was placed
    Regressors_df['TimeSincePlacement'] = Regressors_df['TOD'] - Regressors_df['InitialPlacementTime']
  
    Regressors_df['ExecutedVol'] = np.where(mask_execute.values, Regressors_df['Vol'].values, 0)
    Regressors_df['ActiveCanceledVol'] = np.where(((mask_cancel.values) & (Regressors_df['TOD'].values < config.MARKET_CLOSE_TIME)), Regressors_df['Vol'].values, 0)
    Regressors_df['ExpiredVol'] = np.where(((mask_cancel.values) & (Regressors_df['TOD'].values >= config.MARKET_CLOSE_TIME)), Regressors_df['Vol'].values, 0)
    
    #Saving total vol of specific order types for later
    Regressors_df['TotalOrderExecutedVol'] = Regressors_df.groupby('ID')['ExecutedVol'].transform('sum')
    Regressors_df['TotalOrderCanceledVol'] = Regressors_df.groupby('ID')['ActiveCanceledVol'].transform('sum')
    Regressors_df['TotalOrderExpiredVol'] = Regressors_df.groupby('ID')['ExpiredVol'].transform('sum')
    Regressors_df['TotalOrderFailureVol'] = Regressors_df['TotalOrderCanceledVol'] + Regressors_df['TotalOrderExpiredVol']

    
    
    #Remaining Vol = (Total Order Sum) - (Running Sum)
    Regressors_df['TotalExecutedAfter'] = Regressors_df.groupby('ID')['ExecutedVol'].transform('sum') - Regressors_df.groupby('ID')['ExecutedVol'].cumsum()
    Regressors_df['TotalActiveCanceledAfter'] = Regressors_df.groupby('ID')['ActiveCanceledVol'].transform('sum') - Regressors_df.groupby('ID')['ActiveCanceledVol'].cumsum()
    Regressors_df['TotalExpiredAfter'] = Regressors_df.groupby('ID')['ExpiredVol'].transform('sum') - Regressors_df.groupby('ID')['ExpiredVol'].cumsum()
    Regressors_df['TotalFailureAfter'] = Regressors_df['TotalActiveCanceledAfter'] + Regressors_df['TotalExpiredAfter']
   
    
   #Filtering out full executions and full cancels as they dont predict anything anymore as the order is dead after that  
    state_snapshot_df = Regressors_df[Regressors_df['Type'].isin([66, 67, 69, 83])].copy() 
    
    # Feature extraction
    state_snapshot_df['Is_Initial_Placement'] = np.where(state_snapshot_df['Type'].isin([66, 83]), 1, 0)
    state_snapshot_df['Is_Partial_Fill'] = np.where(state_snapshot_df['Type'] == 69, 1, 0)
    state_snapshot_df['Is_Partial_Cancel'] = np.where(state_snapshot_df['Type'] == 67, 1, 0)
    #state_snapshot_df['Current_Event_Vol'] = state_snapshot_df['Vol']
    
    
    ########### Heartbeat Engine ##########################
    #Trying to create the 'Heartbeat' logic where for an orders life it takes a snapshot of LOB every 1 seconds or so or other custom time ofc
    
    hb_order_info = Regressors_df.groupby('ID').agg(     #agg just puts all this info into one row, syntax like Death( tod, max) means we give this new column name to the old col name tod where we perform the operation max to it
        InitialPlacementTime = ('TOD', 'min'),
        DeathTime = ('TOD', 'max'),
        Price = ('Price', 'first'),
        SideOfBook = ('SideOfBook' , 'first'),
        Vol = ('Vol', 'first')
        ).reset_index()     #To remove the native extra indexing pandas does
    
    heartbeat_interval = config.HEARTBEAT_INTERVAL
    
    hb_order_info['Duration'] = (hb_order_info['DeathTime'] - hb_order_info['InitialPlacementTime']) - 1
    hb_order_info['NumberOfHeartBeats'] = np.minimum(((hb_order_info['Duration'] // heartbeat_interval).astype(int)), (config.MAX_HEARTBEATS))
    
    valid_orders = hb_order_info[hb_order_info['NumberOfHeartBeats'] > 0]
    
    
    #Initializing vectorized grid
    
    ids_repeated = np.repeat(valid_orders['ID'].values , valid_orders['NumberOfHeartBeats'].values)
    starts_repeated = np.repeat(valid_orders['InitialPlacementTime'], valid_orders['NumberOfHeartBeats'])
    
    #Building base df
    
    heartbeats_df = pd.DataFrame({'ID' : ids_repeated, 'BaseTime' : starts_repeated})
    heartbeat_counts = valid_orders['NumberOfHeartBeats'].values
    
    #Doesnt print a heartbeat for initial placement and death time since thats already eincluded in event
    #So for example an order placed at t=0, death at t = 3 has snapshots at 1,2 note c+1 works, since above i subtracted 1 ms from the duration
    
    
    heartbeats_df['Step'] = np.concatenate([np.arange(1, c+1) for c in heartbeat_counts]) * config.HEARTBEAT_INTERVAL   # creates col with 1 * 10000 as first entry, 2*10000 as second entry etc
    heartbeats_df['TOD'] = heartbeats_df['BaseTime'] + heartbeats_df['Step']
    
    heartbeats_df = heartbeats_df.merge(hb_order_info[['ID' , 'Price', 'SideOfBook', 'InitialPlacementTime', 'Vol']], on='ID')
    
   #Look at last event if multiple events happened at same TOD, else programme will crash, ik this isnt optimal but theres not really any other way to sort them for the moment if they come in at the same tod i think
    # 1. Find exact row indices to keep by sorting ONLY the TOD column (Lightning fast)
    # We use reset_index() so we can grab the original row numbers after dropping duplicates
    
    
    #Manually order the list since we might have some skips from deleting 84s and 88s
    temp_tod = Regressors_df[['TOD']].copy()
    temp_tod['position'] = np.arange(len(temp_tod))
    
    keep_indices = temp_tod.sort_values('TOD').drop_duplicates('TOD', keep='last')['position'].values
    
    # 2. Extract Universal Features ONLY for those specific rows
    market_time_general_features = Regressors_df.iloc[keep_indices][config.UNIVERSAL_FEATURES].copy()
    
    del Regressors_df
    del temp_tod
    gc.collect()
    
    # 3. Attach the heavy LOB arrays ONLY for those rows
    for i in range(20):
        market_time_general_features[f'BP_{i}'] = df_BP[i].values[keep_indices]
        market_time_general_features[f'BV_{i}'] = df_BV[i].values[keep_indices]
        market_time_general_features[f'SP_{i}'] = df_SP[i].values[keep_indices]
        market_time_general_features[f'SV_{i}'] = df_SV[i].values[keep_indices]
    
    heartbeats_df = heartbeats_df.sort_values('TOD')
    
    del df_BP, df_BV, df_SP, df_SV
    gc.collect()
    
    #Manual fix to allow merge below
    
    heartbeats_df['TOD'] = heartbeats_df['TOD'].astype('int32')
    market_time_general_features['TOD'] = market_time_general_features['TOD'].astype('int32')
    
    #Merge heartbeats sorted by time with last known (direction = backward) market state before the heartbeat
    hb_order_info_with_market_general_features = pd.merge_asof(
        heartbeats_df,
        market_time_general_features,
        on = 'TOD',
        direction = 'backward'
        )
    
    #Now recalculate the dynamic features, i.e features that change per order ID we look at 
    
    hb_order_info_with_market_general_features['DistanceToTouch'] = np.where( #np.where works like an if condition, its this, else do this
          
          hb_order_info_with_market_general_features['SideOfBook'] == 1,
          hb_order_info_with_market_general_features['BestBid'] - hb_order_info_with_market_general_features['Price'],
          hb_order_info_with_market_general_features['Price'] - hb_order_info_with_market_general_features['BestAsk']
          
          )
    
    hb_order_info_with_market_general_features['DistanceToMicroprice'] = np.where( hb_order_info_with_market_general_features['SideOfBook'] == 1, 
                                                                                hb_order_info_with_market_general_features['Microprice'] -  hb_order_info_with_market_general_features['Price'] ,  
                                                                                hb_order_info_with_market_general_features['Price'] -  hb_order_info_with_market_general_features['Microprice'] )
    
    hb_order_info_with_market_general_features['TimeSincePlacement'] = hb_order_info_with_market_general_features['TOD'] - hb_order_info_with_market_general_features['InitialPlacementTime']
    
    total_buy_vol_ahead = np.zeros(len(hb_order_info_with_market_general_features))
    total_sell_vol_ahead = np.zeros(len(hb_order_info_with_market_general_features))
    total_buy_vol_at_price = np.zeros(len(hb_order_info_with_market_general_features))
    total_sell_vol_at_price = np.zeros(len(hb_order_info_with_market_general_features))
    
    
    price_of_order = hb_order_info_with_market_general_features['Price'].values
    
    for i in range(20):
        lvl_buy_price = hb_order_info_with_market_general_features[f'BP_{i}'].values
        lvl_buy_vol = hb_order_info_with_market_general_features[f'BV_{i}'].values
        lvl_sell_price = hb_order_info_with_market_general_features[f'SP_{i}'].values
        lvl_sell_vol = hb_order_info_with_market_general_features[f'SV_{i}'].values
        
        total_buy_vol_ahead += np.where( lvl_buy_price >= price_of_order, lvl_buy_vol,0)
        total_sell_vol_ahead += np.where(lvl_sell_price <= price_of_order, lvl_sell_vol, 0)
        total_buy_vol_at_price += np.where( lvl_buy_price == price_of_order, lvl_buy_vol,0)
        total_sell_vol_at_price += np.where(lvl_sell_price == price_of_order, lvl_sell_vol, 0)
        
        
    Vol_Ahead = np.where(
        hb_order_info_with_market_general_features['SideOfBook'] == 1,
        total_buy_vol_ahead,
        total_sell_vol_ahead
        )
    

    
    hb_order_info_with_market_general_features['VolAhead'] = Vol_Ahead
    
    hb_order_info_with_market_general_features['LogVolAhead'] = np.log1p(hb_order_info_with_market_general_features['VolAhead'])  
    
    Vol_At_Price = np.where(
        hb_order_info_with_market_general_features['SideOfBook'] == 1,
        total_buy_vol_at_price,
        total_sell_vol_at_price
        )
    
    hb_order_info_with_market_general_features['QueuePositionRatio'] = np.where(
        Vol_At_Price == 0,
        -1, #set this to -1 to give those values to orders who have 'fallen of the LOB', i.e the price levels have moved away from the 20 levels and we dont want to then give it a zero since then it would be the same signal for an order thats 30 levels in the book vs an order thats at the front of the queue at best price which is obviously terrible 
        hb_order_info_with_market_general_features['VolAhead'] / np.where(Vol_At_Price == 0, 1, Vol_At_Price)
        )
    
    #Remove the for loop info from ram as not needed anymore
    
    cols_to_drop = [col for col in hb_order_info_with_market_general_features.columns if col.startswith(('BP', 'BV', 'SP', 'SV'))]
    hb_order_info_with_market_general_features = hb_order_info_with_market_general_features.drop(columns = cols_to_drop)
    
    
   # Put a custom label on this to keep track of these heartbeat snapshots since they arent snapshots of when an actual event takes place  
    hb_order_info_with_market_general_features['Type'] = 26 
    
    #Format them in with the state_snapshot df later
    
    hb_order_info_with_market_general_features['Is_Initial_Placement'] = 0
    hb_order_info_with_market_general_features['Is_Partial_Fill'] = 0
    hb_order_info_with_market_general_features['Is_Partial_Cancel'] = 0

    #Target inheritance for merging afterwards
    
    order_event_targets = state_snapshot_df[['TOD', 'ID','TotalExecutedAfter', 'TotalFailureAfter', 'TotalActiveCanceledAfter', 'TotalExpiredAfter', 'TotalOrderExecutedVol', 'TotalOrderFailureVol']].copy()
    order_event_targets = order_event_targets.sort_values('TOD').drop_duplicates(subset = ['ID', 'TOD'], keep = 'last')
    
    hb_order_info_with_market_general_features = hb_order_info_with_market_general_features.sort_values('TOD')
    
    #Manual fix to allow merge below
    
    hb_order_info_with_market_general_features['TOD'] = hb_order_info_with_market_general_features['TOD'].astype('int32')
    order_event_targets['TOD'] = order_event_targets['TOD'].astype('int32')
    
    hb_order_info_with_market_general_features['ID'] = hb_order_info_with_market_general_features['ID'].astype('int32')
    order_event_targets['ID'] = order_event_targets['ID'].astype('int32')
    
    #Inheriting targets from the most actual event happening
    
    hb_with_targets = pd.merge_asof(
        hb_order_info_with_market_general_features,
        order_event_targets,
        on = 'TOD', #Look backwards from TOD
        by = 'ID',  #look only at this ID
        direction = 'backward'
        )
    
    hb_with_targets.fillna({'TotalExecutedAfter': 0, 'TotalFailureAfter': 0, 'TotalActiveCanceledAfter': 0, 'TotalExpiredAfter': 0}, inplace=True)
    
    #Combining Heartbeats with real events, filtering out noise
    
    final_state_df = pd.concat([state_snapshot_df, hb_with_targets], ignore_index = True)
    
    del state_snapshot_df
    del hb_with_targets
    gc.collect()
    
    final_state_df = final_state_df.sort_values('TOD')    
    
    #Some derivative related features, looking back in time, doing those here is the perfect place as we now have a chronological timeline of real events and hearbeats so we can easily look back
    
    df_past = final_state_df[['ID', 'TOD', 'LogVolAhead',
                              'DistanceToTouch', 
                              'Midprice', 'BestBid','SideOfBook',
                              'BestAsk', 'BidSize', 'AskSize', 'Microprice', 'DistanceToMicroprice', 'OrderFlowImbalance','QImbalance', 
                              #'MicroMidDeviation'
                              ]].copy()
    
    df_past['TOD_+_1000'] = (df_past['TOD'] + config.FEATURE_DELTA).astype('int32')
    
    final_state_df = pd.merge_asof(
        final_state_df,
        df_past.sort_values('TOD_+_1000'),
        by = 'ID',
        left_on = 'TOD',
        right_on = 'TOD_+_1000',
        direction = 'backward',
        suffixes = ['', '_past']    #Suffixes given to col names in left and right df that are merged
        )
    #Some deltas below are ln ratios to take relativeity into account instead of just offering absolute values
    
    
    ######Features for clock time that require ID
    
    #Clock 
    final_state_df['ClockDeltaMidprice'] = (final_state_df['Midprice'] - final_state_df['Midprice_past']).fillna(0)
    #final_state_df['ClockMicroMidDeviation'] = final_state_df['MicroMidDeviation'] - final_state_df['MicroMidDeviation_past']
    distancetobid = final_state_df['BestBid'] - final_state_df['BestBid_past']
    distancetoask = final_state_df['BestAsk'] - final_state_df['BestAsk_past']
    final_state_df['ClockDeltaDistanceToTouch'] = np.where(final_state_df['SideOfBook'] == 1, distancetobid, distancetoask)
    final_state_df["ClockQImbalance"] = (final_state_df['QImbalance'] - final_state_df['QImbalance_past']).fillna(0)
   
    #Some Dynamic events that require dynamic features
    #the deltadistance to microprice is equivalent to deltamicroprice so this is just a name
    
    eventdeltadistancetomicroprice = np.where(
        final_state_df['SideOfBook'] == 1, 
        final_state_df['EventDeltaMicroprice'], 
        -final_state_df['EventDeltaMicroprice']
    )
     
    final_state_df['EventDeltaDistanceToMicroprice'] = np.nan_to_num(eventdeltadistancetomicroprice, nan = 0.0, posinf = 0.0, neginf = 0.0)
    
    #Distance to touch moves with the Bid for Buys, and Ask for Sells
    final_state_df['EventDeltaDistanceToTouch'] = np.where(
        final_state_df['SideOfBook'] == 1, 
        final_state_df['EventDeltaBestBid'], 
        -final_state_df['EventDeltaBestAsk']
    )
     
    #OFI at best bid ask 
    # (Bid side demand change)
    db = np.where(final_state_df['BestBid'] > final_state_df['BestBid_past'], final_state_df['BidSize'],
          np.where(final_state_df['BestBid'] == final_state_df['BestBid_past'], final_state_df['BidSize'] - final_state_df['BidSize_past'],
          - final_state_df['BidSize_past']))

    #(Ask side supply change)
    da = np.where(final_state_df['BestAsk'] < final_state_df['BestAsk_past'], final_state_df['AskSize'],
          np.where(final_state_df['BestAsk'] == final_state_df['BestAsk_past'], final_state_df['AskSize'] - final_state_df['AskSize_past'],
          -final_state_df['AskSize_past']))

    ofi = db - da
    final_state_df['OrderFlowImbalance'] = np.nan_to_num(ofi, nan=0.0, posinf=0.0, neginf=0.0)
    clockofi = final_state_df['OrderFlowImbalance'] - final_state_df['OrderFlowImbalance_past']
    final_state_df['ClockDeltaOrderFlowImbalance'] = np.nan_to_num(clockofi, nan=0.0, posinf=0.0, neginf=0.0)
      
    #Clock
    final_state_df['ClockDeltaDistanceToMicroprice'] = (final_state_df['DistanceToMicroprice'] - final_state_df['DistanceToMicroprice_past'])
    final_state_df['ClockDeltaLogVolAhead'] = (final_state_df['LogVolAhead'] - final_state_df['LogVolAhead_past'])
    final_state_df['ClockDeltaDistanceToTouch'] = (final_state_df['DistanceToTouch'] - final_state_df['DistanceToTouch_past'])
    
    
    #Wipe out NaNs before calculating speed ===
    #Kill the NaNs generated by the morning orders lacking a past
    final_state_df.fillna({
        'EventDeltaDistanceToMicroprice': 0 ,
        'EventDeltaDistanceToTouch': 0,
        'ClockDeltaDistanceToMicroprice': 0,
        'ClockDeltaLogVolAhead': 0, 
        'ClockDeltaDistanceToTouch': 0,
    }, inplace=True)
    
    
    #Speed
    speed_features_to_test = [
        'DeltaOrderFlowImbalance', 
        'DeltaMidprice',
        'DeltaDistanceToMicroprice', 
        'DeltaDistanceToTouch',
    ]
    
    speed_featuresdic = speedmetric(final_state_df, speed_features_to_test)
    final_state_df = final_state_df.assign(**speed_featuresdic)
    
    ########## Adding two Self Exciting Features utilizint that MOs spike LO activity ##################
    #recall np.searchsorted (a,v,side) works like: if i want to insert values of array v into SORTED array a on side where should i put it
    
    mo_tods = df_MO['TOD'].values
    final_tods = final_state_df['TOD']
    
    mo_indices = np.searchsorted(mo_tods, final_tods, side = 'right') - 1   #This ensures if MOs arrive at these times in ms [10,20,20,30] that an event at time 25 gets placed at index 3-1 = 2 which is precisely the TOD of the most recent MO (if the first element is at index 0)
    
    #Just handleing some boundary case where at the start we cant look back yet, so insert some large value manually for time since last MO (10 sec)
    
    last_mo_timestamps = np.where(mo_indices >= 0, mo_tods[mo_indices], np.nan)
    
    final_state_df['TimeSinceLastMO'] = (final_tods - last_mo_timestamps).fillna(100000) #massive value for non existent time since last mos ]
    
    mo_sweeps = df_MO['SweepNoSweep'].values
    final_state_df['SweepNoSweep'] = np.where(mo_indices >=0, mo_sweeps[mo_indices], 0)
    
    sweep_times = df_MO[df_MO['SweepNoSweep'] == 1]['TOD'].values
    
    #boolean of whether there was a sweep in last 2000ms
    final_state_df['SweepInLast_2000ms'] = trailing_calc(
        final_state_df['TOD'], 
        sweep_times, 
        np.ones(len(sweep_times)), 
        2000
    )[1] > 0
    
    # Sweep Intensity (Total volume swept)
    final_state_df['SweepIntensity_2000ms'] = trailing_calc(
        final_state_df['TOD'], 
        sweep_times, 
        df_MO.loc[df_MO['SweepNoSweep'] == 1, 'Vol'].values, 
        2000
    )[0]
       
    final_state_df['MOCount10ms'] = trailing_calc(final_state_df['TOD'], mo_tods, df_MO['Vol'], 10)[1]
    
    
 
    
    del df_MO
    
   
    
    #Remove the cols from regression matrix of the noisy first and last 30 min of trading, but this can be undone later if want to train the model on the whole of cleandata
    if dont_include_full_trading_day is True:
        final_state_df = final_state_df[(final_state_df['TOD'] >= config.SOMARKET_NOISE ) & (final_state_df['TOD'] <= config.EOMARKET_NOISE)]


    #Binary for logistic
    
    # Generate the Success Rows
    fills_bin_df = final_state_df[final_state_df['TotalExecutedAfter'] > 0].copy()
    fills_bin_df[config.TARGET] = 1
    
    #Full logic of the weighting explained in notebook
    #max should just be the sum of the fills not repeatedly summed per snapshot
    fills_bin_df['OrderVolWithoutSnapShots'] = final_state_df['TotalOrderExecutedVol'].fillna(0)
    fills_bin_df['SumVol'] = fills_bin_df.groupby('ID')['TotalExecutedAfter'].transform('sum')
    fills_bin_df['UnitWeight'] = (fills_bin_df['TotalExecutedAfter'] / fills_bin_df['SumVol']) * fills_bin_df['OrderVolWithoutSnapShots']

    # Generate the Failure Rows
    fails_bin_df = final_state_df[final_state_df['TotalFailureAfter'] > 0].copy()
    fails_bin_df[config.TARGET] = 0
     
    fails_bin_df['OrderVolWithoutSnapShots'] = final_state_df['TotalOrderFailureVol'].fillna(0)
    fails_bin_df['SumVol'] = fails_bin_df.groupby('ID')['TotalFailureAfter'].transform('sum')
    fails_bin_df['UnitWeight'] = (fails_bin_df['TotalFailureAfter'] / fails_bin_df['SumVol']) * fails_bin_df['OrderVolWithoutSnapShots']



    # Combine into final training array
    Binary_Regression_Matrix = pd.concat([fills_bin_df, fails_bin_df], ignore_index=True)
    Binary_Regression_Matrix = Binary_Regression_Matrix.sort_values(by = 'TOD')
    
    #Multiclass for other engines
    
    # fills_multi_df = final_state_df[final_state_df['TotalExecutedAfter'] > 0].copy()
    # fills_multi_df[config.TARGET] = 1
    # fills_multi_df['MaxVol'] = fills_multi_df.groupby('ID')['TotalExecutedAfter'].transform('max') #Max here just means since each entry is like at a new snapshot, the max is just teh first entry, i.e the total amount that got filled etc. and .transfrom just gives back a new row with the general max in each entry
    # fills_multi_df['SumVol'] = fills_multi_df.groupby('ID')['TotalExecutedAfter'].transform('sum')
    # fills_multi_df['UnitWeight'] = (fills_multi_df['TotalExecutedAfter'] / fills_multi_df['SumVol']) * fills_multi_df['MaxVol']

    
    # active_cancels_multi_df = final_state_df[final_state_df['TotalActiveCanceledAfter'] > 0].copy()
    # active_cancels_multi_df[config.TARGET] = 0
    # active_cancels_multi_df['MaxVol'] = active_cancels_multi_df.groupby('ID')['TotalActiveCanceledAfter'].transform('max') #Max here just means since each entry is like at a new snapshot, the max is just teh first entry, i.e the total amount that got filled etc. and .transfrom just gives back a new row with the general max in each entry
    # active_cancels_multi_df['SumVol'] = active_cancels_multi_df.groupby('ID')['TotalActiveCanceledAfter'].transform('sum')
    # active_cancels_multi_df['UnitWeight'] = (active_cancels_multi_df['TotalActiveCanceledAfter'] / active_cancels_multi_df['SumVol']) * active_cancels_multi_df['MaxVol']

    
    # expired_multi_df = final_state_df[final_state_df['TotalExpiredAfter'] > 0].copy()
    # expired_multi_df[config.TARGET] = 2
    # expired_multi_df['MaxVol'] =  expired_multi_df.groupby('ID')['TotalExpiredAfter'].transform('max') #Max here just means since each entry is like at a new snapshot, the max is just teh first entry, i.e the total amount that got filled etc. and .transfrom just gives back a new row with the general max in each entry
    # expired_multi_df['SumVol'] =  expired_multi_df.groupby('ID')['TotalExpiredAfter'].transform('sum')
    # expired_multi_df['UnitWeight'] = ( expired_multi_df['TotalExpiredAfter'] /  expired_multi_df['SumVol']) *  expired_multi_df['MaxVol']

    
    # Multi_Class_Regression_Matrix = pd.concat([fills_multi_df, active_cancels_multi_df, expired_multi_df], ignore_index=True)
    # Multi_Class_Regression_Matrix = Multi_Class_Regression_Matrix.sort_values(by = 'TOD')

   
    #Compressing matrices to save RAM
    matrices_to_compress = [Binary_Regression_Matrix]
    for matrix in matrices_to_compress:
        for col in matrix.columns:
            col_type = matrix[col].dtype
            # Compress 64-bit floats to 32-bit
            if col_type == 'float64':
                matrix[col] = matrix[col].astype('float32')
                
            # Compress 64-bit integers to 32-bit
            elif col_type == 'int64':
                matrix[col] = matrix[col].astype('int32') 

   
    
    #Cols that either dont have necessary info or to prevent data leaking i.e we cant train on totalexecuted after since that happnes in the future

    cols_to_drop = ['ActiveCanceledVol', 'BaseTime', 'ExecutedVol', 'ExpiredVol','InitialPlacementTime','TotalOrderCanceledVol' , 'TotalOrderExpiredVol',
                    'SideOfBook_past', 'Step', 'TotalActiveCanceledAfter', 'TotalExecutedAfter', 'TotalExpiredAfter', 'TotalFailureAfter', 'TOD_+_1000', 'TOD_past', 'LogVolAhead_past',
                    'DistanceToTouch_past','TotalOrderFailureVol', 'TotalOrderExecutedVol',  'SumVol' ,'DistanceToMicroprice_past', 'Is_Partial_Fill', 'Is_Partial_Cancel',
                    'VolAhead', 'Midprice_past', 'BestBid_past', 'BestAsk_past', 'BidSize_past', 'AskSize_past', 'Microprice_past', 'OrderFlowImbalance_past', 'QImbalance_past'
                    ]
    

        
   
    Binary_Regression_Matrix = Binary_Regression_Matrix.drop(columns=cols_to_drop)
    #Multi_Class_Regression_Matrix = Multi_Class_Regression_Matrix.drop(columns=cols_to_drop)
    
    #print(Binary_Regression_Matrix.head())
    #print(Multi_Class_Regression_Matrix.head())
    
    #Check for NaNs
    matrix = Binary_Regression_Matrix # Or Multi_Class_Regression_Matrix
    nan_cols = matrix.columns[matrix.isna().any()].tolist()
    
    if nan_cols:
        print("!!! DANGER: NaNs detected in columns:", nan_cols)
        # Force a fill to prevent the crash, but this tells you which ones to fix
        matrix.fillna(0, inplace=True)
    else:
        print("Success: No NaNs detected.")
    
    return {
        
        'Binary Matrix': Binary_Regression_Matrix,
        #'Multi Matrix': Multi_Class_Regression_Matrix
        
        }



#In this environment below we can do the on the fly tests now since i restructered the code this is how it works now
if __name__ == "__main__":
    
    print("\n--- RUNNING DATA ENGINEERING SANDBOX ---")

    main_path, mo_path = get_data_paths()
    
    if main_path and mo_path:
        rawdata = import_data(main_path, mo_path)
        cleandata = clean_data(rawdata)
        
        #Put the custom stuff below here:
            
        rawdata = import_data(main_path, mo_path)
        cleandata = clean_data(rawdata)
        X = data_regressors(rawdata, cleandata, clear_RAM = False, dont_include_full_trading_day = True)['Binary Matrix']
        
        print(X.head())
        
        # Dummy code to show how the output variable regulation works 
        #Just creating a dummy df from the info above
        df_E = pd.DataFrame({
            'TOD': [57237043, 57241428, 57241429, 57241430, 57250845],
            'ID': [211736361,211736361,211736361, 211736361, 211736361],
            'Type': [66, 67, 67, 67, 68],  
            'Vol': [3600, 100, 1200, 1800, 500]  
        })

        # #Just creating a dummy df from the info above
        # df_E = pd.DataFrame({
        #     'TOD': [55548745, 56340804, 57600222],
        #     'ID': [193745485,193745485,193745485],
        #     'Type': [66, 67, 68],  
        #     'Vol': [1000, 500, 500]  
        # })

        # Initialize the Regressors DataFrame with dummy market features
        # We vary 'VolAhead' to simulate the order book changing in real-time
        R_df = pd.DataFrame()
        R_df['TOD'] = df_E['TOD']
        R_df['Type'] = df_E['Type']
        R_df['ID'] = df_E['ID']
        R_df['Vol'] = df_E['Vol']

        print("--- STEP 1: INITIAL COMPILING GRID ---")
        print(df_E)
        print("\n" + "="*60 + "\n")

        # =========================================================================
        # 2. RUN THE CONTINUOUS TARGET LOGIC
        # =========================================================================


        # Isolate exact event behaviors
        R_df['ExecutedVol'] = np.where(R_df['Type'].isin([69, 70]), R_df['Vol'], 0)

        R_df['ActiveCanceledVol'] = np.where(((R_df['Type'].isin([67, 68])) & (R_df['TOD'] < config.MARKET_CLOSE_TIME)), R_df['Vol'], 0)

        R_df['ExpiredVol'] = np.where(((R_df['Type'].isin([67, 68])) & (R_df['TOD'] >= config.MARKET_CLOSE_TIME)), R_df['Vol'], 0)

        # Execute the double-flip reverse cumsum calculation
        R_df['TotalExecutedAfter'] = (R_df.iloc[::-1].groupby('ID')['ExecutedVol'].cumsum().iloc[::-1] - R_df['ExecutedVol'])
        R_df['TotalActiveCanceledAfter'] = (R_df.iloc[::-1].groupby('ID')['ActiveCanceledVol'].cumsum().iloc[::-1] - R_df['ActiveCanceledVol'])
        R_df['TotalExpiredAfter'] = (R_df.iloc[::-1].groupby('ID')['ExpiredVol'].cumsum().iloc[::-1] - R_df['ExpiredVol'])

        print("--- STEP 2: Vol After Now (Looking into the future) ---")
        print(R_df[['Type', 'Vol', 'ExecutedVol', 'ActiveCanceledVol','ExpiredVol' ,'TotalExecutedAfter', 'TotalActiveCanceledAfter', 'TotalExpiredAfter']])
        print("\n" + "="*60 + "\n")

        # =========================================================================
        # 3. STATE-SPACE FILTERING & SNAPSHOT EXTRACTION
        # =========================================================================
        # Isolate rows where order states are born or transformed
        state_snapshots_df2 = R_df[R_df['Type'].isin([66,67,69,83])].copy()

        state_snapshots_df2['TotalFailureAfter'] = state_snapshots_df2['TotalActiveCanceledAfter'] + state_snapshots_df2['TotalExpiredAfter']

        print("--- STEP 3: STATE SNAPSHOTS RETAINED  ---")
        print(state_snapshots_df2[['TOD', 'Type','TotalExecutedAfter', 'TotalFailureAfter']])
        print("\n" + "="*60 + "\n")

        # =========================================================================
        # 4. TARGET GENERATION AND MATRIX PURGE
        # =========================================================================
        # Generate the Success Rows
        fills_df = state_snapshots_df2[state_snapshots_df2['TotalExecutedAfter'] > 0].copy()
        fills_df['FillNoFill'] = 1
        fills_df['Unit_Weight'] = fills_df['TotalExecutedAfter']

        # Generate the Failure Rows
        cancels_df = state_snapshots_df2[state_snapshots_df2['TotalFailureAfter'] > 0].copy()
        cancels_df['FillNoFill'] = 0
        cancels_df['Unit_Weight'] = cancels_df['TotalFailureAfter']

        # Combine into final training array
        Clean_Regression_Data = pd.concat([fills_df, cancels_df], ignore_index=True)



        Clean_Regression_Data = Clean_Regression_Data.sort_values(by = 'TOD')

        # Drop intermediate infrastructure tracking keys
        #Also for the proper code above i should drop type but just kept it in now since easier to check what im doing
        cols_to_drop = ['ID','Vol', 'ExecutedVol', 'ActiveCanceledVol', 'ExpiredVol' ,'TotalExecutedAfter', 'TotalActiveCanceledAfter', 'TotalExpiredAfter', 'TotalFailureAfter' ]
        Clean_Regression_Data = Clean_Regression_Data.drop(columns=cols_to_drop)

        print("--- STEP 4: FINAL CLEAN MACHINE LEARNING MATRIX ---")
        print(Clean_Regression_Data)
        
        # =========================================================================
        # DUMMY DEMONSTRATION: HEARTBEAT & TARGET INHERITANCE ENGINE
        # =========================================================================
        print("\n" + "="*80)
        print("--- HEARTBEAT & TARGET INHERITANCE DEMONSTRATION ---")
        print("="*80 + "\n")

        # STEP 1: Simulate a single order that lives for 35 seconds
        # t=0 (Place), t=15000 (Partial Fill), t=35000 (Cancel)
        df_dummy = pd.DataFrame({
            'TOD': [100000, 115000, 135000],  
            'ID': [999, 999, 999],
            'Type': [66, 69, 68],             
            'Vol': [1000, 400, 600]
        })
        
        print("STEP 1: RAW EVENTS OVER 35 SECONDS")
        print(df_dummy)
        print("\n" + "-"*60 + "\n")

        # STEP 2: Calculate Actual Event Targets (Using the fast Math Trick!)
        df_dummy['ExecutedVol'] = np.where(df_dummy['Type'].isin([69, 70]), df_dummy['Vol'], 0)
        df_dummy['CanceledVol'] = np.where(df_dummy['Type'].isin([67, 68]), df_dummy['Vol'], 0)
        
        df_dummy['TotalExecutedAfter'] = df_dummy.groupby('ID')['ExecutedVol'].transform('sum') - df_dummy.groupby('ID')['ExecutedVol'].cumsum()
        df_dummy['TotalCanceledAfter'] = df_dummy.groupby('ID')['CanceledVol'].transform('sum') - df_dummy.groupby('ID')['CanceledVol'].cumsum()
        
        print("STEP 2: TARGETS CALCULATED FOR REAL EVENTS")
        print(df_dummy[['TOD', 'Type', 'Vol', 'TotalExecutedAfter', 'TotalCanceledAfter']])
        print("\n" + "-"*60 + "\n")

        # STEP 3: Generate Heartbeats (10s intervals = 10000ms)
        interval = 10000
        duration = 135000 - 100000
        num_beats = duration // interval  # 35000 // 10000 = 3 heartbeats
        
        hb_df = pd.DataFrame({
            'ID': [999] * num_beats,
            'TOD': 100000 + (np.arange(1, num_beats + 1) * interval)
        })
        hb_df['Type'] = 26 # Custom flag for Heartbeats
        
        print("STEP 3: GENERATE ARTIFICIAL HEARTBEAT TIMESTAMPS (Every 10s)")
        print(hb_df)
        print("\n" + "-"*60 + "\n")

        # STEP 4: Target Inheritance via merge_asof
        hb_df = hb_df.sort_values('TOD')
        df_dummy = df_dummy.sort_values('TOD')
        
        # Force datatypes to prevent merge crash
        hb_df['TOD'] = hb_df['TOD'].astype('int32')
        df_dummy['TOD'] = df_dummy['TOD'].astype('int32')
        
        hb_with_targets = pd.merge_asof(
            hb_df,
            df_dummy[['TOD', 'TotalExecutedAfter', 'TotalCanceledAfter']],
            on='TOD',
            direction='backward'
        )
        
        print("STEP 4: HEARTBEATS LOOK BACKWARDS AND INHERIT TARGETS")
        print(hb_with_targets)
        print("\n" + "-"*60 + "\n")

        # STEP 5: Stack and Sort to see the final combined timeline!
        final_view = pd.concat([
            df_dummy[['TOD', 'Type', 'Vol', 'TotalExecutedAfter', 'TotalCanceledAfter', 'ID']],
            hb_with_targets
        ], ignore_index=True).sort_values('TOD').fillna({'Vol': 0})
        
        # Convert floats to ints for cleaner printing
        final_view['TotalExecutedAfter'] = final_view['TotalExecutedAfter'].astype(int)
        final_view['TotalCanceledAfter'] = final_view['TotalCanceledAfter'].astype(int)
        final_view['Vol'] = final_view['Vol'].astype(int)
        
        print("STEP 5: FINAL COMBINED CHRONOLOGICAL TIMELINE")
        print("Notice how Heartbeat 1 thinks there are 400 shares left to execute,")
        print("but Heartbeat 2 knows the execution already happened!")
        print("-" * 65)
        print(final_view.to_string(index=False))
        
       