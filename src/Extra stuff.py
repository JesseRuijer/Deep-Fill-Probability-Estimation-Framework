#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 13:48:14 2026

@author: jesseruijer
"""

######################################## Some basic commands ###################################
#print(df_Event.iloc[0:10, 0:5])
#Keys
#print(mat_data_MO.keys())
#Find something specific print(df_Event[df_Event[0] == 24581757])
# print(df_Event.head())
# print(df_Event.tail())

# describe gives some nice stats stuff on dfs 
#print(df_Event_without_noise.describe())

#How to import from other scripts
#from Functions import time_in_hours
#print(time_in_hours(3600000))

#Search in DF
# print(df_Event_without_noise.iloc[15:25,0:7])
######################################################################################






####################Below shows at end of day spams 68s to cancel outstanding orders in full #####################
# mask3 = (
#     (df_Event["Type"] == 68) & 
#     (df_Event["TOD"] > 55800000) & 
#     (df_Event["TOD"] <= 57602000)
#     )
# print(df_Event[mask3])
######################################################################################






#When you add the .values it changes structure from pandas df to a numpy array much faster for calculations
# testdf = pd.DataFrame({
#     '0': [1,2,3],
#     '1':[2,3,4]    
#     }
#     )

# print(testdf)
# print(testdf.values)

#Proving the Vol Ahead works 
# print(regressormatrix.head())
# print(cleandata['Event'].loc[91742])
# print(cleandata['BuyVol'].loc[91742])
# print(cleandata['BuyPrice'].loc[91742][0])


#Below prints how many 1s and 0s we had
# print("\n Total Fills vs Cancels")
# print(regressormatrix["Fill_NoFill"].value_counts())

#Below prints how many counts of Types we had
# print(rawdata["Event"]["Type"].value_counts())


#This was for p values but i dont think they are relevant when are sample size is this big, since i think standard error pretty much goes to zero when n is this big in our case couple hundred thousand 

# #Use statsmodeling library to present statistical evidence for findings p values etc, since technically this doesnt work or is very hard with sci kit learn i think
# #For statsmodeling lib you have to manually enter a col of 1s for intercept
# X_train_sm = sm.add_constant(X_train_standardised)
# #Creating statistical model logistic regression
# stat_model = sm.Logit(y_train.values, X_train_sm)
# #Fitting the model
# result = stat_model.fit(disp=False)
# #Build a pandas df to present result
# stats_df = pd.DataFrame({
#     "Feature": log_mdl_features,
#     "Coefficient (Log Odds)": result.params[1:],  # write [1:] to slice the array removing the first part since that was the artificial col of 1s we had to manually add before in order for statsmodel lib to work
#     "Odds Ratio": np.exp(result.params[1:]),
#     "P-Value": result.pvalues[1:]
# })

# print(stats_df)

######################################################################################
# #The logic for the double []::-1] stuff in regressormatrix function
# df = pd.DataFrame({
#     'Col A' : [1,2,3,4,5],
#     'Col B' : [6,7,8,9,10]
#     })
# print(df)
# print(df.cumsum())
# print(df.iloc[::-1])
# print(df.iloc[::-1].cumsum())
# print(df.iloc[::-1].cumsum().iloc[::-1])
######################################################################################


######################################################################################
# ### check if 88/84 have ID, can i find order with similar ID to the 84 earlier in day 
# #So the answer is no we cant find similar ID earlier in day, but they do both have an ID
# #88 has an ID and you can immediately see that the VOL of those is quite big, they are for bulk auctions for cross events
# #At start and end of day because then you can have crossing i.e buy orders and sell orders on top of eachother, note the actual distinction between buy and sell side does not exist now, so the side of the book variable does not make any sense for 88s 
# print(rawdata['Event'][rawdata['Event']['Type'] == 88].head())
# print(rawdata['Event'][rawdata['Event']['Type'] == 84].head())
# # Theres 84 if and only if ID is 0
# print(rawdata['Event'][(rawdata['Event']['ID'] != 0) & (rawdata['Event']['Type'] == 84)])
# print(rawdata['Event'][(rawdata['Event']['Type'] != 84) & (rawdata['Event']['ID'] == 0)])


######################################################################################





# #####################Proving data only NOT ONLY has DAY type limit orders i.e. a LO placed during day will doesnt exist anymore after 4pm########################3
  
# order_lifespan = rawdata['Event'].groupby('ID')['TOD'].agg(['min', 'max'])  #Creates pd df with buckets grouped by ID then only looks at TOD and then for each bucket calculates min and max and outputs that in pds df
# mask6 = (order_lifespan['min'] < config.MARKET_CLOSE_TIME) & (order_lifespan['max'] > config.MARKET_CLOSE_TIME)
# extract_orders = order_lifespan[mask6].index.values

# print(extract_orders)
# print(len(extract_orders)) #However only like 8 were put in much later then a few seconds past 4 where the cancelations are probably just do to latency 
# print(order_life(147566 , rawdata))     #This was placed in pre market and canceled at end of day 

# ############################################################################














# #############3########## Replicating the slides #############################
# print(f" Total number of events on 1 April 2014 of INTC is {len(cleandata['Event'])}")
# print(f" Amount of Market Orders on 1 April 2014 of INTC is {len(cleandata['MO'])}")
# print(f' Percentage of MO per total number of events on 1 April 2014 INTC is {(len(cleandata["MO"])/len(cleandata["Event"])*100):.04f}%')

# buy_no_walk = (cleandata['MO']['APPS'] == cleandata['MO']['BAP'] ) & (cleandata['MO']['BorS'] == -1)
# sell_no_walk = (cleandata['MO']['APPS'] == cleandata['MO']['BBP'] ) & (cleandata['MO']['BorS'] == 1)
# total_walk = buy_no_walk | sell_no_walk # | is OR operator

# print(f" Percentage of orders that did not walk the book for INTC on April 1 2024 is {(total_walk.sum()/(len(cleandata['MO'])) * 100):.2f} % ")
# #try to recreate the graph at 11 am
# plots(rawdata, 34199640)
# print(time_in_hours(33300000))
# time_to_hours(9.4999)
# print(cleandata['Event'])
# print(9.25*3600000)

# ################################################################################























# ################Proving Unknown 2 is what size of the book an event happens###########
# #84 midprice half int, rounding might be wrong. Answer: no as the hidden orders are executed and that can be done at halfprice if agreed to. it is however true that regular LOs can only be placed in sizes of one cent at the smallest  

# #This shows whenever 66 or 83 we only get 1s and 0s respectively
# print(cleandata["Event"][(cleandata["Event"]["Type"] == 66) & (cleandata["Event"]["SideOfBook"] == 0)])
# print(cleandata["Event"][(cleandata["Event"]["Type"] == 83) & (cleandata["Event"]["SideOfBook"] == 1)])
# #Must also check the side is correct for the other orders, most difficult will be the hidden orders
# hidden0 = (rawdata['Event']['Type'] == 84) & (rawdata["Event"]["SideOfBook"] == 0)
# hidden0index = rawdata['Event'][hidden0].index
# look_up_index = hidden0index - 1 #To see what LOB looked like before LO was placed
# #Adding .values is really important here because that removes the original indices from the df and just hands pandas lists of numbers without any pre fit indices
# best_buy = rawdata["BuyPrice"].loc[look_up_index, 0].values # first col gives best buy price
# best_sell = rawdata["SellPrice"].loc[look_up_index, 0].values # first col gives best sell price
# trade_price84 = rawdata["Event"].loc[hidden0index, "Price"].values

# res_df = pd.DataFrame(
#     { "Trade price of 84": trade_price84,
#      "Best Buy": best_buy,
#      "Best Sell": best_sell}
#     )

# print(f'       Hidden Orders      \n {res_df.head(10)}')
# #Here we may see orders exactly at mid price to be pegged there if they have to sell a lot they dont want to scare the market i think
# #Sell LOs slightly above best buy is to ensure they can capture the incoming buyer immediately
# #That gave some intuition about the location of the Hidden orders, now we can go on to show the sides are still correct
# #For the other order types

# target_event_types = [67, 68, 69, 70, 84]
# mask = rawdata['Event']["Type"].isin(target_event_types)
# index = rawdata['Event'][mask].index
# unknown2val = rawdata['Event'].loc[index, 'SideOfBook'].values
# eventprice = rawdata['Event'].loc[index, 'Price'].values

# #Now look at LOB status before this order was placed
# index_before = index - 1
# best_bid = rawdata['BuyPrice'].loc[index_before, 0].values
# best_ask = rawdata['SellPrice'].loc[index_before, 0].values
# midprice_before = (best_ask + best_bid)/2
# what_type = rawdata['Event'].loc[index, 'Type'].values

# proof_df = pd.DataFrame(
#     {'EventPrice': eventprice,
#      'unknown2' : unknown2val,
#      'Midprice': midprice_before,
#      'Type': what_type
#      })

# #if price is below midprice its on the buy side and vice versa
# proof_df['StrictlyBuySide%'] = (proof_df['EventPrice'] < proof_df['Midprice']).astype(int)*100 #astype just converts the boolean into 1 or 0
# proof_df['StrictlyMidPrice%'] = (proof_df['EventPrice'] == proof_df['Midprice']).astype(int)*100
# proof_df['StrictlySellSide%'] = (proof_df['EventPrice'] > proof_df['Midprice']).astype(int)*100

# bucket_cols = ['Type', 'unknown2']
# target_cols = ['StrictlyBuySide%', 'StrictlyMidPrice%', 'StrictlySellSide%']
# grouped_buckets = proof_df.groupby(bucket_cols)
# filtered_buckets = grouped_buckets[target_cols]
# final_proof = filtered_buckets.mean()

# #i.e for each type of event its calculated for the price of the order where the type was for whether that was on buy
# #or sell side

# print(f'                 Final Proof of unknown 2    \n {final_proof}')

# #########################################################################################



####################Some parquet stuff i used before and some data storage stuff in config##############################

    # train_matrices = prep_data_daily(config.TRAIN_FILE_PATH, config.TRAIN_FILE_PATH_MO)
    # test_matrices = prep_data_daily(config.TEST_FILE_PATH, config.TEST_FILE_PATH_MO)
    
    
    # #Save matrix as a parquet file for binary
    # train_matrices['Binary Training Matrix'].to_parquet(config.TRAIN_BINARY_OUT)
    # test_matrices['Binary Training Matrix'].to_parquet(config.TEST_BINARY_OUT)
    
    # #Save matrix as a parquet file for binary
    # train_matrices['Multi Training Matrix'].to_parquet(config.TRAIN_MULTI_OUT)
    # test_matrices['Multi Training Matrix'].to_parquet(config.TEST_MULTI_OUT)


# #May not be needed anymore when using my new and improved file manager file
# TRAIN_FILE_PATH = '../data/raw/INTC_NASDAQ/INTC_20140401_NASDAQ.mat'
# TRAIN_FILE_PATH_MO = '../data/raw/INTC_NASDAQ/Market Order/INTC_20140401.mat'

# TEST_FILE_PATH = '../data/raw/INTC_NASDAQ/INTC_20140424_NASDAQ.mat'
# TEST_FILE_PATH_MO = '../data/raw/INTC_NASDAQ/Market Order/INTC_20140424.mat'

# TRAIN_BINARY_OUT = "../data/processed/INTC_train_BINARY_2014_04_01.parquet"
# TEST_BINARY_OUT  = "../data/processed/INTC_test_BINARY_2014_04_24.parquet"

# TRAIN_MULTI_OUT  = "../data/processed/INTC_train_MULTI_2014_04_01.parquet"
# TEST_MULTI_OUT   = "../data/processed/INTC_test_MULTI_2014_04_24.parquet"

########################################################################################################################



#######################The orginal FillNoFill code###########################################3

# def creating_simple_labels(train_event_df):
    
#     #My original code which simply labels an order as being cancelled if only a part of it is cancelled and then ignores the rest of that order
#     #Its obviously wrong, but just saved it here in case i need it 
    
#     #Maps all the cancel or fill parts of LOs to df
#     outcome_df = train_event_df[train_event_df["Type"].isin([67,68,69,70])]

#     #This could still be useful if we just want to look at if a random order will be filled partially or fully so dont remove this yet, maybe put it in another function
#     #Groups by ID and looks at what happens last, i.e that will be fill or cancel, but also looks if there was a partial fill at any time during its life 
#     fill_map = outcome_df.groupby("ID")["Type"].apply(lambda x: x.isin([69,70]).any())

#     #Converts it to binary 1 for fill to use in logistic regression
#     fill_map = fill_map.astype(int)
    
#     return fill_map


##########################Paste this into dataandfeatureengineering sandbox to prove the MO and the main files are merged together and saved as parquet###################################3
 # import pandas as pd
 # import os
 
 # # 1. Point directly to one of the new files your batch processor just made
 # test_file = "../data/processed/INTC_BINARY_2014_04_01.parquet"
 
 # if os.path.exists(test_file):
 #     print(f"Loading {os.path.basename(test_file)}...")
 #     df = pd.read_parquet(test_file)
     
 #     # 2. Print the size of the matrix (Should be hundreds of thousands of rows, and lots of columns)
 #     print(f"\nMatrix Shape: {df.shape[0]} rows, {df.shape[1]} columns")
     
 #     # 3. THE ULTIMATE PROOF: Check for the MO Target Column
 #     # If the MO file didn't merge, 'FillNoFill' mathematically cannot exist in this dataframe.
 #     if 'FillNoFill' in df.columns:
 #         print("\nSUCCESS: 'FillNoFill' column found! The MO data was successfully merged.")
         
 #         # Let's see how many orders were actually filled vs not filled on this day
 #         print("\nFill/No Fill Distribution:")
 #         print(df['FillNoFill'].value_counts())
 #     else:
 #         print("\nERROR: 'FillNoFill' is missing. The MO merge failed.")
         
 #     # 4. (Optional) Print all columns just so you can visually inspect them
 #     # print("\nAll Columns in Parquet:")
 #     # print(df.columns.tolist())
     
 # else:
 #     print("File not found. Check the file path!")











