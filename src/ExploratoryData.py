#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 12:48:23 2026

@author: jesseruijer
"""

#Just some quick stuff on when importing new data or creating some graphs


import matplotlib.pyplot as plt
import seaborn as sns
from Functions import time_in_hours, time_to_hours
from DataAndFeatureEngineering import import_data, clean_data, data_regressors
import config
import os
import numpy as np        

def run_exploratory_analysis(rawdata, cleandata, regressormatrix,feature, target_time):
        
    #Function that plots Bar chart at target_time of Vol of bids and asks around midpoint

    row_index_event = (cleandata["Event"]["TOD"] - target_time).abs().idxmin()#this finds the row index closest to the time
    buy_prices = cleandata["BuyPrice"].loc[row_index_event] / 10000
    buy_volume = cleandata["BuyVol"].loc[row_index_event] 
    sell_prices = cleandata["SellPrice"].loc[row_index_event] / 10000
    sell_volume = cleandata["SellVol"].loc[row_index_event]
    
    valid_bids = (buy_prices > 0) & (buy_volume > 0)
    valid_asks = (sell_prices > 0) & (sell_volume > 0)
    
    clean_buy_prices = buy_prices[valid_bids]
    clean_buy_volume = buy_volume[valid_bids]
    clean_sell_prices = sell_prices[valid_asks]
    clean_sell_volume = sell_volume[valid_asks]
    
    plt.figure(figsize=(9,6))
    plt.bar(clean_buy_prices, clean_buy_volume, width = 0.007, color = "red" , alpha = 0.5, edgecolor = 'black',label = 'Bids')
    plt.bar(clean_sell_prices, clean_sell_volume, width = 0.007, color = "blue",alpha = 0.5,edgecolor = 'black', label = 'Asks')
    
    #Autoscaling x axis and adding a tiny buffer so the bars dont touch the edges of the graph
    min_active_price = min(clean_buy_prices.min(), clean_sell_prices.min())
    max_active_price = max(clean_buy_prices.max(), clean_sell_prices.max())
    
    buffer = 0.03
    
    plt.xlim(min_active_price - buffer, max_active_price + buffer)
    
    plt.xlabel("Price")
    plt.ylabel("Vol")
    plt.title(f"Vol of Best Bid Vol and Best Ask Vol at {time_to_hours(TARGET_TIME)}")
    plt.legend()
    plt.show()
    
    sns.histplot(x = cleandata["MO"]["BBV"], color = 'blue', label = "Best Buy Vol", kde=True, bins = 50, alpha = 0.5)
    sns.histplot(x = cleandata["MO"]["BAV"], color = 'red', label = "Best Ask Vol", kde=True, bins = 50, alpha = 0.5)
    plt.xlabel("Vol")
    plt.ylabel("Freq")
    plt.title("Freq vs Vol of Best Bid and Best Ask immediately before MO")
    plt.legend()
    plt.show()
    

    
    #Plots boxplot and density plot of a given feature in regressormatrix
    
    plt.figure(figsize = (9,5))
    
    sns.boxplot(
        data = regressormatrix,
        x = config.TARGET ,
        y = feature,
        )
    plt.xlabel("Filled or Not")
    plt.ylabel(feature)
    plt.title(f'Influence of {feature} on fill or no fill')
    plt.show()
    
    sns.kdeplot(data=regressormatrix[regressormatrix[config.TARGET] == 1], x = feature, color = 'blue', label = 'Filled (1)')
    sns.kdeplot(data=regressormatrix[regressormatrix[config.TARGET] == 0], x = feature, color = 'red', label = 'Not Filled (0)')
    plt.xlabel(f'{feature}')
    plt.ylabel("density")
    plt.legend()
    plt.title(f'Influence of {feature} on fill or no fill')
    plt.show()

    
    #Some Volume metrics
    added_vol_day = rawdata['Event'][rawdata['Event']['Type'].isin([66,83])]['Vol'].sum()
    canceled_vol_day = rawdata['Event'][rawdata['Event']['Type'].isin([67,68])]['Vol'].sum()
    traded_vol_day = rawdata['Event'][rawdata['Event']['Type'].isin([69, 70])]['Vol'].sum()
    hidden_executed_vol_day = rawdata['Event'][rawdata['Event']['Type'].isin([84])]['Vol'].sum()
    bulk_cross_vol_day = rawdata['Event'][rawdata['Event']['Type'].isin([88])]['Vol'].sum()
    absolute_event_vol_day = rawdata['Event']['Vol'].sum()
    
    #Some other basic metrics
    freq_of_events = rawdata["Event"]["Type"].value_counts()
    cancelation_ratio = canceled_vol_day / added_vol_day
    
    buy_no_walk = (cleandata['MO']['APPS'] == cleandata['MO']['BAP'] ) & (cleandata['MO']['BorS'] == -1)
    sell_no_walk = (cleandata['MO']['APPS'] == cleandata['MO']['BBP'] ) & (cleandata['MO']['BorS'] == 1)
    total_walk = buy_no_walk | sell_no_walk # | is OR operator
    
    print('################VOLUME METRICS############## \n')
    print(f'Added Vol in a day (i.e sum during entire day of 66 and 83 vol) is {added_vol_day}')
    print(f'Canceled Vol in a day (i.e sum during entire day of 67 and 68 vol) is {canceled_vol_day}')
    print(f'Traded Vol in a day (i.e sum during entire day of 69 and 70 vol) is {traded_vol_day}')
    print(f'Hidden trades executed Vol in a day (i.e sum during entire day of 84 vol) is {hidden_executed_vol_day}')
    print(f'Bulk Cross Vol in a day (i.e sum during entire day of 88 vol) is {bulk_cross_vol_day}')
    print(f'Absolute Event Vol in a day (i.e sum during entire day of all events, measure of activity, note this double counts vol for adding and canceling orders ) is {absolute_event_vol_day}')
    
    print('\n #########Some other metrics#############\n')
    print(f" Total number of events on 1 April 2014 of INTC is {len(rawdata['Event'])}")
    print(f' Frequency of different event types \n {freq_of_events}')
    print(f' Cancelation Ratio, i.e how many of total added vol in a day were cancelations is {(cancelation_ratio)*100} %')
    
    print(f'Amount of final bulkorder cross section is \n {rawdata["Event"][rawdata["Event"]["Type"] == 88 ]}')
    print(f'Percentage of Vol of Events in the day that were bulk orders is { ((rawdata["Event"][rawdata["Event"]["Type"] == 88 ]["Vol"].sum())/(rawdata["Event"]["Vol"].sum()) * 100) } %')
    
    
    print(' \n #########Some Market Order Info #########\n')
    print(f" Amount of Market Orders on 1 April 2014 of INTC is {len(rawdata['MO'])}")
    print(f' Percentage of MO per total number of events on 1 April 2014 INTC is {(len(rawdata["MO"])/len(rawdata["Event"])*100):.04f}%')
    
    
    print(f" Percentage of orders that did  walk the book for INTC on April 1 2024 is {(1 -((total_walk.sum())/(len(cleandata['MO'])))) * 100:.2f} % ")
    
    print('Checking clock time difference across the day for a lookback of 50 events')
    differences = cleandata['Event']['TOD'].diff(config.EVENT_TIME_DELTA)
    print(f'The max time between 50 events was {differences.max()}')
    print(f'The median time between 50 events was {differences.median()}')
    print(f'The avg time between 50 events was {differences.mean()}')
    
    time_between_two_mos = rawdata['MO']['TOD'].diff(1).mean()
    
    print(f'The average time between two market orders was {time_between_two_mos}')
    
    #Make a plot on x axis TOD and on y axis MO amount placed
    
    plt.figure(figsize = (20,10))
    
    print(rawdata['MO']['TOD'].describe())
    print(time_in_hours(30752880))
    
    mo_counts = rawdata['MO']['TOD']
    counts, bin_edges = np.histogram(mo_counts, bins = 65)  # np.hist gives back the counts and the edges i.e begin and end point of each bin
    
    middle = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    plt.plot(middle, counts, color = 'b', label = 'MO Freq')
    plt.xlim(config.MARKET_OPEN_TIME, config.MARKET_CLOSE_TIME)
    current_ticks = plt.xticks()[0]
    clock_labels = [time_in_hours(int(tick)) for tick in current_ticks]
    plt.xticks(current_ticks, clock_labels)
    plt.xlabel('TOD')
    plt.ylabel('Frequency of MOs')
    plt.title('Freq of MOs vs TOD')
    plt.legend()
    plt.tight_layout()

    plt.show()
    
    mask = rawdata['Event']['Type'].isin([66,83])
    lim_orders = rawdata[mask]
    total_lim_orders = len(lim_orders)
    
    
    print(total_lim_orders)
    
    mask2 = ((rawdata['Event'][rawdata['Price']] == rawdata['BuyPrice'][0]) | (rawdata['Event'][rawdata['Price']] == rawdata['BuyPrice'][0]))
    
    lim_orders_at_best_price = len(lim_orders[mask2])
    
    result = lim_orders_at_best_price/ total_lim_orders
    
    print(f'The percentage of limit orders placed at best price out of all limit orders placed is {result*100}%')
    
    print(f'The percentage of limit orders placed at best price and first level out in the book out of all limit orders placed is {result*100}%')
    
    
    
if __name__ == "__main__":  
    
    TARGET_TIME = time_to_hours(11)
    FEATURE_ANALYSE = 'DistanceToMicroprice'
      
    from FileManager import get_data_paths

    main_path, mo_path = get_data_paths()
    
    if main_path and mo_path:
        rawdata = import_data(main_path, mo_path)
        cleandata = clean_data(rawdata)
    
    print(f'Starting Exploratory Data Analysis on {os.path.basename(main_path)} and {os.path.basename(mo_path)} \n')
   

    regressormatrix = data_regressors(rawdata, cleandata, clear_RAM=False)['Binary Matrix']
    
    run_exploratory_analysis(rawdata, cleandata, regressormatrix, feature = FEATURE_ANALYSE, target_time = TARGET_TIME)

    
    
    
    
    
    
    