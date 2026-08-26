#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 12:48:23 2026

@author: jesseruijer
"""

"""
Exploratory analysis for when importing new asset or trading day file

Returns plots, tables and general information regarding that day, also has a function to plot where in the LOB a specific order is and visualizes that

"""

import matplotlib.pyplot as plt
import seaborn as sns
import config
import os
import numpy as np   
import pandas as pd    

from Functions import time_in_hours, time_to_ms
from DataAndFeatureEngineering import import_data, clean_data, data_regressors

def run_exploratory_analysis(rawdata: dict, cleandata: dict, regressormatrix: pd.DataFrame, feature: str, target_time: float) -> None:
    
    """    
    Gives basic data info and plots regarding the analyzed day

    """

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
    
    plt.xlabel("Price ($)")
    plt.ylabel("Volume")
    plt.yscale('log')
    plt.title(f"Vol of Best Bid Vol and Best Ask Vol at {time_in_hours(target_time)}")
    plt.grid(True, alpha = 0.3)
    plt.legend()
    plt.show()
    
    sns.histplot(x = cleandata["MO"]["BBV"], color = 'blue', label = "Best Buy Vol", kde=True, bins = 50, alpha = 0.5)
    sns.histplot(x = cleandata["MO"]["BAV"], color = 'red', label = "Best Ask Vol", kde=True, bins = 50, alpha = 0.5)
    plt.xlabel("Vol")
    plt.ylabel("Freq")
    plt.title("Freq vs Vol of Best Bid and Best Ask")
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
    
    vol66 = rawdata['Event'][rawdata['Event']['Type'] == 66]['Vol'].sum()
    vol67 = rawdata['Event'][rawdata['Event']['Type'] == 67]['Vol'].sum()
    vol68 = rawdata['Event'][rawdata['Event']['Type'] == 68]['Vol'].sum()
    vol69 = rawdata['Event'][rawdata['Event']['Type'] == 69]['Vol'].sum()
    vol70 = rawdata['Event'][rawdata['Event']['Type'] == 70]['Vol'].sum()
    vol83 = rawdata['Event'][rawdata['Event']['Type'] == 83]['Vol'].sum()
    vol84 = rawdata['Event'][rawdata['Event']['Type'] == 84]['Vol'].sum()
    vol88 = rawdata['Event'][rawdata['Event']['Type'] == 88]['Vol'].sum()

        
    #Some other basic metrics
    freq_of_events = rawdata["Event"]["Type"].value_counts()
    cancelation_ratio = canceled_vol_day / added_vol_day
    
    buy_no_walk = (cleandata['MO']['APPS'] == cleandata['MO']['BAP'] ) & (cleandata['MO']['BorS'] == -1)
    sell_no_walk = (cleandata['MO']['APPS'] == cleandata['MO']['BBP'] ) & (cleandata['MO']['BorS'] == 1)
    total_walk = buy_no_walk | sell_no_walk # | is OR operator
    
    print('Checking clock time difference across the day for a lookback of 50 events')
    differences = cleandata['Event']['TOD'].diff(config.EVENT_TIME_DELTA)
    print(f'The max time between 50 events was {differences.max():.2f} ms')
    print(f'The median time between 50 events was {differences.median():.2f} ms')
    print(f'The avg time between 50 events was {differences.mean():.2f} ms')
    
    
    print('################VOLUME METRICS############## \n')
    print(f'Added Vol in a day (i.e sum during entire day of 66 and 83 vol) is {added_vol_day}')
    print(f'Canceled Vol in a day (i.e sum during entire day of 67 and 68 vol) is {canceled_vol_day}')
    print(f'Traded Vol in a day (i.e sum during entire day of 69 and 70 vol) is {traded_vol_day}')
    print(f'Hidden trades executed Vol in a day (i.e sum during entire day of 84 vol) is {hidden_executed_vol_day}')
    print(f'Bulk Cross Vol in a day (i.e sum during entire day of 88 vol) is {bulk_cross_vol_day}')
    print(f'Absolute Event Vol in a day (i.e sum during entire day of all events, measure of activity, note this double counts vol for adding and canceling orders ) is {absolute_event_vol_day}')
    
    print(f'66 Vol {vol66}')
    print(f'67 Vol {vol67}')
    print(f'68 Vol {vol68}')
    print(f'69 Vol {vol69}')
    print(f'70 Vol {vol70}')
    print(f'83 Vol {vol83}')
    print(f'84 Vol {vol84}')
    print(f'88 Vol {vol88}')

    
    
    print('\n #########Some other metrics#############\n')
    print(f" Total number of events on day of {config.TICK} is {len(rawdata['Event'])}")
    print(f' Frequency of different event types \n {freq_of_events}')
    print(f' Cancelation Ratio, i.e how many of total added vol in a day were cancelations is {(cancelation_ratio)*100} %')
    
    print(f'Amount of final bulkorder cross section is \n {rawdata["Event"][rawdata["Event"]["Type"] == 88 ]}')
    print(f'Percentage of Vol of Events in the day that were bulk orders is { ((rawdata["Event"][rawdata["Event"]["Type"] == 88 ]["Vol"].sum())/(rawdata["Event"]["Vol"].sum()) * 100) } %')
    
    
    print(' \n #########Some Market Order Info #########\n')
    print(f" Amount of Market Orders on day of {config.TICK} is {len(rawdata['MO'])}")
    print(f' Percentage of MO per total number of events on day {config.TICK} is {(len(rawdata["MO"])/len(rawdata["Event"])*100):.04f}%')
    
    
    print(f" Percentage of orders that did  walk the book for {config.TICK} on day is {(1 -((total_walk.sum())/(len(cleandata['MO'])))) * 100:.2f} % ")
    
    time_between_two_mos = rawdata['MO']['TOD'].diff(1)
    print(f'The max time between two market orders was {time_between_two_mos.max():.2f} ms')
    print(f'The median time between two market orders was {time_between_two_mos.median():.2f} ms')
    print(f'The average time between two market orders was {time_between_two_mos.mean():.2f} ms')
    
    #Make a plot on x axis TOD and on y axis MO amount vol placed
    plt.figure(figsize = (20,10))
    mo_tods = rawdata['MO']['TOD']
    mo_vols = rawdata['MO']['Vol']
    counts, bin_edges = np.histogram(mo_tods, bins = 200, weights = mo_vols)  # np.hist gives back the counts and the edges i.e begin and end point of each bin
    
    middle = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    plt.plot(middle, counts, color = 'b', label = 'MO Vol')
    plt.xlim(config.MARKET_OPEN_TIME, config.MARKET_CLOSE_TIME)
    custom_ticks = np.linspace(config.MARKET_OPEN_TIME, config.MARKET_CLOSE_TIME, num=9)
    clock_labels = [time_in_hours(int(tick)) for tick in custom_ticks]
    plt.xticks(custom_ticks, clock_labels)
    plt.xlabel('TOD')
    plt.ylabel('Vol of MOs')
    plt.title('Vol of MOs vs TOD')
    plt.legend()

    plt.show()
        
    plt.figure(figsize = (20,10))
    counts, bin_edges = np.histogram(mo_tods, bins = 200)  # np.hist gives back the counts and the edges i.e begin and end point of each bin
    
    middle = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    plt.plot(middle, counts, color = 'b', label = 'MO Freq')
    plt.xlim(config.MARKET_OPEN_TIME, config.MARKET_CLOSE_TIME)
    custom_ticks = np.linspace(config.MARKET_OPEN_TIME, config.MARKET_CLOSE_TIME, num=9)
    clock_labels = [time_in_hours(int(tick)) for tick in custom_ticks]
    plt.xticks(custom_ticks, clock_labels)
    plt.xlabel('TOD')
    plt.ylabel('Freq of MOs')
    plt.title('Freq of MOs vs TOD')
    plt.legend()
    plt.tight_layout()

    plt.show()
    
    #Same two plots as above but now for LOs  
    
    plt.figure(figsize = (20,10))
    lo_tods = rawdata['Event']['TOD']
    lo_vols = rawdata['Event']['Vol']
    counts, bin_edges = np.histogram(lo_tods, bins = 200, weights = lo_vols)  # np.hist gives back the counts and the edges i.e begin and end point of each bin
    
    middle = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    plt.plot(middle, counts, color = 'b', label = 'LO Vol')
    plt.xlim(config.MARKET_OPEN_TIME, config.MARKET_CLOSE_TIME)
    custom_ticks = np.linspace(config.MARKET_OPEN_TIME, config.MARKET_CLOSE_TIME, num=9)
    clock_labels = [time_in_hours(int(tick)) for tick in custom_ticks]
    plt.xticks(custom_ticks, clock_labels)
    plt.xlabel('TOD')
    plt.ylabel('Vol of LOs')
    plt.title('Vol of LOs vs TOD')
    plt.legend()

    plt.show()

    plt.figure(figsize = (20,10))
    counts, bin_edges = np.histogram(mo_tods, bins = 200)  # np.hist gives back the counts and the edges i.e begin and end point of each bin
    
    middle = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    plt.plot(middle, counts, color = 'b', label = 'LO Frequency')
    plt.xlim(config.MARKET_OPEN_TIME, config.MARKET_CLOSE_TIME)
    custom_ticks = np.linspace(config.MARKET_OPEN_TIME, config.MARKET_CLOSE_TIME, num=9)
    clock_labels = [time_in_hours(int(tick)) for tick in custom_ticks]
    plt.xticks(custom_ticks, clock_labels)
    plt.xlabel('Time of Day')
    plt.ylabel('Frequency of LOs')
    #plt.title('Freq of LOs vs TOD')
    plt.legend()
    plt.tight_layout()
    plt.grid(True, alpha = 0.3)

    plt.show()
    
    df_E = rawdata['Event']
    df_BP = rawdata['BuyPrice']
    df_SP = rawdata['SellPrice']
    
    maskadd = df_E['Type'].isin([66,83])
    lim_orders = df_E[maskadd]
    bps = df_BP[maskadd]
    sps = df_SP[maskadd]
    total_lim_orders = len(lim_orders)

    
    maskadd2 = ((lim_orders['Price'] == bps[0]) | (lim_orders['Price'] == sps[0]))
    
    lim_orders_at_best_price = len(lim_orders[maskadd2])
    
    result = lim_orders_at_best_price/ total_lim_orders
    
    
    maskadd3 = ((lim_orders['Price'] == bps[0]) | 
            (lim_orders['Price'] == sps[0]) |
            (lim_orders['Price'] == bps[1]) | 
            (lim_orders['Price'] == sps[1]))
    lim_orders_at_best_price2 = len(lim_orders[maskadd3])
    result2 = lim_orders_at_best_price2 / total_lim_orders
    
    print(f'The percentage of limit orders placed at best price out of all limit orders placed is {result*100:.2f}%')
    
    print(f'The percentage of limit orders placed at best price and first level out in the book out of all limit orders placed is {result2*100:.2f}%')
    
    
    
    maskcancel = df_E['Type'].isin([67,68])
    lim_orders = df_E[maskcancel]
    bps = df_BP[maskcancel]
    sps = df_SP[maskcancel]
    total_lim_orders = len(lim_orders)

    
    maskcancel2 = ((lim_orders['Price'] == bps[0]) | (lim_orders['Price'] == sps[0]))
    
    lim_orders_at_best_price = len(lim_orders[maskcancel2])
    
    result = lim_orders_at_best_price/ total_lim_orders
    
    
    maskcancel3 = ((lim_orders['Price'] == bps[0]) | 
            (lim_orders['Price'] == sps[0]) |
            (lim_orders['Price'] == bps[1]) | 
            (lim_orders['Price'] == sps[1]))
    lim_orders_at_best_price2 = len(lim_orders[maskcancel3])
    result2 = lim_orders_at_best_price2 / total_lim_orders
    
    print(f'The percentage of limit orders canceled/deleted at best price out of all limit orders placed is {result*100:.2f}%')
    
    print(f'The percentage of limit orders canceled/deleted at best price and first level out in the book out of all limit orders placed is {result2*100:.2f}%')
    
def plot_order_queue_position(ID: int, cleandata: dict, TSP: int) -> None:
    
    """
    This function reads from cleandata hence its still unshifted
    i.e read order book before order was placed is -1 and immediately after placement = just regular idx 
    plots a similar histogram to as above, but includes the specific order as a slice in its bin
    
    """
    
    #Isolate the target order
    order_events = cleandata['Event'][cleandata['Event']['ID'] == ID]
    if len(order_events) == 0:
        print(f"Order ID {ID} not found in cleandata.")
        return

    # Grab the initial placement row
    placement = order_events.iloc[0]
    idx = placement.name     #.name gives back the index of the placement row, cant directly use index since above we converted it to a 1d pandas series, some stupid pandas quirk ig

    price = placement['Price'] / 10000
    order_vol = placement['Vol']
    side = placement['SideOfBook'] 
    target_tod = int(placement['TOD'] + TSP)
    placement_time = int(placement['TOD'])

    
    # position-based lookup instead of idx-1, since filtered index labels can have gaps
    event_index = cleandata['Event'].index
    pos = event_index.get_loc(idx)
    if pos == 0:
        current_vol_ahead = 0.0
    else:
        prev_idx = event_index[pos - 1]
        prev_prices = (cleandata['BuyPrice'].loc[prev_idx] if side == 1 else cleandata['SellPrice'].loc[prev_idx]) / 10000
        prev_vols = (cleandata['BuyVol'].loc[prev_idx] if side == 1 else cleandata['SellVol'].loc[prev_idx])
        current_vol_ahead = prev_vols[prev_prices == price].sum()

    # nothing can be behind you at the instant of placement
    current_vol_behind = 0.0

    # filtered by Price AND SideOfBook, so opposite-side orders at the same numeric price can't leak in
    price_events = cleandata['Event'][
        (cleandata['Event']['TOD'] > placement['TOD']) &
        (cleandata['Event']['TOD'] <= target_tod) &
        (cleandata['Event']['Price'] == placement['Price']) &
        (cleandata['Event']['SideOfBook'] == side)
    ]
    
    arrival_times = cleandata['Event'].groupby('ID')['TOD'].first()
    my_arrival = arrival_times[ID]
   
    for _, event in price_events.iterrows():    #iterrows loops through rows and returns tuple [index, data]
       vol = event['Vol']
       eid = event['ID']
       etype = event['Type']
       
       event_arrival = arrival_times.get(eid, placement_time)
       
       if etype in [66,83]:
           
           if event_arrival < my_arrival:
               current_vol_ahead += vol
           else:
               current_vol_behind += vol
        
      # event arrival (using the arrival_times groupby and first()) is first time an event for this ID was recorded, i.e when the order first joined the queue, thats why we can use the split logic below      
       if etype in [67,68,69,70]:
           if eid == ID:
               order_vol = max(0, order_vol - vol)
           elif event_arrival < my_arrival:
               current_vol_ahead = max(0, current_vol_ahead - vol)
           else:
               current_vol_behind = max(0, current_vol_behind - vol)
               
    valid_events = cleandata['Event'][cleandata['Event']['TOD'] <= target_tod]
    current_idx = valid_events.index[-1]

    buy_prices = cleandata["BuyPrice"].loc[current_idx] / 10000
    buy_vols = cleandata["BuyVol"].loc[current_idx]
    sell_prices = cleandata["SellPrice"].loc[current_idx] / 10000
    sell_vols = cleandata["SellVol"].loc[current_idx]
   
    #Plot the Baseline Book
    plt.figure(figsize=(9, 6))
    
    valid_bids = (buy_prices > 0) & (buy_vols > 0)
    valid_asks = (sell_prices > 0) & (sell_vols > 0)
    
    # Plot all normal bids and asks EXCEPT the specific bin our order is in
    normal_bids_mask = valid_bids & (buy_prices != price)
    normal_asks_mask = valid_asks & (sell_prices != price)
    
    plt.bar(buy_prices[normal_bids_mask], buy_vols[normal_bids_mask], width=0.007, color="red", alpha=0.5, edgecolor='black', label='Bids')
    plt.bar(sell_prices[normal_asks_mask], sell_vols[normal_asks_mask], width=0.007, color="blue", alpha=0.5, edgecolor='black', label='Asks')
    
    #Plot the Stacked Bar at the Order's Price
    if side == 1:
        base_color = "red"
        side_str = "Bid"
    else:
        base_color = "blue"
        side_str = "Ask"
        
    # Layer 1: Vol Ahead (Bottom)
    if current_vol_ahead > 0:
        plt.bar(price, current_vol_ahead, width=0.007, color=base_color, alpha=0.5, edgecolor='black')
    
    # Layer 2: Our Order (Middle, Highlighted)
    if order_vol > 0:
        plt.bar(price, order_vol, bottom= current_vol_ahead, width=0.007, color='yellow', edgecolor='black', label=f'Order {ID} ({side_str})\nQueue Pos: {current_vol_ahead}')
    
    # Layer 3: Vol Behind (Top)
    if current_vol_behind > 0:
        plt.bar(price, current_vol_behind, bottom=(current_vol_ahead + order_vol), width=0.007, color=base_color, alpha=0.5, edgecolor='black')

    plt.title(f"LOB at {time_in_hours(target_tod)} | Queue Position for ID {ID}")
    plt.xlabel("Price")
    plt.ylabel("Vol")
    plt.legend()
    plt.show()
    
    
if __name__ == "__main__":  
    
    TARGET_TIME = time_to_ms(11)
    FEATURE_ANALYSE = 'LogVolAhead'
      
    from FileManager import get_data_paths

    main_path, mo_path = get_data_paths()
    
    if main_path and mo_path:
        rawdata = import_data(main_path, mo_path)
        cleandata = clean_data(rawdata)
        regressormatrix = data_regressors(rawdata, cleandata, clear_RAM= False, dont_include_full_trading_day=False)['Binary Matrix']
    
        print(f'Starting Exploratory Data Analysis on {os.path.basename(main_path)} and {os.path.basename(mo_path)} \n')    
        run_exploratory_analysis(rawdata, cleandata, regressormatrix, FEATURE_ANALYSE, TARGET_TIME)