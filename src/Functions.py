#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 14:27:25 2026

@author: jesseruijer
"""

import matplotlib.pyplot as plt
import seaborn as sns

def time_in_hours(ms_past_midnight):
    
    #Translates time in ms after midnight to regular time
    
    hours = ms_past_midnight // 3600000
    remaining_ms = ms_past_midnight % 3600000
    
    minutes = remaining_ms // 60000
    remaining_ms = remaining_ms % 60000
    
    seconds = remaining_ms // 1000
    ms = remaining_ms % 1000
    
    hours%24
    
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{ms:03d}"



def order_life(order_ID, cleandata):
    
    #Give this function order ID it will return the life of this order ID
    
    return cleandata["Event"][cleandata["Event"]["ID"] == order_ID ].sort_values(by = "TOD")

def plots(cleandata, target_time):
    
    #Function that plots Bar chart at target_time of Vol of bids and asks around midpoint
    
    row_index_event = (cleandata["Event"]["TOD"] - target_time).abs().idxmin()#this finds the row index closest to the time
    buy_prices = cleandata["BuyPrice"].loc[row_index_event] / 10000
    buy_volume = cleandata["BuyVol"].loc[row_index_event] 
    sell_prices = cleandata["SellPrice"].loc[row_index_event] / 10000
    sell_volume = cleandata["SellVol"].loc[row_index_event]
    plt.figure(figsize=(9,6))
    plt.bar(buy_prices, buy_volume, width = 0.007, color = "red" , edgecolor = 'black',label = 'Bids')
    plt.bar(sell_prices, sell_volume, width = 0.007, color = "blue",edgecolor = 'black', label = 'Asks')
    plt.xlabel("Price")
    plt.ylabel("Vol")
    plt.title(f"Vol of Best Bid and Best Ask immediately before MO at {time_in_hours(target_time)}")
    plt.legend()
    plt.show()
    
    sns.histplot(x = cleandata["MO"]["BBV"], color = 'blue', label = "Best Buy Vol", kde=True, bins = 50, alpha = 0.5)
    sns.histplot(x = cleandata["MO"]["BAV"], color = 'red', label = "Best Ask Vol", kde=True, bins = 50, alpha = 0.5)
    plt.xlabel("Vol")
    plt.ylabel("Freq")
    plt.title("Freq vs Vol of Best Bid and Best Ask immediately before MO")
    plt.legend()
    plt.show()
    
    
def plot_feature(regressormatrix, feature):
    
    #Plots boxplot and density plot of a given feature in regressormatrix
    
    plt.figure(figsize = (9,5))
    
    sns.boxplot(
        data = regressormatrix,
        x = "Fill_NoFill" ,
        y = feature,
        )
    plt.xlabel("Filled or Not")
    plt.ylabel(feature)
    plt.title(f'Influence of {feature} on fill or no fill')
    plt.show()
    
    sns.kdeplot(data=regressormatrix[regressormatrix['Fill_NoFill'] == 1], x = feature, color = 'blue', label = 'Filled (1)')
    sns.kdeplot(data=regressormatrix[regressormatrix['Fill_NoFill'] == 0], x = feature, color = 'red', label = 'Not Filled (0)')
    plt.xlabel(f'{feature}')
    plt.ylabel("density")
    plt.legend()
    plt.title(f'Influence of {feature} on fill or no fill')
    plt.show()

def plot_corr_map(regressormatrix):
    
    #Plots Correlation heat plot of all features and of features that appear in model
    
    features_all = ['BASpread', 'QImbalance', 'AbsQImbalance', 'TotalVolImbalance', 'Weighted Vol Imbalance', 
                'Midprice', 'Microprice', "DistanceToTouch", 'LogVolAhead', "LookBackHiddenVol"] 
    
    features_in_model = ['AbsQImbalance', 'Weighted Vol Imbalance', 'Microprice', 
                         "DistanceToTouch", 'LogVolAhead', "LookBackHiddenVol"] 
    corr_matrix = regressormatrix[features_all].corr()
    corr_matrix2 = regressormatrix[features_in_model].corr()

    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("All Feature Correlation Matrix")
    plt.show()
    
    
    sns.heatmap(corr_matrix2, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Feature Correlation Matrix Of Features In Model")
    plt.show()
    
    
    

