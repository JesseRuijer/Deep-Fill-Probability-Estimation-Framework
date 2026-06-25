#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 14:27:25 2026

@author: jesseruijer
"""

#Importing libraries,classes, functions from other scripts

import numpy as np


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

def time_to_hours(time):
    #Input time as follows: 9:15 am = 9.25
    return time*3600000

def order_life(order_ID, df):
    
    #Give this function order ID it will return the life of this order ID
    
    return df["Event"][df["Event"]["ID"] == order_ID ].sort_values(by = "TOD")

def find_order_pattern(df, start_type, middle_type, middle_count, end_type):
    #Pass it some type of TYPE pattern you want a specific order ID to satisfy and it returns those
    def check_rules(history):
        history_list = list(history)
        
        if history_list[0] != start_type:
            return False
            
        if history_list[-1] != end_type:
            return False

        if history_list.count(middle_type) != middle_count:
            return False
            
        # If it survives all the checks above, it's a match!
        return True

    #Group the data by ID, and run the rulebook on every order
    valid_id_mask = df['Event'].groupby('ID')['Type'].apply(check_rules) #Then pandas automaticcaly applies checkrules with history as input where history is all the buckets created before by grouping by ID 
    
    #Extract the IDs that passed the test
    matching_ids = valid_id_mask[valid_id_mask].index
    
    #Filter the original dataframe to only keep those winning IDs
    orders = df['Event'][df['Event']['ID'].isin(matching_ids)]
    
    return orders
    
def speedmetric(df, feature_list):
    
    result = {}
    
    for metric in feature_list:
        event = df[f'Event{metric}'].values
        clock = df[f'Clock{metric}'].values
        
        speed_array = np.divide(
                    event, 
                    clock, 
                    out=np.zeros_like(event, dtype = float), 
                    where=(clock != 0)
                )
        result[f'Speed{metric}'] = speed_array
        
    return result



def trailing_calc(tod_targets, tod_source, vol_source, lookback):
    cum_vol = np.pad(np.cumsum(vol_source), (1,0), constant_values = 0)      #pads to add a zero at thes start and then cumsum calculates the running total so to know the order arrival rate between two different times you just calculate the difference in their total running values
    start_indices = np.searchsorted(tod_source, tod_targets - lookback, side = 'left')   #Finds the row indices where the lookback window starts and below where it finishes
    end_indices = np.searchsorted(tod_source, tod_targets , side = 'right') 
    return (cum_vol[end_indices] - cum_vol[start_indices]), (end_indices - start_indices)
    
