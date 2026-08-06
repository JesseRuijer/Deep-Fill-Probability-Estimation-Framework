#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 14:27:25 2026

@author: jesseruijer
"""

"""

Contains helper functions that are used throughout the code, such as converting time from MS to standard time and vice versa etc.

"""

import numpy as np
import config 
import pandas as pd


def time_in_hours(ms_past_midnight: int) -> str:
    
    """
    Translates time in ms after midnight to regular time
    """
    
    hours = ms_past_midnight // 3600000
    remaining_ms = ms_past_midnight % 3600000
    
    minutes = remaining_ms // 60000
    remaining_ms = remaining_ms % 60000
    
    seconds = remaining_ms // 1000
    ms = remaining_ms % 1000
    
    hours = hours%24
    
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}.{int(ms):03d}"

def time_to_ms(time: float) -> float:
    
    """
    converts time in hours to time in ms, both after midnight 
    Input time as follows: 9:15 am = 9.25
    """
    
    return time*3600000

def order_life(order_ID: int, df: dict) -> pd.DataFrame:
    
    #Give this function order ID it will return the life of this order ID
    
    return df["Event"][df["Event"]["ID"] == order_ID ].sort_values(by = "TOD")

def find_order_pattern(df: dict, start_type: int, middle_type: int, middle_count: int, end_type: int) -> pd.DataFrame:
    """
    Pass it some type of TYPE pattern you want a specific order ID to satisfy and it returns those, note it can only have a beginning, middle and end type not more, although you can specify how often the middle type occurs
    """
    
    def check_rules(history: pd.Series) -> bool:
        history_list = list(history)
        
        if history_list[0] != start_type:
            return False
            
        if history_list[-1] != end_type:
            return False

        if history_list.count(middle_type) != middle_count:
            return False
            
        # If it survives all the checks above it's a match
        return True

    #Group the data by ID, and run the rulebook on every order
    valid_id_mask = df['Event'].groupby('ID')['Type'].apply(check_rules) #Then pandas automaticcaly applies checkrules with history as input where history is all the buckets created before by grouping by ID 
    
    #Extract the IDs that passed the test
    matching_ids = valid_id_mask[valid_id_mask].index
    
    #Filter the original dataframe to only keep those winning IDs
    orders = df['Event'][df['Event']['ID'].isin(matching_ids)]
    
    return orders
    
def speedmetric(df: pd.DataFrame, feature_list: list) -> dict:
    
    """
     Calculates the absolute magnitude of speed (Event Delta / Clock Delta).
     Returns strictly positive values. Relies on the raw Event/Clock deltas 
     in the main dataframe to provide the directional sign to the models.
     """
    
    
    result = {}
    
    for metric in feature_list:
        event = df[f'Event{metric}'].values
        clock = df[f'Clock{metric}'].values
        
        abs_event = np.abs(event)
        abs_clock = np.abs(clock)
        
        speed_array = np.divide(
                    abs_event, 
                    abs_clock,  
                    out=np.zeros_like(event, dtype = float), 
                    where=(clock != 0)
                )
        result[f'Speed_{metric}'] = speed_array
        
    return result



def trailing_calc(tod_targets: np.ndarray, tod_source: np.ndarray, vol_source: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    
    """
    Calculates vol and index related information for trailing features
    """
    
    cum_vol = np.pad(np.cumsum(vol_source), (1,0), constant_values = 0)      #pads to add a zero at the start and then cumsum calculates the running total so to know the order arrival rate between two different times you just calculate the difference in their total running values
    start_indices = np.searchsorted(tod_source, tod_targets - lookback, side = 'left')   #Finds the row indices where the lookback window starts and below where it finishes
    end_indices = np.searchsorted(tod_source, tod_targets , side = 'right') 
    return (cum_vol[end_indices] - cum_vol[start_indices]), (end_indices - start_indices)



def calculate_rolling_moments(df: pd.DataFrame, feature_list: list, window: int = config.EVENT_TIME_DELTA, calc_std: bool = False, calc_skew: bool = False, calc_kurt: bool = False, calc_extremes: bool = False) -> dict:
    
    """
    Calculates moments for features
    """
    
    new_features = {}
    
    for feature in feature_list:
        # Create the rolling window object once per feature for efficiency
        roll = df[feature].rolling(window=window, min_periods=2)
        
        if calc_std:
            new_features[f'RollingStd_{feature}'] = roll.std().fillna(0).astype('float32')
            
        if calc_skew:
            # Skew requires at least 3 data points to not be NaN
            new_features[f'RollingSkew_{feature}'] = roll.skew().fillna(0).astype('float32')
            
        if calc_kurt:
            # Kurtosis requires at least 4 data points
            new_features[f'RollingKurt_{feature}'] = roll.kurt().fillna(0).astype('float32')
            
        if calc_extremes:
            new_features[f'RollingMax_{feature}'] = roll.max().fillna(0).astype('float32')
            new_features[f'RollingMin_{feature}'] = roll.min().fillna(0).astype('float32')

    return new_features