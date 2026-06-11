#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 14:27:25 2026

@author: jesseruijer
"""

#Importing libraries,classes, functions from other scripts


    

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


    
    


