#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 13:48:14 2026

@author: jesseruijer
"""

#To print
#print(df_Event.iloc[0:10, 0:5])
#Keys
#print(mat_data_MO.keys())
#Find something specific print(df_Event[df_Event[0] == 24581757])

# print(df_Event.head())
# print(df_Event.tail())

# describe gives some nice stats stuff on dfs 
#print(df_Event_without_noise.describe())


#Search in DF
# print(df_Event_without_noise.iloc[15:25,0:7])
# print(df_MO_without_noise.iloc[0:5,0:9])

#How to import from other scripts
#from Functions import time_in_hours
#print(time_in_hours(3600000))


# print(order_life(32398125, cleandata))

#Below shows at end of day spams 68s to cancel outstanding orders in full 
# mask3 = (
#     (df_Event["Type"] == 68) & 
#     (df_Event["TOD"] > 55800000) & 
#     (df_Event["TOD"] <= 57600000)
#     )