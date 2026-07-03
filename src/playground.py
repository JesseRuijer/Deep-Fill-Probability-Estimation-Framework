#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 16:50:49 2026

@author: jesseruijer
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import config

test_df = pd.DataFrame({
    
    'a': [1,2,3],
    'b': [4,5,6]
    
    })

print(test_df.shift(1))


test_arr1 = np.array([1,2,3])
print(test_arr1)

test_arr2 = np.array([4,0,5])
print(test_arr2)


speed_array = np.divide(
            test_arr1, 
            test_arr2, 
            out=np.zeros_like(test_arr1, dtype = float), 
            where=(test_arr2 != 0)
        )

print(speed_array)

import pandas as pd


df = pd.DataFrame({
    'OrderID': [997, 998, 999],
    'InitialVolume': [100, 500, 1000]
})


df2 = pd.DataFrame({
    'OrderID2': [997, 998, 999],
    'InitialVolume2': [100, 500, 1000]
})

print("--- Original DataFrame ---")
print(df)
print(df.T)

test_dic = {
    'SpeedDeltaMidprice': [0.5, 1.2, 0.0],
    'SpeedLogVolAhead': [10.5, 8.1, 4.2],
    'JustSomeBacon': [7, 7, 7]
}

# df = df.assign(**test_dic)

# print("\n--- After Unpacking Dictionary ---")
# print(df)

pd.concat([df, df2])


b = np.array([1,2,3,4,5])
print(min(b))
print(np.min(b))
print(np.minimum(b))







sweep_tods = [1,2,3]

final_tods = [2,3,4]









# 2. Find the most recent sweep for every single event/heartbeat
sweep_indices = np.searchsorted(sweep_tods, final_tods, side='right') -1
print(sweep_indices)

plt.plot([0,1], [1,0])


print(config.LGBM_MODEL_FEATURES)

a = np.arange(0,0.401, 0.01)
b = np.arange(0.43,1, 0.03)

print(a)
print(b)
print(np.concatenate([a,b]))




print(np.arange(1,3,1 ))

lst = [1,2,3,4]
print(lst[:-1])
print(lst[-1])

















